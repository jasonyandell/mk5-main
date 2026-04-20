"""Tests for ``burl.modal.gemma_local`` — pin the contract against the
Modal server's ``generate_native`` shape so a bad swap is caught early.

These tests load the real model (bf16 weights, ~6 GB). If HF cache is cold
we skip rather than download in CI — set ``BURL_LOCAL_SKIP_MODEL=1`` to force
skipping regardless.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from burl.modal.gemma_local import (
    DEFAULT_MODEL_REPO,
    GemmaLocalNative,
    make_local_native_model,
)


def _model_cached() -> bool:
    """Return True iff HF cache contains a config.json for the default repo."""
    if os.environ.get("BURL_LOCAL_SKIP_MODEL") == "1":
        return False
    cache_root = Path(
        os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"
    )
    hub_root = cache_root / "hub"
    # repo dirs are named like "models--mlx-community--gemma-4-e2b-it-bf16"
    repo_dir = hub_root / ("models--" + DEFAULT_MODEL_REPO.replace("/", "--"))
    if not repo_dir.exists():
        return False
    # A cached repo has at least one snapshot with a config.json in it.
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return False
    return any(
        (snap / "config.json").exists()
        for snap in snapshots.iterdir() if snap.is_dir()
    )


pytestmark = pytest.mark.skipif(
    not _model_cached(),
    reason=f"Gemma 4 E2B model ({DEFAULT_MODEL_REPO}) not in HF cache.",
)


@pytest.fixture(scope="module")
def server() -> GemmaLocalNative:
    """Load the model once for all tests in this module."""
    return GemmaLocalNative()


def test_loads_without_error(server: GemmaLocalNative) -> None:
    assert server.model is not None
    assert server.tokenizer is not None
    assert server.model_repo == DEFAULT_MODEL_REPO
    # EOS set must include the turn terminator (106) so generation actually stops.
    assert 106 in server.tokenizer.eos_token_ids


def test_generate_native_shape(server: GemmaLocalNative) -> None:
    result = server.generate_native(
        [{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=10,
    )
    assert set(result.keys()) == {"text", "prompt_text", "n_tokens"}
    assert isinstance(result["text"], str)
    assert isinstance(result["prompt_text"], str)
    assert isinstance(result["n_tokens"], int)
    assert result["n_tokens"] > 0
    # prompt_text is the chat-templated string the server fed to MLX
    assert "hi" in result["prompt_text"]


def test_preserves_special_tokens(server: GemmaLocalNative) -> None:
    """When the model is asked to emit a tool call, the returned text must
    retain the ``<|tool_call>...<tool_call|>`` envelope — the parser keys
    off it. We ask for a tool call explicitly with a minimal tool."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Return the sum of two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": (
                "Use the add tool to compute 17 + 25. Emit exactly one tool "
                "call, then stop."
            ),
        }
    ]
    result = server.generate_native(
        messages, tools=tools, max_tokens=256, enable_thinking=False,
    )
    # Either channel-based thinking markers or a tool_call envelope must be
    # preserved in the decode. For this prompt we expect at minimum the
    # tool_call envelope — Gemma is heavily post-trained to use it.
    has_tool_call = (
        "<|tool_call>" in result["text"] and "<tool_call|>" in result["text"]
    )
    has_channel = (
        "<|channel>" in result["text"] and "<channel|>" in result["text"]
    )
    assert has_tool_call or has_channel, (
        "Expected <|tool_call>...<tool_call|> or <|channel>...<channel|> "
        f"in output; got: {result['text']!r}"
    )


def test_factory_returns_str() -> None:
    """``make_local_native_model`` returns a callable that takes
    (messages, tools) and returns a str — matches the signature the rollout
    harness expects."""
    model_fn = make_local_native_model(max_tokens=8)
    out = model_fn([{"role": "user", "content": "hi"}], [])
    assert isinstance(out, str)
    assert len(out) > 0


def test_adapter_name_logs_warning(
    server: GemmaLocalNative, caplog: pytest.LogCaptureFixture,
) -> None:
    """Passing adapter_name at call-time is a no-op locally; must warn."""
    import logging

    caplog.set_level(logging.WARNING, logger="burl.modal.gemma_local")
    server.generate_native(
        [{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=4,
        adapter_name="some-adapter",
    )
    assert any("adapter_name" in rec.message for rec in caplog.records)
