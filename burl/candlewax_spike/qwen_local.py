"""Local Qwen3.6-35B-A3B (MoE, 3B active) VLM inference for candlewax spike.

Runs on Apple silicon via mlx-vlm. Image + text in, structured output out —
parses Hermes-style ``<tool_call>...</tool_call>`` blocks and ``<think>...</think>``
blocks separately so downstream code can treat them like the Claude events.

Single-load, reused across the batch. Model weights live in the HF cache;
first call is slow (load + first forward compile), subsequent calls fast.

Docs:
- Qwen3.6 release (2026-04-16): https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- MLX port: https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit
- Hermes tool template: https://qwen.readthedocs.io/en/latest/framework/function_call.html
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any

from mlx_vlm import generate, load
from mlx_vlm.utils import load_config


DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Qwen3.6 chat template emits nested-XML tool calls:
#   <tool_call>
#   <function=commit_play>
#   <parameter=domino_id>23</parameter>
#   </function>
#   </tool_call>
# Fallback: some templates/fine-tunes still emit Hermes-style JSON.
_FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_JSON_TOOL_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_tool_call_body(body: str) -> dict:
    """Parse a ``<tool_call>...</tool_call>`` body into ``{name, arguments}``.

    Supports both the Qwen3.6 nested-XML shape (``<function=X><parameter=Y>``)
    and the Hermes JSON shape (``{"name": X, "arguments": {...}}``).
    """
    m = _FN_RE.search(body)
    if m is not None:
        name = m.group(1).strip()
        args: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(m.group(2)):
            key = pm.group(1).strip()
            val = pm.group(2).strip()
            # Coerce ints when possible — the JSON schema says integer.
            try:
                args[key] = int(val)
            except ValueError:
                args[key] = val
        return {"name": name, "arguments": args}
    jm = _JSON_TOOL_RE.search(body)
    if jm is not None:
        try:
            parsed = json.loads(jm.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"_parse_error": "no function or JSON block", "_raw_body": body.strip()}


@dataclasses.dataclass
class QwenBlock:
    """One parsed chunk of the raw generation string."""
    kind: str        # "thinking" | "tool_call" | "text"
    content: Any     # str for thinking/text, dict {"name","arguments"} for tool_call
    raw: str         # the exact substring this came from


@dataclasses.dataclass
class QwenResult:
    raw: str                   # full decoded string
    blocks: list[QwenBlock]    # ordered split
    elapsed_s: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_gb: float | None


# --------------------------------------------------------------------------- #
# Load + cache                                                                 #
# --------------------------------------------------------------------------- #


_LOADED: dict[tuple[str, str | None], tuple[Any, Any, Any]] = {}


def load_qwen(
    model_path: str = DEFAULT_MODEL,
    adapter_path: str | Path | None = None,
) -> tuple[Any, Any, Any]:
    """Return (model, processor, config). Cached; safe to call per decision.

    ``adapter_path`` (optional): path to a LoRA adapter directory. When
    provided, ``mlx_vlm.load`` applies the LoRA layers on top of the base
    weights. Cache key is ``(model_path, adapter_path)`` so base and adapted
    variants can coexist in one process.
    """
    adapter_str = str(adapter_path) if adapter_path is not None else None
    key = (model_path, adapter_str)
    if key in _LOADED:
        return _LOADED[key]
    t0 = time.time()
    print(
        f"[qwen] loading {model_path} (adapter={adapter_str}) ...",
        flush=True,
    )
    model, processor = load(model_path, adapter_path=adapter_str)
    config = load_config(model_path)
    elapsed = time.time() - t0
    print(f"[qwen] loaded in {elapsed:.1f}s", flush=True)
    _LOADED[key] = (model, processor, config)
    return _LOADED[key]


# --------------------------------------------------------------------------- #
# Output parsing                                                               #
# --------------------------------------------------------------------------- #


def _parse_blocks(raw: str) -> list[QwenBlock]:
    """Split ``raw`` into an ordered list of think / tool_call / text blocks.

    Processes ``<think>`` and ``<tool_call>`` tags in source order; everything
    outside those tags is flattened into ``text`` blocks (skipping empties).
    """
    blocks: list[QwenBlock] = []

    # Collect all tag spans with type.
    spans: list[tuple[int, int, str, str]] = []   # (start, end, kind, raw)
    for m in _THINK_RE.finditer(raw):
        spans.append((m.start(), m.end(), "thinking", m.group(0)))
    for m in _TOOL_CALL_RE.finditer(raw):
        spans.append((m.start(), m.end(), "tool_call", m.group(0)))
    spans.sort()

    cursor = 0
    for (start, end, kind, raw_span) in spans:
        if start > cursor:
            pre = raw[cursor:start]
            if pre.strip():
                blocks.append(QwenBlock(kind="text", content=pre.strip(), raw=pre))
        if kind == "thinking":
            content = _THINK_RE.search(raw_span).group(1).strip()
            blocks.append(QwenBlock(kind="thinking", content=content, raw=raw_span))
        elif kind == "tool_call":
            body = _TOOL_CALL_RE.search(raw_span).group(1)
            parsed = _parse_tool_call_body(body)
            blocks.append(QwenBlock(kind="tool_call", content=parsed, raw=raw_span))
        cursor = end

    # Trailing text
    if cursor < len(raw):
        trail = raw[cursor:]
        if trail.strip():
            blocks.append(QwenBlock(kind="text", content=trail.strip(), raw=trail))

    return blocks


# --------------------------------------------------------------------------- #
# Main entry                                                                   #
# --------------------------------------------------------------------------- #


def qwen_decide(
    *,
    system_content: str,
    user_content: str,
    image_path: str | Path | None,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    enable_thinking: bool = True,
    temperature: float = 0.0,
    model_path: str = DEFAULT_MODEL,
    adapter_path: str | Path | None = None,
) -> QwenResult:
    """Run one decision through local Qwen3.6.

    ``tools`` is a list of OpenAI-style tool specs (same shape LEM uses for
    Gemma native tool calls). The Qwen chat template consumes them via its
    Hermes block; the model emits ``<tool_call>...</tool_call>`` on its own.

    ``image_path``: pass the candlewax PNG for multimodal; None for plain text.

    ``adapter_path``: optional LoRA adapter directory. Forwarded to
    ``load_qwen`` so the model cache keys on ``(model_path, adapter_path)``.
    """
    model, processor, config = load_qwen(model_path, adapter_path=adapter_path)

    messages: list[dict] = [
        {"role": "system", "content": system_content},
    ]
    if image_path is None:
        messages.append({"role": "user", "content": user_content})
        num_images = 0
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_content},
            ],
        })
        num_images = 1

    # Build prompt string via the processor's chat template. Pass tools so
    # Qwen's Hermes block is emitted into the system preamble; enable_thinking
    # lets the model do <think>…</think> before the tool call.
    tokenizer = getattr(processor, "tokenizer", processor)
    chat_kwargs: dict[str, Any] = {}
    if tools:
        chat_kwargs["tools"] = tools
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking
    prompt_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **chat_kwargs,
    )

    t0 = time.time()
    image_arg = str(image_path) if image_path is not None else None
    result = generate(
        model, processor, prompt_str,
        image=image_arg,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    )
    elapsed = time.time() - t0

    raw = getattr(result, "text", None) or str(result)
    prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
    gen_tokens = int(getattr(result, "generation_tokens", 0) or 0)
    peak = getattr(result, "peak_memory", None)
    peak_gb = float(peak) / (1024 ** 3) if peak else None

    blocks = _parse_blocks(raw)

    return QwenResult(
        raw=raw,
        blocks=blocks,
        elapsed_s=elapsed,
        prompt_tokens=prompt_tokens,
        generation_tokens=gen_tokens,
        peak_memory_gb=peak_gb,
    )


# --------------------------------------------------------------------------- #
# CLI smoke (no game state — just sanity check the model loads and talks)     #
# --------------------------------------------------------------------------- #


def _smoke() -> None:
    result = qwen_decide(
        system_content="You are a concise assistant. When asked a question, answer in one short sentence.",
        user_content="What is 2 + 2? Answer ONLY with the number.",
        image_path=None,
        tools=None,
        max_tokens=64,
        enable_thinking=False,
    )
    print(f"[smoke] raw:\n{result.raw}")
    print(f"[smoke] elapsed={result.elapsed_s:.1f}s "
          f"prompt={result.prompt_tokens}t gen={result.generation_tokens}t "
          f"peak_gb={result.peak_memory_gb}")
    print(f"[smoke] {len(result.blocks)} block(s):")
    for b in result.blocks:
        print(f"  {b.kind:<12} {str(b.content)[:80]!r}")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        _smoke()
    else:
        print("usage: python -m burl.candlewax_spike.qwen_local --smoke")
