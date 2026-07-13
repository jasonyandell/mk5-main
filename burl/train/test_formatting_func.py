"""Pin the preserve_thoughts formatting_func behavior for iter-4+ training.

The SFTTrainer path used by iter-0/1/2/3 calls
``tokenizer.apply_chat_template`` per row, which invokes Gemma 4's
``strip_thinking()`` macro and silently drops every ``<|channel>thought
... <channel|>`` region from assistant content BEFORE tokenization. The
reasoning-style signal therefore never reaches the SFT loss function.

``burl.train.star.build_preserve_thoughts_formatting_func`` bypasses the
chat template on the assistant side: user turn is still rendered via
``apply_chat_template(add_generation_prompt=True)`` so the canonical
``<bos><|turn>user\\n...<turn|>\\n<|turn>model\\n`` prefix is preserved,
but the assistant span is appended verbatim (thoughts intact) followed
by the Gemma 4 turn terminator ``<turn|>\\n``.

This file pins:

  - Pure-string structural correctness (no tokenizer needed).
  - ``<|turn>`` / ``<turn|>`` / ``<|channel>`` / ``<channel|>`` /
    ``<|tool_call>`` / ``<tool_call|>`` are atomic tokens — if any of
    those ever fragment, the formatting_func's assumption breaks.
  - Round-trip: encode(formatting_func(row)) -> decode includes the
    thought prose AND the commit_play tool call.
  - Regression-pin parity: ``apply_chat_template`` on the same row DOES
    strip the thought prose (so the default iter-0/1/2/3 path is
    unchanged — the formatting_func is the new thing, not a swap).

Requires the Gemma 4 tokenizer cache (~/.cache/huggingface/hub/
models--google--gemma-4-E2B-it). If absent, tokenizer-dependent tests
skip; the pure-string tests still run.
"""
from __future__ import annotations

import pytest

from burl.train.star import (
    GEMMA4_TURN_TERMINATOR,
    build_preserve_thoughts_formatting_func,
)

MODEL_ID = "google/gemma-4-E2B-it"

ATOMIC_BOUNDARY_TOKENS = [
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
]


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except (OSError, ValueError) as e:
        pytest.skip(f"Gemma 4 tokenizer not in HF cache: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _row_with_thought() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "Pick the next play."},
            {
                "role": "assistant",
                "content": (
                    "<|channel>thought\n"
                    "THOUGHT_PROSE_MUST_SURVIVE\n"
                    "I'll lead my highest trump to draw out opponents.\n"
                    "<channel|>"
                    "<|tool_call>call:trump_declared{}<tool_call|>"
                    "<|tool_response>{\"declaration\":\"sevens\"}<tool_response|>"
                    "<|tool_call>call:commit_play{domino_id:27}<tool_call|>"
                ),
            },
        ],
    }


# --- Structural tests (no tokenizer) ---------------------------------------

def test_formatting_func_rejects_malformed_rows(tokenizer):
    fmt = build_preserve_thoughts_formatting_func(tokenizer)
    bad_rows = [
        {"messages": [{"role": "user", "content": "x"}]},
        {"messages": [
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "y"},
        ]},
        {"messages": [
            {"role": "user", "content": "x"},
            {"role": "user", "content": "y"},
        ]},
    ]
    for row in bad_rows:
        with pytest.raises(ValueError, match="preserve_thoughts"):
            fmt(row)


# --- Atomic-token pins ------------------------------------------------------

@pytest.mark.parametrize("marker", ATOMIC_BOUNDARY_TOKENS)
def test_turn_boundary_is_atomic(tokenizer, marker):
    """If any of these fragment into multiple ids, the formatting_func
    would emit sequences the trainer can't round-trip cleanly."""
    ids = tokenizer.encode(marker, add_special_tokens=False)
    assert len(ids) == 1, (
        f"marker {marker!r} fragmented to {len(ids)} ids={ids} — "
        f"preserve_thoughts formatting_func is built on atomic-boundary "
        f"assumption; re-check Gemma tokenizer before training"
    )


def test_turn_terminator_constant_matches_tokenizer(tokenizer):
    """``GEMMA4_TURN_TERMINATOR`` must be what apply_chat_template emits."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert rendered.endswith(GEMMA4_TURN_TERMINATOR), (
        f"chat template ends with {rendered[-20:]!r}, not "
        f"{GEMMA4_TURN_TERMINATOR!r} — tokenizer may have changed"
    )


# --- The core behavior: thoughts survive ------------------------------------

def test_formatting_func_preserves_thought_prose(tokenizer):
    """Core pin: the whole point of T17.

    formatting_func(row) must produce a string containing the thought
    prose AND the commit_play tool call. Round-trip through
    encode/decode must keep both visible.
    """
    row = _row_with_thought()
    fmt = build_preserve_thoughts_formatting_func(tokenizer)
    text = fmt(row)

    # Raw string: thought prose and commit_play both present.
    assert "THOUGHT_PROSE_MUST_SURVIVE" in text, (
        "formatting_func dropped the thought prose — strip_thinking "
        "leaked in somewhere"
    )
    assert "call:commit_play{domino_id:27}" in text, (
        "formatting_func dropped the commit tool call"
    )

    # Tokenizer round-trip: encode -> decode recovers both.
    ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    assert "THOUGHT_PROSE_MUST_SURVIVE" in decoded, (
        "tokenizer round-trip dropped thought prose; ids count={}".format(len(ids))
    )
    assert "<|channel>thought" in decoded, "channel marker lost in tokenize"
    assert "call:commit_play{domino_id:27}" in decoded


def test_formatting_func_preserves_user_turn_boundary(tokenizer):
    """The user turn must be rendered via apply_chat_template so
    ``<bos><|turn>user...`` is canonical — otherwise inference-time
    prompts won't match training-time ones."""
    row = _row_with_thought()
    fmt = build_preserve_thoughts_formatting_func(tokenizer)
    text = fmt(row)
    # Must start with bos+user-turn-open and contain model-turn-open.
    assert text.startswith("<bos>") or text.startswith("<|turn>"), (
        f"output does not start with Gemma 4 turn markers: {text[:40]!r}"
    )
    assert "<|turn>user\n" in text, "user turn opener missing"
    assert "<|turn>model\n" in text, "model turn opener missing"
    # Must end with the turn terminator so the model learns when to stop.
    assert text.endswith(GEMMA4_TURN_TERMINATOR), (
        f"output does not end with terminator: ...{text[-30:]!r}"
    )


def test_formatting_func_vs_chat_template_divergence(tokenizer):
    """Direct A/B: on the SAME row, chat-template path drops the thought,
    formatting_func preserves it. This is the whole iter-4 motivation —
    pin that we haven't accidentally broken it.
    """
    row = _row_with_thought()
    fmt = build_preserve_thoughts_formatting_func(tokenizer)

    via_formatting = fmt(row)
    via_chat_template = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )

    # The thought prose survives ONLY through formatting_func.
    assert "THOUGHT_PROSE_MUST_SURVIVE" in via_formatting
    assert "THOUGHT_PROSE_MUST_SURVIVE" not in via_chat_template, (
        "apply_chat_template no longer strips thoughts — this would mean "
        "iter-0/1/2/3 adapters WERE actually trained on thought tokens, "
        "which invalidates the SPIKE_REPORT reading. Re-investigate "
        "before firing iter-4."
    )
    # Both paths preserve the commit tool call (that's the tool-chain signal
    # iter-0/1/2/3 already learn).
    assert "call:commit_play{domino_id:27}" in via_formatting
    assert "call:commit_play{domino_id:27}" in via_chat_template


# --- Label mask: thought tokens are NOT masked ------------------------------

def test_labels_include_thought_token_positions(tokenizer):
    """SFTTrainer typically builds ``labels = input_ids`` with pad→-100. For
    rows fed via formatting_func, there's no assistant-prompt-masking step,
    so every non-pad token — including thought tokens — is a trainable
    position. Pin that the tokenization emits positive thought-token count
    and none of them are the pad id.
    """
    row = _row_with_thought()
    fmt = build_preserve_thoughts_formatting_func(tokenizer)
    text = fmt(row)
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) > 0, "formatting_func produced empty ids"

    # Isolate the thought-prose substring and confirm its tokens appear in ids
    # and none of them equal pad_id.
    thought_tokens = tokenizer.encode(
        "THOUGHT_PROSE_MUST_SURVIVE", add_special_tokens=False
    )
    assert thought_tokens, "thought prose tokenized to zero ids"
    # The contiguous thought-token subsequence should appear in the full ids.
    n = len(thought_tokens)
    windows = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    assert tuple(thought_tokens) in windows, (
        "thought tokens do not appear as a contiguous subsequence in the "
        "formatting_func output — round-trip disagreement"
    )
    # None of the thought tokens collide with pad_id, so label-mask with
    # pad→-100 will leave these as trainable positions.
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None
    assert pad_id not in thought_tokens, (
        f"thought prose contains pad_id={pad_id} — would be masked out "
        f"of the loss"
    )
