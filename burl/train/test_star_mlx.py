"""Pin the MLX-LM port of the preserve_thoughts path (burl/train/star_mlx.py).

Mirrors ``burl/train/test_formatting_func.py``. The PyTorch/SFT reference
lives in ``star.py``; this file pins the behavioral parity for the mlx-lm
side:

  * ``PreserveThoughtsDataset.process(row)`` raises ValueError on malformed
    rows (wrong arity / wrong role order).
  * The 8 Gemma 4 boundary tokens stay atomic under the repo tokenizer.
  * encode/decode round-trip through ``process()`` preserves the thought
    prose AND the commit_play tool call (byte-perfect recovery).
  * Default MLX-LM ``ChatDataset`` DROPS the thought prose on this same row
    -- regression pin: if that ever changes, strip_thinking was removed
    upstream and the whole preserve_thoughts layer becomes unnecessary.

Requires ``mlx-community/gemma-4-e2b-it-bf16`` in the HF cache. If absent,
tokenizer-dependent tests skip.
"""
from __future__ import annotations

import pytest

from burl.train.star_mlx import (
    GEMMA4_TURN_TERMINATOR,
    PreserveThoughtsDataset,
)

MODEL_ID = "mlx-community/gemma-4-e2b-it-bf16"

ATOMIC_BOUNDARY_TOKENS = [
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
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


# --- Validation / malformed rows -------------------------------------------


def test_preserve_thoughts_rejects_malformed_rows(tokenizer):
    ds = PreserveThoughtsDataset([], tokenizer)
    bad_rows = [
        {"messages": [{"role": "user", "content": "x"}]},
        {
            "messages": [
                {"role": "assistant", "content": "x"},
                {"role": "user", "content": "y"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "user", "content": "y"},
            ]
        },
    ]
    for row in bad_rows:
        with pytest.raises(ValueError, match="preserve_thoughts"):
            ds.process(row)


# --- Atomic-token pins ------------------------------------------------------


@pytest.mark.parametrize("marker", ATOMIC_BOUNDARY_TOKENS)
def test_boundary_token_is_atomic(tokenizer, marker):
    """All 8 Gemma 4 boundary tokens must be single-token. If any ever
    fragments, PreserveThoughtsDataset.process would emit sequences the
    tokenizer cannot round-trip cleanly.
    """
    ids = tokenizer.encode(marker, add_special_tokens=False)
    assert len(ids) == 1, (
        f"marker {marker!r} fragmented to {len(ids)} ids={ids} -- "
        f"PreserveThoughtsDataset is built on atomic-boundary assumption; "
        f"re-check Gemma tokenizer before training"
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
        f"{GEMMA4_TURN_TERMINATOR!r} -- tokenizer may have changed"
    )


# --- Core behavior: thoughts survive encode->decode round-trip -------------


def test_preserve_thoughts_round_trip_keeps_thought_prose(tokenizer):
    """Core pin: the whole point of porting preserve_thoughts to MLX-LM.

    ``process(row)`` returns (tokens, 0) where tokens decode back to a
    string containing the thought prose AND the commit_play tool call.
    """
    row = _row_with_thought()
    ds = PreserveThoughtsDataset([row], tokenizer)
    tokens, offset = ds.process(row)

    assert offset == 0, (
        f"preserve_thoughts returned offset={offset}; must be 0 so every "
        f"token (including thoughts) is trainable"
    )
    assert len(tokens) > 0, "process produced empty tokens"

    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    assert "THOUGHT_PROSE_MUST_SURVIVE" in decoded, (
        "tokenizer round-trip dropped thought prose; len(tokens)="
        f"{len(tokens)}"
    )
    assert "<|channel>thought" in decoded, "channel marker lost in tokenize"
    assert "call:commit_play{domino_id:27}" in decoded, (
        "commit_play tool call lost in round-trip"
    )


def test_preserve_thoughts_user_boundary_canonical(tokenizer):
    """User turn must come from apply_chat_template so ``<bos><|turn>user...``
    is canonical -- otherwise inference prompts diverge from training.
    """
    row = _row_with_thought()
    ds = PreserveThoughtsDataset([row], tokenizer)
    tokens, _ = ds.process(row)
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)

    assert decoded.startswith("<bos>") or decoded.startswith("<|turn>"), (
        f"decoded output does not start with Gemma 4 turn markers: "
        f"{decoded[:40]!r}"
    )
    assert "<|turn>user\n" in decoded, "user turn opener missing"
    assert "<|turn>model\n" in decoded, "model turn opener missing"
    assert decoded.endswith(GEMMA4_TURN_TERMINATOR.strip()) or decoded.endswith(
        GEMMA4_TURN_TERMINATOR
    ), f"decoded output does not end with terminator: ...{decoded[-30:]!r}"


def test_thought_tokens_appear_as_subsequence(tokenizer):
    """Pin contiguous-subsequence preservation: the thought-token id
    sequence must appear inside the full tokenized output (no splits,
    no rewrites). Mirrors test_formatting_func::
    test_labels_include_thought_token_positions.
    """
    row = _row_with_thought()
    ds = PreserveThoughtsDataset([row], tokenizer)
    tokens, _ = ds.process(row)

    thought_ids = tokenizer.encode(
        "THOUGHT_PROSE_MUST_SURVIVE", add_special_tokens=False
    )
    assert thought_ids, "thought prose tokenized to zero ids"
    n = len(thought_ids)
    windows = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    assert tuple(thought_ids) in windows, (
        "thought ids not a contiguous subsequence of process() output"
    )

    # Thought tokens must not collide with pad_id (would be masked out of loss).
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        assert pad_id not in thought_ids, (
            f"thought prose contains pad_id={pad_id} -- would be masked out"
        )


# --- Regression pin: default path DOES strip the thought -------------------


def test_default_chat_dataset_strips_thoughts(tokenizer):
    """Direct A/B with mlx-lm's built-in ``ChatDataset`` on the SAME row:
    the default path runs ``apply_chat_template(messages)``, which invokes
    Gemma 4's strip_thinking macro and drops the thought prose.

    If this test ever flips (prose survives), strip_thinking was removed
    upstream -- iter-4 motivation disappears, re-investigate before firing.
    """
    from mlx_lm.tuner.datasets import ChatDataset

    row = _row_with_thought()
    default_ds = ChatDataset([row], tokenizer, chat_key="messages", mask_prompt=False)
    tokens, _ = default_ds.process(row)
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)

    # Default path tokenizes the commit tool call (that's the SFT signal
    # iter-0/1/2/3 already learn), but strips the thought prose.
    assert "call:commit_play{domino_id:27}" in decoded, (
        "default ChatDataset dropped the commit_play tool call -- this "
        "would break all iter-0..3 behavior, not just iter-4"
    )
    assert "THOUGHT_PROSE_MUST_SURVIVE" not in decoded, (
        "apply_chat_template no longer strips thoughts -- this would mean "
        "iter-0/1/2/3 adapters WERE actually trained on thought tokens, "
        "which invalidates the SPIKE_REPORT reading. Re-investigate "
        "before firing iter-4."
    )


# --- Dunder methods pin (mlx-lm iterate_batches / CacheDataset need these) --


def test_preserve_thoughts_dataset_has_len_and_getitem(tokenizer):
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q {i}"},
                {"role": "assistant", "content": f"a {i}"},
            ]
        }
        for i in range(3)
    ]
    ds = PreserveThoughtsDataset(rows, tokenizer)
    assert len(ds) == 3
    # __getitem__ must return the raw row (CacheDataset uses it that way).
    assert ds[0] == rows[0]
    # process on a well-formed row should succeed.
    toks, off = ds.process(rows[0])
    assert off == 0
    assert isinstance(toks, list) and len(toks) > 0
