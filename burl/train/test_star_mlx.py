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
    _EarlyStopSignal,
    _TrajectoryCollector,
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


# --- Val-loss callback: best-tracking + early-stop ------------------------


def test_trajectory_collector_no_val_corpus_no_op():
    """Without --val-corpus the callback's val-tracking machinery is inert."""
    cb = _TrajectoryCollector(losses=[])
    cb.on_train_loss_report(
        {"iteration": 1, "train_loss": 1.5, "learning_rate": 1e-4}
    )
    assert len(cb.losses) == 1 and cb.losses[0]["loss"] == 1.5
    assert cb.val_losses == []
    assert cb.best_val_iter == -1
    assert cb.best_params is None


def test_trajectory_collector_tracks_best_val_loss():
    """Without early-stop enabled, the collector still records the best
    val loss + iter so adapter_config.json can report it."""
    cb = _TrajectoryCollector(losses=[])
    cb.on_val_loss_report({"iteration": 50, "val_loss": 1.2})
    cb.on_val_loss_report({"iteration": 100, "val_loss": 0.9})
    cb.on_val_loss_report({"iteration": 150, "val_loss": 1.1})  # rose; no abort
    cb.on_val_loss_report({"iteration": 200, "val_loss": 0.8})  # new best
    assert [v["step"] for v in cb.val_losses] == [50, 100, 150, 200]
    assert cb.best_val_loss == 0.8
    assert cb.best_val_iter == 200


def test_trajectory_collector_early_stop_after_patience_breaches():
    """Aborts once val_loss has stayed > rise * best for `patience`
    consecutive evals. Streak resets on any improvement or any eval below
    the threshold."""
    cb = _TrajectoryCollector(
        losses=[], early_stop_val_rise=1.3, early_stop_patience=2
    )
    # Establish baseline best=1.0 at iter 50.
    cb.on_val_loss_report({"iteration": 50, "val_loss": 1.0})
    # Eval 2: 1.31 > 1.0 * 1.3 = 1.3 → streak 1, no abort yet.
    cb.on_val_loss_report({"iteration": 100, "val_loss": 1.31})
    assert cb.over_threshold_streak == 1
    # Eval 3: 1.40 > 1.3 → streak 2, abort.
    with pytest.raises(_EarlyStopSignal, match="best=1.0000"):
        cb.on_val_loss_report({"iteration": 150, "val_loss": 1.40})
    assert cb.best_val_iter == 50


def test_trajectory_collector_streak_resets_on_improvement():
    cb = _TrajectoryCollector(
        losses=[], early_stop_val_rise=1.3, early_stop_patience=2
    )
    cb.on_val_loss_report({"iteration": 50, "val_loss": 1.0})
    cb.on_val_loss_report({"iteration": 100, "val_loss": 1.40})  # streak 1
    assert cb.over_threshold_streak == 1
    cb.on_val_loss_report({"iteration": 150, "val_loss": 0.8})   # new best
    assert cb.over_threshold_streak == 0
    assert cb.best_val_loss == 0.8
    # Now need 2 fresh consecutive breaches over the new best (0.8 * 1.3 = 1.04).
    cb.on_val_loss_report({"iteration": 200, "val_loss": 1.05})  # streak 1
    assert cb.over_threshold_streak == 1
    with pytest.raises(_EarlyStopSignal):
        cb.on_val_loss_report({"iteration": 250, "val_loss": 1.10})


def test_trajectory_collector_streak_resets_on_below_threshold_eval():
    """An eval that's higher than best but still below the rise threshold
    counts as 'no breach' and resets the streak."""
    cb = _TrajectoryCollector(
        losses=[], early_stop_val_rise=1.3, early_stop_patience=2
    )
    cb.on_val_loss_report({"iteration": 50, "val_loss": 1.0})
    cb.on_val_loss_report({"iteration": 100, "val_loss": 1.40})  # streak 1
    cb.on_val_loss_report({"iteration": 150, "val_loss": 1.20})  # below 1.3, reset
    assert cb.over_threshold_streak == 0
    cb.on_val_loss_report({"iteration": 200, "val_loss": 1.40})  # streak 1
    assert cb.over_threshold_streak == 1


def test_trajectory_collector_disabled_when_rise_is_none():
    """Without --early-stop-val-rise, even a 100x val_loss spike is silent."""
    cb = _TrajectoryCollector(losses=[], early_stop_val_rise=None)
    cb.on_val_loss_report({"iteration": 50, "val_loss": 1.0})
    cb.on_val_loss_report({"iteration": 100, "val_loss": 100.0})
    cb.on_val_loss_report({"iteration": 150, "val_loss": 1000.0})
    # No exception. Best is still tracked.
    assert cb.best_val_loss == 1.0


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
