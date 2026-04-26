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

import json

import pytest

from burl.train.star_mlx import (
    GEMMA4_TURN_TERMINATOR,
    PreserveThoughtsDataset,
    _EarlyStopSignal,
    _TrajectoryCollector,
    _atomic_save_safetensors,
    _flatten_trainable,
    _load_resume_state,
    _save_crash_snapshot,
    _write_checkpoint,
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


# --- Resumable checkpointing -----------------------------------------------


@pytest.fixture
def tiny_mlx_model():
    """Minimal mlx Module so checkpoint helpers have something to flatten."""
    pytest.importorskip("mlx")
    import mlx.nn as nn

    return nn.Linear(4, 4)


def test_atomic_save_safetensors_roundtrip(tmp_path, tiny_mlx_model):
    import mlx.core as mx

    weights = _flatten_trainable(tiny_mlx_model)
    out = tmp_path / "adapters.safetensors"
    _atomic_save_safetensors(out, weights)
    assert out.exists()
    assert not (tmp_path / "adapters.tmp.safetensors").exists(), "tmp not cleaned"
    loaded = mx.load(str(out))
    assert set(loaded.keys()) == set(weights.keys())


def test_write_checkpoint_with_best_snapshot(tmp_path, tiny_mlx_model):
    """When a best-val snapshot exists, the checkpoint mirrors that, not
    the live model weights."""
    import mlx.core as mx

    best = _flatten_trainable(tiny_mlx_model)
    # Mutate the model to drift from `best`; checkpoint should still serialize `best`.
    tiny_mlx_model.weight = mx.zeros_like(tiny_mlx_model.weight)
    _write_checkpoint(
        checkpoint_dir=tmp_path,
        model=tiny_mlx_model,
        best_params=best,
        iter_=10,
        best_val_loss=0.5,
        best_val_iter=8,
    )
    assert (tmp_path / "adapters.safetensors").exists()
    assert (tmp_path / "checkpoint_iter10" / "adapters.safetensors").exists()
    state_path = tmp_path / "checkpoint_state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["iter"] == 10
    assert state["best_val_loss"] == 0.5
    assert state["best_val_iter"] == 8
    assert state["best_params_in_snapshot"] is True

    # Mirror must be byte-identical to the snapshot dir adapter.
    mirror = mx.load(str(tmp_path / "adapters.safetensors"))
    snap = mx.load(str(tmp_path / "checkpoint_iter10" / "adapters.safetensors"))
    for k in mirror:
        assert mx.array_equal(mirror[k], snap[k]).item()


def test_write_checkpoint_falls_back_to_current_weights(tmp_path, tiny_mlx_model):
    """No best snapshot yet (e.g. crash before first eval) -- the checkpoint
    must capture current trainable params so resume has *something* to load."""
    _write_checkpoint(
        checkpoint_dir=tmp_path,
        model=tiny_mlx_model,
        best_params=None,
        iter_=5,
        best_val_loss=float("inf"),
        best_val_iter=-1,
    )
    state = json.loads((tmp_path / "checkpoint_state.json").read_text())
    assert state["best_params_in_snapshot"] is False
    assert state["best_val_loss"] is None  # inf becomes None on disk


def test_write_checkpoint_failure_does_not_raise(tmp_path, tiny_mlx_model):
    """A doomed write (read-only dir) must NOT abort training."""
    import json as _json

    bad_dir = tmp_path / "doomed"
    bad_dir.mkdir()
    bad_dir.chmod(0o400)  # read-only -> mkdir of subdir fails
    try:
        # Should print a warning and return cleanly.
        _write_checkpoint(
            checkpoint_dir=bad_dir,
            model=tiny_mlx_model,
            best_params=_flatten_trainable(tiny_mlx_model),
            iter_=1,
            best_val_loss=1.0,
            best_val_iter=1,
        )
    finally:
        bad_dir.chmod(0o700)
    # No state written.
    assert not (bad_dir / "checkpoint_state.json").exists()


def test_periodic_checkpoint_callback_writes_every_n_iters(tmp_path, tiny_mlx_model):
    """The collector hook fires a write on every Nth iter (and only on Nth)."""
    cb = _TrajectoryCollector(
        losses=[],
        model=tiny_mlx_model,
        checkpoint_dir=tmp_path,
        steps_per_checkpoint=3,
    )
    # Pretend mlx-lm reported iters 1..7 with steps_per_report=1.
    for i in range(1, 8):
        cb.on_train_loss_report(
            {"iteration": i, "train_loss": 1.0, "learning_rate": 1e-4}
        )
    # Fires at iter 3 and 6; latest snapshot wins on the mirror.
    assert (tmp_path / "checkpoint_iter3").exists()
    assert (tmp_path / "checkpoint_iter6").exists()
    assert not (tmp_path / "checkpoint_iter4").exists()
    state = json.loads((tmp_path / "checkpoint_state.json").read_text())
    assert state["iter"] == 6  # mirror = latest write


def test_periodic_checkpoint_off_when_steps_zero(tmp_path, tiny_mlx_model):
    """steps_per_checkpoint=0 disables the hook entirely (no disk writes)."""
    cb = _TrajectoryCollector(
        losses=[],
        model=tiny_mlx_model,
        checkpoint_dir=tmp_path,
        steps_per_checkpoint=0,
    )
    for i in range(1, 11):
        cb.on_train_loss_report(
            {"iteration": i, "train_loss": 1.0, "learning_rate": 1e-4}
        )
    assert not (tmp_path / "checkpoint_state.json").exists()
    assert not list(tmp_path.glob("checkpoint_iter*"))


def test_load_resume_state_returns_none_when_missing(tmp_path):
    assert _load_resume_state(tmp_path) is None


def test_load_resume_state_skips_when_adapter_missing(tmp_path):
    """A bare state file with no adapter is NOT a valid resume target -- the
    user might have wiped weights but left the JSON; we must restart cold."""
    (tmp_path / "checkpoint_state.json").write_text(json.dumps({"iter": 100}))
    assert _load_resume_state(tmp_path) is None


def test_load_resume_state_reads_paired_files(tmp_path, tiny_mlx_model):
    _write_checkpoint(
        checkpoint_dir=tmp_path,
        model=tiny_mlx_model,
        best_params=_flatten_trainable(tiny_mlx_model),
        iter_=42,
        best_val_loss=0.7,
        best_val_iter=40,
    )
    state = _load_resume_state(tmp_path)
    assert state is not None
    assert state["iter"] == 42
    assert state["best_val_loss"] == 0.7
    assert state["best_val_iter"] == 40


def test_iter_offset_shifts_logged_steps(tiny_mlx_model):
    """On resume, iter_offset shifts the trajectory's step axis so plots
    don't reset to zero mid-run."""
    cb = _TrajectoryCollector(
        losses=[],
        model=tiny_mlx_model,
        iter_offset=100,
    )
    cb.on_train_loss_report(
        {"iteration": 1, "train_loss": 0.5, "learning_rate": 1e-4}
    )
    cb.on_val_loss_report({"iteration": 5, "val_loss": 0.6})
    assert cb.losses[0]["step"] == 101
    assert cb.val_losses[0]["step"] == 105
    assert cb.best_val_iter == 105  # offset propagates into best-tracker


def test_save_crash_snapshot_writes_best_when_present(tmp_path, tiny_mlx_model):
    cb = _TrajectoryCollector(losses=[], model=tiny_mlx_model)
    cb.best_params = _flatten_trainable(tiny_mlx_model)
    cb.best_val_loss = 0.42
    cb.best_val_iter = 7
    cb.last_iter = 9
    exc = RuntimeError("metal::malloc Resource limit (499000) exceeded")
    _save_crash_snapshot(tmp_path, cb, exc, tiny_mlx_model)
    crash_dir = tmp_path / "best_on_crash"
    assert (crash_dir / "adapters.safetensors").exists()
    info = json.loads((crash_dir / "crash_info.json").read_text())
    assert info["iter_when_crashed"] == 9
    assert info["best_val_loss"] == 0.42
    assert info["best_val_iter"] == 7
    assert info["best_params_in_snapshot"] is True
    assert info["exc_type"] == "RuntimeError"
    assert "metal::malloc" in info["exc_msg"]


def test_save_crash_snapshot_falls_back_when_no_best(tmp_path, tiny_mlx_model):
    """Crash before first eval -- still persist *something* (current weights)."""
    cb = _TrajectoryCollector(losses=[], model=tiny_mlx_model)
    cb.last_iter = 3
    _save_crash_snapshot(tmp_path, cb, ValueError("boom"), tiny_mlx_model)
    info = json.loads((tmp_path / "best_on_crash" / "crash_info.json").read_text())
    assert info["best_params_in_snapshot"] is False
    assert info["best_val_iter"] == -1


def test_checkpoint_dir_is_none_when_steps_zero(tmp_path, tiny_mlx_model):
    """Defensive: even if a caller wires checkpoint_dir + steps_per_checkpoint=0,
    no checkpoint should land. (We rely on this in train_mlx.)"""
    cb = _TrajectoryCollector(
        losses=[],
        model=tiny_mlx_model,
        checkpoint_dir=tmp_path,
        steps_per_checkpoint=0,
    )
    cb.on_train_loss_report(
        {"iteration": 100, "train_loss": 0.5, "learning_rate": 1e-4}
    )
    assert not (tmp_path / "checkpoint_state.json").exists()
