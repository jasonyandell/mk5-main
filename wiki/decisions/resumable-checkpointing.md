---
title: Resumable checkpointing for Burl STaR trainer
kind: decision
first_seen: add6a2a
last_updated: add6a2a
status: active
---

## Decision

The [[burl]] STaR trainer (`burl/train/star_mlx.py`) writes an on-disk adapter checkpoint every `--steps-per-checkpoint N` (default 100) iters and accepts a `--resume` flag that continues training from the last checkpoint. Any `Exception` raised inside the training loop also serializes the in-memory best-val snapshot to `{adapter-out}/best_on_crash/adapters.safetensors` before re-raising, so a crash never loses a healthy adapter.

## Context

Run-3 attempt 3 of the [[burl-star-run3]] experiment trained cleanly for 37 minutes (val 2.354 → 0.302 over 9 evals) then died at iter 487/1343 with `RuntimeError: [metal::malloc] Resource limit (499000) exceeded`. The `train_mlx` exception handler caught only `_EarlyStopSignal`, so the in-memory `best_params` snapshot was lost. The same failure mode applied to attempt 1 (parent shell died and `nohup` was missing). Across three attempts the run produced **zero adapters on disk** despite hours of GPU time. The recipe-lessons section of [[star]] §6 named this as the priority blocker before run-3b.

## Mechanism

- **Periodic checkpoint write.** `_TrajectoryCollector.on_train_loss_report` fires every iter (because `steps_per_report=1`). When `iter % steps_per_checkpoint == 0`, it writes:
  - `{adapter-out}/checkpoint_iter{N}/adapters.safetensors` — the per-iter snapshot.
  - `{adapter-out}/adapters.safetensors` — atomic-replaced mirror of the latest snapshot (so `mlx_lm.load(adapter_path=...)` reads the freshest weights without knowing the iter number).
  - `{adapter-out}/checkpoint_state.json` — `{iter, best_val_loss, best_val_iter, best_params_in_snapshot, ts}`.
  - The serialized weights are the **best-val snapshot** when one exists (preferred — overfitting protection), otherwise the current trainable params (fallback for crashes before the first eval).
- **Atomic write.** Every `safetensors` and `json` write goes through a sibling `.tmp` path then `os.replace`, so a process kill mid-write never produces a half-written adapter. mlx-lm's `save_safetensors` enforces a `.safetensors` extension; the tmp path uses the form `adapters.tmp.safetensors`.
- **Checkpoint failure is non-fatal.** `_write_checkpoint` wraps the write in `try/except Exception` and logs `[ckpt] WARN`. A failed checkpoint is strictly less bad than aborting a healthy run.
- **Resume.** `--resume` reads `checkpoint_state.json`, calls `model.load_weights(adapters.safetensors, strict=False)` after `linear_to_lora_layers` is applied, restores `best_val_loss` / `best_val_iter` on the new collector, sets `iter_offset` so logged steps stay continuous, and trims `total_iters` by the resumed count. Optimizer + LR schedule **rebuild from scratch** on the trimmed budget — this is a crash-recovery feature, not bit-perfect continuation.
- **Generic-exception catch.** `train_mlx` now wraps the `train(...)` call in `try/except Exception`. On any non-`_EarlyStopSignal` exception, `_save_crash_snapshot` writes `{adapter-out}/best_on_crash/adapters.safetensors` + `crash_info.json` (including `iter_when_crashed`, `exc_type`, `exc_msg`, full traceback), then re-raises so the failure stays visible to the wrapping shell.

## Tradeoffs

- LR schedule restart on resume is a deliberate non-goal. Restoring AdamW state is non-trivial under `--grad-accum > 1`; the recovery feature is worth more than the schedule purity. Document the resumed-from iter in `adapter_config.json` so postmortems can reason about the kink.
- `--steps-per-checkpoint=0` disables the feature entirely. Cheap default of 100 is appropriate for the sub-1500-iter Burl runs; longer runs may want 250-500 to amortize disk pressure.
- The mirror-at-top-level pattern means the most recent checkpoint is also the first thing `mlx_lm.load` finds; users wanting an older snapshot can name `checkpoint_iter{N}/` directly.

## How to use in run-4

```bash
.venv/bin/python -u burl/train/star_mlx.py \
    --corpus <corpus>/train.jsonl \
    --val-corpus <corpus>/val.jsonl \
    --adapter-out <out>/ \
    --rank 8 --lr 3e-5 --epochs 1 \
    --steps-per-eval 50 \
    --early-stop-val-rise 1.02 --early-stop-patience 2 \
    --steps-per-checkpoint 100 \
    --preserve-thoughts
```

If the run crashes (metal-OOM, SIGTERM, parent-shell death), relaunch with the same args plus `--resume`:

```bash
.venv/bin/python -u burl/train/star_mlx.py ... --resume
```

The trainer reads `checkpoint_state.json`, loads the latest adapter, and trims `total_iters` by the resumed count. `best_on_crash/adapters.safetensors` is also a valid recovery target if the periodic checkpoints lag behind the in-memory best.

## Tested

12 new unit tests in `burl/train/test_star_mlx.py` cover atomic writes, the periodic-checkpoint hook, fallback-to-current-weights when no eval has fired, resume-state read paths, iter-offset propagation, and crash-snapshot persistence (best-snapshot path + current-weights fallback). End-to-end recovery validated on a 50-row tiny corpus: kill at iter ~30, relaunch with `--resume`, verify `iter_offset` log line + correct trimmed iter budget + checkpoints continue from the resumed iter (see `scratch/resume_smoke/`).

## Related

- [[burl-star-run3]] — the run that motivated this decision.
- [[star]] §"Burl 2000-decision corpus ready" — recipe lesson #6, now resolved.
- [[batched-harvest-resilience]] — sibling pattern on the harvest side.
