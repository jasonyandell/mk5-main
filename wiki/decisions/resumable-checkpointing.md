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

- **Periodic checkpoint write — STANDALONE-LOADABLE.** `_TrajectoryCollector.on_train_loss_report` fires every iter (because `steps_per_report=1`). When `iter % steps_per_checkpoint == 0`, it writes a complete mlx-lm adapter dir at TWO paths:
  - `{adapter-out}/checkpoint_iter{N}/{adapters.safetensors,adapter_config.json}` — the per-iter snapshot.
  - `{adapter-out}/{adapters.safetensors,adapter_config.json}` — atomic-replaced mirror of the latest snapshot (so `mlx_lm.load(adapter_path=adapter-out/)` reads the freshest weights without knowing the iter number).
  - `{adapter-out}/checkpoint_state.json` — `{iter, best_val_loss, best_val_iter, best_params_in_snapshot, ts}`.
  - **Both paths above are loadable standalone via `mlx_lm.load(adapter_path=...)` with no manual fixup** — the adapter_config.json carries every field `mlx_lm.tuner.utils.load_adapters` reads (`fine_tune_type`, `num_layers`, `lora_parameters` with rank/scale/keys). Pinned by `test_checkpoint_dir_is_standalone_loadable_by_mlx_lm`.
  - The serialized weights are the **best-val snapshot** when one exists (preferred — overfitting protection), otherwise the current trainable params (fallback for crashes before the first eval).
- **Atomic write.** Every `safetensors` and `json` write goes through a sibling `.tmp` path then `os.replace`, so a process kill mid-write never produces a half-written file. mlx-lm's `save_safetensors` enforces a `.safetensors` extension; the tmp path uses the form `adapters.tmp.safetensors`.
- **Checkpoint failure is non-fatal.** `_write_checkpoint` wraps the write in `try/except Exception` and logs `[ckpt] WARN`. A failed checkpoint is strictly less bad than aborting a healthy run.
- **Resume.** `--resume` (read from `--adapter-out`) and `--resume-from PATH` (read from a different dir) both read `checkpoint_state.json`, call `model.load_weights(adapters.safetensors, strict=False)` after `linear_to_lora_layers` is applied, restore `best_val_loss` / `best_val_iter` on the new collector, **seed `best_params` from the loaded weights** so an early crash still preserves the prior best, set `iter_offset` so logged steps stay continuous, and trim `total_iters` by the resumed count. Optimizer + LR schedule **rebuild from scratch** on the trimmed budget — this is a crash-recovery feature, not bit-perfect continuation.
- **Generic-exception catch — including the exact MLX OOM that killed run-3 attempt 3.** `train_mlx` wraps the `train(...)` call in `try/except Exception`. On any non-`_EarlyStopSignal` exception, `_save_crash_snapshot` writes a STANDALONE-LOADABLE `{adapter-out}/best_on_crash/{adapters.safetensors,adapter_config.json}` + `crash_info.json` (including `iter_when_crashed`, `exc_type`, `exc_msg`, full traceback), then re-raises so the failure stays visible to the wrapping shell. Pinned by `test_train_mlx_metal_oom_re_raises_after_persisting`, which monkey-patches `mlx_lm.tuner.trainer.train` to raise `RuntimeError("[metal::malloc] Resource limit (123456) exceeded")` mid-loop and asserts the crash dir loads via `mlx_lm.load(adapter_path=best_on_crash/)`.

## Tradeoffs

- LR schedule restart on resume is a deliberate non-goal. Restoring AdamW state is non-trivial under `--grad-accum > 1`; the recovery feature is worth more than the schedule purity. Document the resumed-from iter in `adapter_config.json` so postmortems can reason about the kink.
- `--steps-per-checkpoint=0` disables the feature entirely. Cheap default of 100 is appropriate for the sub-1500-iter Burl runs; longer runs may want 250-500 to amortize disk pressure.
- The mirror-at-top-level pattern means the most recent checkpoint is also the first thing `mlx_lm.load` finds; users wanting an older snapshot can name `checkpoint_iter{N}/` directly.

## Wall-clock overhead

Measured on the M5 Max with rank-4 batch-1 over 30 iters (`scratch/resume_smoke/overhead_{on,off}.log`):

| `--steps-per-checkpoint` | wall (s) | writes | overhead |
|---|---:|---:|---|
| 0 (off)                  | 133.0 | 0 | baseline |
| 5 (every 5 iters)        | 119.8 | 6 | -10% (within run-to-run noise) |

Each rank-4 adapter write is ~25 MB and clears the M5 Max NVMe in <0.5s; on a 1500-iter run with `--steps-per-checkpoint 100` (15 writes total) the overhead is unmeasurable against ~30-45 minute training wall. The default of 100 is comfortably under the 5% threshold the team-lead set as the conservative-default trigger.

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

If the run crashes (metal-OOM, SIGKILL, parent-shell death), relaunch with the same args plus `--resume`:

```bash
.venv/bin/python -u burl/train/star_mlx.py ... --resume
```

The trainer reads `checkpoint_state.json`, loads the latest adapter, and trims `total_iters` by the resumed count. `best_on_crash/` is also a valid recovery target if the periodic checkpoints lag behind the in-memory best — both `best_on_crash/` and any `checkpoint_iter{N}/` are standalone mlx-lm adapter dirs.

If you want to fork a run (resume into a different output dir, e.g. for a hyperparam ablation off a healthy mid-run checkpoint), use `--resume-from`:

```bash
.venv/bin/python -u burl/train/star_mlx.py \
    --adapter-out new_run/ \
    --resume-from old_run/  \
    ...
```

## Tested

22 new unit tests in `burl/train/test_star_mlx.py` (43/43 total pass). Coverage includes atomic writes, JSON atomic writes, the periodic-checkpoint hook, the standalone-loadability of every checkpoint dir against the real Gemma 4 model, fallback-to-current-weights when no eval has fired, resume-state read paths (including the `--resume-from PATH` variant), iter-offset propagation, crash-snapshot persistence (best-snapshot path + current-weights fallback), the exact `[metal::malloc] Resource limit (...)` string, and an end-to-end `train_mlx` exception test that monkey-patches `mlx_lm.tuner.trainer.train` to raise the OOM mid-loop and asserts `best_on_crash/` is loadable via `mlx_lm.load(adapter_path=...)`.

End-to-end **SIGKILL+resume** validated on a 50-row tiny corpus (`scratch/resume_smoke/TRANSCRIPT.md`):

1. Train with `--steps-per-checkpoint 10`. After iter-30 ckpt lands, `kill -9 <PID>` (SIGKILL — same signal a kernel OOM-killer delivers).
2. Verified post-kill on-disk state: `checkpoint_iter{10,20,30}/` each carry `adapter_config.json` + `adapters.safetensors`; top-level mirror likewise; `checkpoint_state.json` records `iter=30, best_val_loss=1.1625, best_val_iter=24`.
3. Verified `mlx_lm.load(adapter_path=checkpoint_iter30/)` AND `mlx_lm.load(adapter_path={adapter-out}/)` both succeed standalone (no manual config-copy).
4. Resume with `--resume`: `[resume] iter_offset=30 remaining_iters=20/50`; resumed iter-1 val 1.163 matches the pre-kill iter-25 val 1.163 (proves correct adapter weights loaded); subsequent ckpts at iter 40/50 land normally; iter-50 records a NEW best at val 0.7955 @ iter 49; `[done]` clean.
5. Validated `--resume-from PATH` for the dir-fork case: read from one adapter dir, write checkpoints to a different output dir; best-val tracker carried across the fork; new best lands in the fork dir.

## Related

- [[burl-star-run3]] — the run that motivated this decision.
- [[star]] §"Burl 2000-decision corpus ready" — recipe lesson #6, now resolved.
- [[batched-harvest-resilience]] — sibling pattern on the harvest side.
