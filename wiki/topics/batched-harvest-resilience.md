---
title: Batched harvest resilience pattern
kind: topic
first_seen: 2026-04-24
last_updated: 2026-04-24
status: superseded
---

## Problem

A multi-hour batched [[burl]] harvest hits two failure classes that the 5h-sequential pilot avoided:

1. **Transient Metal OOM** — `metal::malloc`, `Resource limit`, `Resource exhausted`, or mlx-lm 0.31.2's `broadcast_shapes` bug (batch ≥14 with default `prefill_batch_size=8`). Affects one batch at a time, recoverable by skipping or retrying that batch alone.
2. **OS SIGKILL** — Apple Silicon's OS OOM killer or a hard crash takes the whole process down. Cannot be caught from Python.

Both motivate per-batch quarantine + post-hoc retry rather than checkpoint-and-resume, because batches are large enough (6–8 sequences × ~60s each) that losing one isn't fatal but losing 333 of them in sequence is.

## Three-part resilience layer

The layer lives in `scratch/belief_trajectory_rollout/harvest_batched.py` (not in `burl/` proper because it's a harvest scaffold, not part of the production agent).

### 1. OOM classifier + quarantine ledger

`_is_oom_like(exc)` returns True if the exception is a `MemoryError` OR its message contains any of: `metal::malloc`, `Resource limit`, `Resource exhausted`, `broadcast_shapes`, `out of memory`. Each `model.step_batch(...)` call is wrapped:

- If classifier matches: append a record to `<harvest_dir>/quarantine.jsonl` with `{wave_idx, gi_range, seeds, error_kind, error_message, max_tokens_at_failure, quarantined_at}`. Mark every decision in the wave `gen_failed=True` so `_finalize` skips writing `trace_summary.json` (preserves resume idempotency). Continue to the next wave.
- If classifier says no: original behavior — break wave (the model is in an unknown state, don't trust it for the rest of the wave).

Non-OOM `RuntimeError` is rare in practice; the classifier is conservative and lets the original break-wave path own unfamiliar errors.

### 2. SIGKILL recovery via wave sentinel

Before each wave's first `step_batch` call: write `<harvest_dir>/wave_in_progress.txt` with `{wave_idx, gi_range, seeds, max_tokens, started_at, pid}`. Clear it after the wave's generate loop exits. SIGKILL can't be caught — but on next `--resume`, the resume path reads the orphan sentinel, quarantines that wave (with `error_kind="SIGKILL_or_crash"`), and clears the file before continuing.

The sentinel is the single source of truth for "what wave was running when the process died." Once cleared post-wave, a stale or missing sentinel means clean state.

### 3. Retry pass: `--rerun-quarantined`

A second invocation with `--resume <dir> --rerun-quarantined`:

- Reads all `quarantine.jsonl` records.
- Excludes any gi already in `quarantine_resolved.jsonl` (succeeded on retry) or `quarantine_terminal.jsonl` (failed again — drop from corpus).
- Runs the remaining gi list at `--retry-batch-size` (default 4, half of normal 8/6, to stay below memory pressure).
- After retry success, append to `quarantine_resolved.jsonl`. After retry failure, append to `quarantine_terminal.jsonl`.

## Verification on v2 production

The v2 [[burl]] 2000-decision run (see [[burl-2000-harvest]]) ran for 5h 46m and fired **zero quarantine records** across 333 waves. The layer is currently dead code on the success path — its existence is the entire point. Smoke validation pre-launch used `--inject-oom-at-wave 3`, which raises a synthetic `RuntimeError("metal::malloc test injection")` on step 1 of wave 3; the smoke confirmed quarantine + `--rerun-quarantined` end-to-end (4 decisions resolved on retry).

## CLI surface

```
--max-tokens N            Default 2048. Override per-turn generation cap.
--rerun-quarantined       With --resume: retry quarantined gi's only.
--retry-batch-size N      Default 4. Batch size for --rerun-quarantined.
--inject-oom-at-wave N    Test only: raise fake metal::malloc at wave N.
```

## Why this matters for STaR planning

If the harvest pipeline is fragile, you can't iterate on prompt variants or scale up corpus size — every overnight run becomes a roll of the dice. The resilience layer turns a 5h harvest from "0 failures or restart" into "0 failures or quarantine 6 decisions and rerun those."

The harvest cadence this was built for stopped 2026-04-28; zero STaR-harvest commits landed after that date. The pattern is documented and correct, but no weekly-cadence follow-on ever exercised it.

## Links

[[burl-2000-harvest]] · [[wax-museum]] · [[burl]] · [[mlx-lm]] · [[max-tokens-2048-floor]] · [[063fcac]]
