---
title: Batched eval resilience pattern
kind: topic
first_seen: 86334b8
last_updated: 86334b8
status: superseded
---

## Problem

The held-out 560-decision adapter eval (`scratch/belief_trajectory_rollout/star/eval_adapter_smoke.py`) ran sequentially at ~26s/decision = **~4h wall**. That's a fragile budget for an overnight cycle: a single Metal OOM, parent-shell death, or the [[burl-2000-harvest|mlx-lm 0.31.2 broadcast bug]] kills the whole eval, and the next iteration loses a full overnight window. Run-3c had to be re-launched from scratch after the parent-shell footgun (see [[burl-star-run3]]'s "three failed launch attempts").

Fixing this needed two layers stacked: **batched throughput** (so the eval runs in ~1.5h not 4h) **plus** the same resilience the [[batched-harvest-resilience|harvest layer]] earned through the v2 production run. Without the resilience layer, batched throughput just made every failure ~3× more likely (more memory pressure per wave) without any way to recover the partial work.

## Four-part resilience layer

The layer lives in `scratch/belief_trajectory_rollout/star/eval_adapter_smoke.py`. Smoke validation: kill the process mid-wave-1 of an n=12 batch=6 run, relaunch with `--resume-dir`, confirm `resume: 6 already done, 6 TODO` and final summary.json contains all 12 rows in order.

### 1. Atomic per-decision write + per-wave summary roll-up

Every JSON write (per-decision `trace_summary.json`, top-level `summary.json`) goes through `_atomic_write_json`: tmp file + `os.replace`. SIGKILL mid-write can never leave a half-written JSON. Mirror of the [[resumable-checkpointing|`star_mlx.py:_atomic_save_safetensors`]] pattern that trainer-fixer landed for run-2/3 recovery. The eval also rolls forward `summary.json` at the end of every wave, so a kill loses at most the in-flight wave's progress (≤6 decisions, ≤3 min wall).

### 2. In-wave per-decision OOM fallback

Each `model.step_batch(payload)` call is wrapped. On `RuntimeError` or `MemoryError` whose message matches the [[batched-harvest-resilience|harvest's OOM-classifier]] (`metal::malloc`, `Resource limit`, `broadcast_shapes`, etc.), the eval **falls back to one-decision-at-a-time generates within the same wave** instead of aborting the whole wave. The harvest's `--rerun-quarantined` retry pattern collapsed to inline single-decision retries, so a transient batch-shape OOM costs at most one decision instead of six. Single-decision OOMs append to `<out_dir>/quarantine.jsonl` and continue; they're naturally retried via `--resume-dir` on the next launch (no separate retry pass needed).

### 3. `--resume-dir` idempotency

A complete decision is one whose `decision_<gi>/trace_summary.json` parses as JSON and contains `final_play` + `n_turns` keys. Resume scans `--out-dir` for those files, skips the gi's that pass, queues the rest. Idempotent: re-running with `--resume-dir` after EVAL DONE is a no-op (`nothing to do — exiting cleanly`). Combined with the atomic-write layer, kill-then-relaunch loses ≤1 wave even under SIGKILL.

### 4. Same surface for sequential mode (`--batch-size 1`)

The atomic-write + `--resume-dir` paths cover sequential too. The pre-batched eval lacked any incremental write — a kill at decision 559 of 560 lost the entire run. The hardened sequential path writes after each decision, so a kill mid-run preserves progress under both modes. Run-3c's eval would not have needed a 4h re-run if this had been live.

## Verification

Smoke (n=6, batch=6, max_tokens=8192, run-3c adapter, 2026-04-26):

- wall=137.6s vs sequential reference 158s (1.15× at batch=6 amortizing model load — projected ~6× at full waves)
- match=2/6 vs reference 3/6 — drift = 1 decision, exactly at the spec's "≤ 1 of 6" stochastic noise threshold for temp=0.6
- legal=6/6, errors=0, bails=0, belief_used=6/6, forced=2 (gi=2, gi=5 — both also forced in the sequential reference)
- per-decision `trace_summary.json` schema bit-for-bit identical to harvest_batched output

Resume sanity (n=12, batch=6, kill@6-of-12, then `--resume-dir`):

- After kill: 6 trace_summary.json on disk (gi 0–5), 6 init-only dirs (gi 6–11), summary.json contains 6 rows.
- Resume status: `resume: 6 already done, 6 TODO` — exactly correct.
- Final summary.json: 12 rows in gi 0–11 order, no duplicates. wall=278s for the resume wave.

## CLI surface

```
--batch-size N       Default 1 (sequential parity). >1 routes through GemmaLocalNativeBatched.
--max-tokens N       Default 8192. Per-turn generation cap.
--resume-dir         Skip global_idx values whose trace_summary.json is already complete.
--n N                Number of eval decisions (sequential gi 0..N-1).
--global-indices ... Override --n with explicit gi list.
--adapter PATH       Adapter dir; omit for base-model eval.
--variant-name NAME  Default D_required_first.
```

The `--turn-cap N` flag never landed. `burl/eval/` last commit is 2026-04-28; the
eval-adapter-smoke script this resilience layer lives in (as opposed to the
metric/rescore layer under "Promotion status" below) remained in `scratch/` for the
rest of the family's active window.

## Why this matters for STaR planning

The first n=560 base eval launch hit the 5h wall projection and squeezed the run-4 chain out of the overnight window — team-lead rescoped to n=180 paired with the run-3c first 180 indices, which is a *better* analysis (paired comparison) at lower cost. That rescope was only safe because the resilience layer guaranteed the partial 18 decisions on disk would be reused on the relaunch (`resume: 18 already done, 162 TODO`), not thrown away. Without the layer, the rescope decision would have cost 25 min of GPU time to start over from scratch — small in absolute terms but compounding across the iteration cadence.

The same logic that made [[batched-harvest-resilience]] worth its complexity for the harvest pipeline applies to the eval path: turning 4h evals from "all-or-nothing" into "lose at most 3 min on any failure" is the difference between weekly STaR iterations and monthly ones.

## Promotion status

As of 2026-04-26, the pure metric/rescore layer has a tracked home at
`burl/eval/star_metrics.py` with CLI wrapper `burl/eval/star_eval_report.py`.
The filter-only STaR corpus builder likewise has a tracked home at
`burl/train/star_corpus.py` with CLI wrapper `burl/train/build_star_corpus.py`.
The active harvest/eval runner remains under `scratch/` while a cap-12
harvest is running, so a crash/resume of that live process sees the same
script surface it started with.

## Links

[[batched-harvest-resilience]] · [[burl-star-run3]] · [[resumable-checkpointing]] · [[commit-discipline-collapse]] · [[max-tokens-2048-floor]] · [[burl]] · [[star]] · [[mlx-lm]]
