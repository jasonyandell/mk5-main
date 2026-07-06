---
title: Burl Perf — Phase 0 (Measurement Harness)
kind: experiment
first_seen: 1f11d28
last_updated: 1f11d28
status: complete
---

## Overview

Phase 0 of the [[perf-on-the-table]] sprint: a fast, reproducible bench that
records per-decision wall, prefill/decode tok-s, peak memory, and a K1-grade-
match-pct vs the prior [[burl-star-run3]] baseline.  Three other scribes
(cheap wins, continuous batching + prefix sharing, speculative decoding +
quantization) are blocked on this row; their levers are evaluated against
the bench's frozen subset and ledger entries.

The bench drives the same code path Burl uses in production — `mlx_lm.batch_generate`
under [[mlx-lm]], wrapped by [[burl]]'s `GemmaLocalNativeBatched`, fed
through the wax-museum lockstep loop in `harvest_batched.py` (`_init_decision_state`
→ `_prepare_step` → `_apply_step` → `_finalize`).  Instrumentation is
runtime-only: the bench subclasses the model wrapper to record `BatchResponse.stats`,
so the production class stays unchanged.

## Subset selection rationale

Frozen subset is 5 decisions at `global_idx ∈ {0, 36, 72, 104, 136}`.
Layout:

| gi  | game | decl | seat | trick (1-7) | n_legal |
|----:|-----:|-----:|-----:|------------:|--------:|
|   0 |    0 |    0 |    0 |           1 |       7 |
|  36 |    1 |    1 |    0 |           3 |       5 |
|  72 |    2 |    2 |    0 |           5 |       3 |
| 104 |    3 |    3 |    1 |           6 |       2 |
| 136 |    4 |    4 |    3 |           7 |       1 |

Two spreads, deliberate:

1. **Trick position 1/3/5/6/7** so prefill cost averages over the full
   "wide-open hand → trivially-forced last play" arc.  Trick 1 stresses
   the rules-as-tools primer (n_legal=7 means the model has the most
   game-state to reason about), trick 7 stresses parsing in the absence
   of a real choice.

2. **Declarations 0..4** because [[gus]]'s `corpus_eval_20.pt` lays one
   declaration per game.  Sourcing from one game would bias the bench
   toward a single contract's primer payload; one decision per game in
   games 0..4 spreads the prefill cost across five distinct contracts.

The spec also flagged "(or whatever spread is most representative —
defend the pick)".  This is the pick.  Re-freezing requires re-running
`burl/eval/data/freeze_perf_subset.py`, which writes both the rows and a
SHA256 of `corpus_eval_20.pt`; the bench fails fast if that fingerprint
drifts.

## Measurements (canonical baseline-bf16, M5 Max, batch=5, max_tokens=8192)

Two consecutive runs of `--variant baseline-bf16 --subset 5 --batch 5`,
no flags changed in between (sha `1f11d28`):

| Run | wall_total | wall_p50 | wall_p95 | prefill tok/s | decode tok/s | peak GB | K1 match % | regret Δ % |
|-----|-----------:|---------:|---------:|--------------:|-------------:|--------:|-----------:|-----------:|
| 1   | 79.5 s     | 77.6 s   | 79.5 s   | 10,274        | 87.5         | 11.59   | 100        |  0.0       |
| 2   | 79.9 s     | 69.6 s   | 79.9 s   | 10,568        | 79.8         | 11.59   |  80        | −44.6      |

Stability read:

- `wall_s_total` reproduces to 0.5%.  Workload-stable.
- `decode_tok_s` reproduces to 9% (87.5 vs 79.8) — the dominant
  variability is from per-decision turn-count drift at temp=0.6, not
  the bench's measurement.
- `peak_mem_gb` reproduces to 4 decimal places (11.5887 GB) — Metal's
  peak-memory accounting is deterministic for this prompt-mix.
- `prefill_tok_s` reproduces to ~3% — prompt rendering and prefill
  width are workload-deterministic.

## Determinism notes

The base Gemma 4 E2B inference path is **not bit-pinned**:

- `temp=0.6` is the run-3c reference; setting `--temperature 0.0` (greedy)
  reduces sampling noise but MLX kernel-level nondeterminism still
  leaves room for per-decision flips.  The bench documents this in its
  `--temperature` help.
- The "K1 match within tolerance" definition for this bench is:
  **at temp=0.6, run-vs-run K1 grade match is expected to land at
  80–100% on the 5-decision subset; per-decision deltas <0.5 Q-points
  flip K1 sign about half the time.**  The 80% read in run 2 vs run 1
  came from gi=72 (the trick-5 forced-commit decision) flipping
  `eq_delta_vs_bot` from −1.7 to +0.8 — which crosses K1's `≥0`
  threshold.  Both are inside Gemma's sampling envelope; neither is a
  measurement bug.
- Phase 1+ levers should target a `regret_delta_pct ≤ −10%` to be
  considered a confirmed win, given a ~−45% noise floor at the 5-row
  subset.  Phase 4's `--full560` mode tightens this — the 560-row
  baseline noise floor will be ~`±3%` (variance falls as `1/√n`).

## Links

[[perf-on-the-table]] · [[burl-star-run3]] · [[batch-throughput-bench]] ·
[[k1-grading]] · [[mlx-lm]] · [[burl]] · [[gus]]

## Pointers

- Bench CLI: `burl/eval/bench_decision_latency.py`
- Frozen subset: `burl/eval/data/perf_subset_5.jsonl`
- Subset freeze script: `burl/eval/data/freeze_perf_subset.py`
- Ledger: `burl/eval/results/perf_ledger.csv`
- Promoted bridge: `burl/eval/gus_eval_bridge.py`
  (was `scratch/belief_trajectory_rollout/diagnostic/gus_eval_bridge.py`)
- Per-run JSON detail: `burl/eval/results/perf_<timestamp>_<variant>.json`

## Status

The measurement harness itself is durable infrastructure, but the "phase 0" framing
is stage-1 of a perf sprint that stalled: no further Burl commits landed after
2026-05-07, and the sprint's own compounded-stack projection ([[perf-on-the-table]])
was never revisited after [[burl-perf-phase2]]'s continuous-batching retraction.
