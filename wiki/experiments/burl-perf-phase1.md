---
title: Burl Perf — Phase 1 (Cheap Wins)
kind: experiment
first_seen: 160ed1c
last_updated: 1c4f063
status: active
---

## Overview

Phase 1 of the [[perf-on-the-table]] sprint: two **non-output-changing**
levers that should not change what the model emits, only stop the bench
paying for unused capacity.

| Lever | Result |
|---|---|
| Turn-aware token budgets | Wired through; lands as flat-2048 cap on this corpus (the bench's 8192 default was over-provisioned but no per-turn cap lifts further on the 5-row subset). |
| Parallel tool dispatch | Not implemented — the [[wax-museum]] gate produces 1 tool call per assistant turn at sub-millisecond execution; there is no within-turn parallelism to recover. |

Phase 1 ships the plumbing for Lever 1 (per-prompt `max_tokens` end-to-end)
plus an empirical brief about why Lever 2 isn't worth the code.

## Lever 1 — Turn-aware token budgets

### What changed

`mlx_lm.batch_generate` already accepts `max_tokens: Union[int, List[int]]`
(verified at mlx-lm 0.31.2): each prompt in a wave gets its own budget and
the BatchGenerator stops each stream at its own EOS-or-cap, whichever
comes first. Three plumbing edits:

1. **`burl/wax_museum/schemas.py`** — new `max_tokens_for_state(state)`
   policy keyed on [[wax-museum]]'s [[gate-state]]. Currently returns
   `2048` for INITIAL / AFTER_EXPLORE / AFTER_PROBE. The function is the
   knob a future scribe can re-tune; the data structure is the lever, the
   numbers are the placeholder.
2. **`burl/modal/gemma_local_batched.py`** — `step_batch(active)` now
   reads an optional `max_tokens` key per active payload and threads it
   into `batch_generate(max_tokens=List[int])`. Existing callers that
   omit the key fall back to `self.max_tokens` (zero-impact change).
3. **`burl/eval/bench_decision_latency.py`** — `--max-tokens-policy
   turn-aware` now actually does something: per-stream cap from
   `max_tokens_for_state(s.state)`, threaded through `active_payload`.
   The Phase 0 `NotImplementedError` guard is removed.

### Why the budget lands at 2048-flat (for now)

[[max-tokens-2048-floor]] establishes that `max_tokens=2048` is the
production harvest floor — sequential Burl's p99 turn is 1770 chars
(~600 tokens) and max is 2639 chars (~900 tokens). Two empirical
attempts at lower per-state caps on the perf-subset-5:

| Variant | Cap (INITIAL/EXPLORE/PROBE) | Result |
|---|---|---|
| `turn-aware-budgets` | 768 / 768 / 2048 | 2 of 5 decisions hit forced-commit (vs 0 in baseline); decision_0 turn-count went 4 → 8. |
| `turn-aware-1024` | 1024 / 1024 / 2048 | decision_0 still went 4 → 8 turns; decision_72 went from clean-commit to forced-commit on a different play. |
| `turn-aware-2048` | 2048 / 2048 / 2048 | Output preserved (K1 100%, regret_delta 0 vs the same-state baseline run). Wall delta lands inside the 5-row noise floor. |

So the brief's "256 / 512 / 2048" expectation was too aggressive: the
[[wax-museum]] protocol's turn-1 reasoning is consistently 500-700
tokens at temp=0.6, and pinching the budget tighter forces the model to
emit a tool call before it has finished its plan, which **changes the
multi-turn trajectory**. The `wax_museum`-shaped reasoning is fundamentally
multi-paragraph; the cap is a behavior switch, not a free dial.

The end-state: production [[burl-2000-harvest]] already runs at 2048,
and the bench's 8192 default was the only over-provision. Lever 1 closes
that gap (the bench can pass `--max-tokens 2048` directly, or
`--max-tokens-policy turn-aware`, with identical effect on the current
corpus).

### GPU contention with scribe-A — explained, then resolved

The first 90 minutes of Phase 1 measurement was contaminated by an
**undeclared parallel bench from scribe-A on `perf/batch`** (Phase 2,
continuous batching + prefix sharing) running on the same M5 Max GPU.
The same `baseline-bf16-temp0` config drifted from 73.5 s → 82.1 s →
134.5 s in 10 minutes; `turn-aware-budgets` (90 s) and
`turn-aware-1024` (124 s) were measured into that drift. Team-lead
paused scribe-A and re-ran. **Lesson worth remembering** (filed as
[[mlx-cohort-bench-discipline]]): Apple Silicon's unified-memory GPU
is one resource — parallel scribes on the same M5 Max contend even
when their Python processes don't see each other. Multi-agent perf
sprints need either a serialization protocol or a tagged-GPU
discipline.

### Clean-GPU re-measurement (after scribe-A paused)

All temp=0, batch=5, max_tokens=8192 baseline / per-state under lever:

| variant | wall_total | wall_p50 | decode tok/s | peak GB | gi=104 final | sha |
|---|---:|---:|---:|---:|---:|---|
| `baseline-bf16-temp0-clean` run 1 | 36.0 s | 31.3 s | 171.4 | 11.30 | 6 | 1b78269 |
| `baseline-bf16-temp0-clean` run 2 | 38.9 s | 35.1 s | 168.5 | 11.59 | 6 | 1b78269 |
| `turn-aware-256-512-2048` run 1 | 31.0 s | 29.9 s | 160.8 | 11.15 | **19** (forced) | 1b78269 |
| `turn-aware-256-512-2048` run 2 | 30.4 s | 29.7 s | 163.9 | 11.18 | **19** (forced) | 1b78269 |
| `turn-aware-2048-clean` run 1 | 38.0 s | 36.8 s | 172.8 | 11.59 | 19 (one-flip) | 1c4f063 |
| `turn-aware-2048-clean` run 2 | 46.2 s | 41.3 s | 136.2 | 11.38 | 6 | 1c4f063 |

Reads:

- Clean baseline mean: **37.5 s ± 4%** across two consecutive runs.
  This is the same config the Phase 0 reference reported as 79.5 s
  ± 0.5%; the only difference is whether scribe-A was running. The
  Phase 0 reference is now understood to be a contended measurement.
- `256/512/2048` (the original brief spec) is **18% faster (30.7 s mean)**
  but **output-changing**: 2 of 5 decisions hit forced-commit each run
  (gi=36 and gi=104), and gi=104 lands play 19 instead of baseline's
  play 6 — beyond the temp=0 MLX kernel noise that flips gi=72 between
  plays 1 and 13 in baseline-vs-baseline.
- `2048-flat` is **output-equivalent** to the unaware baseline within
  the same MLX-noise envelope. gi=104's one-time flip to 19 in run 1
  did not reproduce in run 2 — same kind of kernel-level flip the
  baseline runs show on gi=72.
- gi=72 alternating between play 1 and play 13 across all four
  baseline-or-2048-flat runs (2/4 each) is the MLX-kernel noise
  signature at temp=0 on this corpus.

### What this leaves Lever 1 as

The plumbing — per-prompt `max_tokens: List[int]` end-to-end, policy
keyed on `GateState` — ships at 2048-flat. Per the Phase 1 brief's
"K1 change → revert and move on" rule, the per-state cap stays at
2048 because every tighter setting tested (256/512/2048, 768/768/2048,
1024/1024/2048) flipped clean-commits to forced-commits.

The bench's prior 8192 default WAS a 4× over-provision — the Phase 1
plumbing closes that gap by making it easy for callers to opt in to
either flat-2048 or per-state caps once a future corpus / SFT round
gives the model less reasoning headroom to use.

## Lever 2 — Parallel tool dispatch (not shipped)

### Empirical brief against implementation

A 60-decision sample from the latest `harvest_batched_20260425_010306`
([[burl-2000-harvest]]) shows the [[wax-museum]] gate produces **exactly
one tool call per assistant turn** in 297 of 297 sampled turns. Tool
execution time per call is sub-millisecond:

| Tool | Calls sampled | Mean exec time |
|---|---:|---:|
| `belief_trajectory` | 27 | 0.000 s |
| `explore_game` | 29 | 0.000 s |
| `probe_best_case` | 32 | 0.000 s |
| `probe_worst_case` | 19 | 0.000 s |

(The `belief_trajectory` 0.000 reading is correct: the call is a
table-cached read of a precomputed payload, not a live forward pass.)

### Why the brief expected parallelism

The brief assumed Gemma 4's native chat template would produce multiple
`<|tool_call|>` blocks per assistant turn — which does happen in some
free-form prompt distributions. Under the [[wax-museum]] gate's
instruction-and-menu design, the model has converged to one-call-per-turn
("call X, then return with the result") because the gate state machine
explicitly walks the model through a single transition per turn.

### What would unlock parallel dispatch

Two prereqs would have to land first, neither of them cheap:

1. The gate state machine would need to advance multiple steps per turn
   (e.g. fold `belief_trajectory()` + `explore_game(X)` into a single
   "plan" turn). That changes [[wax-museum]]'s observable contract.
2. The prompt would need to teach the model to emit batched tool calls.
   Gemma's post-training does emit this shape, but it's hostile to the
   gate's REJECTED-on-out-of-state error path; the model would need
   re-aligned training data to do it reliably.

Neither belongs in a "cheap wins" phase. Filed as
[[questions/open]] for a future iteration if the wall budget shifts back
to tool dispatch (it currently lives in GPU compute).

## Phase 1 wall delta (compounded)

Phase 1 nets out as **0% wall delta** vs the Phase 0 baseline measured
at the same system state:

- Lever 1 is correctness-preserving (K1 100%, regret_delta 0 vs same-state
  baseline) but a no-op on wall because no turn in the perf-subset-5
  hits the budget.
- Lever 2 is not implemented (no within-turn parallelism to recover).

The plumbing landed by Lever 1 (per-prompt `max_tokens` through
`batch_generate`) is reusable by Phase 2's continuous batching path
(see `perf/batch`) and by Phase 3's quantized-runner experiment, where
prompt-length distribution may differ enough to make per-state budgets
matter again.

## Burl-perf noise floor (revised post-contention diagnosis)

The Phase 0 stability read (wall ±0.5%) sampled two back-to-back runs
in a quiet system state. The 3.4× spread (40-134 s) observed during
Phase 1's first 90 minutes turned out to be **mostly explainable by
GPU contention with scribe-A**, not residual MLX noise:

| Run | wall_s_total | decode tok/s | scribe-A active? |
|---|---:|---:|---|
| Phase 0 reference (1f11d28) | 79.5 s | 87.5 | yes (perf/batch) |
| baseline-temp0 run 1 (160ed1c, 01:53) | 73.5 s | 81.1 | yes |
| baseline-temp0 run 2 (160ed1c, 01:54) | 82.1 s | 75.7 | yes |
| baseline-temp0 run 3 (160ed1c, 02:01) | 134.5 s | 47.0 | yes |
| baseline-bf16 re-measure (160ed1c, 22:00) | 40.1 s | 184.5 | no (lull) |
| **clean run 1** (1b78269, 02:30) | **36.0 s** | **171.4** | **no (paused)** |
| **clean run 2** (1b78269, 02:30) | **38.9 s** | **168.5** | **no (paused)** |

After serialization, run-to-run on identical config is **±4% on wall,
±2% on decode tok/s** at temp=0 — much closer to Phase 0's claimed
stability budget. The remaining variance shows up at **temp=0 K1**:
gi=72 flips between plays 1 and 13 across the two clean baseline runs
(MLX kernel non-determinism at greedy decoding); other decisions
reproduce exactly. So **K1 grade match between two clean baseline
runs at temp=0 is 80%, not 100%** — the brief's expectation that
temp=0 would round to 100% K1 was just wrong for MLX-LM 0.31.2.

## Links

[[perf-on-the-table]] · [[burl-perf-phase0]] · [[max-tokens-2048-floor]]
· [[wax-museum]] · [[mlx-lm]] · [[burl-2000-harvest]]

## Pointers

- Bench: `burl/eval/bench_decision_latency.py`
  (`--max-tokens-policy turn-aware`)
- Per-state policy: `burl/wax_museum/schemas.py:max_tokens_for_state`
- Model wrapper hook: `burl/modal/gemma_local_batched.py:step_batch`
  (reads `max_tokens` from each active-payload entry)
- Ledger: `burl/eval/results/perf_ledger.csv` (rows tagged
  `turn-aware-2048`, `turn-aware-1024`, `turn-aware-budgets`)
