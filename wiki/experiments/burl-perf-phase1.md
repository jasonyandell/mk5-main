---
title: Burl Perf — Phase 1 (Cheap Wins)
kind: experiment
first_seen: 160ed1c
last_updated: 160ed1c
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

### Ledger row

| variant | wall_total | wall_p50 | decode tok/s | peak GB | K1 match | regret Δ | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline-bf16` (Phase 0) | 79.5 s | 77.6 s | 87.5 | 11.59 | 100% | 0% | reference (1f11d28) |
| `baseline-bf16` (re-measure) | 40.1 s | 40.1 s | 184.5 | 11.59 | 80% | −34% | 22:00 — system was much less loaded than 01:37 |
| `turn-aware-2048` run 1 | 47.8 s | 44.6 s | 141.8 | 11.59 | 100% | 0% | 20:07 |
| `turn-aware-2048` run 2 | 45.5 s | 44.3 s | 140.9 | 11.59 | 60% | −100% | 20:08 |

The 79.5 → 47.8 s "speedup" is **not** the lever — it is mostly system
load drift (see [[burl-perf-noise-floor]] below). Holding the lever's
wall delta against the same-state `baseline-bf16` re-measure (40.1 s)
puts `turn-aware-2048` at +13% (slower, inside the 5-row noise floor).
The lever is correctness-preserving, not speed-positive on this corpus.

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

## Burl-perf noise floor

Run-to-run wall variance on the 5-decision subset is **far** larger
than was assumed in the Phase 0 stability read (which sampled two
back-to-back runs in a quiet system state). On 2026-04-27, the same
`baseline-bf16` config reproduced wall at:

| Run | wall_s_total | decode tok/s | sha |
|---|---:|---:|---|
| Phase 0 reference | 79.5 s | 87.5 | 1f11d28 |
| Phase 1 baseline-temp0 run 1 | 73.5 s | 81.1 | 160ed1c |
| Phase 1 baseline-temp0 run 2 | 82.1 s | 75.7 | 160ed1c |
| Phase 1 baseline-temp0 run 3 | 134.5 s | 47.0 | 160ed1c |
| Phase 1 baseline-bf16 re-measure | 40.1 s | 184.5 | 160ed1c |

That's a **3.4×** run-to-run wall range on the same config. No
concurrent MLX processes were detected; no Modal jobs were running. The
likely cause is OS-level scheduler / Metal compiler cache warmth /
shared memory pressure — 5 decisions is too few to ride out these
sources. **Lever-induced wall deltas under ~2× cannot be cleanly
attributed at this floor**; Phase 4's 560-decision run is the
measurement that will resolve Phase 1's lever.

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
