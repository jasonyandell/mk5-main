---
title: M5 Max GPU is one resource — cohort bench discipline
kind: topic
first_seen: 1c4f063
last_updated: 1c4f063
status: active
---

## What we learned the hard way

During Phase 1 of the [[perf-on-the-table]] sprint, two scribes (B on
`perf/cheap`, A on `perf/batch`) ran perf benches in parallel on the
same M5 Max. Their Python processes never saw each other (`ps aux`
returned no overlap), but **the GPU saw them just fine**:

- Same-config `baseline-bf16-temp0` reproduced wall at 73.5 s → 82.1 s
  → 134.5 s in 10 minutes.
- After team-lead paused scribe-A, the same config landed at
  **36.0 s ± 4%** across two consecutive runs.
- The Phase 0 reference of 79.5 s was retroactively understood to be
  contended too (taken while scribe-A's Phase 2 work was running on a
  parallel branch).

**The 3.4× wall variance Phase 1 first reported as "perf-subset-5
noise floor" was almost entirely cross-scribe GPU contention.** Real
single-tenant noise floor is ~4%.

## Why M5 Max can't multitask perf benches

Apple Silicon's unified-memory architecture means CPU and GPU compete
for the same SDRAM bandwidth, the Metal command queue is shared
across processes, and the GPU itself is one device. mlx-lm's
`batch_generate` does not negotiate; it issues GPU work and the OS
arbitrates. When two benches run concurrently:

- Wall doubles or worse, unequally distributed (a small bench can
  wait minutes between Metal command submissions).
- decode_tok_s tracks the contention proportionally (47-184 tok/s
  spread on the same config).
- prefill_tok_s is more robust because it's a single bulk dispatch.
- peak_mem_gb grows monotonically across the session as the model
  cache expands; both benches share that pressure.

This is invisible at the Python/process layer. The detection signal
is the per-run JSON's `decode_tok_s` field — if it's >2× off the
recently-measured floor, suspect contention before suspecting code.

## The discipline

For multi-scribe perf sprints on shared M5 Max:

1. **Serialize.** Only one scribe runs the bench at a time. Other
   scribes do non-GPU work (wiki, code, audit) during their pause.
2. **Tag every ledger row** with `notes` describing the system
   state (e.g. "phase1 lever1 256/512/2048 vs clean baseline" makes
   the contention story recoverable; "phase 1 baseline" doesn't).
3. **Sanity-check decode_tok_s** before trusting wall. If decode is
   <80% of the recent clean floor (~170 tok/s for batch=5 bf16
   Gemma 4 E2B at temp=0), the run is probably contended.
4. **Re-bench after team-lead serializes.** Don't comparison-shop
   stale rows; ledger is append-only but the comparison should always
   be against a same-session clean baseline.

## Detection signal: decode_tok_s

For Burl's perf-subset-5 at batch=5, bf16, temp=0, the clean-GPU
floor is **~170 tok/s** (decode), **~27 k tok/s** (prefill),
**~37 s wall**. Anything decoding under 100 tok/s on this config is
contended; anything under 60 tok/s is heavily contended (the 134.5 s
run hit 47 tok/s).

## Pointers

- The contention paper-trail lives in
  `burl/eval/results/perf_ledger.csv` rows 3-9
  (timestamps `20260427_015323` through `20260427_022000`); rows
  11+ (`023012` onward) are clean.
- Contention discovery happened in [[burl-perf-phase1]]
  (Phase 1 cheap wins).

## Links

[[perf-on-the-table]] · [[burl-perf-phase1]] · [[mlx-lm]] ·
[[batched-eval-resilience]]
