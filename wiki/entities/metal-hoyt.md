---
title: metal_hoyt — the GPU searcher under the fp64 referee
kind: entity
first_seen: 2026-07-21
last_updated: 2026-07-21
status: active
---

## What it is

[[hoyt]]'s CFR+ iterate and in-struct gap pricing rebuilt as raw Metal
(MSL) kernels on the M5 Max GPU — a NEW ENGINE, not a port. Code at
`metal_hoyt/` (design doc: `metal_hoyt/DESIGN.md`); kernels dispatched
through `mx.fast.metal_kernel` (MLX, already a repo dependency), so it
ships with zero new dependencies and unified-memory buffers. Born
2026-07-21 from the parked "H6+ is a different build (streaming / Metal /
sampling)" line on the [[hoyt]] page, funded by the perf push reaching
measured exhaustion on CPU ([[perf-log]] 19h–i: 727 evals/hour,
H4-cap-256, every lever priced).

## The one-table design

| stage | engine | precision |
|---|---|---|
| structure build (walk, layouts) | hoyt CPU | exact ints |
| CFR+ iterate + steering gap | **Metal GPU** | fp32 |
| final gap + reference value | hoyt CPU (`_wave_br` / `_wave_values`) | **fp64 exact** |

hoyt stays the referee; metal_hoyt is the searcher. Every banked claim is
*"profile found on GPU, gap priced by exact fp64 best response"* — the
same claim class as a hoyt solve. The fp32 gap only decides WHEN to
certify (exit at target − margin, default 0.01); a wrong fp32 gap can
waste iterations but never mis-claim. Structure truth is single-sourced:
the GPU layout consumes hoyt's `_build_wave`/`_build_fused(par=True)`
arrays — the P13 threading groupings ARE GPU segment layouts — and
asserts three contiguity invariants rather than assuming them.

## What it does NOT claim

Bitwise parity with hoyt's fp64 trajectory (Metal has no fp64), and
therefore interchangeability with hoyt's bit-reference artifacts:
`reference_h4_v1_cap256.jsonl` stays hoyt-produced, and the
[[endgame-equivalence-census]] 29/29 bitwise-tie receipts stay hoyt-only
instruments. Within fp32 it IS deterministic: no atomics, every fold
sequential over precomputed segments, bitwise-repeatable across runs on
the same GPU. Metal-unavailable is a hard error — hoyt is the CPU lane;
there is no fallback.

## Measured (2026-07-21, M5 Max, receipts in [[perf-log]])

- **Searcher speed, 16-root paired bench (cap-256, rung-0 params)**:
  iterate + gap pricing **3.7 s GPU vs 67.7 s CPU (18.2×)**; whole-solve
  wall 1.39× — the solve is now build/certify/export-bound (all CPU), the
  predicted Amdahl shift.
- **The wedge (555090, 42M-slot walk)**: iterate 103.0 → **3.0 s (34×)**,
  gap pricing 47.6 → 0.2 s; identical certified result (v = +11.000, gap
  exactly 0.0, dv = 0.0000). Remaining wall: build 166.7 s + fp64
  certify 57.1 s + upload 13.8 s.
- **fp32 honesty**: steering-gap drift vs fp64, worst observed 8.5e-3
  (near-tied BR argmax flips, not accumulation) — 6× under the 0.05 bar;
  the M4 gate logs it per solve. Metal contracts mul+add to FMA, so fold
  kernels agree with the fp64 same-order fold to ~1.8e-7 (more accurate
  than separate rounding) instead of being bitwise vs numpy.
- **Toys and pins**: all 8 [[hoyt]] toys converge and agree with the CPU
  engine within summed gaps; pinned seats ride through unchanged.
- **Production line** (`refsweep --engine metal`, 19h two-halves
  pattern): **200/200 converged in 960 s** (CPU line: 990), h2 with the
  oversubscription fix at **420 s — the fastest half measured on this
  line** — for **~7.2 user-s/eval, roughly half the CPU line's 15.1
  worker-s/eval**. Wall quote ambient-contaminated (swap); quiet-box
  re-measure in [#87](https://github.com/jasonyandell/mk5-main/issues/87).
- **The 555013 witness**: metal references vs the banked line — p50
  |dv| 0.0040, but one root sits **1.59 pts away at gaps ~0.03** with
  identical worlds and bit-identical walt-BR value. The partnership seam
  measured: single-seat BR gap does not pin the value; distinct
  trajectories land in distinct basins. See [[perf-log]] 21a and the
  question filed in questions/open.

## Consumers

`hoyt.refsweep --engine metal` — the production cascade unchanged (same
ledger, verdicts, resume; the pull-based line's CPU phases parallelize
across workers while each worker's GPU duty cycle stays small). The
line-rate measurement and follow-up levers (build, certification,
cross-root batching) are tracked in [[perf-log]] 21a and the filed
issues.

## Links

[[hoyt]] · [[perf-log]] · [[perf]] · [[walt]] · [[cfr-primer]] ·
[[endgame-equivalence-census]] · [[hoyt-perf-primer]]
