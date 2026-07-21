---
title: Full-metal Hoyt — the H5/H6 problem
kind: topic
first_seen: 2026-07-21
last_updated: 2026-07-21
status: active
---

## Problem statement

Build a **fast, full-Metal [[hoyt]]** that makes H5 and H6 reference solving
practical on the M5 Max.

Hoyt currently solves H4 well on the CPU, but each added tile multiplies CFR
time and memory by roughly 1.5 orders of magnitude. The #78 → #81 → #83 → #84
stack exhausted the credible CPU and exact-compression levers:

- H4 is measured at 727 reference evals/hour;
- forced strategy slots are already compressed away;
- build, iteration, and gap pricing already operate on resident SoA waves;
- exact world/root equivalence compresses nothing; and
- the surviving within-hand class merge removes only about 3% of non-forced
  slots.

The remaining opportunity is the unused parallelism across the solve itself:
move Hoyt's full-width tree construction, CFR sweeps, and reference evaluation
onto the GPU, keep the work device-resident, and batch the ragged population of
roots rather than running a few memory-contending CPU processes.

The goal is **fast H5, then fast H6**. H4 is the calibration set and performance
baseline, not the destination.

## Accuracy requirement

Bit-for-bit replication of the CPU implementation is a non-goal. Metal may use
different layouts, reduction orders, and floating-point arithmetic.

The requirement is **confident, measured error bars**. Against CPU-solvable
reference roots, the Metal lane must establish held-out error distributions for:

- reference value;
- estimated single-seat best-response gap;
- convergence classification; and
- action/argmax changes, stratified by decision margin.

Those intervals must have pre-registered coverage and must be tight enough to
support the downstream decision. A result is usable when its reported error bar
makes the claim honest; it does not need to reproduce the CPU bits. The existing
fp64 lane is a calibration oracle and audit surface, not a required per-root step
in the production H5/H6 path.

## What “full Metal” means

The production path keeps the expensive work on the GPU:

- full-width wave/tree construction;
- information-set and strategy layout;
- CFR forward and backward passes;
- regret and average-policy updates;
- gap estimation; and
- batched scheduling across roots.

Accelerating one kernel while rebuilding or synchronizing every wave on the CPU
does not solve the problem. The unit of acceleration is the **reference solve
fleet**, not an isolated loop.

## Success

The work succeeds when it produces:

1. a calibrated Metal lane whose held-out error bars cover the CPU truth;
2. a measured, practical H5 reference line;
3. a measured, practical H6 reference line; and
4. reference artifacts consumable by the player/distillation pipeline.

It fails if the error bars are too wide to support decisions, if host/device
synchronization dominates, if ragged roots leave the GPU mostly idle, or if the
result is merely a faster H4 solver.

## Non-goals

- bit replication;
- identical intermediate arrays or reduction order;
- per-root CPU certification in production;
- another H4-only CPU optimization;
- reviving exact world/root equivalence; or
- porting Walt's neural field evaluator.

## Evidence

[[hoyt-perf-primer]] · [[perf-log]] · [[endgame-equivalence-census]] ·
[[cfr-primer]] · [PR #78](https://github.com/jasonyandell/mk5-main/pull/78) ·
[PR #81](https://github.com/jasonyandell/mk5-main/pull/81) ·
[PR #83](https://github.com/jasonyandell/mk5-main/pull/83) ·
[PR #84](https://github.com/jasonyandell/mk5-main/pull/84)
