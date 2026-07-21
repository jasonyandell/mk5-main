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
move Hoyt's structural sampling/build, CFR sweeps, and reference evaluation
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

- sampled or streamed wave/tree construction;
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

## Finding: a dense port cannot be H6

The first implementation established a real custom-Metal CFR+ vertical slice
over Hoyt's forced-slot-compressed segment layout. On the quiet M5 Max:

- H4 anchor 555006: Metal iterate 0.225 s versus 3.506 s for four-thread
  fused CPU (15.6x); device-side gap pricing reduced the complete solve to
  3.77 s versus 10.31 s (2.74x). Float32 drift after 40 rounds was 0.0336
  points in value and 0.00129 in measured gap.
- H5 seed 910000 at eight worlds: Metal iterate 0.265 s versus 3.150 s
  (11.9x) for ten rounds; the complete solve was 19.4 s versus 25.4 s
  (1.31x). Host construction and export now dominate. Float32 drift was
  0.00539 points in value and 0.00216 in gap.
- H6 seed 910000 at **one world**: the dense walk exceeded 288,991,264 live
  slots at play 21 and hit the 256M slot cap before any CFR iteration. The
  root has 17,153,136 possible hidden worlds.

This refutes “put the existing full-width tree on Metal” as the H6
architecture. The iterate kernels are viable and worth keeping, but full Metal
must change the structural estimator: sampled trajectories, streamed/chunked
waves, or another bounded representation whose value/gap/action errors can be
calibrated on H4 and small-support H5. The uncertainty license applies to the
structure as well as float arithmetic; otherwise H6 cannot fit.

The next decision is therefore algorithmic, not another kernel micro-pass:
choose an estimator that (1) has bounded device memory independent of the full
tree, (2) exposes statistically honest value and gap intervals, (3) supports
root-fleet batching, and (4) survives held-out CPU calibration.

## Surviving implementation: sparse external sampling

The bounded-memory route now has a working first slice:

- `hoyt/worldsample.py` replaces world enumeration with a tiny exact
  continuation-count DP and samples one legal deal per Metal thread. Counts
  match exhaustive enumeration on every toy; a 24,000-draw Metal gate is
  uniform over the exact population.
- `hoyt/sampled_br.py` now uses the same traversal in two modes.
  `SampledCFR` alternates all four seats through one shared sparse regret table;
  `SampledBR` can fork that candidate, reset one seat, and train while the
  other three policies remain frozen. Both branch every updater action, sample
  one current-policy opponent action, and retain only the current frontier.
  Rows are keyed by seat plus a 62-bit information fingerprint in a sorted
  hard-cap table. `SparsePolicy` artifacts round-trip to `.npz` and expose a
  uniform-fallback lookup for downstream consumers.
- Exact small roots gate both pieces. The four-seat H3 candidate returned
  22.7288 points (SE 0.0641) versus exact CFR 22.7333 and chose the same root
  action. On H4 seed 555184, the exact uniform-opponent BR traversed 96.6M
  nodes in 15.2 s; 64,000 sampled traversals trained in 4.03 s and returned
  10.8560 (SE 0.02275) versus exact 10.7770, inside the registered four-SE
  smoke band and with the exact root move.
- On the established Jud-play H5/910000 root (324,324 physical worlds), the
  shared candidate trained 25,600 traversals in 13.39 s, evaluated in 0.11 s,
  used 392,928 rows, and peaked at 16,442 frontier states. Four frozen-policy
  BR forks plus evaluation took 13.02 s. The provisional profile value was
  25.620 (95% empirical-Bernstein radius 0.427); the simultaneous bounded-
  payoff candidate-deviation band was [0, 1.084] before optimization
  shortfall.
- On the paired H6/910000 root (17,153,136 worlds), 6,400 shared-CFR
  traversals trained in 5.37 s and evaluated in 0.13 s, using 305,277 rows and
  a 27,219-state peak frontier. Four BR forks took another 11.81 s. The
  provisional profile value was 16.360 (95% empirical-Bernstein radius
  0.573); root moves 7 and 15 were effectively tied. The simultaneous
  bounded-payoff candidate-deviation band was [0, 1.528] before optimization
  shortfall.

Those H5/H6 numbers are end-to-end memory/throughput receipts, **not reference
claims**. The independently evaluated candidate BRs give a valid route to a
lower exploitability bound, but are not themselves upper bounds on the true
best response. Until `metalcal.br_shortfall_upper` is fit on an exchangeable
exact-root calibration population, the H5/H6 upper gap is infinite and both
verdicts are correctly `unresolved`. H4/H5 calibration cannot silently be
called an H6 guarantee; that transfer must be demonstrated or exact
bounded-memory fixed-profile H6 anchors must be added.

Version 2 still indexes sparse update records on the host between Metal
microbatches, exports the current rather than a theoretically privileged
average policy, and schedules roots serially. The production sequence is:
device-side sparse sort/merge; root-fleet batching; independently trained
candidate BRs at registered budgets; one-sided shortfall calibration; then
held-out value, gap, convergence, and action-damage coverage. A new root is
converged only when candidate uncertainty plus the shortfall bound lies below
the target; otherwise it remains explicitly unresolved.

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
