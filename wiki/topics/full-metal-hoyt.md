---
title: Full-metal Hoyt — a certified GPU referee past H4
kind: topic
first_seen: 2026-07-21
last_updated: 2026-07-21
status: active
---

[[hoyt]] already expresses late Texas 42 as a static, wave-structured array
program, and its roots are mutually independent. The full-metal hypothesis
survives because those are exactly the two shapes an accelerator can consume:
wide regular work inside a root and a fleet of roots outside it. The CPU
campaign also removed the ambiguity about what remains. H4 micro-performance
has a measured exhaustion receipt, exact world/root quotienting collapses
nothing, and the surviving own-hand class merge removes only about 3% of
non-forced strategy slots ([[hoyt-perf-primer]],
[[endgame-equivalence-census]]).

## Problem statement

Can a **device-resident, root-batched Metal implementation of the complete
Hoyt reference solve** make H5 and then H6 reference production practical on
the M5 Max while preserving Hoyt's only load-bearing claim: every published
profile is a reproducible low-exploitability reference whose final gap is
priced by exact single-seat best response?

"Complete solve" means the full-width tree build, information-set and
compressed strategy layout, CFR+ reach/value sweeps, regret and average-policy
updates, intermediate stop tests, final profile, value, gap, and ledger
verdict. Uploading CPU-built arrays to accelerate one gather or regret kernel
does not answer the question. The throughput path must keep a ragged fleet of
roots resident across iterations, with no host round trip at every wave or
seat update.

The target is not a faster H4 demo. It is a new habitat for the stable referee:
an H5 reference line first, then evidence about whether H6 belongs to exact
enumeration, streaming, or an explicitly licensed approximation. The named
consumers remain reference-backed [[jud]]/`lens:ev` leaves, a distilled student
whose error is measurable against game-native values, and durable referee
values for downstream mechanism work. This is the Hoyt continuation of the
batch-VCT direction parked for the net-bearing [[walt]] path in
[#74](https://github.com/jasonyandell/mk5-main/issues/74), not a port of Walt's
neural field evaluator.

## Why this is the structural move

The stacked evidence removes the cheaper escape hatches:

- [PR #78](https://github.com/jasonyandell/mk5-main/pull/78) made the game a
  net-free SoA wave program and established the frozen H4 reference problem.
- [PR #81](https://github.com/jasonyandell/mk5-main/pull/81) removed the
  Python-object/RSS failure and built the cap-ledger fleet.
- [PR #83](https://github.com/jasonyandell/mk5-main/pull/83) compressed the
  exactly inert 83% of forced strategy slots, fused the iteration, priced the
  gap on the resident structure, made the build resident, and carried the full
  200-root line to 727 certified H4 evals/hour. The subsequent measured levers
  mostly compressed away at fleet scale ([[perf-log]] 18l–19i).
- [PR #84](https://github.com/jasonyandell/mk5-main/pull/84) proved exact
  world/root equivalence collapse is 1.000× by a depth-independent
  co-occurrence mechanism. Its strongest surviving class-CFR form is an exact
  but roughly 3% non-forced-slot tidy, not a width reduction.

One more CPU micro-kernel therefore cannot change Hoyt's habitat. Per
[[hoyt-perf-primer]], CFR costs about 1.5 orders of magnitude per added tile in
both time and memory; H5 CFR is not yet priced as a population, and H6 is
already classified as a different build. The unspent parallelism is the
population itself: solve many independent roots on the 40-core GPU rather than
running a handful of memory-contending CPU processes.

## The numerical collision is part of the problem

The verified fused lane is fp64-only. Its built-and-deleted fp32 arm drifted
`2.59e-3` on the anchor gap, outside the registered `1e-3` license, and bought
no CPU throughput because that kernel was gather-latency-bound
([[perf-log]] 18l). Metal Shading Language does not support `double`
([Metal Shading Language Specification, section 2.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)).

Full-metal Hoyt therefore includes a numerical-method problem, not merely a
kernel-port problem. A successful lane must do one of three things and measure
which one it chose:

1. reproduce the required fp64 behavior with a higher-precision Metal
   representation or compensated scheme;
2. use Metal arithmetic to generate a candidate average profile, then make the
   existing fp64 engine the sparse final certifier and continuation oracle; or
3. earn a new numerical license from decision damage and final exact-BR gap,
   not from raw value drift alone.

The third option may change implementation tolerances; it may not change
Hoyt's published claim. A profile is a reference only after an authoritative
gap measurement says so.

## Contract that must survive

The Metal lane consumes the same `(root, worlds, weights, payoff43)` object as
`hoyt.cfr_solve` and emits the same semantic result:

```
profile + reference value + exact single-seat-BR gap
+ iteration trace + stop/cap verdict + phase timings
```

The following are acceptance conditions, not stretch goals:

- **Rules and structure:** legal moves, trick winners, counts, terminal points,
  chance weights, information-set identity, forced-slot elision, and public
  path identity come from the existing [[play-phase-algebra]]/LUT contract.
  Structural fixtures must match the CPU wave engine exactly: wave sizes,
  parent and move arrays, actors, info-set offsets, terminal points, and slot
  counts.
- **Reference semantics:** regret-matching+, alternating four-seat updates,
  linear averaging, payoff signs, tie rules, and the single-seat deviation gap
  retain the semantics in [[cfr-primer]] and `hoyt/CONTRACTS.md`.
- **Certification:** the existing fp64 engine remains the authority until a
  Metal certifier independently cross-matches it. Every accepted row must end
  with a full four-seat gap at or below the registered target; an intermediate
  early-exit test is not a certificate.
- **Reproducibility:** root/world sampling, arithmetic lane, device identity,
  iteration trace, final verdict, and any CPU repricing are ledgered. Absence
  is never a result; OOM, precision rejection, and cap expiry are first-class
  verdicts.
- **Honesty:** this remains an agent-form, four-seat low-exploitability
  reference. Metal does not promote it to a team equilibrium and does not add
  a team-pair deviation guarantee.

## Hard engineering questions

The implementation has to answer these together because each can erase the
others' speedup:

- **Ragged fleet layout:** roots differ by orders of magnitude and each wave
  expands irregularly. The GPU needs a packed root/wave descriptor algebra,
  stable compaction, and tail handling that do not serialize on the largest
  root.
- **Build residency:** H4 fleet wall is already build-heavy, and the one wedge
  reaches 589.6M slots and about 27 GiB solo. H5 cannot begin by materializing
  several unconstrained replicas. The design must choose and measure full
  residency, root/action sharding, or a streaming frontier.
- **Deterministic reductions:** forward reach, backward value, counterfactual
  value, and regret matching all contain segmented folds. Atomic accumulation
  is easy; a reproducible, certifiable fold with acceptable occupancy is the
  actual requirement.
- **Precision:** fp32 is the native Metal floating-point lane but has not earned
  Hoyt's license. Precision, accumulation order, and certification cadence are
  one design surface.
- **Device boundary:** CPU orchestration may prepare roots and append ledger
  rows. It may not rebuild, regroup, or re-upload every wave; otherwise launch
  and synchronization overhead recreate the Python problem one level lower.
- **Population scheduling:** throughput is H5/H6 evals per wall hour, not one
  photogenic root. Root bucketing, dynamic admission, memory pressure, and the
  wedge path must be graded on a full population.

## Falsification ladder

The build should be killed early if its mechanism does not survive these
gates:

1. **FM0 — size the target.** Freeze a small stratified H5 evalset and measure
   the current CPU lane's tree/slot/memory anatomy under explicit world caps.
   This supplies the budget the Metal lane must close; H5 BR timings are not an
   H5 CFR baseline.
2. **FM1 — structural Metal walk.** On toys and selected H4 roots, build the
   full-width integer structure on-device and match every CPU structural array
   exactly. No CFR arithmetic is licensed before this passes.
3. **FM2 — one-iteration cross-gate.** Match reach, leaf, counterfactual,
   regret, average, and strategy arrays on toys and small real roots under the
   selected precision scheme. Record drift by operation and depth rather than
   only at the final scalar.
4. **FM3 — certified H4 replay.** Run the frozen 200-root H4 line through the
   new lane. Require zero structural discrepancies, zero false converged
   verdicts under fp64 repricing, a full per-root decision-damage report, and a
   measured fleet wall/RSS comparison with the 727-evals/hour reference.
5. **FM4 — horizon proof.** Produce a certified H5 line inside a pre-registered
   wall and memory budget. If the lane only improves H4 while H5 remains
   infeasible, the hypothesis has not achieved its purpose.
6. **FM5 — consumer proof.** Materialize the reference rows in the same stable
   format needed by the first named downstream consumer. Kernel throughput
   without a consumable referee artifact is not completion.

Kill the lane if host synchronization remains wave-shaped, if no numerical
scheme can pass fp64 certification, if ragged packing leaves the GPU mostly
idle, or if the resident footprint makes H5 no more feasible than the CPU
lane. Those are mechanism failures, not invitations to weaken the referee.

## Non-goals

- another H4-only CPU micro-optimization;
- reviving world/root equivalence after the co-occurrence theorem;
- porting the Walt/Jud neural field path to MPS;
- declaring fp32 acceptable because a scalar looks close;
- replacing the exact-given-worlds reference with sampling without a separate,
  explicit error contract; or
- changing Hoyt's equilibrium honesty line.

## Links

[[hoyt]] · [[hoyt-perf-primer]] · [[perf]] · [[perf-log]] ·
[[endgame-equivalence-census]] · [[cfr-primer]] · [[walt-spec]] ·
[[play-phase-algebra]]
