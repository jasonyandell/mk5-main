---
title: "Belief probes: does Zeb's field net encode the teacher's exact posterior over who holds what? (Lem, the paper's G1)"
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

G1 of the paper's work order ([`lem/paper/README.md`](../../lem/paper/README.md) § 5; [[lem-the-teacher]]).
Pre-registered here before any probe was fit. The code is `lem/probe/belief.py` (CC0; tests in
`lem/tests/test_probe.py`).

## Question

The teacher's belief at a position is exact: for every live tile the viewer does not hold and
every other seat, P(seat holds tile | the viewer's information), under uniform worlds that respect
the voids the seats have shown ([[silas]], `SilasBelief.uniform_resolving` on that tile). Does the
student, the shipped field net, encode it? And does it encode more of it than its own inputs
state linearly?

## Pre-registration

- **Net and sample.** `zeb/net/margin_d250k_h128x64.fnet`; the audit's sample
  ([[lem-field-net-audit-2026-10-15]]: 434,535 held-out positions, pip trumps, seat 0 bids 30).
- **Targets.** For each position, `P[rel, slot]` for the three other seats (left, partner, right)
  and every live hidden tile in the net's canonical slot; undefined elsewhere. The **counting
  prior** `B[rel, slot]` = that seat's remaining hand size over the hidden tiles (voids ignored).
  The probe's target is the deviation `P − B`: what the voids tell the teacher. The entries the
  voids move (|P − B| > 0) are reported separately ("moved").
- **Split.** Deals shuffled once (seed 20261016) and halved; 20,000 fit rows from one half,
  20,000 eval rows from the other, so no deal is on both sides.
- **Probes.** Ridge regression (λ = 1, standardized features, one ridge per output entry over
  the rows where it is defined) from four representations: the 343 raw input bits `x`; the
  first hidden layer `a1` (128); the second `a2` (64); `a1+a2`. Scored on the eval rows by R² over
  all defined entries and over the moved entries, plus mean absolute error against the prior's.
- **Chance.** The same probe fit on row-shuffled targets, scored the same way.
- **Predictions, before measuring** (the desk's priors).
  1. The raw inputs state most of it: R² of `x` on the moved entries ≥ 0.8 (the void bits are
     inputs, and the posterior under uniform worlds is close to linear in them).
  2. `a1` recovers nearly what `x` does (within 0.1 of R²); `a2` recovers less (at least 0.1
     below `a1`): the belief is read off the inputs, not built deeper.
  3. No hidden representation beats `x` by more than 0.02 R²: the student adds no belief
     computation beyond what its inputs state linearly.
  4. Shuffled R² within ±0.02 of 0.
- **What a surprise would mean.** `a2` ≥ `x` on the moved entries would say the net computes
  something about ownership its inputs do not state linearly, and would send G3 (causal tests)
  at those units first.
- **Scope.** Play-only evidence (the record and the shown voids). The auction carries nothing
  in these pours (seat 0 always bids 30); the auction-evidence posterior (`Belief::hand_law`,
  #134) is a later arm, and the field's own posterior (reading plays) another.
- **Cost.** Exact targets cost 0.7 ms a position (one 1-tile resolution per live hidden tile),
  about 30 s for 40,000 rows; the ridges are seconds. One `scripts/cpu-slot` call.

## Results

Not yet run.

## Links

[[lem-the-teacher]] · [[lem-field-net-audit-2026-10-15]] · [[silas]] · [[scenario-player]] ·
[[strategy-fusion-fence]]
