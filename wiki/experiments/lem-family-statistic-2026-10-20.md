---
title: "Already-clean test 7: a sharper family statistic — coarse versus mixed, corrected for base rates, family size and lumpiness (Lem, #200)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---

Part of THE ALREADY-CLEAN TEST (#193; [[clean-neurons-42]]). The squeeze's family test
([[lem-squeeze-2026-10-16]]) asked whether a broadened pole's single best two-concept union lies
within one concept family. It had three weaknesses:
- many poles share one union, so the null's band was wide (up to 0.56);
- precision ignores each union's base rate;
- large families have more pairs and so more chances to hold the best one.

This test fixes all three and checks its own calibration before reading it.

Scope: the field net's recipe on pip pours, one position against uniform beliefs, no history.
The vocabulary is list v1, whose concepts the eleven families cover (`lem.probe.squeeze.FAMILIES`).

## Pre-registration

- **The statistic** (`lem.probe families-u`; `squeeze.family_u`, `families_u`).
  - A pole is *broadened* when its best valid union beats its best single concept by at least
    0.05, as before.
  - For such a pole, every valid union (base rate ≤ 0.5) is scored by its **excess**: precision on
    the pole's top 1% minus the union's base rate.
  - **u** is the probability that a random set of unions, as large as the within-family set,
    has a lower best excess than the within-family best, with ties counted half. It is 0.5 in
    expectation when families mean nothing and near 1 for a coarse unit.
  - It is reported as the mean over broadened poles and as the mean over **distinct**
    best-within-family unions (each union once). The null is the same statistic with family labels
    permuted (200 permutations).
- **Nets.** The 21 trained drop-out nets and their 21 untrained twins (7 widths × 3 seeds).
- **Rows and labels.** The 434,535-row audit sample, list v1 (`labels.npz`), and live and dead
  poles from the v1 audits of [[lem-dropout-order-2026-10-17]].
- **Instrument.** Signals conditioned on the play count (`--condition n_plays`;
  [[lem-conditioned-audit-2026-10-20]]).
- **Calibration arms.**
  - The 21 untrained nets.
  - Shuffled labels: each width's seed-0 trained net with the label rows permuted
    (`--shuffle-rows 1`).
- **Predictions.**
  1. **Calibrated.** Pooled over all their broadened poles, the untrained nets' mean u is within
     0.05 of 0.5, and so is the shuffled-label arms' mean u.
  2. **Coarse at 24 and 12 neurons.** At 16×8 and at 8×4 the trained nets' pooled mean u is above
     0.6, and above the permuted-family null's 97.5th percentile in at least 2 of 3 seeds.
  3. **Coarse at every width.** At every width from 8×4 up, the trained pooled mean u is above 0.55.
  4. **Not one lumpy union.** At 16×8 the distinct-union mean u is above 0.55 in at least 2 of
     3 seeds.
- **Cost.** 49 runs of 434,535 rows on the Mac's CPU through `scripts/cpu-slot`, about 1–2
  minutes each.

## Results

Not yet run.

## Links

[[lem-squeeze-2026-10-16]] · [[lem-width-tree-2026-10-17]] · [[lem-dropout-order-2026-10-17]] ·
[[lem-conditioned-audit-2026-10-20]] · [[clean-neurons-42]]
