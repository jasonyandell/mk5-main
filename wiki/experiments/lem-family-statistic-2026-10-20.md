---
title: "Already-clean test 7: a sharper family statistic — coarse versus mixed, corrected for base rates, family size and lumpiness (Lem, #200)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
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

Run 2026-10-20 as registered. Evidence is `lem/family-statistic-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/39aa0d52865c285a0589e2a37a876702cff633a6/lem/family-statistic-2026-10-20):
49 runs, `grade.py`, `grade.txt` and `grade.json`. Pooled means weight each net by its broadened
poles.

| width | trained u, pooled (poles) | trained seeds above the permuted null's 97.5% | distinct-union u (3 seeds) | **untrained u** (poles) | shuffled labels u (poles) |
|---|---:|---:|---|---:|---:|
| 128×64 | 0.576 (832) | 2 / 3 | 0.603 / 0.573 / 0.561 | **0.581** (1,092) | 0.441 (173) |
| 128×8 | 0.555 (619) | 0 / 3 | 0.551 / 0.534 / 0.590 | 0.580 (774) | 0.413 (119) |
| 32×16 | 0.560 (220) | 1 / 3 | 0.536 / 0.647 / 0.490 | 0.588 (273) | 0.431 (40) |
| 16×8 | 0.518 (102) | 0 / 3 | 0.488 / 0.497 / 0.495 | 0.601 (138) | 0.387 (15) |
| 8×4 | 0.580 (48) | 0 / 3 | 0.559 / 0.643 / 0.468 | 0.586 (69) | 0.394 (9) |
| 4×2 | 0.593 (25) | 0 / 3 | 0.519 / 0.543 / 0.632 | 0.636 (34) | 0.324 (3) |
| 2×1 | 0.571 (10) | 1 / 3 | 0.924 / 0.461 / 0.502 | 0.485 (16) | 0.133 (2) |

Pooled over all widths: untrained **0.583** (2,396 poles); shuffled labels **0.425** (361 poles).

**The predictions, graded.**

| | measured | |
|---|---|---|
| 1. calibrated: untrained and shuffled within 0.05 of 0.5 | 0.583 and 0.425 | **wrong** |
| 2. 16×8 and 8×4 above 0.6, above the null in 2 of 3 seeds | 0.518 and 0.580; 0 of 3 each | **wrong** |
| 3. every width from 8×4 up above 0.55 | 16×8 is 0.518 | **wrong** |
| 4. 16×8 distinct-union u above 0.55 in 2 of 3 | 0.488 / 0.497 / 0.495 | **wrong** |

**Reading.**

- **The statistic fails its own calibration, and that is the finding (rung 1, the tools).**
  - Untrained nets score 0.583, well above 0.5. A random unit's top 1% already favours unions
    within one family. The concepts of one family co-occur in the data: trump-held concepts with
    each other, trick-state concepts with each other. So any direction through the inputs carries
    that structure, and a null that permutes family labels removes the structure and sets the bar
    too low.
  - Shuffled labels score below 0.5 (0.425) for the converse reason. With the rows permuted, the
    top set's unions are noise. Correlated within-family pairs then have higher union base rates
    and lower excess.
- **Against the right baseline, trained units are not more coarse than random ones.** At every
  width the trained mean u is at or below its own untrained twins' (16×8: 0.518 against 0.601).
  Read honestly, this statistic shows no coarse-versus-mixed signal at all: neither coarseness nor
  mixing beyond what random units carry.
- **What this does to earlier claims.** The squeeze's "units broaden within one family"
  ([[lem-squeeze-2026-10-16]]) and the width tree's "own family 2.4–2.9× random"
  ([[lem-width-tree-2026-10-17]]) both rest on permuted-family nulls, and neither was compared
  with untrained twins. Until #229 re-checks them against untrained nets, they stand as *not yet
  shown against the right null*.
  - The squeeze's clean shares and the conditioned result (narrower is cleaner) are unaffected:
    they were always read against untrained floors.
- **What this does not say.** It does not say the narrow nets' units are mixed. It says this
  family instrument cannot tell, and that the family claims need the floor every other share
  already gets.

Scope: the field net's recipe on pip pours, list v1, signals conditioned on the play count.

## Links

[[lem-squeeze-2026-10-16]] · [[lem-width-tree-2026-10-17]] · [[lem-dropout-order-2026-10-17]] ·
[[lem-conditioned-audit-2026-10-20]] · [[clean-neurons-42]]
