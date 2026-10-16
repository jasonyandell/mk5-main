---
title: Calibrating the estimator — how often does an A/A miss? (#191, 2026-10-16)
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

## Summary

Issue #191, a methods study that every desk's numbers lean on. Two A/A arms in the Book arena
missed their 95% intervals in opposite directions: +0.12 [+0.02, +0.23] marks/game in #117 and
−0.14 [−0.25, −0.03] in #190 ([[auction-read-2026-10-16]], [[seed-varied-2026-10-16]]). Neither a
per-seed effect nor correlated batches explains it. A first check rules out one more cause:
deployed Zeb's choice for a state does not depend on what else is in its batch. Forty states,
decided alone, in shuffled batches, repeated and by a fresh player object, gave 0 differences.

**Two seeds are two policies.** Plain Zeb at seed 1 and at seed 0 are two deterministic players,
and one can truly be better than the other by the luck of its seed. A nonzero difference between
them is therefore not miscalibration. This design separates the two questions. **F** holds the
pair fixed (seed 1 against seed 0) across twenty replicates on disjoint deals, which pins their
true difference δ(1,0) and tests whether the house intervals cover it. **V** draws a fresh pair of
seeds per replicate, which measures how far apart two seeds' policies typically are.

*Results follow once every registered batch has run.*

## Design (pre-registered before any measured batch; see `scenario/calibration.py`)

- **Players.** Every seat bids by the heuristic rule at caution 0 and plays deployed Zeb 0.3.0 at
  a fixed planner seed. F<r>: seed 1 against seed 0. V<r>: seed 100 + 2r against 101 + 2r.
- **Size.** Twenty replicates an arm, each 288 game pairs (24 batches of 12) on its own base seeds
  (F on 5001–5480, V on 6001–6480); 960 batches.
- **Estimator.** Each replicate's interval exactly as the house reports one (`scenario.small_n`).
- **Registered reads.** (1) R_F, the SD of the F means over their RMS standard error: near 1 if
  calibrated, 95% range about [0.68, 1.31]; **above 1.31 means the intervals undercover**. Also,
  how many F intervals miss the pooled F mean (expect 1 of 20; 4 or more is p < 0.02).
  (2) δ(1,0), pooled over 5,760 pairs, against #117's +0.12. (3) R_V, σ_seed, and how many V
  intervals miss 0. (4) The per-pair tails: excess kurtosis and the share at |margin| ≥ 6.
- **Scope.** This is the Book arena's estimator. The tournament's pair bootstrap (the rematch,
  #165 and #184) and the Lem audit's intervals are #191's other two parts.

## Links

[[book-authority]] · [[auction-read-2026-10-16]] · [[seed-varied-2026-10-16]] · [[book-small-n-2026-10-14]]
