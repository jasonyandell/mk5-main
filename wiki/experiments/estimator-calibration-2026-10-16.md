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

## Part 2: the tournament estimator (pre-registered before any measured faceoff; see `tournament/calibration.py`)

**The estimator under test** is the rematch's ([[rematch-zeb-041-vs-demon-2026-10-15]]): a faceoff of
duplicate pairs on the `marks7` ladder (two games on the same deals, seats swapped), read as A's win
share and marks a game per pair, with a 95% percentile pair bootstrap (resample the pairs, 40,000
resamples, seed 666).

**Participant.** Native Zeb 0.4.1 contested, the rematch's Zeb (`zeb-0.4.1-contested`): 160 worlds,
cutoff 0.8, no clock, so it thinks on a deterministic budget (FAIR CLOCKS). It is seated from the
ladder's own materialized artifact (the pinned JavaScript and wasm; the native binary rebuilt from the
pinned trees, `b54e990f…`, whose bytes differ from the trials' `c76303d0…` only by embedded build paths).

**Why not Zeb against itself.** The native host sends the player `seed: 0` on every request whatever the
config says, a pair's two games are dealt from the same seeds with the dealer at seat 0 in both, and Zeb
reads nothing of the game key. So the second game of a pair is the first, move for move, with the teams
relabeled: every pair splits 1–1 and the margin is exactly 0, an interval of zero width. A pilot pair
(excluded) confirmed it on whole games: 12 of 12 hands identical, both games 7–6 to the same seats.

**The A/A: F, seed 1 against seed 0.** Team A is the same build with every native request's `seed` set to
1 (forecast and play), by a seed shim between the unmodified host and the binary; team B is the ladder's
Zeb exactly as the tournament seats it. The shim at seed 0 reproduces the ladder's Zeb exactly (a pilot
pair: 9 of 9 hands identical, mirror result), so the two teams differ only in the planner seed. As in
part 1, two seeds are two deterministic policies with a true difference δ(1,0) that need not be 0; the
replicates pin it and test whether the intervals cover it.

**Size, and why it is nested.** The timing pilot (excluded: 3 pairs of seed 1 against seed 0, plus the two
check pairs) ran about 36 s and 35 CPU s a game, so a 256-pair replicate is about 5 worker-hours and the
brief's four hours at four jobs buys three replicates at the rematch's size. So the design is nested:
**F0..F11, twelve replicates of up to 256 pairs**, each on its own faceoff seed
(`sha256("issue-191/tournament/F<r>")[:16]`, disjoint deals), run pair-major so the replicates grow
together through the quarters n = 64, 128, 192, 256. The first stage is n = 64 (1,536 games, about four
hours at four `scripts/cpu-slot` jobs, one thread each). The run stops only at a quarter boundary and only
for time or the Mayor's word, never on a look at the results; the reads are taken at the last completed
quarter and reported with its n. Each replicate's interval is the estimator above at that n.
A calibration at n = 64 is the harder test: the percentile bootstrap of a bounded per-pair mean
undercovers less as n grows, so a pass at 64 carries to 256 and a miss at 64 does not by itself convict 256.

**Registered reads (win share primary, as in the rematch; marks a game alongside).**
1. **R_F** = SD of the twelve replicate means / RMS of their standard errors, SE = (hi − lo)/3.92. Near 1
   if calibrated; under normality its 95% range at 11 degrees of freedom is [0.59, 1.41]. **R_F above
   1.41 = the intervals undercover.**
2. **Misses**: how many replicate intervals exclude the pooled mean (expect 0.6 of 12; **3 or more is
   p < 0.02** and also says undercover). Also, how many exclude chance (50%, 0 marks).
3. **δ(1,0)**: the pooled mean over every pair with the same bootstrap, in share and marks.
4. **Tails**: the per-pair marks margin pooled over the replicates, its excess kurtosis, the share of
   pairs at |margin| ≥ 3 marks a game, and how many pairs split 1–1.

**Scope.** Research evidence only: nothing is appended to the tournament's store or ladder, and the
seeded participant is never added. Runner: `python -u -m tournament.calibration drive --arms F0,...,F11
--pairs 64 --out DIR`, then `report`. Data to HF `book/calibration-tournament/`.

## Links

[[book-authority]] · [[rematch-zeb-041-vs-demon-2026-10-15]] · [[auction-read-2026-10-16]] · [[seed-varied-2026-10-16]] · [[book-small-n-2026-10-14]]
