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

*Part 1's results follow once every registered batch has run. Part 2 (the tournament's pair bootstrap) is done: calibrated at n = 64, R_F 0.85 in share and 1.02 in marks, 0 of 24 intervals missed.*

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

### Part 2 results (n = 64, the registered stage; stopped there as registered)

All 1,536 games finished (12 replicates × 64 pairs × 2 games, 14,976 hands); 768 of 768 batches exited
0 on the first attempt and no game was resumed. Wall 4.6 h (20:46–01:20 UTC) at four `scripts/cpu-slot`
jobs beside the Book desk's queue; the two teams spent the same seat CPU (24,214 s and 24,171 s).
Every interval below is the rematch's estimator at n = 64 pairs (95% percentile pair bootstrap, 40,000
resamples, seed 666). Chance is 50% and 0 marks.

| Read | Win share (primary) | Marks a game |
|---|---:|---:|
| **R_F** (calibrated 95% range [0.59, 1.41]) | **0.85** | **1.02** |
| SD of the 12 replicate means / RMS SE | 3.22 / 3.78 points | 0.292 / 0.287 |
| Replicate intervals missing the pooled mean (expect 0.6; 3+ = undercover) | **0 of 12** | **0 of 12** |
| Replicate intervals missing chance | 0 of 12 | 0 of 12 |
| **δ(1,0)**, seed 1 over seed 0, pooled over 768 pairs | 49.28% [47.14, 51.43] | −0.049 [−0.212, +0.115] |

Per replicate, seed 1's win share and marks a game (faceoff seed; hands):

| | Share [interval] | Marks [interval] | | Share [interval] | Marks [interval] |
|---|---:|---:|---|---:|---:|
| F0 | 49.22 [42.19, 56.25] | +0.242 [−0.273, +0.758] | F6 | 51.56 [44.53, 58.59] | +0.078 [−0.430, +0.602] |
| F1 | 51.56 [44.53, 58.59] | +0.094 [−0.492, +0.695] | F7 | 43.75 [36.72, 50.78] | −0.391 [−0.945, +0.156] |
| F2 | 52.34 [44.53, 60.16] | +0.492 [−0.055, +1.031] | F8 | 51.56 [45.31, 57.81] | +0.180 [−0.312, +0.672] |
| F3 | 51.56 [43.75, 59.38] | −0.078 [−0.719, +0.555] | F9 | 49.22 [42.19, 56.25] | −0.109 [−0.633, +0.406] |
| F4 | 44.53 [36.72, 52.34] | −0.422 [−1.000, +0.164] | F10 | 53.12 [45.31, 60.16] | +0.055 [−0.508, +0.617] |
| F5 | 46.09 [38.28, 53.91] | −0.375 [−0.984, +0.242] | F11 | 46.88 [38.28, 55.47] | −0.359 [−0.969, +0.242] |

**Tails** (768 pairs pooled; a pair's margin is seed 1's marks minus seed 0's over its two games):
excess kurtosis of the per-pair margin **−0.15** (lighter than normal: a race to 7 bounds it), SD 2.31
marks a game; **25.4%** of pairs at |margin| ≥ 6 marks over the pair (≥ 3 a game, the registered cut,
the same count); pairs splitting 1–1: 487 (63.4%), seed 1 sweeping 135, seed 0 sweeping 146.

**Reading.** At n = 64 the tournament's pair bootstrap covers what it should: the spread of twelve
replicate means matches the width the intervals claim (R_F 0.85 in share and 1.02 in marks, both inside
the calibrated range), and none of the 24 intervals misses the pooled mean, where 0.6 a read is chance
(0 of 12 has probability 0.54 under calibration, so it is not suspiciously tight either). The per-pair
margin has no heavy tail to fool a percentile bootstrap. The two seeds are indistinguishable: δ(1,0) is
−0.05 [−0.21, +0.12] marks a game. As registered, a pass at 64 carries to the rematch's 256, where the
same independent pairs only make the bootstrap's normal approximation better. **On this evidence the tournament's
intervals need no extra allowance** (for equal players; see Limits), and the ±0.15 marks / ±4 points read-on caveat from #191 does not
apply to them; the rematch's +0.330 [+0.055, +0.606] and 57.03% [53.12, 60.94] stand as reported.

Why this estimator passes where the Book arena's A/As missed is part 1's question, not answered here.
What differs by construction: a faceoff pair is two games on one fresh deal stream, every pair its own
seed, the players deterministic and the duplicate halves played to completion; part 1's arena batches
and its paired halves are a different sampling unit ([[book-small-n-2026-10-14]]).

**Limits.** One participant (Zeb 0.4.1 contested, a mirror match) and one ladder (`marks7`); twelve
replicates locate R_F only within about ±0.4; n = 64, not 256 (any 256 stage would be a new
registration made before more data is seen). A pair of clearly unequal players, where the per-pair
distribution is skewed, is not tested.

Evidence: [`book/calibration-tournament`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/1ee36c0d0c7bd367d6fae5ef2b4323b09b9326d7/book/calibration-tournament)
(games, faceoff records assembled by the tournament code but never appended to the store, `summary.json`,
the excluded pilot, the runner, `CARD.md`, `SHA256SUMS`).

## Links

[[book-authority]] · [[rematch-zeb-041-vs-demon-2026-10-15]] · [[auction-read-2026-10-16]] · [[seed-varied-2026-10-16]] · [[book-small-n-2026-10-14]]
