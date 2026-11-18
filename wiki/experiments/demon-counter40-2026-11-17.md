---
title: "The adaptive Demon raises an opening to 40 and learns who lets it stand"
kind: experiment
first_seen: 2026-11-17
last_updated: 2026-11-17
status: complete
---

**STRATEGY-FUSION FENCE.** [[the-demon]] now beats both shipped native Zeb
0.4.1 contested and its own early-40 predecessor in straight 42 to seven marks.
The predecessor is the root-reuse player from [[demon-instructions-2026-10-16]].
Both use the same pinned native solver, 100 play worlds, 64 bid worlds and nine
straight declarations. The difference is in the auction.

Second to speak over an opponent's opening below 40, the new Demon raises to 40.
The next opponent still has a player behind it. In that spot Zeb's contested
model prices even a weak point overcall above passing, the same mechanism behind
[[demon-ascent-2026-10-16]]. If either Demon partner actually wins a 40 from one
of the first two seats, the new player retires both blind bids for the rest of
that game. It then uses the ordinary forecast cutoff of 0.875 when opening or
raising. Each game starts fresh. A resumed game restores this decision from
completed public auctions.

## Two opponents, two tests

| Straight marks to seven | Game wins | Win rate (95% paired interval) |
|---|---:|---:|
| Incumbent early-40 vs Zeb | 236/256 | 92.2% [89.1%, 95.3%] |
| Counter-40 vs Zeb on the same deals | 245/256 | 95.7% [93.0%, 98.0%] |
| Adaptive counter-40 vs incumbent head-to-head | 116/128 | 90.6% [85.9%, 95.3%] |

Against Zeb the paired difference is **+3.52 percentage points [1.56, 5.86]**
and **+0.42 net marks/game [0.29, 0.56]**, both relative to zero. The next Zeb
seat raised on all 148 eligible counterbids. Across the head-to-head games the
old Demon let its opponent's early 40 stand exactly once in each of the 128 games. The adaptive
Demon won **+3.23 net marks/game [2.86, 3.59]** against the old one. Chance is
50% in each matchup. The table gives opponent-specific results.

The two protocols were fixed before their respective games in
[issue 264](https://github.com/jasonyandell/mk5-main/issues/264). Each seed
supplies two games with teams swapped. The Zeb comparison couples deal seeds,
game keys and search seeds across both arms and alternates run order. The
head-to-head uses a disjoint seed. Intervals resample whole pairs 40,000 times
with seed 264. The record audit regenerates every deal, checks seat ownership
of all played tiles, verifies auction winners and scores, and checks the
completed game's winner.

The final adaptive participant also replayed **all 256 Zeb games** through
the normal seat factory. Every hand record matched the counter-40 run, including
all bids, declarations, tiles and scores. No early 40 stood against Zeb. Its
native instruction ratio to the incumbent on those games was **0.958
[0.940, 0.975]**.

## Cost and reproducibility

Against Zeb the fixed counter used **95.8% [94.0%, 97.5%]** of the old Demon's
native retired instructions per game. The upper bound cleared the fixed 1.20 cost limit.
Games were shorter on average, 9.28 vs 9.55 hands. The cost comparison measures
complete games. Across the head-to-head games the adaptive Demon used **76.4%
[73.7%, 79.3%]** of its opponent's native instructions while playing the same
hands. CPU and shared wall time are descriptive. Native Demon CPU averaged
1.19 vs 1.24 seconds/game in the Zeb trial and 1.83 vs 2.38 seconds/game in
the duel. Runs used deterministic budgets under `scripts/cpu-slot`.

Both Demons used native source `d48909c4`. The adaptive player's Python auction
implementation is pinned separately to `7eec74c0`. The [[tournament]] scheduler
requires the `demon-counter-v1` worker capability to seat the new participant.
The seven-marks trials use the ordinary auction and do not measure the common
bid-30 ladder or the points-to-250 game.


The [public experiment archive](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-11/tree/b4ab50c49343175e6bd857eb7b2d6ac3229c2d42/demon/2026-11-17-counter40)
contains both protocols, all 896 game records, the audited report, participant
configs, binary hashes and reproduction scripts. The archive, report,
provenance and checksum manifest were downloaded and hash-verified.

The [immutable tournament participant](https://huggingface.co/datasets/jasonyandell/texas-42-tournament/blob/f3b74ef9259a45144ebabee5b996de7d5f742a41/participants/demon-pimc-n100-bid928480c1-straight-bounds-support-bait40-adaptive-d48909c4.json)
was downloaded after registration and matched the roster bytes. All 73 earlier
participant records kept their blob IDs.
