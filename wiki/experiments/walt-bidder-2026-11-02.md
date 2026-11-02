---
title: Walt bids — the honest plan's root splat as a live bidder, piloted against the heuristic (2026-11-02)
kind: experiment
first_seen: 2026-11-02
last_updated: 2026-11-02
status: complete
---

## Question

Jason (2026-11-02): make a Walt-strength player that can bid from trick 1. Zeb bids from
trick 1 (`@jasonyandell/zeb`'s `forecast` + `chooseBid`; in the arena `zeb:` in
`arena/zeb_bid.py`), but Zeb is too strong and not the player his family enjoys sitting across
from; Walt, the player they enjoy, has never bid in this repo. Plunge's production Walt looks up
a frozen bid book, and its earlier live bidder was switched off because its forecast overbid:
"the price curve saturates against walt's level-0 field model, so even θ = 1 overbids at the
table" ([[scenario-player]] § A live bidder). Can Walt bid with its own forecast, and is that
forecast honest at the call it chooses?

## Why it should work now

The overbidding Walt had the clairvoyant tail: a max over a one-world cell is
perfect-information play for that world, so the root forecast ran twenty points hot
([[scenario-player]] § The forecast lies). The cell floor (`f=16`, the arena default at
`k ≥ 1`) hands thin cells down to the rung below, and the floored forecast matched the real
endings within a point or two of margin and a point or two of make share, over 200 and 5,000
deals (§ The floor calibrates the forecast; the pours). Zeb's bidder reads that same root
splat, over the net planner. Nothing had read it over Walt's own planner.

## The bidder

`arena/walt_bid.py`, arena spec `waltbid[:<walt play options>][,c<cutoff>]` (default
`k1,n160,m8`, cutoff 0.8). It is Zeb's bidder with Walt's planner: one plan per pip
declaration from the bidder's chair as seat 0 before the opening lead under a 30 contract, the
chosen lead's tails read at every target 30..42 as pmake, `choose_bid` the hand-only rule
(the highest target, at least the lowest legal bid, some declaration makes with probability at
least the cutoff; else pass; never overcall the partner; never 84), declare the declaration
most likely to make the bid. Same class, same code (`ZebBidder` holding a `CompiledWalt` built
from the walt play spec), so every byte of the read is the one graded against Plunge's book
and in [[match-aware-bidding-2026-10-14]]. Walt's random field plays all nine declarations,
but the bidder forecasts pip trumps only, as Zeb's does.

Cost of one forecast (seven declarations, all cores, M5 Max, 2026-11-02): 0.11 s at
`k1,n64,m8`, 0.28 s at `k1,n160,m8`, 1.06 s at `k1,n512,m8` (the champion). A hand needs at
most four (one per seat, cached by hand).

## The pilot (not a registered study)

Sized by the clock, not by power: a hard stop at 2 PM CST left about an hour for the build,
the run and this page. Read it as a pilot with its interval; a registered block at the
champion's size is filed as follow-up work.

**Protocol.** Team A = `waltbid:k1,n160,m8,c0.8` + `walt:k1,n160,m8`; team B =
`heuristic` (the arena's Roberson risk-budget bidder, the standing bidder in every Walt and Zeb
result) + the same `walt:k1,n160,m8`. Mirrored game pairs (identical deals, seats swapped), games
to 7 marks, `ArenaConfig` defaults (reshakes, then the shaker forced to 30). Play is the same
player on both sides, so the margin is the bidding alone. Reported: A's marks/game with a 95%
interval over game pairs; games won; contracts per hand, share made and mean bid for each bidder;
and the calibration of Walt's forecast at the call it chose (mean forecast pmake against the
realized make share, in bins); wall and instructions.

## Results

**Block 1: marks (48 games, 24 mirrored pairs, base seed 1, `scratch/waltbid/run1/`; `python -u -m arena.cli
--team-a waltbid:k1,n160,m8,c0.8+walt:k1,n160,m8 --team-b heuristic+walt:k1,n160,m8 --n-games 48
--base-seed 1 --emit-snapshots …`).** Walt's bidder wins **34 of 48 games (70.8%; both halves
70.8%)**, **+1.25 marks/game [+0.35, +2.12]** (the arena's bootstrap over games; chance 0), point
margin −4.46 a hand (the `make` signature: marks up, points down). Hands per game 11.2.

| bidder | contracts (share of hands) | made | mean bid |
|---|---|---|---|
| Walt's own forecast (A) | 134 (24.9%) | **88.8%** | 32.7 |
| heuristic risk-budget (B) | 404 (75.1%) | 55.4% | 30.4 |

Walt bids a quarter of the hands and makes nine in ten; the heuristic takes the rest, at 30, and
makes just over half. Two shakers were forced to 30 after the reshakes, one on each side; the
table counts them, the calibration below excludes them. The bid
histogram over all 538 contracts: 30 ×347, 31 ×110, 32 ×14, 33 ×5, 34 ×20, 35 ×20, 36 ×13, 40 ×2,
41 ×4, 42 ×3.

**Calibration at the call (`scratch/waltbid/calib.py`: the snapshots joined to the bidder's own
forecast, recomputed; the planner is deterministic by coordinate, so the join is exact; the forced
30 on each side excluded).** Over Walt's 133 own calls the mean forecast pmake at the declared
trump and the bid is **0.840**; the realized make share is **0.887 [0.822, 0.930]** (Wilson). Every
own call prices at or above the cutoff by construction. The forecast is honest at the call it
chooses, if anything a few points shy; the winner's curse over seven declarations and thirteen
targets is not visible at this size.

| forecast bin (Walt's calls) | n | mean forecast | made |
|---|---|---|---|
| [0.80, 0.85) | 91 | 0.818 | 0.868 [0.78, 0.92] |
| [0.85, 0.90) | 28 | 0.868 | 1.000 [0.88, 1.00] |
| [0.90, 0.95) | 12 | 0.921 | 0.750 [0.47, 0.91] |
| [0.95, 1.00] | 2 | 0.994 | 1.000 |

The same forecast priced the heuristic's 403 own calls (its hand, its declared trump, its bid; Walt
plays both sides, so the forecast applies): mean 0.595 against 0.553 made [0.505, 0.601]; below the
cutoff (n = 320) 0.514 against 0.481; at and above it, 0.825 → 0.762 (21), 0.876 → 0.889 (18),
0.918 → 0.800 (20), 0.980 → 0.875 (24). About right in the body, a few points hot in the top bins
where the counts are small.

**Cost.** One forecast per distinct hand, cached: 478 forecasts priced every contract of the block.
The 48 games: wall 507 s, CPU 3,333 s (6.6 cores), 3.04e13 instructions retired (`scripts/cpu-slot
--threads 16`; the forecasts are the auction's whole cost, and the play on both sides is the same
Walt). Descriptive only.

**Replication (base seed 2, the same protocol, 48 games, `scratch/waltbid/run2/`).** Run after block 1
with the time the stop left; not pre-registered, but its seeds are fresh. Walt's bidder wins **24 of 48
(50.0%; halves 58.3% / 41.7%)**, **+0.54 marks/game [−0.50, +1.58]**, point margin −6.03 a hand. The
contract pattern holds: A bids 23.8% of hands at a mean of 32.9 and makes 84.6% (121 own calls, two
forced 30s excluded: mean forecast **0.843**, made **0.843 [0.768, 0.897]**); the heuristic makes 57.5%
at 30.3 (393 own calls, forecast 0.595, made 0.575). Bins for Walt's calls: [0.80, 0.85) n = 79,
0.820 → 0.772 [0.67, 0.85]; [0.85, 0.90) n = 28, 0.866 → 0.964 [0.82, 0.99]; [0.90, 0.95) n = 11,
0.919 → 1.000; [0.95, 1.00] n = 3, 0.975 → 1.000. Wall 471 s, 2.86e13 instructions.

**Pooled, 96 games (`pool.py`: a percentile bootstrap over games, 10,000 resamples).** Walt's bidder
wins **58 of 96 (60.4%)**, **+0.90 marks/game [+0.21, +1.56]**. Over its 254 own calls (three forced
30s excluded) the mean forecast at the call is **0.842** and the make share **0.866**; mean bid 32.8.
The heuristic: 796 own calls, 56.4% made, mean bid 30.3, priced at 0.595 by Walt's forecast.

**Reading.** The margin shrank from +1.25 to +0.54 on fresh seeds, the pattern the project's
registrations name as the winner's curse; pooled it is +0.90 [+0.21, +1.56], a lower bound above
zero from two clock-sized blocks, not a registered result. What replicates exactly is the mechanism:
a Walt that bids on its own honest forecast takes a quarter of the hands at a mean bid near 33,
makes about 86% of them, and its forecast at the call means what it says (0.842 forecast against
0.866 made over 254 calls; block 2 alone 0.843 against 0.843). The heuristic takes three hands in
four at 30 and makes 56%. Whether the caution of the 0.8 rule is right is the cutoff question
(#254), and whether the edge holds at the champion's size and against Zeb's bidder is the registered
block (#254).

**Data.** Both blocks' outputs (the arena logs and summaries, per-game and per-hand CSVs, every hand's
snapshot, the calibration joins and the three scripts, with SHA256SUMS; block 2's files prefixed `run2-`)
are public, CC0, at
[texas-42-zeb-research-2026-11](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-11)
under `walt-bidder/` ([[publish-data-to-huggingface]]).

## Limits

- One machine, two clock-sized blocks, one spec (`k1,n160,m8`, the shipped two-response size, not
  the champion's 512 worlds). The intervals are a pilot's; nothing here is a verdict on the bidder.
- The opponent bidder is the heuristic, not Zeb's bidder; "Walt-strength" is by construction
  (the bidder is Walt's own forecast, the play is Walt's), not a measured fun rating.
- The forecast is planned at 30 under `make`; pmake at higher targets is read off a partly
  tape-played tail, as Zeb's is (#107).
- Pip trumps only; no 84; no match equity; no reading of the other seats' bids.

## Links

[[scenario-player]] · [[match-aware-bidding-2026-10-14]] · [[arena]] · [[tournament]]
