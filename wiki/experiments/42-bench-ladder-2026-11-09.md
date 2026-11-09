---
title: The 42 Bench on the bid30 ladder — clean-room walt1 builds and Zeb rated against walt-wasm (2026-11-09)
kind: experiment
first_seen: 2026-11-09
last_updated: 2026-11-09
status: complete
---

## Question

Seven models (Qwen, Haiku, Opus, Fable, Luna, Sol, Astra) each built, clean-room from the same
spec, a Rust `walt1 play` endpoint: level-1 Walt card play, `n 40` outer worlds, `n0 8` per
level-0 mind (spec and corpus: `~/code/local/walt-corpus/`, `ENDPOINT.md`). Jason (2026-11-09):
"strength is what matters ... a super fast Walt would be neato and allow lots of new
experiments." Walt is the right opponent strength for casual play; Zeb is built from Walt.
Does any build play as strongly as Walt, and how fast?

## Setup

- Ladder `marks7-bid30` ([[tournament]]): first to 7 marks, both sides bid with the pinned
  `Bid30Bidder`, so only card play is compared; duplicate pairs (same deals, seats swapped).
- Participant kind `walt1-binary` (`tournament/walt1_seat.py`, mk5 `89640531`): one binary process
  per decision, sent the seat's dealt hand, declaration, bid 30, bidder and the public record; seed
  rule `walt1-bench-v1` (sha256 of faceoff seed/pair/side/hand/seat/plays, never names or clock).
  Every reply is checked against the referee (legal choice, leader, points, forced). Pins: binary
  sha256, build git commit or none, source digest, tournament source hashes; mismatch fails closed.
- Reference and anchor (1500): `walt1-ref-86ab0d3c`, the real walt-wasm (texas-42 `9d6a5a2e`,
  `walt/`) compiled natively behind the same protocol, at the same n/n0. Plunge's production Walt
  (`plunge-walt`) was not used: its budgets differ (160 worlds on the opening lead, partner check).
- **Chance baseline:** `walt1-ref-reseeded-fc5bcfd3`, the same walt-ref binary behind a wrapper that
  remaps each decision's seed through sha256, so its true rating is 1500 by construction.
- Equivalence bar, fixed 2026-11-09 04:45Z (after Fable had 54 games): a build is Walt-strength
  if its whole 95% interval vs walt-ref lies in 1500 ± 50.
- Store: a local append-only `GitStore` in the canonical record format (not yet on the Hub; see
  Data). 3, then 6 workers under `cpu-slot --threads 2`. W&B board:
  [texas-42-bench/bench-ladder-1](https://wandb.ai/jasonyandell-forge42/texas-42-bench/runs/bench-ladder-1).

## Result

1011 faceoffs, 2022 games, no illegal or malformed reply from any build. Bradley–Terry, 95%
cluster-robust intervals for the gap to walt-ref.

| build | Elo | 95% CI | games | vs walt-ref, marks/game [95% CI] (duplicate pairs) | CPU/decision |
|---|---|---|---|---|---|
| ref-reseeded (control) | 1502 | [1465, 1539] | 246 | +0.05 [−0.40, +0.50] (52) | 678 ms |
| walt-ref (anchor) | 1500 | — | 1280 | — | 678 ms |
| fable | 1467 | [1447, 1487] | 716 | −0.33 [−0.51, −0.15] (283) | 33 ms |
| qwen | 1466 | [1430, 1502] | 260 | −0.93 [−1.73, −0.14] (15) | ~4 s opening lead |
| sol | 1460 | [1434, 1487] | 418 | −0.47 [−0.76, −0.18] (106) | 139 ms |
| opus | 1456 | [1421, 1491] | 214 | −0.31 [−1.07, +0.45] (16) | 89 ms |
| luna | 1456 | [1415, 1497] | 222 | −0.25 [−1.05, +0.55] (10) | 1238 ms |
| astra | 1454 | [1429, 1479] | 450 | −0.42 [−0.66, −0.18] (141) | 66 ms |
| haiku | 1452 | [1415, 1489] | 238 | −0.35 [−1.05, +0.34] (17) | 612 ms |

CPU per decision is the mean over the games up to 04:24Z; the rows for qwen and the control
are estimates.

How to read it:

- **The ladder is fair.** The reseeded control sits at 1502 [1465, 1539], level with walt-ref
  (51–53 in games). The common gap below is not a seat, seed or anchor artifact.
- **Every build is weaker than Walt, by a similar amount (about 35–50 Elo).** Every build's
  interval lies wholly below 1500. The three builds with many head-to-head pairs lose 0.3–0.5
  marks/game to walt-ref, and each of those intervals excludes zero. Seven independent readings of
  one spec landing in one place point at something walt-wasm does that the spec leaves out or
  under-pins (seed derivation, the level-0 mind's details, the sampler). This is a finding about
  the spec, not a ranking of the models. Among the builds the differences are within noise.
- **No build passes the ±50 bar.** Fable comes closest, at −33 Elo [−53, −13] and about 20×
  walt-ref's speed. Its lower end (1447) is just below the 1450 floor, so it is not shown
  equivalent. It is shown weaker.
- The corpus grader agrees in shape: choice match with walt-ref 61–70% and |value error| 420–520 bp
  for all seven. Reseeding walt-ref alone flips its opening lead on the grader's first example
  (19 → 8), so a match rate in that range may be near walt-ref's own reproducibility.

## Round 2 (2026-11-09 19:17Z–22:01Z): Zeb, Fable's tie refinement, v1-spec builds

The same ladder and store. New players:

- **Zeb 0.4.1** (kind `zeb-native`, 160 worlds, seed 0, the common Bid30Bidder). This is
  the only Zeb rated on the canonical `marks7-bid30` ladder, where it is the anchor. The
  canonical record fails closed against today's harness. Its source pin holds
  `arena/hand_metrics.py` from before b41b3121, which adds a function no player reads
  ([#262](https://github.com/jasonyandell/mk5-main/issues/262)). So the same player (same
  native binary pin, wasm, trees and config) is seated as
  `zeb-0.4.1-contested-bid30-src-b41b3121`, re-pinned to the current sources. The record names
  the canonical id.
- **Fable's later builds**, `walt1-fable-tierefine-c9c7192d` (tie refinement added) and
  `-fixedmind-b40ae229`.
- **Two builds written only from the v1.0 Walt spec** (`~/code/local/walt-spec/WALT.md`), which
  makes the n→4n→16n tie refinement mandatory: `walt1-v1-haiku-18f00ffc` and
  `walt1-v1-sol-1facc7b4`.

Final table: 2138 faceoffs, 4276 games, walt-ref = 1500.

| player | Elo | 95% CI | games | vs walt-ref, marks/game (pairs) | CPU/decision |
|---|---|---|---|---|---|
| Zeb 0.4.1 | 1586 | [1552, 1621] | 268 | +1.03 [+0.64, +1.43] (61) | ~3 s/game |
| fable-tierefine | 1517 | [1498, 1535] | 628 | +0.13 [−0.05, +0.30] (274) | 57 ms |
| ref-reseeded (control) | 1510 | [1476, 1544] | 296 | | |
| fable-tierefine-fixedmind | 1503 | [1484, 1523] | 634 | −0.11 [−0.27, +0.04] (317) | 56 ms |
| v1-haiku | 1491 | [1460, 1522] | 276 | −0.08 [−0.35, +0.19] (110) | 938 ms |
| v1-sol | 1486 | [1464, 1508] | 468 | −0.10 [−0.27, +0.08] (220) | 68 ms |

How to read it:

- **Zeb is clearly stronger than Walt.** It leads by about +86 Elo and +1.0 marks/game, and it
  beats every walt1 build on the point estimate except tierefine: −0.38 marks/game [−1.16,
  +0.41] over 8 pairs.
- **Tie refinement closes the round-1 gap.** Every build that does it passes the ±50 bar: both
  v1-spec builds and both tierefine builds. Every round-1 build left it out and sat 35–50 Elo
  low. The spec gap that round 1 pointed at was §6.3.
- **v1-sol is the fast Walt.** It is Walt-strength at 68 ms per decision, about 7× walt-ref
  (CPU is descriptive only).
- **Caveat:** the v1 verdicts were read at repeated looks (every 10 minutes), and play stopped
  at the first look inside. The games played after the stop kept both inside. A
  fresh confirmation block, judged once, would make the verdicts airtight.
- **Failures:** none in 4276 games: no illegal or malformed reply, no hang-guard timeout. One
  outside SIGTERM killed some `walt-play` processes mid-claim, and those claims replayed with
  the same seeds.

Harness fix: `materialize` now takes a per-participant lock. Two parallel games had built
zeb-native into one cargo target dir at once.

## What was learned about the tournament

- The information-gain scheduler starved Sol (14 games, gain 0.000) and later Qwen: from a few
  clusters that mostly split 1–1, the sandwich interval runs narrow. Filed as
  [#261](https://github.com/jasonyandell/mk5-main/issues/261); the workaround used here was a
  per-opponent catch-up (`--only <new>,<opp>`).
- Fixed: two workers sharing a claims directory could crash on a claim that another worker
  removed between listing and locking it. Added `work --exclude`: withdraws ids from scheduling
  and keeps their records.

## Data

Published 2026-11-09, public, CC0:
[jasonyandell/texas-42-bench](https://huggingface.co/datasets/jasonyandell/texas-42-bench)
(round 1: 9 participants and 1011 faceoffs; round 2 adds 6 participants and 1127 faceoffs,
data commit `0f7078f0`; append-only, the canonical record format). It is a separate dataset from the canonical store because `participants/` there
is shared by every ladder. Re-derive the table with
`python -u -m tournament --store hf:jasonyandell/texas-42-bench table --ladder marks7-bid30 --anchor walt1-ref-86ab0d3c`.
Derived copy: `~/code/local/walt-corpus/elo.json`.

## Next

- Confirm the v1 verdicts with a fresh block, judged once.
- Seat v1-sol where a fast Walt opens experiments: arena opponents, Zeb's teacher and the
  bid ladders.
- Fix the source pin that keeps the canonical bid30 anchor from playing on main (#262).

## Links

[[tournament]] · [[scenario-player]] · [[walt-bidder-2026-11-02]]
