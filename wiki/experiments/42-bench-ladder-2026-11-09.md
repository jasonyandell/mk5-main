---
title: The 42 Bench on the bid30 ladder — seven clean-room walt1 builds rated against walt-wasm (2026-11-09)
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

## What was learned about the tournament

- The information-gain scheduler starved Sol (14 games, gain 0.000) and later Qwen: from a few
  clusters that mostly split 1–1, the sandwich interval runs narrow. Filed as
  [#261](https://github.com/jasonyandell/mk5-main/issues/261); the workaround used here was a
  per-opponent catch-up (`--only <new>,<opp>`).
- Fixed: two workers sharing a claims directory could crash on a claim that another worker
  removed between listing and locking it. Added `work --exclude`: withdraws ids from scheduling
  and keeps their records.

## Data

Local store: `scratch/bench/remote.git` in the `bench/42-bench-ladder` worktree
(`participants/`, `faceoffs/marks7-bid30/`); derived table `~/code/local/walt-corpus/elo.json`.
The bench belongs in its own public CC0 dataset, `jasonyandell/texas-42-bench`, because the
canonical store's `participants/` directory is shared by every ladder. Creating it was refused by
this session's permission classifier. The card and the upload command are in
`~/code/local/walt-corpus/TOURNEY-STATUS.md`. Not yet published.

## Next

Diff Fable (the fastest build) against walt-wasm on the positions where they part, to find the
shared gap; a fix there is the road to a fast Walt.

## Links

[[tournament]] · [[scenario-player]] · [[walt-bidder-2026-11-02]]
