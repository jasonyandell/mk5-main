---
title: Zeb 0.3.0's confirmation — soft reading and make-then-margin together beat 0.2.0, +0.21 marks/game (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## Question

Jason approved `@jasonyandell/zeb` 0.3.0: 0.2.0 with two defaults changed, each measured alone.

- **Soft uniform play-reading** at ε = 0.28 ([[reads-plays-soft-2026-10-14]] § Replication 2):
  +0.37 [+0.24, +0.51] marks/game against plain Zeb, at 512 worlds.
- **Make, then margin** on exact ties ([[tie-margin-2026-10-14]]): +0.01 [−0.04, +0.07]
  marks/game and +4.55 points a hand, at 512 worlds.

They were never measured together, and the package plays at 160 worlds by default. Does
0.3.0's plan beat 0.2.0's as the package plays them? A failure would have stopped the release.

## Pre-registration

Committed at `91855dc6` (`scenario/zeb030.py`, top of file) before any measured batch:

- **Players**, built as `zeb/src/lib.rs` builds them under a pip trump, at the package's
  default 160 worlds and seed 0, over the package's own blob `zeb/net/margin_d250k_h128x64.fnet`.
  0.2.0 is `k0,n160,F=<net>,f=16`. 0.3.0 adds `d=posterior,r=uniform:4194304:0.28,t=margin`.
  The arena's `bid30` bidders declare pip trumps only, so doubles and no-trump are not in
  the match.
- **Table.** Both seats of team A play the challenger and both seats of team B the baseline.
  Bidders are `bid30`, games go to 7 marks, and every game seed is played from both sides
  over the same deals (duplicate deals).
- **Arms.** `v3`: 0.3.0 against 0.2.0. `aa`, the A/A control: 0.2.0 at planner seed 1
  against 0.2.0 at seed 0, on the same seeds in the same run.
- **Seeds.** Base seeds 601–616. None overlaps the reading studies' seeds or the tie
  study's.
- **Primary.** `v3`'s margin in marks per game (a game pair's sum / 2), with a 95% percentile
  bootstrap over game pairs (4,000 resamples, seed 42). **Bar: the lower bound is above 0.**
- **Registered secondaries.** The A/A margin, which should cover 0; settled points per
  game; contracts made when bidding.
- **Size.** A 1-minute pilot (seeds 690–691, 16 pairs per arm, excluded) put the per-pair
  SD at 1.22 (`v3`) and 1.56 (`aa`) marks/game, at 1.2–1.8 s a pair. Sizing took SD 1.5:
  (1.96 · 1.5 / 0.15)² = **384 game pairs per arm**, as 16 batches × 24 pairs, for a
  half-width of at most 0.15.
- **Batch drop.** Every batch runs under `timeout 300`. A nonzero exit drops the batch, and
  it is not rerun. No early stop, no top-up.

## Result

**The bar is met.** 0.3.0's plan beat 0.2.0's by **+0.21 [+0.08, +0.34] marks/game** over 384
game pairs. The interval's half-width is 0.13, inside the 0.15 target. The A/A arm sits at
chance.

| registered effect | game pairs | estimate |
|---|---|---|
| **primary: 0.3.0 vs 0.2.0, marks/game** | 384 | **+0.21 [+0.08, +0.34]: passes** |
| 0.3.0 vs 0.2.0, settled points/game | 384 | +37.3 [+31.9, +42.5] |
| A/A: 0.2.0 at seed 1 vs 0.2.0, marks/game | 384 | +0.06 [−0.07, +0.18] |
| contracts made bidding, 0.3.0 / 0.2.0 in the same games | 8,504 hands | 38.5% / 36.6% |
| contracts made bidding, A/A (seed 1 / seed 0) | 8,562 hands | 37.3% / 36.7% |

Exploratory (not registered): 0.3.0 − A/A on the same pairs is +0.15 [−0.02, +0.32]. Per
batch, 0.3.0's margin was positive in 12 of 16 batches, zero in 1 and negative in 3. The
A/A arm's was positive in 7, zero in 1 and negative in 8.

All 32 batches exited 0, in 33–83 s each, three at a time, at a load average of 23–49 on 18
cores (other sessions were running). Run 2026-10-14, 10:15–10:25, at `91855dc6`.

**What it means.** The two defaults together pay at the phone default. The estimate is below
the reading's +0.37 alone at 512 worlds. That reading came from a different world count and
an opponent table, and this match was not built to separate the two changes, so it says
nothing about whether they interact. The points gain, about 37 settled points a game, is
larger than either change's alone. The tie rule buys points, and settled points also follow
the made contracts. Both caveats apply: one match at one world count, pip trumps only.

## Cost in WebAssembly

The package's `npm run bench` (`zeb/scripts/bench.js`) gained a section that times `play`
with reading on and off at the nine opening hands and at every unforced decision of six
deals under sixes. Times are in CPU ms in Node 24 on the M5 Max. The machine was busy (load
average 22–25), so the opening lead took 70 ms against the README's 44 ms when quiet. Read
the ratios.

| | 160 worlds, on / off (median) | 512 worlds, on / off (median) |
|---|---|---|
| opening lead | 70 / 71 | 266 / 264 |
| trick 1 | 72 / 53 | 204 / 162 |
| trick 2 | 72 / 39 | 216 / 132 |
| trick 3 | 78 / 31 | 162 / 96 |
| trick 4 | 19 / 16 | 36 / 41 |
| tricks 5–6 | ≤ 3 | ≤ 3 |

The opening lead, and so the README's 44 ms and 154 ms budget row, is unchanged: before any
play there is nothing to read. At tricks 1–3 reading costs 1.3–2.5× the unread median, more
than the native 1.7× of [[reads-plays-soft-2026-10-14]] § Cost at the tricks where the
rejection draw runs. Scaled to a quiet machine by the opening lead, a trick-2 or trick-3
decision at 160 worlds is about 50 ms median and 120 ms at the 90th percentile. That is
within the budget the opening lead sets at 512 worlds. Two earlier wall-time runs of the
same bench, at the same load, gave ratios of 1.3–2.3× and 1.3–3.1× at tricks 1–3.

## What shipped

At `a0656fe2`, `zeb/src/lib.rs` builds the 0.3.0 plan. Under a pip trump it uses
`Draw::Posterior` with `READING` = `Reading { uniform: true, margin: 1 << 22, eps: 0.28 }`
unless the position request says `reading: false`. Doubles and no-trump are never read.
`Tie::Margin { band: 0.0 }` applies under every declaration, so a forecast's chosen lead (and
its mean and tails) can change, while reading leaves a forecast unchanged (tested). The
parity fixture was regenerated from the Python referee: 38 positions, 3 of them with reading
off. Reading moves q at 5 of them and the tie rule moves the tile at 12, and the generator
refuses a file in which either default does not act. It is bit-identical in native Rust and
in the WebAssembly. The version is 0.3.0. The license is unchanged (MIT). Not published:
the npm publish waits on Jason.

## Data

`jasonyandell/texas-42-zeb-research-2026-10` under `zeb/release-0.3.0/` (CC0, `SHA256SUMS`,
`NOTES.md`): `runs/<arm>_s<seed>/` (`meta.json`, `games.jsonl`), `pilot/`, `logs/progress.log`,
`summary.json`, `run_all.sh`, `bench.txt`. Recompute:

```bash
bash scratch/zeb030/run_all.sh
python -u -m scenario.zeb030 summarize --runs scratch/zeb030/runs/* --out summary.json
```

## Links

[[scenario-player]] · [[reads-plays-soft-2026-10-14]] · [[tie-margin-2026-10-14]] · [[arena]]
