---
title: Opponents who read the book player by the book, within one hand (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

> **Read under [[book-authority]] (2026-10-14).** This page first called its table "the book's
> own table". It is not. The book's opponent remembers the player across hands and changes its
> own play in response; this reader does neither. It uses the book only to place tiles within a
> hand, and its partner-side reading is off (A1, A2; [[book-assumptions]]). The null below is
> narrow: within-hand tile location by the book's verdict sets is not where the book's edge
> lives. The numbers are unchanged.

## Question

[[book-investigation-2026-10-14]] left one table untested, the one the book is written for:
partners who both play the book, and opponents who read their plays by it (#147). Does the
book player do better there? And the converse: does a Zeb that reads the book seats by the
book exploit them more than plain Zeb does?

- *Both partners book* is already the head-to-head's table. The arena seats one play policy
  per team, so the book player has always held both seats of its side ([[book-head-to-head-2026-10-13]],
  `arena/engine.py`, `_run_lockstep`). Block B0 below is that table.
- *Opponents who read by the book* is new here. The investigation's reader read every seat by
  the field net's set, so against the book it was a mis-specified reader (§ IV there).

## The answer, in short

- **A within-hand reader by the book changes nothing for the book player.** Against Zebs that read it by the book,
  the book player scores as it does against plain Zeb: −1.31 [−1.56, −1.06] marks a game and
  33% of games, against −1.27 on the same deals. The paired difference is −0.04 [−0.27, +0.19]
  marks a game, and −2.3 [−6.4, +1.9] points of game share to 250. Both are null.
- **So the converse is null too.** A Zeb that reads the book seats by the book exploits them no
  more than plain Zeb does, at this power.
- **Reading by the book costs the reader against Zeb.** Plain Zeb beats the book-aware reader
  by +0.50 [+0.26, +0.74] marks a game and wins 58.7% [54.2, 63.3] of games to 250. Zeb's plays
  leave the book's sets: the reader's sample holds no world that explains the record at 43% of
  its decisions (87% at trick six), and where some remain, they are the wrong ones.
- **IV's contrast reads below zero, but not because the reader exploits the book.**
  C = −0.54 [−0.89, −0.19] marks a game (−11.0 [−17.0, −4.9] points of game share). Its whole
  weight is the reader's loss to Zeb; its gain against the book is null.
- *Both partners book* was already the head-to-head's table: one play policy holds both seats
  of a team.

## The reader

`arena/book_read.py` (`bookread:consistent/<walt options>`) is deployed Zeb with one change:
its live worlds are weighted by the likelihood that its two opponents played by the book.

- *The likelihood.* At each discretionary play of an opponent, the book stack's verdict at
  that seat's frame of the time: its hand in the world plus the tiles it played since, and the
  record before the play. The deciding policy's set is narrowed as the book player narrows it
  (`book_play.decide`). Consistent: a play weighs 1 when the set holds it, else 0.
- *Silence.* A play no rule decides weighs 1, as Silas reads a fallback (`silas/src/reading.rs`).
  The book player's least-id pick inside a set is a tie-break and is not read.
- *The partner* is unread, uniform as plain Zeb holds it. So the reader differs from plain Zeb
  only in what it believes about the book seats.
- *The draw.* The core samples plain Zeb's 512 uniform worlds and the weights enter through
  the belief seam (`CompiledWalt`). Where the fiber holds at most 512 worlds, those are the
  whole fiber and the weights are the exact posterior (Silas's `exact_given` count). Where it
  is sampled, the weighted sample is an importance estimate of that posterior, with its
  effective sample size logged at every decision. Where no sampled world explains the record,
  the reader plans over the uniform sample, as plain Zeb, and the decision counts as a
  fallback.
- *Why not exact draws.* Compiling the stack into the core, or drawing from Silas's per-seat
  counts, is the faithful route (#147). The pilot below says the weighted sample keeps an
  effective 170–270 worlds through trick 4 against the book, so the cheap route is taken and
  its ESS is reported.
- *The fence.* The likelihood is the book's verdict at a seat's own frame; every value is
  Zeb's plan over the reader's own cell. No world is valued by its hidden hands
  ([[strategy-fusion-fence]]).

**Pilot (seed 900, in no number; seen before this pre-registration).** One batch of 36 games
each: BR read the book team at −1.06 marks a game, ZR read +0.42 for plain Zeb. BR took
118 s and ZR 212 s on a loaded machine (load average 27 from other sessions), so the measured
batches run 24 games.

| trick | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ESS fraction, against the book | 0.78 | 0.53 | 0.42 | 0.36 | 0.38 | 0.50 |
| fallback, against the book | 0% | 0% | 0% | 0.2% | 0% | 0% |
| ESS fraction, against Zeb | 0.69 | 0.33 | 0.18 | 0.10 | 0.05 | 0.04 |
| fallback, against Zeb | 1% | 14% | 31% | 51% | 72% | 88% |
| exact fiber (either table) | 0% | 0% | 0% | 17–19% | 80% | 100% |

Against the book the real deal always fits (the book player's plays are inside the verdict by
construction), so the posterior never empties. Against Zeb it empties often from trick 3: Zeb's
plays leave the book's sets.

## Pre-registration (written and committed before any measured batch)

**Blocks.** Z is deployed Zeb, `walt:k0,n512,F=margin_d250k_h128x64,f=16`; `heuristic` bidding
for both teams; the book player is (b), `book:zeb,least/Z`, with the repaired encoding (#146).

| block | team A | team B | game |
|---|---|---|---|
| B0 | book (b) | Z | 7 marks |
| BR | book (b) | Z reading A by the book (consistent) | 7 marks |
| ZR | Z | Z reading A by the book (consistent) | 7 marks |
| P-B0, P-BR, P-ZR | as above | | 250 points, settled by the book's rule |

- *Seeds and size.* Base seeds 1–11, 24 games a batch (12 deal clusters a batch, 132 a block,
  264 games), every batch under `timeout 300`. A batch that hits the cap is dropped, not rerun,
  and reported; its clusters leave every contrast.
- *B0 and P-B0* are the published repaired blocks b′ and P-b′ (`book/encoding-repairs/`),
  restricted to the same clusters (games 0–11 of each orientation at each seed). A deal depends
  on the base seed, the game index and the hand, so these are the same deals. First, B0 at seed 1
  and 24 games is re-run here and must match the published games game for game; if it does not,
  B0 and P-B0 are re-run here in full at 24 games a batch.
- *Baselines.* Zeb against Zeb reads exactly 0 marks a game and 50% of games (deterministic
  per coordinate, both orientations).
- *Intervals.* 95% percentile bootstrap over deal clusters (both orientations of a game index
  at a base seed), 10,000 resamples. A contrast between blocks is paired by cluster.

**Measures.** Team A's mark margin per game (marks blocks); team A's share of games won and
its settlement margin per hand (points blocks).

**Primary: the book's table.** Δ = BR − B0 in marks a game, and Δ_P = P-BR − P-B0 in game
share, paired by cluster. One contrast answers both questions, with opposite signs:
- interval above 0: the book player does **better** against opponents who read it by the book
  than against plain Zeb;
- interval below 0: the book-aware reader **exploits** the book player more than plain Zeb does;
- otherwise **null**.

**Secondary.**
- *The reader against Zeb* (ZR, P-ZR): plain Zeb's marks a game against 0 and game share
  against 50%. Above: a reader holding the book's expectations is **thrown off** by Zeb, who
  follows no book. Below: it beats plain Zeb.
- *IV's contrast,* C = Δ − ZR: the reader's gain against the book less its gain against Zeb, as
  in [[book-investigation-2026-10-14]] § IV. Above 0: the reader is thrown off by the book's
  play more than by Zeb's; below 0: it exploits the book's play more than Zeb's; else null.
- *The book's table, absolute.* BR's marks a game against 0, P-BR's game share against 50%.

**Mechanism (descriptive, no band).** The reader's ESS fraction and fallback rate by trick,
against the book and against Zeb; the share of its decisions on an exact fiber.

**Not measured, stated as gaps.**
- The uniform likelihood (restricted choice), for cost.
- The book team reading each other by the book: its Zeb fallback stays plain Zeb.
- How often the reading changes the reader's move (needs plain Zeb's tile at each decision).
- Exact draws on sampled fibers (above).

## Results

All 44 batches ran as registered: none near the cap (slowest 176 s), none dropped. B0 at seed 1
re-ran game for game (24 of 24), so B0 and P-B0 are the published games on the same 132
clusters. The analysis reproduces the published full blocks (−1.41 [−1.63, −1.19], 38.4%
[34.6, 42.2]).

| team A | team B | marks/game [95%] | games won to 250 [95%] | settlement/hand (250) | made, A / B (marks) |
|---|---|---|---|---|---|
| book (b) | Z (B0, P-B0) | −1.27 [−1.53, −1.00] | 39.4% [34.8, 43.9] | −2.9 [−4.0, −1.8] | 54.9% / 66.3% |
| book (b) | Z reading A by the book (BR, P-BR) | −1.31 [−1.56, −1.06] | 37.1% [32.2, 42.0] | −3.5 [−4.6, −2.4] | 55.0% / 66.7% |
| Z | Z reading A by the book (ZR, P-ZR) | **+0.50 [+0.26, +0.74]** | **58.7% [54.2, 63.3]** | +2.3 [+1.4, +3.3] | 62.8% / 58.4% |
| Z | Z (the baseline) | 0 | 50% | 0 | |

| contrast (paired by cluster) | marks/game | game share to 250 | band |
|---|---|---|---|
| Δ = BR − B0 (primary) | −0.04 [−0.27, +0.19] | −2.3 [−6.4, +1.9] points | null, null |
| ZR against the baseline | +0.50 [+0.26, +0.74] | +8.7 [+4.2, +13.3] points | thrown off, thrown off |
| C = Δ − ZR (IV's contrast) | −0.54 [−0.89, −0.19] | −11.0 [−17.0, −4.9] points | below 0 |

- *The primary is null in both games.* The interval of ±0.23 marks a game would show an effect
  the size of the repairs (+0.47); it does not show one of a tenth of a mark.
- *The reader is thrown off by Zeb.* Holding the book's expectations of a player who follows no
  book costs it half a mark a game and nine points of game share. Its contracts are made at 58%
  against 63% for plain Zeb's.
- *C's band reads "exploits the book's play more than Zeb's", and that is only true relatively.*
  The reader does better against the book than against Zeb because it reads the book rightly
  and Zeb wrongly, not because it gains against the book: Δ is null.

**The reader's sample** (every decision with a choice; the marks and points blocks agree to a
point):

| trick | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| ESS fraction, against the book | 0.77 | 0.52 | 0.41 | 0.35 | 0.38 | 0.53 |
| effective worlds, against the book | 395 | 267 | 209 | 165 | 72 | 8 |
| fallback, against the book | 0% | 0% | 0% | 0.1% | 0% | 0% |
| ESS fraction, against Zeb | 0.71 | 0.34 | 0.18 | 0.10 | 0.06 | 0.04 |
| fallback, against Zeb | 0.5% | 11% | 29% | 51% | 71% | 87% |
| exact fiber (either) | 0% | 0% | 0% | 14–16% | 78–79% | 100% |

- *Against the book the posterior is well fed:* 165 or more effective worlds through trick 4,
  the exact posterior from trick 5 at most decisions, and almost never empty. So the null is
  not the sample starving.
- *Against Zeb it starves:* fewer than 50 effective worlds from trick 4, and empty at most
  late decisions, where the reader then plays as plain Zeb. Its loss comes from the decisions
  before that, where a few wrong worlds survive.

## What this means

- **A reader who places tiles by the book's rules, within one hand, does not move the book
  player's score.** That reader takes the verdict set as an event. It neither punishes the book
  player's predictability nor is fooled by it. The opponent the book assumes is not seated:
  one who remembers across hands and changes its own play (A1, #149).
- **Reading by a wrong model is costly, and the cost is measurable.** The same reader that
  breaks even against the book loses half a mark a game to Zeb. Policy blindness, in Jason's
  phrase, is real here, but it is Zeb's: a reader who expects the book is thrown off by a
  player who does not follow it.
- **Nothing is adopted.** Zeb's default is unchanged. The reader is an arena player only.

## Deviations from the pre-registration

- None in the blocks, the seeds or the analysis. The bootstrap is this study's own
  (`analyze.py`, seed 147), checked against the published blocks' numbers.
- Batches ran beside other sessions' work (load average 27–54 at the start). 44 batches:
  median 99 s, slowest 176 s.

## What our model leaves out

- **Memory across hands** (A1). The reader forgets the book player between hands and never
  changes its own bidding or play to exploit what it reads, which is the book's S-0080-c (against an
  overbidder, pass and set him). The measurement is #149.
- **The book team reading itself** (A2). Its partner-side worlds stay uniform. The measurement
  is #150.
- **Exploitation rather than location.** A model of an opponent can be used to place tiles,
  which is the only use measured here, or to choose plays that the opponent's predictable
  responses make better. The second use is unmeasured.

## Caveats

- One likelihood (consistent) and one reader: the uniform likelihood, the book team reading
  each other, and exact draws on sampled fibers are not run (the gaps above).
- The weights enter a uniform sample, and the planner's cell floor counts worlds, not weight:
  a cell may count 16 worlds of which few carry weight. That leans toward a thinner plan than
  the floor intends, on both tables alike.
- 132 clusters a block, two-thirds of the investigation's: intervals about ±0.25 marks a game.

## Data

`jasonyandell/texas-42-zeb-research-2026-10` under `book/reads-by-the-book/` (CC0,
`SHA256SUMS`, `CARD.md`): every game and every reader decision (`out/`), the pilot, the seed-1
reproduction of B0, the logs, `results.json`, `analysis.log` and the scripts. B0 and P-B0 are
`book/encoding-repairs/out/b` and `P-b`. Code: `arena/book_read.py`,
`arena/test_book_read.py`, the batch seam in `scenario/compiled.py`.

## Links

[[book-investigation-2026-10-14]] · [[book-head-to-head-2026-10-13]] ·
[[reads-plays-2026-10-13]] · [[silas]] · [[scenario-player]] · [[w42]] ·
[[strategy-fusion-fence]]
