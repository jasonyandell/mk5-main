---
title: The book's own table — opponents who read the book player by the book (2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

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

## Links

[[book-investigation-2026-10-14]] · [[book-head-to-head-2026-10-13]] ·
[[reads-plays-2026-10-13]] · [[silas]] · [[scenario-player]] · [[w42]] ·
[[strategy-fusion-fence]]
