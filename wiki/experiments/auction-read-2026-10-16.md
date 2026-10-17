---
title: The auction read — what a bid, a pass, or a bid that lost tells the table, by itself (#117, 2026-10-16)
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: complete
---

## Summary

Issue #117, assumption A4 ([[book-assumptions]]): the auction is a signal made by a known method.
The book reads it. A 31 is one ten-count off, and a bidder who lost the bid "probably has a double
or two" (S-0021-a, S-0027-b, S-0077-c). Every arena seat has ignored it, so the hidden hands after
the auction are a uniform draw. #149 built the reader that adds each opposing seat's turn in the
auction to Zeb's soft play-reading as a likelihood ([[regulars-table-2026-10-14]]). With the
bidder's method unknown, the read looked worth 0.05–0.19 marks/game there, unregistered and
inside seed noise. This is the registered test of the read by itself, and of the read by a
*known* method, which is the form A4 assumes.

## Results (complete, 2026-10-16; 176 of 176 registered batches ran, none dropped)

**One line.** Reading the auction by the field's known method clears its registered bar: K =
+0.17 [+0.06, +0.29] marks/game against plain deployed Zeb, the sign the book predicts. **But the
A/A cleared too.** Plain Zeb at another planner seed takes +0.12 [+0.02, +0.23] from the same
field, so the read's margin over that seed scatter is K − Za = +0.05 [−0.09, +0.19], a null. What
this table can claim is that the read does not cost. Whether it pays at the size the book implies
waits on a field that is not one planner seed (below).

Data: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
`book/auction-read/` (per batch `meta.json` and `games.jsonl`, `summary.json`, the queue log, `CARD.md`).

### The registered contrasts (paired by game pair; MDE at 80% power)

| contrast | estimate [95%] | pairs | SD/pair | MDE |
|---|---|---|---|---|
| **primary K, the read by a known method** | **+0.17 [+0.06, +0.29] marks/game** | 576 | 1.39 | 0.16 |
| U, the read by an unknown method | +0.07 [−0.04, +0.18] | 576 | 1.35 | 0.16 |
| K − U, the value of knowing the method | +0.10 [−0.03, +0.23] | 576 | 1.65 | 0.19 |
| Za, the A/A (plain Zeb at seed 1) | +0.12 [+0.02, +0.23] | 576 | 1.35 | 0.16 |
| K − Za | +0.05 [−0.09, +0.19] | 576 | 1.72 | 0.20 |
| to 250: PK − PZa, share of games won | +2.6 pts [−1.8, +7.0] | 192 | 0.32 | 6.4 pts |
| to 250: PK − 50% | +3.4 pts [+0.3, +6.5] | 192 | 0.23 | 4.6 pts |

Games to 250 won by team A: PK 53.4% [50.3, 56.5], PZa 50.8% [47.4, 53.9].

### Reading, under [[book-authority]]

- **Assumptions held and violated.** A4 is restored as far as one bidder can restore it: the
  field bids by one fixed, known rule, and K reads every opposing turn of the auction (a bid, a
  pass, a bid since overcalled) under that rule. The book's reader knows a richer method, the
  levels' meanings (31 = one ten-count off, a 35's shape), and reads the declaration and
  partner's bid as well. The heuristic bidder bids only the lowest legal level, so the levels
  carry less than the book's do. A1, A2, A3, A6 and A7 are violated as in every arena block. The
  marks arms violate A5, and the points arms restore it.
- **What our model can see.** The A/A failed: a re-seeded copy of the field beats the field by
  0.12 marks/game, outside its interval. #149 saw the same spread ([[regulars-table-2026-10-14]]:
  Z and Za 0.13 apart against R0). Team B is one planner seed, and planner seed 0 may simply be
  a weaker-than-average draw. Any perturbation of team A's worlds then regresses toward the mean
  and shows as a gain: a new seed (Za, +0.12), a read whose method is unknown (U, +0.07), and a
  read whose method is known (K, +0.17). The read *orders* as the book predicts: the known
  method more than the unknown (K − U = +0.10 [−0.03, +0.23]), and both more than no read
  over the A/A. But no step clears its MDE, and the bootstrap over deals does not see the
  scatter between planner seeds, which is real variance at this size.
- **The book's game.** To 250 the reader wins 53.4% of games, 3.4 points of share over a coin
  ([+0.3, +6.5]), which is the registered secondary. Against the A/A's share the gap is +2.6
  points ([−1.8, +7.0]).
- **The measurement that would settle it.** A field and readers whose planner seed varies by
  game (a seed per game pair, the same in every arm), so that seed scatter becomes part of the
  sampling and the interval sees it. Filed as #190 and run ([[seed-varied-2026-10-16]]): its A/A
  missed too, the other way, with no per-seed effect, so the open question is the estimator's
  calibration (#191).

### Resolution (#191, 2026-10-17)

The pair this A/A used, seed 1 against seed 0, was measured over twenty times the deals: δ(1,0) =
−0.004 [−0.040, +0.031] ([[estimator-calibration-2026-10-16]]). So this page's A/A (+0.12) was
chance, and the estimator is calibrated (R_F 1.06). K = +0.17 [+0.06, +0.29] cleared its
registered bar and stands. #190's K, on disjoint deals, agrees.

### Receipts

Pre-registered at a389af35 (`scenario/auction_read.py`). 176 batches through `scripts/cpu-slot`,
two at a time, 25–149 s each including the wait for a slot. Every batch directory holds its 24 games; none dropped. Timing pilot
seed 3290 excluded. Run 2026-10-16 09:47–10:51.

## Design (pre-registered before any measured batch; see `scenario/auction_read.py`)

- **Players.** Every seat plays deployed Zeb 0.3.0 and bids by the heuristic rule at caution 0.
  Team B: plain Zeb. Team A: **K**, Zeb reading the auction under the field's true method (the
  heuristic bidder at caution 0, trembling 0.1); **U**, the same with the caution flat on {−10,
  −5, 0, 5, 10} (#149's R0); **Za**, plain Zeb at planner seed 1, the A/A. K and U share team B's
  planner seed, so where the auction carries no evidence they play team B's tiles.
- **Arms.** Marks: K, U, Za. The book's game (250 points): PK, PZa.
- **Primary.** K, team A's margin in marks per game. **The book predicts K > 0.**
- **Secondaries.** U; K − U (the value of knowing the method); Za (should cover 0); K − Za; to
  250, PK − PZa and PK − ½ on games won.
- **Size.** 576 game pairs per marks arm (seeds 3301–3348 × 12), 192 per points arm (3401–3416 ×
  12); 176 batches through `scripts/cpu-slot`. MDE on K about 0.14 marks/game. Timing pilot seed
  3290 excluded (20–23 s per 4-pair batch).
- **Estimator, batch-drop rule and fence.** As [[book-small-n-2026-10-14]]. The read is a
  likelihood over this hand's public auction ([[strategy-fusion-fence]]); every player thinks on
  a fixed world budget.

## Links

[[book-authority]] · [[book-assumptions]] · [[regulars-table-2026-10-14]] · [[strategy-fusion-fence]]
