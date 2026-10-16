---
title: The auction read — what a bid, a pass, or a bid that lost tells the table, by itself (#117, 2026-10-16)
kind: experiment
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
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

*Results follow once every registered batch has run.*

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
