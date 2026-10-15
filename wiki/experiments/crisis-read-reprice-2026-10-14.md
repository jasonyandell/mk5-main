---
title: The crisis as read, priced in the book player's stack (#162's next step, 2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## Summary

`silas/examples/book/crisis-read.silas` is the book's crisis as Hand 31 reads it: its two-trump
trigger asks that one setter certainly holds both trumps out, read from the setters' plays under
`protect-the-count`, where `crisis.silas` fires on any two trumps out the bidder cannot draw
([[silas]] § the record as evidence). It is the first book rule that acts on a reading of another
seat's plays ([[book-assumptions]] A2). This study prices it in the book player's stack before the
stack adopts it, under [[book-authority]]: the book's crisis is the read one.

## Results (complete, 2026-10-14; 48 of 48 registered batches ran, none dropped)

**One line.** Putting the crisis as read in the book player's stack costs nothing measurable:
Q = R − C = +0.01 [−0.02, +0.04] marks/game over 192 game pairs, its MDE 0.04 (the per-pair SD came
in at 0.20, far under the 1.0 sized for). The two stacks played out differently in 23 of 192 game
pairs (12%). In the book's game to 250 they are level too. Under the registered rule (adopt unless
Q's interval lies below 0) the stack adopts `crisis-read`.

Data: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
`book/crisis-read-reprice/`.

| contrast | value [95%] | pairs | SD/pair | MDE |
|---|---|---|---|---|
| **primary Q = R − C, marks/game** | **+0.01 [−0.02, +0.04]** | 192 | 0.20 | 0.04 |
| 250: PR − PC, share of games won | +0.5 pts [−1.0, +2.1] | 96 | | 2.5 pts |
| 250: PR − PC, settled points/game | +1.8 [−0.5, +4.2] | 96 | 11.8 | 3.4 |
| game pairs that differ, marks / 250 | 23 of 192 (12%) / 14 of 96 (15%) | | | |

Each arm against deployed Zeb: C −0.90 [−1.13, −0.68], R −0.89 [−1.11, −0.67] marks/game; to 250,
PC 33.9% [28.1, 39.6] and PR 34.4% [28.6, 40.1] of games.

### Reading, under [[book-authority]]

- **Assumptions.** The change restores one reading the book makes (A2, a seat reading another's
  play); everything else is the head-to-head's block c, violating A1, A3, A4 and A6.
- **What our model can see.** The book's own crisis, the one it knows from a count play, plays
  as well as the encoding's broader trigger at a twentieth of a mark's resolution, and differs from
  it in about one game pair in eight. The encoding's two-trump crisis fired where the book would
  still be drawing; the read one waits. Neither changes the book player's standing against Zeb.
- **Adopted.** `arena/book_play.py`'s lead order names `crisis-read` where it named `crisis`;
  `crisis.silas` stays as the encoding the head-to-head and the investigation priced. Every block
  run before this commit used `crisis`.

### Receipts

Pre-registered at f105f56a (`scenario/crisis_reprice.py`); 48 batches through `scripts/cpu-slot`, two
at a time, one queue copy checked; none dropped. Timing pilot seed 2390 excluded (its first run
failed on the binding's default work cap and was rerun after the cap was raised; no measured batch
ran before the registration). At adoption the runner's arms were rewritten so each plays as
registered: C swaps the stack's `crisis-read` back to `crisis`, R plays the stack.

The study ran under a work cap of 10^12 per decision, which is no cap in practice. Each decision's
work is now recorded. Replaying R at seed 2401 and PR at seed 2451, the most any decision of the
whole stack spent was 3.95 × 10^7, four times the binding's default, and the replayed games were
byte-identical to the published batch. The cap is now 10^9 (`arena.book_play.BUDGET`).

## Design (pre-registered before any measured batch; see `scenario/crisis_reprice.py`)

- **Players.** Team B is deployed Zeb 0.3.0; both teams bid by the heuristic rule at caution 0.
  Team A is the head-to-head's book player, block c ([[book-head-to-head-2026-10-13]]): **C** the
  stack as it stands, **R** the same stack with `crisis` compiled from `crisis-read` (`BookPlay`'s
  `swap`). They differ only where `crisis` fires and `crisis-read` does not.
- **Arms.** Marks: C, R. The book's game (250 points): PC, PR.
- **Primary.** Q = R − C marks/game; the prediction is Q ≥ 0, and the stack adopts `crisis-read`
  unless Q's interval lies below 0. Reported with the share of game pairs whose outcome differs.
- **Secondaries.** PR − PC on games won and settled points; each arm's margin; the share of
  differing game pairs by game.
- **Size.** 192 game pairs per marks arm (seeds 2401–2416 × 12), 96 per points arm (2451–2458 ×
  12): MDE 0.20 at SD 1.0. 48 batches through `scripts/cpu-slot`, two at a time. The book
  player's policies run under a work cap of 10^12 per decision (a reading exceeds the binding's
  default 10^7; a cap changes no tile a policy allows within it).
- **Estimator, batch-drop rule and fence.** As [[book-small-n-2026-10-14]].

## Links

[[book-authority]] · [[book-assumptions]] · [[book-head-to-head-2026-10-13]] · [[silas]] ·
[[no-plan-2026-10-14]] · [[strategy-fusion-fence]]
