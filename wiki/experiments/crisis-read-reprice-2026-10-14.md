---
title: The crisis as read, priced in the book player's stack (#162's next step, 2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## Summary

`silas/examples/book/crisis-read.silas` is the book's crisis as Hand 31 reads it: its two-trump
trigger asks that one setter certainly holds both trumps out, read from the setters' plays under
`protect-the-count`, where `crisis.silas` fires on any two trumps out the bidder cannot draw
([[silas]] § the record as evidence). It is the first book rule that acts on a reading of another
seat's plays ([[book-assumptions]] A2). This study prices it in the book player's stack before the
stack adopts it, under [[book-authority]]: the book's crisis is the read one.

*Results follow once every registered batch has run.*

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
