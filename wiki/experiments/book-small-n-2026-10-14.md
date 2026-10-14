---
title: The book at small n — do Winning 42's rules raise a bounded Zeb? (#154, 2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## Summary

Issue #154, the [[book-authority]]'s measurement of assumption A7 ([[book-assumptions]]): the
book is a policy for a reader who cannot enumerate 512 worlds a decision. Every arena block has
compared it against deployed Zeb at full compute; its one measured win is as a planner's
near-tie break in the book's own game ([[book-investigation-2026-10-14]] § II). Here the book
is handed to a Zeb that samples 16, 64 or 160 worlds, and the question is whether the rules
raise it, and whether they raise it more the fewer worlds it sees.

*Results follow once every registered batch has run.*

## Design (pre-registered before any measured batch; see `scenario/small_n.py`)

`scenario/small_n.py` holds the registration in its docstring.

- **Players.** Team B, the opponent in every arm, is deployed Zeb 0.3.0's plan
  (`k0,n160,F=<zeb/net>,f=16,d=posterior,r=uniform:2^22:0.28,t=margin`;
  [[zeb-030-confirmation-2026-10-14]]). Team A is the same plan at n ∈ {16, 64, 160} worlds:
  **Z<n>** the plan alone; **T<n>** with the book as its near-tie break (`book:zeb,zeb,w0.02`,
  the form that won the book's game); **F<n>** with the book as a filter (`book:zeb,zeb`, the
  investigation's variant (c)); **A160**, Z160 at planner seed 1, the A/A. Both teams bid by the
  heuristic rule at caution 0.
- **Arms.** Marks (games to 7): Z16, T16, F16, Z64, T64, Z160, T160, F160, A160. The book's
  game (250 points): PZ16, PT16, PZ160, PT160.
- **Estimator.** A game pair is (base seed, game index) under the arena's paired halves; each
  contrast is taken per pair over the pairs every arm holds; 95% percentile bootstrap over
  game pairs (10,000 resamples, seed 42). Duplicate deals across every arm.
- **Primary.** D16 = T16 − Z16 in marks per game. **The book predicts D16 > 0.** Read against
  the A/A (A160 − Z160, whose interval should cover 0) and the bar: above 0 = the rules raise a
  bounded player; below 0 = they cost even as a tie-break at 16 worlds; else a null with its
  minimum detectable effect.
- **Secondaries.** The trend D16 > D64 > D160 on the paired differences; the filter's cost
  E<n> = F<n> − Z<n> at 16 and 160; the cost of bounded compute itself (Z16 − Z160, Z64 − Z160)
  and whether the book repays it (T16 − Z160); the book's game on games won and settled
  points.
- **Size.** Timing pilot (seed 1690, excluded): 4 game pairs of Z16, T16, Z160 and T160 took
  29–42 s a batch at load average about 40, two batches at once. The primary's per-pair SD was
  0.65 over 4 pairs, too few to trust; the sizing takes SD 1.5, so 288 pairs give a half-width
  of 0.17 marks/game and a minimum detectable D16 of 0.25 (80% power). Marks: seeds 1701–1724
  × 12 pairs = 288 game pairs per arm, nine arms. 250 points: seeds 1801–1816 × 12 = 192 pairs
  per arm, four arms. Two batches at once through `scripts/cpu-slot`, seed-major.
- **Batch-drop rule.** A batch that exits nonzero is dropped, not rerun, and listed. No effect is
  computed until every registered batch has run. Every batch runs under `scripts/cpu-slot`.
- **Fence.** Every value is Zeb's plan over the mover's own cell, at fewer worlds; the rules
  read the frame alone ([[strategy-fusion-fence]]).

## Links

[[book-authority]] · [[book-assumptions]] · [[book-investigation-2026-10-14]] ·
[[scenario-player]] · [[silas]] · [[strategy-fusion-fence]]
