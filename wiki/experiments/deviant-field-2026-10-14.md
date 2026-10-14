---
title: The named deviant field — the book's bidding against the overbidders it was written for (#152, 2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## Summary

Issue #152, the [[book-authority]]'s measurement of assumption A6 ([[book-assumptions]]): the
book's bidding is written for a field with styles, overbidders above all, and every arena block
has seated it against a field with none, where `risk-budget` cost −0.71 [−1.00, −0.41]
marks/game ([[book-head-to-head-2026-10-13]]). Here the field overbids by a named amount, k ∈
{0, 5, 10} points past its hand's ceiling, and the question is the k at which the book's
bidding stops costing. It is also the arena half of cluster E's ten fixed-field rows
(`silas/docs/cross-game.md`, #130 slice 4), and S-0114-b's claim that points punish frivolous
bidding where marks do not.

*Results follow once every registered batch has run.*

## Design (pre-registered before any measured batch; see `scenario/deviant_field.py`)

`scenario/deviant_field.py` holds the registration in its docstring.

- **Players.** Every seat plays Zeb 0.3.0's plan ([[zeb-030-confirmation-2026-10-14]]). Team
  B, the field, bids by the arena's heuristic rule at caution −k, k ∈ {0, 5, 10}: willingness
  is the ceiling plus k, one or two bidding steps past the hand. Team A: **H<k>** Zeb's play
  bidding heuristic at caution 0 (the control); **R<k>** Zeb's play bidding the book's
  `risk-budget`; **B<k>** the book player, the head-to-head's block c, bidding heuristic.
- **Arms.** Marks: H0, H5, H10, R0, R5, R10, B0, B5, B10. The book's game (250 points): PH0,
  PH5, PH10, PR0, PR5, PR10.
- **Estimator.** As [[book-small-n-2026-10-14]]: game pairs under the arena's paired halves,
  contrasts per pair, 95% percentile bootstrap (10,000 resamples, seed 42).
- **Primary.** C<k> = R<k> − H<k> in marks per game at each k. **The book predicts C<k> rises
  with k**; the claim is the first k whose interval covers or lies above 0. Read on C5 − C0 and
  C10 − C0 against the bar: C10 − C0 above 0 = the book's bidding is worth more against its own
  field; below 0 = worth less; else a null with its minimum detectable effect.
- **Secondaries.** The field itself (H<k> against 0; the book predicts overbidders set
  themselves); the book player B<k> − H<k> and B10 − B0; S-0114-b, the same pair under points
  (PR<k> − PH<k> on games won and settled points) against marks; mechanism: contracts per
  hand, share made, mean bid by arm.
- **Size.** Timing pilot (seed 1890, excluded): 4 game pairs of H0, R10, B10 and PR10 took
  29–48 s a batch at load average about 42; H0 read exactly 0 against the k = 0 field, the same
  player both sides. In the pilot the k = 10 field took 71% of the contracts (mean bid 31) and
  `risk-budget` answered it with 84s (mean bid 48 when it bid). Sizing at SD 1.5 per pair: 192
  pairs give a half-width of 0.21 marks/game and a minimum detectable C10 − C0 of 0.30 (80%
  power). Marks: seeds 1901–1916 × 12 pairs = 192 pairs per arm, nine arms. 250 points: seeds
  1951–1958 × 12 = 96 pairs per arm, six arms. Two batches at once through `scripts/cpu-slot`,
  queued after #154's batches.
- **Batch-drop rule and fence.** As [[book-small-n-2026-10-14]]; every batch through
  `scripts/cpu-slot`.

## Links

[[book-authority]] · [[book-assumptions]] · [[book-head-to-head-2026-10-13]] ·
[[book-small-n-2026-10-14]] · [[scenario-player]] · [[silas]] · [[strategy-fusion-fence]]
