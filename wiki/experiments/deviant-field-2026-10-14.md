---
title: The named deviant field — the book's bidding against the overbidders it was written for (#152, 2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-22
status: complete
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

## Results (complete, 2026-10-14; 192 of 192 registered batches ran, none dropped)

**One line.** The book's bidding (`risk-budget`) costs less the more the field overbids, the
direction the book predicts, but the shrink is a null at this size, and the bidding still costs at
k = 10. Against the field that bids its hand (k = 0) it costs C0 = 0.92 [0.60, 1.22] marks/game;
against a field 10 points past its hand, C10 = 0.57 [0.22, 0.91]. The registered read, C10 − C0 =
+0.35 [−0.02, +0.71], leans the book's way and touches 0; its minimum detectable effect is 0.52,
not the 0.30 registered, because the per-pair SD came in at 2.4 against the 1.5 sized for. In the
book's own game, points to 250, the same bidding is level with the control at every k.

Data: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
`book/deviant-field/` (per batch `meta.json` and `games.jsonl`, `summary.json`, the queue log, `CARD.md`).

### The arms (team A's margin against the field; 384 games a marks arm, 192 a points arm)

| arm | team A | field k | marks/game [95%] (points arms: share of games won) | A's contracts / hand | made A / B | mean bid A / B |
|---|---|---|---|---|---|---|
| H0 | heuristic, caution 0 | 0 | 0 exactly (the same bidder both sides) | 50% | 60.2% / 60.2% | 30.6 / 30.6 |
| H5 | heuristic | 5 | −0.24 [−0.47, −0.01] | 36% | 58.4% / 56.5% | 31.0 / 30.8 |
| H10 | heuristic | 10 | −0.02 [−0.29, +0.25] | 23% | 55.8% / 51.9% | 31.4 / 31.2 |
| R0 | `risk-budget` | 0 | −0.92 [−1.22, −0.60] | 55% | 51.2% / 61.7% | 39.6 / 30.9 |
| R5 | `risk-budget` | 5 | −0.86 [−1.18, −0.55] | 41% | 50.1% / 57.3% | 41.5 / 31.4 |
| R10 | `risk-budget` | 10 | −0.59 [−0.90, −0.27] | 30% | 49.1% / 53.8% | 44.4 / 32.0 |
| B0 | the book player (block c) | 0 | −1.00 [−1.23, −0.78] | 50% | 55.7% / 64.9% | 30.6 / 30.6 |
| B5 | the book player | 5 | −1.32 [−1.58, −1.07] | 36% | 53.7% / 61.4% | 31.0 / 30.8 |
| B10 | the book player | 10 | −1.16 [−1.41, −0.90] | 23% | 51.3% / 57.2% | 31.4 / 31.2 |
| PH0, PH5, PH10 | heuristic, to 250 | 0, 5, 10 | 50.0%, 54.7% [49.0, 60.4], 60.4% [54.7, 66.1] | | | |
| PR0, PR5, PR10 | `risk-budget`, to 250 | 0, 5, 10 | 50.5% [43.8, 57.3], 52.6% [45.8, 59.4], 59.4% [52.6, 66.1] | | | |

### The registered contrasts (paired by game pair; MDE at 80% power)

| contrast | marks/game [95%] | pairs | SD/pair | MDE |
|---|---|---|---|---|
| **primary C0 = R0 − H0** | **−0.92 [−1.22, −0.60]** | 192 | 2.24 | 0.45 |
| **primary C5 = R5 − H5** | **−0.62 [−0.97, −0.28]** | 192 | 2.43 | 0.49 |
| **primary C10 = R10 − H10** | **−0.57 [−0.91, −0.22]** | 192 | 2.39 | 0.48 |
| C5 − C0 | +0.29 [−0.00, +0.58] | 192 | 2.05 | 0.42 |
| **C10 − C0 (the read)** | **+0.35 [−0.02, +0.71]** | 192 | 2.58 | 0.52 |
| the field: H5 against 0 | −0.24 [−0.47, −0.01] | 192 | 1.62 | 0.33 |
| the field: H10 against 0 | −0.02 [−0.29, +0.25] | 192 | 1.92 | 0.39 |
| the book player B0 − H0 | −1.00 [−1.23, −0.78] | 192 | 1.59 | 0.32 |
| the book player B5 − H5 | −1.08 [−1.34, −0.82] | 192 | 1.85 | 0.37 |
| the book player B10 − H10 | −1.14 [−1.41, −0.86] | 192 | 1.96 | 0.40 |
| B10 − B0 | −0.16 [−0.45, +0.14] | 192 | 2.12 | 0.43 |

S-0114-b, the same pair in the book's game (96 game pairs per arm):

| k | PR − PH, share of games won | PR − PH, settled points/game | exploratory: R − H in the marks game, share of games won |
|---|---|---|---|
| 0 | +0.5 pts [−6.3, +7.3] | −3.8 [−20.4, +12.8] | −10.7 pts [−15.1, −6.2] |
| 5 | −2.1 pts [−9.9, +5.7] | −5.0 [−24.7, +14.9] | −6.8 pts [−12.5, −1.0] |
| 10 | −1.0 pts [−8.3, +5.7] | −10.4 [−29.5, +8.5] | −5.0 pts [−10.4, +0.5] |

The last column is not registered: it scores the marks arms on games won so that the two games
read in one unit (MDE about 8 points in the marks game, about 10 in the book's).

### Reading, under [[book-authority]]

- **Assumptions held and violated.** A6 is restored in one dimension: the field overbids by a
  named, fixed amount. It is still a style-free field in every other respect: deterministic, the
  same overbid on every hand, no frivolous bids on a partner's hand or to stop the opponents (the
  book's overbidders, S-0080-c, S-0155-a), and it never learns. A1, A2, A3 and A4 are violated as
  in every arena block. The marks arms violate A5; the 250-point arms restore it.
- **What our model can see.** Against a field that overbids, the cost of the book's bidding falls
  from 0.92 to 0.57 marks/game, and the mechanism is the book's: at k = 10 the field takes 70% of the
  contracts against `risk-budget` and makes 53.8% of them, against 61.7% at k = 0. The shrink is
  +0.35 [−0.02, +0.71], a null leaning the book's way, about 1.7 times underpowered for the
  registered 0.30. The bidding still costs at k = 10: its own contracts are made 49.1% of the time at
  a mean bid of 44.
- **The book's own game.** In marks the book's bidding wins 10.7 [6.2, 15.1] points fewer games
  than the control at k = 0 (exploratory). In points to 250, where a set pays the setters and a high
  bid's reward is in the count, it is level at every k (+0.5 [−6.3, +7.3] at k = 0). The two games
  are not paired, so the gap between them carries no interval of its own; what each says alone is
  that the cost of the book's bidding is a marks game's cost, consistent with A5. *Later
  (2026-10-15):* the same pairing on new seeds, [[overbidders-2026-10-14]]'s PRH over 192 pairs,
  wins 38.8% [34.1, 43.5]; PR0 seed 1951 re-run at 363b0c41 is byte-identical, so the gap is the
  draw, and k = 0's level reading here was likely a high one.
- **What the field does to itself.** The book predicts overbidders set themselves (H<k> above 0).
  Against Zeb's caution-0 bidder the k = 5 field gains 0.24 [0.01, 0.47] marks/game, and the k = 10
  field is level (−0.02 [−0.29, +0.25]): in this arena a mechanical overbid of one step pays.
- **S-0114-b, seen in the field's own arms.** In the book's game the overbidding field is punished:
  the plain bidder beats it in 54.7% [49.0, 60.4] of games to 250 at k = 5 and 60.4% [54.7, 66.1] at
  k = 10, against 50% at k = 0, where it is level or ahead in marks. Points punish frivolous bidding
  where marks do not, as the book says (S-0114-b). These are arm margins against their own chance
  baseline, not a registered contrast between the games, which differ in seeds and size.
- **The book player** costs about a mark a game against every field (−1.00, −1.08, −1.14), and the
  field's style does not move it (B10 − B0 −0.16 [−0.45, +0.14]): its play, not its bidding, is the
  head-to-head's gap ([[book-head-to-head-2026-10-13]]).
- **The measurement that restores more of A6.** A field whose overbids are the book's kind
  (frivolous bids on a partner's hand, bids to stop the opponents, a varied style rather than a
  fixed step), and the same three arms at 576 game pairs, which gives the registered MDE of 0.30 on
  C10 − C0 at the observed SD. Filed as #163.

### Receipts

Pre-registered at 98f5d16c (`scenario/deviant_field.py`). 192 batches through `scripts/cpu-slot`, two
at a time by design, 30–50 s a batch at load average 19–42. For about 40 minutes a second copy of
the queue ran alongside (a chained start that survived its shell) and the two together held three
slots; it was stopped. Two batches ran twice, H5 s1912 and R5 s1902: each batch writes one
directory per arm and seed and every batch is seeded, so the second run rewrote the same deals in
place and each seed counts once. Every batch holds its 24 games; none dropped. Timing pilot seed
1890 excluded.

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

## Stack version (#245)

These numbers were played on the book stack as it stood before #245 (2026-10-22), which
changed what the bidder counts as an off's guard (`bare-off`, `bidders-plan`; the double or
the top tile left in the suit, W42 p. 77). They reproduce at commit `64ecc6cf`. Re-runs on
the repaired stack are tracked on #245.
