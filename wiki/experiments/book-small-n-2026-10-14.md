---
title: The book at small n — do Winning 42's rules raise a bounded Zeb? (#154, 2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-22
status: complete
---

## Summary

Issue #154, the [[book-authority]]'s measurement of assumption A7 ([[book-assumptions]]): the
book is a policy for a reader who cannot enumerate 512 worlds a decision. Every arena block has
compared it against deployed Zeb at full compute; its one measured win is as a planner's
near-tie break in the book's own game ([[book-investigation-2026-10-14]] § II). Here the book
is handed to a Zeb that samples 16, 64 or 160 worlds, and the question is whether the rules
raise it, and whether they raise it more the fewer worlds it sees.

## Results (complete, 2026-10-14; 280 of 280 registered batches ran, none dropped)

**One line.** At 16 worlds the book as Zeb's near-tie break neither raises nor costs it: D16 =
−0.01 [−0.18, +0.17] marks/game over 288 game pairs, a null whose minimum detectable effect is 0.25
(the A/A reads −0.03 [−0.18, +0.11]). The prediction D16 > 0 is not seen at this size, and neither is
the bar's negative. The one signed finding is the filter's trend: Zeb's best tile confined to a firing
rule's set costs 1.10 [0.93, 1.27] marks/game against the same Zeb unconfined at 160 worlds, and
0.21 [−0.02, 0.44] at 16, a difference of 0.89 [0.67, 1.13] in the direction A7 predicts. The fewer
worlds the planner sees, the less the book's sets cost it; within this range they never raise it.

Data: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
`book/small-n/` (per batch `meta.json` and `games.jsonl`, `summary.json`, the queue log, `CARD.md`).

### The arms (marks per game against deployed Zeb 0.3.0, 576 games each; made = share of contracts made, team A / team B)

| arm | team A | marks/game [95%] | made A / B |
|---|---|---|---|
| Z16 | Zeb at 16 worlds | −1.08 [−1.25, −0.91] | 51.8% / 61.7% |
| T16 | Z16, the book at its near-ties | −1.09 [−1.28, −0.90] | 51.9% / 62.0% |
| F16 | Z16 inside the book's sets | −1.29 [−1.47, −1.10] | 52.4% / 64.4% |
| Z64 | Zeb at 64 worlds | −0.35 [−0.51, −0.19] | 56.1% / 59.3% |
| T64 | Z64, the book at its near-ties | −0.35 [−0.52, −0.19] | 56.7% / 59.9% |
| Z160 | deployed Zeb (the same player both sides) | 0 exactly | 58.9% / 58.9% |
| T160 | Z160, the book at its near-ties | +0.04 [−0.09, +0.17] | 59.1% / 58.7% |
| F160 | Z160 inside the book's sets | −1.10 [−1.27, −0.93] | 53.7% / 63.8% |
| A160 | Z160 at planner seed 1 (the A/A) | −0.03 [−0.18, +0.11] | 58.5% / 58.8% |

### The registered contrasts (paired by game pair; MDE = minimum detectable effect at 80% power)

| contrast | marks/game [95%] | pairs | SD/pair | MDE |
|---|---|---|---|---|
| **primary D16 = T16 − Z16** | **−0.01 [−0.18, +0.17]** | 288 | 1.51 | 0.25 |
| A/A = A160 − Z160 | −0.03 [−0.18, +0.11] | 288 | 1.22 | 0.20 |
| D64 = T64 − Z64 | −0.00 [−0.17, +0.16] | 288 | 1.44 | 0.24 |
| D160 = T160 − Z160 | +0.04 [−0.09, +0.17] | 288 | 1.09 | 0.18 |
| trend D16 − D160 | −0.05 [−0.25, +0.15] | 288 | 1.74 | 0.29 |
| trend D64 − D160 | −0.05 [−0.24, +0.15] | 288 | 1.71 | 0.28 |
| filter E16 = F16 − Z16 | −0.21 [−0.44, +0.02] | 288 | 1.99 | 0.33 |
| filter E160 = F160 − Z160 | −1.10 [−1.27, −0.93] | 288 | 1.51 | 0.25 |
| filter trend E16 − E160 | **+0.89 [+0.67, +1.13]** | 288 | 2.00 | 0.33 |
| bounded compute Z16 − Z160 | −1.08 [−1.25, −0.91] | 288 | 1.50 | 0.25 |
| bounded compute Z64 − Z160 | −0.35 [−0.51, −0.19] | 288 | 1.43 | 0.24 |
| does the book repay it? T16 − Z160 | −1.09 [−1.28, −0.90] | 288 | 1.68 | 0.28 |

The book's game, points to 250 (192 game pairs per arm; team A's share of games won, and settled
points per game):

| contrast | games won | settled points/game |
|---|---|---|
| D16: PT16 − PZ16 | +0.8 pts [−3.1, +4.7] | +4.8 [−2.7, +12.5] |
| D160: PT160 − PZ160 | +2.1 pts [−0.8, +5.2] | +4.2 [−1.6, +10.2] |
| PT16 − PZ160 | −14.8 pts [−18.8, −10.9] | |

PT160 wins 52.1% [49.2, 55.2] of games to 250 against deployed Zeb, consistent with the
investigation's replicated 53.2% [51.1, 55.2] ([[book-investigation-2026-10-14]] § II) at a third of
its size; PT16 wins 35.2% [31.3, 39.1], PZ16 34.4% [30.7, 38.0].

### Reading, under [[book-authority]]

- **Assumptions held and violated.** This arm restores A7 only in part: team A still enumerates,
  at 16 worlds through the same planner, net and posterior sampler as deployed Zeb. A1, A2, A3, A4,
  A6 and A10 are violated as in every arena block. The marks game violates A5; the 250-point arms
  restore it.
- **What our model can see.** The near-tie break is as cheap at 16 worlds as at 160: D16, D64
  and D160 all cover 0 with MDEs near 0.25, and the A/A covers 0. Bounded compute itself costs
  1.08 [0.91, 1.25] marks/game at 16 worlds and 0.35 [0.19, 0.51] at 64, on both sides of the
  contract (at 16 worlds team A makes 51.8% of its contracts and lets the 160-world opponent make
  61.7%). The book's rules, as a near-tie break, neither add to that cost nor repay it: T16 − Z160
  equals Z16 − Z160.
- **The signed finding and its shape.** Confining a 160-world plan to the book's sets costs it
  1.10 marks/game; confining a 16-world plan costs 0.21 [−0.02, 0.44]. The sets cost the bounded
  player on defense, not on its own contracts: F16's team A makes 52.4% of its contracts against
  Z16's 51.8%, while the opponent's made share rises from 61.7% to 64.4%. At 160 worlds the sets
  cost on both sides (53.7% made, 63.8% conceded). The book's play rules are nearly as good a
  declarer as a 16-world planner, and a worse setter than any planner here.
- **What it cannot yet see.** The reader the book writes for has no planner at all. This design
  stops at 16 worlds, where the fallback inside and outside a rule's set is still a plan over
  sampled worlds. The point the trend extrapolates to, n = 0, is the one the head-to-head's
  variant (a) came nearest ([[book-head-to-head-2026-10-13]]: the book over a lowest-legal fallback,
  against deployed Zeb), and no arm has yet paired that fallback with and without the rules.
- **The measurement that restores A7.** The same design at n = 0: a lowest-legal-tile (or
  random-legal) fallback, alone and under the book's rules, against deployed Zeb and against each
  other, both games. If the rules raise that reader, the gain the book claims for its reader is
  measured where the reader lives; the arm is cheap and is filed as #161. Done: [[no-plan-2026-10-14]], +2.47 [2.28, 2.65] marks/game.

### Receipts

Pre-registered at bf16fdfe (`scenario/small_n.py`, this page's design). 280 batches through
`scripts/cpu-slot`, two at a time, 49–70 s a marks batch and 18–30 s a points batch at load
average 32–42; none dropped. Timing pilot seed 1690 excluded. Estimator and seeds as registered;
`summary.json` carries every number above.

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

## Stack version (#245)

These numbers were played on the book stack as it stood before #245 (2026-10-22), which
changed what the bidder counts as an off's guard (`bare-off`, `bidders-plan`; the double or
the top tile left in the suit, W42 p. 77). They reproduce at commit `64ecc6cf`. Re-runs on
the repaired stack are tracked on #245.
