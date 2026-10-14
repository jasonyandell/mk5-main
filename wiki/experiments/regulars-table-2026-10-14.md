---
title: The regulars' table — an opponent who remembers Zeb's bidding across hands (#149, 2026-10-14)
kind: experiment
first_seen: 2026-10-14
last_updated: 2026-10-14
status: complete
---

## Summary

Issue #149, the [[book-authority]]'s first measurement of assumption A1 ([[book-assumptions]]):
the book's style lessons (S-0080-b, S-0080-c, S-0081-a, S-0089-e, S-0156-b, S-0070-a, S-0189-a)
assume opponents who learn you across hands, and every arena seat forgets between hands. This
table seats the first opponent that remembers, in the channel the pilot showed opens within one
game: the auction.

**What our model can and cannot yet see.** A regular of this form, who learns each opposing
seat's bidding caution from the finished hands of one game and reads the next auction by it,
does not yet show the book's predictability lesson in marks:

- **Primary, the interaction** I = (Zv − Z | R) − (Zv − Z | R0): **+0.04 [−0.15, +0.22]
  marks/game**, 384 game pairs. The book predicts I > 0. A null at this power: the interval
  excludes an effect above +0.22, and the realized minimum detectable interaction is about 0.27
  (per-pair SD 1.86, larger than the 1.5 the sizing assumed).
- **A/A** I_AA (Z re-seeded in place of Zv): −0.02 [−0.20, +0.17]. The interval is calibrated.
- **The regular learns.** Its posterior on a deterministic Z seat ends a game with 0.88 of its
  mass on caution 0 (entropy 0.35 bits of a flat 2.32); on a varied Zv seat, 0.46 (0.72 bits).
  The memory sees the style. Reading it does not move marks at this power: S, the learning gain
  against deterministic Z, is +0.04 [−0.11, +0.20].
- **Varying is free at a memoryless table**, as the book predicts: C = −0.005 [−0.19, +0.18].
- **The game to 250** (192 pairs, secondary) agrees: I on games won +0.008 [−0.044, +0.060]
  (A/A +0.023 [−0.034, +0.081]); I on settled points +6.5 [−2.7, +15.5] per game.

Read under the charter: **"our cross-hand reader of this form cannot yet see the book's edge."**
It learns one number per seat (a fixed caution), uses it only to place tiles within the next
hand, and never responds: it does not bid differently against a learned overbidder (S-0080-c),
does not learn a seat's play, and forgets at the game's end. The book's regular does all of
those. The null bounds this channel (auction reading by a fixed-style learner, information only)
below about a quarter mark a game; it is not evidence that being modeled costs Zeb nothing.

## Design (pre-registered at `d432a216`, before any measured batch)

`scenario/regulars.py` holds the registration in its docstring; `arena/regulars.py` the model.

- **Players.** Every seat plays Zeb 0.3.0's plan (`k0,n160,F=<zeb/net>,f=16,d=posterior,
  r=uniform:2^22:0.28,t=margin`; [[zeb-030-confirmation-2026-10-14]]), deterministic per
  coordinate. Opponents bid by the heuristic rule at caution 0 in every arm.
- **Z**: target team bids `HeuristicBidder` at caution 0. **Zv**: caution drawn per seat per
  hand from {−5, 0, +5} (`VariedBidder`; ceilings move in steps of 5, so three behaviours),
  from the deal's own auction rng, so a hand's draw is identical across arms. **Za**: Z at
  planner seed 1 (A/A).
- **R0**: the soft reader plus the auction in its draw. Each target seat's bid or pass is
  evidence on its dealt hand under the heuristic rule with an unknown caution, flat on
  {−10, −5, 0, 5, 10}, trembling with probability 0.1, reset every hand. **R**: the same, with a
  posterior over a *fixed* caution per seat, updated by Bayes after each hand from the seat's
  revealed hand and action, kept for one game. R − R0 is exactly the cross-hand learning.
- **Refinements of the sketch** ([[book-assumptions]] § the ONE next experiment): the target
  plays Zeb 0.3.0, the deployed plan, instead of lowest-id ties; Zv varies its bidding only.
  The sketch's play mixing (a uniform pick within 0.005 of the best q) and the reader's learned
  play ε were left out. The pilot and #77 put the edge in the auction; the reader's memory
  models bidding only, so play variation could only add cost. Both remain open (below).
- **Estimator.** A game pair is (base seed, game index) under the arena's paired halves; each
  contrast is taken per pair over pairs every arm holds; 95% percentile bootstrap over game pairs
  (10,000 resamples, seed 42). Duplicate deals across all six arms.
- **Size.** Pilot (seeds 1490–1491, excluded): 8 pairs in 67–73 s, interaction SD 0.86 on 8
  pairs, sized at SD 1.5. Marks: seeds 1501–1532 × 12 pairs = 384 pairs per arm, six arms.
  250 points: seeds 1601–1616 × 12 = 192 pairs per arm, six arms. At most 8 batches at once,
  each under `timeout 300`; a nonzero exit drops a batch. **All 288 batches exited 0** (86–160
  s, load average 53–68 on 18 cores with other sessions); none dropped.
- **Bars.** I's interval excludes 0 above: the lesson is visible; below: varying costs more
  against this regular; else a null with its MDE.

## Results

| registered effect | game pairs | estimate |
|---|---|---|
| **primary I = (Zv.R − Z.R) − (Zv.R0 − Z.R0), marks/game** | 384 | **+0.04 [−0.15, +0.22]: null** |
| A/A I_AA = (Za.R − Z.R) − (Za.R0 − Z.R0) | 384 | −0.02 [−0.20, +0.17] |
| S, the regular's gain against Z = −(Z.R − Z.R0) | 384 | +0.04 [−0.11, +0.20] |
| C = Zv.R0 − Z.R0, varying at a memoryless table | 384 | −0.005 [−0.19, +0.18] |
| V = Zv.R − Z.R, varying against the regular | 384 | +0.03 [−0.18, +0.25] |
| 250: I on team A's share of games won | 192 | +0.008 [−0.044, +0.060] |
| 250: I on settled points/game | 192 | +6.5 [−2.7, +15.5] |
| 250: A/A I_AA on games won | 192 | +0.023 [−0.034, +0.081] |
| 250: S on games won (team B's gain) | 192 | +0.008 [−0.036, +0.052] |
| 250: C on games won | 192 | +0.023 [−0.023, +0.070] |

**Arm margins** (team A's marks/game; 0 is a reference, not a null, since team B adds the
auction read): Z vs R0 −0.19 [−0.34, −0.04], Z vs R −0.23 [−0.37, −0.08], Zv vs R0 −0.19
[−0.39, +0.00], Zv vs R −0.20 [−0.38, −0.02], Za vs R0 −0.05 [−0.20, +0.09], Za vs R −0.12
[−0.25, +0.03]. Games to 250 won by team A: 0.45–0.48. Contracts made bidding: 56–58% for the
target, 58–61% for the readers. *Exploratory, not registered:* the auction read itself (R0
against Zeb 0.3.0) looks worth about 0.05–0.19 marks/game here. Z and Za differ only in planner
seed and read 0.13 apart against R0, so that spread is the seed noise; a registered test of the
auction read alone (#117) would be needed to claim it.

**Mechanism.** The regular's final belief per target seat (768 seats per marks R arm):

| arm | mass on caution 0 | entropy (bits; flat 2.32) |
|---|---|---|
| Z.R | 0.88 | 0.35 |
| Za.R | 0.88 | 0.35 |
| Zv.R | 0.46 | 0.72 |
| PZ.R / PZv.R (to 250) | 0.84 / 0.45 | 0.47 / 0.79 |

The memory identifies a deterministic bidder within one game and is visibly confused by a
varied one: the learner works as built. What it does with that knowledge is too little for
marks at this power. A bid is read only as evidence on one hand's ceiling slot, and only to
place tiles in the play that follows. The 0.3.0 soft reader already sees those hands' plays.

## What the model leaves out (the scope of this null)

Held by this table: A1 partly (a memory within one game, of bidding only), A4 (the auction as a
signal from a known method). Still violated:

- **No response.** The regular never bids differently against a learned overbidder or
  underbidder (S-0080-c's "set the overbidder"), and never chooses its own play to exploit a
  style. The book's value of unpredictability is a response (A9); this table measures
  information only.
- **Fixed-style memory of one game.** A regular who tracks a mixture (the Dirichlet form) or who
  remembers across games (a reputation, S-0189-a) is not seated.
- **Play is unread across hands.** The reader's ε is the fitted 0.28 for every seat; it does not
  learn a seat's play, and Zv does not vary its play.
- **The style field is narrow** (A6): both teams bid the heuristic rule, so the styles a regular
  could learn differ by at most ±5 of caution.
- **Not read:** the declaration, the partner's bid, a forced shaker's open (rare).

The measurements that would restore these: a responding regular (bidding against the learned
caution; #116, #130 slice 5), play-ε learning with Zv's play mixing, a named deviant field with
wider styles (#152), and the auction read alone (#117).

## Build

- **The auction in the draw** (`Situation.auction`, `decide(state, bid, auction)` in both
  engines): per read seat, a table of 44 likelihoods over the slot of the hand the world says it
  was dealt (`ceiling_slot`: 0 with no pip suit of three, else 1 + the heuristic ceiling),
  multiplied into the soft reading's per-seat factor, on every draw path. Default off; requires
  a reading. The core's `ceiling_slot` is table-driven and equals the arena's `best_trump`
  ceiling on 120,000 random hands (20,000 of them in the test).
- **Per-game memory** in the arena (`arena/engine.py`): a policy with `remembers` gets each
  state's game key and `observe(game, hand)` after every hand. Every hand is played out, so each
  seat's dealt tiles are public by then.
- **The regular** (`arena/regulars.py`: `BidderBelief`, `RegularsPlay`) and **the varied
  bidder** (`arena/bidders.py`: `VariedBidder`).
- Tests: `scenario/tests/test_auction_read.py` (core slot = arena ceiling; referee = core to the
  bit with auction tables on the exact, enumerated, rejection and budget-out paths; a table of
  ones is no evidence; the draw follows the table; validation),
  `scenario/tests/test_regulars_table.py`, `arena/test_regulars.py`.
- Fence: every input is public when used: this hand's auction and the tiles of finished hands
  ([[strategy-fusion-fence]]).

## Data

Public in `jasonyandell/texas-42-zeb-research-2026-10` under `book/regulars-table/` (CC0,
`SHA256SUMS`, `NOTES.md`): `runs/<arm>_s<seed>/{meta.json,games.jsonl}` for all 288 batches,
`summary.json`, `logs/progress.log` (each batch's exit, wall time and load), `pilot/`, `run.py`.

```bash
timeout 300 python -u -m scenario.regulars arena --arm Zv.R --seed 1501 --games 12 --out runs/Zv.R_s1501
python -u -m scenario.regulars summarize --runs runs/* --out summary.json
```

## Links

[[book-authority]] · [[book-assumptions]] · [[reads-plays-soft-2026-10-14]] ·
[[zeb-030-confirmation-2026-10-14]] · [[book-reads-by-the-book-2026-10-14]] ·
[[exploiter-endgames-2026-10-14]] · [[scenario-player]] · [[strategy-fusion-fence]]
