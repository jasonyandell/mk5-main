---
title: The book's Ch. 9 at the table — doubles trump and no trump priced, provisional (#189, 2026-10-17)
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
---

## Summary

Issue #189, the measurement #155 built for. The book offers two more declarations to a hand
with four doubles or more (catalog rows W42-p075-b, W42-p079-a): doubles as trumps when a pip
trump would leave a high off (W42-p078-b), and no trump when the hand holds neither the 6-6 nor
a high off and no trump prices above every suit (W42-p080-a). `MenuBidder` follows those
conditions, and `MenuWalt` is the deployed Zeb seat that can play what it declares
([[book-assumptions]] A10). This study asks what the menu is worth in marks per hand to the
chair that holds such a hand, at a table where every seat plays alike.

**Every number on this page is provisional (random field for 7–9).** Deployed Zeb's field net
plays pip trumps only. Under doubles trump or no trump, every seat therefore plays as its
random-field twin (`DTNT=random`): in both arms, on both teams, the declarer included (the
Mayor's symmetry ruling on #155). The study is re-run with Lem's first field-v3 net (#185, G4,
[[lem-field-v3-audit-2026-10-16]]) once that net covers declarations 7–9 (#177).

## Results (provisional, random field for 7–9; complete 2026-10-17; 288 of 288 batches, none dropped)

**One line.** At a table where every seat plays doubles and no trump on the random field, the
menu costs the chair −0.045 [−0.065, −0.025] marks per hand on the 4,234 deals where it changes
the declaration. All of that is doubles trump: −0.106 [−0.142, −0.070] on its 1,560 deals. No
trump is level: −0.009 [−0.033, +0.015] on 2,674 deals. Read under [[book-authority]], this is
what our provisional player can and cannot carry. It plans a doubles hand against seats it models
as random legal play, while the book's declarer plays it by Ch. 9's lines. It is not a reading of
the book's declaration.

Data: [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
`book/ch9-provisional/` (per batch `meta.json` and ledgers, `summary.json`, the queue log, `CARD.md`).

| registered read | a − b [95%] | deals | SD/deal | MDE |
|---|---|---|---|---|
| **primary: marks per hand, every four-double deal** | **−0.008 [−0.012, −0.005]** | 23,040 | 0.29 | 0.005 |
| changed deals (the menu chose DT or NT) | −0.045 [−0.065, −0.025] | 4,234 | 0.66 | 0.029 |
| points per hand, every four-double deal | −0.68 [−0.84, −0.53] | 23,040 | 11.9 | 0.22 |
| points per hand, changed deals | −3.72 [−4.55, −2.87] | 4,234 | 27.6 | 1.19 |
| split: the menu declared doubles trump | −0.106 [−0.142, −0.070] | 1,560 | 0.73 | 0.052 |
| split: the menu declared no trump | −0.009 [−0.033, +0.015] | 2,674 | 0.62 | 0.034 |
| split: unchanged | 0 exactly | 18,806 | 0 | — |
| chance baseline: a − the random chair | +0.526 [+0.516, +0.537] | 23,040 | 0.79 | 0.015 |

The table played differently on 2,219 of the 4,234 changed deals. On the rest the chair did not win
the auction, so its declaration never came up. No unchanged deal read anything but 0 (the
determinism check).

### Reading, under [[book-authority]]

- **Assumptions.** See below. A10 is held only in the menu's *declarations*. Its *play* is the
  random field's for both teams, because the deployed net plays pip trumps only. The provisional
  label is exactly this.
- **What our model can see.** Given the choice, our chair declares doubles on the book's
  condition (a high off beside four doubles) and then plays the hand worse than it would play its
  best pip suit, by 0.11 marks a hand. The random field is a k0 planner that models every other
  seat as random legal play. Under doubles, where trump is the question (leading the boss trump,
  ruffing, keeping a sure trump), that is the weakest form a declarer can take. No trump, which
  asks less of a model of the other seats, comes out level.
- **What it cannot see.** Whether the book's doubles declaration pays when the hand is played the
  way Ch. 9 plays it. The pip alternative is played by the deployed net, so the comparison is
  between two declarations played by two different players. Only a field that plays doubles and
  no trump as well as pips can separate the declaration from the player.
- **The restoring measurement.** The re-run with Lem's first field-v3 net, which plays
  declarations 0–7 and 9, as the field for doubles and no trump. It is next, and it is registered
  separately. Lem's audit warns that the net is weakest on trump questions under doubles and on
  the book's own doubles and no-trump hands (regret 0.0163 on 2,110 rows), so even that re-run
  reads "field-v3 for 7 and 9", not "the book's opponent". A play-line price at the bid stays open.

Receipts: pre-registered at 519935de (`scenario/ch9.py`). 288 batches through `scripts/cpu-slot`,
four at a time, 113–147 s of compute each. Pilot seed 4890 excluded. Run 2026-10-17 00:58–04:31.

## Design (pre-registered before any measured batch; see `scenario/ch9.py`)

- **Runner.** The arena's ledger (`arena.ledger.run_ledger`), not the Book desk's game batches
  (`scenario.book_batches`). The question is asked per hand. The deals must be filtered to the
  chair's four doubles, about 4.3% of deals, and the ledger pairs both arms on the same deal and
  shaker with the same auction dice. Every hand is played from 0-0 in marks to 7 with the
  straight auction, so a hand nobody bids is passed out and moves nothing. The ledger gained a
  deal filter for this (`keep`). It keeps the first n deals of the usual stream that the filter
  accepts, and with no filter every existing ledger is byte-identical. A table offering the
  menu now writes all nine declarations in `offered`. Silas replays a menu ledger's
  doubles-trump and no-trump rows under its own rules (`silas/tests/ledger.rs`).
- **Players.** Every seat plays deployed Zeb 0.3.0 (160 worlds, seed 0) as `MenuWalt`. The chair
  is seat 0 at every shaker, and seats 1–3 bid by `HeuristicBidder`. **a**: the chair bids by
  `MenuBidder`, the book's full menu. **b**: the chair bids by `HeuristicBidder`, pip trumps
  only. The ledger's chance arm (a random bidder, lowest legal tile) is the baseline.
- **Deals.** Each batch's `arena.ledger.deals` stream, filtered to the deals where seat 0 holds
  four doubles or more.
- **Primary.** The paired mean of a − b in marks per hand for the chair's side, with a deal's
  value its mean over the four shakers. **The book predicts > 0.** The interval is the house's 95%
  percentile bootstrap over deals, which is calibrated ([[estimator-calibration-2026-10-16]]).
  The bar: above 0 means the menu pays at this table, below 0 means it costs, and otherwise the
  result is a null with its MDE.
- **Secondaries.** (1) The same on the deals where the menu's choice differs from the
  heuristic's (a different declaration, or a different bid below one mark), with the count. On
  every other deal the two chairs bid and declare alike, and the players are deterministic and
  batch-independent, so a − b is exactly 0 there. The summary checks this (`unchanged_nonzero`
  must be 0). (2) Points per hand, on all deals and on the changed deals. (3) The split by what
  the menu chose: doubles, no trump, or unchanged.
- **Mechanism, not tested.** The deals where the table differed (a's calls or contract differ
  from b's at some shaker), and the chair's side's doubles and no-trump contracts and the share
  made.
- **Size.** The timing pilot (seed 4890, 96 deals, excluded) took 170 s wall and 549 CPU-s
  (3.2 cores) under `cpu-slot --threads 4` at a machine load near 20. A census of all 51,108
  four-double hands puts the changed share at 18.3% (doubles 6.9%, no trump 11.4%). The design
  is 80 deals a batch over seeds 4901–5188: **288 batches and 23,040 deals**, about 2.8 hours at
  four batches at once. The MDE (80% power, two-sided 5%) is about **0.026 marks per hand on the
  roughly 4,200 changed deals**, 0.004 on the primary, 1.1 points per hand on the changed deals,
  and 0.03–0.04 within doubles and within no trump.
- **Batch-drop rule.** A batch that exits nonzero, its 300 s cap included, is dropped and
  reported, never rerun.

```
python -u -m scenario.ch9 queue --root scratch/ch9-2026-10-17 --workers 4
python -u -m scenario.ch9 summarize --runs scratch/ch9-2026-10-17/menu-s*/ --out ch9.json
```

## Assumptions (book-authority rule 1)

- **Held, provisionally.** A10, the full menu, in the book's conditions. Its play is the random
  field's, not a net's.
- **Violated.** A1 (memoryless seats), A2 (no partner reads a doubles or no-trump call by the
  book's conventions), A3 (an infallible field), A5 (one hand from 0-0 in marks, not the book's
  game to 250), A9.
- **Also outside the model.** The book prices a Ch. 9 hand by its play lines, and the arena
  prices it by its count. The arena puts Hand 23 at 27 where the book bids 31, so the menu's bids
  may sit below the book's. Walt's auction belief (`b=auction`) assumes pip declarations, but it
  is off in deployed Zeb.
- **Restoring measurements.** A field net covering decls 7–9 (#177; the first field-v3 net,
  #185), then this registration re-run. A play-line price at the bid (open on #189).

## Links

[[book-authority]] · [[book-assumptions]] · [[w42]] · [[scenario-player]] ·
[[estimator-calibration-2026-10-16]] · [[lem-pour-a-2026-10-16]] · [[lem-field-v3-audit-2026-10-16]]
