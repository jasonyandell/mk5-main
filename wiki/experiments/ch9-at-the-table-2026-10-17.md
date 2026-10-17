---
title: The book's Ch. 9 at the table — doubles trump and no trump priced, provisional (#189, 2026-10-17)
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
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

*Results follow once every registered batch has run.*

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
