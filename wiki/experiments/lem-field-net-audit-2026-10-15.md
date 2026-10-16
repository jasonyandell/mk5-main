---
title: "The field-net audit: 384 poles of Zeb's net, labelled by Silas and by Haiku (Lem stage 1)"
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-15
status: active
---

Stage 1 of [[lem-the-teacher]] (#176); the idea is [[clean-neurons-42]]'s experiment A1.
Pre-registered here before any pole of the study sample was scored. The probe kit is
`lem/probe/` (CC0); its tests are `lem/tests/test_probe.py`.

## Question

Does the net Zeb ships already have neurons with one job each, and are those jobs the game's
own words? For every pole of `zeb/net/margin_d250k_h128x64.fnet`: what do its top-activating
positions share, said exactly in [[silas]], and said cheaply by a Haiku reader? And how many
of its 192 neurons does its play need?

## Pre-registration

**The net.** `zeb/net/margin_d250k_h128x64.fnet`, the CC0 field net behind `@jasonyandell/zeb`
(hidden 128 and 64; 343 input features in the `field-v2` layout; integer inference). A neuron
has two poles, `+` (its pre-activation highest) and `−` (lowest); 384 poles. The probe's
forward pass is the referee's own arithmetic (`lem/probe/net.py`; held bit-equal to
`FieldNet.scores` by test).

**The sample.** Every position the net did not train on: the rows with `deal % 10 == 0` of the
five pours that trained it (`scratch/atlas/pour_make_netfield_s100001_d50000` and
`scratch/atlas/zeb200k/pour_s{200001,250001,300001,350001}_d50000`; the public copy is
[texas-42-zeb-pour-250k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k)
`raw/`), the trainer's own validation split: about 434,500 positions. Every row carries its
feature vector, its legal moves, Walt's choice and the frame (declaration, viewer, hand,
bidder, bid, plays). Note for every reader of the table: the pours' contract is always seat 0
bidding 30 on a pip trump, so the `bid` one-hot is constant and the `need`/`slack`
thermometers move only with play.

**The concepts.** 48 yes/no Silas queries over the viewer's frame, written in the book's
vocabulary (`silas/lib/book.silas`), in `lem/probe/concepts.py`; the list's sha256 is
`13e4d525ff9c45b96c337364d0abb95c1b8704a493116501c8a4f0e6b5ed4fe3` and the audit refuses labels
made with any other. Plus, as the trivial baseline, the net's 343 raw input features, each a
concept of its own (`x:<block>:<i>`). Every query is evaluated on a belief that resolves no
hidden tile, so a query reading a hidden hand is refused by Silas ("needs the unresolved
tiles"); the test holds that refusal. **Fence:** no label reads a hand the viewer cannot see;
no clairvoyant value appears anywhere ([[strategy-fusion-fence]]).

**The statistics** (`lem/probe/audit.py`; constants fixed here).
- Top set: the top 1% of rows by the pole's signal (about 4,345 rows), ties by row order.
- Precision of a concept on a pole: its share of the top set; base rate: its share of all
  rows; lift: the ratio. The pole's **label** is the concept of highest precision among those
  with base rate ≤ 0.5. Secondary label: the concept of highest |point-biserial r| over all rows.
- **Clean**: label precision ≥ 0.9. **Partial**: ≥ 0.7. **Mixed**: below. **Dead**: the neuron
  is on (`+`) or off (`−`) in fewer than 0.1% of rows; dead poles are reported and excluded
  from the shares.
- Intervals: Wilson 95% on each precision over the top set; the clean share's interval is the
  share with every precision at its lower, then its upper, bound.
- **Chance**: every concept's column permuted across rows (20 fixed-seed shuffles, seed
  20261015), the whole procedure re-run; the shuffled clean share is the baseline. With 391
  concepts of base rate ≤ 0.5 and a 4,345-row top set it should sit near 0.
- Reported: the clean share (by a Silas concept and by a raw feature, separately), by layer;
  the full pole table (`poles.md`); the concepts that label no pole; the concepts that label
  many.

**The Haiku readers** (`lem/probe/readers.py`; `claude -p --model haiku`, one fresh call per
pass, nothing in context but the cards). Per live pole: *explain* from 12 top-set cards (≤ 15
words, or `none`); *score* 20 held-out cards, 10 from the top set (ranks 13–22) and 10 drawn at
random from all rows, shuffled, Y/N each. Detection accuracy = mean of the hit rate on top cards
and the miss rate on random cards; chance 0.5; Wilson interval over the 20 marks. A pole is
**detected** when the lower bound exceeds 0.5. Second null, **shifted**: the same cards scored
against the description of the pole half a table away. Cards show the viewer's information
only (`lem/probe/cards.py`). Agreement with the exact label is read, not scored: the reader's
words beside the Silas label in the table.

**The neuron-count curve, first form** (`lem/probe/ablate.py`). Fidelity of a reduced net
(a set of hidden neurons kept, the rest zeroed after their ReLU) = share of rows on which it
picks the whole net's tile. Greedy forward selection on 20,000 fit rows, reported on 20,000
disjoint eval rows (seed 20261015); random-k baseline (five draws per k). **Knee** = smallest k
with eval fidelity ≥ 0.95.

**Predictions, written before measuring** (the Lem desk's priors; no stake, a calibration record).
1. Clean share of live poles: between 0.3 and 0.6. Cammarata's regime would be ≥ 0.9; the
   shuffled baseline about 0.
2. Layer 1 has a larger share of raw-feature labels than layer 2.
3. Haiku detects between 0.5 and 0.8 of live poles; the shifted null detects about 0.1.
4. The knee is at k between 40 and 120 of 192 (no small knee).

**Runs.** Sample builds and the concept pass go through `scripts/cpu-slot` (one pour per call,
under 300 s each; the concept pass in row chunks). The readers are about 1,200 CLI calls. The
smoke pour (`scratch/atlas/pour_smoke`, 139 rows) exercised every command before sealing; its
numbers are not the study. Evidence (sample manifest, labels identity, `audit.json`,
`poles.md`, cards, reader outputs, `ablate.json`, `SHA256SUMS`) goes to
`jasonyandell/texas-42-zeb-research-2026-10` under `lem/field-net-audit-2026-10-15/`.

## Results

Not yet run.

## Links

[[lem-the-teacher]] · [[clean-neurons-42]] · [[scenario-player]] · [[silas]] ·
[[strategy-fusion-fence]] · [[gus-probe]]
