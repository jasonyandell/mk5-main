---
title: "The field-net audit: 384 poles of Zeb's net, labelled by Silas and by Haiku (Lem stage 1)"
kind: experiment
first_seen: 2026-10-15
last_updated: 2026-10-16
status: complete
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

Run 2026-10-16 on the full pre-registered sample: 434,535 rows, top set 4,345 per pole, 384 poles,
372 live. Nothing was dropped or re-rolled. Evidence (sample, labels, `audit.json`, `poles.md`,
cards, every reader output, `ablate.json`, runners, logs, `SHA256SUMS`):
[`lem/field-net-audit-2026-10-15`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/0982f55bd74a92d6819cf29e2fe7f309e998dab5/lem/field-net-audit-2026-10-15).

**The pole table.** 222 of 372 live poles are clean: **0.597 [0.581, 0.608]**, *provisional (#191)*: the
interval is the share with every precision at its Wilson lower, then upper, bound, and the Wilson
intervals are over rows, not deals (the split-half A/A below measures how wrong that is). The shuffled-label baseline is
**0.000 [0.000, 0.000]** over 20 shuffles. Partial 136, mixed 14. The clean labels' mean
precision is 0.980.

| | live | clean | by a Silas concept | by a raw feature | partial | mixed |
|---|---:|---:|---:|---:|---:|---:|
| layer 1 (128 neurons) | 256 | 162 | 89 | 73 | 89 | 5 |
| layer 2 (64 neurons) | 116 | 60 | 33 | 27 | 47 | 9 |
| all | 372 | 222 | 122 | 100 | 136 | 14 |

- **Dead poles: 12**, all `+` poles of layer 2: twelve of the 64 second-layer neurons never fire
  on any of the 434,535 rows.
- **The Silas labels** (poles per concept): lead 32, last 23, partner-winning 10, no-trump-held 10,
  trump-led 6, trumps-3plus 5, partner-out-of-trump 5, third 5, setter-out-of-trump 5,
  holds-master 4, late 3, holds-boss-trump 3, count-out-20plus 3, count-on-trick 3,
  winner-cannot-be-beaten 2, doubles-2plus 1, holds-ten 1, to-set-le-5 1. Where in the trick the
  viewer sits (lead, third, last) labels 60 of the 122.
- **26 of the 48 concepts label no pole**: second, trick-1, trick-2, trick-7, hand-le-3, forced,
  bidding-side, is-bidder, holds-trump, holds-sure-trump, trump-out, can-ruff, masters-2plus,
  holds-walker, holds-off, holds-boss-of-led, bidder-winning, safe-trick, ten-on-trick,
  holds-count, count-out, made, set, to-make-le-10, to-make-ge-20, holds-double. Several are
  runners-up (holds-sure-trump sits at precision 1.00 behind holds-boss-trump on L1+009).
- **The raw-feature labels are mostly the declaration.** 69 of the 100 come from the `count`
  block, which records where the ten- and five-count tiles sit after the trump pip is relabelled
  to 6, so it is the only trace of *which pip is trump*. 45 poles have a top set at least 90%
  one declaration (fives 17, fours 13, sixes 11, blanks 2, ones 2; 43 of them labelled by a
  `count` feature). "The trump is fives" is a concept the sealed list did not name, and the net
  spends about an eighth of its live poles on it. The sample's declarations are skewed by the
  pour's own bidder (fours 35.8%, fives 31.9%, sixes 10.6%, each other pip about 5%).

**The Haiku readers.** 372 poles explained from 12 cards each; 19 answered `none`; 349 scored
(4 answers could not be parsed). **Detected: 282 of 372 = 0.758**; mean detection accuracy
0.843. **Shifted null: 46 of 372 = 0.124**, mean 0.531 (chance 0.5). By the exact table's
kind: clean 179 of 213 detected (mean 0.861), partial 93 of 123 (0.820), mixed 10 of 13 (0.769);
67 poles scored 1.0 on all 20 cards. The readers' words are usually conjunctions where the exact
label is one predicate: L1+032's label is `last` and its description "partner is winning the trick
and I play last" (accuracy 1.0); L1+008's `no-trump-held` reads "holds no trump tiles at all"
(0.7); L1-008's `trumps-3plus` reads "leads holding six of the seven trumps" (0.9). Every
description sits beside its Silas label in the evidence.

**The neuron-count curve.** Greedy forward selection, 20,000 fit rows, 20,000 eval rows.

| neurons kept | 0 | 16 | 32 | 64 | 96 | 128 | 160 | 165 | 179 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eval fidelity | 0.389 | 0.497 | 0.562 | 0.636 | 0.702 | 0.775 | 0.928 | **0.952** | 1.000 |
| random k (mean of 5) | | 0.383 | 0.395 | 0.418 | 0.484 | 0.582 | | | |

The **knee is at k = 165 of 192**. Half the play is matched at k = 20, 0.9 at k = 155, and
13 neurons add nothing on the eval rows (the 12 dead ones and one more). The first run of the
curve was cut by the 1,200 s slot cap; the rerun scores candidates incrementally (exactly the
same statistic, held equal by test) and took three minutes.

**The predictions, graded.**

| | predicted | measured | |
|---|---|---|---|
| 1. clean share | 0.3–0.6 | 0.597 [0.581, 0.608]; chance 0.000 | inside, at the top |
| 2. layer 1 has more raw-feature labels | yes | 45% of clean poles in both layers | not supported |
| 3. Haiku detects / shifted null | 0.5–0.8 / about 0.1 | 0.758 / 0.124 | inside |
| 4. knee | k in 40–120 | k = 165 | wrong: later than predicted |

**Reading.** At the level of poles the net is largely clean: three poles in five have one main
job in the game's words or in one input feature, against a chance of zero, and a reader with only
the viewer's cards names most of those jobs. At the level of play it is not sparse: matching its
own move 95% of the time takes 165 of 192 neurons, and no small k stands out. For this net the
"clean neurons" claim of [[clean-neurons-42]] holds and the "small set of features" claim does not;
both are statements about one 192-neuron field net on one contract (seat 0 bids 30), and
[[lem-the-teacher]]'s stage 4 asks the same two questions of nets that grow. Three things the
list should learn before the next audit: a concept for the declaration itself (`trump-is q`), the
conjunctions the readers kept finding (position in the trick × who is winning × trump held), and
the partner's and setters' shown voids by suit.

**Runtime.** Labels about 21 ms a row (the book's definitions over 28 tiles), 55 chunks of 8,000
rows under `scripts/cpu-slot`, about 2.5 hours of wall on two shared slots; the pole table 172 s
(0.89 cores); the readers 47 minutes, about 1,100 `claude -p --model haiku` calls; the curve
3 minutes.

## Split-half A/A of the intervals (#191; registered 2026-10-16 before counting)

The Mayor's ruling on #191: the audit's Wilson intervals are over rows (positions), and a deal's
positions share the hand and the voids, so the headline interval is provisional until an A/A says
how often such an interval is wrong. The test makes no new claim about the net.

- **Design.** The sample's deals halved at random (seed 20261017, disjoint); the audit's procedure
  run on each half with its own top 1% (about 2,170 rows per pole), the pole's label held fixed to
  the audit's; dead poles excluded (372 pairs). Two intervals per half: the audit's **row Wilson**,
  and a **deal-clustered bootstrap** over the top set's deals (1,000 resamples, percentile 2.5/97.5,
  SE = the bootstrap sd; the Wilson SE = half-width / 1.96).
- **Counts** (Inez's request). The rate at which a half's 95% interval misses the *other* half's
  point estimate (calibrated: 0.166, since the other point carries its own noise); the rate of
  non-overlap (calibrated: 0.006); and z = (p_A − p_B) / sqrt(SE_A² + SE_B²), whose sd should be
  near 1 with |z| > 1.96 on 0.05 (the cleanest test). Also the clean share by half.
- **Belief probe** ([[lem-belief-probe-2026-10-16]]). The registered fit (half A) scored on half B,
  and the reverse, each R² with a deal-clustered bootstrap over its eval deals; the same z.
- **Predictions.** (1) Row Wilson is badly undercovered: |z| > 1.96 on more than 0.15 of poles,
  sd of z above 1.5. (2) The deal bootstrap is near calibration: |z| > 1.96 within [0.03, 0.09].
  (3) The two halves' clean shares differ by less than 0.03. (4) The belief R² story
  (x ≫ a1 > a2) holds in both directions with non-overlapping bands between representations.
- Code: `lem/probe/aa.py` (CC0). Cost: one slot call, CPU minutes.

**Results.** Not yet run.

## Exploratory, not pre-registered (2026-10-16)

Read-only looks taken after the sealed study, for the Lem special (Media) and the question
"is there evidence of recursive generalization in Zeb already?" None was predicted in advance;
each is a read of existing data, not a test. Files:
[`series/`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/d2fa52c3be5074e6efca6330975117ef8f96c95d/lem/field-net-audit-2026-10-15/series)
and `audit/reads.json`, `audit/reads.md` in the evidence folder.

**The data series.** The pour repo publishes three nets of this architecture; the same sample and
labels read against each. One net per size, so the trend itself carries no interval.

| net (deals) | live | clean share | by Silas | by raw feature (count block) | mixed | dead | knee |
|---|---:|---|---:|---:|---:|---:|---:|
| 5,000 (`field_make_d5000_v2`, older trainer, different pour and target) | 384 | 0.297 [0.273, 0.333] | 30 | 84 (68) | 35 | 0 | 176 |
| 50,000 (`margin_d50k_h128x64`) | 378 | 0.519 [0.505, 0.553] | 104 | 92 (71) | 20 | 6 | 170 |
| 250,000 (the shipped net) | 372 | 0.597 [0.581, 0.608] | 122 | 100 (69) | 14 | 12 | 165 |

The shuffled chance is 0.000 at every size. The 50k and 250k nets share the trainer, the target
and a pour (the 50k's is one of the 250k's five chunks), so that pair is the clean comparison.
Reading: with more data the nameable concepts rise and the mixing falls; the declaration
detectors (the count-block labels) are there from the start and stay flat; the play stays dense
(the knee moves 176, 170, 165 of 192). This is the direction the recursive-generalization
hypothesis predicts at the level of one snapshot per size ([[recursive-generalization]]); it is not
checkpoint evidence and shows no branching.

**The trump ladder, at the final net** (`audit/reads.md`). Every trump-labelled pole carries the
rungs together: on the three `holds-boss-trump` poles (L1+009, L1+108, L1+118) the precision of
`holds-sure-trump`, `holds-master` and `holds-trump` is 1.00 each; on the four `holds-master` poles
`holds-trump` is 0.88 to 1.00. No live pole has `holds-master` ≥ 0.9 together with `no-trump-held`
≥ 0.5: the nearest is L1-040 (0.70 and 0.28). In this net "master" is "master of trump"; "master of
a side suit" has no unit. Consistent with a concept formed around trump that has not extended, and
equally with a facet the net never needed; the sample's base rates are `holds-boss-trump` 0.257,
`holds-master` 0.314, `no-trump-held` 0.437.

**The structural poles are conjunctions.** For each of the 60 clean `lead`/`last`/`third` poles,
the second predicate of highest lift on its top set: three `lead` poles carry `no-trump-held` at
precision 1.00 (L1+080, L2-003, L2-027; base 0.437), `lead` poles carry `set` at 0.82 to 0.91
(L2+022, L1+057, L2-006; base 0.19), `last` poles carry `safe-trick` 1.00 (L1+032; base 0.10) or
`winner-cannot-be-beaten` 0.68 to 0.70 (L1-064, L1-067, L1-122; base 0.25). The full table is
`audit/reads.md`. These are the conjunctions #185 adds to the list.

## Links

[[lem-the-teacher]] · [[clean-neurons-42]] · [[recursive-generalization]] · [[scenario-player]] · [[silas]] ·
[[strategy-fusion-fence]] · [[gus-probe]]
