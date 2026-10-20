---
title: "Unit paper 1: the 16×8 seed-0 net's unit 5, registered as \"partner has it\", read raw as \"I play last, and how good the trick is for my side\" (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

The first **unit paper** of [[clean-neurons-42]]'s new mode (#235): one unit of one net, read the
way the essay reads a neuron. It asks two things: what does this unit do, and how did it come to
do it? The net is the 16×8 seed-0 field net (`dropout-2026-10-17/nets/field_h16x8_s0.npz`,
bit-identical to the last of its 15 formation checkpoints). The series stays on this one net so
that each paper's circuit can lean on the ones before it.

**Why this unit and not the `late` unit #235 named.** The handoff asked for the seed-0 net's
"coarse `late` (third ∪ last) unit". That unit is not in seed 0; third ∪ last units are in seeds 1
(unit 12) and 2 (units 7 and 13). There is also a naming error. List v1's `late` concept means "the
fifth trick or later" (`concepts.py`), not "third or last to play". So
[[lem-unit-origin-2026-10-20]]'s line that (`third`, `last`) "is named by list v1's own coarse
`late`" is wrong, and so is its use to fail those units on part 1. A correction goes on that page.
Seed 0's seat-family unit is unit 5. #231's conditioned read called it `last` ∪ `second`. The raw
read with the play-count baseline ([[lem-fixed-conditioning-2026-10-20]], #233) names it otherwise:

| pole | on | label | precision | base | runner-up | top r |
|---|---:|---|---|---:|---|---|
| L1+005 | 0.200 | `partner-winning` | 1.000 [0.999, 1.000] | 0.199 | `partner-winning` (1.00) | `x:open:3` +0.75 |
| L1−005 | 0.800 | `x:winning:2` | 1.000 [0.999, 1.000] | 0.337 | `setter-void-off-suit` (0.71) | `x:open:3` −0.75 |

`x:winning:2` is the input bit for "the tile taking the trick so far was played by the seat just
before me" (my right-hand opponent). `partner-winning` already implies that I play third or last.
So the reading to test is a gate: **it opens when my partner has the trick, and it is held shut
hardest when the opponent just before me has it.** It fires on 0.200 of rows against a base rate of
0.199, so the unit is on almost exactly when its concept holds.

Scope: this net, these inputs (one position, uniform beliefs, no history), this vocabulary (list
v2 plus the raw inputs). What we learned from this model, not all of 42, yet.

**What this paper is for.** Unit 5 is close to one input bit. It fires on 0.200 of rows against
a base rate of 0.199, at precision 1.000, and it was picked after its label was known. That makes it
a **calibration unit for the loop**, not a test of the clean-neuron thesis. The essay tests the
thesis on the messiest units, by asking why their things share a neuron. Here we check that each
instrument of the loop reads a unit we can already name: the seat-by-winner table, the circuit,
formation, steering by where play goes, an input flip, and explanations that can lose. Paper 2
takes the messiest pole in this net under read R, which is L1+013 (`void-in-led`, 0.766, partial).

**Two correlates, named before the run.**
- `x:open:3` (I play last) has r = ±0.75 on both poles. Partner can be winning only at third or
  last seat. At second seat the winner is always the leader, who is my right-hand opponent.
- So `x:winning:2` holds on every second-seat row. Its 1.000 precision on the − pole may just mean
  "I play second". The − pole is therefore also read on third- and last-seat rows only.

## The essay's loop, as registered

All reads use the held-out audit sample (`audit-2026-10-15/sample.npz`, 434,535 rows, held-out
deals only) and list v2. Activations are raw pre-activations z₅, never z-scored. Labels use the
audit's labeller under read R (base rate ≤ 0.5, and the label must beat the play-count mix by 0.1).
The code is `python -u -m lem.probe unit` (`lem/probe/unit.py`), smoke-tested on another net
(16×8 seed 1, unit 12) before this registration was sealed.

1. **Label, both poles, as a gate.**
   - The mean z₅ and the on-rate in each possible cell of seat × who is winning, with row counts.
     The cells are: second seat with RHO winning; third seat with partner or RHO winning; last seat
     with LHO, partner or RHO winning.
   - The + pole on the top 1% of all rows. The − pole on the top 1% of all rows, and again on the
     bottom 1% of third- and last-seat rows.
   - **Raw shape.** The deciles of z₅ on rows where partner is winning and on all other rows. The
     overlap is the share of other rows above the 5th percentile of the partner-winning rows.
   - **Dial or detector.** Within partner-winning rows, compare mean z₅:
     - third seat with partner's tile safe (`winner-cannot-be-beaten`) against unsafe;
     - holding count against not;
     - the spread of the means across the seven pip declarations.
   - **The untrained twin** (`dropout-2026-10-17/floor/field_h16x8_s0.npz`). Over its 48 poles, the
     best top-1% precision for `partner-winning` and for `x:winning:2`, both raw and as read R grants
     it.
2. **The circuit in.** On each pole's top 1%, the ten inputs with the largest mean vote
   w₁[f, 5]·x̄_f, beside each one's mean vote on all rows. Read off; no prediction rests on it.
3. **The circuit out.**
   - The layer-2 readers by |w₂[i, 5]|, with signs and their audited labels.
   - The **push**: the change in each legal tile's score from adding σ to a₁[5], on the rows where
     the unit is off and I follow. It is grouped by tile class: count or not, trump or not, beats the
     current winner or not.
4. **Formation.** For each of the 15 checkpoints (steps 0 to 916,200), on the same rows:
   - the + pole's top-1% precision for `partner-winning`;
   - each pole's read-R label;
   - the seat mix of the + pole's top set (third against last).
5. **Steering, read by where play goes (the measure this paper defines).**
   - **Rows.** I follow to an open trick and hold more than one legal tile, and the choice matters:
     some legal tiles beat the current winner and some do not, or some carry count and some do not.
   - **Up.** Add k·σ to a₁[5] for k ∈ {0.25, 0.5, 1, 2}, where σ is the unit's standard deviation
     where it fires. This is scored on the rows where the unit is off.
   - **Down.** Set a₁[5] to 0, scored on the rows where it is on.
   - **Where play goes.** Two rates, each steered minus base:
     - **take**: the pick beats the trick's current winner, by the declaration's trick rank under
       the suit led;
     - **smear**: the pick puts count (a 5 or a 10) on the trick.

     Each is reported with the move-change rate, by seat and pooled, with deal-clustered bootstrap
     95% intervals (1,000 resamples).
   - **The null.** Every other live layer-1 unit (15 of them) is steered on the same rows at the
     same k in its own σ, or ablated for the down arm. A shift counts only if it lies beyond all 15
     in the predicted direction.
   - **The input flip.** On the off rows at third and last seat where an opponent is winning, set
     the input to "partner is winning" (`x:winning:1` on, the others off) and read take and smear
     the same way. Report the unit's own a₁ before and after the flip, and the unit's k = 1 steering
     on the same rows.
   - **Reading.** Eight steered picks are shown in full.
6. **History (#202).** Compare the + pole's top 1% against partner-winning rows below the unit's
   median. Compare their trick number, need and slack, bid, and share at last seat. Read off, not
   graded.

## Predictions, sealed before any of 1–6 is computed on this unit

1. **A three-way gate at last seat (label).**
   - At last seat the mean z₅ orders partner > LHO > RHO.
   - At third seat partner > RHO.
   - Second seat (RHO winning, by force) sits below every partner cell.
2. **The − pole is not just "I play second".** On third- and last-seat rows the − pole's top 1%
   still has `x:winning:2` at precision ≥ 0.9.
3. **A detector, not a dial.** Within partner-winning rows, each of these is under 0.1 of the
   unit's maximum z₅:
   - the safe-against-unsafe gap at third seat;
   - the holds-count gap;
   - the spread across declarations.

   The overlap (raw shape) is under 0.01.
4. **The circuit out.**
   - Unit 5's two largest-|w₂| readers are layer-2 units 4 and 5, whose poles the audit labels
     `partner-winning`, with w₂[4, 5] > 0 and w₂[5, 5] < 0.
   - The push favours count: push(count) − push(no count) > 0.
   - The push disfavours taking: push(beats) − push(does not beat) < 0.
5. **Up means "leave it to partner".** At k = 1 on the off rows, pooled:
   - Δtake < 0 and Δsmear > 0, each interval excluding 0;
   - each shift lies beyond all 15 other units in that direction.
6. **Down means "go get it".** Ablating on the on rows gives Δtake > 0 and Δsmear < 0, intervals
   excluding 0, each beyond all 15 other units.
7. **The unit is one carrier of several.** The input flip moves take and smear the same ways as
   unit 5's k = 1 steering on the same rows. The flip's shifts are larger, because other units also
   read the `winning` bits: |flip| > |unit| on both rates.
8. **The twin is blind.** No twin pole reaches 0.9 for `partner-winning` under read R.
9. **It grew locally, from last seat (formation).**
   - The + pole reaches ≥ 0.9 for `partner-winning` by step 91,620 (10% of training) and stays
     there at every later checkpoint.
   - From step 1,000 on, every checkpoint's + label is in the trick-state set: the position and
     trick families, `can-take-it`, the v2 seat × winner conjunctions, and the raw `x:open:*`,
     `x:winning:*` and `x:trick:*` inputs. There is no below-0.7 escape.
   - At the first checkpoint where the pole is clean, last seat's share of its top set exceeds
     last seat's share of all partner-winning rows. The unit came to partner-winning through last
     seat first.

**What would break the explanation.** Predictions 5 or 6 failing in the opposite direction, or
inside the null, would mean the name says where the unit fires but not what it does. Prediction 7
failing would mean the unit and its input point different ways. Prediction 9 failing with an early
label from an unrelated family would be a lead for the next rung of the ladder (the data, then
recursive generalization), not a reading of mixing.

**Cost.** One net, 15 checkpoints, about 70 forward passes over 434,535 rows. One `scripts/cpu-slot`
run.

## The review before sealing

One fresh-context Opus read the essay's full text (to line 943), then the first draft. Its points,
and what changed:
- **Not a thesis test.** The unit is close to one bit and was chosen after its label was known, so
  it is a positive control for the loop. → Said so above. Paper 2 takes the messiest pole.
- **Prediction 1 was ill-posed.** At third seat LHO has not played. → Made the three-way order a
  last-seat prediction.
- **The − label may only mean "I play second".** → Re-read on third and last seat (prediction 2).
- **`x:open:3` was ignored.** → Named above.
- **Steering may only replay the input.** → Added the input flip (prediction 7).
- **Circuit out could not lose.** → Registered the readers and the push (prediction 4).
- **Raw shape and dial or detector were missing.** The essay asks for raw shape, not percentiles.
  → Prediction 3.
- **Prediction 6 had an escape, and its family was wide.** → Dropped "or below 0.7"; added a
  growth-order prediction (9).
- **Prediction 4 was mechanical** (dose order). → Dropped. Doses are reported, not graded.
- **Prediction 7 followed from the label** (top votes). → Dropped. The votes are read off.
- **Five random units were a thin null.** → All 15.
- **Read steering qualitatively too.** → Eight picks shown in full.
- **Not adopted: a synthetic blind maximum.** Building positions from the explanation alone needs
  consistent deals. The seat × winner table and the dial or detector gaps stand in for near misses:
  partner has played but LHO is winning at last seat, and distant concepts (declaration, count held)
  that should not move it. A true blind maximum is a tool for a later paper.

Sealed 2026-10-20.

## Results

Run 2026-10-20 as registered, in one `scripts/cpu-slot` run of 135 s (2.96e12 instructions). The
evidence (`unit.json`, the log, the code, `SHA256SUMS`) is `lem/unit-partner-has-it-2026-10-20/`
in [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/e597dc9284afd27e747d914648fcc8ad9303807b/lem/unit-partner-has-it-2026-10-20).

**The picture.** Read raw, unit 5 is not a partner-winning detector. It is an **"I play last"** unit
that turns up the louder the trick is for my side. The top 1% saw only the peak of that dial.

**The gate, raw (z₅; the unit's maximum is 38,542).**

| seat | who is winning | rows | mean z₅ | on |
|---|---|---:|---:|---:|
| lead | — | 150,000 | −8,285 | 0.000 |
| second | RHO | 94,926 | −11,004 | 0.027 |
| third | partner | 65,249 | −11,201 | 0.002 |
| third | RHO | 30,162 | **−30,989** | 0.000 |
| last | LHO | 51,579 | 6,268 | 0.915 |
| last | partner | 21,110 | **21,407** | 0.999 |
| last | RHO | 21,509 | 3,097 | 0.738 |

- **It opens at last seat**, for every winner. It is shut at third seat even when partner has the
  trick: 75.6% of partner-winning rows are at third seat, and there it fires on 0.2% of them.
- **Within last seat it is a dial**: partner > LHO > RHO.
- **Its deepest closure is third seat with RHO winning.** That is the negative pole's distinction:
  an opponent on top, with one more opponent still to play.
- **It fires on 0.200 of rows, close to `partner-winning`'s 0.199, by coincidence.** Last seat is
  0.217 of rows, and the unit is off on a quarter of the last-seat rows where RHO is winning.
- **Why the label said `partner-winning`.** `last` is precise at 1.000 on the top set too. But read
  R's play-count baseline explains `last` by construction (seat is the play count mod 4), so it can
  never be granted. The label went to the dial's peak instead. This is a tools-rung finding, filed
  as **#236**. It affects read R's *names* for position units; the clean shares are probably little
  changed, but that is not measured.

**Circuit in.** The + pole's largest votes are `x:winning:1` (+18,524) and `x:open:3` (+4,763). The
− pole's are `x:open:2` (−29,065, third seat) and `x:winning:2` (−5,684). The unit is built from
"I play last" and "partner has it", and shut by "I play third" and "RHO has it". The `need` bits
follow, small.

**Circuit out.** The readers by |w₂| are layer-2 units 5 (−668; its poles read `count-on-trick` and
`partner-winning`), 7 (−413), 3 (−271) and 1 (+237). The push of +σ on legal tiles, in raw score
units: count +864k against no count −497k; beats the winner −2.75M against does not beat +486k;
trump −2.77M against not trump +541k. **Turning it up says "don't take it, put count on it".**

**Steering, read by where play goes** (k = 1, 123,807 off rows, base take 0.394 and smear 0.324):

| arm | Δtake [95%] | Δsmear [95%] | moves changed | the 15 other units |
|---|---|---|---:|---|
| up k = 0.25 | −0.040 | +0.029 | 0.092 | take ≥ −0.038, smear ≤ +0.059 |
| up k = 0.5 | −0.081 | +0.060 | 0.180 | take ≥ −0.077, smear ≤ +0.052 |
| **up k = 1** | **−0.149 [−0.151, −0.147]** | **+0.126 [+0.124, +0.128]** | 0.331 | take ≥ −0.144, smear ≤ +0.110 |
| up k = 2 | −0.200 | +0.219 | 0.476 | take ≥ −0.207, smear ≤ +0.190 |
| down (ablate, 53,250 on rows) | +0.092 [+0.090, +0.095] | −0.041 [−0.044, −0.039] | 0.189 | take ≤ +0.098, smear ≥ −0.027 |
| input flip to "partner has it" (25,348 rows) | −0.181 [−0.186, −0.176] | +0.228 [+0.222, +0.234] | 0.488 | — |
| unit up k = 1 on the flip's rows | −0.144 | +0.034 | — | — |

- **The direction is right, but it is not specific.** Turning unit 5 up makes the net take less and
  smear more. So does turning up almost any other unit. Unit 5 clears the other 15 units at k = 1
  by 0.005 on take and 0.016 on smear. Ablating it raises take less than ablating another unit does.
  Positive drive into this net's first layer pushes play generally toward "leave it, smear". The
  measure can tell a unit's direction. It cannot yet tell a unit from the layer, and a null that
  subtracts the layer's mean push is the next step for the measure.
- **The unit carries the "don't take" half of "partner has it", not the "smear" half.** On the
  flip's rows the unit's own steering gives 0.80 of the flip's take shift (−0.144 against −0.181)
  and 0.15 of its smear shift (+0.034 against +0.228). Flipping the input turns the unit on
  (a₁ 0 → 317, about 0.6σ). Other units carry the count.
- **The steering effect is largest at second seat** (Δtake −0.212 at k = 1), where the unit never
  fires. A dose into a silent unit is a new signal there, which is what steering on off rows means.
- Eight steered picks are in `unit.json` under `examples`. Six of the eight are at second seat.

**Formation.**

| step | + pole's read-R label | precision of `partner-winning` | top set at last seat |
|---:|---|---:|---:|
| 0 | `setter-out-of-trump` (0.78) | 0.19 | 0.54 |
| 1,000 | `x:gone:19` (0.79) | 0.26 | 0.42 |
| 10,000 | `setter-void-off-suit` (0.94) | 0.26 | **0.86** |
| 15,270 | `setter-void-off-suit` (0.88) | 0.30 | **0.99** |
| 30,540 | `partner-winning` (0.84) | 0.84 | 1.00 |
| 45,810 | `partner-winning` (0.97) | 0.97 | 1.00 |
| 76,350 to 916,200 | `partner-winning` (0.99–1.00) | 0.99–1.00 | 1.00 |

It grew in two steps. By step 15,270 it was a last-seat unit, labelled `setter-void-off-suit` because
read R cannot say `last` (#236). Then the dial sharpened toward partner having the trick, between
steps 15,270 and 45,810. The facet grew out of its own seat.

**History (#202).** The top 1% sits slightly later in the hand than partner-winning rows below the
median: trick 3.85 against 3.46. Need and slack are similar, and the bid is 30 in both. Every top
row is at last seat, and no low row is. The gap is the seat, not history.

**The predictions, graded: 4 of 9 hold.**

| | measured | |
|---|---|---|
| 1. partner > LHO > RHO at last; partner > RHO at third; second below every partner cell | last 21.4k > 6.3k > 3.1k; third −11.2k > −31.0k; second −11.0k is *above* third-with-partner −11.2k | **wrong** (third clause) |
| 2. − pole on third/last still `x:winning:2` ≥ 0.9 | 1.000 | holds |
| 3. a detector: gaps < 0.1 of max, overlap < 0.01 | safe gap 0.079 and count gap 0.002 hold; declaration spread 0.170 and overlap 0.909 fail | **wrong** |
| 4. readers 4 and 5 (w₂[4] > 0, w₂[5] < 0); push favours count, disfavours taking | top two readers are 5 (−668) and 7; push both directions hold | **wrong** (readers) |
| 5. up: Δtake < 0, Δsmear > 0, beyond all 15 | −0.149 and +0.126, beyond all 15 by 0.005 and 0.016 | holds, narrowly |
| 6. down: Δtake > 0, Δsmear < 0, beyond all 15 | +0.092 inside the 15 (max +0.098); smear beyond | **wrong** (take) |
| 7. the flip moves the same ways and further | take −0.181 vs −0.144; smear +0.228 vs +0.034 | holds |
| 8. the twin is blind under read R | best `partner-winning` 0.000 (raw 0.475) | holds |
| 9. clean by 91,620 and stays; local labels from 1,000 on; last seat first | clean at 45,810 and after, last seat 1.00 vs 0.24; the labels at 1,000 to 15,270 are not trick-state | **wrong** (labels) |

**Reading.**

- **The loop worked on its calibration unit, by catching the label.** The raw seat-by-winner
  table, the raw shape and formation each said the same thing on their own: last seat first, then
  the dial. Predictions 1, 3 and 9 failed for that reason. They were written for the label, not
  for the unit. This is the essay's warning about percentiles, from one top-1% window: they flatten
  what tells a dial from a detector.
- **Named the essay's way, as a verb**: *"I'm last; lean toward leaving it, more so the better the
  trick already is for us."* It opens at last seat and closes hardest at third seat with an
  opponent on top. It pushes the net away from taking and toward count.
- **The steering measure works and is not yet specific.** Direction by where play goes was the
  missing measure (#205). It reads the right direction here, but the layer's common push nearly
  matches the unit's own. Paper 2 adds a layer-mean null.
- **Next:** paper 2 is the messiest pole in this net under read R, L1+013 (`void-in-led`, 0.766).
  It is read with the seat-by-winner table first (#236).

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## Links

[[clean-neurons-42]] · [[lem-unit-origin-2026-10-20]] · [[lem-lead-units-2026-10-20]] ·
[[lem-fixed-conditioning-2026-10-20]] · [[lem-matched-steering-2026-10-20]] · [[lem-formation-2026-10-17]]
