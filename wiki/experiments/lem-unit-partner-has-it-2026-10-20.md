---
title: "Unit paper 1: the 16×8 seed-0 net's unit 5, \"partner has it\" against \"the player just before me has it\" (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: sealed
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

*(not run)*

## Links

[[clean-neurons-42]] · [[lem-unit-origin-2026-10-20]] · [[lem-lead-units-2026-10-20]] ·
[[lem-fixed-conditioning-2026-10-20]] · [[lem-matched-steering-2026-10-20]] · [[lem-formation-2026-10-17]]
