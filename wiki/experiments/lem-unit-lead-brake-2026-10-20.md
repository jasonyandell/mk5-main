---
title: "Unit paper 3: the 16×8 seed-0 net's unit 0, on lead when I hold neither the 6-5 nor the slot-14 double, and the routes of paper 2's bit (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: sealed
---

The third **unit paper** of [[clean-neurons-42]]'s new mode (#235), on the 16×8 seed-0 field net.
It picks up two threads at once:
- the handoff's "lead unit";
- paper 2's open question ([[lem-unit-big-double-2026-10-20]]): flipping `x:hand:14` moves the
  double's play by only −0.045, while unit 13's own steering moves it by −0.332. What carries the
  rest of the bit, and which way does it push?

Unit 0 is the layer-1 unit with the next-largest weight on that bit (w₁ −13,910, after unit 13's
−31,395). Read R names its poles L1+000 `x:winning:2` (0.907, clean) and L1−000 `x:hand:26`
(0.855, partial).

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## The label, read raw first (exploratory, before registration)

- **It fires on 0.282 of rows**, and mostly on lead: 0.606 of leads, against 0.012–0.335 at the
  other seats.
- **On lead it is a two-tile gate.** It is on for 0.893 of leads when I hold neither canonical 26
  (the **6-5**, the second-highest trump) nor canonical 14 (paper 2's double). It is on for 0.000
  when I hold either. The hand bits explain 0.97 of its variance on leads. Holding the 6-6 does
  not shut it (w₁ +4,028).
- **Off lead, the trick shuts it.** Its largest weights are trumps already on the trick: partner's
  6-5 and 6-4 (`x:trick:55`, `54`, −31,570 and −29,413), the left opponent's 6-6 (`x:trick:27`
  −27,345), and others. Being second to play also shuts it (`x:open:1` −26,411).
- **The read-R label `x:winning:2` names the off-lead facet's top.** The partial `x:hand:26` names
  one of the two lead-side shut-offs.
- **The circuit already visible in the weights, for paper 2's unit:** unit 13 → layer-2 unit 3
  (w₂ −1,434) → slot 14 (w₃ +16,538). Unit 13's silence releases reader 3, and reader 3 lifts the
  double: a brake on a brake.

The data supports only a two-tile label: **"I hold neither the 6-5 nor the slot-14 double"**. The
6-6, the biggest tile, does not shut it. Whether a wider verb holds, for example "no big tile is
mine or on the trick", is what predictions 3 and 5 test. Why these two tiles share a shut state is
left open. Formation (prediction 4) asks which came first.

## The loop, as registered

- **Two gate runs**, `python -u -m lem.probe unit … 0 --gate x:hand:26` and `--gate x:hand:14`
  (`run_gate`, as in paper 2). They give:
  - the held/gone/out × seat tables and the poles;
  - near misses: the other trumps for the 6-5, the other doubles for slot 14;
  - circuit in and out, and formation;
  - steering with the layer-mean null, and the flip.

  New: **plays-gated, lead**, a steering read on lead rows only, reported beside all seats.
- **Steering is off-distribution, and unavoidably so.** The unit is shut on exactly the rows where
  the tile can be played. Up-steering there tests the sign of what the unit writes, not what it
  does in play, as in paper 2. The down arm (ablating where it is naturally on) is read for take
  and smear.
- **Route decomposition**, `python -u -m lem.probe route`, for `x:hand:14` → tile 14 and
  `x:hand:26` → tile 26. On rows where I hold the tile, it is legal and I have a choice, the bit is
  turned off. The tile stays legal, so the flipped input is one training never saw. It is read as
  an intervention, not a position: the hand-derived inputs (count and trump bits) still say I hold
  it. Per unit:
  - its **share**: the flip's effect minus the effect with the unit held at its old value;
  - its **alone** effect: only that unit moved.

  Layer 2 is a ReLU layer, so neither sums to the flip. Both sums are reported, with the 13 × 0
  joint hold and its interaction. Each effect is read on the pick and on the margin (the tile's
  score minus the best other legal score), over all rows, over leads, and per moved row.
- **Off lead** (`u00/offlead.py`): at second, third and last seat, among rows holding neither tile,
  the unit's on-rate with a trump on the trick against none.
- Smoke-tested on 16×8 seed 1.

## Predictions, sealed before any of this is run

The raw read above and the first-layer weights were seen. Predictions those weights nearly settle
are marked.

1. **What it writes is a brake on the two tiles (steering, off-distribution).** At k = 1, on lead
   rows where I hold the 6-5, Δplays-gated, lead < 0, with own − layer < 0 and its interval
   excluding 0. The same holds on lead rows where I hold the slot-14 double.
2. **Unit 0 is not the counter-route of paper 2's bit, but works with unit 13 (route,
   `x:hand:14` → 14, pick, all rows).** If prediction 1's brake is right, unit 0 opens when the bit
   flips and pushes the double down. So:
   - unit 0's share and its alone effect are both negative, the same sign as unit 13's;
   - the unit with the largest positive share is read off and named as paper 4's subject.

   This replaces the first draft's prediction, which wanted unit 0 to be the counter-route. That
   wish contradicted prediction 1.
3. **Two tiles, not "big trumps" (near misses, nearly settled by the weights).** Among rows without
   the 6-5, holding any other trump moves z₀ by less than 0.1 of the 6-5's gap. Among rows without
   slot 14, holding any other double does the same against slot 14's gap.
4. **The 6-5 brake came first (formation).** The first checkpoint at which the unit fires on under
   5% of 6-5-holding rows is **strictly earlier** than the same checkpoint for slot 14. A tie fails:
   the grid is coarse, and a tie says nothing about order.
5. **The off-lead facet shares the verb (off lead).** At third seat and at last seat, among rows
   holding neither tile, the on-rate with a trump on the trick is under half the on-rate with none.
   Then the shut state reads "a big trump is mine or on the trick", one verb across seats. Failing
   means the off-lead facet is a separate job.

Read off, not graded:
- the readers;
- take and smear;
- the route shares for `x:hand:26` (nearly settled by the weights);
- the margin reads;
- the pair interaction.

**What would break the explanation.**
- Prediction 1 failing: the shut state isn't a brake on leading these tiles.
- Prediction 2 failing (unit 0's share positive): unit 0 also counters the gate, and the brake
  reading of what it writes is wrong for the double.
- Prediction 5 failing: the unit holds two jobs split by seat. That is still read as a coarse unit
  before mixing, with its readers checked.
- None would be read as mixing.

**Cost.** Two gate runs of about two minutes, two route runs and the off-lead script of seconds
each, all under `scripts/cpu-slot`.

## The review before sealing

One fresh-context Opus read the essay's full text (to line 943), the draft, paper 2's Results and
the `route` code. Its points, and what changed:
- **Predictions 1 and 2 contradicted each other in sign.** If an open unit 0 brakes the double, its
  share in the double's route is negative, like unit 13's. → Prediction 2 now follows from 1, and
  the counter-route is read off for paper 4.
- **The label claimed "big tile", but the 6-6 doesn't shut the unit.** → Two-tile label.
  Prediction 3 grades the near misses.
- **The off-lead facet had no claim.** → Prediction 5.
- **Steering is off-distribution.** → Stated. The down arm is read where the unit is naturally on.
  The gated play can only be read where the unit is shut.
- **Prediction 3 (the 6-5's main route) was settled by the weights.** → Read off.
- **Prediction 4 passed on a tie, and its reason described the final net.** → Strictly earlier,
  and the reason dropped.
- **The route decomposition** had an impossible flipped input, was non-additive, read the argmax
  only, and mixed effect with how often a unit moved. → It now has:
  - the impossible input named;
  - a share and an alone effect, with both sums;
  - a pair interaction;
  - a margin read;
  - the moved rate and a per-moved-row effect.

Sealed 2026-10-20.

## Results

*(not run)*

## Links

[[clean-neurons-42]] · [[lem-unit-big-double-2026-10-20]] · [[lem-unit-partner-has-it-2026-10-20]] ·
[[lem-fixed-conditioning-2026-10-20]]
