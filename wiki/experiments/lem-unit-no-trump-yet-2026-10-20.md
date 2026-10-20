---
title: "Unit paper 5: the 16×8 seed-0 net's unit 7, shut by any trump on the open trick, a concept list v2 does not have (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: sealed
---

The fifth **unit paper** of [[clean-neurons-42]]'s new mode (#235), on the 16×8 seed-0 field net.
Layer-1 unit 7 is the largest input to the doubles switch of [[lem-unit-lead-doubles-2026-10-20]]
(w₂ +765). Read R names its poles L1+007 `last-setter-winning` and L1−007 `trump-led`.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## The label, read raw first (exploratory, before registration)

- **Its large weights are all trumps already on the trick.**
  - The opponents' trumps: `x:trick:21`–`27` (LHO) and `77`–`83` (RHO), −12k to −27k.
  - Partner's trumps: `x:trick:49`–`55`, −8.6k to −12k, about half as large.
  - Its bias is +1,584.
- **On lead the trick is empty, so it is on 0.998.** Off lead it is on 0.50–0.77 by seat and
  winner. Its R² on seat is 0.156.
- **It feeds all eight layer-2 units positively** (w₂ +8 to +765). The doubles switch, unit 3,
  takes the most.
- **List v2 has no concept for "someone has played a trump to this trick".** Its trump concepts are
  about the hand, what's out, the lead, and declarations: `trump-led`, `last-trump-led`,
  `can-ruff`, `trump-out` and others. `trump-led` is a facet of the missing concept: trump on the
  trick because it was led, not because someone trumped in. This is #202's case (a vocabulary gap
  before superposition) and #236's (read R names a unit by a facet).

**The candidate verb, from the readers' side: "no one has trumped this trick (yet)".** The unit
feeds every layer-2 unit positively, so what it lets through is the signal that the trick has no
trump on it. That is always true on lead, and true off lead until someone trumps. An opponent's
trump shuts it harder, by the weights.

## The loop, as registered

The code is `python -u -m lem.probe unit … 7 --trick` (`lem/probe/unit.py`, `run_trick`). It was
smoke-tested on 16×8 seed 1. The concept T, "a trump is on the open trick", is built from the
record with real tiles (`trumps_on_trick`). It is split by who played the trump (LHO, partner, RHO)
and by whether trump was led. The audit sample has only pip-trump declarations (0–6, layout
field-v2), so T and the net's canonical trump slots 21–27 agree on every row.

1. **Label.**
   - **What readers see, on and off.** At each following seat, the agreement of "on" with "no
     trump on the trick" and with "trump not led". At second seat these are the same rows, since
     only the leader has played.
   - Seat × {no trump, trump led, trumped in} cells.
   - By who: no trump, opponent only, partner only, both.
   - Both poles under read R. The − pole's top-1% precision for T and for `trump led`.
   - R² while following, on z and on a: T, `trump led`, who trumped, who + seat, T + seat, and all
     inputs.
   - The vocabulary check: list v2's five best concepts on the − pole by raw precision, with their
     base rates.
2. **Circuit in.** The 24 largest weights and the bias.
3. **Circuit out.** The readers by |w₂|.
4. **Formation.** At each of the 15 checkpoints, while following:
   - on-rate;
   - R² against T, `trump led` and who trumped;
   - the − pole's precision for T;
   - both labels.
5. **Steering, on following rows.**
   - **Can trump in** (prediction 4's rows): no trump on the trick yet, an opponent winning, trump
     legal and not led, more than one legal tile. Every live layer-1 unit is ablated to 0 on these
     same rows, and its Δplays trump is read. The pool is the units on for at least 200 rows; its
     size is reported. Ranking by |Δ| favours units that are on more, so each unit's on-rate there
     is reported beside it.
   - **An opponent trumped in and is winning** (trump not led): smear and take, read off.
   - **All eligible following rows:** take, smear and plays trump, with the up arm against the
     layer-mean null, and the down arm where each unit is on, split by seat.

## Predictions, sealed before 1–5 are run

The weights, the seat cells and the read-R labels were seen. Every trump-on-trick weight
(−8.6k to −27k) dwarfs the bias (+1,584), so the − pole's precision for T is close to settled, and
is read off. Predictions are graded on what readers see (on or off), not on z's depth below zero.

1. **The missing concept, not the facet (on/off).** At third and at last seat:
   - "on" agrees with "no trump on the trick" on at least 0.85 of rows;
   - that agreement is at least 0.05 above its agreement with "trump not led".
2. **The gap is in the vocabulary.** No list-v2 concept with base rate ≤ 0.5 reaches 0.9 raw
   precision on the − pole.
3. **Any trump, not only the opponents'.** Following, with partner's trump the only trump on the
   trick, the unit is on for fewer than 0.2 of rows. This is the shared premise of predictions 1
   and 4: that T is the unit's variable. The alternative it tells apart is "an opponent's trump is
   on the trick", under which partner-only rows would mostly be on.
4. **"No one has trumped" licenses trumping in (down, on the same rows for every unit).** On
   can-trump-in rows, ablating unit 7 lowers plays trump, with its interval below 0. Its |Δ| is
   among the top three in the pool. Failing on the sign means the verb's use is something else,
   for example "the suit can still win".
5. **Coarse to fine (formation).** R² against T reaches half its final value, and stays there, at
   a strictly earlier checkpoint than the one at which who-trumped first adds 0.05 over T and stays
   there. A tie fails.

Read off, not graded:
- the − pole's T-precision;
- "who + seat" against "T + seat", on z and on a: the weights nearly settle who, and on
  trump-led tricks who is tied to seat;
- up-steering;
- the smear and take reads;
- the lead side: the hand explains R² 0.818 of z on lead, but the unit is on 0.998 there, so
  readers see almost none of it.

**What would break the explanation.**
- Prediction 1 failing: the facet `trump-led` is closer to the unit's own concept.
- Prediction 3 failing: the variable is the opponents' trump, not any trump.
- Prediction 4 failing on the sign: the verb is not about trumping in.
- None would be read as mixing.

**Cost.** One run of about two minutes under `scripts/cpu-slot`.

## The review before sealing

One fresh-context Opus read the essay's full text (to line 943), the draft, paper 4's Results and
`run_trick`. Its points, and what changed:
- **The verb was stated from the − pole.** → It is now stated from the readers' side.
- **R² on z grades a depth no reader sees.** → Predictions are graded on on/off, and R² on a is
  added.
- **Prediction 1's precision clause was settled by the weights, and its `trump led` clause failed
  on arithmetic.** The deepest rows are trump-led tricks. → On/off agreement per seat, at third
  and last, where the two concepts differ.
- **Prediction 3 was settled by the weights and tied to seat.** → Read off. The new prediction 3
  tests the shared premise.
- **The shared premise couldn't be told apart.** Under a pip trump, "a trump on the trick" and
  "the winner is a trump" are the same rows. → The alternative is now "an opponent's trump", told
  apart by partner-only rows.
- **Prediction 4's rows were never measured by the down arm, the row class missed "an opponent
  winning", and units were ranked on different rows.** → Can-trump-in rows, with every unit
  ablated on the same rows. The pool and on-rates are reported, and the bias toward units that are
  on more is stated.
- **`opp_trumped_win` included trump-led tricks.** → Fixed (`~led_t`).
- **Declaration mismatch under field-v3.** → Checked: this sample is field-v2, all pip trumps.

Sealed 2026-10-20.

## Results

*(not run)*

## Links

[[clean-neurons-42]] · [[lem-unit-lead-doubles-2026-10-20]] · [[lem-unit-lead-brake-2026-10-20]] ·
[[lem-unit-big-double-2026-10-20]]
