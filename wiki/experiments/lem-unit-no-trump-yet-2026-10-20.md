---
title: "Unit paper 5: the 16×8 seed-0 net's unit 7, a counter of trumps on the open trick that licenses trumping in (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
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

Run 2026-10-20 as registered, in one `scripts/cpu-slot` run of 102 s (2.25e12 instructions). One
read was added after the run and is marked as such: `u07/posthoc.py`, the on/off agreement with
each list-v2 concept and the on-rate by number of trumps. The evidence (`unit.json`, `posthoc.json`,
the log, the code, `SHA256SUMS`) is `lem/unit-no-trump-yet-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/23f7f85e655183c95ed302d5946dc6dfcc267208/lem/unit-no-trump-yet-2026-10-20).

**The picture.** Unit 7 counts the trumps on the open trick, an opponent's trump counting more
than partner's, and shuts at about one. What it lets through, "this trick has no trump yet",
licenses the net to trump in. On rows where I can trump in, ablating it cuts trumping in by
**0.411** (base 0.797), the largest of all 16 layer-1 units.

**The label: a staircase, not a switch** (following rows; the post-hoc read):

| trumps on the trick | rows | on |
|---|---:|---:|
| none | 177,468 | 0.954 |
| one, partner's | 19,856 | 0.361 |
| one, an opponent's | 48,706 | 0.231 |
| an opponent's and partner's | 22,803 | 0.009 |
| two or more opponents' | 15,702 | ≤ 0.001 |

- "On" agrees with "no trump on the trick" on 0.906 of following rows.
- The best list-v2 concept, `trump-led` (or its negation), agrees on 0.887 (post-hoc).
- By seat:

  | seat | agreement with "no trump on the trick" | agreement with "trump not led" |
  |---|---:|---:|
  | second | 0.942 | 0.942 (the same rows) |
  | third | 0.866 | 0.823 |
  | last | 0.909 | 0.895 |
- A trick trumped in rather than led shuts it less fully: on 0.287 at third and 0.459 at last.
  Those are mostly single trumps.
- **The poles.**
  - The − pole's top 1% is all trump on the trick (1.000) and all `trump led` (0.9998). The deepest
    rows are trump-led tricks with several trumps.
  - The + pole, `last-setter-winning` (0.866), is almost never a lead (0.009).
- **R² while following.** On z: T 0.592, `trump led` 0.551, who trumped 0.754. On a: T 0.479,
  who + seat 0.490, T + seat 0.553.
- **The vocabulary check.** On the − pole, `trump-led` (base 0.187) reaches 0.9998, then `last`
  0.988 and `last-trump-led` 0.988. The list names the pole. What it lacks is the staircase's
  boundary between none and one, where the unit's on/off lives.

**Circuit out.** Every layer-2 unit reads it positively: unit 3 (the doubles switch) +765, unit 5
+425, unit 2 +346, unit 1 +313, unit 0 +267, then three under +90.

**Steering.**
- **Can trump in** (47,011 rows: no trump yet, an opponent winning, trump legal and not led; unit 7
  on for 0.992 of them). Every unit was ablated on these same rows; the pool is all 16:

  | unit | Δplays trump [95%] | on there |
  |---:|---|---:|
  | **7** | **−0.411 [−0.415, −0.407]** | 0.992 |
  | 10 | −0.308 | 0.550 |
  | 1 | −0.209 | 1.000 |
  | 8 | +0.197 | 0.990 |
  | 12 | −0.158 | 0.334 |
- **Down where on**, all eligible following rows: take −0.204, plays trump −0.175, smear −0.034.
  By seat, take falls −0.268 at second, −0.141 at third and −0.183 at last.
- **Up** (k = 1, own − layer): plays trump +0.148 where I can trump in and trump wasn't led; take
  +0.064. Where an opponent has trumped in and is winning, smear is −0.003 [−0.006, +0.001] and
  take +0.054. With the unit forced on, the net trumps more but doesn't throw count to a trick an
  opponent has already ruffed.

**Formation.**

| step | on | R² T | R² `trump led` | R² who | − pole's T precision | − label |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.440 | 0.003 | 0.006 | 0.040 | 0.400 | `setter-void-off-suit` |
| 1,000 | 0.777 | 0.001 | 0.002 | 0.050 | 0.543 | `setter-void-off-suit` |
| 3,000 | 0.933 | 0.029 | 0.009 | 0.100 | 0.723 | `partner-void-off-suit` |
| 15,270 | 0.835 | 0.172 | 0.100 | 0.212 | 0.917 | `partner-void-off-suit` |
| 45,810 | 0.755 | 0.380 | 0.299 | 0.463 | 1.000 | `trump-led` |
| 916,200 | 0.777 | 0.592 | 0.551 | 0.754 | 1.000 | `trump-led` |

- It switched on almost everywhere by step 3,000, then grew the trump shut-off slowly over 60
  epochs.
- R² on who led R² on T at every checkpoint: the weighting by seat grew with the shut-off, not
  after it.
- Its − label wandered through "a void off suit" before settling on `trump-led` at epoch 2.

**The predictions, graded: 1 of 5 holds.**

| | measured | |
|---|---|---|
| 1. on agrees with "no trump" ≥ 0.85, and ≥ 0.05 over "trump not led", at third and last | 0.866 and 0.909; gaps **0.043 and 0.014** | **wrong** (narrowly) |
| 2. no v2 concept reaches 0.9 on the − pole | `trump-led` 0.9998 | **wrong** |
| 3. partner-only rows on < 0.2 | **0.361** | **wrong** |
| 4. ablation where I can trump in lowers it, top three of the pool | −0.411, rank 1 of 16 | holds |
| 5. T before who | both settle at step 45,810 | **wrong** (tie) |

**Reading.**

- **Named the essay's way, as a verb:** *"no one has trumped this trick yet, so trumping in is
  open"*. That is what it lets through, and its use is the largest of the layer: without it, the
  net trumps in 0.41 less often.
- **The variable is a count, not a bit.** A single partner trump leaves it on about a third of the
  time, a single opponent trump about a quarter, two trumps almost never. Predictions 1–3 assumed
  a bit. The staircase is why both the facet test and the "any trump" test missed. The registered
  alternative, "an opponent's trump", is half right: the opponents' trumps weigh about twice as
  much, but partner's still mostly shut it.
- **The pole and the job live at different depths.** Read R's label `trump-led` is right about
  the − pole: the deepest rows are trump-led tricks, full of trumps. The job sits at the boundary
  between none and one. No list-v2 concept names that boundary (best agreement 0.887, against
  0.906 for the built concept and the staircase's sharper picture). This is #236's lesson at a new
  angle: the pole's top 1% is not the unit's decision. It is also #202's: the gap is a missing
  concept, "trumps on the trick, by side", and the list now has a candidate for it.
- **The series' three uses so far:** paper 4's switch leads doubles, and paper 5's unit opens
  trumping in. Both are the largest-use units of their layer on the rows where their verb applies.
  Paper 1's unit, by contrast, shared its push with the layer.
- Not read as mixing anywhere. One input family (trumps on the trick), one shut-off and one write
  feed every layer-2 unit.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## Links

[[clean-neurons-42]] · [[lem-unit-lead-doubles-2026-10-20]] · [[lem-unit-lead-brake-2026-10-20]] ·
[[lem-unit-big-double-2026-10-20]]
