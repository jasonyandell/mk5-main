---
title: "Unit paper 4: the 16×8 seed-0 net's layer-2 unit 3, where papers 2 and 3 meet: the switch the net's off-double leads run through (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

The fourth **unit paper** of [[clean-neurons-42]]'s new mode (#235), and the first on layer 2.
Papers 2 and 3 found that `x:hand:14` reaches play through two layer-1 units of opposite sign:
- unit 13 brakes layer-2 unit 3 (w₂ −1,434), see [[lem-unit-big-double-2026-10-20]];
- unit 0 drives it (w₂ +877), see [[lem-unit-lead-brake-2026-10-20]].

This paper reads that reader. Read R names its poles L2+003 `no-trump-held` and L2−003
`setter-void-off-suit`.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## The label, read raw first (exploratory, before registration)

- **A lead unit.** It is on for 0.942 of leads, against 0.008–0.179 at the other seats. Its mean z
  is +320k on lead and between −724k and −1.04M elsewhere.
- **On lead, it fades with trump length.** By trumps held: 0.995 on at zero trumps, 0.980 at one,
  0.898 at two, 0.789 at three, 0.535 at four. Holding the 6-6 brings it to 0.643, and the 6-5 to
  0.839.
- **Holding the slot-14 double does not shut it** (0.987 held, 0.933 not). Units 0 and 13 cancel
  here, as paper 3 found.
- **Its write** (w₃[:, 3], seen):
  - up on every off double: slot 14 (the 5-5 or 4-4) +16,538, the 3-3 +5,870, the 5-5/4-4
    partner slot 20 +5,358, the 2-2 +4,837, the 0-0 +4,543, the 1-1 +4,204;
  - down on the 6-5 (−9,613);
  - small negatives on several non-double off tiles.

**The candidate verb: "lead an off double, not trump".** It is on when I lead and am not long in
trump. The nearest book rule is C23, "the partner leads doubles"
([[book-study-2026-10-12]], where the arena sided with the book). The net has no partner-convention
input and is not graded against the book here. The bidder split is read off for comparison only.

## The loop, as registered

The code is `python -u -m lem.probe unit … 3 --layer 2` (`lem/probe/unit.py`, `run_reader`). It
was smoke-tested on 16×8 seed 1, layer-2 unit 2. The rows are the 434,535 held-out audit rows and
list v2.

1. **Label.**
   - The seat × winner cells.
   - On lead: the on-rate by trumps held, by off doubles held, by each trump tile held, and by
     bidder or not.
   - Both poles under read R.
   - R² of z on lead against trumps held (one-hot), against off doubles held (one-hot), both, the
     hand bits, and all inputs. Also R² of z against seat.
   - The same increments over the trick number (#226), and the on-rate when short in trump with no
     off double held.
2. **Circuit in.** Each layer-1 unit's contribution w₂[3, u]·a₁[u]: its mean on lead and off lead,
   and its sd on lead (ranked).
3. **Circuit out.** The write w₃[:, 3]: the mean over off doubles, over trumps and over the rest,
   and its correlation with the off-double indicator. The untrained twin's maximum |correlation|
   over its eight layer-2 writes is the floor.
4. **Formation**, at each of the 15 checkpoints:
   - on-rate on lead and off it;
   - R² of z on lead against trumps held;
   - the write's correlation with the off doubles;
   - both poles' labels.
5. **Steering on leads.** Three measures, each read on lead rows where the choice exists:
   - **leads off double**: an off double and another tile are legal;
   - **leads trump**: a trump and a non-trump are legal;
   - **leads 6-5**: the 6-5 and another tile are legal.

   Take and smear are read on following rows.
   - **Up:** k ∈ {0.25, 0.5, 1, 2}, against the layer-mean null (the other live layer-2 units, each
     in its own σ).
   - **Down, the natural-use read papers 2 and 3 lacked:** ablate to 0 where the unit is on. On lead
     it is on 0.94 of the time, so this reads what it does in play. Reported:
     - the share of picks changed;
     - where a changed pick that was an off double goes: trump, the 6-5, another off double, or
       other;
     - the share of off-double leads removed, within each trick number (so the base rate's
       headroom and the play count are handled), for 0–1 trumps against 3 or more, weighted by
       rows over the tricks both groups reach;
     - the bidder split.
   - **The null for the down arm:** each other live layer-2 unit ablated on the same row classes
     *where it is on*. The unit's own effect is ranked among them; it is not compared with their
     mean on its rows, which would sit near 0 because they are mostly off on lead.
   - Eight ablated picks are shown in full.

## Predictions, sealed before 1–5 are run

The cells, the trumps-held and trump-tile tables, and the write were seen. They are not graded.
Since w₃ is the last layer, the *sign* of ablation's effect on each tile's score follows from the
seen write, so signs are not predicted.

1. **It is the layer's double-leader, and its doubles give way to trump (down).**
   - Of the 8 layer-2 units, each ablated where it is on, unit 3 has the largest |Δleads off
     double|. Chance is 1 in 8.
   - Of the changed picks that were an off double, at least half go to a trump. The write's only
     large trump term is the 6-5's, so where they go is not set by the write.
2. **Its use lives where trump is short (down, within trick number).** The share of off-double
   leads removed is larger at 0–1 trumps than at 3 or more.
3. **"When" to lead them, not "whether I have them" (label, within trick number).**
   - On lead, the increment in R² from trumps held over the trick number is larger than the
     increment from off doubles held.
   - The near miss: on lead, short in trump (0–1) and holding no off double, the unit is still on
     for at least 0.9 of rows.
4. **The circuit in includes both carriers of paper 2's bit.** Unit 0 and unit 13 are both among
   the four layer-1 units with the largest contribution sd on lead. Chance is about 1 in 20. The
   signs of their mean contributions follow from w₂ (a₁ ≥ 0) and are read off.
5. **"When" before "what" (formation).**
   - The lead gating means on lead minus on off lead is ≥ 0.5, measured while the unit is alive
     (on ≥ 0.05 of rows), and staying ≥ 0.5 at every later checkpoint.
   - The lead gating appears at a strictly earlier checkpoint than the one at which the write's
     correlation with the off doubles reaches half its final value and stays there.
   - A tie fails.

**The shared premise, named (paper 3's lesson).** Predictions 2 and 3 both assume that trump length
gates the unit's use. If the unit is mainly "my lead" with a weak trump modulation, both fail
together, and that premise is what fails.

Read off, not graded:
- up-steering, take and smear;
- the negative pole (`setter-void-off-suit`) and its lead share;
- the twin's floor for the write's correlation, which was close to settled by the seen write;
- the bidder split: the book's C23 is about the partner, which this input does not mark;
- the ablated examples.

**What would break the explanation.**
- Prediction 1 failing: another unit leads the doubles more, and this one is a part of that.
- Predictions 2 and 3 failing together: the premise above.
- Prediction 4 failing: paper 2's two routes meet elsewhere too.
- None would be read as mixing.

**Cost.** One run of about two minutes under `scripts/cpu-slot` (8 layer-2 units × 4 doses, plus
8 ablations).

## The review before sealing

One fresh-context Opus read the essay's full text (to line 943), the draft, paper 3's Results and
`run_reader`. Its points, and what changed:
- **Prediction 1's sign was arithmetic.** The write is the last layer, and it was seen. → Signs are
  not predicted. Prediction 1 now grades rank among the layer and where the doubles' picks go.
- **The down-arm null was near 0 by construction.** The other units are mostly off where unit 3 is
  on. → Each unit is ablated where it is on, and the unit is ranked.
- **Predictions 2 and 3 were mechanical or partly seen.** z falls with trump length, there is
  headroom near 0 and 1, and the play count confounds both. → Predictions 2 and 3 are read within
  trick number, as a share of the base rate, with the pooling rule registered.
- **Prediction 4's sign clause followed from w₂.** → Dropped. Its chance baseline is stated.
- **Prediction 5 could pass at step 0 or during a die-and-revive.** → "Reaches and stays", taken
  while the unit is alive.
- **The gate itself was untested.** → The no-double near miss is in prediction 3. The shut cases
  (long trump, the 6-6) were seen, so they are not graded. The negative pole is read off.
- **Paper 3's lesson.** → The shared premise of predictions 2 and 3 is named.

Sealed 2026-10-20.

## Results

Run 2026-10-20 as registered, in one `scripts/cpu-slot` run of 63 s (1.36e12 instructions). The
evidence (`unit.json`, the log, the code, `SHA256SUMS`) is `lem/unit-lead-doubles-2026-10-20/`
in [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/95e2cd48c6186f29a3fa0132b55b2bf6a92a4e96/lem/unit-lead-doubles-2026-10-20).

**The picture.** Layer-2 unit 3 is the switch the net's off-double leads run through. On lead rows
where an off double and another tile are legal, the net leads an off double 0.757 of the time.
Ablating this unit where it is on cuts that by **0.686** [0.683, 0.689], and changes 0.75 of the
picks. No other layer-2 unit, ablated where it is on, moves off-double leads by more than 0.147.
Trump length decides how strongly it is on, but wherever it is on, nearly all the off-double leads
depend on it.

**The label.**
- On lead it is on 0.942; off lead, 0.007–0.178 (R² on seat 0.479).
- Read R names its + pole `no-trump-held` 0.879 (0.963 of its top rows are leads), and its − pole
  `setter-void-off-suit` 0.840 (no leads).
- **Within trick number** (the trick number alone explains R² 0.072), trumps held adds 0.359 and off
  doubles held adds 0.065. Given trumps held, off doubles add 0.014.
- On-rate on lead:
  - by trumps held: 0.995, 0.979, 0.898, 0.789, 0.535 and 0.316 for 0 to 5 trumps;
  - by off doubles held: 0.900, rising to 0.994 at four;
  - short in trump with no off double: **0.975**. It fires when it is time to lead doubles,
    whether or not I hold one.

**Circuit in** (contribution sd on lead):
- unit 7, w₂ +765, the largest, its mean +636k;
- unit 13, −1,434 (mean −296k);
- unit 0, +877 (mean +160k);
- unit 14, −970 (mean −131k).

So paper 2's and paper 3's units are its second and third inputs. Unit 7 is a lead for paper 5.

**Circuit out.** The write's mean is +6,892 over the off doubles, −1,566 over trumps and −1,901
over the rest. Its correlation with the off-double indicator is 0.794. The untrained twin's best
layer-2 write reaches 0.325.

**Down, the natural use** (ablated where on; 99,048 lead rows with an off double and another tile
legal):
- leads off double −0.686, leads trump +0.250, leads 6-5 +0.625;
- on following rows (where it is on 0.04–0.18), take −0.144 and smear −0.001.

| trumps held | rows | Δleads off double [95%] |
|---:|---:|---|
| 0 | 36,628 | −0.812 [−0.816, −0.807] |
| 1 | 32,931 | −0.717 [−0.722, −0.712] |
| 2 | 20,895 | −0.607 [−0.614, −0.600] |
| 3 | 7,350 | −0.235 [−0.245, −0.225] |
| 4 | 1,150 | −0.119 [−0.140, −0.100] |

As a share of the off-double leads in each group, within trick number (tricks 1–4, where both
groups have rows), it removes **0.875 at 0–1 trumps and 0.996 at 3 or more**. Long in trump, the
net leads an off double less often, but when it does, it is all this unit.

**Where the doubles' leads go** (70,097 changed picks that were an off double):
- 0.672 to a non-double off tile;
- 0.297 to a trump, the 6-5 in 0.106;
- 0.031 to another off double.

**Up** (k = 1, against the layer-mean null): leads off double +0.133 (own − layer +0.193
[+0.192, +0.195]), leads trump −0.137 beyond the layer, leads 6-5 −0.306 beyond the layer.

**The bidder split, read off:** −0.626 for the bidder, −0.737 for the others. The book's C23 is
about the partner, which these inputs do not mark.

**Formation.**

| step | on lead | on off lead | R² lead on trumps held | write corr with off doubles | + pole |
|---:|---:|---:|---:|---:|---|
| 0 | 0.092 | 0.105 | 0.017 | 0.201 | `setter-void-off-suit` |
| 300 | 0.119 | 0.054 | 0.001 | 0.305 | `setter-void-off-suit` |
| 1,000 | **0.980** | 0.110 | 0.125 | **0.886** | `x:need:12` |
| 3,000 | 0.962 | 0.047 | 0.246 | 0.870 | `no-trump-held` |
| 45,810 | 0.963 | 0.121 | 0.166 | 0.856 | `count-out-20plus` |
| 916,200 | 0.942 | 0.106 | 0.254 | 0.794 | `no-trump-held` |

- Born between steps 300 and 1,000, with "when" (lead) and "what" (the doubles) arriving in the same
  interval.
- The trump modulation came after, by step 3,000, and wandered: R² 0.275 at 10k, 0.105 at 153k,
  back to 0.254 at the end.
- The + pole's label went from `no-trump-held` to `count-out-20plus` and back: the trump-length
  facet and a count facet took turns at the top 1% while the lead switch held.

**The predictions, graded: 3 of 5 hold.**

| | measured | |
|---|---|---|
| 1. the layer's double-leader; its doubles' leads go to trump at least half the time | rank 1 of 8 (−0.686 against ≤ 0.147); **but 0.297 go to trump, 0.672 to other off tiles** | **wrong** (where they go) |
| 2. a larger share removed at 0–1 trumps than at 3+ | 0.875 against 0.996 | **wrong** |
| 3. trumps over doubles within trick number; near miss on ≥ 0.9 | +0.359 against +0.065; 0.975 | holds |
| 4. units 0 and 13 among the top four inputs | ranks 3 and 2 (unit 7 first) | holds |
| 5. "when" strictly before "what" | both first at step 1,000 and stay | **wrong** (tie) |

**Reading.**

- **Named the essay's way, as a verb:** *"lead an off double"*. It is the net's doubles-lead switch,
  on when I lead, dimmed as my trump grows. Without it, the net's off-double leads mostly become
  leads of other off tiles, not trump.
- **The shared premise held, and the use differs from the gate.** Prediction 3 says trump length
  sets how strongly the unit is on. Prediction 2 says trump length does not set how much of the
  doubles' lead it carries: everywhere it is on, almost all of it. The dimming is in the gate, not
  in the use. That is a cleaner unit than registered.
- **"Lead a double" is not "don't lead trump".** The write's trump term is small except on the 6-5,
  and ablation sends the doubles' leads to other off tiles. The 6-5 is the exception: steering up
  cuts leading it by 0.306 beyond the layer. That is paper 3's "the double, not the 6-5", now one
  level down.
- **Born whole.** The lead gate and the doubles write appeared together between steps 300 and
  1,000, and the trump dimming came after. "Coarse-to-fine wins right away" is the essay's
  phrase for this, and here it was a coarse unit refined by a modulation, not a merge.
- **Papers 2–4 together.** Holding the slot-14 double reaches this switch by two opposite routes
  (unit 13 brakes it, unit 0 drives it). The switch itself barely notices whether I hold doubles
  (+0.065). It reads whether it is time to lead them, and the doubles' own legality does the rest.
  That is why paper 2's flip of the bit moved play so little: the net's question "should I lead a
  double?" does not depend on which doubles I have.
- Not read as mixing anywhere. The + pole's alternating label (trump length, then count out) is two
  facets of one switch's gate, with one write.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## Links

[[clean-neurons-42]] · [[lem-unit-lead-brake-2026-10-20]] · [[lem-unit-big-double-2026-10-20]] ·
[[lem-unit-partner-has-it-2026-10-20]] · [[book-study-2026-10-12]]
