---
title: "Unit paper 2: the 16×8 seed-0 net's unit 13, a gate whose silence plays the double: shut when I hold canonical tile 14 (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

The second **unit paper** of [[clean-neurons-42]]'s new mode (#235), on the same net as
[[lem-unit-partner-has-it-2026-10-20]]: the 16×8 seed-0 field net. Paper 1 promised the messiest
pole in this net under read R, which is **L1+013**: `void-in-led` at 0.766 [0.753, 0.778], *partial*.
Its other pole, L1−013, is clean at 1.000 for `x:hand:14`.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## The label, read raw first (exploratory, before registration)

Paper 1 found that read R can name a unit by one facet of its main job (#236). So this paper reads
the unit raw before writing any prediction, and says so. The read is
`unit-papers-2026-10-20/explore13.py`: the mean raw pre-activation z₁₃ and the on-rate, by the
state of canonical tile 14 and by seat (the maximum is 41,472, the minimum −70,319):

| canonical tile 14 | on-rate, by seat (lead / 2nd / 3rd / last) | mean z₁₃ |
|---|---|---|
| **in my hand** (17.4% of rows) | 0.000 / 0.000 / 0.000 / 0.000–0.026 | −23,700 to −32,200 |
| gone (played) | 0.98 / 0.68–0.77 / 0.77–0.81 / 0.59–0.64 | +2,200 to +9,200 |
| out (unseen) | 0.99 / 0.69–0.73 / 0.76–0.79 / 0.52–0.54 | +200 to +6,600 |

- **It is a gate: off when I hold canonical tile 14, mostly on otherwise.** The one input bit
  `x:hand:14` explains 0.70 of z₁₃'s variance. Adding `gone:14` and `void-in-led` adds 0.0006.
- **Whether I can follow suit barely moves it.** The + pole's `void-in-led` label is one facet of
  a broad open side, which explains the partial precision.
- **Canonical 14 is the 4-4 slot.** The field net relabels pips so that trump is 6 and the other six
  pips keep their order. So slot 14 is the double of the second-highest non-trump pip. Under trump
  blanks to fours that is the real **5-5, a ten-count double** (57.6% of rows). Under fives or sixes
  it is the real 4-4, which carries no count. That is why `holds-ten` correlates −0.31 with z₁₃.
- **The other correlates** over all rows: `x:gone:14` +0.31, `doubles-2plus` −0.23, `holds-walker`
  −0.20, and the trick state weakly (`third-partner-winning` +0.16).

**What the gate is for, the reading to test.** A double is the highest tile of its suit. So holding
slot 14 means I hold the **boss of canonical suit 4**. When the double is out against me, my other
4s can lose when led (`holds-walker` −0.20). The candidate verb is: **"the boss of this suit is
not mine; don't treat the suit as mine."** The shut side means "I hold the boss". The open side is
graded by seat and by whether the double is gone or still out. The essay's questions: what does
the gate let through, to whom, and how did it grow?

## The loop, as registered

The code is `python -u -m lem.probe unit … --gate x:hand:14` (`lem/probe/unit.py`, `run_gate`). It
was smoke-tested on another net (16×8 seed 1, unit 3, with gates `x:hand:19` and `x:hand:9`). The
rows are the held-out audit sample (434,535 rows) and list v2. Activations are raw.

1. **Label.**
   - The table above, from the run itself.
   - Both poles under read R.
   - R² of z₁₃ on the gate.
   - **Slot or tile:** the held-minus-not-held gap in z₁₃, split by whether slot 14 is the 5-5 (ten
     count) or the 4-4 (no count). The input's `count` block tells the net which.
   - **Near misses:** among rows without the gated tile, the gap from holding another off double
     (0-0, 1-1, 2-2, 3-3 and 5-5 slots), and from holding a non-double 4-x. Each is a fraction of the
     gate's gap.
   - **The open side:** the twelve strongest correlates of z₁₃ among rows without the tile.
   - **The floor for this kind of clean:** the share of layer-1 poles, trained and in the untrained
     twin, whose top 1% is ≥ 0.9 one raw input bit. L1−013 is clean that way.
2. **Circuit in.** The ten largest mean votes on each pole's top 1%. Read off.
3. **Circuit out.**
   - The readers by |w₂|, read off.
   - The push of +σ on legal tiles where the unit is off (I hold the double). It is split into the
     double, its suit's other tiles, other non-trump tiles, count against no count, trump, and
     beats against does not beat.
4. **Formation as growth.** At each of the 15 checkpoints, starting with step 0 as the baseline:
   - R² on the gate;
   - R² of the open side on seat;
   - the on-rates with the tile held and not held;
   - both poles' read-R labels.
5. **Steering, read by where play goes, against a layer-mean null.**
   - **Up:** add k·σ to a₁₃ for k ∈ {0.25, 0.5, 1, 2} where the unit is off.
     - Take and smear are read on following rows where the choice matters.
     - **Plays-gated** (the pick is the double) is read on rows where I hold it, it is legal, and I
       have a choice. Leads are included.
     - **Plays-its-suit** (the pick is a non-double tile of suit 4) is read on lead rows where I hold
       the double and such a tile is legal.
   - **Down:** ablate on the rows where the unit is on, scored on take and smear.
   - **The layer-mean null** (new): every other live layer-1 unit is steered at the same k in its
     own σ. Their shifts are averaged row by row, and the own-minus-layer difference gets a
     deal-clustered 95% interval (1,000 resamples). It answers paper 1's open question of whether
     a shift is the unit's or the layer's. That matters here because the layer's generic
     "leave it, smear" push could, on its own, make the net play a 5-5 more often. Steering the unit
     up on held rows injects a signal it never carries there, which is what steering is.
   - **The input flip:** on held rows where the tile is legal and I have a choice, set `x:hand:14`
     to 0 (the tile stays legal). Read all seats and leads only, beside the unit's own k = 1 on the
     same rows.
   - Eight steered picks are shown in full.

## Predictions, sealed before 1–5 are run

The raw table was seen first. Predictions 1 and 4 sit closest to it. Prediction 1 follows from the
table's single-bit R², and prediction 4's gate half follows from it too. Predictions 2, 3 and 5 are
about what the table does not show.

1. **Slot, not tile.** The held-minus-not-held gap is within a factor of 1.33 (a quarter either
   way) between the 5-5 and the 4-4 slot. The unit reads the slot, whatever real tile fills it.
   Failing means it reads the count, and is a 5-5 unit.
2. **Near misses stay near.** Holding any other off double, or a non-double 4-x, moves z₁₃ by less
   than 0.1 of the gate's gap.
3. **"The boss isn't mine" (circuit out and steering, the walker reading).**
   - Push of +σ where I hold the double: the double's push is below other non-trump tiles'. Its
     suit's other tiles' push is also below other non-trump, non-suit tiles'.
   - Up at k = 1 on rows where I hold it: Δplays-gated < 0, and own-minus-layer < 0 with its
     interval excluding 0. Turning the gate on removes "I hold the boss", and with it the play of
     the boss.
   - The flip moves plays-gated the same way. The unit's k = 1 shift is at least half the flip's,
     so the gate is the majority route of the bit. The flip turning the unit on is arithmetic,
     since z₁₃ is linear in the bit, and is not graded.
4. **The gate came first, then the open side branched.** At step 0, R² on the gate is under 0.05.
   The gate's R² reaches half its final value at an earlier checkpoint than the open side's seat R²
   reaches half of its own.
5. **The open side branches along its own suit.** Among the five strongest open-side correlates,
   at least one is an input about canonical suit 4: the hand, gone or trick bit of slot 10, 11, 12,
   13 or 19.

Read off, not graded: the readers, take and smear for both arms, and the floor share. Paper 1 gave
no basis for their direction.

**What would break the explanation.**
- If prediction 3 fails, or plays-gated sits inside the layer's shift or rises, the gate tells the
  net "I hold it" without steering what it does with it. The walker verb would then be wrong.
- If prediction 1 fails, the unit reads count.
- If prediction 2 fails, the unit is about doubles in general, which is coarser, not mixed.
- None of these would be read as mixing.

**Not done, and why: a synthetic blind maximum.** Building consistent positions from the
explanation alone needs a deal generator, a tool for a later paper. The near misses stand in.

**Cost.** One run of about two minutes, under `scripts/cpu-slot`.

## The review before sealing

One fresh-context Opus read the essay's full text (to line 943), then the first draft and paper 1's
results. It also checked the code: the input has a `count` block, so prediction 1 is a real test.
Its points, and what changed:
- **The draft tested the input bit, not the unit's meaning.** Its verb was a noun ("the double
  isn't mine"). → The walker reading ("the boss of this suit isn't mine") is now the explanation
  that can lose: prediction 3, its push and its play.
- **The open side carries the structure and had no claim.** → Prediction 5 (branching along its
  suit) and prediction 4 (growth order).
- **"The flip turns the unit on" is arithmetic.** → Not graded.
- **"The twin is blind on this bit" was decided in advance.** → Replaced by the floor share for
  "clean on a raw bit", read off.
- **"R² ≥ 0.5 by 10k" was close to free.** → Replaced with a step-0 baseline and an order claim.
- **Near misses were missing.** → Prediction 2.
- **The thresholds were unjustified.** → 1.33 is "a quarter either way" and 0.5 is "the majority
  route". Predictions that sit closest to the pre-read table are marked.
- **The steering confound** (the generic smear push meets a ten-count double). → Stated, and
  handled by the layer-mean null.
- **Not adopted: a graded readers prediction.** Read off, since nothing yet says which reader
  should carry the gate.

Sealed 2026-10-20.

## Results

Run 2026-10-20 as registered, in one `scripts/cpu-slot` run of 105 s (2.32e12 instructions). The
evidence (`unit.json`, the log, the code, the exploration script, `SHA256SUMS`) is
`lem/unit-big-double-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/8c558bea5382447f80593cf32ed06285c27c9da8/lem/unit-big-double-2026-10-20).

**The picture.** Unit 13 is a **switch for playing one tile**. It is open whenever I don't hold
the double in slot 14, which is exactly when that tile can't be played. When I do hold it, the unit
falls silent, and the silence is what lets the net play the double. Turned up on rows where I hold
the double, the net stops playing it, far beyond what the rest of the layer does. The walker
reading, "the boss isn't mine, so leave its suit alone", is wrong: an open gate makes the suit's
other tiles more attractive.

**The label.** The run reproduces the raw table: held rows on 0.000–0.010 with mean z₁₃ −25,060 to
−30,689; gone and out rows on 0.53–0.99. Read R gives `void-in-led` 0.766 for the + pole and
`x:hand:14` 1.000 for the − pole. R² on the gate is 0.699, and adding `gone:14` gives 0.700.
- **Slot, not tile.** Held minus not held is 32,079 where slot 14 is the 5-5 and 31,921 where it
  is the 4-4, a ratio of 1.005.
- **Near misses.** Holding the 0-0, 1-1, 2-2, 3-3 or 5-5 slot moves z₁₃ by −0.012 to +0.009 of
  the gate's gap. Holding a non-double 4-x moves it by 0.038.
- **The floor for this kind of clean.** 0.625 of the trained net's layer-1 poles read ≥ 0.9 on one
  raw input bit, against 0.219 in the untrained twin. The twin's best top-1% precision for this
  gate is 0.503, and its best R² is 0.085.
- **The open side** (rows without the tile). Its strongest correlates are trick state:
  - `x:trick:83`, my right-hand opponent has played the trump double (−0.42);
  - `x:trick:42`, partner has played this double (+0.27);
  - `third-partner-winning` (+0.26), `partner-winning` (+0.26), `third` (+0.21).

  The + pole's top 1% has no stable job. Its read-R label went from `trump-is-fours` to
  `void-in-led` to `x:gone:12` and back across formation. A partial + pole on a gate unit is the
  open door, not a job.

**Circuit in.**
- The − pole is `x:hand:14` (vote −31,388), then the trump double on the trick from RHO (−9,793).
- The + pole's largest votes are the left opponent's tile on the trick being a non-double 4
  (`x:trick:10` to `13`, +2,777 to +3,687), then partner's tile being this double (`x:trick:42`,
  +2,712).
- So the open side opens widest when this suit is being played and someone else has its boss.

**Circuit out.**
- One reader dominates: layer-2 unit 3, at w₂ −1,434. The next reader is under 130. Its poles read
  `no-trump-held` and `setter-void-off-suit`.
- The push of +σ where I hold the double:
  - on the double, −5.99M;
  - on its suit's other tiles, +293k;
  - on other non-trump tiles, −77k;
  - on trump, +759k.
- The open gate holds the double down and lifts trump and the rest of its suit.

**Steering, read by where play goes, against the layer-mean null.**

| k | Δplays the double (59,745 rows held) | own − layer [95%] | Δleads its suit (16,433 lead rows, double held) | own − layer [95%] |
|---:|---:|---|---:|---|
| 0.25 | −0.155 | **−0.161 [−0.164, −0.159]** | +0.029 | +0.030 [+0.027, +0.032] |
| 0.5 | −0.263 | −0.262 [−0.265, −0.258] | +0.062 | +0.060 [+0.056, +0.063] |
| 1 | −0.332 | **−0.318 [−0.321, −0.315]** | +0.104 | **+0.087 [+0.082, +0.092]** |
| 2 | −0.344 | −0.308 [−0.311, −0.306] | +0.099 | +0.048 [+0.043, +0.053] |

- **The base rate of playing the double on those rows is 0.402.** A quarter of a σ removes 0.155
  of it, and the layer's mean shift is +0.007. This is the first strongly specific steering result
  in the program. Paper 1's unit beat the other units by 0.005.
- **On leads alone**, k = 1 drops leading the double by 0.651 [0.644, 0.658].
- **Take and smear** (read off): at k = 1, take −0.021 against the layer's −0.023, and smear −0.022
  against +0.002. Smear falls because the double is often the 5-5. Down (ablate where open): take
  −0.005, smear +0.021 (own − layer +0.008).
- **Eight steered picks**, all leads. In five the base lead was the double and the steered lead is
  something else. Two of those are now leading a trump (5-4 under fives; 6-3 under fours).
- **The input flip** (`x:hand:14` → 0 on 59,908 held rows; the unit goes from a₁ 0.35 to 303, near
  0.9σ). Plays the double −0.045 [−0.049, −0.042], against the unit's own k = 1, −0.332, on the same
  rows. The bit's other routes push the other way: flipping the bit moves play far less than the
  gate alone. The gate is the majority route by the registered rule, but not the whole story. Some
  other unit also reads "I hold it" and works against it.

**Formation.**

| step | R² on the gate | R² of the open side on seat | on, held / not held | − pole |
|---:|---:|---:|---|---|
| 0 | 0.002 | 0.031 | 0.33 / 0.30 | `setter-void-off-suit` |
| 1,000 | 0.002 | 0.044 | 0.05 / 0.05 | `x:gone:0` |
| 3,000 | 0.000 | 0.066 | 0.05 / 0.08 | `x:gone:6` |
| 10,000 | 0.322 | **0.300** | 0.00 / 0.18 | `x:hand:14` |
| 15,270 | 0.403 | 0.154 | 0.00 / 0.33 | `x:hand:14` |
| 30,540 | 0.641 | 0.030 | 0.00 / 0.58 | `x:hand:14` |
| 916,200 | 0.699 | 0.054 | 0.00 / 0.79 | `x:hand:14` |

- **It nearly died first.** By step 1,000 it fired on 5% of rows.
- **Then it came back as the gate**, between steps 3,000 and 10,000. In that window it was also,
  briefly, a seat unit on its open side (R² 0.30).
- **The seat dependence then faded** as the open side widened to fire on 0.79 of not-held rows.
- **The shut side was settled at once**: from step 10,000 on, holding the tile shuts it on
  ≥ 0.997 of rows.

**The predictions, graded: 2 of 5 hold.**

| | measured | |
|---|---|---|
| 1. slot, not tile (gap ratio within 1.33) | 1.005 | holds |
| 2. near misses under 0.1 of the gap | all within 0.038 | holds |
| 3. the walker reading: double and suit pushed down; plays-gated falls beyond the layer; the gate is the majority route | the double −5.99M; plays-gated −0.332, own − layer −0.318; unit 7× the flip. **But its suit is pushed up** (+293k against −77k), and leading it rises (+0.087 beyond the layer) | **wrong** (the suit) |
| 4. the gate first: step-0 R² < 0.05, gate reaches half before the open side's seat R² | 0.002 at step 0; but the open side's seat R² is at half its final value at step 0 (0.031 of 0.054) | **wrong** |
| 5. a suit-4 input among the five strongest open-side correlates | trick state and the double itself (`x:trick:42`); the suit-4 inputs lead the + pole's votes, not the correlates | **wrong** |

**Reading.**

- **Named the essay's way, as a verb**: *"the double isn't in my hand"*. It is on exactly when the
  double can't be played, and its silence plays the double. A ReLU unit can't fire for "I hold X,
  play X" and still be quiet elsewhere for free. This net built the inverse: an always-on brake on
  one tile, lifted when the tile is mine. One reader carries it.
- **The messiest pole in the net was the open side of a clean gate.** Its partial label, `void-in-led`
  at 0.766, is where the open side's other votes add up. Its − pole is clean at 1.000. A unit is two
  poles: read the shut side as the gate before calling the open side unnamed. This is the essay's
  negative pole as a gate, in 42.
- **The steering measure separated a unit from its layer for the first time**, at 0.318 against
  paper 1's 0.005. Where play goes is the right read. A unit tied to a tile shows it plainly. A unit
  tied to a situation (paper 1) shares its push with the layer.
- **The walker reading lost, and that is the finding.** With the gate open the net favours its suit
  and trump. Holding the double is not about the suit's other tiles here. It is about this tile.
- **The flip is the open question.** The bit's other routes nearly cancel the gate. Paper 3 can
  follow the second carrier of `x:hand:14`.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## Links

[[clean-neurons-42]] · [[lem-unit-partner-has-it-2026-10-20]] · [[lem-fixed-conditioning-2026-10-20]] ·
[[lem-formation-2026-10-17]]
