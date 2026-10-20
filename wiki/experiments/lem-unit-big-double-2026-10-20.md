---
title: "Unit paper 2: the 16×8 seed-0 net's unit 13, a gate that shuts when I hold canonical tile 14 (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: sealed
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

*(not run)*

## Links

[[clean-neurons-42]] · [[lem-unit-partner-has-it-2026-10-20]] · [[lem-fixed-conditioning-2026-10-20]] ·
[[lem-formation-2026-10-17]]
