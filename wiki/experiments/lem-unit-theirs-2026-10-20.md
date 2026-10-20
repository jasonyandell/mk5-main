---
title: "Unit paper 6: the 16×8 seed-0 net's unit 14, 'not partner's trick', loudest when an opponent's off double is winning (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: complete
---

The sixth **unit paper** of [[clean-neurons-42]]'s new mode (#235), on the 16×8 seed-0 field net.
Unit 14 is the handoff's third named unit, and C1 of [[lem-lead-units-2026-10-20]] (#232). There,
read raw, it was "mostly a `ten-on-trick`-side unit" and its lead half was weak (0.127 of max).
Read R now names its poles L1+014 `setter-void-off-suit` and L1−014 `partner-winning`. It is the
second brake on paper 4's doubles switch (w₂ −970; [[lem-unit-lead-doubles-2026-10-20]]).

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## The label, read raw first (exploratory, before registration)

- **On 0.658 of rows.** By seat and winner:
  - partner winning: 0.040 at third, 0.128 at last;
  - an opponent winning: 0.766–0.828;
  - on lead: 0.816, with a low mean z (2,026 against 8.7k–11.4k for an opponent winning).
- **By what an opponent is winning with** (following rows):

  | an opponent winning with | on | mean z |
  |---|---:|---:|
  | an off double | **1.000** | 18,589 |
  | a trump | 0.716 | 6,455 |
  | another tile | 0.622 | 1,498 |
  | (partner winning) | 0.061 | −14,688 |

  Whether the winner can still be beaten (`winner-cannot-be-beaten`) barely matters: 0.716 against
  0.847. Nor does count on the trick: 0.759 against 0.814.
- **Weights.**
  - `x:winning:1` (partner) is −18,490.
  - The opponents' off doubles on the trick (LHO `x:trick:0`, `2`, `5`, `9`, `14`, `20`; RHO `56`,
    `58`, `61`, `65`, `70`, `76`) are +11.6k to +21.1k.
  - A few opponents' trumps are negative, including the 6-6, and others positive.
  - The bias is +392.
- **Readers:** layer-2 unit 3 (the doubles switch) −970, unit 1 −443, unit 0 +295, unit 2 +191.

**The candidate verb: "this trick isn't partner's".** It is on when an opponent is winning, and
on lead (0.816), where nobody is winning yet. It is loudest when an opponent's off double, the boss
of the led suit, is winning. On lead, "theirs" would predict off; "not partner's" predicts on. The
seen on-rate there already favours "not partner's". The variable is graded: mean z climbs 1.5k,
6.5k, 18.6k by what an opponent is winning with. So the label is read in magnitudes, not on-rates.
Read R's + label `setter-void-off-suit` is discussed with the results.

## The loop, as registered

The code is `python -u -m lem.probe unit … 14 --winner` (`lem/probe/unit.py`, `run_winner`). It
was smoke-tested on 16×8 seed 1. The winning tile is replayed from the record (`winning_tile`).
Its kind is one of: partner winning, or an opponent winning with an off double, a trump or another
tile. The audit sample has only pip-trump declarations (0–6), so slot 27 is always the trump
double and the off doubles are always slots 0, 2, 5, 9, 14 and 20.

1. **Label.**
   - Seat × kind cells.
   - **An opponent's off double on the trick**, with mean z (deal-clustered 95% interval) and
     on-rate in each of these cells:
     - **in the led suit with an opponent winning:** the double itself is winning;
     - **in the led suit with partner winning:** partner has trumped it, since a double is the boss
       of its suit;
     - **discarded from another suit**, with partner or an opponent winning;
     - **none on the trick**, with partner or an opponent winning;
     - **lead.**
   - Opponent winning with an off double, split by whether the double was led.
   - Both poles under read R.
   - On/off agreement, and R² on z and on a.
2. **Circuit in.** The 24 largest weights and the bias.
3. **Circuit out.** The readers.
4. **Formation.** At each checkpoint:
   - the on-rate per kind and on lead;
   - **the partner term:** mean z with an opponent winning with another tile, minus mean z with
     partner winning;
   - **the double term:** mean z with an opponent's off double winning, minus mean z with an
     opponent winning with another tile;
   - both labels.
5. **Steering.** Every live layer-1 unit is ablated on the same rows, and its Δ is ranked. Up is
   read against the layer-mean null. The row classes:
   - **opponent's double, I can trump in:** an opponent winning with an off double, trump not led,
     both a trump and a non-trump legal. Read plays trump, also by seat; at second seat RHO can
     still overtrump. Base rates are printed.
   - **opponent's double, nothing of mine wins:** an opponent's double winning, both count and
     non-count tiles legal, and no legal tile beats the winner. Read **count given away**: a count
     tile played that does not take the trick.
   - **partner winning, count choice:** read count given, not overtaking.

## Predictions, sealed before 1–5 are run

The cells, the kind table, the beatable and count splits, the weights, the readers and the lead
on-rate were seen. A layer-1 unit is linear in its inputs, so the label cells are sums of weights
over which inputs co-occur. Only predictions about co-occurrence, use and growth can lose.

1. **"Not partner's" licenses trumping in (down, same rows).** On opponent's-double rows where I
   can trump in, ablating unit 14 lowers plays trump, with its interval below 0. Its |Δ| is among
   the top three of the pool. The split by seat is read off.
2. **It holds the count back from a lost trick (down, same rows).** On opponent's-double rows where
   nothing of mine wins, ablating it raises count given away, with its interval above 0.
3. **Forced on partner's trick, it holds count back (up, off-distribution).** On partner-winning
   rows with a count choice, at k = 1, count given falls beyond the layer: own − layer < 0, with
   its interval excluding 0. The unit is off there, so this tests what it writes.
4. **Partner's trump overrides the double.** With an opponent's off double in the led suit but
   partner winning (partner trumped it), mean z is below 0, with its interval below 0. Partner's
   trump bits on the trick were not among the seen weights, so this can lose.
5. **Whose, then with what (formation, graded).** The partner term reaches half its final value,
   and stays there, at a strictly earlier checkpoint than the double term reaches half of its own
   and stays. A tie fails.

Read off, not graded:
- the lead rows, which were seen and favour "not partner's";
- the discard cells;
- the led-or-not split;
- on/off agreement and R²;
- the poles;
- the other doses.

**What would break the explanation.**
- Predictions 1 and 2 failing: its use is not the ruff-or-hold decision.
- Prediction 4 failing: the double term outweighs whose trick it is, so the unit leans toward
  "an opponent's double is here".
- None would be read as mixing.

**Cost.** One run of about two minutes under `scripts/cpu-slot`.

## The review before sealing

One fresh-context Opus read the essay's full text (to line 943), the draft, paper 5's Results and
`run_winner`. Its points, and what changed:
- **The lead rows contradict "theirs".** → The verb is "not partner's". The lead rows are read
  off, since they were seen.
- **The variable is graded again** (paper 5's mistake). → Label cells are read as mean z with
  intervals, and formation as z terms, not on-rate thresholds.
- **Prediction 1 was half settled by the weights, and "detector against theirs" was a false
  choice for a linear unit.** → Dropped. Prediction 4 asks only what the weights don't settle:
  partner's trump against the double.
- **"Partner winning" with a double on the trick includes discards**, where partner wins without
  trumping. → The double is restricted to the led suit, and discards are reported apart.
- **The smear read counted a count trump that wins the count.** → Count given away (count and
  not taking), on rows where nothing of mine wins. Can-trump-in rows are kept out of
  prediction 2.
- **`can_ruff` let only-trump hands in.** → A non-trump must also be legal. The seat split and
  base rates are added.
- **Prediction 5's thresholds were arbitrary.** → Graded z terms, "reaches half and stays".
- **The slot-27 classification under follow-me.** → Checked: this sample is all pip trumps.
- **Read R's + label went undiscussed.** → It is discussed with the results.

Sealed 2026-10-20.

## Results

Run 2026-10-20 as registered, in one `scripts/cpu-slot` run of 93 s (2.19e12 instructions). The
evidence (`unit.json`, the log, the code, `SHA256SUMS`) is `lem/unit-not-partners-2026-10-20/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/a811456336e4da17110bd83c901b6641740a9b44/lem/unit-not-partners-2026-10-20).

**The picture.** Unit 14 tells the net **not to throw count onto a trick an opponent's double is
winning**. Where nothing of mine can win such a trick, the net gives count away 0.044 of the time.
Ablate this unit and it gives count away 0.426 of the time (Δ +0.383 [+0.375, +0.390]). That is
the largest effect of the 16 units, each ablated on the same rows. The unit's reading of the trick
is "not partner's", with an opponent's winning double the loudest case.

**The label, as mean z [95%]** (following unless noted):

| | rows | on | mean z |
|---|---:|---:|---|
| an opponent's off double in the led suit, an opponent winning | 76,515 | 1.000 | +18,813 [18,778, 18,849] |
| an opponent's off double discarded, an opponent winning | 3,760 | 0.905 | +12,777 |
| no opponent's double, an opponent winning | 117,901 | 0.663 | +3,703 |
| lead (nobody winning) | 150,000 | 0.814 | +2,026 |
| an opponent's off double in the led suit, partner winning (partner trumped it) | 5,677 | 0.352 | **−2,475** [−2,650, −2,293] |
| an opponent's off double discarded, partner winning | 11,087 | 0.247 | −2,488 |
| no opponent's double, partner winning | 69,595 | 0.008 | −17,628 |

- Partner winning subtracts about 21k (−17.6k against +3.7k without a double).
- An opponent's double adds about 15k, wherever it is: in the led suit, discarded, or beaten.
- They are two additive terms, as a linear unit must be. Partner winning outweighs a double, just.
- **Lead rows sit between the two**, on 0.814: "not partner's", not "theirs".
- The double, led or played later, is on 1.000 either way.

**The poles.**
- The − pole is `partner-winning` 0.929.
- The + pole's top 1% is 0.551 opponents'-double tricks. Read R's label `setter-void-off-suit`
  (0.826) is a correlate of those tricks: the defenders are void in an off suit and can trump.

**R²** on a: opponent winning 0.297, the winning kind 0.615.

**Circuit out:** layer-2 unit 3 (the doubles switch) −970, unit 1 −443, unit 0 +295, unit 2 +191.

**Steering, every unit ablated on the same rows (pool of 16).**

| rows | base | unit 14's Δ [95%] | rank | the top unit |
|---|---:|---|---:|---|
| an opponent's double winning, I can trump in: plays trump (25,819) | 0.800 | −0.187 [−0.192, −0.183] | 4 | unit 7 (paper 5) −0.453 |
| an opponent's double winning, nothing of mine wins: count given away (20,011) | 0.044 | **+0.383 [+0.375, +0.390]** | **1** | — (next: unit 10 +0.271) |
| partner winning, count choice: count given, not overtaking (41,370; unit 14 on for 0.060) | 0.583 | — | — | unit 2 −0.251 |

- By seat, ablation lowers trumping in by 0.214 at second, 0.177 at third and 0.163 at last.
- **Up** (k = 1, own − layer):
  - trumping in +0.117 [+0.114, +0.120] where an opponent's double wins;
  - count given away −0.097 [−0.099, −0.095] there;
  - on partner's tricks, count given −0.013 [−0.015, −0.011].

**Formation.**

| step | on | on, partner winning | on, an opponent's double | partner term (z) | double term (z) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.607 | 0.558 | 0.666 | −971 | **+11,572** |
| 1,000 | 0.963 | 0.993 | 0.988 | −10,865 | +5,777 |
| 10,000 | 0.981 | 0.991 | 0.998 | −7,525 | +9,933 |
| 45,810 | 0.834 | 0.717 | 0.996 | −3,022 | +8,711 |
| 76,350 | 0.787 | 0.276 | 0.998 | +4,933 | +11,450 |
| 152,700 | 0.756 | **0.070** | 0.997 | **+15,184** | +15,453 |
| 916,200 | 0.657 | 0.061 | 1.000 | +16,187 | +17,091 |

- **The double term was there at step 0**: an untrained unit already at 0.68 of its final value.
  It dipped while the unit turned on almost everywhere (steps 300–30k), and has stayed above half
  from step 10,000.
- **The partner shut-off came last.** Partner-winning rows were on 0.99 at step 3,000. The term
  went negative before it went positive, and reached half its final value only at epoch 10
  (step 152,700).
- The − label became `partner-winning` at epoch 5 and stayed.

**The predictions, graded: 3 of 5 hold.**

| | measured | |
|---|---|---|
| 1. ablation lowers trumping in, top three | −0.187, interval below 0; **rank 4** (unit 7, 10 and 8 ahead) | **wrong** (rank) |
| 2. ablation raises count given away on a lost trick | +0.383 [+0.375, +0.390], rank 1 of 16 | holds |
| 3. forced on partner's trick, less count given beyond the layer | −0.013 [−0.015, −0.011] | holds (small) |
| 4. partner's trump overrides the double: mean z < 0 | −2,475 [−2,650, −2,293] | holds |
| 5. "whose" before "with what" | the double term stays above half from step 10k; the partner term from step 152.7k | **wrong** (the reverse) |

**Reading.**

- **Named the essay's way, as a verb:** *"not partner's trick; keep your count off it"*. Its main
  use is to hold count back from a trick the opponents hold, most of all one an opponent's double
  wins. Without it, the net throws count there nearly ten times as often. Trumping in on that trick
  belongs mainly to paper 5's unit 7. Unit 14 helps, ranked fourth.
- **Two additive terms, as the review said a linear unit must be.** "An opponent's double is on
  the trick" (+15k) and "partner is winning" (−21k). The unit sums them; downstream, its
  readers see one quantity. Prediction 4 shows the partner term wins when both apply, which is
  what makes "not partner's" the verb rather than "a double is here".
- **"With what" came first and "whose" last.** The double term was in the random initial weights
  and was kept. The partner shut-off was carved out between epochs 3 and 10, turning a unit that
  fired for any trick with an opponent's double into one that stays quiet when partner has the
  trick. In the essay's terms this is formation as growth: a facet present at birth, refined by a
  gate added late. It is not two concepts merged.
- **The series' uses, side by side.**
  - Paper 4's switch leads doubles.
  - Paper 5's counter opens trumping in.
  - Paper 6's unit keeps count off the opponents' tricks.

  Each is the largest-use unit of its layer on its own rows. Paper 1's unit, which shared its push
  with the layer, is the exception so far.
- Not read as mixing. One sum of two related terms, one quantity downstream.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## Links

[[clean-neurons-42]] · [[lem-lead-units-2026-10-20]] · [[lem-unit-lead-doubles-2026-10-20]] ·
[[lem-unit-no-trump-yet-2026-10-20]] · [[lem-unit-partner-has-it-2026-10-20]]
