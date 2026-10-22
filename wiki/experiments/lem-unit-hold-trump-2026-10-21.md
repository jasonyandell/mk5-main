---
title: "Unit paper 10: the 16×8 seed-0 net's layer-1 unit 1, a trump vote shut short of trump and when the second trump falls before me (Lem, #235)"
kind: experiment
first_seen: 2026-10-21
last_updated: 2026-10-21
status: active
---

The tenth **unit paper** of [[clean-neurons-42]]'s new mode (#235), on the 16×8 seed-0 field net.
Layer-1 unit 1 is the last gap in the #236 audit's boundary column (#202).
- It is on for 0.829 of rows.
- Its best list-v2 on/off match is `¬lead-no-trump` at 0.860, against a chance level of 0.829.
- Read R labels both poles clean:
  - L1+001 is `last-setter-winning` (0.961);
  - L1−001 is `x:gone:26` (1.000), the second trump gone.

Both names come from the deepest 1%, and the pole is not the job ([[clean-neurons-42]]). The negative
pole's `x:gone:26` is a superset label. Every row of its deepest 1% is the narrower event
`x:trick:82`, the second trump just played by the player before me, which is 2.1% of the sample. The
question is what the unit does where it is on, and how it came to do it. Its neighbour is paper 7's
L1 8 ([[lem-unit-pull-trump-2026-10-20]]), which shuts on leads from trump length. On leads the two
are anticorrelated (r −0.40). Paper 5's L1 7 ([[lem-unit-no-trump-yet-2026-10-20]]) counts trumps on the
open trick and licenses trumping in, so it is unit 1's rival on prediction 2.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

**The net's frame.** The field net sees tiles in canonical slots, relabelled per declaration so that
trump is canonical pip 6. The slots that matter here:

| canonical slot | blanks–threes | fours | fives | sixes | in every declaration |
|---|---|---|---|---|---|
| 27 | d-d | 4-4 | 5-5 (a ten) | 6-6 | the trump double |
| 26 | 6-d | 6-4 (a ten) | 6-5 | 6-5 | the second trump |
| 21–25 | | | | | the lower trumps |
| 15–19 | 6-x | 6-x | 6-x | 5-x | the top off suit's singles |
| 0, 2, 5, 9, 14 | | | | | the low off doubles |
| 20 | 6-6 | 6-6 | 6-6 | 5-5 | the top off double |

The trick block of the input records who played what: "right" is the player just before me.
`x:trick:82` is "the player before me played slot 26".

## The label, read raw first (exploratory, before registration)

The scripts are in `scratch/lem/unit-papers-2026-10-20/l1u01/`: `raw.py`, `explore.py`, `explore2.py`
and `explore3.py`, with their logs. No label was imposed first.

**Where it is off: leads short of trump, and one tile before me.**
- Off on 0.170 of rows. Off by seat: lead 0.421, second 0.036, third 0.038, last 0.043.
- **On leads its off rate falls with trumps held.** (R² of z on trumps held is 0.29, so z is mostly
  something else; the ladder below reads the rest.) Off by trumps held, under blanks–threes: 0.45–0.47 with
  none, 0.22 with one, 0.05 with two, 0.00 with three or more. Under fours it is 0.58, 0.37, 0.15,
  0.02; under fives 0.78, 0.61, 0.43, 0.17; under sixes 0.68, 0.45, 0.27, 0.05. It also rises by
  trick, from 0.19 on the first lead to 0.66 on the sixth.
- Holding the trump double lowers it (0.152 against 0.456). So do the 4-4 in slot 14 (0.193) and
  slot 25 (0.190). Holding slot 19 raises it (0.769 against 0.358).
- **On follows, one input shuts it: the player before me just played the second trump** (slot 26).
  Off is 1.000 on those 9,149 rows, which are 0.83 of the follow off-set. The weight is −26,135,
  the largest in the unit.
  - The same player's other trumps keep it on: off 0.002 for 21–23, 0.014 for 24, 0.045 for 25,
    0.007 for the trump double.
  - The second trump played by partner (off 0.016) or by the leader when I am last (0.017) does
    not shut it.
  - Other trump-double weights are smaller: −12,462 when the leader led it (I am last), −5,079 when
    partner played it, −2,208 when the player before me played it.
  - On those rows, trump was led on 0.80; the player before me holds the trick on 0.90; I hold the
    trump double on 0.18; the trump double is gone on 0.51.
- On follows, mean z is highest at last seat with the leader holding the trick (17,400, against
  about 10,600–11,900 elsewhere). The weight on "the left-hand opponent holds the trick" is +8,112,
  the third largest.
- The remaining 1,819 follow off-rows (not the second trump) lean toward partner having played
  the trump double (0.358 against 0.025), partner holding the trick, and holding slot 19.
- The best v2 concept for off is `lead-no-trump`, at Cohen's κ 0.446. On leads alone the best is
  `¬holds-trump` at 0.307.

**What it writes (through the net, seen).** This is the mean change in the canonical tile scores when
the unit is ablated, base minus ablated, on rows where it is on.
- On leads:
  - for: the top off suit's singles (slots 15–19, +0.92M to +1.45M) and the second trump (+1.46M);
  - for, less: the lower trumps (+0.08M to +0.52M) and the other off singles (+0.17M to +0.37M);
  - against: the trump double (−1.44M) and the low off doubles (−0.07M to −1.48M);
  - the top off double is +0.29M.
- On follows the pattern is the same and larger. The top off suit's singles are +1.65M to
  +2.62M, the second trump +2.41M, the trump double −2.17M, and the low off doubles −0.38M to
  −1.42M.

**What it does (exploratory ablation, seen).** It is ablated where it is on (rows with more than one
legal tile). About a quarter of picks change: 0.249 overall.
- **Leads with a trump and a non-trump legal (64,108 rows):** trump led 0.389 → 0.225; off doubles
  0.500 → 0.654; count 0.157 → 0.188.
- **All leads where on (86,649 rows):** off doubles 0.575 → 0.689; changed 0.172.
- **Follows, by seat:**
  - second: trump 0.324 → 0.267, take 0.363 → 0.321, count 0.180 → 0.258, off doubles 0.110 → 0.177;
  - third: trump 0.258 → 0.241, count 0.246 → 0.287;
  - last: trump 0.275 → 0.247, take 0.225 → 0.205.
  - At last seat with partner holding the trick, trump rose (0.176 → 0.204) and take rose (0.083 →
    0.123).
- **Forced on where the second trump was just played** (the median on value on follows, 790, added):
  where I hold the trump double, the trump double played falls 0.898 → 0.826. Overall it changes
  0.112 of picks.

**Implied by arithmetic before sealing (the paper-9 lesson).**
- **Leads.** The two lead rows above give the remainder: leads where on with a choice but not a
  trump-and-non-trump choice. That is 22,541 rows, almost all leads holding no trump.
  - Any non-trump double moves there by 86,649 × 0.114 − 64,108 × 0.154 ≈ 5 row flips. Each Δ
    carries ±0.001 of rounding, so Δ ≈ 0.000 ± 0.007.
  - Changed picks there are about (86,649 × 0.172 − 64,108 × 0.225) / 22,541 ≈ 0.021, about 480
    rows.
  - This bounds any non-trump double, not low doubles alone: a swap between slot 20 and the low
    doubles would not show.
  - **Strongly implied: on leads the unit acts through trump.** The rise in doubles is the slack from
    fewer trump leads.
  - One confound: on no-trump leads the unit is mostly off (0.627), so where it is on there the dose
    may be low. The cell doses are read below.
- **Follows.** A trump-or-not choice on a follow exists only where I am void in a led off suit (the
  F-R rows). If trump is led I must follow trump; if an off suit is led and I hold it, no trump is
  legal.
  - So the seen per-seat trump rows are F-R's trump Δ: 91,554 × (−0.057) + 91,824 × (−0.017) + 90,179 ×
    (−0.028) ≈ −9,305 rows, ±about 270.
  - **The direction of "it trumps in" is settled.** At last seat with partner holding the trick,
    trump rose (0.176 → 0.204).

**Formation and the twin (seen).**
- At step 0 the unit is on 0.892, with weight −2,907 on the second trump before me, and off on 0.079
  of those rows.
- Off at leads with no trump rose from 0.03–0.20 (steps 1,000–15,270) to 0.54 at step 30,540 and 0.87
  at 76,350, then settled at 0.63.
- **The second-trump kill came late.** The weight was −9,340 at step 76,350 (off 0.444), then −23,761
  at 152,700 (off 0.967). Its final value is −26,135 (off 1.000).
- R² of z on trumps held, on leads, was 0.05 at step 0 and 0.29 at the end.
- The untrained twin (the step-0 net) is on 0.892. It is off on 0.102 of leads and 0.152 of
  no-trump leads.

**Seen on another net (the smoke test).** The code was smoke-tested on seed 1's 16×8 net, on the
20k smoke rows. Seed 1's L1 1 is an unrelated unit, on 0.233. Its cell directions were seen and
inform nothing below: L-0 −0.075, F-D −0.018, F-S −0.160, F-R −0.053, both rank 11–13.

**The working verb (to test, not a label).** "Take it from the opponents with trump": on a lead,
lead a lesser trump; on a follow, trump in and take. The positive pole sits there: z is highest
at last seat with the leader (an opponent) holding the trick. The unit is weaker on partner's
trick, where ablating it raises trump and take. Doubles and count rise when it is ablated, as
slack.
- On leads the gate is "I hold trump".
- In the net's frame the write prefers the second trump to the trump double.

**The second-trump kill: its reason is open.** One candidate is "let the trump double cover". A
second trump from the player before me is beaten only by the trump double. The unit votes against
the double, so shutting it would lift that vote where the double is needed. The forced-on read is
consistent and small (0.898 → 0.826). Three seen facts cut against it:
- When the leader leads the second trump and I am last holding the double, covering is just as
  needed. Yet the unit stays on (off 0.017).
- Only 0.18 of the kill rows hold the double, yet the kill is the unit's largest weight.
- When the player before me plays the trump double itself, the unit stays on (0.007). So the reason
  is not "my trumps can't take any more" either.

Graded prediction 5 tests the candidate where the gate is missing.

Two readings on follows are open, and the graded predictions part them:
- **A, "trump and take":** on follows, too, it acts through trump and taking. Where no trump can be
  played, it moves nothing.
- **B, "keep the doubles back":** on follows, unlike leads, the anti-double write moves discards.
  Where I am void with no trump, it holds doubles back.

## The loop, as registered

The code is `python -u -m lem.probe unit … 1 --hold` (`lem/probe/unit.py`, `run_hold`). It was
smoke-tested on seed 1, as above.

1. **Label.**
   - Off and z by seat. On leads: by declaration × trumps held, by trick, by low doubles held (with
     no trumps and with some).
   - On follows: by which trump the player before me played. The leader's second trump. The share
     of the follow off-set explained by the second trump.
   - Cohen's κ and κ against a constant for:
     - `¬lead-no-trump` (best v2, seen);
     - "following, unless the player before me played the second trump";
     - the v3 candidate, "a lead holding a trump, or following unless the player before me played
       the second trump".
   - The R² ladder on leads: trumps held, + trick, + low doubles held, + declaration.
   - Both poles under read R.
2. **Circuit in.** The weights, the hand row, and the trick block as a table: who played which slot.
3. **Circuit out.** The layer-2 readers (w₂ column, with their labels) and the write through the net,
   on leads and follows where on. The twin: on rate, its second-trump weight, off on leads with 0 and
   3+ trumps, and its write.
4. **Formation.** At each of 15 checkpoints:
   - on, off on leads with 0 and 3+ trumps, off and weight on the second trump before me;
   - the write on leads and follows;
   - on the final net's rows: trump led on choice leads, and its Δ when ablated; Δ trump played on
     F-R rows when ablated;
   - both poles.
5. **Steering.** Every live layer-1 unit is ablated on the same rows. The reads are: low double, any
   double, trump double, trump, second trump, the top off suit's singles, count, take, and double
   taking. Each has a deal-clustered interval, and unit 1's rank among the live units both ways.
   The cells, all where unit 1 is on unless named:
   - **L-T:** leads holding a trump, a low double and a single legal (focus: low double);
   - **L-0:** leads holding no trump, a low double and a single legal (focus: low double);
   - **L-27:** leads with the trump double and another trump legal (focus: trump double);
   - **F-R:** void in a led off suit, a trump and a non-trump legal (focus: trump);
   - **F-D:** discards (void, no trump legal), a low double and a single legal (focus: low double);
   - **F-S:** following an off suit with that suit's double and another of the suit legal (focus: any
     double);
   - **F-27:** trump led, the trump double and another trump legal (focus: trump double);
   - **the near miss:** the leader led the second trump and I am last, with the trump double legal.
     The unit is on there;
   - **the second trump before me**, the trump double legal. The unit is off there; ablation is a
     no-op and is read for completeness.

   The pooled leads with a trump and a non-trump legal, ablated (seen, confirmatory).

   **Up:** add k·σ (k = 0.25, 0.5, 1, 2), then the median on value on follows, own − layer. The rows
   are leads holding no trump with a low double and a single legal, where it is off; and the
   second-trump rows with the trump double legal.

## Predictions, sealed before 1–5 are run on seed 0

**Seen before sealing** (above):
- the off rates by seat, trick, declaration × trumps held, and per slot held, on leads;
- the follow off-set and the trump table for the player before me, the leader and partner;
- every weight; the write through the net on leads and follows;
- the pooled ablation on leads (trump-and-non-trump, and all) and by seat × holder on follows;
- the forced-on read on the second-trump rows;
- the formation's on rate, off rates on leads, the second-trump weight and off rate, and R² on
  trumps held;
- the twin's on rate, weight and lead off rates;
- the correlation with L1 8;
- seed 1's smoke run.

**Not seen on seed 0:** any of the cells above, except as implied by the pooled rows (c2, c4). Also unseen:
the formation's use (Δ trump led per checkpoint) and write per checkpoint, the up doses, the
boundary κ for the candidates, the R² ladder past trumps held, and the poles' fine print.

**Confirmatory, reported, not graded:**
- **c1, leads act through trump:** on pooled leads with a trump and a non-trump legal, ablating
  lowers trump led (seen: 0.389 → 0.225);
- **c2, no trump, little effect:** on leads holding no trump, ablating moves doubles led by little.
  The arithmetic above strongly implies it (any non-trump double, ±0.007 on the remainder), so L-0
  is reported here, with its dose, not graded;
- **c3, the kill:** forced on where the second trump was just played and I hold the trump double,
  the trump double played falls (seen: 0.898 → 0.826);
- **c4, it trumps in:** on F-R rows, ablating lowers trump played (settled: ≈ −9,305 rows pooled).

**Graded:**

1. **On follows it is still a take-with-trump vote: a discard does not move (A over B).** On F-D
   rows, ablating unit 1 moves the low double discarded, and count discarded, each by less than
   0.03 either way: both intervals lie within [−0.03, +0.03].
   - B, "keep the doubles back", is the low-double interval wholly above +0.03.
   - An interval that straddles ±0.03, or a count interval outside the band, is a miss for 1 but not
     B. The page then says what moved.
   - A cell under 200 rows leaves 1 ungraded. Its dose (mean activation) is reported beside it.
2. **Where it can trump in, it is the layer's main trump vote.** On F-R rows, unit 1's fall in trump
   played ranks in the top 3 of the live layer-1 units, most negative first. The page reports n, the
   number of live units. The direction is c4. The rival is paper 5's L1 7, which licenses trumping
   in.
3. **On leads it plays a lesser trump and keeps the trump double (L-27).** On leads with the trump
   double and another trump legal, ablating raises leading the trump double, interval above 0. The
   write's sign (−1.44M) is a mean over all leads. Whether it flips picks on these rows is not seen,
   and the low doubles (up to +1.48M ablated) compete. This tests the write on leads only. It says
   nothing about the follow kill.
4. **The trump-lead job came before the second-trump kill (formation).** At step 30,540, Δ trump led
   when ablated is already at least half its final value. That Δ is read on the final net's choice
   leads where on; the final value is seen as about −0.164, so the bar is ≤ −0.082.
   - At that step the kill shuts only 0.180 of its rows (seen), against 0.967 at step 152,700.
   - The order, a gate on the negative side forming after the job it serves, is our derivation, not
     the essay's claim. The essay's ordering is a branch growing out from a local origin, and
     consolidation coming after use.
   - A miss means the trump-lead job was not yet in place when the kill began to form.
5. **The kill's candidate reason, tested where the gate is missing.** On the near-miss rows (the
   leader led the second trump, I am last, the trump double legal, the unit on there), ablating unit
   1 raises the trump double played, interval above 0. If the candidate is right, the unit holds the
   double back where it must cover, and the gate's miss here costs something. A miss (interval
   containing or below 0) means the unit's vote does not hold the double back there. Then "let the
   double cover" is not the kill's reason, c3 is a dose effect of forcing it on, and the reason
   stays open. A cell under 200 rows leaves 5 ungraded.

**Read off, not graded:**
- L-T, F-S, F-27, every read and rank in every cell, and each cell's dose (on, mean activation,
  mean z);
- L-27 split by the bidding side and the defenders, before it is set beside the book;
- the rest of the follow off-set (inputs that separate it);
- the readers' w₂ per checkpoint: which reader came to depend on it;
- every unit's Δ in each cell;
- the up doses and the on value;
- the boundary κ, with the v3 candidate;
- the R² ladder;
- the rest of the formation (write, poles, Δ trump on F-R);
- the twin's write;
- the readers;
- L1 8 beside it (paper 7's brake on trump leads from length), and L1 7's rank on F-R (paper 5);
- the trump leads beside *Winning 42*'s guidance on pulling trump, and on when to lead the trump
  double ([[book-authority]]).

**What would break the explanation.**
- 1 missing upward: on follows the unit is a doubles brake as well as a take-with-trump vote, so
  the verb is too narrow.
- 2 failing: it trumps in, but it is not the layer's main trump-in vote.
- 3 failing: the write's preference for the second trump does not act on leads.
- 4 failing: the trump-lead job was not in place before the kill began.
- 5 failing: "let the double cover" is not why the kill exists.

None would be read as mixing. A miss names what the unit does instead.

**Cost.** One run under `scripts/cpu-slot`, with a timeout of 900 s. Seed 1's smoke on 20k rows took
6.7 s. The full sample is about 22 times larger, with 16 units' ablations, the up doses and 15
checkpoints (each with two ablations and two writes).

## The review before sealing

One fresh-context Opus read the essay's full text (allow_large, to line 944), the draft, `run_hold` and
`Trick`, the four exploratory scripts and logs, paper 9's review, papers 5 and 7, the friction log on
arithmetic, and the card's "The work". Its points, and what changed:
- **Prediction 2's direction was already settled.** On a follow, a trump-or-not choice exists only
  on F-R rows. So the seen per-seat trump rows are F-R's, ≈ −9,305 rows pooled.
  → The direction is confirmatory (c4); only the rank is graded, with n of live units reported.
- **c2's arithmetic was right but its bound was not.** The bound is ±0.007, not ±0.004. It covers
  any non-trump double, and about 480 changed picks leave room on the L-0 subset. There is also a
  dose confound, since the unit is mostly off on no-trump leads.
  → Reworded to "strongly implied". Every cell now reports its dose.
- **Prediction 1 parted the verbs on low doubles only.** The pooled follows also show count held
  back.
  → Count on F-D joins the band. The straddle case is defined.
- **Prediction 4 at step 76,350 tested "both half-grown", not order.**
  → It is graded at step 30,540, where the kill shuts 0.180 of its rows. The order is labelled our
  derivation, not the essay's.
- **Prediction 3's link to the kill did not hold.** It tests leads; the kill acts on follows. The
  near miss and the trump-double-before-me rows cut against "let the double cover".
  → The link is cut. The kill's reason is stated as open, with the three facts against the
  candidate, and graded prediction 5 tests it where the gate is missing.
- **The positive pole was not read into the verb.**
  → The verb is now "take it from the opponents with trump". The weaker effect on partner's trick is
  named.
- **"A dial on trumps held" overstated the fit** (R² 0.29).
  → Softened to the off rate falling with trumps held.
- **The pole label is a superset.**
  → Said: `x:gone:26` covers `x:trick:82`.
- **Missing reads.** → Added: the rest of the follow off-set; the other trump-double weights; the
  readers' w₂ per checkpoint.
- **The book.** → L-27 is split by bidding side before it is set beside the book.
- **The code's masks checked out.** Minor: `single_l` admits trumps 21–26, which affects L-T
  (read off). `G` was dead code.
- Nothing weighs the net against the book, and nothing reads the unit as mixed.

Sealed 2026-10-21.

## Links

[[clean-neurons-42]] · [[lem-unit-pull-trump-2026-10-20]] · [[lem-unit-no-trump-yet-2026-10-20]] · [[lem-unit-top-double-2026-10-21]] ·
[[lem-unit-count-on-it-2026-10-21]] · [[book-authority]]
