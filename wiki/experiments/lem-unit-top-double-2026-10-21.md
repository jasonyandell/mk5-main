---
title: "Unit paper 9: the 16×8 seed-0 net's layer-2 unit 1, leading the top off double under fives and sixes (Lem, #235)"
kind: experiment
first_seen: 2026-10-21
last_updated: 2026-10-21
status: complete
---

The ninth **unit paper** of [[clean-neurons-42]]'s new mode (#235), on the 16×8 seed-0 field
net. Layer-2 unit 1 is the next gap in the #236 audit's boundary column (#202). It is on for 0.201
of rows. Its best list-v2 on/off match is `trump-is-sixes` at 0.820 against a chance level of
0.799. Read R labels both poles clean:
- L2+001 is `x:count:39` (0.995), the input bit for a 5-count in canonical slot 11, which marks the
  declaration;
- L2−001 is `winner-cannot-be-beaten` (0.921).

Both names come from the deepest 1% (the pole is not the job: [[clean-neurons-42]]). The negative
pole still says what the unit keeps out. Its runners-up are `trump-led` (0.81) and `last` (0.70):
following into a trick already decided, kept out of a unit that acts on an open lead. The question
is what the unit does where it is on and how it came to do it. Its neighbours are paper 4's L2 3,
the switch the net's off-double leads run through ([[lem-unit-lead-doubles-2026-10-20]]), and
paper 8's L2 4 ([[lem-unit-count-on-it-2026-10-21]]), whose feeding job is split by declaration
group.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

**The net's frame.** The field net sees tiles in canonical slots, relabelled per declaration so
that trump is always canonical pip 6. Three slots matter here:

| canonical slot | 0–4 (blanks–fours) | 5 (fives) | 6 (sixes) | what it is in every declaration |
|---|---|---|---|---|
| 20 | 6-6 | 6-6 | 5-5 (a ten) | the top off double |
| 14 | 5-5 (a ten) | 4-4 | 4-4 | the second off double |
| 19 | 6-5 | 6-4 (a ten) | 5-4 | the top off suit's second tile |

## The label, read raw first (exploratory, before registration)

The scripts are in `scratch/lem/unit-papers-2026-10-20/l2u01/`: `raw.py`, `explore.py` and
`explore3.py`, with their logs. No label was imposed first.

**Where it is on: leads under fives and sixes.**

| declaration | lead | second | third | last |
|---|---:|---:|---:|---:|
| 0–3 | 0.047–0.058 | 0.000 | 0.000–0.001 | 0.000 |
| 4 | 0.092 | 0.000 | 0.003 | 0.000 |
| 5 | **0.963** | 0.001 | 0.165 | 0.049 |
| 6 | **0.994** | 0.285 | 0.608 | 0.290 |

- The best v2 concept for on is `lead`, at Cohen's κ 0.437. No v2 concept is "trump is fives or
  sixes": the list has each declaration as its own bit, and the audit tries single concepts and
  their negations, not unions.
- **The gate is a pattern in the net's frame, not a list of declarations: "slot 14 holds no ten".**
  Under blanks–fours it is held shut on leads by layer-1 unit 15 (contribution on leads −1,033k under 0–4,
  0 under 5 and 6). L1 15's on/off boundary in the #236 audit is `x:count:14` (0.943): it is on
  where canonical slot 14 holds a ten, the 5-5, which is under blanks–fours. On follows the
  largest closer is L1 10 (−5.57M mean contribution on off rows against −1.19M for L1 15; its
  on/off boundary is the same `x:count:14`).
- **Within fives leads it dims when I hold the 6-4** (canonical slot 19, a ten under fives):

  | fives leads | rows | on | mean z | the 6-6 led | a trump led |
  |---|---:|---:|---:|---:|---:|
  | no 6-6, no 6-4 | 30,090 | 0.999 | 283k | — | 0.211 |
  | no 6-6, 6-4 | 8,436 | 0.895 | 104k | — | 0.416 |
  | 6-6, no 6-4 | 8,154 | 0.988 | 231k | 0.692 | 0.171 |
  | 6-6 and 6-4 | 1,752 | 0.567 | 16k | 0.072 | 0.373 |

  Under sixes, holding slot 19 (the 5-4, no count) does not dim it (on 0.999 and 0.986). Holding
  another five-count off trump under fives (the 4-1) raises it slightly (on 0.174 against off 0.080).
- **Under sixes it dims with long trump.** On by trumps held: 0.999 (0–1), 0.997 (2), 0.986 (3),
  0.911 (4), 0.641 (5). Under fives it does not (0.93 at none, 0.98–0.99 at one to four).

**What it writes.** Its column of w₃ in canonical slots:
- **+7,048 on slot 20, the top off double**, the only large positive weight; the next is +997.
- −11,648 on slot 14, the second off double;
- −7,925 on slot 21, the lowest trump; −5,371 on slot 27, the trump double; trumps average −4,030.

In real tiles, the push leads the 6-6 under fives and the 5-5 under sixes, and holds back the
4-4 and trump. Under blanks–fours, where it is mostly shut, the same push would lead the 6-6 over
the 5-5.

**What it does (exploratory ablation, seen).**
- On leads with an off double legal, every layer-2 unit was ablated where it was on. The rows are
  pooled over holding the top double or not.

  | rows | base: top double led | ablate L2 1 | the next largest fall |
  |---|---:|---:|---|
  | fives (31,709) | 0.180 | 0.056 | L2 3, 0.181 → 0.111 |
  | sixes (11,034) | 0.176 | 0.053 | L2 3, 0.186 → 0.127 |

  Ablated, trump leads rise (fives 0.183 → 0.270, sixes 0.245 → 0.315), and slot-14 leads rise a
  little (0.178 → 0.212, 0.154 → 0.195).
- **Beside paper 4's unit.** L2 3 is on for 0.92–0.95 of leads under every declaration. Its write is
  +16,538 on slot 14 and +5,358 on slot 20. Ablated, it cuts off-double leads under every
  declaration (fives 0.801 → 0.200) and slot 14 to zero. So L2 3 leads an off double with a tilt to
  slot 14, which is the 5-5, a ten, under 0–4. L2 1 is on only where slot 14 is the plain 4-4, and
  it writes against slot 14 and for slot 20.
- On sixes follows, ablating it lowered count played at third seat (0.352 → 0.250) and last seat
  (0.350 → 0.259); the tile most often withdrawn was the 5-5. The holder split was not looked at.
- Under fives, follows are almost all off.
- In paper 8's published run (`l2u04/unit.json`), on partner-holds choice follows under fives and
  sixes, ablating L2 1 moved count played by −0.039 (on there 0.406), against L2 2's −0.367.
- Adding 1σ (245) on leads where it is off, pooled over declarations 0–4, cut trump leads 0.243 →
  0.199.

**Seen on another net (the smoke test).** The code was smoke-tested on seed 1's 16×8 net, on the
20k smoke rows. Seed 1 has no unit gated to fives and sixes. Its closest unit is L2 7: on 0.94 of
leads under 0–4 and sixes, 0.31 under fives, with −21,315 on slot 14 and +3,944 on slot 20.
- Seed 1's top-double vote under fives and sixes is shared: ablated where each is on, L2 0 −0.399,
  L2 6 −0.357, L2 7 −0.184.
- On its rows with both doubles legal, ablating L2 7 moved leads from the top double (0.101 → 0.000)
  to slot 14 (0.517 → 0.779).
- On its sixes follows, its adjusted coefficient for partner holding is −274k and for the top
  double's suit led +136k.
- Its formation went through a death (on 0.000 from step 10,000 to 15,270) and a rebirth.

All of that is seed 1, a differently organised net, not the unit read here. It informs predictions
1b and 2 and is recorded as seen. Seed 0's per-row-set ranks on top-double-holding rows, the
both-doubles rows, the sixes follows split, the formation and the twin are unseen.

**The working verb (to test, not a label).** "Lead the top off double, not trump, under fives and
sixes." In the net's frame the unit is a correction to paper 4's off-double switch: L2 3 leads an
off double and favours slot 14, which is the right double (the 5-5) under blanks–fours. Where slot
14 is the plain 4-4, L2 1 opens and moves the lead to slot 20, the 6-6 or, under sixes, the 5-5. It
also holds trump back. That is composition, two units with their own jobs, not packing. Its gate
reads the same pattern the correction needs: it opens where slot 14 holds no ten.

Two readings of the dims are open and are read off, not graded:
- the 6-4 dim under fives: holding the off ten, the net leads trump first (0.416 against 0.211);
- the long-trump dim under sixes: with four or five trumps, pull trump first.

On sixes follows, where it is on 0.29–0.61 of the time with mean z near 0, two verbs are open. Its
shape there (dial or detector) is not assumed.
- **A, "cash the top double":** it serves the 5-5 following its own suit (fives led), where the 5-5
  can take the trick;
- **B, "feed the ten":** it serves the 5-5 laid, while void, on partner's trick. Under sixes the
  5-5 is a ten.

The exploratory ablation saw the 5-5 withdrawn pooled over holders (third and last seat), so
"it moves the 5-5 on follows" is seen; which cell it moves is not.

## The loop, as registered

The code is `python -u -m lem.probe unit … 1 --double` (`lem/probe/unit.py`, `run_double`). It was
smoke-tested on seed 1, as above.
- **Top-double leads** are lead rows under fives or sixes where the top off double (slot 20) is
  held, so legal, and another tile is legal.
- **Both-doubles leads** are lead rows under fives or sixes where slots 20 and 14 are both legal and
  the unit is on.

1. **Label.** On rate, mean z and mean a by declaration × seat.
   - On leads, by declaration group: holds the top double × holds slot 19, by trumps held, by
     holding the boss trump, and the on-set's make-up (including the 0–4 leads where it is on).
   - On sixes follows: a regression of z on seat, trick, side, the top double's suit led (A) and
     partner holding (B), with the four cells beside it.
   - Cohen's κ and κ against a constant for the boundary `lead` (best v2, seen) and the v3
     candidates `lead ∧ slot 14 holds no ten` (¬`x:count:14`) and the same or any sixes follow.
   - The R² ladder on leads: declaration group, + holding slot 19 by group, + trumps held by group.
   - Both poles under read R.
2. **Circuit in.** Each layer-1 unit's mean contribution w₂·a₁ on leads under 0–4, 5 and 6, on
   fives leads with and without slot 19, and on follows under 0–4, 5 and 6.
3. **Circuit out.** Every layer-2 unit's write: slots 20, 14 and 19, the trump and off-double means,
   and its top and bottom real tiles under declarations 0, 4, 5 and 6. The untrained twin's on
   rate, on rate at leads by group, R² on declaration group, and write.
4. **Formation** (graded 4 below; the rest read off). At each of 15 checkpoints:
   - on rate, and on at leads under 0–4, 5, 6 and 5–6 pooled, on sixes follows, and on sixes follows with the
     5-5 legal;
   - the write on slot 20, slot 14 and trumps;
   - on the final net's top-double leads where it is on, the top double led and its Δ when
     ablated;
   - both poles.
5. **Steering.**
   - **Down:** ablate every live layer-2 unit, each on the same rows: top-double leads where unit 1
     is on (fives and sixes pooled, then apart). Read the top double, slot 14, any off double,
     trump, count and take, with deal-clustered intervals and unit 1's rank.
   - **Both doubles:** on both-doubles leads, the top-double and slot-14 lead rates at base, with
     each unit ablated, and with L2 1 and L2 3 ablated together.
   - **Sixes follows, the 5-5 legal**, where unit 1 is on, in three cells:
     - A: fives led;
     - B: void in the led suit, partner holds;
     - void in the led suit, an opponent holds.

     Unit 1 is ablated. The reads are the 5-5 played, taking and not taking, and count not taking,
     with deal-clustered intervals, every live unit's Δ, and unit 1's rank.
   - **Sixes choice follows** (a count tile and a non-count tile legal) where unit 1 is on, by
     holder: Δ count played for every unit and its rank, with count, not taking, read beside it.
   - **Up:** add k·σ (k = 0.25, 0.5, 1, 2), then a typical on value (the median on top-double
     leads), own − layer, on two row sets:
     - leads under 0–4 with the 6-6 legal, where it is off;
     - fives leads holding the 6-6 and the 6-4.

## Predictions, sealed before 1–5 are run on seed 0

**Seen before sealing** (above):
- the on rates by declaration and seat; the fives-leads table by 6-6 × 6-4; the sixes trump-length
  dim; L1 15's gate under 0–4 and L1 10 on follows;
- the write;
- the pooled ablation of every layer-2 unit on leads with an off double legal. **This settles 1a and
  1b by arithmetic** (the review's finding): only rows holding the top double can lead it, about
  9,049 of the 31,709 fives rows. So unit 1's Δ there is about −0.124 × 31,709 / 9,049 ≈ −0.43 (sixes
  ≈ −0.45). No other unit's whole seen fall can reach it on the subset;
- L2 3's write, on rates and pooled ablation;
- the sixes-follows ablation by seat, the 5-5 withdrawn pooled over holders;
- paper 8's sixes-and-fives feeding ranks (L2 1 −0.039 against L2 2 −0.367);
- the 1σ up on off leads under 0–4, pooled;
- seed 1's smoke run.

**Not seen on seed 0:** the joint ablation of L2 1 and L2 3, the sixes-follow cells with the 5-5
legal, the formation, the twin, and the circuit for the 6-4 dim.

**Confirmatory, reported, not graded:**
- **1a, size:** on top-double leads where it is on, ablating it lowers the top double led by at
  least 0.40 (seen by arithmetic: ≈ −0.43 / −0.45);
- **1b, rank:** it ranks first of the live layer-2 units on those rows (seen by bound);
- **the both-doubles direction:** at base the top double is led more than slot 14, and ablating
  unit 1 reverses that (mostly seen pooled, and on seed 1's L2 7).

**Graded:**

2. **The slot-14 lead it holds back belongs to L2 3 (composition, both doubles).** On both-doubles
   leads where unit 1 is on, ablating L2 1 and L2 3 together brings slot 14's lead rate to at most
   half of what ablating L2 1 alone gives, with the interval of their difference below 0. That is,
   the slot-14 lead that appears when unit 1 is removed is L2 3's push showing through. A miss means
   the slot-14 lead comes from elsewhere, and unit 1 is not a correction of L2 3 but a separate
   vote.
3. **On sixes follows it serves the 5-5 laid on partner's trick, B (causal, by cell).** With unit 1
   ablated, the fall in the 5-5 played is larger in cell B (void, partner holds) than in cell A
   (fives led), with B's interval below 0. A miss the other way, with A's interval below 0, favours
   A, "cash the top double". Both intervals containing 0 means it does not move the 5-5 by cell on
   these rows. A cell under 200 rows leaves 3 ungraded.
4. **The sixes-follow job came after the lead job (formation).** Let the lead step be the first
   checkpoint where on at fives-and-sixes leads first reaches half its final value. Let the offshoot
   step be the same for on at sixes follows with the 5-5 legal. The offshoot step is no earlier than
   the lead step. That is the essay's order for a branch grown from a main job. A unit that dies and
   is reborn is read from its last rebirth (the last checkpoint where its overall on rate is below
   0.01), and the page says so.

**Read off, not graded:**
- the sixes-follows regression (A's and B's coefficients), without intervals; unit 1's rank in each
  cell; count not taking;
- paper 8's sixes feeding comparison, re-measured on this sample with count not taking beside it;
- unit 1's Δ on every read where it is ablated, including trump led (seen in direction) and slot
  14;
- the up doses and the on value. Forced on under 0–4, the push's direction follows from the write;
  its size, and whether the net leads the 6-6 there, does not;
- the fives rows with the 6-6 and the 6-4, forced on;
- the 0–4 on-set's make-up;
- the circuit in, and which layer-1 units carry the 6-4 dim;
- the boundary κ. `lead ∧ slot 14 holds no ten` is near 0.78 by arithmetic from the seen rates, so
  it is reported, not graded;
- the R² ladder on leads;
- the rest of the formation (write, poles, the top double's use per checkpoint);
- the twin;
- the leads beside *Winning 42*'s guidance on leading off doubles (walkers) before or after pulling
  trump ([[book-authority]]).

**What would break the explanation.**
- 2 failing: the slot-14 lead it holds back is not L2 3's, so unit 1 is a separate vote rather than
  a correction of L2 3's tilt.
- 3 failing toward A: on follows it serves cashing the top double in its own suit, not feeding a
  ten.
- 4 failing: the sixes-follow job grew first or with the lead job, not as a later branch.

None would be read as mixing. A miss names what the unit does instead.

**Cost.** One run of under two minutes under `scripts/cpu-slot`, with a timeout of 900 s. Seed 1's
smoke on 20k rows took 4 s. The full sample is about 22 times larger, plus 8 units' ablations and
the up doses, plus 15 checkpoints.

## The review before sealing

One fresh-context Opus read the essay's full text (allow_large, to line 944), the draft,
`run_double`, papers 4 and 8 with paper 8's review, the card's "The work" and synthesis, and the
exploratory scripts and logs. Its points, and what changed:
- **Three of four graded predictions were already settled.**
  - 1a and 1b follow by arithmetic and a bound from the pooled ablation (≈ −0.43 / −0.45).
  - Prediction 4 (L2 2 feeds under sixes) was in paper 8's published `unit.json`.
  - Prediction 2's base and unit-1 direction were mostly seen.

  → 1a, 1b and that direction are moved to confirmatory, with the arithmetic shown. Old 4 is read
  off. Paper 8's row is listed as seen.
- **Prediction 2 did not test its heading** (the correction of L2 3). → It is now the joint ablation
  of L2 1 and L2 3, which `run_double` gained.
- **Prediction 3 ran over all sixes follows, had no interval, and mixed feeding with taking.**
  → It is rebuilt as a causal test on the 5-5, by cell (A: fives led; B: void, partner holds), with
  taking and not taking read apart and deal-clustered intervals. The regression is read off.
- **"Dial" was assumed on follows.** → Dropped. The shape there is not assumed.
- **The gate was named as a list of declarations.** → Named by its pattern, "slot 14 holds no ten"
  (L1 15's boundary is `x:count:14`). The v3 candidate is renamed to match.
- **L1 10, the largest closer on follows, was missing.** → Named, and a 0–4 follows row set is
  added to the circuit in.
- **The negative pole was dismissed.** → One line reads it as what the unit keeps out: a decided
  trick.
- **Small fixes.** → Trumps held is crossed with the declaration group in the R² ladder. A
  formation-order prediction is added (graded 4).
- **No tilt against the book**, and nothing reads the unit as mixed.

Sealed 2026-10-21.

## Results

The run of 2026-10-21 followed the registration: one `scripts/cpu-slot` run of 82 s
(1.84e12 instructions). The evidence (`unit.json`, the log, the code, the raw scripts, the run
script, `SHA256SUMS`) is `lem/unit-top-double-2026-10-21/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/a2566f4d7f8afe459ea1f89fc73deaf630eac107/lem/unit-top-double-2026-10-21).

**The picture.** Under fives and sixes, unit 1 is the net's **"lead the top off double"** unit, and
on sixes follows the same push **lays the 5-5 on partner's trick**.
- On leads where I hold the top double (the 6-6, or the 5-5 under sixes), ablating it drops that
  lead from 0.633 to 0.197. It ranks first of 8.
- Its gate reads one pattern in the net's frame: slot 14 holds no ten. That boundary scores Cohen's κ
  0.786, against 0.437 for the best v2 concept.
- It works against a group, not one unit. The slot-14 lead it holds back is pushed by L2 0, L2 3 and
  L2 4 together.
- Training gave it the slot-20 push and the gate. Its slot-14 brake was there at initialisation.

**Confirmatory (seen by arithmetic before sealing).**

| | claim | result | |
|---|---|---|---|
| 1a | top-double leads where it is on, ablated: falls ≥ 0.40 | −0.436 [−0.446, −0.428] from 0.633 (fives −0.433, sixes −0.446) | held |
| 1b | rank 1 of the live layer-2 units on those rows | rank 1; next L2 3 −0.229, L2 2 −0.144 | held |
| — | both doubles legal: base leads the top double more than slot 14, and ablating unit 1 reverses it | base top double 0.244, slot 14 0.447; ablated 0.001 and 0.771 | **base half missed** |

The base half was called "mostly seen". It was not. With both doubles in hand, the net leads the 4-4
first (0.447) more often than the top double (0.244). Unit 1 moves that choice by −0.243 and +0.323,
but it does not decide it.

**Graded.**

| | prediction | result | |
|---|---|---|---|
| 2 | both doubles: ablating L2 1 and L2 3 together brings slot 14 to at most half of L2 1 alone, difference below 0 | 0.040 against 0.771; Δ −0.730 [−0.753, −0.707] | **held** |
| 3 | sixes follows with the 5-5 legal: the 5-5's fall is larger in B (void, partner holds) than in A (fives led), B below 0 | B −0.571 [−0.604, −0.538] of 0.622; A −0.279 [−0.340, −0.223] of 0.870 | **held** |
| 4 | the sixes-follow job reaches half its final on rate no earlier than the lead job (read from the rebirth) | lead job at step 30,540 (0.92 of 0.971); follow job at step 152,700 (0.255 of 0.452) | **held** |

All three graded predictions held.

**What the results say.**
- **2.** The joint ablation held, but the read-off beside it widens the picture. Ablated alone, L2 0,
  L2 3 and L2 4 each take slot 14's lead on these rows to 0.000. Their writes on slot 14 are +14,696,
  +16,538 and +18,398. So the slot-14 lead is a vote shared by three layer-2 units, and unit 1 is
  the one unit that writes against it (−11,648) and for slot 20 (+7,048). "The correction of L2 3"
  was too narrow. It is the correction of the layer's slot-14 vote, which is the right double (the
  5-5, a ten) under blanks–fours.
- **3.** On sixes follows unit 1 serves both cells, and B is larger. In B the 5-5 is withdrawn almost
  entirely (−0.571 of 0.622), never by taking. In A, where the 5-5 follows suit, it falls by
  −0.279, −0.186 of that taking. Where an opponent holds and I am void, it falls −0.113 of 0.125.
  The regression agrees: the partner-holds coefficient is 633k, and fives led is 10k. So on follows
  the push is "lay the 5-5 where partner has the trick", which feeds a ten. Unit 1 ranks first in
  every cell.
- **4.** The unit was born on, at on 0.055 at step 0, with a write of −23,660 on slot 14 and −683 on
  slot 20. It died from step 300 to step 15,270. At step 30,540 it came back as the unit read here:
  - on 0.90 at fives leads, 0.97 at sixes leads, 0.03 elsewhere;
  - slot 20 at +8,155;
  - ablated, top-double leads fall −0.415.

  The lead job and its gate came back together. The sixes-follow job stayed near 0.03 until step
  76,350 and grew between steps 152,700 and 305,400 (0.255 → 0.487): a later branch, as the
  essay's account of growth has it. Before the death, at step 30, sixes follows were on 0.27 against
  0.14 at leads. That early state is read but not counted.

**Read off.**
- **Feeding under sixes, re-measured** (the demoted prediction). On sixes choice follows where unit 1
  is on and partner holds, ablating it lowers count played by −0.218 [−0.231, −0.204], count not
  taking −0.214. It ranks first; L2 2 is −0.125. Where an opponent holds, L2 2 ranks first (−0.220)
  and unit 1 second (−0.091). The demoted prediction would have missed on these rows. Paper 8's
  rows (fives and sixes pooled, where unit 4 is on) gave L2 1 −0.039. On sixes rows where unit 1 is
  on, unit 1 is the main feeder.
- **Forced on.** Adding the on value (417) where it is off raises the top-double lead beyond the
  layer:
  - under blanks–fours with the 6-6 legal: +0.585 [0.579, 0.592] from a base of 0.315; slot 14
    −0.085; trump −0.170;
  - on fives leads holding the 6-6 and the 6-4: +0.443 [0.420, 0.467] from 0.072.

  The doses rise monotonically: +0.043, +0.134, +0.458, +0.628 at k = 0.25, 0.5, 1, 2 under 0–4.
- **Ablated on top-double leads, every read.** Trump leads +0.169, off doubles −0.170, slot 14
  +0.049. Under sixes count falls −0.437, because the 5-5 is a ten.
- **The 6-4 dim under fives** is carried by L1 12, which is active only under fives (−164k when I
  hold the 6-4, −51k when I don't), and L1 14 (−129k against −57k). L2 2 writes +16,364 on slot 19,
  the 6-4 under fives.
- **The gate.**
  - On leads, L1 15 closes it under blanks–fours (−1,033k).
  - On follows, L1 10 does (−11,764k under blanks–fours, 0 under fives, −3k under sixes). That is why
    sixes follows alone are open.
  - Its 0–4 on-leads (0.05–0.09) are mostly hands holding the 5-5 (0.79). There it holds the 5-5 lead
    back.
- **Boundary.** `lead ∧ slot 14 holds no ten` scores Cohen's κ 0.786 and κ against a constant 0.692,
  as the arithmetic said. Adding any sixes follow scores 0.769. The best v2 concept, `lead`, scores
  0.437.
- **R² on leads.** Declaration group 0.588; + holding slot 19 by group 0.595; + trumps held by group
  0.625.
- **Poles.** + (top 1%): all fives or sixes (2,785 and 1,539 of 4,345; 21 fours), 0.63 leads.
  − : all blanks–fours, 0.70 last seat and 0.24 second seat: following into a decided trick.
- **The twin** is on 0.055, with R² 0.071 on the declaration group. Its write is the step-0 write,
  with slot 14 at −23,660. The brake was there before training.

**Against the frame.**
- The unit is clean in the essay's sense: one push in the net's frame (lead or lay the top off
  double, not the second, not trump) behind one gate (slot 14 holds no ten).
- Its two jobs are one push read in two places. On a lead it chooses the double. On a sixes follow,
  void on partner's trick, it lays the 5-5, which is a ten. The follow job grew later, as a branch.
- With the layer it is composition: three units vote for slot 14, the right double under 0–4, and
  one unit, opened where slot 14 holds no ten, votes for slot 20.
- **The vocabulary gap is named.** List v2 has `trump-is-fives` and `trump-is-sixes` but no word for
  "slot 14 holds no ten", or, in the game's own terms, "the second off double is not a count double".
  The v3 candidate is that word: κ 0.786.
- Not read as mixing.
- **Beside the book.** With both doubles in hand the net leads the 4-4 before the 6-6 or 5-5 (0.447
  against 0.244). It also leads trump first when it holds the 6-4 under fives. *Winning 42*'s
  guidance on leading doubles and walkers before or after pulling trump is the authority on reasons
  ([[book-authority]]). These are the net's choices, read beside the book, not weighed against it.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## Links

[[clean-neurons-42]] · [[lem-unit-lead-doubles-2026-10-20]] · [[lem-unit-count-on-it-2026-10-21]] ·
[[lem-unit-big-double-2026-10-20]] · [[lem-unit-lead-brake-2026-10-20]] · [[book-authority]]
