---
title: "Unit paper 8: the 16×8 seed-0 net's layer-2 unit 4, a dial on feeding the trick (Lem, #235)"
kind: experiment
first_seen: 2026-10-21
last_updated: 2026-10-21
status: complete
---

The eighth **unit paper** of [[clean-neurons-42]]'s new mode (#235), on the 16×8 seed-0 field
net. Layer-2 unit 4 is the next gap in the #236 audit's boundary column (#202). Its best list-v2
on/off match is `x:bid:0` at 0.987, which is exactly what a constant scores (κ 0.00). The unit is
on for 0.987 of rows. Read R labels both of its poles clean:
- L2+004 is `partner-winning` (1.000);
- L2−004 is `trump-led` (0.965).

The question is what the unit does across the 98.7% of rows where it is on, and why it shuts in
the 1.3% where it doesn't. Its neighbours are paper 6's unit, "not partner's trick; keep your count
off it" ([[lem-unit-theirs-2026-10-20]]), and paper 7's unit ([[lem-unit-pull-trump-2026-10-20]]),
which writes into this one with weight +241.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## The label, read raw first (exploratory, before registration)

The scripts are in `scratch/lem/unit-papers-2026-10-20/l2u04/`: `raw.py`, `follow.py`, `write.py`
and `smear.py`. No label was imposed first.

**Where it shuts.**
- Never on a lead. Off rate by seat: second 0.002, third 0.036, last 0.023.
- 0.949 of off rows follow a trump lead. Off on trump-led follows is 0.067, against 0.0014 on
  other follows.
- On trump-led follows, off depends on who holds the trick:
  - partner holds it: 0.000;
  - LHO holds it: 0.084;
  - RHO holds it: 0.103.
- The densest cell is third seat, with RHO holding partner's trump lead. Off there is 0.48 on the
  bidding side and 0.365 for defenders (mean z 20k and 34k). Among defenders in that cell, off
  falls with trick number, from 0.744 in trick 1 to 0.133 in trick 6.
- Off is higher when I hold the boss of the led suit (0.131 against 0.047 on trump-led follows).
- By declaration: fours 0.024, sixes 0.000, the others 0.008–0.010.
- No v2 concept draws the line. The best Cohen's κ for off is `holds-boss-of-led` at 0.165, then
  `trump-led` at 0.103.

**What it is while on: a dial on who holds the trick.** Mean z over follows (k = thousands):

| seat, holder | not trump led | trump led |
|---|---:|---:|
| second, RHO | 442k | 313k |
| third, partner | 536k | 450k |
| third, RHO | 285k | **22k** |
| last, LHO | 328k | 206k |
| last, partner | 700k | 619k |
| last, RHO | 286k | 157k |

R² of z on follows:
- winner: 0.356;
- winner + seat: 0.509;
- adding trump led: 0.622;
- adding declaration: 0.653;
- adding trick number: 0.659.

So the off-set is the bottom of a graded dial. The dial is high when partner holds the trick and
low when an opponent does. A trump lead lowers it further.

**What it writes: count, under some declarations.** Its column of w₃, read in real tiles per
declaration:

| declaration | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean write on count tiles | +3,112 | +5,471 | +5,845 | +5,845 | +8,794 | −127 | +297 |
| mean write on other tiles | +10 | −502 | −584 | −584 | −1,225 | +715 | +622 |

- The largest single weight is on the 5-5, at +18,398. It is the 5-5 under declarations 0–4,
  where the 5-5 is a 10-count. Under 5 and 6 the same canonical slot holds the 4-4, which is not
  count.
- The heaviest negative weight is the boss trump, at −7,032.
- **In the net's own frame (canonical slots), the write is the same under every declaration**:
  hold the top trump back, and lay slot 14, the low trump 6-1, and slots 10 and 8. Under
  declarations 0–4 those slots carry the 5-5, the 5-0 and the 3-2 (and under fours the 4-1), so
  the push lands on count. Under 5 and 6 the same push lays the 4-4 and the 4-0, which carry no
  count. "Count" is what the canonical push means under 0–4, not a second job.

**What it does (exploratory steering, seen).** The rows are following rows where a count tile and
a non-count tile are both legal.
- Ablated where it is on and partner holds the trick, count played falls from 0.676 to 0.320. That
  is pooled over declarations, on the unit's own on-rows. On the same kind of rows, L2 2 ablated
  falls 0.728 → 0.312.
- Up by 1σ where an opponent holds a trump lead, count played rises 0.177 → 0.278 and taking falls
  0.214 → 0.181. That is own minus base, not against the layer.
- Adding a typical on value (783) on off rows: take 0.349 → 0.237, count 0.115 → 0.209.

**Seen on another net (the smoke test).** The code was smoke-tested on seed 1's 16×8 net, whose
unit 4 is dead, using its L2 0 on the 20k smoke rows. That unit has the same poles: on 0.992, +
`partner-winning`, − `trump-led`. Its write is count-aligned under declarations 5–6 instead
(count +7,929 and +4,548). Ablating each unit on the same rows split the count smear by
declaration:
- 5–6: L2 0, −0.404;
- 1–4: L2 2, −0.423;
- 0: L2 3, −0.321.

Its formation went through a near-death: on 0.002 at step 1,000, then reborn. A second smoke
run, after the review, showed more of seed 1's unit. In third seat with RHO holding a trump lead,
its dial dips where I can take the trick (mean z 67k, off 0.34) and stays up where I cannot
(315k, off 0.02). Its adjusted trumped-in / trump-led ratio is 0.58.

All of that is seed 1, not the unit read here, but it informs predictions 1, 2 and 6. Seed 0's
per-declaration steering, near misses and formation are unseen. A unit per declaration group
would be composition, separate specialists reading one shared dial, not packing.

**Two working verbs (to test, not labels).** The unit is a dial, not a gate. Its low end is the
bottom of a graded quantity (the R² ladder above), not a second pole that sends flow elsewhere.
- **A, "put count on it".** The dial reads how welcome count is on this trick: high when partner
  holds it, low when an opponent's trump does. Its push lays count under declarations 0–4.
- **B, "feed, don't take".** The dial reads whether to feed this trick or to win it back. It is
  low where an opponent holds the trick and I can take it, and its push holds the boss trump back
  and lays a side tile. This fits what was seen:
  - the boss trump is the heaviest negative write;
  - off is higher when I hold the boss of the led suit;
  - the densest off cell is RHO holding partner's trump lead, where I can often overtrump;
  - raising the unit on off rows cuts taking;
  - seed 1's dip sits where I can take.

Both verbs predict the same thing where partner holds the trick. They part where an opponent
holds it: A says the dip is about an opponent's trump on the trick, and B says it is about my
being able to take it. B does not depend on the declaration in canonical slots, and the count
reading is what B's push means under 0–4. Prediction 6 separates them, and prediction 3 is read
with B in mind (below).

These can lose:
- The unit may not carry the feeding decision under declarations 1–4. It might only correlate
  with a unit that does.
- The split by declaration may be a weight artefact with no use behind it.
- The dip may be about a trump being led (a noun about the trick's suit), with neither verb.

## The loop, as registered

The code is `python -u -m lem.probe unit … 4 --count` (`lem/probe/unit.py`, `run_count`). It was
smoke-tested on seed 1, as above.
- **Choice follows** are following rows where a count tile and a non-count tile are both legal.
- **Holds** means the open trick's current winner (`winning` block).
- **Held by a trump** means the winning tile is a trump.

1. **Label.** Graded in mean z, the dial. Mean a, what the readers see, is reported beside it.
   - Mean z and off rate for each seat × holder × kind of trick. The three kinds are: not trump
     led and held by a non-trump; not trump led and held by a trump (trumped in); trump led.
   - The same by declaration.
   - The third-seat RHO-holds-a-trump-lead cell, by trick, side and holding the boss of the led
     suit.
   - The off-set's make-up.
   - **Near misses:** for third and last seat, N, T and R are mean z where **RHO** holds the
     trick. At third seat RHO is the only opponent who has played. At last seat only RHO can have
     trumped in on LHO's non-trump lead, so LHO-held rows are left out of all three.
     - N: not trump led, held by a non-trump;
     - T: trump led;
     - R: not trump led, held by RHO's trump.

     Each cell is also split by `can-take-it`. An adjusted version regresses z on seat, trick,
     side and declaration plus the trump-led and trumped-in indicators, over third- and last-seat
     rows held by RHO, and reports the ratio of their coefficients. Partner can hold with a trump
     on a non-trump lead only at last seat, so that near miss is last seat only, against N − T at
     last seat.
   - The R² ladder on follows. It adds a winner × trump-led term, reproducing the explored 0.622 /
     0.653, and a holds-with-a-trump term, which differs from what was explored. It also scores
     the list-v3 candidate `trick-holder-kind` × seat (partner; opponent; opponent with a trump).
   - Cohen's κ, and κ against a constant, for each candidate boundary.
   - Both poles under read R.
2. **Circuit in.** Each layer-1 unit's mean contribution w₂·a₁, by kind of trick.
3. **Circuit out.** The write per declaration in real tiles (count mean against non-count mean,
   top and bottom tiles) for every layer-2 unit, and the untrained twin's write.
4. **Formation** (read off; see below). At each of 15 checkpoints:
   - on rate;
   - R² of z on the holder;
   - partner minus opponent;
   - the trump-lead subtraction (N − T, third and last seat);
   - on rate in the third-seat RHO-trump-lead cell;
   - the d1–4 write's count alignment;
   - both poles.
5. **Steering** on choice follows. The reads are count played, count played without taking (a
   count tile can be a trump under declarations 4 and 5), take, and the boss trump played.
   - **Up:** add k·σ (k = 0.25, 0.5, 1, 2), then a typical on value: the median activation on
     partner-holds choice follows. Read on four row sets:
     - partner holds;
     - an opponent holds;
     - an opponent holds a trump lead;
     - the unit is off.

     Each is read own − layer, where the layer is the mean of the other live layer-2 units under
     their own dose of the same kind.
   - **Down:** ablate every live layer-2 unit on the same rows (the rows where unit 4 is on) for
     each declaration group (1–4, 5–6, 0, and 4 apart from 1–3) × holder (partner, opponent).
     Report:
     - each unit's Δ in count played, with a deal-clustered interval;
     - unit 4's Δ on every read;
     - unit 4's rank;
     - which count tiles are withdrawn.

     The other layer-2 units' writes by declaration were not looked at before sealing.

## Predictions, sealed before 1–5 are run on seed 0

**Seen before sealing** (above):
- the seat, holder, trump-led and declaration off rates; the cell means of z; the R² ladder;
- the raw-input and concept differences; the write by declaration;
- the pooled exploratory steering;
- seed 1's smoke run, including its per-declaration division of labour and its formation.

**Not seen on seed 0:**
- the steering by declaration group and its ranks;
- the near-miss cells (R, and partner holding with a trump);
- the formation; the candidate boundary κ beyond the best v2 one; the twin.

1. **It carries the feeding vote where its write lands (down, declarations 1–4).** The rows are
   choice follows in declarations 1–4 where partner holds the trick and unit 4 is on.
   - **1a, size (confirmatory):** ablating it lowers count played by at least 0.45, with its
     interval below 0. The layer after it is linear and its write was seen, so this checks size,
     not direction. Declaration 4 is about two thirds of these rows; 4 and 1–3 are reported apart.
   - **1b, rank (the real test):** it ranks first (most negative) of the live layer-2 units
     ablated on the same rows. Seen against it: pooled over declarations, L2 2 fell further on its
     own rows (−0.416). On seed 1, L2 2 carried 1–4.
2. **The vote is divided by declaration (down, 5–6).** On the same kind of rows in declarations
   5–6, another live layer-2 unit's Δ in count played is more negative than unit 4's. A miss would
   mean 5–6 have no single feeding unit in this net, not that unit 4 carries them. (Unit 4's own
   |Δ| there follows from its write and is read off.)
3. **An opponent's trump on the trick, not the suit led (near miss, adjusted).** Over third- and
   last-seat follows held by RHO, the adjusted coefficient for "RHO trumped in" is at least half
   the coefficient for "trump led". The raw (N − R) / (N − T) per seat is reported beside it, and
   a cell under 200 rows leaves that seat's raw ratio ungraded.
   - Read with B: in trumped-in rows I usually cannot take the trick back. So a miss here favours
     B ("I can take it") over a noun about the suit led. It is read against prediction 6 before
     the noun is considered.
4. **Partner's trump does not lower it (near miss, last seat).** At last seat, partner holding
   with a trump on a non-trump lead and partner holding with a non-trump differ in mean z by at
   most 0.25 of N − T at last seat, in absolute value.
5. **The dial has a list-v3 word.** The graded candidate `trick-holder-kind` × seat (partner;
   opponent without a trump; opponent with a trump) explains at least 0.59 of the variance of z on
   follows. That is 0.08 above winner + seat (0.509, seen). Its gain over the trump-led version
   (0.622, seen) is reported, not graded.
6. **B: the dip sits where I can take it (near miss, separates the verbs).** At third seat with
   RHO holding, the trump-lead dip where I cannot take the trick ((N − T) among `¬can-take-it`
   rows) is under half of the dip overall. Seen on seed 1 in direction, for the T cells only.
   A miss favours A: the dip holds where I cannot take.

**Read off, not graded:**
- the up doses and the on value. Their direction on off rows was seen;
- unit 4's own Δ under 5–6 (it follows from its write), and its Δ take and Δ boss trump in every
  group. If they move under 5–6 as under 1–4, the verb does not depend on the declaration;
- the off rate by `can-take-it` (seen in `follow.py`: 0.106 can, 0.056 cannot, on trump-led
  follows);
- the 5-5's share of the count tiles withdrawn under 1–4. **Set in advance:** above 0.7, the label
  narrows to "put the 5-5 on it";
- d0, whose write is half-aligned;
- the withdrawn tiles;
- the formation. Seed 1's unit died near step 1,000 and was reborn, so an order fixed in advance
  would be ill-posed, as paper 7's review found;
- the circuit in and out; the twin;
- the side and trick crosses against *Winning 42*'s guidance on smearing count to partner and
  keeping it off the opponents' tricks ([[book-authority]]).

The κ against a constant (#202's 0.00) is reported, not graded, and so is the Cohen's κ of the
bit `opponent holds with a trump`. Off is 1.3% of rows. Any bit much commoner than that scores
below a constant, and the base rates already seen cap that bit's Cohen's κ near 0.15. For a dial,
the on/off boundary is the wrong instrument (paper 7's lesson). Prediction 5 scores the graded
candidate instead.

**What would break the explanation.**
- 1b failing: unit 4 is not the main feeding unit under 1–4. It shares the vote, or the vote
  lives in a unit it correlates with.
- 2 failing: the write split is unused in play, and the feeding unit is shared.
- 3 failing with 6 holding: the dip is about my taking it back (B), not about an opponent's trump.
  3 and 6 both failing: the dip is the suit led, a noun about the trick.
- 4 failing: any trump on the trick lowers the dial.
- 5 failing: the dial needs more than who holds the trick and with what.
- 6 failing: the dip holds where I cannot take, which favours A.

None would be read as mixing. A miss names what the unit reads instead.

**Cost.** One run of a few minutes under `scripts/cpu-slot`, with a timeout of 900 s. Paper 7's
layer-1 run took 117 s. This one adds per-group ablations for 8 units and 15 checkpoints.

## The review before sealing

One fresh-context Opus read the essay's full text (allow_large, to line 944), the draft,
`run_count`, paper 7's page, the card's "The work" and synthesis, and the raw scripts. Its points,
and what changed:
- **The verb was named after our concept, not the write.** → The write is described first in the
  net's canonical frame (hold the top trump back, lay slots 14, 22, 10, 8), the same under every
  declaration. "Count" is what that push means under 0–4.
- **"Gate" and "negative side" were used for a dial.** → "Dial" throughout. The sentence on the
  negative side was dropped.
- **Prediction 5 (a κ for a bit) was the probe temptation, and the seen base rates fixed it near
  0.15.** → Demoted to read-off. The graded v3 candidate is the dial word, scored by R² with a
  bar set in advance. The ladder now reproduces the explored terms, and the new term is named.
- **The cells averaged z while the readers see a.** → Mean a beside mean z, and z graded.
- **1 was nearly settled by the seen write, and mostly a fours test.** → Split into 1a, size
  (bar raised to 0.45, called confirmatory), and 1b, rank (the real test). 4 and 1–3 are reported
  apart.
- **2a was mechanical.** → Read off. 2b is kept as prediction 2.
- **Bug: 4's row sets mixed seats.** Partner can hold with a trump on a non-trump lead only at
  last seat. → 4 is last seat only, with an absolute difference against last seat's N − T.
- **Bug: 3 mixed holders at last seat.** → N, T and R are all RHO-held. Graded on an adjusted
  coefficient ratio (trick, side, declaration), with the raw ratio beside it. Small cells are
  ungraded.
- **"Count played" mixed feeding with taking.** → A "count, not taking" read is added.
- **Verb B ("feed, don't take") passes the same tests, and a miss on 3 would be misread under
  it.** → B is named beside A, prediction 6 (`can-take-it`) separates them, and 3's outcome is read
  with B. The second smoke on seed 1 favoured B's direction. That is recorded as seen.
- **The 5-5 may be the whole count vote.** → The 5-5's share of withdrawn count tiles is read off
  with a threshold set in advance (0.7).
- **Small fixes.** → `d5-6` is now `decl ∈ {5, 6}`; every read is recorded for the ablations; the
  `can-take-it` off rates are reported as seen.
- **"Division of labour" could read as spreading.** → One line: a unit per declaration group is
  composition, separate specialists, not packing.
- **No tilt against the book**, and nothing reads the unit as mixed.

Sealed 2026-10-21.

## Results

The run of 2026-10-21 followed the registration: one `scripts/cpu-slot` run of 76 s
(1.67e12 instructions). The evidence (`unit.json`, the log, the code, the raw scripts, the run
script, `SHA256SUMS`) is `lem/unit-count-on-it-2026-10-21/` in
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/f0ac72ebde7ff19b7e161d16f30fe31266fd55a1/lem/unit-count-on-it-2026-10-21).

**The picture.** Under ones through fours, unit 4 is the net's main **"feed partner's trick"**
unit. It is a dial on whose trick it is, and an opponent's trump on the trick lowers it.
- Where partner holds the trick, ablating it drops count played from 0.669 to 0.086.
- No other layer-2 unit moves count on those rows.
- Under blanks (declaration 0), whose write is only half aligned, it is still the only feeder, but
  smaller: −0.242. *Corrected 2026-10-21: the first version of this page said "blanks through
  fours" for the −0.583, which is declarations 1–4 (Media caught it).*
- Its 1.3% off tail is where an opponent's trump holds the trick: 0.998 of off rows.
- Whether I could take the trick back barely matters, so verb A stands and verb B lost.
- Under fives and sixes, the same push lands on non-count tiles, and L2 2 does the feeding.

**Graded.**

| | prediction | result | |
|---|---|---|---|
| 1a | ablated, partner holds, declarations 1–4: count played falls ≥ 0.45 | −0.583 [−0.589, −0.575] from 0.669 (4: −0.621; 1–3: −0.504) | **held** (confirmatory) |
| 1b | that Δ ranks first of the live layer-2 units on the same rows | rank 1 of 8; the next most negative is −0.003 | **held** |
| 2 | under 5–6, another unit's Δ is more negative than unit 4's | L2 2 −0.367 against unit 4 −0.091 | **held** |
| 3 | adjusted coefficient, trumped in / trump led ≥ 0.5 | 0.569 (−143k / −251k); raw third seat 0.418, last 0.763 | **held** (adjusted) |
| 4 | last seat: partner with a trump vs without, \|Δz\| ≤ 0.25 of N − T | 0.265 (675k against 729k) | **missed, narrowly** |
| 5 | `trick-holder-kind` × seat explains ≥ 0.59 of z on follows | 0.640 | **held** |
| 6 | B: at third seat, the trump-lead dip where I cannot take is under half the dip overall | 0.894 (can take: 1.173) | **missed: B loses to A** |

So 1, 2, 3 and 5 held. 4 missed narrowly, and 6 missed.

**What the misses say.**
- **6.** At third seat with RHO holding partner's trump lead, the dial falls almost as far where I
  cannot take the trick (z 327k → 42k) as overall (341k → 22k). It falls a little further where I
  can (349k → −25k). The dip is about an opponent's trump holding the trick, not about my chance
  to win it back. Holding the boss of the led suit still deepens it (off 0.63 against 0.40), so
  taking back is a small extra term, not the variable. Seed 1's smoke showed B's direction on its
  own unit, and that did not carry over to seed 0. Two seeds' units can differ in this term.
- **4.** Partner holding with a trump lowers the dial by 54k at last seat. That is 0.265 of N − T,
  just over the bar. Any trump on the trick costs the dial a little, and an opponent's trump costs
  it about four times as much.
- **3** held only when adjusted. At third seat the raw ratio is 0.418, because trumped-in rows
  there come later in the hand and from the other side than trump leads. The adjusted ratio
  controls for that.

**Read off.**
- **The push is declaration-free in the net's frame. What it hits is not.**

  Unit 4's own Δ when ablated where partner holds:

  | declarations | count | count, not taking | boss trump played |
  |---|---:|---:|---:|
  | 1–4 | −0.583 | −0.528 | +0.043 |
  | 5–6 | −0.091 | −0.146 | +0.079 |
  | 0 | −0.242 | | |

  The boss trump is released under every declaration. Count is withdrawn only where count sits
  in its slots.
- **The count tiles withdrawn** (partner holds, 1–4), by canonical slot: slot 10 (the 5-0) 0.313,
  14 (the 5-5) 0.219, 8 (the 3-2) 0.184, 22 (fours' 4-1) 0.125. The 5-5's share is 0.219, under the
  0.7 bar set in advance, so the label stays "count", not "the 5-5".
- **Forced on.** Adding the typical on value (1,098) raises count played beyond the layer on every
  row set, and most on the rows where it was off: count +0.407, take −0.162, boss trump −0.188
  (own − layer). Forced on, the net feeds the trick instead of taking it back.
- **The division of labour** (composition, not packing):
  - **L2 2:** its write is count-aligned under 5 (+6,403 against −3,485) and against count under
    0–4. Ablated, it drops feeding under 5–6 (−0.367).
  - **L2 0:** its write runs the other way, count under 0–4 and against it under 5–6. Ablated
    under 5–6 where an opponent holds, count played rises +0.495, so it works as a brake there.
  - Seed 1 split the same job across its own units by declaration group (smoke).
- **Circuit in.** L1 8 ("trump still out against me", paper 7) gives the largest steady push
  (+194k to +211k).
  - L1 2 (`partner-winning`) gives +125k where partner holds, +26k on a trump lead and +2k when
    trumped in.
  - L1 11 and L1 15 (`winner-cannot-be-beaten`) subtract hardest on trump leads (−90k and −60k).
  - So the dial is partner holding, minus an opponent's trump holding, fed by the trump count
    behind it.
- **The tail cell.** At third seat with RHO holding a trump lead, off falls with trick number from
  0.528 in trick 1 to 0.167 in trick 6. It is 0.48 on the bidding side and 0.37 for defenders.
- **The boundary.** The bit `opponent holds with a trump` scores Cohen's κ 0.125. The narrower
  `opponent holds a trump lead, third seat` scores 0.506. The κ against a constant is below 0 for
  every bit, as expected for a 1.3% tail. The dial word (prediction 5) is where the vocabulary
  fits: R² 0.640 against 0.622 for the trump-led version.
- **The twin.** The untrained twin's unit 4 is on 0.967 of rows, with R² 0.042 on winner + seat
  (trained: 0.509).
- **Formation.**
  - **Near-death, as on seed 1.** The unit fell to on 0.045 at step 300 and 0.024 at step 1,000.
  - **Reborn on partner.** At step 1,000 its + pole is `partner-winning` (0.75), and by step 3,000
    it is 0.99. At step 3,000 R² on the holder is 0.285.
  - **Its write was count-aligned under 1–4 from initialisation** (+4,659 at step 0) and grew to
    +7,000–9,000.
  - **The trump tail came late.** The − pole became `trump-led` only at step 152,700. Off in the
    third-seat RHO-trump-lead cell rose from 0.12 at step 3,000 to 0.53 at the end.
  - So the dial came first and the tail later, as in papers 6 and 7, after a death and rebirth as
    in paper 3.

**Against the frame.**
- The unit is clean in the essay's sense: one dial (who holds the trick, and with what) and one
  push (feed it, hold the top trump back).
- "Count" is what that push means under the declarations whose canonical slots carry count. The
  net covers the other declarations with another unit: composition, not packing.
- The off-set is the bottom of the dial, not a second meaning. Its κ 0.00 in #202 was the wrong
  instrument, a bit against a 1.3% tail.
- **The vocabulary gap is named.** List v2 has `partner-winning` and `trump-led` as bits but no
  graded `trick-holder-kind` (partner; opponent; opponent with a trump). The v3 candidate is that
  word, R² 0.640.
- Not read as mixing.
- **Beside the book.** *Winning 42*'s advice on when to put count on partner's trick and keep it
  off the opponents' is the authority on reasons ([[book-authority]]). The side and trick crosses
  above are read beside it, not weighed against it.

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## Links

[[clean-neurons-42]] · [[lem-unit-theirs-2026-10-20]] · [[lem-unit-pull-trump-2026-10-20]] ·
[[lem-unit-lead-doubles-2026-10-20]] · [[book-authority]]
