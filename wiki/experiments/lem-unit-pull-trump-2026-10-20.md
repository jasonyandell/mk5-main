---
title: "Unit paper 7: the 16×8 seed-0 net's unit 8, on almost everywhere and shut on leads where trump is settled (Lem, #235)"
kind: experiment
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---

The seventh **unit paper** of [[clean-neurons-42]]'s new mode (#235), on the 16×8 seed-0 field
net. Unit 8 is the deepest vocabulary gap in the #236 audit's boundary column (#202): the list-v2
concept that best matches its on/off is `¬lead-boss-trump`, at 0.909 against a constant's 0.907
(κ 0.02). Read R labels its poles L1+008 `no-trump-held` (partial, 0.894) and L1−008 `x:count:39`
(partial, 0.869). The unit is on 0.907 of rows, so what it does lives in the rare rows where it
switches off. This paper reads them. Its neighbours on the lead are papers 3–5:
[[lem-unit-lead-brake-2026-10-20]], [[lem-unit-lead-doubles-2026-10-20]] and
[[lem-unit-no-trump-yet-2026-10-20]].

Scope: this net, these inputs (one position, uniform beliefs, no history), list v2 plus the raw
inputs. What we learned from this model, not all of 42, yet.

## The label, read raw first (exploratory, before registration)

Scripts and notes are in `scratch/lem/unit-papers-2026-10-20/u08/` (`raw.py`, `lead.py`,
`NOTES.md`). No label was imposed first.

- **The off-set is a lead set.**
  - Off rate by seat: lead 0.234, second 0.005, third 0.032, last 0.015.
  - By play count, leads (n_plays ≡ 0 mod 4) run 0.20–0.28, and every other play is ≤ 0.07.
  - 0.876 of off rows are leads, against 0.291 of on rows.
- **Within leads, off grows with trumps held:**

  | trumps in hand | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | off, on lead | 0.114 | 0.217 | 0.313 | 0.443 | 0.895 | 1.0 | 1.0 |
  | rows | 53,663 | 48,463 | 32,729 | 11,949 | 2,802 | 374 | 20 |

- **Other inputs rise on off-leads:**
  - the 6-5, `x:hand:26`: 0.386 against 0.106;
  - the 6-4, `x:hand:25`: 0.29 against 0.14;
  - the sixes declaration's count bits, `x:count:20`, `25`, `43`: 0.336 against 0.034;
  - trumps gone, `x:gone:21`–`24`: about 0.15 higher.
- **Off by declaration** (all rows): sixes 0.326, against 0.046–0.099 for the others.
- **The best v2 concept on leads alone** is `trump-is-sixes`: it agrees with on/off 0.818 of the
  time, against 0.766 for a constant. Next come `¬trump-out` (0.796) and `trumps-3plus` (0.774).
  None draws the line.

**Seen by accident, before sealing.** The second smoke test was run with `--formation` pointed at
seed 0's checkpoints instead of seed 1's. That printed seed-0 unit 8's growth on the 20k smoke rows:
- the lead term, the trump term and off-on-lead at each checkpoint;
- the final R² on leads: trumps still out in other hands 0.524, trumps held 0.411.

Both are therefore seen. Formation and the held-against-out comparison are read off, not graded.

**Two candidate verbs.** The unit is on almost everywhere, so its job is its vote while on, and
the leads where that vote is withdrawn.
- **A, a brake on leading trump.** On, it votes against leading trump. On a lead from trump length
  it shuts, and the net may pull trump.
- **B, "trump is still contested".** On while trumps are spread among hands. It shuts when trump
  is settled: I hold most of what is left, or what is left is gone. This covers the off-leads with
  0–1 trumps late in the hand. Half the off-leads have 0–1 trumps (0.47 by the staircase above),
  which A alone does not explain. The seen R² (out ahead of held) leans to B.

Both are verbs about the lead, and both make the same use prediction: forced on, fewer trump
leads. They part on what the off state is for when there is no trump to lead, which is read off
here (where the pick goes: trump, an off double, or count). The variable is graded, so the label
is read in mean z as well as off rates.

The negative side of a mostly-on unit is the gate in the essay's sense: it marks where the vote is
let through, not an opposite. Read R's + pole `no-trump-held` is where leading trump is impossible
and the vote costs nothing; that fits both verbs. The − pole `x:count:39` is a declaration-specific
count bit, and the − pole's sixes share is read off to check it.

These can lose:
- The unit may not touch whether the net leads trump.
- The staircase may be trick number, the bidder's side or the sixes declaration, not trump.
- The vote may be about trump in general (trump scarcity, a noun) rather than the lead.

## The loop, as registered

The code is `python -u -m lem.probe unit … 8 --lead` (`lem/probe/unit.py`, `run_lead`). It was
smoke-tested on 16×8 seed 1. "Trumps held" counts the canonical trump slots 21–27 in hand. The
boss is slot 27 (the 6-6) and the 6-5 is slot 26. The audit sample has only pip-trump
declarations (0–6).

1. **Label.** Off rate and mean z on leads:
   - by trumps held, by trumps still out in other hands, and held × out;
   - **the 0–1-trump off-leads**: their share of off-leads, by trick number, by trumps gone and by
     declaration;
   - **by trumps held within controls**: no boss, not sixes, tricks 1–2 apart from tricks 3+;
   - by trumps held × the boss; × the 6-5 without the boss; × the leader's side (bidding side or
     defenders);
   - by trick number with 3 or more trumps; by declaration.

   **Near misses:**
   - following with 4 or more trumps, by seat;
   - leading with 0–1 trumps: 3 or more off doubles against none, as a difference.

   **R² of z on non-sixes leads:**
   - trick number + boss + 6-5, and the gain from adding trumps held;
   - trumps held, trumps out, trumps gone, held + gone;
   - off doubles held.

   R² on all inputs is 1 by construction for a layer-1 unit, so it is not reported. On/off
   agreement on leads with "fewer than t held" and "more than t out", against chance. Both poles
   under read R, with the − pole's lead share, mean trumps held and out, and sixes share.
2. **Circuit in.** The 24 largest weights, the bias, and the weight on every trump slot in hand and
   gone.
3. **Circuit out.** The readers.
4. **Formation** (read off, seen). At each of the 15 checkpoints, in mean z:
   - **the lead term**: following minus lead, on 0–1 trumps held;
   - **the trump term**: leads with 0–1 trumps minus leads with 3 or more, tricks 1–2;
   - off on lead, R² of held and of out on lead, and both labels.
5. **Steering.**
   - **Rows:**
     - **choice leads**: on lead, with both a trump and a non-trump in hand (every tile is legal on
       a lead);
     - **the use near miss**: following, void in a led suit that isn't trump, with a trump and a
       non-trump legal.
   - **Reads:**
     - leads trump (also split unit off and unit on);
     - leads an off double;
     - leads count;
     - on the near-miss rows, trumps in.
   - **Up:** add k·σ to the unit after its ReLU (k = 0.25, 0.5, 1, 2). Then add **the gate-closed
     value**: the unit's median activation on leads with 0–1 trumps where it is on. That is a
     realistic "on", where σ may be small next to it. Each is read as own − layer, the layer being
     the mean of the other live units under their own dose of the same kind. The share of picks
     changed is reported, so a null can be read as low power.
   - **Down:** ablate every live layer-1 unit on the same rows, one at a time. Report Δ with an
     interval and unit 8's rank by point estimate.

A consistency check asserts that trumps held, counted from the canonical slots, equals the rules'
trumps in hand on every lead.

## Predictions, sealed before 1–5 are run

Seen before sealing:
- the seat and play-count off rates, the declaration off rates and the trumps-held staircase on
  leads;
- the raw-input differences and the best v2 concepts;
- by accident, seed 0's formation and final held/out R² on the smoke rows.

Not seen:
- the controlled crosses, the near misses, the weights and the readers;
- the steering on seed 0 (the smoke steering was seed 1's unit 8, another unit).

A layer-1 unit is linear in its inputs, so the label cells are sums of weights over which inputs
co-occur. Only predictions about co-occurrence, use and growth can lose.

1. **Its vote is against leading trump (up, gate-closed value, unit off).** On choice leads where
   unit 8 is off, adding the gate-closed value lowers leads trump beyond the layer: own − layer < 0,
   with its interval excluding 0.
2. **It brakes trump leads where it is on (down).**
   - **2a, sign:** on choice leads where unit 8 is on, ablating it raises leads trump, with its
     interval above 0.
   - **2b, rank:** its Δ there is the largest in the pool of live layer-1 units.

   A miss on 2b alone means it shares the brake (with paper 3's unit 0, say), not that it isn't one.
3. **Length beyond position, the boss and the 6-5 (label, controlled).**
   - On non-sixes leads, adding trumps held to trick number + boss + 6-5 raises R² by at least 0.05.
   - On leads in tricks 1–2, without the boss and not sixes, the off rate at 4+ trumps is at least
     0.3 above the rate at 1 trump.
4. **The lead, not the holding (near misses).**
   - Following with 4 or more trumps, the off rate is at most 0.05 at every following seat.
   - Leading with 0–1 trumps, 3 or more off doubles against none differ in off rate by at most
     0.05 either way.
5. **About the lead, not trump in general (use near miss).** At the gate-closed value, the
   own − layer shift in trumps in on the near-miss rows is less than half the size of prediction
   1's shift in leads trump.

Read off, not graded:
- formation and the held/out R² (seen);
- the 0–1-trump off-leads and what they read;
- the side and trick-number crosses;
- where the pick goes instead (off double, count);
- the k·σ doses, the down rows over all choice leads, and the poles.

The series' formation record cuts both ways: #194 saw position first across units, and paper 6
saw the reverse in its unit. With the trajectory seen, no order is graded.

**What would break the explanation.**
- 1 or 2a failing: the unit does not carry the decision to lead trump, and the staircase is a
  correlate.
- 3 failing: the variable is position, the boss or the 6-5, not length.
- 4 failing: the gate is about strength or holding, not the lead.
- 5 failing: the vote is about trump in general; the noun "trump scarcity" fits better than the
  lead verb.

None would be read as mixing. A miss names what the unit reads instead, and the list-v3 candidate
changes with it.

**Candidate list-v3 concepts** (#202), each to be scored on held-out rows against κ 0.02:
`lead-trumps-3plus`, `lead-trumps-4plus` and `trumps-out-at-most-1` (verb B).

**Cost.** One run of about two minutes under `scripts/cpu-slot`.

## The review before sealing

One fresh-context Opus read the essay's full text (to line 944), the draft, `run_lead`, paper 6's
page, the six-paper synthesis and the raw notes. Its points, and what changed:
- **The label explains the minority.** About 47% of off-leads have 0–1 trumps. → Verb B, "trump
  is still contested", is now beside verb A. The 0–1-trump off-leads are read by trick number,
  trumps gone and declaration, and trumps out is a variable.
- **Trick number, the bidder and sixes confound the staircase.** → Prediction 3 is graded within
  controls: non-sixes, no boss, tricks 1–2, with an R² gain over trick + boss + 6-5. The side
  cross is added.
- **A noun (trump scarcity) passes the same tests.** → Prediction 5, a use near miss on trumping
  in while following.
- **The two use predictions share the read.** → What the pick moves to (off double, count) is read
  off beside leads trump.
- **The 1σ dose is weak.** → Graded at the gate-closed value, with the share of picks changed.
- **Sign and rank were fused.** → 2a and 2b graded apart; a rank-only miss means a shared brake.
- **The near misses were nearly settled.** → Tightened: followers at most 0.05, doubles as a
  difference within ±0.05.
- **The formation prediction was badly posed and ignored paper 6.** → Moot: the trajectory was
  then seen by accident (above), so formation is read off.
- **Code.** R² on all inputs dropped (identically 1); the row count renamed; the formation
  comment fixed; a held-trumps consistency assert added.
- **No drift toward superposition or against the book.** The reviewer suggested reading the
  side cross against *Winning 42*'s guidance on when the bidder pulls trump and when defenders
  lead it. That goes with the results, as agreement with the book's reasons
  ([[book-authority]]).
- **Name.** The slug is distinct from paper 5's `no-trump-yet`, and the verb no longer says "yet".

Sealed 2026-10-20.
