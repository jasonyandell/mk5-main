# Cold critic: does Zeb find the optimal move in finite time?

Research night 2026-10-12, critic role (a separate Fable 5.1 session). Written 02:20 to
03:10 CDT for Jason at 7am. Everything below was checked against the files on disk:
I rebuilt the Lean project, re-ran every empirical report script on its JSON, recomputed
fable2's E4/E5/E6/E7/E8 from their JSON, re-ran theory's two check scripts and the
empirical E3 and agreement scripts, and ran two small experiments of my own
(`critic/truth_rank.py`, `critic/agree_vs_stability.py`, 9 s and 15 s). Nothing outside
`critic/` was written. Reports read in their final versions: theory 01:41, lean 01:38,
fable2 01:49, empirical 02:09, counter 02:15.

## 0. The verdict in ten lines

1. **The question has an answer, and it is a conditional.** Exact Walt finds the O1 move
   in finite time against a fixed field F when its belief is F's posterior (theory T1,
   proved; checked by brute force at 50 lab information sets by counter). Every other
   reading of "Zeb finds the optimal move" is false or undefined.
2. **For deployed Zeb the referent is degenerate.** With the deterministic net as F,
   the posterior puts zero mass on the true deal at 586 of 674 self-play positions
   after the lead (empirical, 40 of 40 deals). O1-vs-net is well defined only at a
   table that really plays the net.
3. **At such a table, uniform Zeb is measurably not O1.** Sampling from the net's
   posterior gains +0.028 ± 0.006 marks per hand, 120 flips for and 61 against, over
   2,120 paired deals (empirical E1b; I re-ran the report). That is C1 in real 42.
4. **In self-play the softened repair is unmeasured, not null.** +0.010 ± 0.008 marks
   per hand on 1,200 deals has a 95% interval that contains the whole +0.028. My
   truth-rank test shows the softened net likelihood ranks the true deal at the 71st
   percentile of 512 uniform worlds, at every ε and every phase. The belief carries
   signal in self-play. What is missing is power and a phase-restricted design.
5. **Zeb after the lead is a one-step rollout of the net.** Fable2's E7 (82% to 100%
   same tile as a floor above the sample, in-sample loss 0.0007 pmake) is confirmed
   from the JSON, and the hub's own cell sizes (3.1 worlds at the second own decision
   at 160 worlds) imply the same for the champion the +0.56 was measured on. The
   contingent plan is exercised at the root only.
6. **Δ_execution has one definition and three regimes.** Zero with coherent beliefs
   (theory T1′, counter 0 of 150). Non-negative with uniform beliefs and no floor
   (empirical E2: 300 of 300, strictly positive at 11, ≤ 2 pmake points). Negative with
   the floor (E2: 16 of 300, ≤ 3.5 points). Fable2's "−2.7 points" is the floored sign on
   mixed world samples; the same-world number is −0.07 points.
7. **T2 is proved and bounded, and its measured depth is tiny.** Lean: 24 applications
   from any base settle the opening lead under fixed beliefs; the bound is attained by
   an abstract operator. Counter: every fixed-belief ladder in mini-42 settled by rung 2
   against a bound of 7. With coherent beliefs the ladder cycles, with certified strict
   cycles (every change forced) in 8 of 16 lab configurations.
8. **The seed catch is real and harmless to anything shipped.** Rung k seeds its
   sample with k (planner.py line 304), so the sampled ladder is not one operator.
   Deployed Zeb is a single application. "Data stops at parity is the ladder
   stabilizing" (fable2 §4) is a conjecture, not a measurement, and the seed catch is
   the smaller of two reasons: no one measured whether successive nets' Zebs agree.
9. **The floor-off-when-exact arena null (−0.02 [−0.35, +0.29]) was predictable.**
   Fable2 E6 priced the value at stake at 0.0001 pmake per late decision over 1.7% of
   moves; the floor role's own diagnostic finds the rule changes 0.9% of late tiles.
   384 games cannot see that. The arena was the wrong instrument, and it ran 584 s
   against a 300 s rule.
10. **What the five reports still miss mathematically** is in §4e: the name of the
    fixed point Zeb actually sits at (a self-confirming equilibrium with a
    misspecified model, the Berk–Nash shape), why agreement is the wrong fidelity
    metric (40% of decisions are exact ties; the net imitates decided moves at 88%),
    and that the whole O1 frame cannot see conventions.

## 1. Per report

### 1.1 Theory (Opus 5.5)

**Established.** T1 and T1′ are correct as stated; Lemma W is the load-bearing step
and its perfect-recall argument (distinct cells are distinct information sets, so the
max distributes) is right. Lemma 1 (Bayes) is right, including the legality factors.
Props A, B, C and the constants table check: ε_K = osc·√(ln(7/0.025)/2K) gives 13.3 /
7.4 / 5.2 / 2.6 / 1.3 pp at 160 / 512 / 1,024 / 4,096 / 16,384 (recomputed). Fiber
sizes (3k)!/(k!)³ check for k = 1..7. The m-action family arithmetic
(3(m+2) vs 2m+5 over 2(m+1)²) checks. The §3.6 cycle and the §3.5 sharpness family
re-run clean (`cycle_check.py`: P² = P⁶, P²..P⁵ distinct; `sharp_check.py`: first
change at Pⁿ⁻¹ for n = 2..8). Line 304 of planner.py mixes k into the seed, as
claimed. The ledger's 0.16 s/ρ (512 × 12 × 26 µs) and the 0.53^m table check.

**Overreach.**
- "The tie rule is not a gap" is true for value at one decision (Prop T) and misses
  what the tie rule does to the pour. 285 of 714 planned positions in the 40-deal pour
  have an exact tie at the top (my `agree_vs_stability.py`; empirical finds 557 of
  1,390 on its lines). At those positions Zeb's move is a function of the seed, and the
  net is then trained to imitate a coin. That is not an O1 gap, it is a teacher defect,
  and the theory report is the place it should have been named.
- §4.2's "Repair: floor 1 whenever the fiber was enumerated… closes this row of the gap
  completely" is correct and not worth a ledger row: fable2 E6 measured 6 of 349 moves
  changed, mean regret 0.0001 pmake, and the arena confirmed nothing is there to find.
- §6.3's prediction that the true deal survives the filter with probability ≈ 0.53^m is
  refuted upward: measured 0.178 vs 0.062 at m = 4–5 and 0.059 vs 0.012 at m = 6–8
  (empirical E1d, re-run). Theory pre-registered this reading ("errors cluster on
  positions the real deal does not visit") and should now say it plainly: the net's
  disagreements with Zeb are positively correlated within a deal, the independence
  model is wrong by 3–5× mid-hand, and the filter is less lethal than §6.3 says while
  still reaching 0.8% survival by play 20.
- §2.5 "if Zeb committed to its root plan, it would lose at most 2(1−ρ)" needs "with no
  floor" stated. The floored plan is not the unrestricted optimum Prop B requires.

**Errors.** None material. §5.3's C(21,7) = 116,280 is right. The report says the
theory of §4.2 (composition of the one-cell optimism bound through the floored
recursion) is conjecture; it is, and fable2 E7 makes it nearly moot (see §4a).

**A staff reviewer would send back:** (i) name the tie-rule-as-teacher defect; (ii)
replace the 0.53^m paragraph with the measured curve and the correlation reading;
(iii) the ledger's row 1 cost "0.16 s/ρ" is for the Python forward; in the Rust core the
filter is µs per world and the row is simply affordable at any ρ above 1%, say so;
(iv) the one theorem still missing from the whole night is Lemma W in Lean (§1.2
below).

### 1.2 Lean

**Established.** `lake build` clean (0.13 s warm, 9 jobs); `lake env lean Axioms.lean`
prints only propext, Classical.choice, Quot.sound for all 15 theorems; `grep sorry`
finds only a comment. I read all six files. `settle`, `stable_from_height`,
`limit_indep`, `fixed_point`, `fixed_point_unique`, `settled_by_rank`,
`fixed_point_by_rank`, `settled_42`, `bound_attained`, `floored_local`,
`rung_seeded_no_settle`, `value_shift`, `regret_le`, `certificate`, `early_stop_argmax`
and the pennies theorems say what the report says they say. `settled_42` concludes
`iter R k p c = iter R 24 q c` for k ≥ 24: the 24th application, i.e. P²³, for any base,
exactly as theory's indexing.

**Honest scope, stated by the role and worth repeating.** `settled_42` assumes `hgen`
(every coordinate is a genuine decision) and `hlater` (the response reads the rung
below only at later positions). Both were checked against planner.py by reading, not
by formalizing. `floored_local` fixes the floored predicate; in the code the predicate
depends on cell sizes, which depend on modeled moves at later coordinates. The appendix
argues the extra step. The argument is right because the operator is defined at the
root coordinate and everything inside the plan is later than the root, so Local holds
with the same dep. Cosmetic, not a hole.

**Overreach.** None. The report's "What it does not say" list is complete and accurate.

**What a staff reviewer sends back.** The night's only piece of actual game content,
Lemma W (the cell recursion over a weighted world list equals the max over legal
contingent policies, and it is here that perfect recall is used), is in prose only.
Without it, "T1 in Lean" is not true: Lean has T2, the TV bounds and the argmax. Lemma W
plus Lemma D would make T1 and T1′ formal, and they are a few hundred lines over a
finite game tree. That is the next Lean target, not more ladder theorems.

### 1.3 Empirical

Every headline number re-ran to the digit from the JSON on disk:

| claim | re-run result | file |
|---|---|---|
| +0.028 marks/hand, 120/61, p = 1.4e-5, 2,120 deals | +0.0278 ± 0.0063, 120/61, p = 1.38e-5, 2,120 | `lines_report.py` on a,b,c,d |
| fresh β^F draw +0.53 ± 0.13; consistent subset +0.00 ± 0.15 | +0.53 ± 0.13; +0.00 ± 0.15 (n = 1,353) | `e1b_report2.py` |
| certificate fires at 8 of 1,230 after play 0; median 2δ 1.8, median Δ 0.4 pp | 8; 1.809; 0.0039 | same |
| E2 floor 1: D ≥ 0 at 300/300, > 0 at 11, max 0.020; floor 16: < 0 at 16, max 0.035 | identical | `e2_report.py` |
| E3: floor changes the root move at 101/714, 25/40 openings, forecast −15.0 pp at the lead | 101, 25, −15.02 | `e3.py` re-run, 20 s |
| true deal consistent at 88/674 after the lead, 40/40 deals | 88/674, 40/40 | `e1a_pour40.json` |
| net agrees with Zeb 58% | 0.584 | `agree.py` re-run |
| E1c softened: +0.010 ± 0.008, 53/41, p = 0.26 | +0.0100 ± 0.0081, 53/41, 0.256 | `lines_report.py` on e1c |

**Established.** Items 1–7 of the report's "answers first". The setting-B design (Zeb in
one seat, the net in three, paired deals, a V line as noise control that comes out
18/19) is the cleanest experiment of the night. The E2 design (plan and execution scored
on the same enumerated fiber) is the right way to measure Δ_execution and the only one
that is.

**Overreach.**
- Item 8, "the softened posterior is null in self-play… the likelihood carries little
  usable signal". The interval [−0.006, +0.026] contains the entire setting-B effect.
  And the likelihood does carry signal: my truth-rank test (§3.1) puts the true deal at
  the 71st percentile of the uniform sample under the ε-softened net at every ε from
  0.1 to 0.8 and every phase. "Smaller than the well-specified gain and not yet
  distinguished from zero" is what the data say. The report's own power statement
  ("2.3 SE for a setting-B-sized effect") undersells its SE: 0.028/0.008 is 3.5 SE, so a
  full-size effect would have shown. The truth is in between.
- "Its effective sample stayed healthy": median ESS 1,456 at plays 8–11, yes, but 26
  at plays 16–19 and 4 at 20+ (recomputed from the JSON). Late fibers are small anyway,
  so this is not the old collapse, but "healthy" should carry the phase.
- "Exact planning bounds execution, as T1 predicts": E2 is Lemma W under the uniform
  belief (the plan is the max over legal policies on the fiber; the executed policy is
  one of them). T1 is about β^F. Right result, wrong theorem named.

**Errors.** None found. One phrasing: "the true deal goes F-inconsistent at the first
modeled play: median play index 1" is the index of the first play where the net
disagrees with Zeb's actual tile, over all seats. Viewer-relative, the median first
inconsistent decision is at 3 plays. Same fact, two clocks.

**What a staff reviewer sends back:** E1c again at ε = 0.8 (same truth-rank, 4× the
late ESS), applied only at plays 4–15 where E1b located the gain, 2,000+ deals, with
the interval reported against +0.028. That is four 5-minute runs on code that exists.

### 1.4 Fable2

Recomputed from the JSON: E7 same-tile 0.825 / 0.926 / 0.961 / 0.996 by phase, in-sample
loss mean 0.0007 and never negative (trivially: it is measured under the floor-16 plan's
own q, whose argmax is Zeb's move). E2: 0.19 planned of 2.60 own decisions pooled over
the first two tricks, 80.8% of world-paths with no second planned decision. E1: curse
+0.0122 ± 0.0030, 38% flips, out-of-sample regret 0.0049 mean, 0.045 max. E5: unanimous
across 10 seeds at 17% of leads, 44% of plays 1–7, 97% at 16+. E6: 3/300, 3/43, 0/6
moves differ; regret mean 0.0001, max 0.024. E4: q_in 0.299, Zeb 0.326, net 0.312; Zeb −
net +0.013 with position-level SE 0.0075. E8: median N needed 10,198 at the lead,
0% certified at 512; the "8–15" row is 216 + 32 positions from 16+.

**Established.** The object definition (§1, Prop 1: the floored plan is a legal policy
of the live seat; Prop 2: SAA optimism; Prop 3: deterministic per seed). E7 and E2 as
measurements. E3's consistency numbers match the empirical role's independent code
(66% / 91% / 98% true-deal inconsistency by phase, both). E6 as a measurement of the
floor on exact fibers. E8's conclusion.

**Overreach.**
- Headline 1, "Zeb is a one-step rollout": 82% at leads is not identity, and the 18%
  is scored with the floor-16 plan's own in-sample q, which fable2's own E1 shows is
  biased by +1.2 pp. The decisive test, the arena at f = 513 vs f = 16, was not run.
  The claim is nonetheless well supported (see §4a for why the hub's data back it);
  state it as "indistinguishable within 0.1 pmake point in-sample", not as identity.
- Headline 5 / E4, "Δ_execution is negative for Zeb, −2.7 points". The forecast is on
  512 worlds (seed 424242) and the realized value on 64 different worlds (seed 11),
  so the −0.027 (position-level SE 0.010) is sample noise plus the curse plus the
  floor effect, in unknown shares. The sign is right and the diagnosis (the floor puts
  the replanning player outside the plan's class) is right. The scale belongs to
  empirical E2: −0.07 pmake points mean, −3.5 worst, on the plan's own worlds.
- §3, "T1. Instantiated and checked at 306 real positions (E6)". E6 is under the uniform
  belief. It instantiates Lemma W, not T1.
- Headline 6 and §4, "'Data stops at parity' is the ladder stabilizing", stated as a
  finding. Theory marks the same sentence a conjecture, correctly. See §4d.
- E8's empirical-Bernstein "certificate" is computed on the sample that chose the plan,
  so it ignores the selection effect its own E1 measures. The N it reports is a lower
  bound on the N truly needed. Since the conclusion is "0% at 512", it survives.
- §6, "Zeb plays within a point of its own best response": under its own belief, yes
  (E1). Against the field it models, empirical says +0.53 pmake points per decision is
  on the table from play 4 on. Both are "within a point". Say which.

**Errors.** "100%" at 16+ is 99.6%. The 8–15 row of E8 lumps 32 late positions. Nothing
that changes a conclusion.

**What a staff reviewer sends back:** the f = 513 arena block, and E4 redone on the
plan's own worlds.

### 1.5 Counter

**Established.** Every output quoted is in `out/`. Instance A (mini42-8, bidder holding
3-1 3-2 after partner's 1-1 lead and LHO's 2-0 sluff: fiber 9, F-consistent 1,
uniform plays the 3-2 and makes 0%, coherent plays the 3-1 and makes 100%) is the
best teaching example of C1 produced tonight, because the two field plays that carry
the information are both legal from every hand. Instance B (420-world posterior, gap
0.174) shows it is not a 1-world artifact. `delta_exec`: coherent 0 of 150, uniform
33 of 150 positive, max 0.0208. `c2a`: exact trinomial flip probabilities (0.234 at
N = 512 for a 0.011 gap). `c2b`: the floor is a pure tax under exact beliefs (0.0009
mean, 0.105 worst at f = 16) and a guard under thin samples. T2 fixed: settled at rung 0
at all 1,984 mini42-8 sets, rung 2 worst in mini42-12 against a bound of 7. T2 coherent
and tilt: certified cycles, including strict ones under the sticky tie rule (points /
coherent period 4; make / tilt trump 0 period 6). C3: 150 + 400 + 947 + 343 = 1,840
priors, the Walt partnership map reached O2 every time.

**Overreach.** The C3 "finding" is 2- and 3-trick games with ≤ 5-deal priors, and the
report says so. The reading "sequential rationality removes bad fixed points" is a
hypothesis about why, not a theorem; in real 42 the off-path fallback belief is
exactly where a wrong convention would hide. The "bad pair" example is admitted to be
reasoned, not re-run. The cycles move 3 to 24 of ~2,000 information sets with
ex-ante value swings of 0.0004 to 0.01: real, certified, and tiny, which the report
shows and should say in one sentence.

**Errors.** None found. Bounds R−1 = 3 and 7 are right for 2 and 3 tiles per seat.

**What a staff reviewer sends back:** the mini-42 lab is the right instrument for the
one thing nobody has: a 42-shaped position where a convention (a hand-to-lead map)
beats person-by-person optimality. The pooling search found none in 1,290 priors. A
4-tile-per-seat game with two count tiles is the next size up and is where I would look.

### 1.6 The floor arena (`floor/arena_k0`, `floor/REPORT.md` of 02:22)

`k0,n512,F=net` with `sampled_floor=True` (floor off when the fiber is enumerated) vs
the deployed player: 384 mirrored games, 4,311 hands, mean mark margin −0.02 [−0.35,
+0.29], made 36.3% vs 36.5%, halves 54.2% / 44.8%. A null, and the floor role's own
diagnostic says why: the rule changes the chosen tile at 26 of 2,771 late positions
(0.9%), never with 20 or more plays down, because with two tricks left the viewer's
only interior decision is its forced last tile. Fable2 E6 said the same from the other
side (6 of 349 exact-fiber moves, mean regret 0.0001 pmake). The per-game effect is of
order 0.001 marks and 384 games have an SE of 0.16. The run took 584 s against the
brief's 300 s limit, with no written exception in the directory; a k1 block is still
pending in the floor report (the `K1ROW` placeholder). The floor report's own verdict
("a correctness change, yes-leaning; a strength lever, no") is the right one, and its
branch is unmerged, so nothing shipped changed tonight.

## 2. Cross-checks between reports

- Empirical E3 (101/714 root moves change f16 vs f1; 25/40 at the lead) equals
  fable2 E7's `same_1` column (37.5% / 75.5% / 88.3% / 98.8% → 25 + 46 + 27 + 3 = 101)
  on the same pour. Two codes, one number.
- Empirical E1a true-deal inconsistency by phase equals fable2 E3's (66% / 91% / 98%).
- Counter's `delta_exec` (uniform ≥ 0, coherent = 0) and empirical E2 floor 1 (≥ 0 at
  300/300) agree in sign and shape; theory T1′ is the reason.
- The hub's floor table (floors 8..161 give identical forecasts; cells average 3.1
  worlds at the second own decision at 160 worlds) and fable2 E2/E7 (0.19 planned of
  2.60 decisions; 81% of paths never see a second planned decision) are the same fact
  from two sides.

## 3. Two critic experiments

### 3.1 Does the softened net likelihood point at the truth in self-play? (`truth_rank.py`, 8.6 s)

At each of the 674 planned positions after the lead in the 40-deal self-play pour,
weight the 512 uniform worlds and the true deal by ∏ (ε/n + (1−ε)[net picks the played
tile]) over the other seats' discretionary plays, and take the percentile rank of the
true deal's weight among the 512 (midrank for ties). 0.5 means the likelihood is noise
about the truth.

| ε | plays 1–3 | 4–7 | 8–11 | 12–15 | 16–19 | 20–27 | all (SE) | share of positions with rank > 0.5 |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.677 | 0.723 | 0.713 | 0.742 | 0.726 | 0.680 | **0.712** (0.009) | 0.77 |
| 0.5 | 0.678 | 0.728 | 0.718 | 0.745 | 0.723 | 0.674 | 0.713 (0.009) | 0.77 |
| 0.8 | 0.682 | 0.733 | 0.724 | 0.744 | 0.717 | 0.674 | 0.714 (0.009) | 0.78 |

Median ESS of the 512 weights at ε = 0.5: 229 / 142 / 83 / 68 / 23 / 3 by the same
phases; at ε = 0.8: 368 / 324 / 274 / 235 / 80 / 5.

**Reading.** The rank is 0.71, twenty SEs above noise, flat in ε and flat across the
hand. The 0/1 filter kills the true deal (empirical E1d) because it demands every play
match; the soft version keeps it and ranks it in the top third. So "the likelihood
carries little usable signal in self-play" is wrong. The information is there; ε sets
only the ESS. ε = 0.8 keeps the same pointing power with 3–4× the late ESS of ε = 0.5.
That is the setting to run E1c at.

### 3.2 Where does the net disagree with Zeb? (`agree_vs_stability.py`, 15 s)

Join fable2's E5 (Zeb's modal move share across 10 seeds, per position) with the net's
tile on the true hands at the same 714 positions.

| Zeb's seed self-agreement (mode share) | n | net agrees with Zeb |
|---|---|---|
| ≤ 0.5 | 70 | 0.271 |
| 0.6–0.9 | 158 | 0.544 |
| 1.0 (unanimous) | 486 | 0.642 |

| in-sample gap, best minus runner-up (pmake) | n | net agrees | regret of the net's tile under Zeb's q (pmake pts) |
|---|---|---|---|
| = 0 (exact tie) | 285 | 0.502 | 0.07 |
| 0–0.01 | 142 | 0.401 | 0.70 |
| 0.01–0.05 | 168 | 0.667 | 1.07 |
| > 0.05 | 119 | **0.882** | 1.30 |

**Reading.** Zeb's own seed-to-seed agreement is 0.89 overall (0.67 at leads), well
above the net's 0.58, so the imitation ceiling I expected (the net cannot beat the
target's own reproducibility) is not what caps the net; it sits 30 points below that
ceiling. But 40% of decisions are exact ties where the move is the tie rule and the
net's "disagreement" costs 0.07 pmake points, and where the gap is above 5 pmake points
the net agrees 88% of the time. Agreement is the wrong fidelity metric. The hub's regret
metric (0.26–0.35 points) is the right one, and by it the 250k net already reproduces
every decided move. This is why more data stopped moving marks: there was no decided
move left to learn, only coin flips.

## 4. Reconciliations

### 4a. Is Zeb a one-step rollout of its net, and what happens to "+0.56"?

**Established to within measurement.** Three independent facts: (i) fable2 E7, same
tile as the floor-above-sample player at 82% / 93% / 96% / 100% by phase, in-sample
loss 0.0007 pmake; (ii) fable2 E2, 0.19 of 2.6 own decisions planned, 81% of paths
with none after the root; (iii) the hub, 2026-10-10: cells average 3.1 worlds at the
second own decision at 160 root worlds and floors 8 through 161 give identical
forecasts. At 512 root worlds the second-decision cell is about 3.2 × 3.1 ≈ 10, still
under 16. So the champion `k1,n512,m8,f=16` that the +0.56 was measured on is also
one-step after the root, with rung-0 Walt at 8 dice worlds as its tail instead of the
net. That last step is arithmetic on the hub's numbers, not a measurement.

**What it does to the story.** Nothing to the marks and everything to the romance.
- The +0.56 was already decomposed by the hub: +0.30 [−0.00, +0.61] for 160 → 512 root
  worlds (both honest) and +0.13 [−0.20, +0.45] for floor 1 → 16 at 160 worlds. The
  floor's own block is a null. The honest sentence is: **three times the root worlds
  for a one-step rollout, plus removing the clairvoyant tail, beat the clairvoyant
  160-world player by +0.56**. The floor "bought" calibration and compute, not marks.
- "One contingent plan over the whole cell" is exercised at the root only. The splat's
  interior is the field's play. The hub's "the splat is now a calibrated forecast" is
  then explained rather than surprising: the forecast is a Monte Carlo estimate of a
  nearly fixed policy's value (root move, then the field everywhere), and such an
  estimate is unbiased except for the 19% of paths with a second planned decision,
  which is where fable2's residual +1.2 pp curse lives.
- The positive reading: the root is a true joint max over a 512-world cell, that is
  the whole of Walt's contribution, and it is where the marks came from (512 vs 160).
  Zeb is one policy-iteration step of the net, evaluated by 512-world rollout. Parity
  with the two-rung champion says one rollout step of a 58%-faithful imitation equals
  two nested Walt rungs on dice. That is a clean, known object (rollout policy
  improvement), and its known limitation is the one fable2 §4 names: its fixed point is
  not the optimal policy.

### 4b. Δ_execution: one quantity, three regimes

Note 1 defines Δ = V_plan − V_execute with both terms under the root belief against
the same F. All three reports measure that, under different (belief, floor, world-set)
conditions:

| who | belief | floor | worlds for V_execute | result |
|---|---|---|---|---|
| theory T1′ | coherent β^F | none | the root belief | **0**, proved |
| counter `delta_exec` | coherent | none | root posterior, exact | 0 at 150/150 |
| counter `delta_exec` | uniform | none | root fiber, exact | +0.0013 mean, 33/150 positive, max 0.021 |
| empirical E2 | uniform | 1 | the whole root fiber (≤ 512) | ≥ 0 at 300/300, 11 positive, max 0.020 |
| empirical E2 | uniform | 16 | same | −0.0007 mean, **16 negative**, 3 positive, max 0.035 |
| fable2 E4 | uniform | 16 | 64 fresh worlds vs a 512-world forecast | −0.027 (SE 0.010), positions at plays 0–13 |

Same quantity. The sign is governed by two things. With no floor, the plan is the max
over all legal policies on the root world set (Lemma W) and the executed policy is one
of them, so Δ ≥ 0 by definition; the strictly positive cases are the uniform replanner
forgetting what the field's plays implied (Prop C), and coherent beliefs make them
zero (T1′). With the floor, the plan's class excludes the policy actually executed (the
root is never floored, so the replanner at the next turn optimizes where the plan
handed down), and Δ can go negative. Fable2's −2.7 points is that floored sign measured
early in the hand on mixed samples, where the curse (+1.2 pp) and 64-world noise are
of the same order as the effect; empirical's −0.07 points is the same sign on the
plan's own worlds late in the hand. Nothing contradicts. The scale that should be
quoted is E2's, and E2 does not cover the early hand, where fable2's rough number says
the floor effect is bigger. Measuring it there needs the plan and the execution scored
on one fresh world set (both are cheap); nobody did that.

### 4c. The belief repair: what is known, and the next experiment

Known, each with its number:
1. The deterministic net's posterior is empty of the true deal in self-play: 586 of 674
   positions after the lead, every one of 40 deals, survival 0.8% by play 20
   (empirical E1a/E1d). Theory's 0.53^m underpredicts survival mid-hand by 3–5×:
   disagreements are correlated within a deal.
2. When the net really sits in the other three chairs, planning on draws from its
   posterior is worth +0.028 ± 0.006 marks per hand, from either chair, p = 1.4e-5 on
   2,120 deals; out of sample the chosen move improves by +0.53 ± 0.13 pmake points
   per decision from play 4 on. Reweighting the deployed 512 is worth +0.00 ± 0.15: the
   consistent subset is ~20 worlds at plays 8–11. The repair is "sample from β^F", and
   the hub's ESS lesson is exact (empirical E1b; counter C1 shows the mechanism in a
   9-world fiber).
3. Theory's certificate Δ > 2δ is sound (never violated) and useless (8 of 1,230),
   because δ = TV over the whole sample is 0.9 while the typical Δ is 0.004 (empirical).
4. In self-play the ε = 0.5 softened posterior, sampled properly, is +0.010 ± 0.008 on
   1,200 deals. Not a null: an interval that contains +0.028 and excludes −0.006.
5. **New tonight (§3.1):** the softened likelihood ranks the true deal at the 71st
   percentile of the uniform sample in self-play, at every ε and every phase. The
   signal exists. The theory's misspecification worry (§6.3, "the filter helps when
   survival stays well above ρ") is answered: the soft filter over-represents the truth
   by a factor of about 1.4 in rank terms, everywhere.

So the belief repair stands exactly where the hub left it ("paused, not dead") with
one thing changed: the mechanism of the old failure is now fully accounted for (0/1
support collapse plus reweighting instead of sampling), and the soft likelihood is
shown to point the right way. Fable2's "stochastic field with a tape" is the right
definitional fix (it makes β^F a proper posterior, the plan a proper expectation, and
the ladder a smooth map), and it is the prerequisite for anything beyond the next step.

**The next experiment**, in order of cost:
1. E1c at ε = 0.8 (same rank, 3–4× the late ESS), belief applied only at plays 4–15
   (where E1b found the gain and ESS is above 200), 2,400 deals in four 5-minute
   batches, interval reported against +0.028. Code exists. If the point estimate stays
   near +0.010, the self-play value of the repair is a third of the well-specified one
   and the decision is Jason's; if it reaches +0.02 the lever is back.
2. The stochastic field with a tape (softmax over the net's integer scores, drawn on a
   record-keyed tape) in `_field_move`, then the same E1b/E1c pair with F_σ as both the
   field and the likelihood. This is the version in which O1 is defined at a self-play
   table.
3. Only then an arena block, 384 games, and only if (1) or (2) shows a per-hand effect
   above 0.02, since 384 games resolve ±0.3 marks per game ≈ ±0.03 per hand.

### 4d. The seed catch and "data stops at parity"

The catch is real: rung k's sample seed is `_mix(seed, k, viewer, own, tape, forbid)`
(planner.py line 304, verified), so rung k is operator R_k and the sampled ladder is a
composition of different operators. Lean's `rung_seeded_no_settle` shows that locality
alone does not then give stabilization; theory's remark that base-independence survives
(T2.1 works for a composition of operators local for one dep) is right.

Does it undercut "the parity plateau is the stabilized ladder"? Only a little, because
that reading was never a measurement. Three separate facts:
1. Nothing shipped iterates. Zeb is one application of R_0 to the net; the champion is
   R_1 applied to R_0[dice]. The seed catch changes no shipped move.
2. Expert iteration (pour → net → Zeb → pour) iterates a map only under three
   idealizations: exact imitation (it is 58% by agreement, ~88% on decided moves by
   §3.2), a fixed planner seed across pours (the pour seeds changed: 1001.., 100001..,
   200001..), and a fixed root sample size. Theory lists the first two and marks the
   conclusion a conjecture. Fable2 states it as line 6 of its answer. Theory is right.
3. The measured quantity, "+0.11 → +0.13 marks per game vs the champion", is parity
   against a third party. Two different policies can both be at parity with a third.
   The fixed-point test is root-move agreement between successive nets' Zebs on a
   fixed position set (theory §3.7 proposes it; nobody ran it). Until it is run, "the
   ladder stabilized" is a conjecture and "the data lever stopped moving marks" is
   the measurement.

What §3.2 adds: the plateau has a mechanical explanation that needs no fixed-point
story. The net already reproduces the decided moves (88% where the gap exceeds 5 pp;
regret 0.3 points); what it does not reproduce is the tie rule's coin (0.07 points at
stake). More data of the same kind had nothing left to teach that marks could see.

### 4e. What the five reports still miss mathematically

Jason's instruction was "follow the math". Five things the math says that no report
says:

1. **The fixed point has a name, and it is not "equilibrium".** Zeb-vs-Zeb with a net
   trained on Zeb's own pour is a profile where each seat best-responds (one step, under
   uniform belief) to a *model* of the others, and the model is fit to on-path data from
   that same profile. That is a self-confirming equilibrium with a misspecified model
   class, and the expert-iteration limit is the Berk–Nash shape: the model converges to
   the KL-closest member of its class to the actual play, and the policy best-responds
   to that projection, not to the truth. The prediction of that frame is exactly what
   the hub measured: imitation regret keeps falling (0.38 → 0.26 points) while marks do
   not move, because the projection, not the data, is the limit. The lever that moves
   it is the model class or the response operator, never more rows. Fable2 §4 says
   "only a different operator can" without naming why; this is why.
2. **Agreement is the wrong fidelity metric; the tie rule is a teacher defect.** 40% of
   decisions are exact ties under `make`. At those the move is the lowest tile id of a
   seed-dependent set, and the net is trained to imitate a hash. The pour should carry
   the q-vector (it does, in the splat) and the net should be fit to the decided
   margin, not the tile. The hub's margin target already does half of this; the other
   half is to stop scoring the net on "agree" at all.
3. **O1 cannot see conventions, by construction.** Every referent on the table tonight
   holds the other three seats fixed or lets them move one at a time. Counter searched
   1,840 priors and found the one-seat map always reached O2 in 2–3-trick games. That
   is evidence about tiny games, and the theory's self-confirming-conventions
   conjecture (§5.2) remains the one mechanism by which the pour could be structurally
   unable to learn something humans do. The math that addresses it is the
   common-information coordinator (one agent choosing prescriptions over the mover's
   possible hands), and it is tractable exactly where fibers are small: the last three
   tricks, 1,680 worlds before voids. An O2-vs-O1 gap measured there would be the first
   number on this question.
4. **The sampled player is a mixed strategy, and nobody wrote its value.** Against a
   fixed field, mixing over seeds is harmless (fable2 Prop 3). Against an adapting
   opponent, or as the teacher of an imitator, it is not. A one-line theorem is missing:
   the value of the seed mixture against a fixed F equals the mean of the per-seed
   values, and the imitation target's entropy is a lower bound on any deterministic
   imitator's disagreement rate. §3.2 measures both sides of that bound (0.89 target
   self-agreement, 0.58 imitator agreement) and shows the net is far from it.
5. **"Finite time" has three limits and the question conflated them.** N → |Fib| (exact
   late in the hand, certifiable never at the lead: C2.1, E8), k → ∞ in the ladder (24
   under fixed beliefs, cycles under coherent ones, moot for one-rung Zeb), and t → ∞
   in expert iteration (a conjecture with a mechanical rival explanation). The reports
   together settle the first two and leave the third. Any future statement of the form
   "Zeb converges" should say which limit.
6. **The floored value is a two-player mixture and its bias theorem is easy now.** With
   E7 in hand, the floored root value is the Monte Carlo value of "a, then the field
   everywhere" on 81% of paths and a max over at most one further decision on 19%. Its
   optimism is bounded by osc × 0.19 × √(ln 7 / (2·16)) ≈ 0.05 and measured at 0.012.
   Theory's §4.2 conjecture is a two-line corollary of E7, not an open problem.

## 5. Verdict table

| claim | status | who | the one number |
|---|---|---|---|
| T1: exact Walt with β^F is an O1 best response at the root, finite | proved | theory; counter checked | 0 of 50 lab information sets disagree with brute force |
| T1′: replanning with β^F realizes the plan, Δ_execution = 0 | proved; measured | theory; counter | 0 of 150 |
| Lemma W: the cell recursion is the max over legal policies | proved in prose, not in Lean | theory | — |
| T2: fixed-belief ladder settles; 42 opening by the 24th application | proved, sorry-free | lean, theory | 24 applications, 0 sorry |
| T2 bound attained | proved (abstract); verified (game family) | lean; theory | first change at Pⁿ⁻¹, n = 2..8 |
| T2 with coherent beliefs | false, cycles | theory (4-cycle, exact); counter (certified strict cycles); lean (pennies) | period 4 from rung 3, mini42-8 |
| Measured ladder depth in the lab | measured | counter | settles by rung 2 vs bound 7 |
| Seed catch: the sampled ladder is not one operator | proved (toy); verified in code | lean, theory | planner.py line 304 |
| C1 in the lab: uniform ≠ O1, loss up to the whole range | measured, exact | counter | gap 1.00 in a 9-world fiber |
| C1 in real 42 vs the net at the table | measured | empirical | +0.028 ± 0.006 marks/hand, p = 1.4e-5 |
| β^F must be sampled, not reweighted | measured | empirical | +0.53 ± 0.13 vs +0.00 ± 0.15 pmake pts |
| 0/1 posterior empty in self-play | measured | empirical, fable2 | 88 of 674 positions keep the true deal |
| True-deal survival ≈ 0.53^m | wrong (too low mid-hand) | theory predicted; empirical measured | 0.178 vs 0.062 at m = 4–5 |
| Softened posterior in self-play is null | **not established** (underpowered) | empirical | +0.010 ± 0.008; CI contains +0.028 |
| Softened likelihood points at the truth in self-play | measured | critic | rank 0.71 ± 0.009, flat in ε and phase |
| Certificate Δ > 2δ sound | measured | empirical | 0 violations |
| Certificate Δ > 2δ useful | wrong | empirical, fable2 | fires at 8 of 1,230 |
| C2 sampling: move flips, value does not | measured | empirical, fable2, counter | 41% of leads flip; −0.09 ± 0.06 pmake pts |
| Sampling certificate affordable at the lead | wrong (needs ~10⁴ worlds, optimistic) | fable2 E8; theory C2.1 | 0% certified at 512 |
| Zeb ≈ one-step rollout of the net after the root | measured (in-sample); inferred for the champion | fable2; hub cell sizes | 82–100% same tile; 0.19 of 2.6 decisions planned |
| Floor restricts the policy class; Δ_execution < 0 under it | measured | empirical E2, fable2 E4 | 16 of 300 negative, max 3.5 pts |
| Floor "bought +0.56 marks" | overstated; the worlds did | hub attribution | floor alone +0.13 [−0.20, +0.45] |
| Floor off when exact helps | measured null, predicted null | floor arena + diagnostic; fable2 E6 | −0.02 [−0.35, +0.29]; 0.9% of late tiles change |
| Floor with exact beliefs is a pure tax | measured, exact | counter | 0.0009 mean, 0.105 worst at f = 16 |
| Tie rule costs no value at a decision | proved | theory | — |
| Tie rule is a teacher defect | measured | critic | 285 of 714 decisions are exact ties |
| Net is far below Zeb's seed self-agreement ceiling | measured | critic | 0.58 vs 0.89 |
| Net reproduces decided moves | measured | critic | 88% agreement where the gap > 0.05 |
| "Data stops at parity" = ladder stabilized | conjecture | theory (marked); fable2 (unmarked) | no successive-net agreement measured |
| C3 in the lab: Walt partnership map lands below O2 | not found | counter | 0 of 1,840 priors |
| C3 in real 42 (self-confirming conventions) | conjecture | theory | — |
| O2 / O3 affordable | wrong | theory | prescriptions up to 7^116,280 |
| Arena rule compliance for the floor block | violated | orchestrator | 584 s vs 300 s |

CRITIC DONE
