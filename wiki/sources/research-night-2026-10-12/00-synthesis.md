# Research night 2026-10-12: does Zeb find the optimal move in finite time?

*Orchestrator's synthesis of six reports: theory (Opus), lean (Opus), counter (Opus),
empirical (Opus), floor (Opus), fable2 (a separate Fable 5.1 session), reviewed by a
cold critic (a second Fable 5.1 session) that re-ran the numbers and added two
experiments of its own. Reports and code: `scratch/research-night/<role>/`. Every
experiment ran under 5 minutes wall except two arena blocks (granted 30-minute
exceptions in writing; the critic notes the first ran 584 s against the 300 s rule, and
that it was the wrong instrument for a 0.9% effect). Jason's instruction: "analyze the
Zeb object and find what we are missing mathematically. follow the math."*

## The answer

**Exact Walt finds the optimal move in finite time, against a fixed field, when its
belief is that field's posterior.** Proved (theory T1), checked by brute force at 50
lab information sets (counter), pieces formalized in Lean (argmax, the coherence
bound, the action certificate), and realized exactly by replanning (theory T1′;
counter: execution gap 0 on 150 of 150 lab positions). "Optimal" there means the best
response for one seat to a declared model of the other three. Every other reading of
"Zeb finds the optimal move" is false or undefined.

**Deployed Zeb is not that object.** It differs in four places, each measured tonight:

| gap | what Zeb does | what it costs | the repair | verdict |
|---|---|---|---|---|
| belief | uniform over the void-consistent fiber | **+0.028 ± 0.006 marks/hand** when the other three seats really are the net (2,120 paired deals, 120 flips for / 61 against, p = 1.4e-5); a total loss in a lab instance (0% vs 100%) | **sample** from the field's posterior by rejection from uniform draws; reweighting the 512 is worth exactly nothing (+0.00 ± 0.15) | real and affordable. With the deterministic net the posterior is empty in self-play (the real deal survives at 88 of 674 positions). The softened likelihood still ranks the real deal at the 71st percentile of the uniform sample, flat in ε and phase (critic), so the signal exists; the self-play value is **unmeasured, not null** (+0.010 ± 0.008, interval contains +0.028). Re-run in progress. |
| sampling | 512 iid worlds | flips the opening lead in 41% of hands, costs −0.09 ± 0.06 pmake points per decision (null); a 1-in-90 call flips with exact probability 0.234 | a certificate needs ~10,000 worlds at the median lead; 512 certifies 0% | "finite time" yes; "certified" no at any phone budget; regret below a point |
| floor | thin cells hand the move to the net | changes the root move at 14% of positions and 25 of 40 openings; the floorless plan is 15 pmake points more optimistic at the lead (the clairvoyance it bans); on an exactly enumerated fiber it is a pure tax, 0.0009 mean, 0.105 worst | floor off when the fiber is enumerated | built, correct, bit-identical, 92 tests; **null in marks** at both k0 and k1 (−0.02 and −0.01, intervals ±0.3) because it changes 0.9% of late tiles. A Zeb 0.2.0 decision for Jason. |
| referent | best response to the net | the net picks Zeb's own tile 58% of the time, but 40% of decisions are exact ties and where the gap exceeds 5 pmake points it agrees 88% (critic) | a stochastic field, so a posterior is defined; and stop scoring the net by agreement | agreement is the wrong fidelity metric; regret is right, and by regret the 250k net already reproduces every decided move. That is why data stopped. |

**The ladder stabilizes in finite time under fixed beliefs**, by the 24th application
of the response operator at the opening lead (P^23 in note 2's indexing), from any
starting policy, to a unique fixed point; the bound is attained. Proved in Lean,
sorry-free, 15 theorems in core Lean 4 with no Mathlib, clean build in a second
(`scenario/lean/`). In the lab every fixed-belief ladder settled by rung 2 against a
bound of 7. With beliefs recomputed from the rung below it can cycle forever: Lean's
matching-pennies witness, theory's exact signalling game, and certified cycles of
period 4, 6 and 12 in mini-42, strict in 8 of 16 configurations. One catch found by
reading the code against the theorem: Walt mixes the rung index into its sample seed
(`planner.py:304`), so the sampled ladder iterates a different operator at every
rung; the theorem covers exact Walt or a rung-independent seed. Deployed Zeb is one
application of rung 0 to the net, so the ladder question is moot for it.

**One-seat optimality is the most the mathematics gives cheaply.** An O1 fixed point
need not be partnership-optimal (O2) or team-maxmin (O3); note 1's tables check out
with a gap of 0.97. But in mini-42 the Walt partnership map reached O2 in all 1,840
priors tried: every bad mutual best response stood on a partner's move at an
information set the pair never reaches, and replanning at every information set, off
path included, removes it. A finding about tiny games, not a theorem about 42. O3 was
not touched.

## What Zeb is, exactly

The separate Fable session measured the plan itself, and the critic confirmed it from
the JSON. With 512 worlds and a floor of 16, the plan decides the root and then only
0.19 of the live seat's 2.6 remaining discretionary decisions; 81% of worlds never see
a second planned decision. A player with the floor set above the sample, so every
interior own decision goes to the net, plays the same tile as shipped Zeb at 82% of
leads, 93% of trick-1 and trick-2 positions, 96% mid-hand and 100% late, and when it
differs the in-sample loss is 0.0007 pmake. The hub's own cell sizes (3.1 worlds at the
second own decision at 160 root worlds) put the champion `k1,n512,m8` in the same
regime, with rung-0 Walt on dice as its tail instead of the net.

**Deployed Zeb is one policy-iteration step of its own net, evaluated by a 512-world
rollout.** "One contingent plan over the whole cell" is exercised at the root only;
the splat's interior is the field's play. Three things follow.

1. **The +0.56 was the worlds, not the floor.** The hub's own attribution already said
   it: +0.30 for 160 → 512 root worlds, +0.13 [−0.20, +0.45] for the floor alone. The
   honest sentence is "three times the root worlds for a one-step rollout, plus
   removing the clairvoyant tail, beat the clairvoyant 160-world player." The floor
   bought calibration and compute. The splat is a calibrated forecast because it is a
   Monte Carlo estimate of a nearly fixed policy's value, unbiased except on the 19%
   of paths with a second planned decision, where the residual +1.2 pmake points of
   optimism lives.
2. **"Data stops at parity" has a name.** Zeb-versus-Zeb with a net fit to Zeb's own
   pour is a self-confirming equilibrium with a misspecified model class: the net
   converges to the closest member of its class to the actual play, and the policy
   best-responds to that projection, not to the truth (the Berk–Nash shape). The
   prediction is what the hub measured: imitation regret keeps falling while marks do
   not move. The lever is the model class or the response operator, never more rows.
   This is a conjecture with a mechanical rival explanation (the critic: no one has
   measured whether successive nets' Zebs agree), and the seed catch is the smaller
   caveat.
3. **"We need more search" means the live seat's own later decisions**, which the
   floor hands to the net almost everywhere below the root. Not depth: every world
   already plays to the end. Not data. What the plan decides: thicker cells (more
   worlds, which the phone cannot afford) or a smarter floor that hands down only
   when the net's move is within ε of the cell's max, a certificate the floor never
   checks today. The floored value's optimism bound is now a two-line corollary:
   osc × 0.19 × √(ln 7 / 32) ≈ 0.05, measured at 0.012.

## What we are missing mathematically

In the order the math points:

1. **A stochastic field.** With a deterministic net the posterior has empty support in
   self-play, so every belief experiment on the seam, past and future, is degenerate.
   A softmax or ε-mix over the net's integer scores, drawn on a record-keyed tape so
   worlds still partition (the dice-tape construction with the net's distribution).
   One line in `_field_move`, referee and core. Then β^F is a proper posterior, the
   plan a proper expectation, the ladder a smooth map, and the hub's "uniform beat
   every belief" becomes a real question again. The old failure is now fully
   accounted for: 0/1 support collapse plus reweighting instead of sampling.
2. **The right fidelity metric for the net.** Agreement counts coin flips: 40% of
   decisions under `make` are exact ties where the move is the tie rule on a
   seed-dependent set, and the net is trained to imitate a hash. Score the net by
   regret under the splat's q (the hub's 0.26 to 0.35 points), fit it to the decided
   margin, and stop reporting "agree." A one-line theorem is missing with it: the
   imitation target's own seed entropy (self-agreement 0.89) lower-bounds any
   deterministic imitator's disagreement, and the net (0.58) sits 30 points below
   that ceiling.
3. **A per-cell certificate.** Δ > 2δ is sound (0 violations) and vacuous (8 of 1,230
   certified; median 2δ = 1.8 pmake points against a median Δ of 0.4). The usable one
   bounds belief error only on the cells that decide the move, and the same object
   would certify the floor.
4. **O1 cannot see conventions, by construction.** Every referent tonight holds the
   other seats fixed or moves them one at a time. The math that addresses it is the
   common-information coordinator, one agent choosing prescriptions over the mover's
   possible hands, tractable exactly where fibers are small: the last three tricks,
   1,680 worlds before voids. An O2-versus-O1 gap measured there would be the first
   number on the self-confirming-conventions conjecture.
5. **"Finite time" has three limits and the question conflated them.** N → |fiber|
   (exact late, never certifiable at the lead), k → ∞ in the ladder (24 under fixed
   beliefs, cycles under coherent ones, moot for one-rung Zeb), and t → ∞ in expert
   iteration (the conjecture above). Any future "Zeb converges" must say which.
6. **The right referent for the table.** O1 against the net is a proxy twice removed
   from Plunge's field of three humans and a Walt partner. The target that fits the
   purpose is low regret under a belief robust to not knowing F, and uniform over the
   fiber is the maximum-entropy such belief. Its out-of-sample regret under its own
   objective is half a pmake point per decision. "Plays within a point of its own best
   response, in 44 ms, honestly forecast" is the property Plunge needs, and Zeb has
   it. That does not make the belief gap imaginary: against a table the net models
   well, it is worth 3 hands in 100.

## What was proved, measured, and conjectured

| claim | status | who | the number |
|---|---|---|---|
| T1: exact Walt + β^F is an O1 best response, finite time | proved; brute-force checked | theory; counter | 0 of 50 lab information sets disagree |
| T1′: replanning with β^F realizes the plan (Δ_execution = 0) | proved; measured | theory; counter; empirical | 0 / 150 lab; real 42 at floor 1: plan ≥ execution 300 / 300 |
| Δ_execution under the floor | measured, negative | empirical E2; fable2 E4 | 16 of 300 negative, ≤ 3.5 pmake pts (fable2's −2.7 was mixed-sample; same-world −0.07) |
| coherence bound, no horizon factor; certificate Δ > 2δ | proved, formalized | theory, lean | make constant 2; points 4r ≤ 168 |
| certificate Δ > 2δ sound / useful | sound; useless | empirical | 0 violations; fires at 8 of 1,230 |
| T2: fixed-belief ladder settles by the 24th application; unique fixed point; bound attained | proved, formalized; lab-checked | theory, lean, counter | 15 Lean theorems, 0 sorry; lab settles by rung 2 vs bound 7 |
| T2 with coherent beliefs | false | lean, theory, counter | cycles of period 4, 6, 12; strict in 8 of 16 configs |
| the seed catch | proved (toy), code-verified | lean | `planner.py:304` |
| C1 in the lab | measured, exact | counter | gap 1.00 in a 9-world fiber |
| C1 in real 42 vs the net | measured | empirical; critic re-ran | +0.028 ± 0.006 marks/hand, p = 1.4e-5 |
| β^F must be sampled, not reweighted | measured | empirical | +0.53 ± 0.13 vs +0.00 ± 0.15 pmake pts |
| 0/1 posterior empty in self-play | measured | empirical, fable2 | 88 of 674 positions keep the real deal |
| true-deal survival ≈ 0.53^m | wrong mid-hand | theory; empirical | 0.178 vs 0.062 at m = 4–5 (correlated within a deal) |
| softened posterior in self-play | **not established** | empirical | +0.010 ± 0.008; interval contains +0.028 |
| softened likelihood points at the truth in self-play | measured | critic | rank 0.71 ± 0.009, flat in ε and phase |
| sampling: moves flip, value does not | measured | empirical, counter, fable2 | 41% of leads flip; −0.09 ± 0.06; exact flip 0.234 at gap 0.011 |
| sampling certificate at the lead | not affordable | fable2, theory | ~10⁴ worlds; 0% certified at 512 |
| Zeb ≈ one-step rollout of its net | measured (in-sample); inferred for the champion | fable2; critic | 82–100% same tile; 0.19 of 2.6 own decisions planned |
| floor "bought +0.56" | overstated | hub attribution | floor alone +0.13 [−0.20, +0.45]; worlds +0.30 |
| floor off when exact | correct; null | theory, counter, floor | 0.0009 mean tax; arena −0.02 [−0.35, +0.29] at k0, −0.01 at k1 |
| tie rule costs no value / is a teacher defect | proved / measured | theory / critic | 285 of 714 decisions are exact ties |
| net far below Zeb's self-agreement ceiling; reproduces decided moves | measured | critic | 0.58 vs 0.89; 88% where gap > 0.05 |
| "data stops at parity" = a self-confirming fixed point | conjecture | theory, fable2, critic | no successive-net agreement measured |
| O1 fixed point need not be O2/O3 | cited + checked | counter | gap 0.97 at m = 99 |
| Walt partnership map reaches O2 in mini-42 | measured, all priors tried | counter | 1,840 / 1,840 |
| O2 / O3 affordable | no | theory | prescriptions up to 7^116,280 |

## What to do next, ranked by what the math says and what the phone can afford

1. **E1c at ε = 0.8, belief at plays 4–15 only, 2,400 deals** (running now at the
   critic's recommendation; result appended below if it lands before 7:00). If the
   point estimate stays near +0.010 the self-play value of the repair is a third of
   the well-specified one and the decision is Jason's; at +0.02 the lever is back.
2. **The stochastic field with a tape** in `_field_move`, referee and core. The one
   change that makes the belief seam a live question again, and the prerequisite for
   anything beyond step 1.
3. **Score the net by regret, fit it to the margin, drop agreement.** Half is done
   (the margin target); the other half is to stop reporting "agree."
4. **Zeb versus a table of nets with the posterior sampler in the core.** The +0.028
   is a Python-referee result at 2 to 6 s per decision; rejection sampling is
   microseconds in Rust. Then decide whether Plunge's bot partner, which is Walt, is
   "net enough" for the filter to pay.
5. **Merge floor-off-when-exact as correctness** in a Zeb 0.2.0 with regenerated
   parity fixtures. Not a strength lever; it removes the one place where the floor's
   reason for existing is false.
6. **A rung-independent seed** for the ladder, so the champion ladder is an iteration
   of one operator. Cosmetic for play, necessary for the theorem to describe the code.
7. **An O2-versus-O1 gap on three-trick endgames** with the common-information
   coordinator. The first number on conventions.
8. **Draw the plan tree** (the induction path Jason wanted last night), now with a
   sharper reason: it will show that the live seat decides the root and almost nothing
   else.

## The floor arena (floor role)

Built in both engines, bit-identical: `Rung.sampled_floor`, off by default; when on, a
rung whose worlds were enumerated (the sampler's own `fiber_size <= n` test, now
returned as a bit) is never floored. Spec letter `x`. 92 tests green (2 new). Default
behavior unchanged: the same decision digest on 120 positions, rule off, old and new
core, at both `k0,n512,F=net` and `k1,n160,m8`.

| block (384 mirrored games, seed 1; A = rule on) | A wins | marks/game (95% CI) | made, A vs B | wall |
|---|---|---|---|---|
| `k0,n512,F=v3 net` | 190/384 | −0.02 [−0.35, +0.29] | 36.3% vs 36.5% | 584 s |
| `k1,n512,m8` | 192/384 | −0.01 [−0.33, +0.29] | 36.6% vs 36.7% | 1208 s |

On 2,771 late positions at k0 the rule changed the tile in 26 (0.9%), none from play
20 on. Left on branch `worktree-agent-a72be62a882667abe` (`134b253f`, `f4078fa4`),
not merged: it changes the core's rung tuple and the deployed bytes.

## E1c-2 (the critic's experiment)

[appended if it lands before 7:00]
