---
title: Research night 2026-10-12 — does Zeb find the optimal move in finite time?
kind: experiment
first_seen: 2026-10-12
last_updated: 2026-10-12
status: complete
---

## Summary

One night, seven agents, one question from Jason: *"try to mathematically prove that
zeb finds the optimal move in finite time. or counterexample that and try to find what
we'd need to prove that … my goal is to analyze the Zeb object and find what we are
missing mathematically. follow the math."* Roles: theory, Lean, counterexample lab,
empirical, floor (all Opus), an independent Fable 5.1 session, and a second Fable 5.1
session as cold critic. Every experiment under 5 minutes wall; two arena blocks on a
written exception. Full reports, code and the orchestrator's synthesis:
[[research-night-2026-10-12-sources|wiki/sources/research-night-2026-10-12/]]. The
Lean project is checked in at `scenario/lean/` (15 theorems, core Lean 4, no Mathlib,
no `sorry`, builds in a second). Prior art: Jason's two notes of 2026-10-10 and
2026-10-12 on stabilization, the optimality gap and complementary plans (in the
sources).

## The answer

**Exact Walt finds the optimal move in finite time against a fixed field F when its
belief is F's posterior.** Proved (T1), checked by brute force at 50 lab information
sets, pieces formalized in Lean (argmax, the coherence bound, the action certificate),
and realized exactly by replanning (T1′: Δ_execution = 0, 150 of 150 lab positions).
"Optimal" means the best response for one seat to a declared model of the other three
(O1). Every other reading is false or undefined.

**Deployed Zeb** (`k0,n512,F=net,f=16`, [[scenario-player]]) differs from that object
in four places, each measured:

| gap | cost | repair | verdict |
|---|---|---|---|
| belief: uniform over the void-consistent fiber | **+0.028 ± 0.006 marks/hand** when the other three seats really are the net (2,120 paired deals, 120 flips for / 61 against, p = 1.4e-5); a total loss (0% vs 100%) in a 9-world lab instance | *sample* from the posterior by rejection; reweighting the 512 is worth +0.00 ± 0.15 | real, affordable, and only defined when the model of the other seats has support: with the deterministic net the real deal is filtered out at 586 of 674 self-play positions. The softened likelihood still ranks the real deal at the 71st percentile of the uniform sample, flat in ε and phase (critic): the signal exists in self-play but is weak: at power (ε = 0.8, plays 4–15, 2,400 deals) the softened belief is **null, +0.003 ± 0.004, interval [−0.006, +0.011], the well-specified +0.028 excluded**. The binding constraint is the model of the other seats, not the sample. |
| sampling: 512 iid worlds | flips the opening lead in 41% of hands at −0.09 ± 0.06 pmake points per decision (null); a 1-in-90 call flips with exact probability 0.234 | a certificate needs ~10⁴ worlds at the median lead; 512 certifies 0% | finite time yes; certified never at a phone budget; regret below a point |
| floor: thin cells hand the move to the net | changes the root move at 14% of positions and 25 of 40 openings; the floorless plan is 15 pmake points more optimistic at the lead; on an exactly enumerated fiber a pure tax of 0.0009 mean, 0.105 worst | floor off when the fiber is enumerated | built, bit-identical, 92 tests; **null in marks** (−0.02 [−0.35, +0.29] at k0, −0.01 at k1) because it changes 0.9% of late tiles; a Zeb 0.2.0 decision |
| referent: best response to the net | the net picks Zeb's tile 58% of the time, but 40% of decisions are exact ties and where the gap exceeds 5 pmake points it agrees 88% | a stochastic field; score the net by regret, not agreement | by regret the 250k net already reproduces every decided move, which is why the data lever stopped |

**The ladder (T2).** Under fixed beliefs the response ladder settles at the opening
lead by the 24th application (P^23 in note 2's indexing), from any starting policy, to
a unique fixed point; the bound is attained. Lean-proved. The lab's ladders settled by
rung 2 against a bound of 7. With beliefs recomputed from the rung below it can cycle
forever: Lean's matching-pennies witness, an exact signalling game, and certified
cycles of period 4, 6 and 12 in mini-42, strict in 8 of 16 configurations. One catch
from reading the theorem against the code: `planner.py:304` mixes the rung index into
the sample seed, so the sampled ladder iterates a different operator at every rung;
the theorem covers exact Walt or a rung-independent seed. Deployed Zeb is a single
application, so this is moot for it. (The seed is rung-independent since #112, the same day.)

**Partnership (C3).** An O1 fixed point need not be O2 (partnership) or O3 (team
maxmin); note 1's gap of 0.97 checks out. But in mini-42 the Walt partnership map
reached O2 in all 1,840 priors tried: every bad mutual best response stood on a
partner's move at an unreached information set, and replanning everywhere, off path
included, removes it. A finding about 2- and 3-trick games, not a theorem about 42.

## What Zeb is, exactly

With 512 worlds and a floor of 16 the plan decides the root and then only 0.19 of the
live seat's 2.6 remaining discretionary decisions; 81% of worlds never see a second
planned decision. A player with the floor above the sample plays the same tile as
shipped Zeb at 82% of leads, 93% of trick-1/2 positions, 96% mid-hand, 100% late, at an
in-sample loss of 0.0007 pmake (fable2 E7, confirmed by the critic). The hub's cell
sizes put the champion `k1,n512,m8` in the same regime.

**Deployed Zeb is one policy-iteration step of its own net, evaluated by a 512-world
rollout.** "One contingent plan over the whole cell" is exercised at the root only.
Three consequences:

1. **The +0.56 was the worlds, not the floor.** The hub's own attribution: +0.30 for
   160 → 512 root worlds, +0.13 [−0.20, +0.45] for the floor alone. The floor bought
   calibration and compute. The splat is calibrated because it is a Monte Carlo
   estimate of a nearly fixed policy's value.
2. **"Data stops at parity" has a name.** Zeb-vs-Zeb with a net fit to Zeb's own pour
   is a self-confirming equilibrium with a misspecified model class (the Berk–Nash
   shape): the net converges to the closest member of its class to the actual play and
   the policy best-responds to that projection. Imitation regret keeps falling, marks
   do not. The lever is the model class or the operator, never more rows. Conjecture;
   the rival mechanical explanation (successive nets' Zebs agreeing) is unmeasured.
3. **"More search" means the live seat's own later decisions**, which the floor hands
   to the net almost everywhere below the root: thicker cells, or a floor that hands
   down only when the net's move is within ε of the cell's max.

## What we are missing mathematically

1. **A stochastic field** (softmax or ε-mix over the net's scores on a record-keyed
   tape; one line in `_field_move`, referee and core). Without it the posterior has
   empty support and every belief experiment on the seam, past and future, is
   degenerate. The old tilt failure is now fully accounted for: 0/1 support collapse
   plus reweighting instead of sampling.
2. **Regret, not agreement, as the net's fidelity metric.** Agreement counts coin
   flips: 40% of `make` decisions are exact ties resolved by the tie rule on a
   seed-dependent set, and the net is trained to imitate a hash.
3. **A per-cell certificate.** Δ > 2δ is sound (0 violations) and vacuous (8 of 1,230
   certified). The same object would certify the floor.
4. **O1 cannot see conventions.** The common-information coordinator, tractable on the
   last three tricks (1,680 worlds), would give the first O2-vs-O1 number.
5. **Three limits, one word.** N → |fiber|, k → ∞ in the ladder, t → ∞ in expert
   iteration. Any "Zeb converges" must say which.
6. **The right referent for the table.** Uniform over the fiber is the maximum-entropy
   belief robust to not knowing F; its out-of-sample regret under its own objective is
   half a pmake point per decision. "Within a point of its own best response, in 44 ms,
   honestly forecast" is what Plunge needs, and Zeb has it.

## Next, ranked

Filed as issues #108 (stochastic field), #109 (regret, not agreement), #110 (floor-off-when-exact as Zeb 0.2.0), #111 (posterior sampler in the core, then marks), #112 (rung-independent seed), #113 (O2 vs O1 on endgames).

1. ~~E1c at ε = 0.8, plays 4–15, 2,400 deals~~ done: null at power (above).
2. ~~The stochastic field with a tape~~ done: `field_eps`, default 0 (#108, [[scenario-player]]).
3. ~~Score the net by regret; drop agreement~~ done: 0.0064 make probability per decision (#109).
4. ~~Zeb vs a table of nets with the posterior sampler in the core, in marks~~ done: +0.026 ± 0.003 marks/hand, 22,000 paired deals; not turned on (#111).
5. ~~Floor-off-when-exact as a Zeb 0.2.0 correctness change~~ done: the only behavior, Zeb 0.2.0 (#110).
6. ~~A rung-independent seed for the ladder~~ done (#112).
7. ~~O2 vs O1 on three-trick endgames~~ done: +0.027 margin per endgame vs the Walt map, half a convention on reached sets ([[endgames-o2-vs-o1-2026-10-12]]).
8. Draw the plan tree.

## Links

[[scenario-player]] · [[the-wall]] · [[candlewax]] · [[strategy-fusion]] ·
[[expected-q-value]] · [[splat-visualizer]] · [[arena]]
