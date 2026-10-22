---
title: "The Walt endgame probe: Walt's tile against the exact optimum of its own model"
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Issue #244, feasibility item 3 of [[comparison-algebra-2026-10-21]]. At three or fewer tricks left, is Walt's tile in
the exact information-state maximizing set of Walt's own model? Where it is not, what make (or set) probability
does it give up? A zero gap inside the noise is a finding: it would mean Walt's endgame is already optimal for its
model. Read-only on Walt and Zeb. Night-Math-2, night of 2026-10-21.

## What Walt plans on there (read in code, `scenario/planner.py`)

- **The belief.** Uniform on the viewer's void-consistent fiber (`draw="uniform"`, the default for both Walts below).
  It is not the field's posterior: the other seats' past plays enter only through their shown voids. When the
  fiber holds at most the live rung's n it is enumerated outright, and the exact belief is used (`sample_worlds`).
  Above n it is n draws, with replacement.
- **The model of the others.** At k = 0 with a field net (Zeb), each other seat plays the net's move from its own
  hand and the record. At k = 1 (the champion), each plays what a rung-0 Walt holding its hand would play at that
  record: 8 sampled worlds, a horizon of 4 plays and then the random tape. Either way the move is a pure function
  of (seat, hand, record), cached as such (`Walt._modeled_move`).
- **The plan.** One tile per cell of scenarios the viewer cannot tell apart (`_Invocation._value`), to the end of the
  hand. Under `make` a cell is valued as soon as the contract is decided.
- **The floor (f = 16).** Below the root, a sampled cell thinner than 16 scenarios does not maximize. It hands the
  viewer's move down a rung. An enumerated invocation is never floored.
- **Fiber sizes.** With three tricks left the fiber holds at most 1,680 worlds, when the viewer leads trick 5. With
  two tricks left it holds at most 90. So with two left every n512 decision is enumerated, and with three left it is
  sampled only when the fiber exceeds 512.

So on an enumerated fiber, Walt's own computation is the exact information-state optimum of its model, in floats.
The probe checks this independently. It measures the gap where Walt samples: three tricks left, fiber above 512.

## Method

- **Positions.** Real endgames from existing pours (Walt-family self-play, every position recorded):
  - **Champion** (`walt:k1,n512,m8`, `make`): `pour_make_s1001_n512_d5000`, the 5,000-deal make pour
    ([texas-42-walt-pour-make](https://huggingface.co/datasets/jasonyandell/texas-42-walt-pour-make), train split).
  - **Zeb** in its arena form (`walt:k0,n512,F=margin_d250k_h128x64,f=16`, `make`, the committed
    `zeb/net/margin_d250k_h128x64.fnet`): positions from `zeb200k/pour_s200001_d50000`, a k0, n512 pour over the
    50k field net.
  - Kept: every planned position with at most three own tiles (16 or more plays) and more than one legal tile.
- **Walt's tile is today's planner's.** The pours date from 2026-10-10, before the #110 floor and #112 seed repairs.
  Today's planner does not reproduce their recorded decisions: the q values differed at 12 of the first 25 champion
  positions, and the tile at 4. So the pours supply real positions, and Walt's tile at each is recomputed by today's
  Python referee (`scenario.planner.Walt`, the compiled core's reference). A decision is a pure function of
  (config, state, bid).
- **The exact optimum, three ways.**
  1. *The twin.* The same Walt with its live rung at n = 4,096. Every fiber is then enumerated, so by Walt's own rule
     there is no floor. This is Walt's own exact computation, in floats.
  2. *The course.* `lem/endgame.py` writes the decision as a family course (`silas-family-course-v1`) over every world
     of the fiber, the others played by the twin's own model (`Walt._modeled_move`, read only). A choice is the
     viewer's decision at what it has seen. Its mass is a count of worlds. A payoff is a count of worlds where the
     viewer's side gets its way: the bidders make, the defenders set.
  3. *Two exact solves.* `solve` is the Python reference, in Fractions. `silas family --courses` reads the file into
     `silas::family::Family` and solves it in BigRational. Silas also reports the controller's decision quotient
     (`Reach::Every`, `Path`, `Plan`; raw and fork interfaces).
- **Checks, which void the run if they fail.** At every position the twin's q must equal the course's values (to
  10⁻⁹), and Silas's exact root values and maximizing set must equal the reference's.
- **Fence.** Every maximum is over the viewer's plays at one information set. No per-world value is taken as the
  seat's, and no clairvoyant API is touched.

## Pre-registration (fixed before the measured runs)

**Sample.** The champion: the pour's deals 0–1,499. Zeb: deals 0–2,999. All kept positions of those deals, in pour
order. Runs go through `scripts/cpu-slot` under its 300 s cap, resumable, with no new position after 240 s.

**Primary, per Walt and by tricks left (3, 2), and at three left split by sampled (fiber > 512) and enumerated:**
1. The share of positions where Walt's tile is outside the exact maximizing set.
2. The mean regret: the exact maximum minus the exact value of Walt's tile, in the viewer side's success
   probability (make for the bidders, set for the defenders).

Each comes with a 95% interval from a deal-resampling bootstrap (4,000 resamples, seed 0, percentile; a deal's
positions are resampled together). **Chance baseline:** a uniformly random legal tile, with the share outside the set
being 1 − |best|/|menu|, and its mean regret.

**Predictions.**
- **Enumerated** (every two-left position, and three-left with fiber ≤ 512): outside = 0 and regret = 0 exactly. That
  is Walt's rule, so any nonzero is a bug in Walt or in the probe, not a finding.
- **Three left, sampled.** A nonzero share, prior 2–20% of positions, far below chance. The mean regret is below 0.02
  success probability.
- **Reading.** If the sampled stratum's regret interval holds 0, or its upper end is below 0.001 (a tenth of a
  percentage point), then "Walt's endgame is optimal for its model inside the noise". Otherwise the gap is real and
  the repair is to enumerate: at most 1,680 worlds, and the twin's cost is reported beside the live cost.

**Descriptive, with no verdict.**
- Two arms on the same three-left positions: the floor off (n512, f = 1), and n = 160, the zeb package's default
  budget.
- The controller's raw states (every, path, plan) at open positions, where not every tile ties.
- The CPU time of the live decision and of the exact twin.

**Pilot (seen before this section was written).** Deals 0–9 of the Zeb pour (60 positions, before the arms were
added) and deals 0–7 of the champion pour (49 positions). Twin, reference and Silas agreed at all 109. Zeb: 0
outside. Champion: 1 of 5 sampled positions outside, regret 0.0014; every enumerated position inside. Both pilots'
deals stay in the sample.

## Result

**Walt's endgame is the exact optimum of its own model wherever it enumerates, and within the pre-registered noise
bar where it samples.** Both Walts pass. Every number below has its 95% deal-resampled interval.

**The checks held everywhere.** At all 27,118 positions the twin's q equals the course's exact values. At all 27,118,
Silas's BigRational solve (`silas family`) equals the Python Fraction reference, value for value and in the
maximizing set.

**Enumerated decisions: 0 outside, as predicted by construction.** That covers every two-left decision and every
three-left decision with a fiber of at most 512: 16,133 for Zeb and 8,046 for the champion. In each, Walt's tile is
in the exact maximizing set. Chance would miss 8–26% of them.

**Sampled decisions (three left, fiber 560–1,680):**

| Walt | n (deals) | outside the set | chance | regret, success prob. | chance regret |
|---|---:|---|---|---|---|
| Zeb (`k0,n512,F=250k,f=16`) | 1,941 (3,000) | 0.057 [0.047, 0.067] | 0.325 [0.310, 0.341] | 0.00077 [0.00058, 0.00098] | 0.038 [0.035, 0.041] |
| champion (`k1,n512,m8`) | 998 (1,500) | 0.056 [0.042, 0.071] | 0.343 [0.321, 0.365] | 0.00052 [0.00036, 0.00070] | 0.036 [0.032, 0.041] |

- **Against the pre-registration.** The miss share sits inside the prior of 2–20%, and the regret is far under the
  predicted ceiling of 0.02. Both regret intervals exclude 0 but end below 0.001, so the registered reading applies:
  *optimal for its model inside the noise*. The gap is measurable and small. Zeb's upper end, 0.00098, passes by a
  hair.
- **What a miss is.** It is a real but small error, never a tie broken badly: every miss has regret > 0. A miss
  costs 0.0136 on average for Zeb (max 0.077) and 0.0093 for the champion (max 0.038). Bidders miss a little more
  often than defenders: Zeb 0.063 against 0.048, the champion 0.070 against 0.038.
- **Per hand.** Summed over every seat's decisions in a deal, Walt's sampling gives up 0.0005 success probability
  per deal for Zeb and 0.00035 for the champion. For scale, in a different unit (one seat, one three-trick
  endgame), an exploiter that knows Zeb takes +0.0040 ([[exploiter-endgames-2026-10-14]]).

**Whole strata, by tricks left (every position, enumerated or not):**

| Walt | tricks left | n | open | outside | chance | regret | chance regret |
|---|---:|---:|---:|---|---|---|---|
| Zeb | 3 | 8,922 | 3,894 | 0.012 [0.010, 0.015] | 0.255 [0.246, 0.264] | 0.00017 [0.00013, 0.00022] | 0.041 [0.039, 0.043] |
| Zeb | 2 | 9,152 | 1,472 | 0 | 0.080 [0.076, 0.085] | 0 | 0.025 [0.023, 0.027] |
| champion | 3 | 4,418 | 2,028 | 0.013 [0.010, 0.016] | 0.278 [0.264, 0.291] | 0.00012 [0.00008, 0.00016] | 0.040 [0.037, 0.043] |
| champion | 2 | 4,626 | 731 | 0 | 0.079 [0.072, 0.086] | 0 | 0.024 [0.022, 0.026] |

"Open" means not every legal tile ties. At one trick left the play is forced, so that stratum is not probed.

**Descriptive arms (three left, the same positions).**

| arm | Zeb outside | Zeb regret | champion outside | champion regret |
|---|---|---|---|---|
| as configured (f = 16) | 0.0123 [0.0102, 0.0146] | 0.00017 [0.00013, 0.00022] | 0.0127 [0.0095, 0.0161] | 0.00012 [0.00008, 0.00016] |
| floor off (f = 1) | 0.0095 [0.0076, 0.0116] | 0.00009 [0.00007, 0.00011] | 0.0127 [0.0095, 0.0161] | 0.00011 [0.00008, 0.00015] |
| n = 160 (the zeb package's default) | 0.0356 [0.0318, 0.0395] | 0.00069 [0.00058, 0.00080] | 0.0389 [0.0334, 0.0449] | 0.00064 [0.00052, 0.00078] |

- **The floor, for Zeb.** Below the floor, Zeb hands the move to the field net. The floor is a net 25 of its 110
  misses (41 fixed with it off, 16 introduced), and about half of its endgame regret: 0.00017 with the floor
  against 0.00009 without.
- **The floor, for the champion.** Below the floor, the champion hands the move to a rung-0 Walt. The floor changes
  44 tiles and nets zero (18 fixed, 18 introduced).
- **The rest is the sample.** n512 draws with replacement from a fiber of up to 1,680. At n = 160 the share outside
  roughly triples.

**Silas's controller at open three-left decisions.** This counts the raw states of the exact plan's controller:
how much of what the viewer will see the plan must remember.

| Walt | every | path | plan |
|---|---|---|---|
| Zeb | median 124, mean 201, max 1,378 | 38, 61, 471 | 38, 60, 470 |
| champion | 138, 256, 1,664 | 42, 79, 676 | 42, 78, 673 |

At two left it is 3 (every) and 2 (path, plan) at every open decision.

**Cost (CPU seconds per sampled decision, Python referee, descriptive).** Zeb live 0.052, and the exact twin with
the course and solve 0.145. The champion live 0.28, and exact 0.54.

**Historical note.** The pours' own tiles come from the 2026-10-10 planner, before #110 and #112. They lie outside
today's exact set at 463 of the champion's 9,044 positions. That is not a clean measure of those repairs: the model
of the others changed with them (the #112 seed), so the exact set moved too.

## Reading

Walt plans an information-state optimum by construction, and in the endgame the probe confirms it exactly: wherever
the fiber fits the rung, the tile is the exact argmax of Walt's model. The only gap is the 512-world sample on the
fat three-trick fibers. It is about 1 decision in 18 there, about 0.01 success probability per miss, and about
0.0005 make probability per deal. Under the pre-registered bar that is noise.

- **The cheap repair.** Enumerate the live rung whenever the fiber holds at most 1,680 worlds. It would remove the
  gap exactly, at about three times the cost of those decisions. It is a change under `scenario/`, so it goes to the
  Mayor first. Filed as #246, low priority.
- **What this does not test.** It tests Walt against its own model only. Whether that model is right (the uniform
  belief, the field net or rung-0 Walts as the others) is the scope of [[exploiter-endgames-2026-10-14]],
  [[endgames-o2-vs-o1-2026-10-12]] and the belief work. No per-world clairvoyant value enters
  ([[strategy-fusion-fence]]).

**Data.** Per-position rows (exact values, Silas's agreement, controller counts, arms), the report and the driver
are at [texas-42-zeb-research-2026-10 / zeb/walt-endgame-probe-2026-10-22](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/main/zeb/walt-endgame-probe-2026-10-22).
The courses (about 3 GB) were deleted after Silas solved them; `python -u -m lem.endgame run` rewrites any of them
exactly.

## Links

[[comparison-algebra-2026-10-21]] · [[scenario-player]] · [[exploiter-endgames-2026-10-14]] ·
[[endgames-o2-vs-o1-2026-10-12]] · [[strategy-fusion-fence]] · [[silas]]
