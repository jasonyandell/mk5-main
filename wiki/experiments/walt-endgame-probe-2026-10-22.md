---
title: "The Walt endgame probe: Walt's tile against the exact optimum of its own model"
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: active
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

Pending.

## Links

[[comparison-algebra-2026-10-21]] · [[scenario-player]] · [[exploiter-endgames-2026-10-14]] ·
[[endgames-o2-vs-o1-2026-10-12]] · [[strategy-fusion-fence]] · [[silas]]
