---
title: "The exact player, S0: complete endgames of at most three tricks, exact under a declared model"
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Stage S0 of the exact-player track (Exact-1, 2026-10-22). Can a seat play an endgame of at most
three tricks exactly: the exact counted posterior over the worlds it cannot tell apart, the
exact information-state optimum under a declared model of the other three seats, every decision
checked by an independent construction? What does each decision cost, and what does the
compiled controller have to carry? Walt-free: nothing from mk5 `scenario/`'s planner is a
component. The comparison with Walt comes last.

## The player (built, `silas/src/player/`)

- **The model.** A library of table types, one seat law per other seat (`lowest`, `highest`,
  `uniform`, `lean-low`, `lean-high`), under a prior affine in a parameter p: the family of
  [[comparison-algebra-2026-10-21]]'s `silas::family`.
- **The belief.** The exact counted posterior over (world, type) atoms: each world of the viewer's
  fiber, weighted by the type's prior times the likelihood of every play the others made, replayed
  in that world. A record impossible under every type is refused, never smoothed. A second root,
  `fiber`, drops the likelihood (Walt's belief kind, and the night's 560-deal ending).
- **The decision.** From the viewer's turn every atom is carried through the others' laws to its
  next turn; atoms the viewer cannot tell apart are merged into one block before any maximum. The
  value is a sum over blocks and a maximum over the viewer's own tiles only. STRATEGY-FUSION FENCE:
  nothing is clairvoyant. The course is solved by `Family` for every p at once, in BigRational.
- **The act.** The least maximizing tile, re-planned at every own turn (`play_hand`).
- **Budgets.** Kernel reads, a deterministic count. A budget refuses; it never truncates
  (FAIR CLOCKS).
- **CLI.** `silas exact --positions --library [--root] [--settle] [--p] [--play] [--recount N]
  [--quotient]`; positions from pours by `silas/examples/exact_positions.rs`.

## Gates (tests, `silas/tests/player.rs`, 13 s)

- **G1, the second construction.** `player::recount` shares only the rules and the declared laws:
  its own fiber enumeration by combinations, its own replay likelihood, plain rationals at one p,
  settlement only at the hand's end, no course and no `family.rs`. It agrees exactly with the player
  in mass, value and maximizing set on the note's 31 classes and on 20 seeded endgames (uniform
  posterior; the note's family on the fiber; the leaning mixture's posterior). No Python is the
  oracle: Python checks are banned on this track (Jason, 2026-10-22).
- **G2, time consistency.** Every re-plan from the fresh posterior has the previous course's child
  mass and maximizing set (Bayes and Bellman at every turn), six seeded hands.
- **G4, the fixture.** The note's ending built through the player's own course builder gives 24,089
  nodes, the 14 thresholds, 31 classes, 767 raw states (root ladder 1, 18, 31, 31) and 76 fork.
- **G6, determinism**, and a budget that refuses.

The note's own record is impossible under the note's deterministic Low/High types once the
likelihood is counted, so the exact posterior refuses it. Under the uniform kernel the record
weighs 1/384, and the posterior's values differ from the flat fiber's (`[150523/311040, …]` against
`[415/896, …]`). Both pick 5-1.

## Pre-registration

`scratch/desks/family-2026-10-22/PREREG-S0.md`, fixed before the measured batch. The probe's
positions of deals 0–499 of both pours (6,065 decisions) under the uniform kernel; complete
endgames from each deal's first three-tile position, with seeds 0–3. Gates G1, G2, G6, and G7:
realized success against the exact root value, played against the declared model. Endpoints:
work, instructions and controller size per decision, by tricks left. Pilots (40 positions) are
excluded.

## Results (measured at `37b663c4`, 2026-10-22, 08:16–08:27)

The orchestrator approved the pre-registration. Batch A ran 41 shards and batch B 80, each through
`scripts/cpu-slot`, none over the cap, none failed, none refused by the budget. The driver's own
limit ("at most 6 at once") was dead code. Its `jobs -r` ran in a subshell, so all 121 jobs
queued at once and cpu-slot alone set the concurrency. Instructions retired do not depend on that.
Wall is descriptive and was measured under that load. Data:
[texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)
at `exact/s0-2026-10-22/`.

**The gates.**

- **G1 holds everywhere.** The second construction agreed exactly, in mass, every root value and
  the maximizing set, at all 6,065 decisions of A and all 5,271 of B.
- **G2 holds on every hand.** On all 4,000 hands of B, every re-plan was the previous course's
  child.
- **G6 holds.** 102 rows of A, rerun, are byte-identical apart from `wall_ns`.
- **G7 passes, under the bar.** The realized success minus the exact root value:
  - champion: −0.0118 per hand, z = −1.83;
  - zeb: −0.0147, z = −2.24;
  - pooled z ≈ −2.9.

  It passes |z| < 3, but both pours lean the same way.

**G7 was designed weaker than its name.** It does not isolate the construction, and the page states
this as a scope finding:

- The future plays were drawn from the declared model, as pre-registered.
- The hidden deal and the record before the first decision came from Walt's self-play pours, and
  the counted posterior weighs worlds as if the uniform kernel had made that record.
- So G7 also measures how far the uniform kernel misreads Walt's past plays. A small optimism (the
  viewer side does about 1.3 points worse than its value says) is what a misread record predicts.
- A pure calibration draws the world from the posterior itself. That is the next gate (S1/S2
  successors).

**Work per decision** (exact counts, both pours; decisions decided at the root are one choice):

| own tiles | decisions | decided at the root | course choices, undecided: median / p90 / max | wall, undecided (descriptive) |
|---|---|---|---|---|
| 3 | 2,993 | 35.1% [32.7, 37.5] champion, 36.2% [33.9, 38.7] zeb | 4,742 / 86,773 / 478,240 | 32 ms / 594 ms / 6.1 s |
| 2 | 3,072 | 51.9% [49.4, 54.4] champion, 50.4% [47.9, 52.9] zeb | p90 114, max 565 | p90 ≈ 1 ms |

**Instructions.** These are instructions retired (FAIR CLOCKS), not wall.

- **Per decision, one process each.** On 61 decisions, every 100th of A, solve only, from
  `/usr/bin/time -l`:
  - a one-choice decision costs 1.9×10⁷ (process start and parse);
  - two own tiles, undecided (n = 16): median 2.4×10⁷, max 6.7×10⁷;
  - three own tiles, undecided (n = 24): median 4.5×10⁸, 90th percentile 9.9×10⁹, max 5.0×10¹⁰;
  - about 1.8×10⁵ instructions per course choice, which is BigRational arithmetic.
- **Per decision in the shards, including the recount and the controller.** Pooled (all
  instructions over all decisions): A 2.31×10¹³ over 6,065 decisions, 3.8×10⁹ each. B 5.97×10¹³
  over 5,271 decisions, 1.13×10¹⁰ each, replanning included.

**The compiled controller** (undecided decisions; both pours alike):

| own tiles | Reach Every, raw: median / p90 / max | Reach Plan, raw: median / p90 / max | fork (Every, Plan): max |
|---|---|---|---|
| 3 | 198 / 1,358 / 4,111 | 57 / 367 / 1,496 | 7, 6 |
| 2 | 3 / 3 / 3 | 2 / 2 / 2 | 2, 2 |

The plan the player actually follows needs about 57 states at three tricks left. Across every
shape of what it might see next, the fork interface needs at most 7.

**Against the predictions.**

- These held:
  - the gates;
  - no refusals;
  - "about a third decided at the root" (35–36%);
  - fork at most 10 (it is 7), and at two own tiles raw under 60 and fork at most 5.
- These missed:
  - **The median work at three own tiles.** Predicted 3×10⁴ choices; measured 4.7×10³ among the
    undecided, and 6×10² over all decisions. The pilot's deals 0–6 were larger than the sample.
  - **The controller sizes.** Predicted median 800 raw (Every) and 300 (Plan); measured 198 and 57.
  - **The median wall.** Predicted 0.2 s; measured 32 ms among the undecided.

  The pilot overstated the cost.
- Walt was not compared in S0.

**What this says.** At two or three tricks left, a seat can play exactly under a declared
one-type model with the counted posterior. Every decision is checked by a second construction and
is time-consistent with its own plan. A typical decision costs 10⁸–10⁹ instructions, and the
controller it carries is small. What it does not say: anything about strength against Walt, or
about any model but the uniform kernel. The calibration gate as run is a model-fit reading, not a
construction check.
