---
title: "The exact player, S0: complete endgames of at most three tricks, exact under a declared model"
kind: experiment
first_seen: 2026-10-22
last_updated: 2026-10-22
status: active
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

## Results

Pending.
