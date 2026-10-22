---
title: "The minimal comparison algebra in Lean and Silas"
kind: experiment
first_seen: 2026-10-21
last_updated: 2026-10-22
status: complete
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question

Jason's note *A minimal comparison algebra for a declared 42 family* (2026-10-21) makes two claims. The first is local: holding 3-1, 4-1 and 5-1, the viewer's six programs collapse to three indicator vectors, and every decision reads two numbers. The second is global: a solved course reduces to its coarsest output-keeping right congruence, the decision quotient. Night-Math's job (night of 2026-10-21) was to make both the project's own: Lean statements first, then a native implementation that reproduces [[gus-decision-quotient-2026-10-21]] exactly, then tests. Nothing was trained and no model was loaded.

## Result

**Lean** (`scenario/lean/Stab/Hexagon.lean`, `Residual.lean`, `Mixing.lean`). There are 22 headline theorems, all on the three standard axioms, with no sorry and no Mathlib; the statements are in `scenario/lean/README.md`.

- **Local algebra.** With `x = E[A − B]`, `y = E[C − B]` and `β = E[B]`: Q31 = β + max(x, 0), Q41 = β + max(y, 0), Q51 = β + max(x, y). The second decisions compare x, y and x − y.
- Mixing the six programs gains nothing, and no linear scalar statistic carries both numbers.
- A six-type belief realizes (x, y) exactly on the hexagon |x|, |y|, |x − y| ≤ 1.
- The sign patterns of (x, y, x − y) number exactly 13, and they are the sequential decision classes. The note's segment lies on 19x − 23y = 4 and crosses five classes, split at p = 1/5.
- **Residual congruence.** It is the coarsest right congruence that keeps outputs, and an exact realization never merges two of its classes. The quotient machine attains it.
- Refinement round k is exactly agreement on block words of length at most k. With the HALT/REJECT totalization and at most n own tiles, refinement is exact after n − 1 blocks.
- **Mixing boundary.** The observation mass (1 + 18p)/560 is injective in p. Take any relation that keeps the decision sgn(5p − 1) and is kept by every convex mixture: on [0, 1] it is equality. The witness is explicit, λ = 1/5.

**Two claims in the night's brief are false, and Lean proves it.**

1. **The thirteen.** The root triple (Q31, Q41, Q51) takes only 4 weak orders (`four_root_orders`), because Q51 is never the unique maximum. The 13 are the weak orders of (E[A], E[B], E[C]), which are the sequential classes. The note itself says this; the brief's paraphrase did not.
2. **One block.** "The last tile is forced, so refinement stabilizes after one block" is false. The bound is n − 1 blocks, and `Witness.one_block_not_enough` is a three-tile course that needs two. [[gus-decision-quotient-2026-10-21]] had said every ladder stabilizes after one block; its own shape ladder, 11, 462, 477, 477, needs the second, so that line is corrected there.

**Silas** (`silas/src/quotient.rs`, `silas quotient --course FILE`). The quotient is generic over tiles and observations, and every value is an exact BigRational. It is built twice, by bottom-up signatures and by Moore refinement, and is refused if the two disagree. It runs under the raw, shape and fork interfaces. On the 256 eligible maskbreadth roots it reproduces `gus-decision-quotient-v1/report.json` (sha256 14fde1ac…) in every count, with zero differences:

| Interface | States | Ladder by block depth | Root ladder |
| --- | ---: | --- | --- |
| raw | 8,257 | 753, 8,257, 8,257 | 253, 256, 256 |
| shape | 477 | 11, 462, 477, 477 | 10, 237, 252, 252 |
| fork | 144 | 11, 144, 144 | 10, 139, 139 |

Every other count matches too: nodes by stage and ply, menus, the discretionary tallies, the 13 weak orders at three-way nodes, per-root states and compression, classes shared across roots, and coverage. Silas takes 0.4 s and 4.3 × 10⁹ instructions; the Python reference takes 3.6 s and 6.9 × 10¹⁰ (both under `scripts/cpu-slot`).

## Where the quotient lives, and why

The brief asked for a decision on placement: a new Silas module, or an exporter beside the existing Python. The answer is both, split at a neutral file.

- **Silas holds the mathematics.** `quotient.rs` is a generic module, `Course<A, X>` over any ordered tile and observation types. It holds no Gus tokens and no 42 rules, so it serves Silas's own solved endings as well as a Gus course. It belongs in Silas because Silas is where exact beliefs, information states and exact rationals already live, and its arithmetic is exact by construction (clippy denies float arithmetic).
- **Python keeps the reading of Gus's files.** `lem/quotient.py --export` writes `silas-decision-course-v1`: menus, maximizing sets, q, masses and block labels. The hidden hands in `teacher.json` stay behind. The `course` feature reads that file and re-audits mass, Bellman and argmax exactly.
- **The design leaves the gated core alone.** It touches no Silas core pin, `physics.rs` or `frame.rs`. The Lean files are additive proofs in `scenario/lean`, where the brief put them, and touch no Walt code.

## Tests

- **`silas/tests/quotient.rs`.** All six tests run in under a second:
  - the Lean witness, whose raw ladder 3, 4, 5, 5 needs two blocks;
  - the note's hexagon on 61 lattice points, with 4 root outputs and 13 classes after one block, one class per sign pattern;
  - shape against fork;
  - six malformed courses, each refused;
  - a broken Bellman row, refused;
  - real root test-0007 (52 decisions), checked against the Python counts.
- **`lem/tests/test_quotient.py`.** It adds a check that the export is the course without hidden state.

## Scope: what is not proved

- **Not reproduced: the note's 767-state controller and its 1, 18, 31, 31 ladder.** These need the 560-deal fixture, which is a computation, not a theorem; it is the night's stretch goal.
- **Taken as given:** that (5p − 1)/(1 + 18p) is that fixture's gap is the definition of the threshold, not a derivation.
- **Not proved:** that 0.08 and 0.12 share a cell of the note's atlas. Lean proves only that both lie between 1/16 and 1/7.
- **Measured, not proved:** that the raw and fork ladders of the Gus course stop at one block. That is a fact of this course.
- **Untouched:** nothing here reads a net. The quotient is a property of a course and an interface, as [[gus-decision-quotient-2026-10-21]] says, not of a learner.
