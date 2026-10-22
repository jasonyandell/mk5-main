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

**Lean** (`scenario/lean/Stab/Hexagon.lean`, `Residual.lean`, `Mixing.lean`, `Plans.lean`). There are 24 headline theorems, all on the three standard axioms, with no sorry and no Mathlib; the statements are in `scenario/lean/README.md`.

- **Local algebra.** With `x = E[A − B]`, `y = E[C − B]` and `β = E[B]`: Q31 = β + max(x, 0), Q41 = β + max(y, 0), Q51 = β + max(x, y). The second decisions compare x, y and x − y.
- Mixing the six programs gains nothing, and no linear scalar statistic carries both numbers.
- A six-type belief realizes (x, y) exactly on the hexagon |x|, |y|, |x − y| ≤ 1.
- The sign patterns of (x, y, x − y) number exactly 13, and they are the sequential decision classes. The note's segment lies on 19x − 23y = 4 and crosses five classes, split at p = 1/5.
- **Residual congruence.** It is the coarsest right congruence that keeps outputs, and an exact realization never merges two of its classes. The quotient machine attains it.
- Refinement round k is exactly agreement on block words of length at most k. With the HALT/REJECT totalization and at most n own tiles, refinement is exact after n − 1 blocks.
- **Mixing boundary.** The observation mass (1 + 18p)/560 is injective in p. Take any relation that keeps the decision sgn(5p − 1) and is kept by every convex mixture: on [0, 1] it is equality. The witness is explicit, λ = 1/5.
- **Fixed plans (the note's five-policy argument).** A fixed plan's value is a line in p, and the optimum is the upper envelope of those lines. The parameters where a plan is optimal form an interval. Where one plan is optimal across a piece, any plan optimal at an interior point is optimal across it too. So the fewest fixed plans optimal at every p is the number of affine pieces of the optimum.

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

## The whole ending, in Silas (2026-10-22)

`silas/tests/controller.rs` builds the note's ending from Silas's own frame, worlds and trick rules: seat 3 at 31 no-trump, holding 2-1, 3-1, 4-1, 5-1 before its trick-4 play. The frame leaves 560 deals. The partner is High with probability p and Low otherwise; the opponents play low. `silas/src/family.rs` solves the course once for every p, exactly. The night's Python prototype, written separately, gave the same numbers.

**The note's numbers, reproduced.**
- **Thresholds.** The ending has 24,089 information nodes and the note's 14 thresholds: 1/16, 1/7, 2/11, 1/5, 1/4, 1/3, 2/5, 5/12, 3/7, 1/2, 7/13, 4/7, 2/3, 8/11.
- **Witnesses.** Every threshold is witnessed: some decision has disjoint maximizing sets on its two sides.
- **Classes.** There are 31 parameter classes, and the root's best play is 2-1 throughout.
- **The two gaps Lean took as given.** The local history's mass is (1 + 18p)/560. After 3-1, the gap between 4-1 and 5-1 is (5p − 1)/(1 + 18p). Both are derived from the rules, no longer defined. The note's 0.08 and 0.12 also share a cell: every decision has the same maximizing set at both.

| Controller | Contexts | Raw states | Root ladder | Shape | Fork |
| --- | ---: | ---: | --- | ---: | ---: |
| every history | 722,863 | 767 | 1, 18, 31, 31 | 445 | 76 (root 1, 3, 22, 22) |
| 2-1 committed (5,935 nodes, 10 thresholds, 23 classes) | | 302 | 1, 10, 23, 23 | | |

**The note's message counts, checked.** The root's optimum, a convex function of p, has five affine pieces, with knots at 1/16, 1/5, 1/3 and 1/2. So, as the note says, five fixed policies are optimal from the root and fewer cannot be. Lean now carries the note's argument (`Stab/Plans.lean`), and Silas checks each piece's plan against the envelope at both ends. Every threshold has a witness history whose only interior switch it is, with disjoint maximizing sets on its two sides, so any two open cells need different policies. That gives the note's 15, and 11 with 2-1 committed. Its 31 and 23 are the raw root classes, and 769 and 304 add HALT and REJECT.

**New: good play meets only the knots of the optimum.** An ex-ante optimal plan is optimal at every history it reaches with positive mass. So along the plan's own path, the maximizing set can change only at those four knots, and 1/5 is the note's own threshold. The other ten thresholds move the best play only at histories reached after a play the plan never makes. Kept to the plan's path, the controller shrinks:

| Reach | Contexts | Raw states | Root ladder | Root classes are |
| --- | ---: | ---: | --- | --- |
| every history (maximizing sets) | 722,863 | 767 | 1, 18, 31, 31 | the 31 classes |
| the plan's path (maximizing sets) | 26,533 | 142 | 1, 10, 11, 11 | 5 pieces, 4 knots, the 2 pure partners |
| the plan's path (the plan's play) | 26,533 | 132 | 1, 5, 7, 7 | 5 plans, the 2 pure partners |

At p = 0 and p = 1 one partner type has no mass, so fewer histories live; those two are classes of their own. The test asserts each partition directly against the envelope's knots. The note's gap between 5 messages and 15 already says the extra cells come from deviations; the path controller says the same in states. **Most of the 31-class atlas is counterfactual.** It describes what the viewer should do after its own mistakes. A viewer who plays well needs five plans and a 132-state controller, and on its path the best play changes only where the optimum bends.

**How many numbers the viewer keeps.** Take a free belief over the 1,120 deal-and-type atoms. The comparison dimension of a decision is the fewest linear coordinates that keep every comparison of the viewer's programs. Each program's value is a 0/1 vector over the atoms, so this is the dimension of the span of the program differences (`silas/src/comparison.rs`).

| Decision | Histories | Comparison dimension |
| --- | ---: | --- |
| the note's local history (after 2-1, 1-0, 6-2, 5-2) | 1 | 2, the note's x and y |
| the root | 1 | 168 |
| second | 160 | 0: 8, 1: 10, 2: 83, 3: 43, 4: 14, 5: 2 |
| third | 480 | 0: 65, 1: 415 |
| fourth | 23,448 | 0 |

The note's "two numbers per decision" holds at its local history and is typical one decision deeper. It is not a property of the ending as a whole. Under a free belief the root needs 168 numbers. The one-parameter family walks a curve through that space, and it crosses only 14 of its walls.

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

- **Prose, not Lean: the count of fixed plans.** That the fewest is the number of pieces is two Lean lemmas plus a two-line argument; there is no `pieces` definition.
- **Measured, not proved:** the path statement (that good play meets only the knots) is a standard argument (an optimal plan cannot be improved at a history it reaches with positive mass) checked in Silas; it is not formalized.
- **Measured, not proved:** that the raw and fork ladders of the Gus course stop at one block. That is a fact of this course.
- **Scoped:** every fixture number is for one ending, one partner family (Low/High by tile order) and opponents who play low. The arena seats no such partner; the family is the note's, chosen to be exactly solvable, not a model of a person.
- **Applied since (2026-10-22, #244):** the family form now reads Walt's real endgames whole. Walt's tile is
  the exact optimum of its own model wherever it enumerates, and within noise where it samples
  ([[walt-endgame-probe-2026-10-22]]). Feasibility items 1 and 2 are #242 and #243.
- **Untouched:** nothing here reads a net. The quotient is a property of a course and an interface, as [[gus-decision-quotient-2026-10-21]] says, not of a learner.
