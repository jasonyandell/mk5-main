---
title: Endgame equivalence census — no hidden-moving symmetries in the tested roots
kind: topic
first_seen: 2026-07-20
last_updated: 2026-10-18
status: complete
---

The 200-root census found no hidden-moving automorphisms under its tested residual
signatures. It found useful equivalences within the viewer's hand. This is a scoped
measurement, not a proof that hidden worlds never compress or that hidden swaps never
preserve value. With one trick left and the viewer holding the highest trump, reassigning
the other three tiles leaves the winner and total trick points unchanged. Scalar value
equality is weaker than game isomorphism.

Measured 2026-07-20 on the frozen 200-root H4 evalset (the endgame solver `hoyt`, branch `origin/worktree-equiv-census`, not on main) with a certified instrument, `hoyt/equivcensus.py`.

## What co-occurrence does and does not establish

Tiles in one hand cannot meet in a trick. Hidden tiles in different hands can meet when
the capacities, void constraints and legal continuations permit it. Such co-occurrence
can expose a rank difference and defeat a proposed structural swap. It does not prove
that every swap changes the final payoff or optimal value.

No hidden-moving symmetry was found in this H4 sample, including its no-void regime;
the reported dead-suit loophole occurred zero times in 200 roots. Other roots, horizons,
query-specific quotients and accidental value equalities are not excluded by that census.

Arbitrary pip relabeling is not a symmetry of the full rule system. Preserving numeric-max
lead selection for every pair requires an order-preserving permutation, hence identity
([[suit-algebra-spec]] §9, [[play-phase-algebra]]). Special residual states may still admit
equivalences; this global rule observation is not an impossibility result from the census.

## The instrument

The play phase touches tiles only through led suit, follow-legality, trick-order comparisons and count ([[suit-algebra-spec]] §5-§7). The residual signature of a trick-boundary root encodes exactly that and no pip identity: per-lead legality rows, per-row dense trick ranks, counts, holders, void-forbidden bits, and a seat/payoff header. A signature isomorphism is a game isomorphism, so values transport exactly. Canonicalization is individualization-refinement over the 16 live tiles, gated on invariance under index relabeling (50/50) and exact recovery of planted symmetries.

## Verdicts

| quantity | measured |
|---|---|
| cross-root canonical classes (strict, points-payoff, relaxed, no-void regimes) | 200/200 distinct in every regime, 1.000x |
| within-root world orbits (all 200 roots) | **3,206,646 -> 3,206,646**, zero hidden-moving automorphisms |
| within-hand interchangeable pairs (relaxed signature) | 25/200 roots, 29 pairs |
| solver receipt for the 29 pairs | 29/29 bitwise value ties against a uniform field; non-paired moves differ, median 1.54 pts |
| pip-relabeled roots matching their originals | 0/200 |

The measured signatures supplied no reduction for world-orbit quotients, cross-root dedup
or orbit-aware world caps on this corpus. A new quotient still needs to prove preservation
of its chosen query, transition law or value; the census is evidence about expected savings,
not a universal obstruction.

## What survives: own-hand equals

Within-hand ties are provable exact Q-ties, found structurally in about 1 ms without solving ("my 2-2 and my 4-4 are the same domino now"). Consumers:

- **Solver testing:** value-tie invariants are a parity-style gate class; the 29/29 receipt doubles as a correctness witness.
- **Indifference detection:** provable argmax ties for [[jud]] serving and claim logic; tie-breaks among these pairs are convention, never value.
- **Narration:** the tie makes the family-table "equals" concept exact ([[w42]]).

CFR strategy among tied moves is not a valid tie observable: the RM+ max(0, .) kink amplifies floating-point noise between value-tied branches (4.5e-04 drift on an exact tie). Per-move values are.

Scope: exactness claims are for physics worlds with uniform weights. A sigma-filtered production line and capped world samples are not signature-respecting, and nets read pips.

## Sizing class-CFR

Class-CFR constrains each seat's strategy to be uniform across its own current equals at each info set, one regret accumulator per class. Sized across all reachable info sets (four seats, every depth), 10 stratified roots, world cap 64:

| quantity | measured |
|---|---|
| reachable info sets | 39,389,562 |
| forced (one legal move) | 82.6% |
| non-forced slots removed | **2.94%** (14,044,679 -> 13,631,733) |
| non-forced info sets with at least one merge | 6.00% |
| class-forced (one class, several legal moves) | 376,486 (5.50% of non-forced), nearly all at the 2-tile trick |
| per-root spread | min 0.63%, median 3.27%, max 4.79% |

Reduction by trick: leading 4.38%, second 3.67%, third 2.87%, last 0% (one tile each, fully forced). Incidence falls with depth. More worlds merge slightly less (cap 16: 3.22%; cap 64: 2.94%, same roots). The visible seat's slots reduce 0.68%; the three hidden seats 3.60%.

The per-info-set equals test drops the acting seat's own-hand tiles from rank comparisons (a tile the seat holds can never resolve a trick against the pair) and keeps all live tiles for follow-legality; a mid-trick guard adds the tiles on the table. Detection costs about 1 microsecond per info set.

Verdict: exact and cheap, but about 3% on top of the 83% forced-slot compression already in the fused lane. A correctness tidy and an indifference gate, not a width win. Receipts live on the branch: `scratch/equiv-census/class_cfr_census.py`, `class_cfr_results_n10.json`.

## Prior work

Related earlier evidence in texas-42, concerning different equivalence criteria:

- `wiki/idea-hierarchical-fibers.md` §7, the "rung 2" probe: 28% of 160,012 swaps change the outcome.
- `walt/CENSUS-RULINGS.md:1364`, Corollary S-rigid (on paper): at the first play the seat-side quotient is the identity, 1,184,040 classes. "Compression in this project is bought with deadness."
- `lean/Texas42/Transport.lean:95,201` (mechanized): only the identity and the 2<->3 pip swap preserve count.
- 2026-10-13 re-measurement on random deals with the [[silas]] double-dummy solver (scratch code `scratch/detective-classes/`, not committed; its sources are `zeb/detective-classes/` in [texas-42-zeb-research-2026-10](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10)):

| position | compression | pattern pairs with differing values |
|---|---|---|
| start of trick 3 | 2.53x | 227 of 450 |
| start of trick 4 | 1.39x | 90 of 210 |

With one representative per pattern at trick 4, the best move flipped once in 13 positions and the mean error reached about 7 points.

## Constellations: swapping a live tile for a dead one

The null above is about hidden *live* tiles. texas-42's constellations exchange a live tile
for a *dead* one with the same standing among the living, which changes the history and not the
remaining game. That equivalence holds exactly, and on 2026-10-14 it was rebuilt in the core as
a key for Zeb's late-hand plan values. It is exact when the field reads only constellations. It
is not for the deployed net, which reads dead tiles. Classes number about 10^6 with one trick
left and 10^8 with two. [[constellations-2026-10-14]].

## Links

[[constellations-2026-10-14]] · [[walt]] · [[scenario-player]] · [[silas]] · [[suit-algebra-spec]] · [[play-phase-algebra]] · [[jud]]
