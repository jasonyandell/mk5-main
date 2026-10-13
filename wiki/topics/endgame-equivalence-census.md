---
title: Endgame equivalence census — worlds never compress
kind: topic
first_seen: 2026-07-20
last_updated: 2026-10-13
status: complete
---

Can exact equivalence classes over hidden dominoes shrink the endgame solve space with fully correct results? No. **Swapping two hidden tiles never preserves value, at any depth.** The one exact equivalence lives inside the viewer's own hand. A proposal to "collapse equivalent hidden tiles to solve the middle of the hand" is answered by this page.

Measured 2026-07-20 on the frozen 200-root H4 evalset (the endgame solver `hoyt`, branch `origin/worktree-equiv-census`, not on main) with a certified instrument, `hoyt/equivcensus.py`.

## The co-occurrence theorem

In a 4-seat trick game, any two comparable hidden tiles co-occur in a trick in some consistent world, one in each of two seats. Their relative rank is therefore always game-live, and no exact swap of hidden tiles exists. The only tiles that can never resolve a trick against each other are two tiles of one hand (one play per seat per trick), so interchangeability exists only within the observer's own hand.

The mechanism is depth-independent: world-level dedup is dead at every horizon, not only H4. The sole loophole, mutually incomparable "dead trash" pairs in entirely dead suits, occurred 0 times in 200 roots. Voids are irrelevant to the null; a no-void regime also found nothing.

Corollary: pip relabeling is not a per-deal game isomorphism. A mixed domino leads its higher end and in-suit ranks are pip sums, so only the unique monotone relabeling transports even the unscored play tree ([[suit-algebra-spec]] §9, [[play-phase-algebra]]). The game re-reads its pips at every mixed lead.

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

Dead: world-orbit quotients, cross-root dedup, orbit-aware world caps, and any "solve one representative per class" scheme over hidden-tile assignments.

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

The same null was reached earlier and independently in texas-42:

- `wiki/idea-hierarchical-fibers.md` §7, the "rung 2" probe: 28% of 160,012 swaps change the outcome.
- `walt/CENSUS-RULINGS.md:1364`, Corollary S-rigid (on paper): at the first play the seat-side quotient is the identity, 1,184,040 classes. "Compression in this project is bought with deadness."
- `lean/Texas42/Transport.lean:95,201` (mechanized): only the identity and the 2<->3 pip swap preserve count.
- 2026-10-13 re-measurement on random deals with the [[silas]] double-dummy solver (scratch code `scratch/detective-classes/`, not committed):

| position | compression | pattern pairs with differing values |
|---|---|---|
| start of trick 3 | 2.53x | 227 of 450 |
| start of trick 4 | 1.39x | 90 of 210 |

With one representative per pattern at trick 4, the best move flipped once in 13 positions and the mean error reached about 7 points.

## Links

[[walt]] · [[scenario-player]] · [[silas]] · [[suit-algebra-spec]] · [[play-phase-algebra]] · [[jud]]
