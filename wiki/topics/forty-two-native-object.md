---
title: The 42-native object — hand as marked substructure, play as node deletion
kind: topic
first_seen: 2026-07-22
last_updated: 2026-07-22
status: active
---

## The claim (Jason, 2026-07-22)

The project has assumed that because actions are individual dominoes, the
natural computational unit is an individual domino. The claim: it isn't.
A hand is a **seven-node marked substructure of the game algebra**, and
action selection is **choosing which node to remove**, valued by the
structure that remains and the world transition (observation) the
removal causes. The unit is the structure; the domino is just its
handle.

## Why 42 specifically (the card-game contrast)

In a card game, a king of hearts stays a king and stays a heart; trump
adds power but does not relocate the card into a different
legal-following structure. In 42 that relocation is the most basic
mechanic: `can_follow = contains(pip) AND NOT in_called_suit` — declare
twos and the 6-2 is not a stronger six, it is **no longer a six at
all**. And the base structure is a *covering*, not a partition: every
mixed tile inhabits two suits at once ([[suit-algebra-spec]] §2). Suit
in 42 is a relation between the tile and the contract, not a property
of the tile.

## The receipts (all mechanical, all gated)

1. **The game reads only structure** — the
   [[endgame-equivalence-census]] residual signature carries everything
   the game can consult and no pip identity. Theorem-grade.
2. **Structure cannot be cashed as state-merging** — census: world/root
   compression exactly 1.000×; within-hand ties ~3% of non-forced slots
   at H4.
3. **Structure cannot be cashed as symmetry** — measured 2026-07-22
   (gate R6, `roles/tests/test_symmetry.py`): of 5,039 nontrivial pip
   relabelings, exactly ONE is a game isomorphism — 2↔3, transporting
   the twos-game onto the threes-game, surviving only because the 2-3
   tile is trump in both (the higher-end lead rule kills it everywhere
   else; count marks kill everything else). Contrast bridge: an S₃ of
   interchangeable non-trump suits per contract. 42 is measurably about
   the least symmetric trick game constructible — BECAUSE identity is
   contract-entangled.
4. **Therefore the only cashable form is the function class** — and its
   first measured instance exists: the [[role-threat-tensor]] spread
   probe (n=4000 exact values, role basis beats pip identity, dMAE
   −0.071 CI [−0.094, −0.048]). Functions of this game are simpler in
   structural coordinates even though states never merge.

The formula's second clause closes the triangle: "the world transition
it causes" is an observation partitioning belief over the ambient
structure (the 21 hidden nodes) — value = f(belief over structure, my
substructure, context), which is [[belief-policy-value-algebra]] /
otis's value = f(belief, policy, context) arrived at from the
representation side. The deletion-sequence view of a hand IS the
[[count-fate-ledger]]: a node's fate is when and how it leaves the
structure.

## How big is the object (measured 2026-07-22, 400 deals × Knuth estimator)

One complete hand, one seat's view: C(21,7)·C(14,7) ≈ 4×10⁸ worlds.
Naive per-world path bound (7!)⁴ ≈ 10¹⁴·⁸; MEASURED legal sequences per
deal: median **10⁹·⁷** (p90 10¹⁰·⁷) — legality alone narrows five orders
of magnitude. Branching by play position is a heartbeat: leads carry the
width (7.0 → 6.0 → 5.0 → … → 1.0 per trick), follows hum along at ~2.6
early and collapse to 1.0 by the last trick. **39.5% of all plays are
forced; 48% of follows are forced and 73% have ≤2 choices; a hand is
~17 real decisions dressed as 28 plays.** One decl per hand — the ×9
decl axis lives only in the auction's belief, never in the play object.

Datacenter arithmetic ($10⁹ ≈ 100–250 PB of RAM): the HISTORY-tree
representation of one hand is tens of exabytes — no by ~100×. The
native state-DAG with forced-chain collapse (the compression [[hoyt]]
already performs bitwise at H4, where 83% of info sets are forced) is
~10¹⁴–10¹⁵ slots ≈ **2–20 PB — one hand of 42 fits in a fraction of
one datacenter, but ONLY in native coordinates**. The representation
choice is worth the entire distance between absurd and buildable; and
nobody needs to build it (CFR gap ≤0.05 in ≤40 iters, world-scale-
invariant — the cascade exists so this bill is never paid).

## atlas exists (2026-07-22)

The representation this page argues for is built: [[atlas]] (`atlas/`,
Python/numpy) is the founding v1 — a hand as an exact `CoordinateV1` with a
canonical blake2b-128 address, play as `transition` (node deletion), the
unknowns as an exact `fiber`, roles over the resident decl-indexed algebra,
and the 2↔3 arrow as `transport`. It is a library (no solver, no nets, no
players; belief quarantined to the uniform default), correct by construction
and gated against every authority: engine parity on 550 games × 4 viewers,
fiber parity vs [[walt]]'s enumerator, role parity vs the
[[role-threat-tensor]], and — the receipt this page's sufficiency claim
wanted — C4: 20 different histories reaching an equal coordinate give
identical [[hoyt]] values to 1e-9. No perf work in v1 by design; the
P-ENC/P-DEL/P-DECL program and the consumers below build on it.

## The perf stance

Where the wrong unit is paid for today: NOT the engine (hoyt/walt
already compute on masks and relations — the native object at the
feature level runs at 1.23M hands/s). The compensation is paid in the
learned stack: per-decl relearning of the re-wiring, capacity spent on
hand-order invariance, domino-by-domino feature recomputation where one
AND updates a neighborhood. Prior (registered, Jason concurring): the
native object will be FASTER end-to-end because it solves the right
problem — consistent with the perf-log's own history, where the largest
wins were representation corrections (forced-slot compression,
in-struct pricing), not micro-optimization.

## The falsifiable program

- **P-ENC (decisive, priors below)**: one small policy head, two input
  bases — nodes-with-algebra-relations (no pip embeddings) vs
  pip-identity embeddings — same task: match exact best moves on
  generated H4 roots (targets free via `compile_rule_sigma` +
  `br_solve`, ~30 ms/root). Registered priors: structural encoder
  reaches pip-encoder accuracy with fewer parameters and better sample
  efficiency at n ∈ {500, 2k, 8k}; refuted if pip wins at every n or
  parity requires more params. Either verdict banks.
- **P-DEL**: the deletion derivative — how much of value decomposes into
  role-local first-order terms (the ledger's role-weighted
  participation) vs second-order interactions (guard-walker).
- **P-DECL**: cross-decl sharing — one structural encoder with decl as
  re-marking vs per-decl heads; the R6 receipt says literal isomorphism
  is absent, so any sharing that works is function-class sharing.

Consumers: jud-next featurization, the auction decoder
([#42](https://github.com/jasonyandell/mk5-main/issues/42)), the
count-fate referees, table42 narration. Tracking:
[#90](https://github.com/jasonyandell/mk5-main/issues/90).

## Links

[[role-threat-tensor]] · [[endgame-equivalence-census]] ·
[[suit-algebra-spec]] · [[count-fate-ledger]] ·
[[belief-policy-value-algebra]] · [[texas-42]] · [[the-wall]]
