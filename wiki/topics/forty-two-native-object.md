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
