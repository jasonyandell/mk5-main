---
title: The fundamental factorization of 42 — algebra × cells × record × delimited ignorance
kind: topic
first_seen: 2026-07-22
last_updated: 2026-07-22
status: active
---

## What it is

The founding decomposition, ratified in discussion (Jason, 2026-07-22, the
PR #92 review session): every situation in 42 factors as **algebra ×
location cells × public record × delimited ignorance**, and every motion of
the game is a monotone refinement of the factors. [[atlas]]'s
`CoordinateV1` is verified as one valid *residue* of this structure — the
factorization demotes the coordinate from "the native representation" to a
quotient of it, and names the primary object: the marked substructure on
the shared trunk ([[forty-two-native-object]]), with each seat's unknowns
as delimited subset cells. Analogy that guided the session: unique prime
factorization — not the last theorem, the first one; the thing everything
else is built on.

## The congruence (what gate C4 actually establishes)

- **Dynamics factor through the coordinate by construction.** `transition`,
  `legal`, `fiber`, `roles` are functions of the packed bytes; the code has
  no channel to the path. Same coordinate ⇒ same future evolution, exactly
  — a congruence, not an empirical tendency.
- **Value factors through the coordinate as a theorem with a scope.**
  Against a field whose behavior depends only on situations
  (non-signaling), induction down the subgame gives value = f(coordinate).
  Against a history-reading field it does not — there the finest valid
  coordinate is the whole path.
- **C4 is therefore a leakage check on the implementation, not a
  probability estimate.** Verified live 2026-07-22: seed 60013, two 20-play
  histories (adjacent same-leader tricks swapped), all four viewer
  addresses byte-identical, [[hoyt]] best-response values equal to 0.0
  (`atlas/tests/test_sufficiency.py`).
- The census's "essentially never" ([[endgame-equivalence-census]]) is a
  *different* equivalence — cross-coordinate value coincidence — and is not
  load-bearing anywhere in this structure. The real within-game equivalence
  is the within-hand tie (29/29 bitwise), which is the same-hand
  localization of the intuition.

## Coordinate vs path

The coordinate keeps exactly the physics residue of history (voids, counts,
banked points) and discards exactly the choice evidence. Belief
([[belief-policy-value-algebra]]) is a functional on the discarded part: a
path is `(coordinate₀, action sequence)`, its likelihood decomposes as a
product of per-step discretionary terms, each computable locally at its own
coordinate, and the resulting measure tilts the endpoint's fiber. Physics =
fiber(coordinate); evidence = likelihood(path). The C4 pairs are the
natural instrument for the **history premium** — identical physics,
potentially divergent posteriors under a discretionary field (probe P-HIST,
[#94](https://github.com/jasonyandell/mk5-main/issues/94)).

## The tower of quotients

How coarse a coordinate may validly be is **field-relative**: value factors
through a quotient exactly when the opposing field is blind to what the
quotient drops.

1. Full path — required against signaling fields.
2. `CoordinateV1` — valid against non-signaling fields (gated, PR #92).
3. Attribution-coarse — `played[4]` → (union, per-seat counts, voids): no
   rule consults *who* played a past tile beyond counts and voids (atlas
   `roles.py` reads only the union; `fiber` consumes only pool + counts +
   forbidden masks). Candidate congruence, untested (probe P-ATTR,
   [#94](https://github.com/jasonyandell/mk5-main/issues/94)).
4. Payoff-stripped — `dealer` is strategically inert in play; `team_points`
   factor additively under the points lens and collapse to a
   remaining-threshold under the marks lens; `bidder` matters only as team
   orientation.
5. The floor — the marked substructure on the trunk: the part no field
   class allows dropping.

Corollary: a `CoordinateV1` address means *same engine situation*, not
*same strategic situation* — it is finer than the value-relevant residue
for every sane field. The SPEC's "naive addresses are already canonical"
is scoped to tile relabelings (census gate R6), not seat symmetry
([#93](https://github.com/jasonyandell/mk5-main/issues/93)).

## The founding sequence (from the 0 object)

The 0 object: 28 nodes, all ten algebras latent — before a declaration,
absorption and power are a stack of potentials (why
`decl_stack_features` refracts one hand through every decl). Then, each
step an information event: seats + teams (the frame) → the deal (a
location partition of the 28 — conservation is the invariant: every tile in
exactly one place, which is what makes "I hold the 5-3 so nobody else does"
inference) → the auction (public record AND the game's purest discretionary
evidence: no voids exist yet, so nearly everything a bid says is choice) →
**the declaration selects the algebra** (ten latents collapse to one; the
moment a hand becomes a marked substructure of a single algebra) → play
(deletion on the trunk, each play also an information event).

## The factorization

Every situation = **algebra** (selected by the declaration) × **location
cells** (the trunk: where every tile is) × **public record** (append-only:
bids, plays, revealed voids) × **delimited ignorance** (per-viewer cells
over the hidden part). Motion is monotone refinement, always: the pool
depletes, the record appends, fibers shrink, roles promote
([[role-threat-tensor]]). Everything estimated — belief, policy, value — is
a function *on* the factors, never a fifth factor. The uniqueness and
completeness edges are testable with C4-pattern congruence probes
([#94](https://github.com/jasonyandell/mk5-main/issues/94)).

## The delimited-unknown primitive

A location cell is `(P, k)`: some k-subset of possible-mask `P`, subject
unknown.

- Own hand, played pile, current trick: the degenerate case `|P| = k`.
  "Known" is not a different type — it is the tight end of one slide.
- **Upper-bound-only physics** (points games): plays remove tiles from `P`;
  voids remove whole follow-sets; nothing in the rules ever produces a
  must-hold constraint (following consumes its own witness — the follower
  proven held is the tile just played; leading proves nothing). Hence the
  intension — pool `U`, per-hidden-seat `(P_s, k_s)`, disjointness —
  generates the fiber **losslessly**. `atlas/fiber.py` already consumes
  exactly this tuple; the enumerated `(N, 3)` array is a query result, not
  the state.
- Transition law, uniform across all four seats: seat `s` plays `t` ⇒
  `U ← U∖{t}`, `k_s ← k_s − 1`, `P_s ← P_s∖{t}` minus any revealed
  follow-set (strictly smaller, always — a play replaces one unknown with a
  strict subset); other cells lose `t` (weakly smaller). "Node deletion" is
  this law evaluated at `|P| = k`.
- Consequences: the early-game fiber scale wall (≈4×10⁸ worlds at play 1)
  dissolves — carry the intension, refine monotonically, enumerate on
  demand once shrunk; consistency is a bipartite feasibility check
  (Hall / max-flow) with no enumeration; exact counting stays genuinely
  hard (a disjoint-representatives count). The [[walt]]/[[hoyt]]
  endgame-exact regime is the degenerate limit of the same object, not a
  separate land.

## Epistemic shape

42's knowledge structure is: **one common public record** (everything
happens face-up on the table) **plus one private mask per seat**. Viewers'
cells differ only by subtracting their own hand from the pool; all
higher-order knowledge ("I know that you know…") collapses to this pair,
and hypothetical reasoning is exact — a viewer reconstructs any other
seat's cells perfectly per supposed world, which is what makes
suppose-and-search well-posed. A game state = trunk + a partition of the
live marks into four hands; a perspective = trunk + one part + cells over
the rest; "viewer" is which marks are given, not a structural field.

## Hands in situ

A hand has no context-free coordinate: role-hood is relational — against a
live complement, under a declaration. "The same hand in another game" is a
morphism (a compatible transport — 2↔3 the one global instance, within-hand
ties the local ones), not an equality. Proximity is intrinsic: the geometry
is generated by legal transitions — the closest hands to a hand are the
ones a play reaches. Any other metric is a modeling claim requiring the
faithfulness check (probe P-METRIC,
[#94](https://github.com/jasonyandell/mk5-main/issues/94)).

## Links

[[atlas]] · [[forty-two-native-object]] · [[endgame-equivalence-census]] ·
[[belief-policy-value-algebra]] · [[role-threat-tensor]] ·
[[count-fate-ledger]] · [[walt]] · [[hoyt]] · [[texas-42]]
