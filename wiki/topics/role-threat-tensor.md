---
title: Role threat tensor — domino roles as monotone bitmasks
kind: topic
first_seen: 2026-07-22
last_updated: 2026-07-22
status: active
---

## The idea (Jason, 2026-07-21, late night)

A domino's role — walker, high trump, good lead — is not an attribute but
a relation to what's still out. Make it mechanical: precompute

    THREAT[decl][d] = 28-bit mask of tiles that beat d when d is LED

(10 decls × 28 tiles × 4 bytes ≈ **1.1 KB**, derived from `walt.tables`,
rules never reimplemented). Then with `out` = the outstanding tiles:

- walker(d) ⟺ `THREAT[decl][d] & out == 0` — the [[count-fate-ledger]]
  walker, one AND
- threat_count(d) = `popcount(THREAT[decl][d] & out)` — lead quality as
  a number that only counts down
- `out` only shrinks ⇒ threat masks only empty ⇒ **roles only promote**
  (gate R4). A domino's position-power is a monotone lattice walk, which
  is the formal content of "as dominoes are played, we eliminate members
  of the set."

The decl-unknown hand is the 28×D stack — role columns for every game
declaration at once; bidding is column selection. Code: `roles/`
(`threat.py`, probes in `probes.py`, gates R1–R5 in `roles/tests/`).
Gates include the rules authority itself: R2 drives 5,000+ constructed
tricks through `forge.oracle.tables.resolve_trick` and demands the mask
predict the winner relation; R1 pins popcounts to `walt.tables.
beat_count`; R5 pins the injectivity facts below.

## Measured verdicts (2026-07-22; receipts `scratch/roles_probes.json`)

**1. The alphabet: exact interchangeability at the auction is ZERO —
the brainstorm's horizon prior runs BACKWARD.** Under the full
behavioral key (trick rank under all 8 led suits + count), every game
declaration separates all 28 tiles (28/28 distinct, every decl). No two
tiles are cold-interchangeable anywhere. Merging appears only as
distinguishing columns die: the [[endgame-equivalence-census]] measured
25/200 roots with a true interchangeable pair at H4 (~3% of non-forced
slots). So identity COARSENS toward the endgame while role-power
PROMOTES toward it — two monotone walks in opposite directions. The
lead-power-only key (rank when led + count) is much coarser — 13–20
distinct roles/decl, 55–83% of random hands hold a duplicate — but it is
lossy (equal leads, different follows), and is the honest shadow of the
"roles repeat" intuition.

**2. Factorization: role coordinates beat pip coordinates where the
hand matters.** On 4,000 fresh H4 roots with exact BR-vs-lowest-legal
values (net-free, `compile_rule_sigma` + `br_solve`, ~30 ms/root),
ridge + 5-fold CV + paired bootstrap:

| target | scalars | roles (10 feats) | pips (28 one-hot) | verdict |
|---|---|---|---|---|
| root value | 3.836 | 3.838 | 3.849 | hand rep below noise — context dominates |
| root value SPREAD (max−min over my leads) | 1.640 | **1.554** | 1.625 | roles−pips **−0.071 [−0.094, −0.048]** |

Same information (the role map is injective — R5), different
conditioning: the value functional is measurably SMOOTHER in role
coordinates. That is the brainstorm's "simplification" as a number — the
compression is functional, not stateful (the census already killed
stateful).

**3. Policy shadow**: the exact best lead is a walker only **55%** of
the time a walker exists, and the min-threat tile 58% (chance 42%) —
holding the boss and cashing it are different decisions. Harvest timing
is real; that is the count-fate ledger's thesis arriving from an
independent instrument.

**4. Throughput receipt**: full 9-decl role stack for **1.23M hands/s**
(one numpy sweep, `decl_stack_features`) — auction-scale feature
extraction is free.

## Consumers (distill-for-what)

The auction lane ([#42](https://github.com/jasonyandell/mk5-main/issues/42)):
decl-stack columns as bid features / cross-decl weight sharing. The
[[count-fate-ledger]] referees: walkers and guards as mask predicates.
table42 narration and [[w42]] detectors: "that 6-4 just became a
walker" is a THREAT mask hitting zero, live — Roberson's vocabulary made
mechanical. Follow-up probes filed in the tracking issue: the
live-interchangeability horizon curve (auction 0% → H4 12.5%), and a
policy-factorization probe with a trained head instead of rules.

The tensor turned out to be the first brick of a larger claim — the
computational unit is the marked substructure, not the domino; see
[[forty-two-native-object]] (registered 2026-07-22, with the symmetry
receipt this page's R5 gate foreshadowed).

## Links

[[count-fate-ledger]] · [[endgame-equivalence-census]] ·
[[suit-algebra-spec]] · [[hoyt]] · [[w42]] · [[texas-42]] ·
[[forty-two-native-object]]
