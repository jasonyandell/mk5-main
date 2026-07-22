---
title: atlas — the game as an addressed space
kind: entity
first_seen: 2026-07-22
last_updated: 2026-07-22
status: active
---

## What it is

The from-scratch native substrate for [[texas-42]]: a hand as an exact
**coordinate**, play as **node deletion**, the unknowns as an exactly
enumerable **fiber**, a domino's role as the state of a precomputed threat
relation. Code at `atlas/` (Python, numpy only — no torch, no numba). A
library, not a solver: no nets, no players, no belief model beyond the
uniform default in v1 (`atlas/SPEC.md`, ratified by Jason 2026-07-22 —
"a fresh from-scratch no-retrofit build, not because perf, but because
correctness").

atlas is the first BUILT instance of the [[forty-two-native-object]]
program: the representation that page argues for, made into an addressed
object with parity gates against every existing authority. Rules are never
reimplemented — every plane in the algebra derives once from
`forge.oracle.tables`, the rule authority, and the gates enforce it. atlas
runtime imports no `walt` / `roles` / `hoyt`; those are parity authorities
the tests lean on.

## The modules

- **`algebra`** — the resident, immutable, decl-indexed rule planes, built
  once: `led`, `rank`, `can_follow` masks, `count`, `THREAT[decl][tile]`
  (beats-when-led) and the full `BEATS[decl][led_suit][tile]` family (the
  [[role-threat-tensor]] generalized from "when led" to any position), and
  the pips-2↔3 arrow as data. Indexed by `decl_id` 0..9; decl 8
  (doubles-suit) is present for table completeness, flagged non-game.
- **`coordinate`** — `CoordinateV1`, a frozen dataclass with a 53-byte
  fixed-width little-endian packed form; `address()` is blake2b-128 of the
  bytes — THE address, stable forever (a zeroed `history_digest` extension
  slot is reserved for signaling fields). `from_engine`, `transition` (node
  deletion from the viewer's information), `legal`, and `transport` (the one
  symmetry). team_points are stored declaring-team-first
  (orientation-canonical). Special contracts raise; `CoordinateV0Auction` is
  a pack/unpack stub for the auction lane
  ([#42](https://github.com/jasonyandell/mk5-main/issues/42)).
- **`fiber`** — `fiber(coord)` enumerates every hidden deal consistent with
  count conservation, played membership, and the voids matrix. The physics
  fiber, exact; belief is a measure a consumer hands in (the quarantine
  doctrine — the ONLY estimated object anywhere).
- **`roles`** — walkers / bosses / threat counts / count exposure for the
  viewer's hand or any hypothetical world hand at a coordinate, plus the
  decl-unknown 28×D auction stack. Mirrors (and will eventually absorb)
  `roles/threat.py`.
- **`narrate`** — coordinate deltas as family-vocabulary events ("the 6-4
  became a walker", "seat 2 shown void in fives", "trick banked N"); pure
  functions, the count-fate referees' hook.

## The gates (all green, `atlas/tests/`, no skips)

Correct by construction, gated against the authorities — zero perf ambition
in v1. Full suite 33/33.

- **A1 / C5 resolve authority** — atlas trick resolution == `forge.oracle`
  `.tables.resolve_trick` on exhaustive tricks (>1,500, all 10 decls); the
  threat plane is bit-identical to `roles/threat.py` and every role
  predicate matches it on 3,000 random states.
- **C1 pack/unpack** — 10k fuzz round-trip, version-byte enforcement,
  address stable across a subprocess.
- **C2 engine parity** — 550 random full games × 4 viewers, all 10 decls
  incl. 8: `transition` from the observed action equals `from_engine` of the
  engine's next state field-for-field, and `legal` equals the engine's legal
  set for the viewer to act.
- **C3 fiber parity** — atlas.fiber == `walt.worlds.enumerate_worlds` on the
  equivalent engine-state root at >200 trick boundaries; mid-trick fibers
  vs an independent brute-force filter (incl. decl 8); the monotone-shrink
  law `fiber(transition(c,t)) ⊆ fiber(c)` on >200 checks.
- **C4 sufficiency spot-gate** — 20 engineered pairs of DIFFERENT histories
  reaching an EQUAL coordinate (adjacent tricks both led and won by one
  seat, swapped; 240 available in the search window) give identical [[hoyt]]
  best-response values to 1e-9 — a measured check that play order carries no
  game information against a non-signaling field.
- **C6 the symmetry arrow** — `transport` is an involution on every
  coordinate and commutes with `transition` for decls 2 and 3 (where gate R6
  makes 2↔3 a game isomorphism) across >300 steps, addresses included.

## Consumers (later; not v1)

jud-next featurization, the auction decoder
([#42](https://github.com/jasonyandell/mk5-main/issues/42)), the
[[count-fate-ledger]] referees, [[w42]] detectors, and table42
narration. Perf comes after correctness; v1 has none by design. Tracking:
[#90](https://github.com/jasonyandell/mk5-main/issues/90).

## Links

[[forty-two-native-object]] · [[role-threat-tensor]] ·
[[endgame-equivalence-census]] · [[texas-42]] · [[hoyt]] · [[walt]] ·
[[engine]]
