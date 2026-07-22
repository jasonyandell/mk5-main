# atlas — the game as an addressed space (founding spec, 2026-07-22)

Ratified by Jason 2026-07-22, from the 2026-07-21→22 sessions: "a hand
as a coordinate as the native representation... a fresh from scratch
no-retrofit build, not because perf, but because correctness."

## The idea being captured (do not dilute this section)

The project has assumed that because actions are individual dominoes,
the computational unit is an individual domino. It isn't. The unit is
the **situation**, and a situation has an **exact coordinate**: the
knowns known, the unknowns not vague but *delimited* — an exactly
enumerable fiber of consistent hidden deals over an exactly encoded
public+private base point. A hand is a seven-node **marked substructure
of the decl-indexed game algebra**; play is **node deletion**; a
domino's role (walker, boss, lead quality) is the state of precomputed
threat relations against a monotonically shrinking outstanding set —
roles only promote. Suit is a relation between tile and contract, not a
property of the tile (`can_follow = contains(pip) AND NOT
in_called_suit` — declaring twos removes the 6-2 from the sixes).

Receipts already banked (wiki: [[forty-two-native-object]],
[[role-threat-tensor]], [[endgame-equivalence-census]]):
- the game reads only structure, never pip identity (census signature);
- structure cashes as neither state-merging (census 1.000×) nor
  symmetry (gate R6: exactly one nontrivial isomorphism in 5,039 — pips
  2↔3 transporting twos↔threes); the only cashable form is the
  function class, first measured instance = roles-beat-pips on the
  spread target (n=4000, dMAE −0.071 CI [−0.094, −0.048]);
- the object is thin: 39.5% of plays forced, ~17 real decisions per
  28-play hand, 10^9.7 legal sequences per deal (measured);
- the coordinate is small: a viewer's entire epistemic situation packs
  into tens of bytes, and the census's no-hidden-isomorphisms verdict
  means naive addresses are already canonical.

The estimated layer is quarantined by construction: the ONLY estimated
object anywhere is the belief measure on the exact fiber
([[belief-policy-value-algebra]]: belief = a tilt of uniform by the
field's discretionary likelihood). Everything else is exact.

## What atlas is

The from-scratch native substrate: coordinates, the decl-indexed
algebra, role state, transitions, and fibers — **correct by
construction, gated against the existing authorities, zero perf
ambition in v1** (perf comes after; correctness is the point). atlas is
a LIBRARY: no solver, no nets, no players in v1. Consumers (jud-next
featurization, auction lane #42, count-fate referees, w42 detectors,
narration) build on it later.

Python, numpy only (no torch, no numba in v1). Rules are NEVER
reimplemented: every rule fact derives from `forge.oracle.tables` /
`walt.tables` at build time (the walt CONTRACTS doctrine), and parity
gates enforce it.

## Modules

### atlas/algebra.py — the decl-indexed game algebra, resident

One immutable object holding, for every decl in
`forge.oracle.declarations.GAME_DECL_IDS` (+ DOUBLES_SUIT for table
completeness, flagged non-game):
- membership planes: `can_follow[decl][led_suit] : uint32 mask` (8 per
  decl);
- led-suit map `led[decl][tile]`, rank tables `rank[decl][led_suit][tile]`;
- count marks (decl-invariant) and the count-tile mask;
- threat masks `THREAT[decl][tile]` (beats-when-led) AND the full
  per-led-context beat family `BEATS[decl][led_suit][tile] : uint32`
  (tiles that outrank `tile` under that led context — the roles/
  threat tensor generalized from "when led" to "in any position");
- the 2↔3 arrow as data (the game's one symmetry, from gate R6).

All stacked numpy arrays, built once from the authorities, read-only.

### atlas/coordinate.py — the exact situation coordinate

`CoordinateV1` — a frozen dataclass + canonical packed encoding. Fields
(play phase, points contracts; special contracts are out of v1 scope
and must raise, never approximate):

- `version` (=1), `decl_id`, `viewer` seat, `viewer_hand` uint32 mask
- `played[4]` uint32 masks (per-seat tiles played, all tricks)
- `trick_leader`, `current_trick` (0–3 tile ids, seat-implied)
- `team_points[2]` (banked, declaring-team index 0), `bid_value`,
  `bidder`, `dealer`
- `voids[4]` uint8 — observed void bits per seat per led-suit domain
  (0..6 pips + called), exactly the follow-failures all seats have seen

Operations:
- `pack() -> bytes` (fixed-width, little-endian, versioned; the
  canonical address bytes) / `unpack()`; `address()` = blake2b-128 of
  `pack()` — THE address, stable forever;
- `from_engine(state, viewer)` — build from a `forge.zeb.game` state;
- `transition(coord, tile) -> CoordinateV1` — node deletion: apply one
  play *from the viewer's information* (legal only when knowable:
  viewer's own plays always; others' plays given as observations);
  updates masks, trick state, banked points on trick completion, and
  the voids matrix from observed follow-failures;
- `legal(coord) -> uint32 mask` for the seat to act (when that seat is
  the viewer or the acting tile set is queried per-world, the fiber
  module owns the latter);
- the auction-phase variant is DECLARED (`CoordinateV0Auction`,
  fields: viewer hand, bid history, dealer) but may be a stub with
  tests only for pack/unpack in v1.

Sufficiency doctrine: the coordinate is claimed sufficient for the
exact game vs non-signaling fields (census argument). Gate C4 tests
this claim rather than assuming it. Against signaling fields the full
history is the coordinate; v1 records this in docstrings and offers
`history_digest` as an optional extension field in pack (zeroed in v1).

### atlas/fiber.py — the exact unknowns

- `fiber(coord) -> np.ndarray[(N, 3), uint32]` — every deal of the
  hidden 21 (tiles not in viewer hand, not played) among the three
  hidden seats consistent with: per-seat remaining-count conservation,
  played-tile membership, and the voids matrix. This is the physics
  fiber — exact, enumerable, no modeling.
- `weights` are NOT atlas's business beyond the uniform default: belief
  is a measure handed in by consumers (the quarantine doctrine).
- Monotonicity: `fiber(transition(c, t)) ⊆ fiber(c)` projected — gated.

### atlas/roles.py — role state at a coordinate

Thin, derived, exact: walkers / bosses / threat counts / count exposure
for any seat's holding (viewer's hand, or a hypothetical world's hand)
at a coordinate; the decl-unknown 28×D stack for auction-phase hands.
Reuses the algebra's BEATS/THREAT planes; mirrors (and will eventually
absorb) `roles/threat.py` — but atlas does NOT import from `roles/`
(fresh build; `roles/` stays until consumers migrate, then no-legacy
deletes it).

### atlas/narrate.py (small, v1-optional but desired)

Coordinate deltas as sentences: given `c -> transition(c, t)`, emit the
role-promotion events exactly ("6-4 became a walker", "seat 2 shown
void in fives"). Pure functions over the two coordinates; this is the
family-vocabulary surface and the count-fate referees' hook.

## Gates (all pytest, `atlas/tests/`, no skips ever)

- **C1 pack/unpack**: round-trip identity, address stability across
  processes, version byte enforced; fuzz 10k random coordinates.
- **C2 engine parity**: 500+ random full games via `forge.zeb.game`;
  at every play, `from_engine` coordinates for all 4 viewers; atlas
  `transition` from the observed action must equal `from_engine` of the
  engine's next state (field-exact), and `legal` must equal the
  engine's legal set for the viewer-to-act.
- **C3 fiber parity**: for random mid-game coordinates, `fiber` equals
  `walt.worlds.enumerate_worlds` on the equivalent root (same set,
  any order) — including void-constrained cases; plus the monotone
  shrink gate.
- **C4 sufficiency spot-gate**: pairs of DIFFERENT histories reaching
  EQUAL coordinates must produce identical hoyt solve values (build
  subgames from both, `br_solve` vs a fixed rule sigma, values equal to
  1e-9) — a measured check on the coordinate-sufficiency claim, ~20
  engineered pairs.
- **C5 role parity**: atlas role predicates equal `roles/threat.py`
  outputs on random states (until roles/ is absorbed), and the R2-style
  resolve_trick authority gate is re-run against atlas's BEATS planes.
- **C6 the symmetry arrow**: transporting a coordinate through 2↔3
  (tiles + decl 2↔3) commutes with transition — the one symmetry, as
  an API-level law.

## Style and law

CLAUDE.md governs: immutable state transitions, every line a
liability, correct by construction; ES-modules rule is TS-side, here it
is: pure functions, frozen dataclasses, numpy, `python -u`. No
`roles/`, `walt/`, `hoyt/` imports in atlas runtime code (tests may
import them as parity authorities — walt.tables/forge.oracle.tables ARE
the rule authorities for algebra.py's build). No perf work in v1: no
numba, no caching cleverness beyond building the algebra once; if a
gate is slow, shrink its N, never its meaning. Temporary files in
`scratch/`. Wiki update on landing: [[forty-two-native-object]] gains
an "atlas exists" section; new entity page `entities/atlas`;
`log.md` + index entries; lint must pass 0 errors.

## Non-goals (v1)

Solvers, players, nets, belief models beyond the uniform default, the
auction coordinate beyond a stub, perf tuning, Metal, and any retrofit
of existing consumers. Also NOT a goal: deleting `roles/` yet — that
happens when consumers migrate (no-legacy applies at that moment, not
before).
