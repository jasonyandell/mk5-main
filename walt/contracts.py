"""walt module contracts — the frozen interfaces the v0 modules build against.

See walt/DESIGN.md for the algorithm and rationale. Ground rules that apply
to every module:

- Interpreter: the repo's .venv python, run from the repo root. walt is a
  regular package (walt/__init__.py); no sys.path hacks anywhere — `forge.*`,
  `champion.*`, `arena.*` resolve from the same checkout walt lives in.
- Tiles are forge domino ids 0..27 (`forge.oracle.tables.DOMINOES`). Inside
  walt everything speaks DOMINO IDS; the zeb engine's slot indices (0..6
  into a fixed hand tuple) appear only at the arena boundary in grade.py.
- Rules are never reimplemented. All rule LUTs are TABULATED by calling
  `forge.oracle.tables` functions (`led_suit_for_lead_domino`, `can_follow`,
  `trick_rank`, `resolve_trick`, `DOMINO_COUNT_POINTS`) and parity-tested
  against the zeb engine on random states. Trick winner = first max of
  trick_rank in play order from the leader (ties broken by play order).
- Hands and tile sets are uint32 bitmasks (bit d = domino id d) in numpy;
  worlds are struct-of-arrays, never lists of objects, in hot paths.
- The solver's value unit is E[final declaring-team hand points, 0..42]
  (banked + remaining). The walt seat maximizes sign*value with
  sign = +1 if me%2 == bidder%2 else -1 — same orientation and sign rule
  as arena.jud_play.JudPlay. Never double-flip.
- Featurization parity: champion.jud_net.featurize_state(state, seat=mover)
  on CHILD states (post-move) with seat passed explicitly is the reference.
  The POV default flips to the next player if seat is omitted — always pass
  seat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# Solve horizon: consult walt when the mover has <= HORIZON tiles left.
HORIZON = 4


@dataclass(frozen=True)
class EndgameRoot:
    """Everything walt knows at a decision point — its information set.

    play_history covers the whole hand from trick 1 (completed tricks plus
    the current partial trick), as ((seat, domino_id), ...) in play order —
    same shape as ZebGameState.play_history. banked team_points are the
    engine's (team0, team1) at the root (credited on completed tricks only).
    bids/dealer are needed verbatim by the jud featurizer's auction block.
    """

    decl_id: int
    bidder: int
    bid_value: int
    bids: tuple  # seat-indexed, as in ZebGameState.bid_state.bids
    dealer: int
    me: int  # the walt seat
    my_hand: tuple  # sorted domino ids still held by me
    play_history: tuple  # ((seat, domino_id), ...) from trick 1
    trick_leader: int  # leader of the current (possibly empty) trick
    current_trick: tuple  # domino ids already on the current trick, in order
    team_points: tuple  # banked (team0, team1)


@dataclass
class SolveResult:
    value: float  # E[declaring-team final points] under the computed strategy
    best_move: int  # domino id to play now
    n_worlds: int
    n_nodes: int
    n_field_queries: int
    # walker instrumentation: moves (led by me, this decision) that win in
    # every alive world while ranking bottom-half by global beat-count.
    walker_flags: dict = field(default_factory=dict)
    # exact value of EVERY legal root move (declaring orientation, same unit
    # as .value), keyed by domino id. The wavefront engine computes these for
    # free; consumers: top-2 gaps, tie-band width (#77 mixing), dense
    # distillation targets, count-fate receipts.
    root_values: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# walt/tables.py  (builder B1)
# ---------------------------------------------------------------------------
# LUTs, all tabulated from forge.oracle.tables, cached per decl_id:
#
#   LUTS = get_luts(decl_id) with numpy arrays:
#     .led_suit[28]        int8   led suit if this tile is led
#     .can_follow[8, 28]   bool   can_follow(tile, led_suit) for led_suit 0..7
#     .rank[8, 28]         int8   trick_rank(tile, led_suit)
#     .count[28]           int8   DOMINO_COUNT_POINTS
#     .beat_count[28]      int8   global "how many of the other 27 tiles this
#                                 tile beats when it is LED" (walker ranking)
#
#   resolve_tricks(leader, tiles4, decl_id) -> (winner_seat, points)
#     vectorized over a batch axis: leader (B,), tiles4 (B,4) in play order.
#     Parity-tested against forge.oracle.tables.resolve_trick.
#
#   legal_moves_mask(hand_mask, led_tile_or_None, decl_id) -> uint32 mask
#     follow-suit legality per the engine: if cannot follow, whole hand is
#     legal. Vectorized over a batch of hand masks. Parity-tested against
#     zeb legal_actions on random mid-hand states.
#
# walt/worlds.py  (builder B1)
# ---------------------------------------------------------------------------
#   enumerate_worlds(root: EndgameRoot) -> np.ndarray
#     shape (N, 3) uint32 hand masks for the three non-me seats, in
#     ascending absolute seat order. Physics-exact u: tile conservation,
#     per-seat remaining counts implied by the history, and void
#     constraints — for every observed play where a seat did not follow
#     the led suit, the seat's reconstructed hand AT THAT MOMENT (current
#     candidate holding + their own later plays... i.e. original hand minus
#     plays made before that moment) contains no tile that could follow.
#     Brute-force cross-checked on small cases (deal random full hands,
#     play a prefix with a random policy, verify enumeration == filter of
#     all C(unknown) assignments).
#
#   seat_order(root) -> tuple  # the three absolute seats matching columns
#
# ---------------------------------------------------------------------------
# walt/field.py  (builder B2)
# ---------------------------------------------------------------------------
#   class FieldOracle:
#       """jud argmax as a deterministic field, batched + memoized.
#
#       __init__(net_path=<walt worktree>/champion/jud_net.pt, device='cpu')
#       decisions(queries) -> list[int]
#           queries: list of (seat, hand_mask, pub) where pub is a PubState.
#           Returns the domino id jud plays for each query. Memoized on
#           (seat, hand_mask, pub.key). Internally: vectorized featurizer
#           (numpy) building the 350-dim child features for every legal
#           move of every query, one torch forward, argmax of sign*E[pts]
#           with ties -> lowest slot index in the engine's hand ordering
#           (= lowest domino id among that seat's remaining sorted hand —
#           verify against JudPlay's torch.argmax-first-max semantics).
#       """
#
#   @dataclass(frozen=True)
#   class PubState:
#       """Public state a field decision conditions on: decl_id, bidder,
#       bids, dealer, play_history-so-far, trick_leader, current_trick,
#       banked team_points, plus a precomputed hashable .key."""
#
#   sigma_consistent(root, worlds, oracle, moves_filter) -> np.ndarray[bool]
#       For each candidate world, replay root.play_history; at every
#       observed move (k, seat, tile) with seat != root.me and
#       moves_filter(seat, k) True, ask the oracle for that seat's move
#       given the world-hypothetical hand; keep worlds reproducing every
#       filtered observed move. Batched across worlds per history step.
#       This is exact B(sigma) for a deterministic field (the e^g filter).
#
#   Parity gates (test_field.py): (1) featurizer bit-equality vs
#   champion.jud_net.featurize_state on >=2000 random (state, seat) pairs
#   across decls/phases; (2) decision equality vs arena.jud_play.JudPlay
#   on >=500 random mid-hand states (must be 100%; investigate any miss).
#
# ---------------------------------------------------------------------------
# walt/solver.py  (builder B3)
# ---------------------------------------------------------------------------
#   solve(root: EndgameRoot,
#         worlds: np.ndarray, weights: np.ndarray,
#         oracle: FieldOracle,
#         payoff: str = 'points',   # or 'make'
#         ) -> SolveResult
#
#   Exact best response of root.me in the information-set game vs the
#   deterministic field, per DESIGN.md: recursion with observation-branch
#   world partitioning, memo on (my_hand_mask, history_key), leaves scored
#   with tables.resolve_tricks + banked points; payoff 'make' maps final
#   declaring points to 1[pts >= bid_value] (defenders: 1 - that) before
#   weighting. Correctness gates T1-T4 and T6 from DESIGN.md live in
#   test_solver.py and must pass before grade.py trusts it.
#
# ---------------------------------------------------------------------------
# walt/grade.py + walt/bench.py  (builder B4)
# ---------------------------------------------------------------------------
#   WaltPlay: an arena play policy (choose(states, bid_values, marks=None,
#   marks_to_win=7) -> list[slot]) that delegates to arena.jud_play.JudPlay
#   while the mover holds > HORIZON tiles and to the solver at <= HORIZON,
#   with per-decision fresh solves, sigma-filtering opponents always and
#   partner only for pre-horizon moves. Slot conversion at the boundary:
#   slots index the FIXED original 7-tuple state.hands[mover] (they do not
#   shift as tiles are played), so slot = state.hands[mover].index(id) —
#   and the slot must be unplayed, which it is iff walt only proposes ids
#   from the mover's remaining hand. Solves for a batch
#   of live games run on a multiprocessing pool (workers each load their
#   own FieldOracle); pool size tuned in bench.py.
#   Driver: bespoke runner reusing arena.engine/match summarize (the
#   jud_vs_burl/run_h2h.py pattern) OR a parse_play hook — builder's call,
#   smallest diff wins. Paired seeds, bootstrap CI via arena.match code.
#   Logging: out_dir/events.jsonl + tail.log, a line at least every 60s.
#   bench.py: 200 random endgame roots at HORIZON 3 and 4 -> p50/p95 solve
#   wall, world counts, memo stats; total budget <= 10 min.
"""Module docstring ends here; the commented specs above are the contract."""
