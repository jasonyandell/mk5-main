"""atlas — the game as an addressed space (see atlas/SPEC.md).

The from-scratch native substrate for Texas 42: coordinates, the
decl-indexed algebra, role state, transitions, and fibers — correct by
construction, gated against the existing authorities (forge.zeb, walt, hoyt,
roles), zero perf ambition in v1.  A library, not a solver: no nets, no
players, no belief model beyond the uniform default.  The only estimated
object anywhere is the belief measure a consumer hands onto the exact fiber;
everything atlas computes is exact.

Public surface:

    algebra     resident decl-indexed rule planes + the 2<->3 arrow
    coordinate  CoordinateV1: pack/address, from_engine, transition, legal,
                transport (the one symmetry); CoordinateV0Auction stub
    fiber       exact enumeration of the hidden deals of a coordinate
    roles       walkers / bosses / threat counts / count exposure, decl stack
    narrate     coordinate deltas as family-vocabulary events
"""
from atlas.algebra import (
    Algebra,
    N_DECLS,
    N_DOMINOES,
    N_LED_SUITS,
    beat_count_when_led,
    follow_mask,
    get_algebra,
    led_suit,
    legal_from_hand,
    mask_of,
    tiles_of,
    trick_points,
    trick_winner_offset,
)
from atlas.coordinate import (
    CoordinateV0Auction,
    CoordinateV1,
    VERSION,
    from_engine,
    legal,
    transition,
    transport,
)
from atlas.fiber import fiber, hidden_seats, uniform_weights, unknown_tiles
from atlas.narrate import Event, narrate, suit_name, tile_name
from atlas.roles import (
    FEATURE_NAMES,
    boss_mask,
    decl_stack_features,
    hand_features,
    outstanding,
    role_summary,
    threat_counts,
    walker_mask,
)

__all__ = [
    # algebra
    "Algebra", "N_DECLS", "N_DOMINOES", "N_LED_SUITS", "get_algebra",
    "beat_count_when_led", "follow_mask", "led_suit", "legal_from_hand",
    "mask_of", "tiles_of", "trick_points", "trick_winner_offset",
    # coordinate
    "CoordinateV1", "CoordinateV0Auction", "VERSION", "from_engine", "legal",
    "transition", "transport",
    # fiber
    "fiber", "hidden_seats", "unknown_tiles", "uniform_weights",
    # roles
    "FEATURE_NAMES", "boss_mask", "decl_stack_features", "hand_features",
    "outstanding", "role_summary", "threat_counts", "walker_mask",
    # narrate
    "Event", "narrate", "suit_name", "tile_name",
]
