"""roles — domino roles as monotone threat masks (see roles/threat.py).

The role algebra from the 2026-07-21 brainstorm: THREAT[decl][d] is the
mask of tiles that beat d when led; roles (walker, high trump, boss) are
its state against the shrinking outstanding set, and the decl-unknown
hand is the 28xD stack read column-wise by the auction.
"""
from roles.threat import (
    ALL_TILES,
    FEATURE_NAMES,
    GAME_DECL_IDS,
    boss_mask,
    build_threat,
    decl_stack_features,
    hand_features,
    hand_to_mask,
    threat_counts,
    threat_tensor,
    walker_mask,
)

__all__ = [
    "ALL_TILES", "FEATURE_NAMES", "GAME_DECL_IDS", "boss_mask",
    "build_threat", "decl_stack_features", "hand_features", "hand_to_mask",
    "threat_counts", "threat_tensor", "walker_mask",
]
