"""otis — the fate-ledger-native player (GitHub issue #49, count-fate ledger).

This package hosts the offline **fate parser** that turns a replayed Texas 42 game
into a per-count-tile fate ledger, plus the corpus adapter and export CLI that feed
the TypeScript referee.

The parser reuses forge's rule tables (``forge.oracle.tables`` /
``forge.oracle.declarations``) — the same suit algebra the GPU engine
(``forge.eq.game_tensor.GameStateTensor``) is derived from — rather than
reimplementing trick logic. It is pure CPU and torch-free.
"""

from otis.fates import (
    COUNT_TILE_IDS,
    COUNT_TILE_PIPS,
    GameFates,
    NeutralGame,
    TileFate,
    domino_id_to_pips,
    parse_game_fates,
    pips_to_domino_id,
)

__all__ = [
    "COUNT_TILE_IDS",
    "COUNT_TILE_PIPS",
    "GameFates",
    "NeutralGame",
    "TileFate",
    "domino_id_to_pips",
    "parse_game_fates",
    "pips_to_domino_id",
]
