# GPU Solver for Texas 42
"""
Precomputed lookup tables for game logic.

This module replicates the TypeScript domino-tables.ts for use in the GPU solver.
"""

from .tables import (
    DOMINO_PIPS,
    EFFECTIVE_SUIT,
    SUIT_MASK,
    HAS_POWER,
    RANK,
    CALLED_SUIT,
)

__all__ = [
    'DOMINO_PIPS',
    'EFFECTIVE_SUIT',
    'SUIT_MASK',
    'HAS_POWER',
    'RANK',
    'CALLED_SUIT',
]
