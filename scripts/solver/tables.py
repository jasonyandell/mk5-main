"""
Domino Tables - Precomputed lookup tables for game logic

This module implements the "Crystal Palace" - the single source of truth
for base game rules via table lookups. See bead t42-9xy3 for the theory.

Key insight: Trump conflates two independent operations:
1. Absorption - restructures which dominoes belong to which suit
2. Power - determines which dominoes beat others

These are factored into separate table dimensions for clean composition.
"""

import numpy as np
from numpy.typing import NDArray

# ============= CONSTANTS =============

# The called suit index (suit 7) - where "called" dominoes go
CALLED_SUIT: int = 7

# ============= DOMINO_PIPS =============

def _build_domino_pips() -> NDArray[np.int8]:
    """
    Build DOMINO_PIPS: (28, 2) array of [lo, hi] pairs.

    Triangular encoding: domino_id = hi*(hi+1)//2 + lo where lo <= hi
    """
    result = np.zeros((28, 2), dtype=np.int8)
    idx = 0
    for hi in range(7):
        for lo in range(hi + 1):
            result[idx, 0] = lo
            result[idx, 1] = hi
            idx += 1
    return result

DOMINO_PIPS: NDArray[np.int8] = _build_domino_pips()

# ============= EFFECTIVE_SUIT =============

def _build_effective_suit() -> NDArray[np.int8]:
    """
    Build EFFECTIVE_SUIT: (28, 9) array.

    EFFECTIVE_SUIT[d][absorption_id] -> suit (0-7)

    Determines what suit a domino belongs to given the absorption config:
    - When absorbed: returns 7 (the absorbed suit)
    - Otherwise: returns high pip (the domino's natural suit for leading)

    Absorption IDs:
    - 0-6: pip trump (dominoes containing that pip -> suit 7)
    - 7: doubles trump (doubles -> suit 7)
    - 8: no absorption
    """
    result = np.zeros((28, 9), dtype=np.int8)

    for d in range(28):
        lo, hi = DOMINO_PIPS[d]

        # Pip absorptions (0-6)
        for pip in range(7):
            if lo == pip or hi == pip:
                result[d, pip] = CALLED_SUIT
            else:
                result[d, pip] = hi

        # Doubles absorption (7)
        if lo == hi:
            result[d, 7] = CALLED_SUIT
        else:
            result[d, 7] = hi

        # No absorption (8)
        result[d, 8] = hi

    return result

EFFECTIVE_SUIT: NDArray[np.int8] = _build_effective_suit()

# ============= SUIT_MASK =============

def _build_suit_mask() -> NDArray[np.uint32]:
    """
    Build SUIT_MASK: (9, 8) array of 28-bit masks.

    SUIT_MASK[absorption_id][suit] -> bitmask of dominoes that can follow that suit

    A domino can follow a suit if:
    - For absorbed suit (7): domino must be absorbed
    - For regular suit: domino must contain that pip AND not be absorbed
    """
    result = np.zeros((9, 8), dtype=np.uint32)

    for abs_id in range(9):
        for suit in range(8):
            mask = 0
            for d in range(28):
                lo, hi = DOMINO_PIPS[d]
                effective_suit = EFFECTIVE_SUIT[d, abs_id]
                is_absorbed = (effective_suit == CALLED_SUIT)

                if suit == CALLED_SUIT:
                    # Absorbed suit led: must be absorbed to follow
                    can_follow = is_absorbed
                elif is_absorbed:
                    # Domino is absorbed: cannot follow non-absorbed suits
                    can_follow = False
                else:
                    # Non-absorbed domino, regular suit led
                    # Can follow if either pip matches
                    can_follow = (lo == suit or hi == suit)

                if can_follow:
                    mask |= (1 << d)

            result[abs_id, suit] = mask

    return result

SUIT_MASK: NDArray[np.uint32] = _build_suit_mask()

# ============= HAS_POWER =============

def _build_has_power() -> NDArray[np.bool_]:
    """
    Build HAS_POWER: (28, 9) boolean array.

    HAS_POWER[d][power_id] -> True if domino has trump power

    Determines if a domino has power (can beat non-power dominoes).

    Power IDs:
    - 0-6: dominoes containing that pip have power
    - 7: doubles have power (doubles-trump)
    - 8: nothing has power (doubles-suit, no-trump)
    """
    result = np.zeros((28, 9), dtype=np.bool_)

    for d in range(28):
        lo, hi = DOMINO_PIPS[d]
        is_double = (lo == hi)

        for power in range(9):
            if power == 8:
                result[d, power] = False
            elif power == 7:
                result[d, power] = is_double
            else:
                result[d, power] = (lo == power or hi == power)

    return result

HAS_POWER: NDArray[np.bool_] = _build_has_power()

# ============= RANK =============

def _build_rank() -> NDArray[np.int8]:
    """
    Build RANK: (28, 9) array.

    RANK[d][power_id] -> encodes (tier << 4) + rank for power dominoes

    This table encodes:
    - Power dominoes: (2 << 4) + rank = 32-46 (full Tier 2 value)
    - Non-power dominoes: just their rank (0-14) for potential Tier 1

    Rank values:
    - 14: Doubles (highest in suit)
    - 0-12: Non-doubles (pip sum)
    - Exception: Doubles-trump (power_id=7) -> rank = pip value (0-6)
    """
    result = np.zeros((28, 9), dtype=np.int8)

    for d in range(28):
        lo, hi = DOMINO_PIPS[d]
        is_double = (lo == hi)
        pip_sum = lo + hi

        for power in range(9):
            if power <= 6:
                # Pip power: dominoes containing that pip have power
                has_power = (lo == power or hi == power)
                if has_power:
                    # Tier 2: (2 << 4) + rank
                    rank = 14 if is_double else pip_sum
                    result[d, power] = (2 << 4) + rank
                else:
                    # No power: just raw rank for potential Tier 1
                    result[d, power] = 14 if is_double else pip_sum
            elif power == 7:
                # Doubles power: doubles have power, rank by pip value
                if is_double:
                    # Tier 2, rank = pip value (not 14!)
                    result[d, power] = (2 << 4) + lo
                else:
                    # No power: raw rank for potential Tier 1
                    result[d, power] = pip_sum
            else:
                # No power (8): all dominoes just have raw rank
                result[d, power] = 14 if is_double else pip_sum

    return result

RANK: NDArray[np.int8] = _build_rank()
