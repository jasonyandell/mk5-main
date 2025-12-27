"""
Texas 42 Solver - Global Lookup Tables

Ported from src/game/core/domino-tables.ts

This module implements the "Crystal Palace" - precomputed lookup tables
for game logic. All rule evaluation becomes O(1) table lookups.

Key insight: Trump conflates two independent operations:
1. Absorption (kappa) - restructures which dominoes belong to which suit
2. Power (pi) - determines which dominoes beat others
"""

from typing import List, Tuple
import numpy as np

# ============= CONSTANTS =============

CALLED_SUIT = 7  # The called suit index (suit 7) - where "called" dominoes go

# Pip values for each domino index [low, high]
# Uses triangular number encoding: index = hi*(hi+1)/2 + lo
DOMINO_PIPS: List[Tuple[int, int]] = []
for hi in range(7):
    for lo in range(hi + 1):
        DOMINO_PIPS.append((lo, hi))

assert len(DOMINO_PIPS) == 28, f"Expected 28 dominoes, got {len(DOMINO_PIPS)}"

# ============= CONVERSION FUNCTIONS =============

def domino_to_id(high: int, low: int) -> int:
    """
    Convert Domino pips to table index (0-27).
    Uses triangular number formula: index = hi*(hi+1)/2 + lo
    """
    lo = min(high, low)
    hi = max(high, low)
    return (hi * (hi + 1)) // 2 + lo


def get_absorption_id(decl_id: int) -> int:
    """
    Extract absorption configuration from declaration ID.

    Declaration IDs:
    - 0-6: pip-trump (absorbs dominoes containing that pip)
    - 7: doubles-trump (absorbs all doubles)
    - 8: doubles-suit/nello (absorbs all doubles, no power)
    - 9: no-trump (no absorption)

    Returns:
    - 0-6: Suit-based absorption
    - 7: Doubles absorption (doubles-trump, doubles-suit, nello)
    - 8: No absorption (no-trump)
    """
    if decl_id <= 6:
        return decl_id
    elif decl_id <= 8:
        return 7  # doubles form separate suit
    else:
        return 8  # no absorption


def get_power_id(decl_id: int) -> int:
    """
    Extract power configuration from declaration ID.

    Returns:
    - 0-6: Pip-based power
    - 7: Doubles power
    - 8: No power (no-trump, nello)
    """
    if decl_id <= 6:
        return decl_id
    elif decl_id == 7:
        return 7  # doubles have power
    else:
        return 8  # nothing has power


# ============= PRECOMPUTED TABLES =============

def _generate_effective_suit_table() -> np.ndarray:
    """
    EFFECTIVE_SUIT[d][absorptionId] -> SuitId (0-7)

    Determines what suit a domino belongs to given the absorption config.
    - When absorbed: returns 7 (the absorbed suit)
    - Otherwise: returns high pip (the domino's natural suit for leading)

    28 x 9 = 252 entries
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


def _generate_suit_mask_table() -> np.ndarray:
    """
    SUIT_MASK[absorptionId][suit] -> bitmask of dominoes that can follow that suit

    A domino can follow a suit if:
    - For absorbed suit (7): domino must be absorbed
    - For regular suit: domino must contain that pip AND not be absorbed

    9 x 8 = 72 entries (each entry is a 28-bit mask)
    """
    result = np.zeros((9, 8), dtype=np.uint32)

    for abs_id in range(9):
        for suit in range(8):
            mask = 0
            for d in range(28):
                lo, hi = DOMINO_PIPS[d]
                effective_suit = EFFECTIVE_SUIT[d, abs_id]
                is_absorbed = (effective_suit == CALLED_SUIT)

                can_follow = False
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


def _generate_rank_table() -> np.ndarray:
    """
    RANK[d][powerId] -> number (higher wins)

    Implements the ranking portion of tau from SUIT_ALGEBRA.md S8.

    This table encodes:
    - Power dominoes: (2 << 4) + rank = 32-46 (full Tier 2 tau value)
    - Non-power dominoes: just their rank (0-14) for potential Tier 1

    Rank values:
    - 14: Doubles (highest in suit)
    - 0-12: Non-doubles (pip sum)
    - Exception: Doubles-trump (powerId=7) -> rank = pip value (0-6)

    28 x 9 = 252 entries
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


def _generate_has_power_table() -> np.ndarray:
    """
    HAS_POWER[d][powerId] -> boolean

    Determines if a domino has power (can beat non-power dominoes).

    28 x 9 = 252 entries
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


def _generate_points_table() -> np.ndarray:
    """
    POINTS[d] -> {0, 5, 10}

    Count domino values for scoring.
    - 5-5 = 10 points
    - 6-4 = 10 points
    - 5-0, 4-1, 3-2 = 5 points each
    - All others = 0 points

    28 entries
    """
    result = np.zeros(28, dtype=np.int8)

    for d in range(28):
        lo, hi = DOMINO_PIPS[d]
        total = lo + hi

        if hi == 5 and lo == 5:
            result[d] = 10  # 5-5 = 10 points
        elif (hi == 6 and lo == 4) or (hi == 4 and lo == 6):
            result[d] = 10  # 6-4 = 10 points
        elif total == 5:
            result[d] = 5   # 5-0, 4-1, 3-2 = 5 points each
        else:
            result[d] = 0

    return result


# Generate all tables at module load time
EFFECTIVE_SUIT = _generate_effective_suit_table()
SUIT_MASK = _generate_suit_mask_table()
RANK = _generate_rank_table()
HAS_POWER = _generate_has_power_table()
POINTS = _generate_points_table()

# ============= GAME LOGIC FUNCTIONS =============

def get_led_suit(domino_id: int, absorption_id: int) -> int:
    """Get the suit that a domino leads."""
    return int(EFFECTIVE_SUIT[domino_id, absorption_id])


def get_legal_plays_mask(hand: int, absorption_id: int, lead_domino_id: int | None) -> int:
    """
    Get legal plays from a hand given what was led.

    Args:
        hand: Bitmask of dominoes in hand (bit i = domino i present)
        absorption_id: Current absorption configuration
        lead_domino_id: The domino that was led (None if leading)

    Returns: Bitmask of legal plays
    """
    if lead_domino_id is None:
        return hand  # leading: any domino

    led_suit = EFFECTIVE_SUIT[lead_domino_id, absorption_id]
    can_follow = hand & int(SUIT_MASK[absorption_id, led_suit])
    return can_follow if can_follow != 0 else hand  # must follow if able


def can_follow(domino_id: int, absorption_id: int, led_suit: int) -> bool:
    """Check if a domino can follow the led suit."""
    return (int(SUIT_MASK[absorption_id, led_suit]) & (1 << domino_id)) != 0


def is_trump(domino_id: int, power_id: int) -> bool:
    """Check if a domino has trump power."""
    return bool(HAS_POWER[domino_id, power_id])


def hand_to_bitmask(hand: list) -> int:
    """Convert a hand (list of (high, low) tuples) to a bitmask."""
    mask = 0
    for high, low in hand:
        mask |= (1 << domino_to_id(high, low))
    return mask


# ============= VERIFICATION =============

def print_table_summary():
    """Print summary of generated tables for verification."""
    print("=== Global Tables Generated ===")
    print(f"DOMINO_PIPS: {len(DOMINO_PIPS)} entries")
    print(f"  First 5: {DOMINO_PIPS[:5]}")
    print(f"  Last 5: {DOMINO_PIPS[-5:]}")
    print()
    print(f"EFFECTIVE_SUIT: {EFFECTIVE_SUIT.shape} ({EFFECTIVE_SUIT.dtype})")
    print(f"  [0-0 (id=0)]: {list(EFFECTIVE_SUIT[0])}")
    print(f"  [6-6 (id=27)]: {list(EFFECTIVE_SUIT[27])}")
    print()
    print(f"SUIT_MASK: {SUIT_MASK.shape} ({SUIT_MASK.dtype})")
    print(f"  [abs=0, suit=0]: {bin(SUIT_MASK[0, 0])} ({bin(SUIT_MASK[0, 0]).count('1')} dominoes)")
    print(f"  [abs=7, suit=7]: {bin(SUIT_MASK[7, 7])} ({bin(SUIT_MASK[7, 7]).count('1')} doubles)")
    print()
    print(f"RANK: {RANK.shape} ({RANK.dtype})")
    print(f"  [0-0 (id=0)]: {list(RANK[0])}")
    print(f"  [6-6 (id=27)]: {list(RANK[27])}")
    print()
    print(f"HAS_POWER: {HAS_POWER.shape} ({HAS_POWER.dtype})")
    print(f"  [0-0 (id=0)]: {list(HAS_POWER[0])}")
    print(f"  [6-6 (id=27)]: {list(HAS_POWER[27])}")
    print()
    print(f"POINTS: {POINTS.shape} ({POINTS.dtype})")
    print(f"  Non-zero: {[(i, DOMINO_PIPS[i], POINTS[i]) for i in range(28) if POINTS[i] > 0]}")
    print(f"  Total points in deck: {sum(POINTS)}")


if __name__ == "__main__":
    print_table_summary()
