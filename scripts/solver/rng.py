"""
Park-Miller LCG random number generator matching TypeScript implementation.

This module replicates the RNG and dealing logic from:
- src/game/core/random.ts
- src/game/core/dominoes.ts
- src/game/constants.ts

Used by the GPU solver for deterministic game state generation.
"""

from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray


# Park-Miller LCG constants
LCG_A = 16807
LCG_M = 2147483647  # 2^31 - 1


# DOMINO_VALUES in the same order as TypeScript constants.ts
# Each tuple is (lo, hi) where lo <= hi
DOMINO_VALUES: List[Tuple[int, int]] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),  # lo=0
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),          # lo=1
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),                  # lo=2
    (3, 3), (3, 4), (3, 5), (3, 6),                          # lo=3
    (4, 4), (4, 5), (4, 6),                                  # lo=4
    (5, 5), (5, 6),                                          # lo=5
    (6, 6),                                                   # lo=6
]


def lcg_next(seed: int) -> int:
    """
    Returns next seed value using Park-Miller LCG.

    a=16807, m=2147483647 (2^31 - 1)
    """
    return (LCG_A * seed) % LCG_M


def to_float(seed: int) -> float:
    """
    Convert seed to float in range [0, 1).

    Matches TypeScript: return this.seed / m
    """
    return seed / LCG_M


def next_int(seed: int, min_val: int, max_val: int) -> Tuple[int, int]:
    """
    Generate random integer between min (inclusive) and max (exclusive).

    Returns (random_int, new_seed).

    Matches TypeScript:
        nextInt(min, max) {
            return Math.floor(this.next() * (max - min)) + min;
        }
    """
    new_seed = lcg_next(seed)
    rand_float = to_float(new_seed)
    result = int(rand_float * (max_val - min_val)) + min_val
    return result, new_seed


def shuffle_with_seed(items: List[int], seed: int) -> Tuple[List[int], int]:
    """
    Fisher-Yates shuffle matching TypeScript implementation.

    TypeScript iterates from len-1 down to 1, using rng.nextInt(0, i+1).

    Returns (shuffled_items, final_seed).
    """
    shuffled = items.copy()
    current_seed = seed

    for i in range(len(shuffled) - 1, 0, -1):
        j, current_seed = next_int(current_seed, 0, i + 1)
        # Swap
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    return shuffled, current_seed


def domino_to_global_id(lo: int, hi: int) -> int:
    """
    Convert (lo, hi) domino pair to global triangular ID.

    Global ID formula: hi*(hi+1)//2 + lo

    This maps dominoes to a triangular numbering:
    (0,0)=0, (0,1)=1, (1,1)=2, (0,2)=3, (1,2)=4, (2,2)=5, ...
    """
    return hi * (hi + 1) // 2 + lo


def deal_with_seed(seed: int) -> NDArray[np.int32]:
    """
    Deal dominoes to 4 players matching TypeScript implementation.

    Process:
    1. Create 28 dominoes in DOMINO_VALUES order (NOT triangular order)
    2. Convert each (lo, hi) pair to global ID
    3. Shuffle using Fisher-Yates with the seed
    4. Return as 4x7 array where result[player][local_idx] = global domino ID

    Returns numpy array of shape (4, 7) with dtype int32.
    """
    # Create dominoes in DOMINO_VALUES order, convert to global IDs
    domino_ids = [domino_to_global_id(lo, hi) for lo, hi in DOMINO_VALUES]

    # Shuffle
    shuffled, _ = shuffle_with_seed(domino_ids, seed)

    # Deal to 4 players (7 each)
    result = np.array([
        shuffled[0:7],
        shuffled[7:14],
        shuffled[14:21],
        shuffled[21:28]
    ], dtype=np.int32)

    return result


def get_domino_values_order() -> List[int]:
    """
    Get the list of global domino IDs in DOMINO_VALUES order.

    This is the order BEFORE shuffling - useful for testing.
    """
    return [domino_to_global_id(lo, hi) for lo, hi in DOMINO_VALUES]


def global_id_to_domino(global_id: int) -> Tuple[int, int]:
    """
    Convert global triangular ID back to (lo, hi) pair.

    Inverse of domino_to_global_id.
    """
    # Find hi such that hi*(hi+1)//2 <= global_id < (hi+1)*(hi+2)//2
    hi = 0
    while (hi + 1) * (hi + 2) // 2 <= global_id:
        hi += 1
    lo = global_id - hi * (hi + 1) // 2
    return lo, hi


class SeededRandom:
    """
    Stateful RNG matching TypeScript SeededRandom class.

    Useful for step-by-step debugging and verification.
    """

    def __init__(self, seed: int):
        # Ensure seed is a positive integer (matches TypeScript createSeededRandom)
        self.seed = abs(int(seed)) or 1

    def next(self) -> float:
        """Generate next random number between 0 and 1."""
        self.seed = lcg_next(self.seed)
        return to_float(self.seed)

    def next_int(self, min_val: int, max_val: int) -> int:
        """Generate random integer between min (inclusive) and max (exclusive)."""
        return int(self.next() * (max_val - min_val)) + min_val

    def get_seed(self) -> int:
        """Get current seed value."""
        return self.seed
