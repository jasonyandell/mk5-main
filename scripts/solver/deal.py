"""
Texas 42 Solver - Seeded RNG and Dealing

Ported from src/game/core/random.ts and src/game/core/dominoes.ts

Uses Park & Miller's "minimal standard" LCG for deterministic randomness.
This ensures cross-platform reproducibility with the TypeScript implementation.
"""

from typing import List, Tuple
from .tables import DOMINO_PIPS, domino_to_id


class SeededRandom:
    """
    Seeded random number generator using Linear Congruential Generator (LCG).

    Uses Park and Miller's "minimal standard" LCG constants:
    - a = 16807 (multiplier)
    - m = 2^31 - 1 (modulus)

    This matches the TypeScript implementation exactly.
    """

    def __init__(self, seed: int):
        # Ensure seed is a positive integer
        self.seed = abs(int(seed)) if seed else 1

    def next(self) -> float:
        """Generate next random number between 0 and 1."""
        a = 16807
        m = 2147483647  # 2^31 - 1

        self.seed = (a * self.seed) % m
        return self.seed / m

    def next_int(self, min_val: int, max_val: int) -> int:
        """Generate random integer between min_val (inclusive) and max_val (exclusive)."""
        return int(self.next() * (max_val - min_val)) + min_val


def create_dominoes() -> List[Tuple[int, int]]:
    """
    Creates a complete set of 28 dominoes as (high, low) tuples.

    Order matches TypeScript's DOMINO_VALUES constant:
    [0,0], [0,1], [0,2], ... [0,6], [1,1], [1,2], ... [6,6]
    """
    dominoes = []
    for hi in range(7):
        for lo in range(hi + 1):
            dominoes.append((hi, lo))
    return dominoes


# Pre-built domino set matching TypeScript order
DOMINO_SET: List[Tuple[int, int]] = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
    (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
    (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
    (3, 3), (4, 3), (5, 3), (6, 3),
    (4, 4), (5, 4), (6, 4),
    (5, 5), (6, 5),
    (6, 6)
]


def shuffle_with_seed(array: list, seed: int) -> list:
    """
    Shuffles an array using Fisher-Yates algorithm with seeded RNG.

    Matches TypeScript's shuffleWithSeed exactly.
    """
    rng = SeededRandom(seed)
    shuffled = array.copy()

    for i in range(len(shuffled) - 1, 0, -1):
        j = rng.next_int(0, i + 1)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    return shuffled


def deal_dominoes_with_seed(seed: int) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]]
]:
    """
    Deals dominoes to 4 players (7 each) with a seed for deterministic results.

    Returns: Tuple of 4 hands, each a list of 7 (high, low) tuples.
    """
    dominoes = shuffle_with_seed(DOMINO_SET.copy(), seed)

    return (
        dominoes[0:7],
        dominoes[7:14],
        dominoes[14:21],
        dominoes[21:28]
    )


def hands_to_bitmasks(hands: Tuple[List, List, List, List]) -> Tuple[int, int, int, int]:
    """Convert 4 hands to 4 bitmasks (28-bit each)."""
    masks = []
    for hand in hands:
        mask = 0
        for high, low in hand:
            mask |= (1 << domino_to_id(high, low))
        masks.append(mask)
    return tuple(masks)


def format_hand(hand: List[Tuple[int, int]]) -> str:
    """Format a hand for display: [6-4, 5-3, ...]"""
    return "[" + ", ".join(f"{h}-{l}" for h, l in hand) + "]"


def print_deal(seed: int):
    """Print a deal for verification."""
    hands = deal_dominoes_with_seed(seed)
    masks = hands_to_bitmasks(hands)

    print(f"=== Deal for seed {seed} ===")
    for i, (hand, mask) in enumerate(zip(hands, masks)):
        print(f"  Player {i}: {format_hand(hand)}")
        print(f"           mask: {bin(mask)} ({bin(mask).count('1')} dominoes)")

    # Verify all 28 dominoes dealt exactly once
    combined = masks[0] | masks[1] | masks[2] | masks[3]
    assert combined == (1 << 28) - 1, f"Expected all 28 bits set, got {bin(combined)}"
    assert masks[0] & masks[1] & masks[2] & masks[3] == 0, "Duplicate dominoes!"
    print(f"  Verification: All 28 dominoes dealt exactly once ✓")


if __name__ == "__main__":
    # Test with a few seeds
    for seed in [1, 12345, 99999]:
        print_deal(seed)
        print()
