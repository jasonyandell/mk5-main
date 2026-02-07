from __future__ import annotations

import random

from .tables import N_DOMINOES


def deal_from_seed(seed: int) -> list[list[int]]:
    """Return 4 hands of 7 unique domino IDs, deterministically from `seed`."""
    rng = random.Random(seed)
    dominos = list(range(N_DOMINOES))
    rng.shuffle(dominos)
    hands = [sorted(dominos[i * 7 : (i + 1) * 7]) for i in range(4)]
    if sorted(hands[0] + hands[1] + hands[2] + hands[3]) != list(range(N_DOMINOES)):
        raise AssertionError("invalid deal")
    return hands


def deal_with_fixed_p0(p0_hand: list[int], seed: int) -> list[list[int]]:
    """Return 4 hands with P0's hand fixed, remaining dominoes dealt from seed.

    Args:
        p0_hand: List of 7 domino IDs for player 0
        seed: RNG seed for distributing remaining 21 dominoes to P1, P2, P3

    Returns:
        List of 4 hands (each sorted), with p0_hand as the first hand
    """
    if len(p0_hand) != 7:
        raise ValueError(f"P0 hand must have 7 dominoes, got {len(p0_hand)}")
    if len(set(p0_hand)) != 7:
        raise ValueError("P0 hand has duplicate dominoes")
    if not all(0 <= d < N_DOMINOES for d in p0_hand):
        raise ValueError(f"Invalid domino ID in P0 hand (must be 0-{N_DOMINOES-1})")

    p0_set = set(p0_hand)
    remaining = [d for d in range(N_DOMINOES) if d not in p0_set]

    rng = random.Random(seed)
    rng.shuffle(remaining)

    hands = [
        sorted(p0_hand),
        sorted(remaining[0:7]),
        sorted(remaining[7:14]),
        sorted(remaining[14:21]),
    ]

    if sorted(hands[0] + hands[1] + hands[2] + hands[3]) != list(range(N_DOMINOES)):
        raise AssertionError("invalid deal")
    return hands
