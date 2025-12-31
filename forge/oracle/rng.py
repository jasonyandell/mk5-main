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
