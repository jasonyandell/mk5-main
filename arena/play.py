"""Play policies for the arena.

A PlayPolicy chooses one action per state for a batch of games — batching
is the contract, so GPU players amortize across all games they are playing
this tick. The oracle-backed player lives in `lens_play.py` to keep this
module (and the engine) importable without the GPU stack.
"""
from __future__ import annotations

import random
from typing import Protocol, Sequence

from forge.zeb.game import legal_actions
from forge.zeb.types import ZebGameState


class PlayPolicy(Protocol):
    """A card player: slot indices (0-6) for a batch of PLAYING states."""

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
    ) -> list[int]:
        ...


class RandomPlay:
    """Uniform-legal baseline."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
    ) -> list[int]:
        return [self._rng.choice(legal_actions(s)) for s in states]

    def __repr__(self) -> str:
        return "RandomPlay()"
