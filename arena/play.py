# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

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
    """A card player: slot indices (0-6) for a batch of PLAYING states.

    ``marks`` carries each game's absolute (team0, team1) mark score so a
    score-conditioned player can shape its play risk (rung #27); ``None`` and
    score-blind players ignore it. The engine always supplies it.
    """

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
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
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        return [self._rng.choice(legal_actions(s)) for s in states]

    def __repr__(self) -> str:
        return "RandomPlay()"


class LowestPlay:
    """The lowest legal tile by id (Silas's order: `h-l` is h(h+1)/2 + l):
    the chance arm's play, and a Silas policy's fallback."""

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        return [min(legal_actions(s), key=lambda slot, s=s: s.hands[_mover(s)][slot]) for s in states]

    def __repr__(self) -> str:
        return "LowestPlay()"


def _mover(state: ZebGameState) -> int:
    return (state.trick_leader + len(state.current_trick)) % 4
