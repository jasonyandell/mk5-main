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


PIP_DECLS = tuple(range(7))


def plays(policy, decl: int) -> bool:
    """Whether a play policy can play declaration `decl`: its own `plays`
    if it states one, else every inner play policy it wraps (`inner`,
    `zeb`, `walt`) can; a policy that states nothing plays every
    declaration."""
    own = getattr(policy, "plays", None)
    if callable(own):
        return own(decl)
    inner = (getattr(policy, name, None) for name in ("inner", "zeb", "walt"))
    return all(plays(p, decl) for p in inner if p is not None and hasattr(p, "choose"))


def check_menu(bid_policies, play_policies) -> None:
    """Refuse a table where a bidder may declare what a seat cannot play
    (#155): a bid policy names the declarations it may make in
    `declarations` (the pip trumps when it names none)."""
    menu = sorted({d for b in bid_policies for d in getattr(b, "declarations", PIP_DECLS)})
    for policy in play_policies:
        missing = [d for d in menu if not plays(policy, d)]
        if missing:
            raise ValueError(
                f"#155: a bidder here may declare {missing}, which {policy!r} cannot play "
                "(the field net plays pip trumps only); seat Zeb with walt's DTNT=random "
                "(arena.walt_play.MenuWalt) at a menu table"
            )


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
