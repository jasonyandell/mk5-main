# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""A seat that loses the count (#172): any play policy, shown a game in which
some tiles it saw played are remembered as still out.

The book's fallible opponent loses track of which tiles have gone, and its
ploys (the straight-off 84 that hopes for a misdiscard, the low-looking master
late in a hand) lean on that ([[book-assumptions]] A3). In the planner a lost
count is no fiber: its worlds share the one true record. It is a policy's
view, so this module corrupts the view, not the game.

The forgetter's view is the true state with a played tile t and a live tile u
exchanged everywhere: in the hands, the history and the points. A relabeled
deal is a deal, so the view is consistent by construction (every hand its
size, 42 points, every void it shows real), and in it t is still out and u has
gone. The game, the record and every other seat stay true; the inner policy
decides at the view, and since neither tile is in the forgetter's hand its
slots are the same.

*Which tiles.* A tile another seat played in a completed trick is forgotten
when its coin, the top 53 bits of `_splitmix(_mix(deal, seat, COUNT, t))`,
falls below ⌈p·2^53⌉. The coin is keyed on the deal, not the decision, so a
tile once forgotten stays forgotten for the rest of the hand: the count is
lost, not lapsed. Its stand-in u is the first tile, in an order keyed the same
way, that is live, held by another seat, not already exchanged, and leaves a
legal history with the same trick leaders (the view must be a game that could
have been played). Where none does, t is remembered.

    CountLapse(player(), p=0.1)
"""
from __future__ import annotations

import dataclasses
import math
from typing import Sequence

from forge.zeb.game import apply_action, legal_actions
from forge.zeb.types import ZebGameState
from scenario.planner import _mix, _splitmix

COUNT = 0x636F756E            # "coun": a seat's count coin is _mix(deal, seat, COUNT, tile)


def deal_key(state: ZebGameState) -> int:
    """The deal's identity: every seat's dealt hand, in order."""
    return _mix(*(d for hand in state.hands for d in hand))


def _swap(state: ZebGameState, t: int, u: int) -> ZebGameState:
    sigma = {t: u, u: t}
    m = lambda d: sigma.get(d, d)  # noqa: E731
    return dataclasses.replace(
        state,
        hands=tuple(tuple(m(d) for d in hand) for hand in state.hands),
        played=frozenset(m(d) for d in state.played),
        play_history=tuple((s, m(d)) for s, d in state.play_history),
        current_trick=tuple(m(d) for d in state.current_trick),
    )


def replayed(state: ZebGameState) -> ZebGameState | None:
    """`state` rebuilt by playing its history from the deal, or None when a
    play is illegal or a trick's winner is not the next trick's leader. The
    points are the history's own."""
    if not state.play_history:
        return state
    s = dataclasses.replace(state, played=frozenset(), play_history=(), current_trick=(),
                            trick_leader=state.play_history[0][0], team_points=(0, 0))
    for seat, d in state.play_history:
        if (s.trick_leader + len(s.current_trick)) % 4 != seat:
            return None
        slot = s.hands[seat].index(d)
        if slot not in legal_actions(s):
            return None
        s = apply_action(s, slot)
    if s.trick_leader != state.trick_leader or s.current_trick != state.current_trick:
        return None
    return s


def lost_count(state: ZebGameState, coin: int) -> tuple[ZebGameState, tuple[int, ...], tuple[int, ...]]:
    """The view of the seat to play, the tiles it has forgotten, and those of
    them that found a stand-in. With coin 0 it is `state` itself."""
    if coin == 0 or not state.play_history:
        return state, (), ()
    viewer = (state.trick_leader + len(state.current_trick)) % 4
    done = len(state.play_history) - len(state.current_trick)
    deal = deal_key(state)
    forgotten = tuple(d for s, d in state.play_history[:done]
                      if s != viewer and _splitmix(_mix(deal, viewer, COUNT, d)) >> 11 < coin)
    view, touched, stood = state, set(), ()
    for t in forgotten:
        key = _mix(deal, viewer, COUNT, t, 1)
        live = [d for s in range(4) if s != viewer for d in view.hands[s]
                if d not in view.played and d not in touched]
        for u in sorted(live, key=lambda d: _mix(key, d)):
            swapped = replayed(_swap(view, t, u))
            if swapped is not None:
                view, stood = swapped, stood + (t,)
                touched |= {t, u}
                break
    return view, forgotten, stood


class CountLapse:
    """`inner`, deciding at the view of a seat that has lost the count with
    probability `p` per tile it saw played (see the module docstring)."""

    def __init__(self, inner, p: float):
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must lie in [0, 1]")
        self.inner, self.p = inner, p
        self.coin = math.ceil(p * 2.0 ** 53)
        self.decisions = self.forgotten = self.stood = 0

    def choose(self, states: Sequence[ZebGameState], bid_values: Sequence[int],
               marks: Sequence[tuple[int, int]] | None = None, marks_to_win: int = 7) -> list[int]:
        views = []
        for s in states:
            view, forgotten, stood = lost_count(s, self.coin)
            views.append(view)
            self.decisions += 1
            self.forgotten += len(forgotten)
            self.stood += len(stood)
        return self.inner.choose(views, bid_values, marks, marks_to_win)

    @property
    def lapse(self) -> dict:
        """Decisions taken, tiles forgotten, and forgotten tiles that found a stand-in."""
        return {"decisions": self.decisions, "forgotten": self.forgotten, "stood": self.stood}

    def __repr__(self) -> str:
        return f"CountLapse({self.inner!r}, p={self.p})"
