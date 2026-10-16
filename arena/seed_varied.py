# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""A Zeb seat whose planner seed varies with the deal (#190).

A Zeb player is one planner seed: its worlds at each decision are drawn from
a stream keyed by the seed and the state, so two seeds are two fixed,
slightly different players. A study that seats the field at one seed and
bootstraps over deals does not see that difference. #117's A/A found it:
plain Zeb at seed 1 beat the seed-0 field by +0.12 [+0.02, +0.23]
marks/game, and #149 saw the same spread. A single arm's margin against
such a field carries the field seed's luck.

`SeedVaried` makes the seed part of the sampling. It holds `pool` copies of
one player, at seeds `offset`, ..., `offset + pool - 1`, and plays each hand
at the copy its deal picks: a keyed hash of the four dealt hands as a set,
so the pick is the same in every arm, in both halves of a game pair (which
deal the same hands to the other team), and at every decision of the hand.
Two `SeedVaried` seats with the same offset play every deal at the same
seed, so an arm that differs from the field only by a read still differs
only by the read. With different offsets they are two independent draws of
a planner seed per deal, which is an honest A/A. At `pool` = 1 it is the
player at seed `offset`, byte for byte.
"""
from __future__ import annotations

from typing import Sequence

from forge.zeb.types import ZebGameState

from arena.walt_play import WaltPlay

POOL = 16
_FNV_OFFSET, _FNV_PRIME, _MASK = 0xCBF29CE484222325, 0x100000001B3, (1 << 64) - 1


def deal_key(state: ZebGameState) -> int:
    """A keyed 64-bit hash of the deal: the four dealt hands as a set, so a
    rotation of the seats is the same deal."""
    h = _FNV_OFFSET
    for hand in sorted(sum(1 << int(t) for t in hand) for hand in state.hands):
        for byte in hand.to_bytes(4, "little"):
            h = ((h ^ byte) * _FNV_PRIME) & _MASK
    # FNV's low bits barely mix, and the pool reads them: finish with
    # splitmix64's mixer.
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & _MASK
    return h ^ (h >> 31)


class _Pooled:
    """The planner a `SeedVaried` holds as `walt`, for wrappers that call
    the planner directly (`arena.regulars`, `arena.book_play`): each state
    goes to the planner at its deal's seed."""

    def __init__(self, seat: "SeedVaried"):
        self._seat = seat
        self.config = seat.players[0].walt.config

    @property
    def belief(self):
        return self._seat.players[0].walt.belief

    @belief.setter
    def belief(self, value) -> None:
        for p in self._seat.players:
            p.walt.belief = value

    def decide_batch(self, states, bids, auctions=None):
        out: list = [None] * len(states)
        for k, idx in self._seat.by_seed(states).items():
            sub = None if auctions is None else [auctions[i] for i in idx]
            planner = self._seat.players[k].walt
            for i, r in zip(idx, planner.decide_batch([states[i] for i in idx], [bids[i] for i in idx], sub)):
                out[i] = r
        return out


class SeedVaried:
    """`player`'s spec at a planner seed picked by the deal from `pool`
    seeds starting at `offset`. `player` must be a `WaltPlay`; its own seed
    is ignored."""

    def __init__(self, player: WaltPlay, pool: int = POOL, offset: int = 0):
        if pool < 1:
            raise ValueError("a seed pool holds at least one seed")
        self.pool, self.offset = pool, offset
        self.players = [WaltPlay(**dict(player.kwargs, seed=offset + k)) for k in range(pool)]
        self.walt = _Pooled(self)
        self.last_values: list[dict[int, float]] = []

    def seed_of(self, state: ZebGameState) -> int:
        return self.offset + deal_key(state) % self.pool

    def by_seed(self, states: Sequence[ZebGameState]) -> dict[int, list[int]]:
        """The states' indices by their pool position, in order."""
        by: dict[int, list[int]] = {}
        for i, s in enumerate(states):
            by.setdefault(self.seed_of(s) - self.offset, []).append(i)
        return dict(sorted(by.items()))

    def plays(self, decl: int) -> bool:
        return self.players[0].plays(decl)

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
        auctions: Sequence | None = None,
    ) -> list[int]:
        out: list[int | None] = [None] * len(states)
        values: list = [None] * len(states)
        for k, idx in self.by_seed(states).items():
            pick = lambda col: None if col is None else [col[i] for i in idx]  # noqa: E731
            player = self.players[k]
            slots = player.choose([states[i] for i in idx], pick(bid_values), pick(marks), marks_to_win,
                                  pick(auctions))
            for i, slot, q in zip(idx, slots, player.last_values):
                out[i], values[i] = slot, q
        self.last_values = values
        return out  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"SeedVaried({self.players[0]!r}, seeds={self.offset}..{self.offset + self.pool - 1})"
