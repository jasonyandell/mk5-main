# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The fiber — every hidden-hand completion of an information state — and
exact-uniform sampling from it.

Given the unseen pool, the three hidden seats' hand sizes, and the tiles
each seat provably cannot hold, the number of lawful assignments is
counted by a small dynamic program over (next tile, remaining capacities),
and a sample is drawn tile by tile in proportion to those counts. Uniform
by construction; no rejection, no invalid worlds, never empty when the
information state is real (the true deal is always a member).

When the fiber is no larger than the sample asked for — 90 completions with
two tricks left, 1,680 with three — it is enumerated outright instead: the
exact belief at no extra cost, and never a duplicate scenario.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from scenario.rules import ALL_TILES, bit, tiles_of


class Uniform(Protocol):
    """Any source of uniform floats on [0, 1) — `random.Random` or the
    planner's SplitMix stream."""

    def random(self) -> float: ...


@lru_cache(maxsize=4096)
def _fiber(tiles: tuple[int, ...], sizes: tuple[int, int, int],
           forbid: tuple[int, int, int]):
    """Counting table for assigning `tiles` to three seats with the given
    capacities, never giving a seat a forbidden tile."""
    n = len(tiles)

    @lru_cache(maxsize=None)
    def ways(i: int, c0: int, c1: int, c2: int) -> int:
        if i == n:
            return 1 if c0 == c1 == c2 == 0 else 0
        t = tiles[i]
        total = 0
        if c0 and not (forbid[0] >> t) & 1:
            total += ways(i + 1, c0 - 1, c1, c2)
        if c1 and not (forbid[1] >> t) & 1:
            total += ways(i + 1, c0, c1 - 1, c2)
        if c2 and not (forbid[2] >> t) & 1:
            total += ways(i + 1, c0, c1, c2 - 1)
        return total

    return ways


def fiber_size(pool: int, sizes: tuple[int, int, int],
               forbid: tuple[int, int, int]) -> int:
    tiles = tuple(tiles_of(pool))
    if sum(sizes) != len(tiles):
        raise ValueError(f"pool has {len(tiles)} tiles but sizes sum to {sum(sizes)}")
    return _fiber(tiles, sizes, forbid)(0, *sizes)


def sample_fiber(pool: int, sizes: tuple[int, int, int],
                 forbid: tuple[int, int, int], rng: Uniform) -> tuple[int, int, int]:
    """One assignment of the pool to three seats, uniform over the fiber."""
    tiles = tuple(tiles_of(pool))
    ways = _fiber(tiles, sizes, forbid)
    caps = list(sizes)
    total = ways(0, *caps)
    if total == 0:
        raise ValueError("empty fiber: no lawful completion exists")
    masks = [0, 0, 0]
    for i, t in enumerate(tiles):
        u = rng.random() * total
        acc = 0
        chosen = -1
        for j in range(3):
            if caps[j] and not (forbid[j] >> t) & 1:
                caps[j] -= 1
                w = ways(i + 1, *caps)
                caps[j] += 1
                acc += w
                if u < acc:
                    chosen = j
                    total = w
                    break
        if chosen < 0:  # floating-point edge: take the last admissible seat
            for j in (2, 1, 0):
                if caps[j] and not (forbid[j] >> t) & 1:
                    caps[j] -= 1
                    total = ways(i + 1, *caps)
                    caps[j] += 1
                    chosen = j
                    break
        caps[chosen] -= 1
        masks[chosen] |= bit(t)
    return (masks[0], masks[1], masks[2])


def enumerate_fiber(pool: int, sizes: tuple[int, int, int],
                    forbid: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    """Every assignment of the pool to three seats, in tile-by-tile,
    seat-0-first order (the same order the counting table walks)."""
    tiles = tuple(tiles_of(pool))
    ways = _fiber(tiles, sizes, forbid)
    out: list[tuple[int, int, int]] = []
    caps = list(sizes)
    masks = [0, 0, 0]

    def walk(i: int) -> None:
        if i == len(tiles):
            out.append((masks[0], masks[1], masks[2]))
            return
        t = tiles[i]
        for j in range(3):
            if caps[j] and not (forbid[j] >> t) & 1:
                caps[j] -= 1
                if ways(i + 1, *caps):
                    masks[j] |= bit(t)
                    walk(i + 1)
                    masks[j] ^= bit(t)
                caps[j] += 1

    walk(0)
    return out


class Fiber:
    """The viewer's fiber: the hidden seats (absolute, in order), the unseen
    pool, and each one's hand size and provable voids. Its members are
    scenarios — 4-tuples of remaining-hand masks in absolute seat order, the
    viewer's own hand in its slot."""

    def __init__(self, viewer: int, own: int, played: int, sizes4: tuple[int, int, int, int],
                 forbid4: tuple[int, int, int, int]):
        self.viewer, self.own = viewer, own
        self.others = [s for s in range(4) if s != viewer]
        self.pool = ALL_TILES & ~played & ~own
        self.sizes = (sizes4[self.others[0]], sizes4[self.others[1]], sizes4[self.others[2]])
        self.forbid = (forbid4[self.others[0]], forbid4[self.others[1]], forbid4[self.others[2]])
        self.size = fiber_size(self.pool, self.sizes, self.forbid)

    def _world(self, hidden: tuple[int, int, int]) -> tuple[int, int, int, int]:
        hands = [0, 0, 0, 0]
        hands[self.viewer] = self.own
        for s, m in zip(self.others, hidden):
            hands[s] = m
        return (hands[0], hands[1], hands[2], hands[3])

    def worlds(self) -> list[tuple[int, int, int, int]]:
        """Every member, in `enumerate_fiber`'s order."""
        return [self._world(h) for h in enumerate_fiber(self.pool, self.sizes, self.forbid)]

    def draw(self, rng: Uniform) -> tuple[int, int, int, int]:
        """One member, uniform."""
        return self._world(sample_fiber(self.pool, self.sizes, self.forbid, rng))


def sample_worlds(viewer: int, own: int, played: int, sizes4: tuple[int, int, int, int],
                  forbid4: tuple[int, int, int, int], n: int,
                  rng: Uniform) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Scenarios from the viewer's fiber: the whole fiber when it has at most
    `n` members, else `n` uniform draws. The flag says which: True when the
    scenarios ARE the fiber (the exact belief), False when they are a sample
    of it."""
    fiber = Fiber(viewer, own, played, sizes4, forbid4)
    if fiber.size <= n:
        return fiber.worlds(), True
    return [fiber.draw(rng) for _ in range(n)], False
