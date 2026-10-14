# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Rule tables for the scenario player.

Nothing here restates Texas 42. Every entry is tabulated once per
declaration by calling `forge.oracle.tables` — the rules authority — and
the planner only ever reads the tables. The compiled core derives the same
tables in Rust (`scenario/core/src/rules.rs`), and `test_compiled` holds the
two equal entry for entry. Tiles are forge domino ids 0..27;
tile sets are Python int bitmasks (bit d set == domino d present).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    can_follow,
    led_suit_for_lead_domino,
    trick_rank,
)

N_TILES = 28
ALL_TILES = (1 << N_TILES) - 1
N_LED_SUITS = 8  # pip suits 0..6, 7 = the called suit


def bit(tile: int) -> int:
    return 1 << tile


def tiles_of(mask: int) -> list[int]:
    """Ascending tile ids present in a bitmask."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def mask_of(tiles) -> int:
    m = 0
    for t in tiles:
        m |= 1 << int(t)
    return m


@dataclass(frozen=True)
class Rules:
    decl_id: int
    led_suit: tuple[int, ...]              # [28] led suit when the tile leads
    follow_bits: tuple[int, ...]           # [8]  tiles that follow each led suit
    rank: tuple[tuple[int, ...], ...]      # [8][28] trick rank under each led suit
    count: tuple[int, ...]                 # [28] count points on the tile

    def legal(self, hand: int, led_tile: int | None) -> int:
        """Follow-suit-legal subset of `hand` (whole hand when void or leading)."""
        if led_tile is None:
            return hand
        followers = hand & self.follow_bits[self.led_suit[led_tile]]
        return followers if followers else hand

    def follows(self, tile: int, led_tile: int) -> bool:
        return bool((self.follow_bits[self.led_suit[led_tile]] >> tile) & 1)

    def winner_offset(self, trick: tuple[int, int, int, int]) -> int:
        """Index (0..3, play order) of the winning tile; first max wins ties."""
        ranks = self.rank[self.led_suit[trick[0]]]
        best, best_rank = 0, ranks[trick[0]]
        for i in (1, 2, 3):
            r = ranks[trick[i]]
            if r > best_rank:
                best, best_rank = i, r
        return best

    def trick_points(self, trick: tuple[int, int, int, int]) -> int:
        return 1 + sum(self.count[t] for t in trick)


@lru_cache(maxsize=None)
def rules_for(decl_id: int) -> Rules:
    led_suit = tuple(led_suit_for_lead_domino(t, decl_id) for t in range(N_TILES))
    follow_bits = tuple(
        mask_of(t for t in range(N_TILES) if can_follow(t, ls, decl_id))
        for ls in range(N_LED_SUITS)
    )
    rank = tuple(
        tuple(trick_rank(t, ls, decl_id) for t in range(N_TILES))
        for ls in range(N_LED_SUITS)
    )
    return Rules(
        decl_id=decl_id,
        led_suit=led_suit,
        follow_bits=follow_bits,
        rank=rank,
        count=tuple(DOMINO_COUNT_POINTS),
    )
