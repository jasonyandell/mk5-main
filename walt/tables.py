"""walt/tables.py — rule LUTs, all TABULATED from forge.oracle.tables.

Nothing here reimplements Texas 42 rules; every value is derived by calling
`forge.oracle.tables` functions once at LUT-build time and cached per decl_id.
Parity against the zeb engine is enforced in tests/test_tables.py.

Inside walt everything speaks forge domino ids 0..27; hands/tile-sets are
uint32 bitmasks (bit d set == domino id d present).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, "/Users/jason/code/mk5-main/.claude/worktrees/walt")

from forge.oracle.tables import (  # noqa: E402
    DOMINO_COUNT_POINTS,
    can_follow,
    led_suit_for_lead_domino,
    trick_rank,
)

N_DOMINOES = 28
# led_suit domain: 0..6 pip suits, 7 = called (trump) suit.
N_LED_SUITS = 8


@dataclass(frozen=True)
class LUTs:
    decl_id: int
    led_suit: np.ndarray      # int8  [28]     led suit if this tile leads
    can_follow: np.ndarray    # bool  [8, 28]  can_follow(tile, led_suit)
    rank: np.ndarray          # int8  [8, 28]  trick_rank(tile, led_suit)
    count: np.ndarray         # int8  [28]     count points on the tile
    beat_count: np.ndarray    # int8  [28]     #tiles (of other 27) this beats when LED
    can_follow_bits: np.ndarray  # uint32 [8]  bitmask of tiles that follow led_suit


_CACHE: dict[int, LUTs] = {}


def _build(decl_id: int) -> LUTs:
    led_suit = np.array(
        [led_suit_for_lead_domino(t, decl_id) for t in range(N_DOMINOES)],
        dtype=np.int8,
    )
    cf = np.array(
        [[can_follow(t, ls, decl_id) for t in range(N_DOMINOES)] for ls in range(N_LED_SUITS)],
        dtype=bool,
    )
    rank = np.array(
        [[trick_rank(t, ls, decl_id) for t in range(N_DOMINOES)] for ls in range(N_LED_SUITS)],
        dtype=np.int8,
    )
    count = np.array(DOMINO_COUNT_POINTS, dtype=np.int8)

    # beat_count[t]: when t is LED (led_suit = led_suit[t]), how many of the
    # other 27 tiles it beats. t plays first, so ties break to t (first-max):
    # t beats o iff rank(o) <= rank(t) under t's led suit.
    beat_count = np.zeros(N_DOMINOES, dtype=np.int8)
    for t in range(N_DOMINOES):
        col = rank[led_suit[t]]  # ranks of all tiles under t's led suit
        beat_count[t] = int(np.sum(col <= col[t]) - 1)  # -1 excludes self

    tile_bit = (np.uint32(1) << np.arange(N_DOMINOES, dtype=np.uint32))
    can_follow_bits = np.array(
        [np.bitwise_or.reduce(tile_bit[cf[ls]]) if cf[ls].any() else np.uint32(0)
         for ls in range(N_LED_SUITS)],
        dtype=np.uint32,
    )

    return LUTs(
        decl_id=decl_id,
        led_suit=led_suit,
        can_follow=cf,
        rank=rank,
        count=count,
        beat_count=beat_count,
        can_follow_bits=can_follow_bits,
    )


def get_luts(decl_id: int) -> LUTs:
    """Return cached LUTs for a declaration (decl 8 doubles-suit is purged)."""
    luts = _CACHE.get(decl_id)
    if luts is None:
        luts = _build(decl_id)
        _CACHE[decl_id] = luts
    return luts


def resolve_tricks(leader, tiles4, decl_id: int):
    """Vectorized trick resolution.

    leader: (B,) leader SEAT index (0..3). tiles4: (B,4) domino ids in play
    order (tiles4[:,0] is the leader's tile). Returns (winner_seat (B,),
    points (B,)). Parity-tested against forge.oracle.tables.resolve_trick.
    """
    luts = get_luts(decl_id)
    leader = np.asarray(leader, dtype=np.int64)
    tiles4 = np.asarray(tiles4, dtype=np.int64)
    led = luts.led_suit[tiles4[:, 0]].astype(np.int64)     # (B,)
    ranks = luts.rank[led[:, None], tiles4]                 # (B,4)
    offset = np.argmax(ranks, axis=1)                       # first-max tie break
    winner = (leader + offset) % 4
    points = 1 + luts.count[tiles4].astype(np.int64).sum(axis=1)
    return winner.astype(np.int64), points.astype(np.int64)


def legal_moves_mask(hand_mask, led_tile, decl_id: int):
    """Follow-suit-legal move mask per the engine.

    hand_mask: uint32 bitmask, scalar or array (B,). led_tile: the led domino
    id, or None when leading (whole hand legal). If no tile can follow, the
    whole hand is legal. Returns a uint32 mask matching the input shape.
    Parity-tested against zeb legal_actions.
    """
    luts = get_luts(decl_id)
    arr = np.asarray(hand_mask, dtype=np.uint32)
    scalar = arr.ndim == 0
    hm = np.atleast_1d(arr)

    if led_tile is None:
        out = hm.copy()
    else:
        led = int(luts.led_suit[int(led_tile)])
        fb = np.uint32(luts.can_follow_bits[led])
        followers = hm & fb
        out = np.where(followers != np.uint32(0), followers, hm).astype(np.uint32)

    return np.uint32(out[0]) if scalar else out


def hand_to_mask(tiles) -> np.uint32:
    """OR of single-bit masks for an iterable of domino ids."""
    m = np.uint32(0)
    for t in tiles:
        m |= np.uint32(1) << np.uint32(int(t))
    return m


def mask_to_tiles(mask) -> list[int]:
    """Sorted domino ids present in a uint32 bitmask."""
    m = int(mask)
    return [d for d in range(N_DOMINOES) if (m >> d) & 1]
