"""atlas/algebra.py — the decl-indexed game algebra, resident.

One immutable object holding, for every declaration indexed directly by
``decl_id`` (0..9; decl 8 = doubles-suit is present for table completeness
and flagged non-game, per forge.oracle.declarations), the membership /
rank / threat planes the rest of atlas reads.  Suit is a relation between a
tile and a contract, never a property of the tile: ``can_follow`` is a mask
that changes with the declaration (declaring twos removes the 6-2 from the
sixes).  A domino's role is the state of a precomputed threat relation
against a monotonically shrinking outstanding set; roles only promote.

Every value is derived once, at import time, from ``forge.oracle.tables`` —
the rule authority — and never reimplemented (the walt CONTRACTS doctrine).
atlas runtime imports only ``forge.oracle.tables``; ``walt`` / ``roles`` /
``hoyt`` are parity authorities used by the gates alone.

The one symmetry the endgame census found (gate R6: exactly one nontrivial
isomorphism in 5,039) rides here as data — the pips-2<->3 arrow, an
order-automorphism of the whole algebra transporting twos<->threes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_LOW,
    N_DOMINOES,
    can_follow,
    led_suit_for_lead_domino,
    trick_rank,
)

# Every declaration id, indexed directly. 0..6 pip suits, 7 doubles-trump,
# 8 doubles-suit (non-game, table completeness), 9 notrump.
N_DECLS = 10
# led-suit domain: 0..6 pip suits, 7 = called (trump) suit.
N_LED_SUITS = 8

ALL_TILES = np.uint32((1 << N_DOMINOES) - 1)
_BIT = (np.uint32(1) << np.arange(N_DOMINOES, dtype=np.uint32))


def _swap23(pip: int) -> int:
    if pip == 2:
        return 3
    if pip == 3:
        return 2
    return pip


def _arrow_tile() -> np.ndarray:
    """The pips-2<->3 relabelling as a permutation of domino ids."""
    hl_to_id = {
        (max(DOMINO_HIGH[d], DOMINO_LOW[d]), min(DOMINO_HIGH[d], DOMINO_LOW[d])): d
        for d in range(N_DOMINOES)
    }
    out = np.zeros(N_DOMINOES, dtype=np.int64)
    for d in range(N_DOMINOES):
        h, l = _swap23(DOMINO_HIGH[d]), _swap23(DOMINO_LOW[d])
        out[d] = hl_to_id[(max(h, l), min(h, l))]
    return out


def _arrow_decl() -> np.ndarray:
    out = np.arange(N_DECLS, dtype=np.int64)
    out[2], out[3] = 3, 2
    return out


def _arrow_led() -> np.ndarray:
    """led-suit domain under the arrow: pip suits 2<->3 swap, called fixed."""
    out = np.arange(N_LED_SUITS, dtype=np.int64)
    out[2], out[3] = 3, 2
    return out


@dataclass(frozen=True)
class Algebra:
    """Resident, read-only, decl-indexed rule planes (all numpy)."""

    led: np.ndarray          # int8  [10, 28]     led suit if a tile leads
    rank: np.ndarray         # int8  [10, 8, 28]  trick_rank(tile, led_suit)
    can_follow: np.ndarray   # uint32 [10, 8]     mask of tiles following led_suit
    count: np.ndarray        # int8  [28]         count points on a tile (decl-invariant)
    count_mask: np.uint32    # uint32             tiles carrying count points
    threat: np.ndarray       # uint32 [10, 28]    tiles that beat a tile WHEN LED
    beats: np.ndarray        # uint32 [10, 8, 28] tiles that outrank a tile in a led context
    arrow_tile: np.ndarray   # int64 [28]         2<->3 tile permutation
    arrow_decl: np.ndarray   # int64 [10]         2<->3 decl permutation
    arrow_led: np.ndarray    # int64 [8]          2<->3 led-suit permutation


def _build() -> Algebra:
    led = np.array(
        [[led_suit_for_lead_domino(t, decl) for t in range(N_DOMINOES)]
         for decl in range(N_DECLS)],
        dtype=np.int8,
    )
    rank = np.array(
        [[[trick_rank(t, ls, decl) for t in range(N_DOMINOES)]
          for ls in range(N_LED_SUITS)]
         for decl in range(N_DECLS)],
        dtype=np.int8,
    )
    cf = np.zeros((N_DECLS, N_LED_SUITS), dtype=np.uint32)
    for decl in range(N_DECLS):
        for ls in range(N_LED_SUITS):
            m = np.uint32(0)
            for t in range(N_DOMINOES):
                if can_follow(t, ls, decl):
                    m |= _BIT[t]
            cf[decl, ls] = m

    count = np.array(DOMINO_COUNT_POINTS, dtype=np.int8)
    count_mask = np.uint32(0)
    for t in range(N_DOMINOES):
        if count[t]:
            count_mask |= _BIT[t]

    # beats[decl][ls][t] = tiles strictly outranking t under led suit ls; the
    # leader wins ties (plays first), so "beats" is strict. threat is the
    # when-led slice: threat[decl][t] = beats[decl][led[t]][t].
    beats = np.zeros((N_DECLS, N_LED_SUITS, N_DOMINOES), dtype=np.uint32)
    threat = np.zeros((N_DECLS, N_DOMINOES), dtype=np.uint32)
    for decl in range(N_DECLS):
        for ls in range(N_LED_SUITS):
            r = rank[decl, ls]
            for t in range(N_DOMINOES):
                beats[decl, ls, t] = np.bitwise_or.reduce(
                    _BIT[r > r[t]], initial=np.uint32(0))
        for t in range(N_DOMINOES):
            threat[decl, t] = beats[decl, led[decl, t], t]

    alg = Algebra(
        led=led, rank=rank, can_follow=cf, count=count, count_mask=count_mask,
        threat=threat, beats=beats,
        arrow_tile=_arrow_tile(), arrow_decl=_arrow_decl(), arrow_led=_arrow_led(),
    )
    for a in (alg.led, alg.rank, alg.can_follow, alg.count, alg.threat,
              alg.beats, alg.arrow_tile, alg.arrow_decl, alg.arrow_led):
        a.setflags(write=False)
    return alg


_ALGEBRA: Algebra | None = None


def get_algebra() -> Algebra:
    """The resident algebra, built once."""
    global _ALGEBRA
    if _ALGEBRA is None:
        _ALGEBRA = _build()
    return _ALGEBRA


# --------------------------------------------------------------------------- #
#  tile-set helpers (bit d == domino id d)                                     #
# --------------------------------------------------------------------------- #

def tiles_of(mask: int) -> np.ndarray:
    """Ascending domino ids present in a uint32 bitmask."""
    return np.flatnonzero((int(mask) >> np.arange(N_DOMINOES)) & 1)


def mask_of(tiles) -> int:
    m = 0
    for t in tiles:
        m |= 1 << int(t)
    return m


# --------------------------------------------------------------------------- #
#  trick resolution from the planes (the C5 authority check lives on this)     #
# --------------------------------------------------------------------------- #

def led_suit(lead_tile: int, decl_id: int) -> int:
    return int(get_algebra().led[decl_id, lead_tile])


def follow_mask(led_tile: int | None, decl_id: int) -> int:
    """Mask of tiles that can legally follow ``led_tile`` (whole board if
    leading). Follow-suit is applied against a hand by the caller."""
    if led_tile is None:
        return int(ALL_TILES)
    alg = get_algebra()
    return int(alg.can_follow[decl_id, alg.led[decl_id, led_tile]])


def legal_from_hand(hand_mask: int, led_tile: int | None, decl_id: int) -> int:
    """Follow-suit-legal subset of ``hand_mask``: if any held tile follows the
    led suit you must follow, else the whole hand is legal."""
    hand_mask = int(hand_mask)
    if led_tile is None:
        return hand_mask
    followers = hand_mask & follow_mask(led_tile, decl_id)
    return followers if followers else hand_mask


def trick_winner_offset(lead_tile: int, tiles4, decl_id: int) -> int:
    """Play-order offset (0..3) of the winning tile; leader is offset 0 and
    wins ties (first-max), matching forge.oracle.tables.resolve_trick."""
    alg = get_algebra()
    ls = int(alg.led[decl_id, lead_tile])
    ranks = [int(alg.rank[decl_id, ls, t]) for t in tiles4]
    best = 0
    for i in range(1, 4):
        if ranks[i] > ranks[best]:
            best = i
    return best


def trick_points(tiles4) -> int:
    alg = get_algebra()
    return 1 + int(sum(alg.count[t] for t in tiles4))


# --------------------------------------------------------------------------- #
#  role queries (R1-style beat_count cross-check anchors on these)             #
# --------------------------------------------------------------------------- #

def beat_count_when_led(decl_id: int) -> np.ndarray:
    """int64 [28]: how many of the other 27 tiles each tile BEATS when it is
    led (the walt.beat_count quantity, derived from the threat plane:
    27 - popcount(threat))."""
    alg = get_algebra()
    return 27 - np.bitwise_count(alg.threat[decl_id]).astype(np.int64)
