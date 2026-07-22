"""atlas/roles.py — role state at a coordinate.

A domino's role — walker, boss, lead quality — is not an attribute but the
STATE of a precomputed threat relation against a monotonically shrinking
outstanding set; roles only promote.  This module is thin, derived and
exact: every query is ANDs and popcounts over the algebra's THREAT plane and
the coordinate's masks.

It mirrors (and will eventually absorb) roles/threat.py, but atlas does NOT
import roles/ — the planes come from atlas.algebra.  The parity is gated in
tests; roles/ stays until consumers migrate, then no-legacy deletes it.
"""
from __future__ import annotations

import numpy as np

from forge.oracle.declarations import GAME_DECL_IDS

from atlas.algebra import ALL_TILES, N_DOMINOES, get_algebra, tiles_of

FEATURE_NAMES = (
    "n_walkers",       # hand tiles unbeatable-when-led vs out
    "n_trump",         # hand tiles in the called suit
    "has_high_trump",  # hand holds the boss of the called suit
    "n_boss",          # hand tiles that are bosses of the live set
    "sum_threats",     # total live threats over hand tiles
    "min_threats",     # the safest lead's threat count
    "count_in_hand",   # count points held
    "count_walking",   # count points on walkers (cashable now)
    "count_out",       # count points still outstanding
    "n_out",           # outstanding tile count
)

_STACK_NAMES = ("n_trump", "has_high_trump", "n_walkers_cold",
                "sum_threats_cold", "count_in_hand")


# --------------------------------------------------------------------------- #
#  outstanding set at a coordinate                                             #
# --------------------------------------------------------------------------- #

def outstanding(coord, hand_mask: int) -> int:
    """Tiles live against ``hand_mask`` at a coordinate: everything not played
    by anyone and not in the hand itself."""
    played_union = 0
    for m in coord.played:
        played_union |= int(m)
    return int(ALL_TILES) & ~played_union & ~int(hand_mask)


# --------------------------------------------------------------------------- #
#  role queries — ANDs + popcounts over the monotone `out`                     #
# --------------------------------------------------------------------------- #

def threat_counts(hand: int, out: int, decl: int) -> np.ndarray:
    """int64 [k]: live threats against each hand tile (ascending tile id) if
    it were led now. 0 == walker."""
    T = get_algebra().threat
    tiles = tiles_of(hand)
    return np.bitwise_count(T[decl, tiles] & np.uint32(out)).astype(np.int64)


def walker_mask(hand: int, out: int, decl: int) -> int:
    """Mask of hand tiles that are walkers (unbeatable when led vs out)."""
    alg = get_algebra()
    tiles = tiles_of(hand)
    w = tiles[(alg.threat[decl, tiles] & np.uint32(out)) == 0]
    bit = np.uint32(1) << w.astype(np.uint32)
    return int(np.bitwise_or.reduce(bit, initial=np.uint32(0)))


def boss_mask(out_or_hand: int, decl: int) -> int:
    """Mask of boss tiles among a live set: tiles no OTHER live tile beats
    when they lead."""
    alg = get_algebra()
    tiles = tiles_of(out_or_hand)
    live = np.uint32(out_or_hand)
    b = tiles[(alg.threat[decl, tiles] & live) == 0]
    bit = np.uint32(1) << b.astype(np.uint32)
    return int(np.bitwise_or.reduce(bit, initial=np.uint32(0)))


# --------------------------------------------------------------------------- #
#  fixed-decl hand features                                                    #
# --------------------------------------------------------------------------- #

def hand_features(hand: int, out: int, decl: int) -> np.ndarray:
    """float64 [10] role-basis summary of a hand vs the outstanding set, fixed
    decl. Pure mask algebra — no pip identity."""
    alg = get_algebra()
    tiles = tiles_of(hand)
    out32 = np.uint32(out)
    live = np.uint32(out) | np.uint32(hand)
    tc = np.bitwise_count(alg.threat[decl, tiles] & out32).astype(np.int64)
    trump = ((int(alg.can_follow[decl, 7]) >> tiles) & 1).astype(bool)
    boss = (alg.threat[decl, tiles] & live) == 0
    cnt = alg.count[tiles].astype(np.int64)
    out_tiles = tiles_of(out)
    high_trump = bool(np.any(trump & boss)) if len(tiles) else False
    return np.array([
        float(np.sum(tc == 0)),
        float(np.sum(trump)),
        float(high_trump),
        float(np.sum(boss)),
        float(tc.sum()),
        float(tc.min()) if len(tc) else 0.0,
        float(cnt.sum()),
        float(cnt[tc == 0].sum()),
        float(alg.count[out_tiles].sum()),
        float(len(out_tiles)),
    ])


# --------------------------------------------------------------------------- #
#  the decl-unknown stack — one 7-tile hand, every game declaration            #
# --------------------------------------------------------------------------- #

def decl_stack_features(hands: np.ndarray) -> np.ndarray:
    """float64 [B, len(GAME_DECL_IDS), 5]: every hand's role summary in every
    game declaration, vectorized (`out` = the 21 unseen tiles — the auction
    state). Columns: trump length, high-trump ownership, cold walkers, total
    live threats, count points held. Bidding reads this as column selection."""
    alg = get_algebra()
    hands = np.asarray(hands, dtype=np.uint32).reshape(-1)
    B = len(hands)
    member = ((hands[:, None] >> np.arange(N_DOMINOES, dtype=np.uint32))
              & np.uint32(1)).astype(bool)
    out = (~hands) & ALL_TILES
    feats = np.zeros((B, len(GAME_DECL_IDS), len(_STACK_NAMES)))
    for k, decl in enumerate(GAME_DECL_IDS):
        thr = np.bitwise_count(alg.threat[decl][None, :] & out[:, None]).astype(np.int64)
        trump = ((int(alg.can_follow[decl, 7]) >> np.arange(N_DOMINOES)) & 1).astype(bool)
        cold = thr == 0
        rank7 = alg.rank[decl, 7].astype(np.int64)
        boss_trump = int(np.argmax(np.where(trump, rank7, -1))) if trump.any() else -1
        feats[:, k, 0] = (member & trump[None, :]).sum(1)
        feats[:, k, 1] = member[:, boss_trump] if boss_trump >= 0 else 0.0
        feats[:, k, 2] = (member & cold).sum(1)
        feats[:, k, 3] = np.where(member, thr, 0).sum(1)
        feats[:, k, 4] = np.where(
            member, alg.count[None, :].astype(np.int64), 0).sum(1)
    return feats


# --------------------------------------------------------------------------- #
#  coordinate-level convenience                                                #
# --------------------------------------------------------------------------- #

def role_summary(coord, hand_mask: int | None = None) -> np.ndarray:
    """hand_features for a holding at a coordinate — the viewer's own hand by
    default, or any hypothetical world's hand — under the coordinate's decl."""
    hand = coord.viewer_hand if hand_mask is None else int(hand_mask)
    return hand_features(hand, outstanding(coord, hand), coord.decl_id)
