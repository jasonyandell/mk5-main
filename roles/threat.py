"""roles/threat.py — the threat tensor: domino roles as monotone bitmasks.

Jason's 2026-07-21 brainstorm, made mechanical: a domino's role (walker,
high trump, good lead) is not an attribute — it is the STATE of a
precomputed threat mask against the outstanding set.

    THREAT[decl][d] = 28-bit mask of tiles that beat d when d is LED

That single 10x28 uint32 table (1.1 KB) is the whole role algebra:

    walker(d)       <=>  THREAT[decl][d] & out == 0
    threat_count(d)  =   popcount(THREAT[decl][d] & out)
    high trump      <=>  the trump-suit tile whose threats are all dead

`out` (outstanding tiles: not played, not in the queried hand) only ever
shrinks, so threat masks only empty and roles only PROMOTE — a domino's
position-power over the hand is a monotone lattice walk. That is the
formal content of "as dominoes are played, we eliminate members of the
set": the whole role trajectory is determined by which bits leave `out`,
and the update is one AND per query.

The decl axis is first-class: a hand's decl-unknown meaning is its role
row across all GAME_DECL_IDS columns, computed in one vectorized sweep
(`decl_stack_features`). Bidding is column selection.

Rules are never reimplemented here (walt CONTRACTS doctrine): every value
derives from walt.tables LUTs, which tabulate forge.oracle.tables. The
gates in roles/tests cross-check against `beat_count` (an independent
code path over the same authority) and against `resolve_trick` itself.
"""
from __future__ import annotations

import numpy as np

from forge.oracle.declarations import GAME_DECL_IDS, N_DECLS
from walt.tables import N_DOMINOES, get_luts, hand_to_mask

__all__ = [
    "GAME_DECL_IDS", "ALL_TILES", "build_threat", "threat_tensor",
    "threat_counts", "walker_mask", "boss_mask", "hand_features",
    "decl_stack_features", "FEATURE_NAMES", "hand_to_mask",
]

ALL_TILES = np.uint32((1 << N_DOMINOES) - 1)
_BIT = (np.uint32(1) << np.arange(N_DOMINOES, dtype=np.uint32))


def build_threat() -> np.ndarray:
    """uint32 [N_DECLS, 28]: THREAT[decl][d] = tiles that beat d when d
    is led under decl (strictly higher trick rank — the leader wins
    ties because the leader plays first)."""
    T = np.zeros((N_DECLS, N_DOMINOES), dtype=np.uint32)
    for decl in range(N_DECLS):
        luts = get_luts(decl)
        for d in range(N_DOMINOES):
            ranks = luts.rank[luts.led_suit[d]]
            T[decl, d] = np.bitwise_or.reduce(
                _BIT[ranks > ranks[d]], initial=np.uint32(0))
    return T


_THREAT: np.ndarray | None = None


def threat_tensor() -> np.ndarray:
    global _THREAT
    if _THREAT is None:
        _THREAT = build_threat()
        _THREAT.setflags(write=False)
    return _THREAT


# --------------------------------------------------------------------------- #
#  role queries — every one is ANDs + popcounts over the monotone `out`        #
# --------------------------------------------------------------------------- #

def _tiles_of(mask: int) -> np.ndarray:
    return np.flatnonzero((int(mask) >> np.arange(N_DOMINOES)) & 1)


def threat_counts(hand: int, out: int, decl: int) -> np.ndarray:
    """int64 [k]: live threats against each hand tile (ascending tile id)
    if it were led now. 0 = walker."""
    T = threat_tensor()
    tiles = _tiles_of(hand)
    return np.bitwise_count(T[decl, tiles] & np.uint32(out)).astype(np.int64)


def walker_mask(hand: int, out: int, decl: int) -> int:
    """Bitmask of hand tiles that are walkers (unbeatable when led) against
    the outstanding set `out`."""
    T = threat_tensor()
    tiles = _tiles_of(hand)
    w = tiles[(T[decl, tiles] & np.uint32(out)) == 0]
    return int(np.bitwise_or.reduce(_BIT[w], initial=np.uint32(0)))


def boss_mask(out_or_hand: int, decl: int) -> int:
    """Bitmask of the boss tiles among a live set: tiles no OTHER live tile
    beats when they lead. (The high trump is the boss of the called suit;
    every suit's current boss is one AND away.)"""
    T = threat_tensor()
    tiles = _tiles_of(out_or_hand)
    live = np.uint32(out_or_hand)
    b = tiles[(T[decl, tiles] & live) == 0]
    return int(np.bitwise_or.reduce(_BIT[b], initial=np.uint32(0)))


# --------------------------------------------------------------------------- #
#  fixed-decl hand features (probe 2's role basis)                             #
# --------------------------------------------------------------------------- #

FEATURE_NAMES = (
    "n_walkers",            # hand tiles unbeatable-when-led vs out
    "n_trump",              # hand tiles in the called suit
    "has_high_trump",       # hand holds the boss of the called suit
    "n_boss",               # hand tiles that are bosses of the live set
    "sum_threats",          # total live threats over hand tiles
    "min_threats",          # the safest lead's threat count
    "count_in_hand",        # count points held
    "count_walking",        # count points on walkers (cashable now)
    "count_out",            # count points still outstanding
    "n_out",                # outstanding tile count
)


def hand_features(hand: int, out: int, decl: int) -> np.ndarray:
    """float64 [len(FEATURE_NAMES)] role-basis summary of a hand vs the
    outstanding set, fixed decl. Pure mask algebra — no pip identity."""
    luts = get_luts(decl)
    T = threat_tensor()
    tiles = _tiles_of(hand)
    out32 = np.uint32(out)
    live = np.uint32(out) | np.uint32(hand)
    tc = np.bitwise_count(T[decl, tiles] & out32).astype(np.int64)
    trump = luts.can_follow[7, tiles]
    boss = (T[decl, tiles] & live) == 0
    cnt = luts.count[tiles].astype(np.int64)
    out_tiles = _tiles_of(out)
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
        float(luts.count[out_tiles].sum()),
        float(len(out_tiles)),
    ])


# --------------------------------------------------------------------------- #
#  the decl-unknown stack — one hand, every declaration, one sweep             #
# --------------------------------------------------------------------------- #

_STACK_NAMES = ("n_trump", "has_high_trump", "n_walkers_cold",
                "sum_threats_cold", "count_in_hand")


def decl_stack_features(hands: np.ndarray) -> np.ndarray:
    """float64 [B, len(GAME_DECL_IDS), 5]: every hand's role summary in
    every game declaration, vectorized (`out` = the 21 unseen tiles —
    the auction state, where nothing has been played).

    Columns per decl: trump length, high-trump ownership, cold walkers
    (unbeatable even with all 21 unseen tiles live), total live threats,
    count points held. This is the flat 28xD object from the brainstorm:
    bidding reads it as column selection."""
    T = threat_tensor()
    hands = np.asarray(hands, dtype=np.uint32).reshape(-1)
    B = len(hands)
    member = ((hands[:, None] >> np.arange(N_DOMINOES, dtype=np.uint32)) &
              np.uint32(1)).astype(bool)                       # [B, 28]
    out = (~hands) & ALL_TILES                                 # [B]
    feats = np.zeros((B, len(GAME_DECL_IDS), len(_STACK_NAMES)))
    for k, decl in enumerate(GAME_DECL_IDS):
        luts = get_luts(decl)
        thr = np.bitwise_count(T[decl][None, :] & out[:, None]) \
            .astype(np.int64)                                  # [B, 28]
        trump = luts.can_follow[7]                             # [28]
        cold = thr == 0
        # high trump: hand holds the tile that beats every other tile of
        # the called suit — the trump whose full-set threat mask is empty
        # of trumps; equivalently the max-rank trump, precomputed:
        boss_trump = int(np.argmax(np.where(
            trump, luts.rank[7], -1))) if trump.any() else -1
        feats[:, k, 0] = (member & trump[None, :]).sum(1)
        feats[:, k, 1] = member[:, boss_trump] if boss_trump >= 0 else 0.0
        feats[:, k, 2] = (member & cold).sum(1)
        feats[:, k, 3] = np.where(member, thr, 0).sum(1)
        feats[:, k, 4] = np.where(
            member, luts.count[None, :].astype(np.int64), 0).sum(1)
    return feats
