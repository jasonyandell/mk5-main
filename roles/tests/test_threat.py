"""roles gates R1-R5: the threat tensor against the rules authority.

R1 pins THREAT to walt.tables.beat_count (an independent code path over
the same LUTs); R2 pins it to forge.oracle.tables.resolve_trick — the
rules authority itself — on exhaustive led-tile tricks; R3 checks the
walker predicate against brute force; R4 checks monotone promotion; R5
registers the alphabet facts probe 1 relies on (injectivity of the
role-vector map).
"""
import numpy as np

from forge.oracle.declarations import GAME_DECL_IDS, N_DECLS
from forge.oracle.tables import resolve_trick
from walt.tables import N_DOMINOES, get_luts
from roles.threat import (
    ALL_TILES,
    build_threat,
    decl_stack_features,
    hand_features,
    threat_counts,
    threat_tensor,
    walker_mask,
)


def test_r1_beat_count_crosscheck():
    """popcount(THREAT[d]) == 27 - beat_count[d] for every decl and tile."""
    T = build_threat()
    for decl in range(N_DECLS):
        luts = get_luts(decl)
        got = np.bitwise_count(T[decl]).astype(np.int64)
        want = 27 - luts.beat_count.astype(np.int64)
        assert np.array_equal(got, want), f"decl {decl}"


def test_r2_resolve_trick_authority():
    """For every decl, led tile d, and follower e: e threatens d iff a
    trick (d, e, filler, filler) with provably-inert fillers is won by e.
    Fillers are chosen from tiles that neither threaten d nor e under d's
    led suit, so the winner relation isolates the (d, e) pair."""
    T = threat_tensor()
    rng = np.random.default_rng(0)
    checked = 0
    for decl in GAME_DECL_IDS:
        luts = get_luts(decl)
        for d in range(N_DOMINOES):
            ranks = luts.rank[luts.led_suit[d]]
            for e in range(N_DOMINOES):
                if e == d:
                    continue
                inert = np.flatnonzero(
                    (ranks <= min(ranks[d], ranks[e]))
                    & (np.arange(N_DOMINOES) != d)
                    & (np.arange(N_DOMINOES) != e))
                if len(inert) < 2:
                    continue
                f = rng.choice(inert, size=2, replace=False)
                out = resolve_trick(d, (d, e, int(f[0]), int(f[1])), decl)
                threatens = bool((int(T[decl, d]) >> e) & 1)
                assert threatens == (out.winner_offset == 1), \
                    (decl, d, e, out.winner_offset)
                checked += 1
    assert checked > 5000          # exhaustive-ish, not a token sample


def test_r3_walker_bruteforce():
    """walker_mask == brute force over outstanding tiles, random states."""
    rng = np.random.default_rng(1)
    T = threat_tensor()
    for _ in range(300):
        decl = int(rng.choice(GAME_DECL_IDS))
        luts = get_luts(decl)
        tiles = rng.permutation(N_DOMINOES)
        hand = int(np.bitwise_or.reduce(
            (np.uint32(1) << tiles[:4].astype(np.uint32))))
        out_tiles = tiles[4:4 + int(rng.integers(0, 12))]
        out = int(np.bitwise_or.reduce(
            (np.uint32(1) << out_tiles.astype(np.uint32)), initial=np.uint32(0)))
        got = walker_mask(hand, out, decl)
        want = 0
        for t in tiles[:4]:
            ranks = luts.rank[luts.led_suit[t]]
            if not any(ranks[o] > ranks[t] for o in out_tiles):
                want |= 1 << int(t)
        assert got == want, (decl, hand, out)


def test_r4_monotone_promotion():
    """Threat counts never increase as tiles leave the outstanding set."""
    rng = np.random.default_rng(2)
    for _ in range(100):
        decl = int(rng.choice(GAME_DECL_IDS))
        tiles = rng.permutation(N_DOMINOES)
        hand = int(np.bitwise_or.reduce(
            (np.uint32(1) << tiles[:7].astype(np.uint32))))
        out = int(ALL_TILES & ~np.uint32(hand))
        prev = threat_counts(hand, out, decl)
        for t in tiles[7:]:
            out &= ~(1 << int(t))
            cur = threat_counts(hand, out, decl)
            assert np.all(cur <= prev)
            prev = cur
        assert np.all(prev == 0)   # empty out: every tile is a walker


def test_r5_role_vector_injectivity():
    """The role-vector map (per-decl trick ranks when led) separates all
    28 dominoes — the lossless-relabeling fact the brainstorm assumed —
    and no single game decl separates them alone."""
    cols = []
    for decl in GAME_DECL_IDS:
        luts = get_luts(decl)
        ranks = np.array([luts.rank[luts.led_suit[d], d]
                          for d in range(N_DOMINOES)])
        cols.append(ranks)
    M = np.stack(cols, axis=1)                     # [28, D]
    assert len({tuple(r) for r in M.tolist()}) == N_DOMINOES
    for k, decl in enumerate(GAME_DECL_IDS):
        assert len(set(M[:, k].tolist())) < N_DOMINOES, \
            f"decl {decl} alone separates all tiles?!"


def test_features_shapes():
    f = hand_features(0b1111, int(ALL_TILES & ~np.uint32(0b1111)), 6)
    assert f.shape == (10,) and np.isfinite(f).all()
    hands = np.array([0b1111111, (1 << 27) | (1 << 26) | 0b11111],
                     dtype=np.uint32)
    s = decl_stack_features(hands)
    assert s.shape == (2, len(GAME_DECL_IDS), 5) and np.isfinite(s).all()
