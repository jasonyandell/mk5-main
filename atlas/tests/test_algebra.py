"""atlas algebra gates.

A1 pins atlas trick resolution to the rule authority itself
(forge.oracle.tables.resolve_trick) on exhaustive led-tile tricks — this is
gate C5's resolve_trick authority check.  A2 is the R1-style beat_count
cross-check against walt.tables (an independent code path over the same
LUTs).  A3-A5 pin the threat/beats planes, and A6-A8 the 2<->3 arrow as a
genuine order-automorphism of the whole algebra (the one symmetry, gate R6).
"""
import numpy as np

from forge.oracle.declarations import GAME_DECL_IDS
from forge.oracle.tables import resolve_trick
from walt.tables import get_luts

from atlas.algebra import (
    ALL_TILES,
    N_DECLS,
    N_DOMINOES,
    N_LED_SUITS,
    beat_count_when_led,
    get_algebra,
    legal_from_hand,
    trick_points,
    trick_winner_offset,
)


def test_a1_resolve_trick_authority():
    """atlas trick resolution == forge.oracle.tables.resolve_trick on
    exhaustive (decl, lead, random-fillers) tricks, all 10 decls."""
    rng = np.random.default_rng(0)
    checked = 0
    for decl in range(N_DECLS):
        for lead in range(N_DOMINOES):
            others = [t for t in range(N_DOMINOES) if t != lead]
            for _ in range(6):
                f = rng.choice(others, size=3, replace=False)
                tiles4 = (lead, int(f[0]), int(f[1]), int(f[2]))
                ref = resolve_trick(lead, tiles4, decl)
                assert trick_winner_offset(lead, tiles4, decl) == ref.winner_offset, \
                    (decl, tiles4)
                assert trick_points(tiles4) == ref.points, (decl, tiles4)
                checked += 1
    assert checked > 1500


def test_a2_beat_count_crosscheck():
    """beat_count_when_led == walt.tables beat_count for every game decl."""
    for decl in GAME_DECL_IDS:
        got = beat_count_when_led(decl)
        want = get_luts(decl).beat_count.astype(np.int64)
        assert np.array_equal(got, want), f"decl {decl}"


def test_a3_threat_is_beats_when_led():
    """threat[decl][t] == beats[decl][led[t]][t] for every decl and tile."""
    alg = get_algebra()
    for decl in range(N_DECLS):
        for t in range(N_DOMINOES):
            assert alg.threat[decl, t] == alg.beats[decl, alg.led[decl, t], t]


def test_a4_beats_matches_rank_order():
    """beats[decl][ls][t] holds exactly the tiles whose rank strictly exceeds
    t's under led suit ls."""
    alg = get_algebra()
    rng = np.random.default_rng(1)
    for _ in range(400):
        decl = int(rng.integers(0, N_DECLS))
        ls = int(rng.integers(0, N_LED_SUITS))
        t = int(rng.integers(0, N_DOMINOES))
        r = alg.rank[decl, ls]
        want = int(np.bitwise_or.reduce(
            (np.uint32(1) << np.flatnonzero(r > r[t]).astype(np.uint32)),
            initial=np.uint32(0)))
        assert int(alg.beats[decl, ls, t]) == want


def test_a5_legal_follow_suit():
    """legal_from_hand honours follow-suit: follow if able, else whole hand."""
    alg = get_algebra()
    rng = np.random.default_rng(2)
    for _ in range(500):
        decl = int(rng.integers(0, N_DECLS))
        tiles = rng.permutation(N_DOMINOES)
        hand = int(np.bitwise_or.reduce(
            (np.uint32(1) << tiles[:7].astype(np.uint32))))
        lead = int(tiles[7])
        got = legal_from_hand(hand, lead, decl)
        followers = hand & int(alg.can_follow[decl, alg.led[decl, lead]])
        want = followers if followers else hand
        assert got == want
        assert legal_from_hand(hand, None, decl) == hand


def test_a6_arrow_is_involution():
    alg = get_algebra()
    assert np.array_equal(alg.arrow_tile[alg.arrow_tile], np.arange(N_DOMINOES))
    assert np.array_equal(alg.arrow_decl[alg.arrow_decl], np.arange(N_DECLS))
    assert np.array_equal(alg.arrow_led[alg.arrow_led], np.arange(N_LED_SUITS))
    # only twos<->threes move (decls and pip led-suits); everything else fixed
    assert list(alg.arrow_decl) == [0, 1, 3, 2, 4, 5, 6, 7, 8, 9]
    fixed_tiles = [t for t in range(N_DOMINOES) if alg.arrow_tile[t] == t]
    assert len(fixed_tiles) == 16  # 28 minus the 12 tiles that carry a 2 xor a 3


def _is_iso(alg, decl: int) -> bool:
    """The arrow transports the decl-d game onto its image (arrow_decl[d]):
    count, led suit, follow set and every within-context rank ORDER carried.
    Mirrors gate R6's _is_iso, on atlas's own planes."""
    at, ad, als = alg.arrow_tile, alg.arrow_decl, alg.arrow_led
    d2 = int(ad[decl])
    if not np.array_equal(alg.count[at], alg.count):
        return False
    if not np.array_equal(als[alg.led[decl]][at], alg.led[d2]):
        return False
    for ls in range(N_LED_SUITS):
        ls2 = int(als[ls])
        cf = ((int(alg.can_follow[decl, ls]) >> np.arange(N_DOMINOES)) & 1)
        cf2 = ((int(alg.can_follow[d2, ls2]) >> np.arange(N_DOMINOES)) & 1)
        if not np.array_equal(cf.astype(bool), cf2[at].astype(bool)):
            return False
        ra = alg.rank[decl, ls].astype(int)
        rb = alg.rank[d2, ls2].astype(int)[at]
        if not np.array_equal(np.sign(ra[:, None] - ra[None, :]),
                              np.sign(rb[:, None] - rb[None, :])):
            return False
    return True


def test_a7_arrow_transports_twos_to_threes_only():
    """The arrow is a game isomorphism for EXACTLY decls 2 and 3 (it carries
    the twos-game onto the threes-game); count marks and the higher-end lead
    rule kill 2<->3 in every other declaration (gate R6)."""
    alg = get_algebra()
    assert {d for d in range(N_DECLS) if _is_iso(alg, d)} == {2, 3}


def test_a8_arrow_matches_r6_tile_permutation():
    """atlas's arrow tile permutation is the R6 pips-2<->3 relabelling and
    leaves count marks globally invariant."""
    from forge.oracle.tables import DOMINOES

    alg = get_algebra()
    tid = {frozenset(d): i for i, d in enumerate(DOMINOES)}
    swap23 = (0, 1, 3, 2, 4, 5, 6)
    want = np.array([tid[frozenset({swap23[h], swap23[l]})] for h, l in DOMINOES])
    assert np.array_equal(alg.arrow_tile, want)
    assert np.array_equal(alg.count[alg.arrow_tile], alg.count)


def test_a9_arrays_readonly():
    alg = get_algebra()
    for a in (alg.rank, alg.threat, alg.beats, alg.can_follow, alg.count):
        assert not a.flags.writeable
    assert int(ALL_TILES) == (1 << N_DOMINOES) - 1
