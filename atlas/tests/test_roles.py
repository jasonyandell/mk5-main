"""atlas gate C5 — role parity against roles/threat.py.

atlas.roles reimplements the role algebra on atlas.algebra's planes (no
roles/ import at runtime). C5 pins every predicate to roles/threat.py — the
existing authority — on random states: the threat plane itself, threat
counts, walker/boss masks, the 10-feature hand basis, and the decl-unknown
28xD stack. (The resolve_trick authority half of C5 is gate A1 in
test_algebra.) Also checks role_summary reads hypothetical world hands at a
coordinate.
"""
import random

import numpy as np

from forge.oracle.declarations import GAME_DECL_IDS
from forge.zeb.game import apply_action, is_terminal, legal_actions, new_game

# parity authority (tests only)
from roles import threat as R

from atlas import roles as A
from atlas.algebra import ALL_TILES, get_algebra
from atlas.coordinate import from_engine
from atlas.fiber import fiber, hidden_seats


def _random_hand_out(rng):
    decl = int(rng.choice(GAME_DECL_IDS))
    tiles = rng.permutation(28)
    n_hand = int(rng.integers(1, 8))
    hand = int(np.bitwise_or.reduce((np.uint32(1) << tiles[:n_hand].astype(np.uint32))))
    n_out = int(rng.integers(0, 15))
    out_tiles = tiles[n_hand:n_hand + n_out]
    out = int(np.bitwise_or.reduce(
        (np.uint32(1) << out_tiles.astype(np.uint32)), initial=np.uint32(0)))
    return hand, out, decl


def test_c5a_threat_plane_matches():
    T_atlas = get_algebra().threat
    T_roles = R.threat_tensor()
    assert np.array_equal(T_atlas, T_roles)


def test_c5b_role_queries_match():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        hand, out, decl = _random_hand_out(rng)
        assert np.array_equal(A.threat_counts(hand, out, decl),
                              R.threat_counts(hand, out, decl))
        assert A.walker_mask(hand, out, decl) == R.walker_mask(hand, out, decl)
        assert A.boss_mask(out, decl) == R.boss_mask(out, decl)
        assert A.boss_mask(hand | out, decl) == R.boss_mask(hand | out, decl)


def test_c5c_hand_features_match():
    rng = np.random.default_rng(1)
    for _ in range(3000):
        hand, out, decl = _random_hand_out(rng)
        fa = A.hand_features(hand, out, decl)
        fr = R.hand_features(hand, out, decl)
        assert np.allclose(fa, fr), (decl, fa, fr)
    assert A.FEATURE_NAMES == R.FEATURE_NAMES


def test_c5d_decl_stack_matches():
    rng = np.random.default_rng(2)
    hands = np.array(
        [int(np.bitwise_or.reduce(
            (np.uint32(1) << rng.permutation(28)[:7].astype(np.uint32))))
         for _ in range(500)], dtype=np.uint32)
    assert np.allclose(A.decl_stack_features(hands), R.decl_stack_features(hands))


def test_c5e_role_summary_on_world_hands():
    """role_summary reads the viewer's hand and any hypothetical world hand at
    a coordinate, matching roles/threat on the derived (hand, out)."""
    rng = random.Random(4)
    checked = 0
    for g in range(60):
        st = new_game(70_000 + g)
        r = random.Random(g)
        target = r.randint(3, 5)
        while len(st.play_history) < target * 4:
            st = apply_action(st, r.choice(legal_actions(st)))
        if is_terminal(st) or st.decl_id == 8:
            continue
        viewer = r.randint(0, 3)
        coord = from_engine(st, viewer)
        # viewer's own hand
        out_v = A.outstanding(coord, coord.viewer_hand)
        assert np.allclose(A.role_summary(coord),
                           R.hand_features(coord.viewer_hand, out_v, coord.decl_id))
        # a hypothetical world's hand for a hidden seat
        worlds = fiber(coord)
        if len(worlds):
            seats = hidden_seats(coord)
            w = worlds[rng.randrange(len(worlds))]
            j = rng.randrange(3)
            hyp = int(w[j])
            out_h = A.outstanding(coord, hyp)
            assert np.allclose(A.role_summary(coord, hyp),
                               R.hand_features(hyp, out_h, coord.decl_id))
            checked += 1
    assert checked > 20
