"""atlas fiber gates C3.

C3a: at trick boundaries, atlas.fiber equals walt.worlds.enumerate_worlds on
the equivalent root (built from the same engine state), as a set of rows.
C3b: mid-trick fibers equal a brute-force filter written here (walt's public
enumerator targets trick-boundary roots; the brute force is the independent
authority mid-trick), including decl 8 (doubles-suit) which atlas handles.
C3c: the monotone-shrink law fiber(transition(c,t)) subset fiber(c).
"""
import random
from itertools import combinations

import numpy as np

from forge.zeb.game import (
    apply_action,
    current_player,
    is_terminal,
    legal_actions,
    new_game,
)
from walt.contracts import EndgameRoot
from walt.worlds import enumerate_worlds

from atlas.coordinate import from_engine, transition
from atlas.fiber import fiber, hidden_seats, unknown_tiles, _forbidden


def _rows(arr: np.ndarray) -> set:
    return {tuple(int(x) for x in row) for row in arr}


def _play_to_boundary(seed: int, min_tricks: int, max_tricks: int):
    """Random play to a trick boundary with the given number of completed
    tricks; returns the engine state (never terminal)."""
    st = new_game(seed)
    rng = random.Random((seed << 3) ^ 0x11)
    target = rng.randint(min_tricks, max_tricks)
    while len(st.play_history) < target * 4:
        st = apply_action(st, rng.choice(legal_actions(st)))
    return st


def _root_from_state(st, me: int) -> EndgameRoot:
    rem = tuple(sorted(d for d in st.hands[me] if d not in st.played))
    return EndgameRoot(
        decl_id=st.decl_id, bidder=st.bidder, bid_value=st.bid_state.high_bid,
        bids=st.bid_state.bids, dealer=st.dealer, me=me, my_hand=rem,
        play_history=st.play_history, trick_leader=st.trick_leader,
        current_trick=st.current_trick, team_points=st.team_points)


def test_c3a_walt_parity_at_boundaries():
    checked = 0
    for g in range(400):
        # boundaries 4-6 tricks in: <= 9 hidden tiles, the regime walt runs in
        st = _play_to_boundary(20_000 + g, 4, 6)
        if st.decl_id == 8:      # doubles-suit is non-game; walt lane skips it
            continue
        me = current_player(st)
        want = _rows(enumerate_worlds(_root_from_state(st, me)))
        got = _rows(fiber(from_engine(st, me)))
        assert got == want, (g, me, len(got), len(want))
        checked += 1
    assert checked > 200


def _brute_fiber(coord) -> set:
    seats = hidden_seats(coord)
    unknown = [int(t) for t in unknown_tiles(coord)]
    counts = [7 - bin(int(coord.played[s])).count("1") for s in seats]
    forb = [int(_forbidden(coord, s)) for s in seats]

    def ok(mask, f):
        return (mask & f) == 0

    def m(tiles):
        v = 0
        for t in tiles:
            v |= 1 << t
        return v

    out = set()
    for c0 in combinations(unknown, counts[0]):
        m0 = m(c0)
        if not ok(m0, forb[0]):
            continue
        rem1 = [t for t in unknown if t not in c0]
        for c1 in combinations(rem1, counts[1]):
            m1 = m(c1)
            if not ok(m1, forb[1]):
                continue
            c2 = [t for t in rem1 if t not in c1]
            m2 = m(c2)
            if not ok(m2, forb[2]):
                continue
            out.add((m0, m1, m2))
    return out


def test_c3b_midtrick_bruteforce():
    checked = 0
    decls = set()
    for g in range(2000):
        st = _play_to_boundary(30_000 + g, 4, 6)
        rng = random.Random(g)
        # step into a partial trick (1..3 tiles down) to make it mid-trick
        n_into = rng.randint(1, 3)
        ok = True
        for _ in range(n_into):
            if is_terminal(st) or not st.current_trick and len(st.play_history) and \
                    len(st.play_history) % 4 == 0 and n_into == 0:
                ok = False
                break
            legal = legal_actions(st)
            st = apply_action(st, rng.choice(legal))
        if is_terminal(st) or not st.current_trick:
            continue
        viewer = rng.randint(0, 3)
        coord = from_engine(st, viewer)
        if len(unknown_tiles(coord)) > 10:   # keep brute force cheap
            continue
        got = _rows(fiber(coord))
        want = _brute_fiber(coord)
        assert got == want, (g, viewer, len(got), len(want))
        decls.add(st.decl_id)
        checked += 1
        if checked >= 250 and 8 in decls:
            break
    assert checked > 150
    assert 8 in decls, "wanted at least one doubles-suit mid-trick fiber"


def test_c3c_monotone_shrink():
    checked = 0
    for g in range(120):
        st = new_game(40_000 + g)
        rng = random.Random(g ^ 0x5A)
        viewer = rng.randint(0, 3)
        while not is_terminal(st):
            parent = from_engine(st, viewer)
            actor = current_player(st)
            slot = rng.choice(legal_actions(st))
            dom = st.hands[actor][slot]

            # only enumerate in the endgame regime (bounds the combinatorics)
            if len(unknown_tiles(parent)) <= 12:
                child = transition(parent, dom)
                parent_set = _rows(fiber(parent))
                seats = hidden_seats(parent)
                if actor == viewer:
                    # viewer's own play removes nothing from the hidden fiber
                    assert _rows(fiber(child)) == parent_set
                else:
                    j = seats.index(actor)
                    for w in fiber(child):
                        ext = list(int(x) for x in w)
                        ext[j] |= 1 << dom
                        assert tuple(ext) in parent_set, (g, actor, dom)
                checked += 1
            st = apply_action(st, slot)
    assert checked > 200
