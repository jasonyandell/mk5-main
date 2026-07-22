"""atlas gate C6 — the symmetry arrow, at the API level.

Transporting a coordinate through the pips-2<->3 arrow (tiles + decl +
led-suit) must COMMUTE with transition for decls 2 and 3 (the twos/threes
games the arrow is a game isomorphism of, gate R6):

    transport(transition(c, t)) == transition(transport(c), arrow_tile[t])

and transport is an involution on every coordinate.  The public surface is
also smoke-checked: the whole atlas API imports and round-trips.
"""
import random

import numpy as np

from forge.zeb.game import (
    apply_action,
    current_player,
    is_terminal,
    legal_actions,
    new_game,
)

import atlas
from atlas import from_engine, get_algebra, transition, transport
from atlas.coordinate import CoordinateV1


def test_c6_involution_everywhere():
    rng = random.Random(0)
    for g in range(80):
        st = new_game(90_000 + g)
        r = random.Random(g)
        viewer = r.randint(0, 3)
        while not is_terminal(st):
            c = from_engine(st, viewer)
            assert transport(transport(c)) == c
            st = apply_action(st, r.choice(legal_actions(st)))


def test_c6_transport_commutes_with_transition():
    at = get_algebra().arrow_tile
    checked = 0
    seed = 90_000
    while checked < 400 and seed < 96_000:
        seed += 1
        st = new_game(seed)
        if st.decl_id not in (2, 3):    # arrow is a game iso only here
            continue
        r = random.Random(seed)
        viewer = r.randint(0, 3)
        while not is_terminal(st):
            before = from_engine(st, viewer)
            actor = current_player(st)
            slot = r.choice(legal_actions(st))
            dom = st.hands[actor][slot]
            after = transition(before, dom)

            lhs = transport(after)
            rhs = transition(transport(before), int(at[dom]))
            assert lhs == rhs, (seed, dom)
            assert lhs.address() == rhs.address()
            # transport genuinely swaps the declaration
            assert transport(before).decl_id == (3 if before.decl_id == 2 else 2)
            checked += 1
            st = apply_action(st, slot)
    assert checked > 300


def test_public_surface_smoke():
    st = new_game(2)
    actor = current_player(st)
    c = atlas.from_engine(st, actor)
    assert atlas.CoordinateV1.unpack(c.pack()) == c
    assert isinstance(atlas.legal(c), int)
    assert atlas.fiber(c).shape[1] == 3
    assert atlas.role_summary(c).shape == (len(atlas.FEATURE_NAMES),)
    assert set(atlas.__all__) <= set(dir(atlas))
    # a couple of round-trips through transition + narrate via the top module
    dom = st.hands[actor][legal_actions(st)[0]]
    after = atlas.transition(c, dom)
    assert any(e.kind == "play" for e in atlas.narrate(c, after))
