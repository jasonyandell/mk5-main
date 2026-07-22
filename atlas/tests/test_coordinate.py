"""atlas coordinate gates C1 (pack/unpack) and C2 (engine parity).

C1: round-trip identity, version-byte enforcement, cross-process address
stability, 10k fuzz. C2: 500+ random full games through forge.zeb.game; at
every play, for all four viewers, atlas.transition from the observed action
equals from_engine of the engine's next state (field-exact), and legal
equals the engine's legal set for the viewer to act.
"""
import random
import subprocess
import sys

import numpy as np

from forge.zeb.game import (
    apply_action,
    current_player,
    is_terminal,
    legal_actions,
    new_game,
)

from atlas.algebra import N_DECLS
from atlas.coordinate import (
    CoordinateV0Auction,
    CoordinateV1,
    VERSION,
    _PACK_SIZE,
    from_engine,
    legal,
    transition,
)


def _random_coord(rng: random.Random) -> CoordinateV1:
    n_ct = rng.randint(0, 3)
    return CoordinateV1(
        version=VERSION,
        decl_id=rng.randint(0, N_DECLS - 1),
        viewer=rng.randint(0, 3),
        viewer_hand=rng.getrandbits(28),
        played=tuple(rng.getrandbits(28) for _ in range(4)),
        trick_leader=rng.randint(0, 3),
        current_trick=tuple(rng.sample(range(28), n_ct)),
        team_points=(rng.randint(0, 42), rng.randint(0, 42)),
        bid_value=rng.randint(0, 42),
        bidder=rng.randint(0, 3),
        dealer=rng.randint(0, 3),
        voids=tuple(rng.getrandbits(8) for _ in range(4)),
    )


def test_c1_pack_roundtrip_fuzz():
    rng = random.Random(0)
    for _ in range(10_000):
        c = _random_coord(rng)
        data = c.pack()
        assert len(data) == _PACK_SIZE
        assert CoordinateV1.unpack(data) == c
        assert CoordinateV1.unpack(data).address() == c.address()


def test_c1_version_byte_enforced():
    c = _random_coord(random.Random(1))
    bad = bytes([2]) + c.pack()[1:]
    try:
        CoordinateV1.unpack(bad)
    except ValueError as e:
        assert "version" in str(e)
    else:
        raise AssertionError("bad version byte was not rejected")
    for short in (b"", c.pack()[:-1], c.pack() + b"\x00"):
        try:
            CoordinateV1.unpack(short)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong-length packed bytes were not rejected")


def test_c1_address_stable_across_processes():
    c = _random_coord(random.Random(7))
    data = c.pack()
    prog = (
        "import sys; from atlas.coordinate import CoordinateV1; "
        "d=bytes.fromhex(sys.argv[1]); "
        "print(CoordinateV1.unpack(d).address().hex())"
    )
    out = subprocess.run(
        [sys.executable, "-c", prog, data.hex()],
        capture_output=True, text=True, cwd=".", check=True,
    )
    assert out.stdout.strip() == c.address().hex()


def test_c1_auction_stub_roundtrip():
    rng = random.Random(3)
    for _ in range(2000):
        c = CoordinateV0Auction(
            version=VERSION, viewer=rng.randint(0, 3),
            viewer_hand=rng.getrandbits(28), dealer=rng.randint(0, 3),
            bids=tuple(rng.randint(0, 42) for _ in range(rng.randint(0, 7))),
        )
        assert CoordinateV0Auction.unpack(c.pack()) == c


def test_special_contract_raises():
    st = new_game(0)
    bad = st.__class__(**{**st.__dict__, "decl_id": 12})
    try:
        from_engine(bad, 0)
    except NotImplementedError as e:
        assert "special contracts" in str(e)
    else:
        raise AssertionError("out-of-range decl_id did not raise")


def _legal_mask(state) -> int:
    actor = current_player(state)
    hand = state.hands[actor]
    m = 0
    for slot in legal_actions(state):
        m |= 1 << hand[slot]
    return m


def test_c2_engine_parity():
    n_games = 550
    steps = 0
    legal_checks = 0
    decls_seen = set()
    for g in range(n_games):
        st = new_game(1_000 + g)
        rng = random.Random((g << 8) ^ 0xA7)
        decls_seen.add(st.decl_id)
        while not is_terminal(st):
            coords = {v: from_engine(st, v) for v in range(4)}
            actor = current_player(st)

            # legal parity for the viewer who is to act
            assert legal(coords[actor]) == _legal_mask(st), (g, actor)
            # a hidden seat's legality is not answerable from the coordinate
            for v in range(4):
                if v != actor:
                    try:
                        legal(coords[v])
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("legal() answered for a non-actor")
            legal_checks += 1

            slot = rng.choice(legal_actions(st))
            dom = st.hands[actor][slot]
            st2 = apply_action(st, slot)

            for v in range(4):
                got = transition(coords[v], dom)
                want = from_engine(st2, v)
                assert got == want, (g, v, dom)
                assert got.address() == want.address()
            st = st2
            steps += 1

    assert steps > 500
    assert legal_checks > 500
    # skip_bidding samples all 10 decls incl. 8 (doubles-suit, non-game)
    assert decls_seen == set(range(N_DECLS))


def test_c2_team_points_declaring_first():
    """team_points[0] is always the declaring (bidder's) team's banked points."""
    for g in range(60):
        st = new_game(5_000 + g)
        rng = random.Random(g)
        while not is_terminal(st):
            st = apply_action(st, rng.choice(legal_actions(st)))
        c = from_engine(st, 0)
        bt = st.bidder % 2
        assert c.team_points == (st.team_points[bt], st.team_points[1 - bt])
        assert sum(c.team_points) == sum(st.team_points)
