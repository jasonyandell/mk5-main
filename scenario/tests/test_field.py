"""The field net. Its features read only the mover's information; its
forward pass in the core is the referee's to the integer; a Walt whose
field is the net decides and analyzes identically in both; the spec names
it; and the trainer learns from a pour, deterministically, into a net that
survives the disk."""
import math

import numpy as np
import pytest

from forge.zeb.game import is_terminal, legal_actions

from arena.walt_play import _declared_trump, parse_walt_spec
from scenario.atlas import deal, pour
from scenario.compiled import CompiledWalt
from scenario.field import CANON, N_FEATURES, FieldNet, features
from scenario.field_train import dataset, train
from scenario.planner import Config, Rung, Tilt, Walt, situation_from_state
from scenario.record import remaining_hand
from scenario.rules import rules_for
from scenario.tests.test_compiled import _same, _same_analysis, _tilt
from scenario.tests.test_rules_record import random_playing_states


def random_net(seed: int, h1: int = 24, h2: int = 12) -> FieldNet:
    rng = np.random.default_rng(seed)

    def ints(*shape, scale=2000):
        return rng.integers(-scale, scale, size=shape, dtype=np.int64)

    return FieldNet(ints(N_FEATURES, h1), ints(h1, scale=4000), ints(h2, h1), ints(h2),
                    ints(28, h2), ints(28), 4, 9)


def _pip_states(n: int, seed: int, min_plays: int = 0, unforced: bool = True):
    out = []
    for s in random_playing_states(200, seed=seed):
        if (not is_terminal(s) and s.decl_id in CANON and len(s.played) >= min_plays
                and (not unforced or len(legal_actions(s)) > 1)):
            out.append(s)
        if len(out) == n:
            break
    assert len(out) == n
    return out


def _req(state, bid=30):
    sit = situation_from_state(state, bid)
    rec = sit.record
    return (state.decl_id, sit.viewer, remaining_hand(state, sit.viewer), rec.played, rec.leader,
            list(rec.trick), rec.banked, rec.forbid, bid, sit.bidder_team, list(sit.plays))


def test_canon_sends_trump_to_six_and_keeps_the_set():
    assert CANON[6] == tuple(range(28))
    for d, c in CANON.items():
        assert sorted(c) == list(range(28))
        trumps = [t for t in range(28) if rules_for(d).led_suit[t] == 7]
        assert sorted(c[t] for t in trumps) == list(range(21, 28))  # the 6-suit: 6-0 .. 6-6


def test_features_are_distinct_indices_in_range():
    """(The signature is the information guarantee: a modeled seat's features
    are a function of its own hand, the public record, the bid and its side.)"""
    for state in _pip_states(30, seed=3):
        sit = situation_from_state(state, 30)
        f = features(rules_for(state.decl_id), sit.viewer, remaining_hand(state, sit.viewer),
                     sit.record, 30, sit.bidder_team)
        assert len(set(f)) == len(f) and all(0 <= i < N_FEATURES for i in f)


def test_core_forward_matches_the_referee_to_the_integer():
    net = random_net(1)
    fast = CompiledWalt(Config((Rung(4, None),), field=net))
    states = _pip_states(300, seed=5, unforced=False)
    for d in {s.decl_id for s in states}:
        fast._rules(d)
    got = fast._core.field_scores([_req(s, 30 + i % 4) for i, s in enumerate(states)])
    tiles = set()
    for i, (state, (tile, z)) in enumerate(zip(states, got)):
        bid = 30 + i % 4
        sit = situation_from_state(state, bid)
        rules = rules_for(state.decl_id)
        own = remaining_hand(state, sit.viewer)
        assert z == net.scores(features(rules, sit.viewer, own, sit.record, bid, sit.bidder_team)).tolist()
        assert tile == net.move(rules, sit.viewer, own, sit.record, bid, sit.bidder_team)
        tiles.add(tile)
    assert len(tiles) > 10


def test_ties_go_to_the_lowest_tile():
    z = np.zeros(28, dtype=np.int64)
    net = FieldNet(np.zeros((N_FEATURES, 4), np.int64), np.zeros(4, np.int64), np.zeros((2, 4), np.int64),
                   np.zeros(2, np.int64), np.zeros((28, 2), np.int64), z, 0, 0)
    fast = CompiledWalt(Config((Rung(4, None),), field=net))
    for state in _pip_states(20, seed=7):
        sit = situation_from_state(state, 30)
        rules = rules_for(state.decl_id)
        own = remaining_hand(state, sit.viewer)
        lowest = min(t for t in range(28) if (rules.legal(own, sit.record.led) >> t) & 1)
        assert net.move(rules, sit.viewer, own, sit.record, 30, sit.bidder_team) == lowest
        fast._rules(state.decl_id)
        assert fast._core.field_scores([_req(state)])[0][0] == lowest


NET = random_net(2)
FIELD_CONFIGS = [
    Config((Rung(12, None),), "make", seed=1, field=NET),                    # the field is the net
    Config((Rung(16, None, floor=6),), "points", seed=2, field=NET),         # ... and takes the floor
    Config((Rung(8, 5, floor=4),), "make", seed=3, tilt=Tilt("dice"), field=NET),  # frontier playouts stay dice
    Config((Rung(4, 4), Rung(10, None, floor=4)), "points", seed=4, field=NET),   # the net under rung 0
]


def test_compiled_matches_the_planner_with_a_field_net():
    for cfg in FIELD_CONFIGS:
        ref, fast = Walt(cfg), CompiledWalt(cfg)
        for state in _pip_states(8, seed=11 + cfg.seed, min_plays=4):
            _same(ref.decide(state, 30), fast.decide(state, 30))
            assert ref.nodes == fast.nodes


def test_compiled_analysis_matches_the_planner_with_a_field_net():
    for cfg in FIELD_CONFIGS[1:]:
        ref, fast = Walt(cfg, belief=_tilt), CompiledWalt(cfg, belief=_tilt)
        for state in _pip_states(5, seed=31 + cfg.seed, min_plays=6):
            _same_analysis(ref.analyze(state, 31), fast.analyze(state, 31))
            assert ref.nodes == fast.nodes


def test_the_net_changes_the_field():
    """With the net in the field the live seat plans against a different
    field than the dice: its action values move."""
    cfg = Config((Rung(12, None),), "points", seed=1)
    moved = 0
    for state in _pip_states(6, seed=13, min_plays=4):
        _, q0 = CompiledWalt(cfg).decide(state, 30)
        _, q1 = CompiledWalt(Config(cfg.rungs, "points", 1, field=NET)).decide(state, 30)
        moved += q0 != q1
    assert moved >= 4


def test_floor_at_rung_zero_needs_a_field_net():
    with pytest.raises(ValueError):
        Config((Rung(8, None, floor=4),))
    Config((Rung(8, None, floor=4),), field=NET)


def test_a_field_net_refuses_non_pip_declarations():
    state = next(s for s in random_playing_states(200, seed=9)
                 if not is_terminal(s) and s.decl_id not in CANON and len(legal_actions(s)) > 1)
    with pytest.raises(ValueError):
        CompiledWalt(Config((Rung(4, None),), field=NET)).decide(state, 30)
    with pytest.raises(ValueError):
        Walt(Config((Rung(4, None),), field=NET)).decide(state, 30)


def test_spec_names_the_field_net(tmp_path):
    path = tmp_path / "net.npz"
    NET.save(path)
    p = parse_walt_spec(f"k0,n8,F={path}")
    c = p.walt.config
    assert c.level == 0 and c.rungs[0].floor == 16 and c.field is not None
    assert np.array_equal(c.field.w1, NET.w1) and (c.field.s1, c.field.s2) == (NET.s1, NET.s2)
    assert parse_walt_spec(f"k0,n8,f=4,F={path}").walt.config.rungs[0].floor == 4
    assert parse_walt_spec("k0,n8").walt.config.field is None
    assert parse_walt_spec(f"k1,n8,m2,F={path},py").walt.config.rungs == (Rung(2, 4), Rung(8, None, 16))
    assert "field=net" in repr(p)


# -- training ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_pour():
    walt = parse_walt_spec("k0,n8,u=points").walt
    return pour([deal(s) for s in range(1, 9)], 0, 30, _declared_trump, walt)


def test_dataset_is_the_movers_view(small_pour):
    data = dataset(small_pour, tau=2.0)
    assert data.x.shape == (len(small_pour), N_FEATURES)
    assert np.allclose(data.target.sum(axis=1), 1.0)
    assert not (data.target[~data.legal] > 0).any()
    for i, pos in enumerate(small_pour.positions()):
        canon = CANON[pos.decl_id]
        best = max(pos.actions, key=lambda t: (pos.sign * pos.q[pos.actions.index(t)], -t))
        assert data.target[i, canon[best]] == data.target[i].max()


def test_a_make_pour_trains_in_points(tmp_path):
    """The target is read off the margin splat, so a pour planned under `make`
    (q in [0, 1]) still gives sharp targets in points — not the flat ones a
    probability divided by two points would."""
    walt = parse_walt_spec("k0,n8").walt
    assert walt.config.utility == "make"
    p = pour([deal(s) for s in range(1, 5)], 0, 30, _declared_trump, walt)
    data = dataset(p, tau=2.0)
    on = np.where(data.legal, data.target, 0.0)
    assert np.abs(data.adv).max() > 2.0 and on.max(axis=1).max() > 0.9


def test_training_is_deterministic_and_the_net_round_trips(small_pour, tmp_path):
    a = train(small_pour, epochs=3, seed=7, hidden=(16, 8))
    b = train(small_pour, epochs=3, seed=7, hidden=(16, 8))
    assert a.losses == b.losses and a.losses[-1] < a.losses[0]
    for k in ("w1", "b1", "w2", "b2", "w3", "b3"):
        assert np.array_equal(getattr(a.net, k), getattr(b.net, k))
    a.net.save(tmp_path / "f.npz")
    back = FieldNet.load(tmp_path / "f.npz")
    cfg = Config((Rung(8, None),), "points", seed=1, field=back)
    for state in _pip_states(3, seed=17, min_plays=8):
        _same(Walt(cfg).decide(state, 30), CompiledWalt(cfg).decide(state, 30))
    assert 0.0 <= a.metrics["val"]["agree"] <= 1.0 and not math.isnan(a.metrics["val"]["regret"])
