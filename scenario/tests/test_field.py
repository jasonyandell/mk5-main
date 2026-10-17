# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: MIT

"""The field net. Its features read only the mover's information; its
forward pass in the core is the referee's to the integer; a Walt whose
field is the net decides and analyzes identically in both; the spec names
it; and the trainer learns from a pour, deterministically, into a net that
survives the disk — on either target, in numpy or on the GPU."""
import math

import numpy as np
import pytest
import scenario_core

from forge.zeb.game import is_terminal, legal_actions

from arena.walt_play import _declared_trump, parse_walt_spec
from scenario.atlas import deal, in_plan_units, pour
from scenario.compiled import CompiledWalt
from pathlib import Path
from scenario.field import BIDS, CANON, FieldNet, LAYOUTS, N_FEATURES, N_FEATURES_OF, OFFSET, canon_for, features
from scenario.field_train import _forward, _loss_dlogits, dataset, dequantize, int_scores, regret, targets, train
from scenario.planner import Config, Rung, Walt, situation_from_state
from scenario.record import remaining_hand
from scenario.rules import rules_for
from scenario.tests.test_compiled import _belief, _same, _same_analysis
from scenario.tests.test_rules_record import random_playing_states


def random_net(seed: int, h1: int = 24, h2: int = 12, layout: str = "field-v2") -> FieldNet:
    """A random integer net. The v2 arrays are drawn first, so a v3 net of the
    same seed is the v2 net plus three fresh kind rows."""
    rng = np.random.default_rng(seed)

    def ints(*shape, scale=2000):
        return rng.integers(-scale, scale, size=shape, dtype=np.int64)

    n2 = N_FEATURES_OF["field-v2"]
    w1, b1, w2, b2, w3, b3 = ints(n2, h1), ints(h1, scale=4000), ints(h2, h1), ints(h2), ints(28, h2), ints(28)
    if layout == "field-v3":
        w1 = np.concatenate([w1, ints(N_FEATURES_OF[layout] - n2, h1)])
    return FieldNet(w1, b1, w2, b2, w3, b3, 4, 9, layout)


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
            list(rec.trick), rec.banked, rec.forbid, bid, sit.bidder, list(sit.plays))


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


def test_the_bid_is_one_hot_and_a_mark_bid_needs_all_42():
    state = _pip_states(1, seed=3)[0]
    sit = situation_from_state(state, 30)
    rules, own = rules_for(state.decl_id), remaining_hand(state, sit.viewer)

    def bid_rows(bid):
        f = features(rules, sit.viewer, own, sit.record, bid, sit.bidder_team)
        return [i - OFFSET["bid"] for i in f if OFFSET["bid"] <= i < OFFSET["kind"]]

    assert [bid_rows(b) for b in BIDS] == [[i] for i in range(len(BIDS))]
    assert bid_rows(84) == bid_rows(42) and OFFSET["bid"] + len(BIDS) == OFFSET["kind"] == N_FEATURES_OF["field-v2"]


def test_core_forward_matches_the_referee_to_the_integer():
    net = random_net(1)
    fast = CompiledWalt(Config((Rung(4, None),), field=net))
    states = _pip_states(300, seed=5, unforced=False)
    bids = [*BIDS, 84]
    got = fast._core.field_scores([_req(s, bids[i % len(bids)]) for i, s in enumerate(states)])
    tiles = set()
    for i, (state, (tile, z)) in enumerate(zip(states, got)):
        bid = bids[i % len(bids)]
        sit = situation_from_state(state, bid)
        rules = rules_for(state.decl_id)
        own = remaining_hand(state, sit.viewer)
        assert z == net.scores(features(rules, sit.viewer, own, sit.record, bid, sit.bidder_team, net.layout)).tolist()
        assert tile == net.move(rules, sit.viewer, own, sit.record, bid, sit.bidder_team)
        tiles.add(tile)
    assert len(tiles) > 10


def test_dequantize_is_the_net_up_to_its_floors():
    """The warm start computes the integer net's scores, each shift a division."""
    net = FieldNet.from_bytes(open("zeb/net/margin_d250k_h128x64.fnet", "rb").read())
    rows = []
    for state in _pip_states(60, seed=23, unforced=False):
        sit = situation_from_state(state, 30)
        rows.append(features(rules_for(state.decl_id), sit.viewer, remaining_hand(state, sit.viewer),
                             sit.record, 30, sit.bidder_team, net.layout))
    x = np.zeros((len(rows), net.n_features), np.uint8)
    for i, f in enumerate(rows):
        x[i, f] = 1
    z_int = int_scores(net, x).astype(np.float64)
    z_float = _forward(dequantize(net), x.astype(np.float32))[2].astype(np.float64)
    assert np.abs(z_float - z_int).max() <= 1e-2 * np.abs(z_int).max()   # the floors, about 0.2%


def test_batched_integer_scores_are_the_referees():
    net = random_net(3)
    rows = []
    for state in _pip_states(40, seed=19, unforced=False):
        sit = situation_from_state(state, 30)
        rows.append(features(rules_for(state.decl_id), sit.viewer, remaining_hand(state, sit.viewer),
                             sit.record, 30, sit.bidder_team, net.layout))
    x = np.zeros((len(rows), net.n_features), np.uint8)
    for i, f in enumerate(rows):
        x[i, f] = 1
    assert np.array_equal(int_scores(net, x), np.array([net.scores(f) for f in rows]))


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
        assert fast._core.field_scores([_req(state)])[0][0] == lowest


NET = random_net(2)
FIELD_CONFIGS = [
    Config((Rung(12, None),), "make", seed=1, field=NET),                    # the field is the net
    Config((Rung(16, None, floor=6),), "points", seed=2, field=NET),         # ... and takes the floor
    Config((Rung(8, 5, floor=4),), "make", seed=3, field=NET),               # frontier playouts stay dice
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
        ref, fast = Walt(cfg, belief=_belief), CompiledWalt(cfg, belief=_belief)
        for state in _pip_states(5, seed=31 + cfg.seed, min_plays=6):
            _same_analysis(ref.analyze(state, 31), fast.analyze(state, 31))
            assert ref.nodes == fast.nodes


LATE_CONFIGS = [  # the floor guards only sampled cells
    Config((Rung(512, None, 16),), "make", seed=8, field=NET),   # the shipped shape
    Config((Rung(4, 4), Rung(160, None, 16)), "points", seed=9, field=NET),
]


def test_compiled_matches_the_planner_where_the_fiber_is_enumerated(monkeypatch):
    """Late positions, enumerated and sampled alike (16 plays down is 1,680
    worlds, sampled at n=512; from about 18 the fiber fits and the floor
    stands down): the rule is the same in both implementations, and it
    changes a plan at every config (against a referee told every fiber is a
    sample, which floors as before the rule)."""
    import scenario.planner as planner
    states = {cfg.seed: _pip_states(8, seed=51 + cfg.seed, min_plays=16) for cfg in LATE_CONFIGS}
    plans = {}
    for cfg in LATE_CONFIGS:
        ref, fast = Walt(cfg), CompiledWalt(cfg)
        for i, state in enumerate(states[cfg.seed]):
            a = ref.analyze(state, 31)
            _same_analysis(a, fast.analyze(state, 31))
            assert ref.nodes == fast.nodes
            _same((a.tile, a.q), fast.decide(state, 31))
            assert ref.nodes == fast.nodes
            plans[cfg.seed, i] = a.q
    sample_worlds = planner.sample_worlds
    monkeypatch.setattr(planner, "sample_worlds", lambda *args: (sample_worlds(*args)[0], False))
    for cfg in LATE_CONFIGS:
        assert any(Walt(cfg).analyze(state, 31).q != plans[cfg.seed, i]
                   for i, state in enumerate(states[cfg.seed])), cfg


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
    data = dataset(small_pour)
    target = targets(data, "margin", 2.0)
    assert data.x.shape == (len(small_pour), N_FEATURES)
    assert np.allclose(target.sum(axis=1), 1.0)
    assert not (target[~data.legal] > 0).any() and not (data.win[~data.legal] > 0).any()
    for i, pos in enumerate(small_pour.positions()):
        canon = CANON[pos.decl_id]
        best = max(pos.actions, key=lambda t: (pos.sign * pos.q[pos.actions.index(t)], -t))
        assert target[i, canon[best]] == target[i].max()
        slots = [canon[t] for t in pos.actions]
        assert np.allclose(data.q[i, slots], pos.sign * pos.q, atol=1e-5) and not data.q[i, ~data.legal[i]].any()


def test_several_pours_train_as_one(small_pour, make_pour):
    """A chunked pour is several directories: their rows concatenate in
    order and the second's deals are numbered after the first's, so the
    held-out tenth is still whole deals. Pours of different utilities do not
    mix: their q are in different units."""
    a, both = dataset(small_pour), dataset(small_pour, small_pour)
    assert len(both.x) == 2 * len(a.x)
    assert (both.x[len(a.x):] == a.x).all() and (both.q[len(a.x):] == a.q).all()
    assert (both.deal[:len(a.x)] == a.deal).all()
    assert (both.deal[len(a.x):] == a.deal + len(small_pour.hands)).all()
    assert len(set(both.deal.tolist())) == 2 * len(small_pour.hands)
    with pytest.raises(ValueError, match="utility"):
        dataset(small_pour, make_pour)


def test_regret_is_what_the_pick_gives_up_against_the_best_legal_move():
    """The best move costs nothing; any move of an exact tie for best costs
    nothing (which of them the teacher played is its tie rule, not play);
    a worse move costs the gap; an illegal slot's value never counts as best."""
    value = np.array([[0.2, 0.7, 0.5, 0.0], [0.4, 0.4, 0.1, 0.9], [0.3, 0.3, 0.0, 0.0]])
    legal = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 0, 0]], bool)
    assert np.allclose(regret(value, legal, np.array([1, 0, 0])), 0)        # the best; a tie, either member
    assert regret(value, legal, np.array([1, 1, 1]))[1:].tolist() == [0, 0]  # the tie's other member
    assert np.allclose(regret(value, legal, np.array([0, 2, 1])), [0.5, 0.3, 0])


def test_the_grades_are_regrets_under_the_splats_q(make_pour):
    """`regret_q` is the mean, over decisions, of the best legal move's
    sign·q minus the net's move's, recomputed here from the positions
    themselves; `at_best` is the share where that is 0, ties included."""
    data = dataset(make_pour)
    t = train(data, "pmake", epochs=2, seed=5, hidden=(16, 8))
    g = t.metrics["train"]
    rows = np.flatnonzero(data.deal % 10 != 0)
    scores = int_scores(t.net, data.x[rows])
    gaps = []
    for r, i in enumerate(rows):
        pos = make_pour.position(int(i))
        canon = CANON[pos.decl_id]
        pick = max(pos.actions, key=lambda a: (scores[r, canon[a]], -canon[a]))   # `_pick`: first slot of a tie
        v = (pos.sign * pos.q).astype(np.float32)
        gaps.append(float(v.max() - v[pos.actions.index(pick)]))
    assert g["positions"] == len(rows)
    assert g["regret_q"] == pytest.approx(np.mean(gaps), abs=1e-6)
    assert g["at_best"] == pytest.approx(np.mean(np.array(gaps) == 0))


def test_a_make_pour_trains_in_points(tmp_path):
    """The target is read off the margin splat, so a pour planned under `make`
    (q in [0, 1]) still gives sharp targets in points — not the flat ones a
    probability divided by two points would."""
    walt = parse_walt_spec("k0,n8").walt
    assert walt.config.utility == "make"
    p = pour([deal(s) for s in range(1, 5)], 0, 30, _declared_trump, walt)
    data = dataset(p)
    on = np.where(data.legal, targets(data, "margin", 2.0), 0.0)
    assert np.abs(data.adv).max() > 2.0 and on.max(axis=1).max() > 0.9


@pytest.fixture(scope="module")
def make_pour():
    walt = parse_walt_spec("k0,n8").walt
    return pour([deal(s) for s in range(1, 7)], 0, 30, _declared_trump, walt)


def test_the_pmake_target_is_the_movers_success(make_pour):
    """Under `make` the plan's q is the make probability, so the pmake
    target read off the margin splat is q for the bidding side and 1 − q
    for a defender (who wins by setting); the targets are sharp — some
    position's moves range from hopeless to sure."""
    data = dataset(make_pour)
    assert make_pour.utility == "make"
    spread = 0.0
    for i, pos in enumerate(make_pour.positions()):
        slots = [CANON[pos.decl_id][t] for t in pos.actions]
        q = pos.weights @ in_plan_units(pos, pos.outcomes).T / pos.weights.sum()
        assert np.allclose(pos.q, q)
        assert np.allclose(data.win[i, slots], q if pos.sign > 0 else 1 - q, atol=1e-6)
        spread = max(spread, float(np.ptp(data.win[i, slots])))
    assert spread > 0.9


def test_the_bce_sees_only_the_legal_tiles():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((6, 28)).astype(np.float32) * 3
    legal = rng.random((6, 28)) < 0.3
    legal[:, 0] = True
    target = np.where(legal, rng.random((6, 28)), 0).astype(np.float32)
    loss, d = _loss_dlogits(logits, legal, target, "pmake")
    moved = np.where(legal, logits, logits + 50)
    assert _loss_dlogits(moved, legal, target, "pmake")[0] == pytest.approx(loss)
    assert not d[~legal].any()
    p = 1 / (1 + np.exp(-logits.astype(np.float64)))
    want = -(target * np.log(p) + (1 - target) * np.log(1 - p))[legal].sum() / len(logits)
    assert loss == pytest.approx(want, rel=1e-5)
    assert np.allclose(d[legal], ((p - target) / len(logits))[legal], atol=1e-6)


def test_training_is_deterministic_and_the_net_round_trips(small_pour, tmp_path):
    data = dataset(small_pour)
    a = train(data, epochs=3, seed=7, hidden=(16, 8))
    b = train(data, epochs=3, seed=7, hidden=(16, 8))
    assert a.losses == b.losses and a.losses[-1] < a.losses[0]
    for k in ("w1", "b1", "w2", "b2", "w3", "b3"):
        assert np.array_equal(getattr(a.net, k), getattr(b.net, k))
    a.net.save(tmp_path / "f.npz")
    back = FieldNet.load(tmp_path / "f.npz")
    cfg = Config((Rung(8, None),), "points", seed=1, field=back)
    for state in _pip_states(3, seed=17, min_plays=8):
        _same(Walt(cfg).decide(state, 30), CompiledWalt(cfg).decide(state, 30))
    assert 0.0 <= a.metrics["val"]["at_best"] <= 1.0 and not math.isnan(a.metrics["val"]["regret_q"])


@pytest.mark.parametrize("target", ["margin", "pmake"])
def test_mlx_trains_the_numpy_net(make_pour, target):
    """The GPU path is the same float net: the same init, batches, loss and
    Adam step, so the losses and weights agree to float32 rounding (Adam
    amplifies it only where a gradient is rounding noise) and the nets pick
    the same moves."""
    data = dataset(make_pour)
    a = train(data, target, epochs=4, seed=5, hidden=(24, 12))
    b = train(data, target, epochs=4, seed=5, hidden=(24, 12), device="mlx")
    assert a.losses[-1] < a.losses[0]
    assert np.allclose(a.losses, b.losses, rtol=2e-3)
    for pa, pb in zip(a.params, b.params):
        assert np.abs(pa - pb).mean() < 1e-2 * np.abs(pa).mean()
    assert b.metrics["train"]["at_best"] == pytest.approx(a.metrics["train"]["at_best"], abs=0.03)


def test_an_old_net_is_refused(tmp_path):
    np.savez(tmp_path / "old.npz", layout="field-v1")
    with pytest.raises(ValueError, match="layout"):
        FieldNet.load(tmp_path / "old.npz")


def test_the_blob_round_trips_and_refuses_what_it_did_not_write():
    net = random_net(5)
    net.w1[0, 0], net.b2[0] = -(1 << 15), 1 << 40      # int16's floor; a bias far past int32
    blob = net.to_bytes()
    back = FieldNet.from_bytes(blob)
    for k in ("w1", "b1", "w2", "b2", "w3", "b3", "s1", "s2"):
        assert np.array_equal(getattr(back, k), getattr(net, k)), k
    assert back.to_bytes() == blob
    for bad in (b"XNET" + blob[4:], blob.replace(b"field-v2", b"field-v1"), blob[:-1], blob + b"\0", blob[:7]):
        with pytest.raises(ValueError):
            FieldNet.from_bytes(bad)
    net.w3[0, 0] = 1 << 15
    with pytest.raises(ValueError, match="int16"):
        net.to_bytes()


def test_the_core_canon_is_the_referees():
    for layout in LAYOUTS:
        for d in range(10):
            c = scenario_core.canon(layout, d)
            assert (None if c is None else tuple(c)) == canon_for(layout, d), (layout, d)
    with pytest.raises(ValueError):
        scenario_core.canon("field-v9", 0)


def _states_of(decls, n: int, seed: int):
    out = []
    for s in random_playing_states(400, seed=seed):
        if not is_terminal(s) and s.decl_id in decls and len(legal_actions(s)) > 1:
            out.append(s)
        if len(out) == n:
            break
    assert len(out) == n
    return out


def test_field_v3_plays_doubles_and_no_trump_with_the_kind_bit():
    """A v3 row is the v2 row plus one kind bit; under doubles trump and no
    trump the tiles are not relabelled, and a v2 net refuses those games."""
    assert N_FEATURES_OF["field-v2"] + 3 == N_FEATURES_OF["field-v3"] == N_FEATURES
    for state in _states_of({4, 7, 9}, 30, seed=11):
        sit = situation_from_state(state, 30)
        rules, own = rules_for(state.decl_id), remaining_hand(state, sit.viewer)
        v3 = features(rules, sit.viewer, own, sit.record, 30, sit.bidder_team, "field-v3")
        kind = [i for i in v3 if i >= OFFSET["kind"]]
        assert kind == [OFFSET["kind"] + {4: 0, 7: 1, 9: 2}[state.decl_id]]
        if state.decl_id == 4:
            v2 = features(rules, sit.viewer, own, sit.record, 30, sit.bidder_team, "field-v2")
            assert v2 == v3[:-1]
        else:
            assert canon_for("field-v3", state.decl_id) == tuple(range(28))
            assert canon_for("field-v2", state.decl_id) is None
            with pytest.raises(ValueError, match="pip-trump declarations only"):
                features(rules, sit.viewer, own, sit.record, 30, sit.bidder_team, "field-v2")
    v2net, v3net = random_net(3), random_net(3, layout="field-v3")
    assert v3net.layout == "field-v3" and v3net.plays(7) and v3net.plays(9) and not v3net.plays(8)
    assert v2net.plays(4) and not v2net.plays(7) and not v2net.plays(9)
    assert np.array_equal(v3net.w1[:N_FEATURES_OF["field-v2"]], v2net.w1), "a v3 random net is the v2 one plus kind rows"
    assert FieldNet.from_bytes(v2net.to_bytes()).layout == "field-v2"
    assert FieldNet.from_bytes(v3net.to_bytes()).layout == "field-v3"
    with pytest.raises(ValueError, match="layout"):
        FieldNet.from_bytes(b"FNET\x08field-v9" + v3net.to_bytes()[13:])


def test_core_forward_matches_the_referee_under_doubles_and_no_trump():
    net = random_net(4, layout="field-v3")
    fast = CompiledWalt(Config((Rung(4, None),), field=net))
    states = _states_of({7, 9}, 60, seed=12) + _states_of({0, 3, 6}, 60, seed=13)
    got = fast._core.field_scores([_req(s) for s in states])
    for state, (tile, z) in zip(states, got):
        sit = situation_from_state(state, 30)
        rules, own = rules_for(state.decl_id), remaining_hand(state, sit.viewer)
        assert z == net.scores(features(rules, sit.viewer, own, sit.record, 30, sit.bidder_team, net.layout)).tolist()
        assert tile == net.move(rules, sit.viewer, own, sit.record, 30, sit.bidder_team)


def test_the_shipped_v2_net_still_refuses_doubles_and_no_trump():
    v2 = FieldNet.from_bytes(Path("zeb/net/margin_d250k_h128x64.fnet").read_bytes())
    assert v2.layout == "field-v2" and v2.n_features == N_FEATURES_OF["field-v2"]
    state = _states_of({7, 9}, 1, seed=14)[0]
    with pytest.raises(ValueError, match="pip-trump declarations only"):
        CompiledWalt(Config((Rung(4, None),), field=v2)).decide(state, 30)
    with pytest.raises(ValueError, match="pip-trump declarations only"):
        Walt(Config((Rung(4, None),), field=v2)).decide(state, 30)


def test_checkpoints_come_from_the_one_run_and_warm_means_warm(make_pour):
    """`on_checkpoint` sees the integer net at the asked epochs (0 is the initial net) of the
    same run: the last checkpoint is the returned net, and checkpointing leaves the run
    unchanged. `warm` is true only when the caller passed `init`."""
    data = dataset(make_pour)
    seen = {}
    t = train(data, "pmake", epochs=3, seed=5, hidden=(16, 8), checkpoints=(0, 1, 3),
              on_checkpoint=lambda e, n: seen.__setitem__(e, n))
    assert sorted(seen) == [0, 1, 3]
    assert np.array_equal(seen[3].w1, t.net.w1) and np.array_equal(seen[3].b3, t.net.b3)
    plain = train(data, "pmake", epochs=3, seed=5, hidden=(16, 8))
    assert np.array_equal(plain.net.w1, t.net.w1) and plain.losses == t.losses
    assert not np.array_equal(seen[0].w1, seen[3].w1)
    assert t.metrics["warm"] is False
    assert train(data, "pmake", epochs=1, seed=5, hidden=(16, 8), init=dequantize(t.net)).metrics["warm"] is True
    with pytest.raises(ValueError):
        train(data, "pmake", epochs=1, seed=5, hidden=(16, 8), checkpoints=(1,))
