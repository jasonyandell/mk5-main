# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The hand-written field net: it compiles, every rule unit is a truth value,
and the compiled core plays it exactly as the referee does."""
from __future__ import annotations

import numpy as np

from lem import handnet
from lem.probe.net import forward
from scenario.compiled import CompiledWalt
from scenario.field import FieldNet, features
from scenario.planner import Config, Rung, situation_from_state
from scenario.rules import rules_for
from scenario.tests.test_field import _pip_states, _req
from scenario.record import remaining_hand


def _rows(net: FieldNet, n: int, seed: int):
    states = _pip_states(n, seed=seed, unforced=False)
    xs = np.zeros((n, net.n_features), np.uint8)
    for i, s in enumerate(states):
        sit = situation_from_state(s, 30 + i % 13)
        xs[i, features(rules_for(s.decl_id), sit.viewer, remaining_hand(s, sit.viewer), sit.record,
                       30 + i % 13, sit.bidder_team, net.layout)] = 1
    return states, xs


def test_every_rule_unit_is_a_truth_value():
    B = handnet.build()
    net = B.compile()
    _, xs = _rows(net, 400, seed=11)
    a = forward(net, xs)
    for j, u in enumerate(B.l1.values()):
        if u.kind == "bool":
            assert set(np.unique(a.a1[:, j]).tolist()) <= {0, 1}, u.name
    for j, u in enumerate(B.l2.values()):
        assert set(np.unique(a.a2[:, j]).tolist()) <= {0, 1}, u.name


def test_the_core_plays_the_hand_net_as_the_referee_does():
    net = handnet.build().compile()
    fast = CompiledWalt(Config((Rung(4, None),), field=net))
    states = _pip_states(300, seed=5, unforced=False)
    got = fast._core.field_scores([_req(s) for s in states])
    for state, (tile, z) in zip(states, got):
        sit = situation_from_state(state, 30)
        rules = rules_for(state.decl_id)
        own = remaining_hand(state, sit.viewer)
        assert z == net.scores(features(rules, sit.viewer, own, sit.record, 30, sit.bidder_team, net.layout)).tolist()
        assert tile == net.move(rules, sit.viewer, own, sit.record, 30, sit.bidder_team)


def test_the_build_is_deterministic_and_round_trips():
    a, b = handnet.build().compile(), handnet.build().compile()
    assert a.to_bytes() == b.to_bytes()
    assert FieldNet.from_bytes(a.to_bytes()).to_bytes() == a.to_bytes()
    assert a.s1 == a.s2 == 0 and a.layout == "field-v2"
