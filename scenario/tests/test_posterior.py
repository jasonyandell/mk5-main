# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: MIT

"""The posterior draw: the live rung's worlds from β^F ∝ uniform × the
likelihood that the field played what the other seats played. An enumerated
fiber is weighted by that likelihood normalized; a sampled one is n draws by
rejection from the uniform stream, equally weighted; a decision that cannot
find n within its budget plans over the uniform sample, exactly as the
uniform player does. Referee = core to the bit, proposals and fallbacks
included."""

import pytest
import scenario_core

from forge.zeb.game import legal_actions

from arena.walt_play import parse_walt_spec
from scenario.compiled import CompiledWalt
from scenario.planner import DRAW_BUDGET, Config, Rung, Walt, situation_from_state
from scenario.record import remaining_hand
from scenario.tests.test_compiled import _belief, _same, _same_analysis
from scenario.tests.test_field import NET
from scenario.tests.test_field_sigma import _net_filter, _net_played


def _with(cfg: Config, **kw) -> Config:
    return Config(**{**cfg.__dict__, **kw})


def _positions(cases):
    """Hands the net plays in every seat (the real deal is in the posterior),
    at positions where the seat to play has a choice."""
    out = [_net_played(seed, k) for seed, k in cases]
    return [s for s in out if len(legal_actions(s)) > 1]


SAMPLED = _positions([(3, 5), (6, 6), (8, 9), (11, 10), (6, 17), (9, 13), (10, 7)])
LATE = _positions([(7, 21), (11, 17), (16, 16), (13, 20), (16, 20)])   # fibers of 12 to 350

POSTERIOR_CONFIGS = [
    Config((Rung(16, None, 6),), "make", seed=3, field=NET, draw="posterior", draw_budget=4000),
    Config((Rung(12, None, 4),), "points", seed=5, field=NET, field_eps=0.2, draw="posterior", draw_budget=4000),
    Config((Rung(16, 6, 6),), "make", seed=7, field=NET, draw="posterior", draw_budget=60),  # budget runs out
]


def test_compiled_matches_the_planner_under_the_posterior():
    drawn = fell = moved = 0
    for cfg in POSTERIOR_CONFIGS:
        ref, fast = Walt(cfg), CompiledWalt(cfg)
        for state in SAMPLED + LATE:
            a = ref.analyze(state, 30)
            _same_analysis(a, fast.analyze(state, 30))
            assert (ref.nodes, ref.proposals, ref.fallbacks) == (fast.nodes, fast.proposals, fast.fallbacks)
            _same((a.tile, a.q), fast.decide(state, 30))
            assert (ref.nodes, ref.proposals, ref.fallbacks) == (fast.nodes, fast.proposals, fast.fallbacks)
            drawn += ref.proposals > 0 and not ref.fallbacks
            fell += ref.fallbacks
            moved += a.q != Walt(_with(cfg, draw="uniform")).analyze(state, 30).q
    assert drawn >= 8 and fell >= 2 and moved >= 8


def test_an_enumerated_fiber_is_weighted_by_the_likelihood():
    zeros = 0
    for eps in (0.0, 0.1):
        cfg = Config((Rung(512, None, 16),), "make", seed=1, field=NET, field_eps=eps, draw="posterior")
        for state in LATE:
            sit = situation_from_state(state, 30)
            ref = Walt(cfg)
            a = ref.analyze(state, 30)
            uniform = Walt(_with(cfg, draw="uniform")).analyze(state, 30)
            assert a.worlds == uniform.worlds and len(a.worlds) <= 512   # the whole fiber
            like = ref.field_likelihood(sit, a.worlds)
            assert a.weights == [x / sum(like) for x in like]
            assert ref.proposals == 0 and ref.fallbacks == 0
            _same_analysis(a, CompiledWalt(cfg).analyze(state, 30))
            zeros += a.weights.count(0.0)
    assert zeros  # at ε = 0 the net's filter excludes worlds outright


def test_every_drawn_world_reproduces_the_observed_moves_at_eps_zero():
    cfg = Config((Rung(64, None, 16),), "make", seed=2, field=NET, draw="posterior")
    fast = CompiledWalt(cfg)
    for state in SAMPLED:
        sit = situation_from_state(state, 30)
        a = fast.analyze(state, 30)
        assert fast.fallbacks == 0 and fast.proposals >= 64 and len(a.worlds) == 64
        assert all(_net_filter(sit, w) == 1.0 for w in a.worlds)
        assert a.weights == [1.0] * 64
        # The proposals are the uniform stream, so the uniform sample's
        # consistent worlds are the posterior's first ones, in order.
        uniform = CompiledWalt(_with(cfg, draw="uniform")).analyze(state, 30).worlds
        kept = [w for w in uniform if _net_filter(sit, w) == 1.0]
        assert len(kept) < 64 and a.worlds[:len(kept)] == kept


def test_the_likelihood_enters_once():
    """`field_likelihood` is the one likelihood of the plays, and it enters
    once: a sampled fiber is drawn from it and weighted equally (never
    reweighted by it again), an enumerated one is weighted by it normalized
    (above), and an external belief multiplies in exactly once on top.
    Non-vacuous: at ε > 0 the drawn worlds' likelihoods differ, so a second
    application would show in the weights."""
    cfg = POSTERIOR_CONFIGS[1]
    uneven = 0
    for state in SAMPLED[:4]:
        sit = situation_from_state(state, 30)
        for walt in (Walt(cfg, belief=_belief), CompiledWalt(cfg, belief=_belief)):
            a = walt.analyze(state, 30)
            assert not walt.fallbacks
            assert a.weights == _belief(sit, a.worlds)
        uneven += len(set(Walt(cfg).field_likelihood(sit, a.worlds))) > 1
    assert uneven


def test_a_spent_budget_and_the_opening_lead_plan_as_the_uniform():
    """With one proposal the posterior falls back to the uniform sample and
    plans exactly as the uniform player; before any other seat has played,
    every proposal is accepted and the first n are the uniform sample."""
    base = Config((Rung(32, None, 8),), "make", seed=4, field=NET)
    assert base.draw == "uniform" and base.draw_budget == DRAW_BUDGET
    spent = _with(base, draw="posterior", draw_budget=1)
    opening = _net_played(15, 0)
    for state in SAMPLED[:3] + [opening]:
        u = Walt(base).analyze(state, 30)
        _same_analysis(u, CompiledWalt(_with(base, draw_budget=7)).analyze(state, 30))  # unused under uniform
        for cfg in (spent, _with(base, draw="posterior")) if state is opening else (spent,):
            ref, fast = Walt(cfg), CompiledWalt(cfg)
            _same_analysis(u, ref.analyze(state, 30))
            _same_analysis(u, fast.analyze(state, 30))
            assert ref.fallbacks == fast.fallbacks == (cfg is spent)


def test_the_draw_is_validated_and_named_by_the_spec(tmp_path):
    with pytest.raises(ValueError):
        Config((Rung(8, None),), draw="posterior")                       # no net to explain the plays
    with pytest.raises(ValueError):
        Config((Rung(4, 4), Rung(8, None)), field=NET, draw="posterior")  # the one-rung player's posterior
    with pytest.raises(ValueError):
        Config((Rung(8, None),), field=NET, draw="tilted")
    with pytest.raises(ValueError):
        Config((Rung(8, None),), field=NET, draw="posterior", draw_budget=0)
    with pytest.raises(ValueError):
        scenario_core.Core([(8, None, 1)], "make", 0, None, 0.0, "posterior", 10)
    path = tmp_path / "net.npz"
    NET.save(path)
    p = parse_walt_spec(f"k0,n8,F={path},d=posterior,D=500")
    assert (p.walt.config.draw, p.walt.config.draw_budget) == ("posterior", 500) and "draw=posterior" in repr(p)
    assert parse_walt_spec(f"k0,n8,F={path}").walt.config.draw == "uniform"
    with pytest.raises(ValueError):
        parse_walt_spec("k1,n8,d=posterior")
