"""The compiled core is the Python planner, bit for bit: same tile, same
action values, same analyses, on random states, across the ladder's
configurations, with and without a belief, batched or one at a time."""
import math
import random

from forge.zeb.game import is_terminal, legal_actions

from arena.walt_play import parse_walt_spec
from scenario.compiled import CompiledWalt
from scenario.planner import Config, Rung, Tilt, Walt
from scenario.tests.test_rules_record import random_playing_states


def _states(n: int, seed: int, min_plays: int = 0):
    out = []
    for s in random_playing_states(80, seed=seed):
        if not is_terminal(s) and len(s.played) >= min_plays and len(legal_actions(s)) > 1:
            out.append(s)
        if len(out) == n:
            break
    assert len(out) == n
    return out


CONFIGS = [
    Config((Rung(12, None),), "make", seed=1),           # rung 0, full horizon
    Config((Rung(6, 5),), "points", seed=2),             # rung 0, frontier playouts, points
    Config((Rung(4, 4), Rung(10, None)), "make", seed=3),  # the two-response form
    Config((Rung(3, 3), Rung(3, 3), Rung(6, 6)), "make", seed=4),  # three rungs
]


def _same(a, b):
    ta, qa = a
    tb, qb = b
    assert ta == tb
    assert qa.keys() == qb.keys()
    for t in qa:
        assert qa[t] == qb[t] or (math.isnan(qa[t]) and math.isnan(qb[t])), (t, qa[t], qb[t])


def test_compiled_matches_the_planner_without_a_belief():
    for cfg in CONFIGS:
        ref, fast = Walt(cfg), CompiledWalt(cfg)
        for state in _states(12, seed=11 + cfg.seed, min_plays=6):
            bid = 30 + (state.decl_id % 3)
            _same(ref.decide(state, bid), fast.decide(state, bid))
            assert ref.nodes == fast.nodes


def test_compiled_matches_the_planner_at_trick_one():
    cfg = Config((Rung(2, 3), Rung(8, None)), "make", seed=9)
    ref, fast = Walt(cfg), CompiledWalt(cfg)
    for state in _states(4, seed=99):
        if len(state.played) > 1:
            continue
        _same(ref.decide(state, 30), fast.decide(state, 30))


def _tilt(sit, worlds):
    """A deterministic non-uniform belief: a function of the worlds alone."""
    return [1.0 + ((w[(sit.viewer + 1) % 4] * 2654435761) % 97) / 50.0 for w in worlds]


def test_compiled_matches_the_planner_with_a_belief():
    for cfg in CONFIGS[:3]:
        ref, fast = Walt(cfg, belief=_tilt), CompiledWalt(cfg, belief=_tilt)
        for state in _states(8, seed=21 + cfg.seed, min_plays=8):
            _same(ref.decide(state, 31), fast.decide(state, 31))


def test_batch_equals_one_at_a_time_and_handles_forced_moves():
    cfg = Config((Rung(3, 3), Rung(8, 8)), "make", seed=5)
    fast = CompiledWalt(cfg)
    states = []
    for s in random_playing_states(40, seed=31):
        if not is_terminal(s):
            states.append(s)
        if len(states) == 24:
            break
    bids = [30 + i % 5 for i in range(len(states))]
    batch = fast.decide_batch(states, bids)
    for state, bid, got in zip(states, bids, batch):
        _same(fast.decide(state, bid), got)
        if len(legal_actions(state)) == 1:
            assert math.isnan(got[1][got[0]])


def test_compiled_matches_the_planner_with_the_tilt_and_the_auction():
    from arena.walt_play import declaration_belief
    cfgs = [
        Config((Rung(4, 4), Rung(10, None)), "make", seed=6, tilt=Tilt("dice")),
        Config((Rung(4, 4), Rung(10, None)), "make", seed=7, tilt=Tilt("field", eps=0.2)),
        Config((Rung(12, None),), "points", seed=8, tilt=Tilt("dice")),
    ]
    auction = declaration_belief(0.1)
    for cfg in cfgs:
        ref, fast = Walt(cfg, belief=auction), CompiledWalt(cfg, belief=auction)
        for state in _states(10, seed=41 + cfg.seed, min_plays=4):
            _same(ref.decide(state, 30), fast.decide(state, 30))
            assert ref.nodes == fast.nodes


def test_compiled_enumerates_the_endgame_exactly_as_the_planner():
    cfg = Config((Rung(8, None), Rung(160, None)), "make", seed=12, tilt=Tilt("dice"))
    ref, fast = Walt(cfg), CompiledWalt(cfg)
    n = 0
    for state in random_playing_states(80, seed=55):
        if not is_terminal(state) and len(state.played) >= 18 and len(legal_actions(state)) > 1:
            _same(ref.decide(state, 30), fast.decide(state, 30))
            n += 1
            if n == 8:
                break
    assert n == 8


def test_spec_selects_the_engine():
    assert type(parse_walt_spec("k0,n8").walt).__name__ == "CompiledWalt"
    assert type(parse_walt_spec("k0,n8,py").walt).__name__ == "Walt"
    p = parse_walt_spec("k1,n8,m2,b=auction+field,e=0.2")
    assert p.walt.config.tilt == Tilt("field", 0.2) and p.walt.belief is not None
    assert parse_walt_spec("k0,n8,b=dice").walt.belief is None


ANALYSIS_CASES = [
    (Config((Rung(12, None),), "make", seed=1), None),
    (Config((Rung(6, 5),), "points", seed=2, tilt=Tilt("dice")), None),
    (Config((Rung(4, 4), Rung(10, 6)), "make", seed=3, tilt=Tilt("field", eps=0.2)), None),
    (Config((Rung(4, 4), Rung(10, 6)), "points", seed=13, tilt=Tilt("dice")), _tilt),
    (Config((Rung(3, 3), Rung(3, 3), Rung(6, None)), "make", seed=4), _tilt),
]


def _same_analysis(a, b):
    _same((a.tile, a.q), (b.tile, b.q))
    assert a.worlds == b.worlds
    assert a.weights == b.weights
    assert a.outcomes == b.outcomes


def test_compiled_analysis_matches_the_planner():
    for cfg, belief in ANALYSIS_CASES:
        ref, fast = Walt(cfg, belief=belief), CompiledWalt(cfg, belief=belief)
        for state in _states(6, seed=61 + cfg.seed, min_plays=6):
            a, b = ref.analyze(state, 31), fast.analyze(state, 31)
            _same_analysis(a, b)
            assert ref.nodes == fast.nodes
            assert all(type(w) is tuple for w in b.worlds)


def test_analysis_batch_equals_one_at_a_time_and_handles_forced_moves():
    cfg, belief = ANALYSIS_CASES[3]
    fast = CompiledWalt(cfg, belief=belief)
    states = []
    for s in random_playing_states(40, seed=37):
        if not is_terminal(s) and len(s.played) >= 4:
            states.append(s)
        if len(states) == 16:
            break
    assert any(len(legal_actions(s)) == 1 for s in states)
    bids = [30 + i % 5 for i in range(len(states))]
    for state, bid, got in zip(states, bids, fast.analyze_batch(states, bids)):
        _same_analysis(fast.analyze(state, bid), got)
        if len(legal_actions(state)) == 1:
            assert got.worlds == [] and got.outcomes == {got.tile: ()} and math.isnan(got.q[got.tile])
