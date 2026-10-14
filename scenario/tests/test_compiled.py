"""The compiled core is the Python planner, bit for bit: same tile, same
action values, same analyses, on random states, across the ladder's
configurations, with and without a belief, batched or one at a time."""
import math
import random

import pytest
import scenario_core

from forge.zeb.game import is_terminal, legal_actions

from arena.walt_play import parse_walt_spec
from scenario.compiled import CompiledWalt
from scenario.planner import Config, Rung, Walt
from scenario.rules import rules_for
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


def test_the_core_tables_are_the_rules_authority():
    """The core derives its own rules tables; they are `forge.oracle.tables`'s,
    entry for entry, under all ten declarations."""
    for d in range(10):
        r = rules_for(d)
        led_suit, follow_bits, rank, count = scenario_core.rules(d)     # byte tables arrive as bytes
        assert list(led_suit) == list(r.led_suit), d
        assert follow_bits == list(r.follow_bits), d
        assert [list(row) for row in rank] == [list(row) for row in r.rank], d
        assert list(count) == list(r.count), d


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


def _belief(sit, worlds):
    """A deterministic non-uniform belief: a function of the worlds alone."""
    return [1.0 + ((w[(sit.viewer + 1) % 4] * 2654435761) % 97) / 50.0 for w in worlds]


def test_compiled_matches_the_planner_with_a_belief():
    for cfg in CONFIGS[:3]:
        ref, fast = Walt(cfg, belief=_belief), CompiledWalt(cfg, belief=_belief)
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


def test_compiled_matches_the_planner_with_the_auction():
    from arena.walt_play import declaration_belief
    cfgs = [
        Config((Rung(4, 4), Rung(10, None)), "make", seed=6),
        Config((Rung(12, None),), "points", seed=8),
    ]
    auction = declaration_belief(0.1)
    for cfg in cfgs:
        ref, fast = Walt(cfg, belief=auction), CompiledWalt(cfg, belief=auction)
        for state in _states(10, seed=41 + cfg.seed, min_plays=4):
            _same(ref.decide(state, 30), fast.decide(state, 30))
            assert ref.nodes == fast.nodes


def test_compiled_enumerates_the_endgame_exactly_as_the_planner():
    cfg = Config((Rung(8, None), Rung(160, None)), "make", seed=12)
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
    assert parse_walt_spec("k1,n8,m2,b=auction,e=0.2").walt.belief is not None
    assert parse_walt_spec("k1,n8,m2").walt.belief is None
    with pytest.raises(ValueError):
        parse_walt_spec("k1,n8,m2,b=field")


ANALYSIS_CASES = [
    (Config((Rung(12, None),), "make", seed=1), None),
    (Config((Rung(6, 5),), "points", seed=2), None),
    (Config((Rung(4, 4), Rung(10, 6)), "make", seed=3), None),
    (Config((Rung(4, 4), Rung(10, 6)), "points", seed=13), _belief),
    (Config((Rung(3, 3), Rung(3, 3), Rung(6, None)), "make", seed=4), _belief),
]


def _same_analysis(a, b):
    _same((a.tile, a.q), (b.tile, b.q))
    assert a.worlds == b.worlds
    assert a.weights == b.weights
    assert a.outcomes == b.outcomes
    assert a.line == b.line


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


def _trick_one_state(seed: int, plays: int):
    """A trick-1 state with `plays` tiles already down and a real choice."""
    for s in random_playing_states(80, seed=seed):
        if len(s.played) == plays and len(legal_actions(s)) > 1:
            return s
    raise AssertionError("no such state")


# The Python referee costs minutes at n160 from trick one (37 s for one state),
# so the floor cases run at n=FLOOR_N: still two rungs, and the same three
# floor regimes (a small floor, a mid floor, a floor at the full world count,
# where every cell is floored) at the same depths. Floors scale with n.
FLOOR_N = 40
FLOOR_CASES = [  # (floor, utility, plays already down) at k1,n40,m8
    (4, "points", 0),
    (16, "make", 2),
    (FLOOR_N, "points", 3),
    (4, "make", 12),
    (16, "points", 14),
    (FLOOR_N, "make", 13),
]


def test_compiled_matches_the_planner_under_a_cell_floor():
    for floor, utility, plays in FLOOR_CASES:
        cfg = Config((Rung(8, 4), Rung(FLOOR_N, None, floor)), utility, seed=17)
        state = _states(1, seed=71, min_plays=plays)[0] if plays >= 4 else _trick_one_state(73, plays)
        ref, fast = Walt(cfg), CompiledWalt(cfg)
        a = ref.analyze(state, 31)
        _same_analysis(a, fast.analyze(state, 31))
        assert ref.nodes == fast.nodes
        _same((a.tile, a.q), fast.decide(state, 31))
        assert ref.nodes == fast.nodes


def test_compiled_matches_the_planner_on_the_champion_ladder():
    """The champion's shape (`k1,n512,m8`, floor 16) at a cheaper n: the
    rung-0 minds and the live rung share each coordinate's worlds, in both
    implementations alike."""
    ref = parse_walt_spec("k1,n48,m8,py").walt
    fast = parse_walt_spec("k1,n48,m8").walt
    assert ref.config == fast.config and ref.config.level == 1
    for state in _states(5, seed=91, min_plays=4):
        a = ref.analyze(state, 30)
        _same_analysis(a, fast.analyze(state, 30))
        assert ref.nodes == fast.nodes


def test_compiled_line_matches_the_planner():
    """The plan's line (`analyze(line=True)`) is the referee's, step for step:
    the same viewer cells in the same order, the same tiles, the same
    observations, the same floored moves — and asking for it moves nothing
    else: tile, values, worlds, weights, outcomes and nodes are the plain
    analysis's, in both implementations."""
    cases = ANALYSIS_CASES + [(Config((Rung(8, 4), Rung(FLOOR_N, None, f)), u, seed=17), None)
                              for f, u, _ in FLOOR_CASES[:3]]
    floored = 0
    for cfg, belief in cases:
        ref, fast = Walt(cfg, belief=belief), CompiledWalt(cfg, belief=belief)
        for state in _states(3, seed=61 + cfg.seed, min_plays=6):
            plain = fast.analyze(state, 31)
            nodes = fast.nodes
            a, b = ref.analyze(state, 31, line=True), fast.analyze(state, 31, line=True)
            assert plain.line is None and fast.nodes == ref.nodes == nodes
            assert (b.tile, b.q, b.worlds, b.weights, b.outcomes) == \
                (plain.tile, plain.q, plain.worlds, plain.weights, plain.outcomes)
            _same_analysis(a, b)
            assert a.line == b.line and a.line.keys() == a.q.keys()
            floored += sum(s.floored for steps in a.line.values() for s in steps)
    assert floored > 0, "some line should cross a floored cell"
