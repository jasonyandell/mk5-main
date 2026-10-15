# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The fallible tracker (#151, `Config.void_miss` = p): at each invocation
each hidden seat's shown voids are forgotten with probability p, by a coin
on `_mix(seed, VOID, seat)`, and the worlds are drawn from what is
remembered. p = 0 is deployed Zeb byte for byte; the coin is the same
integer in the referee and the core, so the two agree at p > 0; at p = 1
a seat that showed out of a suit may hold it in a drawn world."""

from scenario.compiled import CompiledWalt
from scenario.planner import Config, Reading, Rung, Walt, missed_voids, situation_from_state
from scenario.tests.test_compiled import _same, _same_analysis
from scenario.tests.test_field import NET, _pip_states
from scenario.tests.test_reading import MARGIN


def _with(cfg: Config, **kw) -> Config:
    return Config(**{**cfg.__dict__, **kw})


def _voided(n: int, seed: int):
    """n states where some hidden seat has shown a void."""
    out = []
    for state in _pip_states(60, seed=seed, min_plays=8):
        rec = situation_from_state(state, 30).record
        viewer = situation_from_state(state, 30).viewer
        if any(f for s, f in enumerate(rec.forbid) if s != viewer):
            out.append(state)
        if len(out) == n:
            break
    assert len(out) == n
    return out


VOID_CONFIGS = [
    Config((Rung(12, None),), "make", seed=1, field=NET, void_miss=0.5),
    Config((Rung(16, None, floor=6),), "points", seed=2, field=NET, void_miss=0.2),
    Config((Rung(16, None, 6),), "make", seed=3, field=NET, draw="posterior", draw_budget=20_000,
           reading=Reading("uniform", MARGIN, 0.28), void_miss=0.3),     # deployed Zeb's draw, smaller n
    Config((Rung(4, 4), Rung(10, None, floor=4)), "points", seed=4, field=NET, void_miss=0.5),  # under rung 0
]


def test_zero_void_miss_is_deployed_zeb():
    """The default is p = 0, and a p so small that the coin is drawn yet never
    fires (⌈p·2^53⌉ = 1) plans exactly as p = 0, in both engines."""
    base = Config((Rung(16, None, 6),), "make", seed=3, field=NET, draw="posterior", draw_budget=20_000,
                  reading=Reading("uniform", MARGIN, 0.28))
    assert base.void_miss == 0.0 and base.void_coin == 0
    tiny = _with(base, void_miss=1e-30)
    assert tiny.void_coin == 1
    for state in _voided(4, seed=31):
        a = Walt(base).analyze(state, 30)
        _same_analysis(a, Walt(tiny).analyze(state, 30))
        _same_analysis(a, CompiledWalt(tiny).analyze(state, 30))
        _same_analysis(a, CompiledWalt(base).analyze(state, 30))


def test_compiled_matches_the_planner_when_voids_are_missed():
    """The coin is the same integer in both engines: the referee and the core
    draw the same worlds and plan the same at p > 0, uniform and posterior."""
    moved = 0
    for cfg in VOID_CONFIGS:
        ref, fast = Walt(cfg), CompiledWalt(cfg)
        for state in _voided(4, seed=41 + cfg.seed):
            a = ref.analyze(state, 31)
            _same_analysis(a, fast.analyze(state, 31))
            assert ref.nodes == fast.nodes
            _same((a.tile, a.q), fast.decide(state, 31))
            moved += a.worlds != Walt(_with(cfg, void_miss=0.0)).analyze(state, 31).worlds
    assert moved >= 4  # forgetting reaches the worlds


def test_a_forgotten_void_puts_the_suit_back():
    """At p = 1 every hidden seat's voids are forgotten: some drawn world gives
    a seat a tile it was proved not to hold; at p = 0 none does."""
    cfg = Config((Rung(64, None),), "make", seed=9, field=NET)
    found = 0
    for state in _voided(6, seed=53):
        sit = situation_from_state(state, 30)
        forbid = sit.record.forbid
        def breaks(worlds):
            return any(w[s] & forbid[s] for w in worlds for s in range(4) if s != sit.viewer)
        assert not breaks(Walt(cfg).analyze(state, 30).worlds)
        found += breaks(CompiledWalt(_with(cfg, void_miss=1.0)).analyze(state, 30).worlds)
    assert found >= 3


def test_the_coin_forgets_each_seat_with_probability_p():
    """Over many seeds, each hidden seat's mask is cleared about p of the time,
    the viewer's never, and p = 0 clears nothing."""
    forbid = (0b1010, 0b0110, 0b1100, 0b0011)
    coin = Config((Rung(4, None),), void_miss=0.25).void_coin
    cleared = [0, 0, 0, 0]
    for seed in range(4000):
        kept = missed_voids(seed, 1, forbid, coin)
        assert kept[1] == forbid[1]
        assert missed_voids(seed, 1, forbid, 0) == forbid
        for s in range(4):
            cleared[s] += kept[s] == 0
    assert cleared[1] == 0
    for s in (0, 2, 3):
        assert 0.22 < cleared[s] / 4000 < 0.28, cleared
