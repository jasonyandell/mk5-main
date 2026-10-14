"""The tie rule ≺ (#120): `lowest` takes the lowest tile id among a
decision's exact ties; `margin` (make, then margin) takes the tie with the
best weighted bidder-team margin for the decider's side under the plan, then
the lowest id. Referee = core to the bit under `margin` (tile, q, nodes,
analyses, batch); the margin tile is the one an independent reading of the
analysis's outcomes names; a one-rung player's q is untouched, so the rule
moves the tile only inside the tied set; the rule does fire. The band δ
(`Config.tie_band`) widens margin's tied set to the near ties: δ = 0 is the
exact rule, and a band's pick is bit-identical across engines and is the one
the analysis names within δ of the best q."""
import pytest
import scenario_core

from arena.walt_play import parse_walt_spec
from scenario.compiled import CompiledWalt
from scenario.convention import Convention
from scenario.planner import Config, Rung, Walt
from scenario.tests.test_compiled import _same, _same_analysis, _states
from scenario.tests.test_field import NET, _pip_states


def _with(cfg: Config, **kw) -> Config:
    return Config(**{**cfg.__dict__, **kw})


MARGIN = [
    Config((Rung(12, None),), "make", seed=1, tie="margin"),                    # rung 0, dice field
    Config((Rung(8, 5),), "make", seed=2, tie="margin"),                        # frontier playouts
    Config((Rung(4, 4), Rung(10, None)), "make", seed=3, tie="margin"),         # modeled rung 0 breaks its ties too
    Config((Rung(3, 3), Rung(3, 3), Rung(6, 6)), "make", seed=4, tie="margin"),  # three rungs
]
FIELD_MARGIN = [
    Config((Rung(16, None, floor=6),), "make", seed=5, field=NET, tie="margin"),          # the shipped shape, smaller
    Config((Rung(4, 4), Rung(10, None, floor=4)), "make", seed=6, field=NET, tie="margin"),
]


def _belief(sit, worlds):
    return [1.0 + ((w[(sit.viewer + 1) % 4] * 2654435761) % 97) / 50.0 for w in worlds]


def test_core_matches_the_referee_under_make_then_margin():
    moved = 0   # the rule is not idle on these states: parity is not vacuous
    for cfg in MARGIN:
        ref, fast, low = Walt(cfg), CompiledWalt(cfg), CompiledWalt(_with(cfg, tie="lowest"))
        for state in _states(6, seed=71 + cfg.seed, min_plays=10):
            bid = 30 + state.decl_id % 4
            want = ref.decide(state, bid)
            _same(want, fast.decide(state, bid))
            assert ref.nodes == fast.nodes
            _same_analysis(ref.analyze(state, bid), fast.analyze(state, bid))
            moved += want[0] != low.decide(state, bid)[0]
    assert moved > 0


def test_core_matches_the_referee_with_a_field_net_a_belief_and_a_batch():
    moved = 0
    for cfg in FIELD_MARGIN:
        ref, fast = Walt(cfg, belief=_belief), CompiledWalt(cfg, belief=_belief)
        low = CompiledWalt(_with(cfg, tie="lowest"), belief=_belief)
        states = _pip_states(5, seed=81 + cfg.seed, min_plays=10)
        batch = fast.decide_batch(states, [31] * len(states))
        for state, got in zip(states, batch):
            want = ref.decide(state, 31)
            _same(want, got)
            _same(want, fast.decide(state, 31))
            _same_analysis(ref.analyze(state, 31, line=True), fast.analyze(state, 31, line=True))
            moved += want[0] != low.decide(state, 31)[0]
    assert moved > 0


def test_a_convention_sender_breaks_its_ties_by_margin_in_both_engines():
    cfg = Config((Rung(16, None, floor=6),), "make", seed=7, field=NET, tie="margin",
                 convention=Convention("biggest", send=True, read=False))
    ref, fast = Walt(cfg), CompiledWalt(cfg)
    for state in _pip_states(12, seed=91, min_plays=1):
        _same(ref.decide(state, 30), fast.decide(state, 30))
        assert ref.sent == fast.sent


def _margin_pick(a, maximize: bool) -> int:
    """The rule read off an analysis: among the exact ties of q, the best
    Σ_r w_r·m_r, then the lowest tile."""
    top = (max if maximize else min)(a.q.values())
    tied = sorted(t for t, v in a.q.items() if v == top)
    m = {t: sum(w * x for w, x in zip(a.weights, a.outcomes[t])) for t in tied}
    best = (max if maximize else min)(m.values())
    return min(t for t in tied if m[t] == best)


def test_the_margin_tile_is_the_analysis_reading_and_q_is_unchanged():
    """One rung: no modeled Walt decides, so q is the lowest rule's; the
    margin tile is a tie of q and the one the outcomes name. The rule
    moves the tile at some of these positions (it is not idle)."""
    moved = 0
    for base in (Config((Rung(12, None),), "make", seed=1),
                 Config((Rung(16, None, floor=6),), "make", seed=5, field=NET)):
        low, mar = CompiledWalt(base), CompiledWalt(_with(base, tie="margin"))
        states = _pip_states(40, seed=101 + base.seed, min_plays=8)
        for state in states:
            bid = 30 + state.decl_id % 5
            a, b = low.analyze(state, bid), mar.analyze(state, bid)
            assert a.q == b.q and a.outcomes == b.outcomes
            seat = (state.trick_leader + len(state.current_trick)) % 4
            assert b.tile == _margin_pick(b, maximize=seat % 2 == state.bidder % 2)
            tied = sorted(t for t, v in a.q.items() if v == a.q[a.tile])
            assert a.tile == tied[0] and b.tile in tied
            moved += a.tile != b.tile
    assert moved > 0


def test_lowest_is_the_default_and_margin_is_make_only():
    assert Config((Rung(4, None),)).tie == "lowest"
    with pytest.raises(ValueError):
        Config((Rung(4, None),), "points", tie="margin")
    with pytest.raises(ValueError):
        Config((Rung(4, None),), tie="coin")
    with pytest.raises(ValueError):
        scenario_core.Core([(4, None, 1)], "points", 0, tie="margin")
    with pytest.raises(ValueError):
        scenario_core.Core([(4, None, 1)], "make", 0, tie="coin")
    assert parse_walt_spec("k0,n8,t=margin").walt.config.tie == "margin"
    assert parse_walt_spec("k0,n8").walt.config.tie == "lowest"
    with pytest.raises(ValueError):
        parse_walt_spec("k0,n8,u=points,t=margin")


BAND = [
    Config((Rung(12, None),), "make", seed=1, tie="margin", tie_band=0.05),
    Config((Rung(4, 4), Rung(10, None)), "make", seed=3, tie="margin", tie_band=0.1),   # modeled rung's band too
    Config((Rung(16, None, floor=6),), "make", seed=5, field=NET, tie="margin", tie_band=0.07),
]


def test_core_matches_the_referee_within_a_band():
    moved = 0   # the band is not idle against the exact rule: parity is not vacuous
    for cfg in BAND:
        ref, fast = Walt(cfg, belief=_belief), CompiledWalt(cfg, belief=_belief)
        exact = CompiledWalt(_with(cfg, tie_band=0.0), belief=_belief)
        states = _pip_states(6, seed=111 + cfg.seed, min_plays=8)
        batch = fast.decide_batch(states, [31] * len(states))
        for state, got in zip(states, batch):
            want = ref.decide(state, 31)
            _same(want, got)
            _same(want, fast.decide(state, 31))
            assert ref.nodes == fast.nodes
            _same_analysis(ref.analyze(state, 31), fast.analyze(state, 31))
            moved += want[0] != exact.decide(state, 31)[0]
    assert moved > 0


def _band_pick(a, maximize: bool, band: float) -> int:
    """The band read off an analysis: among the q within δ of the best, the
    best Σ_r w_r·m_r, then the lowest tile."""
    top = (max if maximize else min)(a.q.values())
    tied = sorted(t for t, v in a.q.items() if (v >= top - band if maximize else v <= top + band))
    m = {t: sum(w * x for w, x in zip(a.weights, a.outcomes[t])) for t in tied}
    best = (max if maximize else min)(m.values())
    return min(t for t in tied if m[t] == best)


def test_a_band_picks_the_analysis_reading_and_zero_is_the_exact_rule():
    """One rung: q is the lowest rule's under any band. The band's tile is
    the one the outcomes name within δ, and δ = 0 decides as `t=margin`."""
    moved = 0
    for base, band in ((Config((Rung(12, None),), "make", seed=1, tie="margin"), 0.1),
                       (Config((Rung(16, None, floor=6),), "make", seed=5, field=NET, tie="margin"), 0.2)):
        exact, zero = CompiledWalt(base), Walt(_with(base, tie_band=0.0))
        wide = CompiledWalt(_with(base, tie_band=band))
        for state in _pip_states(40, seed=101 + base.seed, min_plays=8):
            bid = 30 + state.decl_id % 5
            a, b = exact.analyze(state, bid), wide.analyze(state, bid)
            assert a.q == b.q and a.outcomes == b.outcomes
            seat = (state.trick_leader + len(state.current_trick)) % 4
            assert b.tile == _band_pick(b, seat % 2 == state.bidder % 2, band)
            assert a.tile == _band_pick(a, seat % 2 == state.bidder % 2, 0.0) == zero.decide(state, bid)[0]
            moved += a.tile != b.tile
    assert moved > 0


def test_a_band_is_margin_only_and_parses():
    assert Config((Rung(4, None),)).tie_band == 0.0
    for bad in (dict(tie_band=0.01), dict(tie="margin", tie_band=-0.01), dict(tie="margin", tie_band=1.5)):
        with pytest.raises(ValueError):
            Config((Rung(4, None),), "make", **bad)
    for tie, band in (("lowest", 0.01), ("margin", -0.01), ("margin", 1.5)):
        with pytest.raises(ValueError):
            scenario_core.Core([(4, None, 1)], "make", 0, tie=tie, tie_band=band)
    c = parse_walt_spec("k0,n8,t=margin:0.01").walt.config
    assert (c.tie, c.tie_band) == ("margin", 0.01)
    assert parse_walt_spec("k0,n8,t=margin").walt.config.tie_band == 0.0
    with pytest.raises(ValueError):
        parse_walt_spec("k0,n8,t=lowest:0.01")
