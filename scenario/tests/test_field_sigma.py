# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: MIT

"""The stochastic field F_σ: a modeled seat explores with probability ε,
playing the dice's tile, else the net's move. ε = 0 is the deterministic net
byte for byte; the coin and the dice are the same integers in both engines,
so the referee and the core agree with ε > 0; the tape realizes P_σ; and the
likelihood of what the other seats played under F_σ is a probability,
positive on the whole fiber when ε > 0 and the net's 0/1 filter at ε = 0."""

import pytest

from arena.walt_play import parse_walt_spec
from scenario.atlas import _state, deal
from scenario.compiled import CompiledWalt
from scenario.fiber import sample_worlds
from scenario.planner import Config, Rung, SplitMix, Situation, Walt, situation_from_state
from scenario.record import empty_record, remaining_hand
from scenario.rules import bit, rules_for, tiles_of
from scenario.tests.test_compiled import _same, _same_analysis
from scenario.tests.test_field import NET, _pip_states


def _with(cfg: Config, **kw) -> Config:
    return Config(**{**cfg.__dict__, **kw})


SIGMA_CONFIGS = [
    Config((Rung(12, None),), "make", seed=1, field=NET, field_eps=0.1),
    Config((Rung(16, None, floor=6),), "points", seed=2, field=NET, field_eps=0.3),     # floored moves stay the net's
    Config((Rung(4, 4), Rung(10, None, floor=4)), "points", seed=4, field=NET, field_eps=0.2),  # F_σ under rung 0
    Config((Rung(64, None, 16),), "make", seed=8, field=NET, field_eps=0.05),           # the shipped shape, smaller n
]


def test_zero_eps_is_the_deterministic_net():
    """The default is ε = 0, and an ε so small that the coin is drawn yet
    never explores (⌈ε·2^53⌉ = 1) plans exactly as the deterministic net."""
    base = Config((Rung(16, None, floor=6),), "points", seed=2, field=NET)
    assert base.field_eps == 0.0 and base.field_coin == 0
    tiny = _with(base, field_eps=1e-30)
    assert tiny.field_coin == 1
    for state in _pip_states(4, seed=21, min_plays=4):
        a = Walt(base).analyze(state, 30)
        _same_analysis(a, Walt(tiny).analyze(state, 30))
        _same_analysis(a, CompiledWalt(tiny).analyze(state, 30))


def test_eps_one_is_the_dice():
    """At ε = 1 every modeled move is the dice's own draw from the tape: a
    one-rung plan over F_σ is the plan over the random field, to the bit."""
    dice = Config((Rung(12, None),), "make", seed=5)
    sigma = _with(dice, field=NET, field_eps=1.0)
    for state in _pip_states(4, seed=23, min_plays=4):
        a = Walt(dice).analyze(state, 30)
        _same_analysis(a, Walt(sigma).analyze(state, 30))
        _same_analysis(a, CompiledWalt(sigma).analyze(state, 30))


def test_compiled_matches_the_planner_under_the_stochastic_field():
    moved = 0
    for cfg in SIGMA_CONFIGS:
        ref, fast = Walt(cfg), CompiledWalt(cfg)
        for state in _pip_states(4, seed=41 + cfg.seed, min_plays=4):
            a = ref.analyze(state, 31)
            _same_analysis(a, fast.analyze(state, 31))
            assert ref.nodes == fast.nodes
            _same((a.tile, a.q), fast.decide(state, 31))
            assert ref.nodes == fast.nodes
            moved += a.q != Walt(_with(cfg, field_eps=0.0)).analyze(state, 31).q
    assert moved >= 8  # the exploration reaches the plans


def test_the_tape_realizes_p_sigma():
    """Over many scenario tapes at one record, the modeled seat's tile has
    frequency ε/n + (1 − ε)·[the net's move] for each legal tile."""
    eps = 0.3
    cfg = Config((Rung(8, None),), "make", seed=3, field=NET, field_eps=eps)
    walt = Walt(cfg)
    state = _pip_states(1, seed=29, min_plays=5)[0]
    sit = situation_from_state(state, 30)
    rules = rules_for(state.decl_id)
    seat, hand = sit.viewer, remaining_hand(state, sit.viewer)
    legal = tiles_of(rules.legal(hand, sit.record.led))
    n = len(legal)
    assert n > 1
    viewer = (seat + 1) % 4
    inv = walt._invocation(0, viewer, remaining_hand(state, viewer), sit.record, rules, 30, sit.bidder_team)
    rng = SplitMix(99)
    trials = 40000
    inv.tapes = [rng.next_u64() for _ in range(trials)]
    counts = {t: 0 for t in legal}
    for r in range(trials):
        counts[inv._field_move(seat, r, hand, sit.record)] += 1
    m = NET.move(rules, seat, hand, sit.record, 30, sit.bidder_team)
    for t in legal:
        p = eps / n + (1 - eps) * (t == m)
        assert abs(counts[t] / trials - p) < 5 * (p * (1 - p) / trials) ** 0.5, (t, counts[t], p)


def _net_filter(sit: Situation, world) -> float:
    """1 if the deterministic net plays every observed discretionary move of
    the other seats in this world, else 0 — computed from the net directly."""
    rules = rules_for(sit.decl_id)
    hands = [world[s] | sit.played_by(s) for s in range(4)]
    rec = empty_record(sit.plays[0][0])
    for seat, tile in sit.plays:
        if seat != sit.viewer:
            legal = rules.legal(hands[seat], rec.led)
            if legal & (legal - 1) and NET.move(rules, seat, hands[seat], rec, sit.bid, sit.bidder_team) != tile:
                return 0.0
        hands[seat] &= ~bit(tile)
        rec = rec.advance(tile, rules)
    return 1.0


def _fiber(state, sit):
    rec = sit.record
    sizes = tuple(rec.hand_size(s) for s in range(4))
    own = remaining_hand(state, sit.viewer)
    worlds, exact = sample_worlds(sit.viewer, own, rec.played, sizes, rec.forbid, 4000, SplitMix(1))
    assert exact
    return worlds


def _net_played(seed: int, plays_down: int):
    """A pip-trump hand the net plays in every seat, `plays_down` tiles in:
    the real deal passes the net's filter."""
    hands = deal(seed)
    decl, bidder, bid = seed % 7, 0, 30
    rules = rules_for(decl)
    rec, plays = empty_record(bidder), ()
    while len(plays) < plays_down:
        seat = rec.seat
        tile = NET.move(rules, seat, hands[seat] & ~rec.played, rec, bid, bidder % 2)
        plays += ((seat, tile),)
        rec = rec.advance(tile, rules)
    return _state(tuple(tuple(tiles_of(h)) for h in hands), bidder, bid, decl, plays, rec)


def test_the_likelihood_is_positive_on_the_fiber_and_the_filter_at_zero():
    zeros = ones = 0
    for state in _pip_states(2, seed=33, min_plays=16) + [_net_played(s, 17) for s in (3, 4)]:
        sit = situation_from_state(state, 30)
        worlds = _fiber(state, sit)
        real = tuple(remaining_hand(state, s) for s in range(4))
        assert real in worlds
        lik = Walt(Config((Rung(8, None),), field=NET, field_eps=0.1)).field_likelihood(sit, worlds)
        assert all(x > 0 for x in lik)
        flt = Walt(Config((Rung(8, None),), field=NET)).field_likelihood(sit, worlds)
        assert flt == [_net_filter(sit, w) for w in worlds]
        zeros += flt.count(0.0)
        ones += flt.count(1.0)
    assert zeros and ones


def test_the_likelihood_is_a_probability_over_the_next_move():
    """Extend the record by each legal tile of the seat to play, as seen by
    another seat: the likelihoods' increments P_σ(t) sum to 1, and match
    ε̂/n + (1 − ε̂)·[the net's move]; under the random field they are 1/n."""
    state = _pip_states(1, seed=37, min_plays=6)[0]
    sit = situation_from_state(state, 30)
    rules = rules_for(state.decl_id)
    mover = sit.viewer
    viewer = (mover + 2) % 4
    real = tuple(remaining_hand(state, s) for s in range(4))
    hand = real[mover]
    legal = tiles_of(rules.legal(hand, sit.record.led))
    n = len(legal)
    assert n > 1
    m = NET.move(rules, mover, hand, sit.record, 30, sit.bidder_team)
    for cfg in (Config((Rung(8, None),), field=NET, field_eps=0.05),
                Config((Rung(8, None),), field=NET, field_eps=0.7),
                Config((Rung(8, None),))):
        walt = Walt(cfg)
        before = walt.field_likelihood(Situation(sit.decl_id, viewer, 30, sit.bidder, sit.record, sit.plays), [real])[0]
        assert before > 0
        p = {}
        for t in legal:
            after_world = tuple(h & ~bit(t) if s == mover else h for s, h in enumerate(real))
            after = Situation(sit.decl_id, viewer, 30, sit.bidder, sit.record.advance(t, rules),
                              sit.plays + ((mover, t),))
            p[t] = walt.field_likelihood(after, [after_world])[0] / before
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-12)
        eps = cfg.field_eps_exact if cfg.field is not None else 1.0
        for t in legal:
            assert p[t] == pytest.approx(eps / n + (1 - eps) * (t == m), rel=1e-12)


def test_the_stochastic_field_is_validated_and_named_by_the_spec(tmp_path):
    with pytest.raises(ValueError):
        Config((Rung(8, None),), field_eps=0.1)              # no net: the random field is the dice
    with pytest.raises(ValueError):
        Config((Rung(8, None),), field=NET, field_eps=1.5)
    with pytest.raises(ValueError):
        Walt(Config((Rung(4, 4), Rung(8, None)), field=NET, field_eps=0.1)).field_likelihood(
            situation_from_state(_pip_states(1, seed=37, min_plays=6)[0], 30), [])
    assert Config((Rung(8, None),), field=NET, field_eps=0.5).field_eps_exact == 0.5
    assert abs(Config((Rung(8, None),), field=NET, field_eps=0.05).field_eps_exact - 0.05) <= 2.0 ** -53
    path = tmp_path / "net.npz"
    NET.save(path)
    p = parse_walt_spec(f"k0,n8,F={path},E=0.1")
    assert p.walt.config.field_eps == 0.1 and "net+eps0.1" in repr(p)
    assert parse_walt_spec(f"k0,n8,F={path}").walt.config.field_eps == 0.0
    with pytest.raises(ValueError):
        parse_walt_spec("k0,n8,E=0.1")
