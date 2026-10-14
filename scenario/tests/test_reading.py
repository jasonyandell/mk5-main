"""Reading the other seats' plays by the net's set (`Config.reading`): the
posterior's likelihood is the legal tiles the field net scores within a
margin of its best, and a play weighs [tile ∈ A] (consistent) or
[tile ∈ A]/|A| (uniform: restricted choice). Referee = core to the bit,
proposals and fallbacks included; a set that allows every legal tile says
nothing under the consistent likelihood; a margin of zero is the net's own
filter; and on the restricted-choice deal of the lumps study the uniform
likelihood reproduces Silas's exact counts (615/2708 and 715/2031)."""

from math import comb

import pytest
import scenario_core

from arena.walt_play import parse_walt_spec
from scenario.atlas import _state
from scenario.compiled import CompiledWalt
from scenario.fiber import Fiber
from scenario.planner import READ_ENUM, Config, Reading, Rung, Situation, Walt, situation_from_state
from scenario.record import empty_record, remaining_hand
from scenario.rules import bit, mask_of, rules_for
from scenario.tests.test_compiled import _same_analysis
from scenario.tests.test_field import NET
from scenario.tests.test_posterior import LATE, SAMPLED, _positions

EARLY = _positions([(3, 2), (5, 3), (12, 3), (14, 2)])   # fibers above READ_ENUM: rejection

MARGIN = 10_000_000          # about the 40th percentile of the test net's score gaps
EVERY = (1 << 62) - 1        # wider than every gap: the any-legal-tile policy


def _with(cfg: Config, **kw) -> Config:
    return Config(**{**cfg.__dict__, **kw})


READING_CONFIGS = [
    Config((Rung(16, None, 6),), "make", seed=3, field=NET, draw="posterior", draw_budget=20_000,
           reading=Reading("consistent", MARGIN)),
    Config((Rung(12, None, 4),), "points", seed=5, field=NET, draw="posterior", draw_budget=20_000,
           reading=Reading("uniform", MARGIN)),
    Config((Rung(16, 6, 6),), "make", seed=7, field=NET, draw="posterior", draw_budget=40,
           reading=Reading("uniform", MARGIN)),                       # the budget runs out
]


def _fiber(state) -> int:
    sit = situation_from_state(state, 30)
    rec = sit.record
    own = remaining_hand(state, sit.viewer)
    return Fiber(sit.viewer, own, rec.played, tuple(rec.hand_size(s) for s in range(4)), rec.forbid).size


def test_compiled_matches_the_planner_under_a_reading():
    """Every path of the draw, referee = core: the exact fiber (at most n),
    the enumerated draw (at most `READ_ENUM`), rejection above it, and the
    fallback when the budget runs out."""
    paths = {"exact": 0, "enumerated": 0, "rejection": 0}
    fell = moved = 0
    for cfg in READING_CONFIGS:
        ref, fast = Walt(cfg), CompiledWalt(cfg)
        for state in EARLY + SAMPLED + LATE:
            a = ref.analyze(state, 30)
            _same_analysis(a, fast.analyze(state, 30))
            assert (ref.nodes, ref.proposals, ref.fallbacks) == (fast.nodes, fast.proposals, fast.fallbacks)
            d = fast.decide(state, 30)
            assert d[0] == a.tile and d[1] == a.q
            size = _fiber(state)
            path = "exact" if size <= cfg.rungs[-1].n_worlds else "enumerated" if size <= READ_ENUM else "rejection"
            paths[path] += not ref.fallbacks
            assert ref.proposals == {"exact": 0, "enumerated": size}.get(path, ref.proposals)
            fell += ref.fallbacks
            moved += a.q != Walt(_with(cfg, reading=None)).analyze(state, 30).q
    assert min(paths.values()) >= 3 and fell >= 1 and moved >= 8


def test_the_enumerated_draw_follows_the_likelihood():
    """On a fiber larger than n and at most `READ_ENUM`, the n worlds are
    draws from the whole fiber weighted by L: every one has L > 0, and the
    mass of a holding among them matches its exact posterior mass."""
    checked = 0
    for state in LATE:
        size = _fiber(state)
        if size <= 64:
            continue
        cfg = Config((Rung(min(1024, size - 1), None, 16),), "make", seed=9, field=NET, draw="posterior",
                     reading=Reading("uniform", MARGIN))
        sit = situation_from_state(state, 30)
        ref = Walt(cfg)
        a = ref.analyze(state, 30)
        assert ref.proposals == size and not ref.fallbacks
        every = Walt(_with(cfg, draw="uniform", reading=None, rungs=(Rung(size, None, 16),))).analyze(state, 30).worlds
        like = ref.reading_likelihood(sit, every)
        assert all(x > 0 for x in ref.reading_likelihood(sit, a.worlds))
        left = (sit.viewer + 1) % 4
        for t in range(28):
            exact = sum(x for w, x in zip(every, like) if w[left] >> t & 1) / sum(like)
            if 0.05 < exact < 0.95:
                seen = sum(w[left] >> t & 1 for w in a.worlds) / len(a.worlds)
                assert abs(seen - exact) < 4 * (exact * (1 - exact) / len(a.worlds)) ** 0.5
                checked += 1
    assert checked >= 3


def test_an_enumerated_fiber_is_weighted_by_the_reading():
    """On an enumerated fiber the weights are the likelihood normalized: the
    consistent one a 0/1 filter, the uniform one unequal among the kept."""
    zeros = uneven = 0
    for likelihood in ("consistent", "uniform"):
        cfg = Config((Rung(512, None, 16),), "make", seed=1, field=NET, draw="posterior",
                     reading=Reading(likelihood, MARGIN))
        for state in LATE:
            sit = situation_from_state(state, 30)
            ref = Walt(cfg)
            a = ref.analyze(state, 30)
            assert a.worlds == Walt(_with(cfg, draw="uniform", reading=None)).analyze(state, 30).worlds
            like = ref.reading_likelihood(sit, a.worlds)
            assert a.weights == [x / sum(like) for x in like]
            assert ref.proposals == 0 and ref.fallbacks == 0
            _same_analysis(a, CompiledWalt(cfg).analyze(state, 30))
            zeros += a.weights.count(0.0)
            uneven += len({x for x in like if x}) > 1
    assert zeros and uneven


def test_a_set_of_every_legal_tile_says_nothing_when_consistent():
    """A seat that may play any legal tile is silent under the consistent
    likelihood: L is 1 on every world, and where the draw is by rejection
    every proposal is kept, so the worlds are the uniform sample and the plan
    is the uniform player's. Under the uniform likelihood the same seat is
    evidence (it favours hands with fewer legal tiles): L varies."""
    base = Config((Rung(32, None, 8),), "make", seed=4, field=NET)
    silent = _with(base, draw="posterior", reading=Reading("consistent", EVERY))
    random = _with(base, draw="posterior", reading=Reading("uniform", EVERY))
    varied = same = 0
    for state in EARLY + SAMPLED[:4]:
        sit = situation_from_state(state, 30)
        u = Walt(base).analyze(state, 30)
        assert Walt(silent).reading_likelihood(sit, u.worlds) == [1.0] * 32
        varied += len(set(Walt(random).reading_likelihood(sit, u.worlds))) > 1
        if _fiber(state) > READ_ENUM:
            for walt in (Walt(silent), CompiledWalt(silent)):
                _same_analysis(u, walt.analyze(state, 30))
                assert walt.proposals == 32 and not walt.fallbacks
            same += 1
    assert varied >= 4 and same >= 2


def test_margin_zero_is_the_nets_filter():
    """With no exact tie in the net's scores, the set at margin 0 is the
    net's move alone, and the consistent likelihood is the deterministic
    field's 0/1 filter (`field_likelihood` at ε = 0), world for world."""
    zero = Walt(Config((Rung(64, None, 16),), field=NET, draw="posterior", reading=Reading("consistent", 0)))
    field = Walt(Config((Rung(64, None, 16),), field=NET, draw="posterior"))
    kept = 0
    for state in SAMPLED:
        sit = situation_from_state(state, 30)
        worlds = CompiledWalt(_with(zero.config, draw="uniform", reading=None)).analyze(state, 30).worlds
        like = zero.reading_likelihood(sit, worlds)
        assert like == field.field_likelihood(sit, worlds)
        kept += sum(like)
    assert 0 < kept < 64 * len(SAMPLED)


# -- the restricted-choice deal (wiki/experiments/lumps-voi-2026-10-13.md) -------

def _tile(name: str) -> int:
    from forge.oracle.tables import DOMINOES
    hi, lo = map(int, name.split("-"))
    return DOMINOES.index((max(hi, lo), min(hi, lo)))


def test_restricted_choice_reproduces_silas():
    """Sixes, bid 30. The bidder (seat 0) leads the 5-4 and the left setter
    (seat 1) follows with the 5-0; the partner (seat 2) holds 6-2 6-1 6-0
    2-2 4-3 4-2 3-2. Every seat read as allowing every legal tile (a margin
    wider than every gap). The posterior over the left setter's six unseen
    tiles is the likelihood times the ways the other two seats complete the
    deal — C(13, 6) for every hand, so it cancels — and the bidder's lead
    weighs 1/7 in every world. Silas's exact counts (`silas query
    left-holds-55.silas --read left=any.silas`, the lumps study's Data):

        P(left holds the 5-5)       consistent 6/19,       uniform 615/2708
        P(the 5-0 was his only five) consistent 715/3876,  uniform 715/2031
    """
    decl, bidder, viewer = 6, 0, 2
    rules = rules_for(decl)
    plays = ((0, _tile("5-4")), (1, _tile("5-0")))
    rec = empty_record(bidder)
    for _, t in plays:
        rec = rec.advance(t, rules)
    partner = mask_of(_tile(n) for n in ("6-2", "6-1", "6-0", "2-2", "4-3", "4-2", "3-2"))
    unseen = [t for t in range(28) if not (partner | bit(plays[0][1]) | bit(plays[1][1])) >> t & 1]
    assert len(unseen) == 19
    sit = Situation(decl, viewer, 30, bidder, rec, plays)
    fives = {_tile(n) for n in ("5-5", "5-3", "5-2", "5-1")}

    from itertools import combinations
    worlds = []
    for left in combinations(unseen, 6):
        rest = [t for t in unseen if t not in left]
        worlds.append((mask_of(rest[:6]), mask_of(left), partner, mask_of(rest[6:])))
    assert len(worlds) == comb(19, 6)
    holds55 = [w[1] >> _tile("5-5") & 1 for w in worlds]
    only = [not any(w[1] >> t & 1 for t in fives) for w in worlds]

    def p(like, event):
        return sum(x for x, e in zip(like, event) if e) / sum(like)

    for likelihood, (p55, ponly) in (("consistent", (6 / 19, 715 / 3876)), ("uniform", (615 / 2708, 715 / 2031))):
        walt = Walt(Config((Rung(16, None, 4),), field=NET, draw="posterior", reading=Reading(likelihood, EVERY)))
        like = walt.reading_likelihood(sit, worlds)
        assert abs(p(like, holds55) - p55) < 1e-12
        assert abs(p(like, only) - ponly) < 1e-12
    # The bidder's lead is one of seven in every world; the follow is one of his fives.
    walt = Walt(Config((Rung(16, None, 4),), field=NET, draw="posterior", reading=Reading("uniform", EVERY)))
    assert walt.reading_likelihood(sit, [worlds[0]])[0] == 1 / 7 / (1 + sum(worlds[0][1] >> t & 1 for t in fives))
    # And a state the planner reads the same way plans without complaint.
    hands = tuple(tuple(sorted(t for t in range(28) if (w | bit(p)) >> t & 1))
                  for w, p in zip(worlds[0], (plays[0][1], plays[1][1], 0, 0)))
    state = _state(tuple(h if s != 2 else tuple(sorted(t for t in range(28) if partner >> t & 1))
                         for s, h in enumerate(hands)), bidder, 30, decl, plays, rec)
    assert CompiledWalt(walt.config).decide(state, 30)[0] in {t for t in range(28) if partner >> t & 1}


def test_the_reading_is_validated_and_named_by_the_spec(tmp_path):
    with pytest.raises(ValueError):
        Reading("bayes", 1)
    with pytest.raises(ValueError):
        Reading("uniform", -1)
    with pytest.raises(ValueError):
        Config((Rung(8, None),), field=NET, reading=Reading("uniform", 1))    # a reading needs the posterior
    with pytest.raises(ValueError):
        scenario_core.Core([(8, None, 1)], "make", 0, NET.to_bytes(), 0.0, "uniform", 10, None, ("uniform", 1))
    with pytest.raises(ValueError):
        scenario_core.Core([(8, None, 1)], "make", 0, NET.to_bytes(), 0.0, "posterior", 10, None, ("bayes", 1))
    path = tmp_path / "net.npz"
    NET.save(path)
    p = parse_walt_spec(f"k0,n8,F={path},d=posterior,r=uniform:{MARGIN}")
    assert p.walt.config.reading == Reading("uniform", MARGIN) and f"reading=uniform:{MARGIN}" in repr(p)
    assert parse_walt_spec(f"k0,n8,F={path},d=posterior").walt.config.reading is None
