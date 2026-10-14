# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The reason report: on the book's four-fives 84 the separating fact is the
book's own set condition, an opponent holding the 6-5 with another five
(exactly 33/95 over the fiber), found as one fact; a split made of two facts
is found as those two; a real price the space cannot explain is reported as
no reason found, with the best candidates; an exact tie is indifferent; a
price inside the noise bar is indistinguishable; and the report is the
referee's too."""
import dataclasses
from fractions import Fraction
from itertools import islice
from pathlib import Path

import pytest
from scenario_core import SilasAuthority, SilasBelief, SilasQuery

from scenario.atlas import _state
from scenario.compiled import CompiledWalt
from scenario.planner import Analysis, Config, Rung, situation_from_state
from scenario.reason import (
    INDIFFERENT, INDISTINGUISHABLE, NO_REASON, ONE_FACT, REASON, SEPARATION, TWO_FACTS, _explain, candidates,
    noise_configs, reason, render, space, tile_name, two_fact,
)
from scenario.record import empty_record, remaining_hand
from scenario.silas import frame
from scenario.tests.test_rules_record import random_playing_states

SIX_FIVE_GUARDED = Path(__file__).resolve().parents[2] / "silas" / "examples" / "six-five-guarded.silas"


def tile(high: int, low: int) -> int:
    return high * (high + 1) // 2 + low


def eighty_four():
    """Winning 42, p. 113: fives trump, the bidder (seat 0, to lead) holds
    5-5 5-4 5-2 5-0 4-4 4-2 2-2 and bids 84 — every point, the 42 contract."""
    hand = [tile(5, 5), tile(5, 4), tile(5, 2), tile(5, 0), tile(4, 4), tile(4, 2), tile(2, 2)]
    rest = [t for t in range(28) if t not in hand]
    deal = (tuple(sorted(hand)), tuple(rest[:7]), tuple(rest[7:14]), tuple(rest[14:]))
    return _state(deal, 0, 42, 5, (), empty_record(0))


def test_the_eighty_fours_reason_is_the_guarded_six_five(monkeypatch):
    """Zeb's lead against the book's 84 line, the double-five: they part in
    the worlds where an opponent guards the 6-5 with another five — the 5-5
    is set in every one of them — and that fact, found by search, reads on
    Zeb's own worlds exactly as the book's query does, at 33/95 exactly
    over the fiber."""
    cfg = Config((Rung(4, None), Rung(64, None)), "make", seed=1)
    state = eighty_four()
    # The report plans the decision once; the book check below reads the same
    # analysis (the line is the only thing `line=True` adds) rather than re-plan it.
    planned = []
    analyze = CompiledWalt.analyze

    def spy(self, *args, **kw):
        out = analyze(self, *args, **kw)
        planned.append(out)
        return out

    monkeypatch.setattr(CompiledWalt, "analyze", spy)
    r = reason(state, 42, tile(5, 5), cfg, bar=[("seed 2", dataclasses.replace(cfg, seed=2))])
    monkeypatch.undo()
    assert len(planned) == 1
    assert r.tile != tile(5, 5) and r.verdict == REASON and r.compare == "better"
    assert r.authority == "sampled(64, 1)" and r.n_worlds == 64
    assert r.price > float(r.bar) > 0 and abs(float(r.share_a - r.share_b) - r.price) < 1e-12
    assert r.found == ONE_FACT
    best = r.facts[0]
    assert best.candidate.gloss == "an opponent holds 6-5 and another of trump"
    assert (best.prior, best.prior_authority) == (Fraction(33, 95), "exact")
    assert best.on_a == 1 and best.on_b < Fraction(1, 10) and best.authority == "sampled(64, 1)"
    assert best.count_a == len(r.side_a)
    assert all(best.separation >= f.separation for f in r.facts)
    assert best.coverage == (1, 1 - best.on_b) and best.separation >= SEPARATION and best.separation > r.chance[0]
    # The book's own text reads the same masses off the same worlds.
    sit = situation_from_state(state, 42)
    f = frame(sit, remaining_hand(state, 0))
    analysis = planned[0]
    assert [r.tile, r.q] == [analysis.tile, analysis.q]
    sampled = SilasAuthority.sampled(64, 1)
    book = SilasQuery(SIX_FIVE_GUARDED.read_text())
    side = SilasBelief.from_worlds(f, [analysis.worlds[i] for i in r.side_a],
                                   [analysis.weights[i] for i in r.side_a], sampled)
    assert book.summarize(side, 10_000_000).event_probability.value == best.on_a
    # The counterexample: the guarded 6-5 holds, Zeb's tile makes, the 5-5 is set.
    ex = r.examples[0]
    assert ex.fact_holds and ex.paths[tile(5, 5)][0] == (0, tile(5, 5))
    assert ex.margins[r.tile] >= 42 > ex.margins[tile(5, 5)]
    text = render(r)
    assert "33/95" in text and "an opponent holds 6-5 and another of trump" in text and "sampled(64, 1)" in text


def _state_at(i: int):
    """The i-th state of the seeded random games (a fixed decision)."""
    return next(islice(random_playing_states(30, seed=5), i, None))


CHEAP = Config((Rung(32, None),), "make", seed=3)


def test_an_exact_tie_is_indifferent_and_gives_no_reason():
    state = _state_at(4)
    r = reason(state, 30, 8, CHEAP)
    assert r.tile == 1 and r.q[1] == r.q[8]
    assert r.verdict == INDIFFERENT and r.beats == Fraction(1, 2) and r.facts == () and r.examples == (None, None)
    assert "Indifferent — no reason" in render(r)


def test_a_price_inside_the_noise_bar_is_indistinguishable():
    """The default bar: the next seed and one rung stronger. The price is
    real but smaller than the re-plans move it, so no reason is given — and
    the referee writes the same report, byte for byte."""
    state = _state_at(11)
    r = reason(state, 30, 4, CHEAP)
    assert r.tile == 5 and [a.label for a in r.arms] == [label for label, _ in noise_configs(CHEAP)]
    assert [a.config.level for a in r.arms] == [0, 1] and r.arms[0].config.seed == 4
    assert 0 < abs(r.price) < float(r.bar)
    assert r.verdict == INDISTINGUISHABLE and r.compare == "indistinguishable" and r.facts == ()
    assert "Indistinguishable — no reason" in render(r)
    assert render(reason(state, 30, 4, CHEAP, compiled=False)) == render(r)


def test_the_candidate_space_is_declared_and_compiles():
    state = eighty_four()
    sit = situation_from_state(state, 42)
    rec = sit.record
    space = candidates(5, 0, remaining_hand(state, 0), rec.played, [rec.hand_size(s) for s in range(4)])
    texts = [c.text for c in space]
    assert len(set(texts)) == len(texts)
    kinds = {"holds": 0, "another": 0, "void": 0}
    for c in space:
        SilasQuery(c.text)
        kinds["another" if "another" in c.gloss else "void" if "void" in c.gloss else "holds"] += 1
    # 21 hidden tiles by three seats and by "an opponent"; seven suits with a hidden tile.
    assert kinds["holds"] == 4 * 21 and kinds["void"] == 4 * 7
    assert any(c.gloss == "an opponent holds 6-5 and another of trump" for c in space)


def test_the_grown_space_is_declared_and_composes():
    """Every family compiles, leading and following; every fact is also one
    clause, so any two compose into one Silas query, and/or."""
    for state, bid, pair in ((eighty_four(), 42, (tile(2, 2), tile(5, 5))), (_state_at(13), 30, ())):
        sit = situation_from_state(state, bid)
        own = remaining_hand(state, sit.viewer)
        facts = space(sit, own, pair)
        assert len({c.text for c in facts}) == len(facts)
        families = {c.family for c in facts}
        assert families == {"holds", "holds with", "void", "position", "count at risk", "prelude"}
        for c in facts:
            SilasQuery(c.text)
        for a, b in zip(facts, facts[1:]):
            assert SilasQuery(two_fact(a, b, "and").text) and SilasQuery(two_fact(a, b, "or").text)
    leading = [c.gloss for c in space(situation_from_state(eighty_four(), 42), remaining_hand(eighty_four(), 0),
                                      (tile(2, 2), tile(5, 5)))]
    assert "S1 (opponent, next to play) can beat 2-2 led" in leading
    assert "S3 (opponent, the last to play) is void in the suit 2-2 leads (deuces) and holds trump" in leading
    assert "a count tile I could lead is capturable by the next seat (S1)" in leading
    assert "an opponent holds a trump none of mine beats" in leading


def _worlds(state, bid: int, n: int, seed: int):
    """n deals of the hidden tiles that fit the viewer's record, seeded."""
    import random
    sit = situation_from_state(state, bid)
    own, rec = remaining_hand(state, sit.viewer), sit.record
    hidden = [t for t in range(28) if not (own | rec.played) >> t & 1]
    rng, out = random.Random(seed), []
    while len(out) < n:
        rng.shuffle(hidden)
        world, at = [0, 0, 0, 0], 0
        world[sit.viewer] = own
        for s in range(4):
            if s != sit.viewer:
                k = rec.hand_size(s)
                world[s] = sum(1 << t for t in hidden[at:at + k])
                at += k
        out.append(tuple(world))
    return sit, own, out


def test_a_split_made_of_two_facts_is_found_as_two():
    """On the 84's frame, a fixed sample of 128 deals split by a rule known in
    advance — S1 holds the 6-6 and S3 the 6-5 — against the rest: the two-fact
    explanation is that rule, separating perfectly, and it is the reason,
    since no single fact does as well."""
    state = eighty_four()
    sit, own, worlds = _worlds(state, 42, 128, 7)
    f = frame(sit, own)
    six_six, six_five = 1 << tile(6, 6), 1 << tile(6, 5)
    one = tuple(r for r, w in enumerate(worlds) if w[1] & six_six and w[3] & six_five)
    two = tuple(r for r in range(len(worlds)) if r not in one)
    a = Analysis(tile=tile(2, 2), q={}, worlds=list(worlds), weights=[1.0] * len(worlds), outcomes={})
    found, facts, pairs, chance, searched = _explain(
        f, a, one, two, SilasAuthority.sampled(128, 7), sit, own, (tile(2, 2), tile(5, 5)), 5)
    assert 0 < len(one) < 32 and found == TWO_FACTS
    best = pairs[0]
    assert best.candidate.gloss == "S1 (opponent) holds 6-6 and S3 (opponent) holds 6-5"
    assert best.candidate.text == "(query (roles) (out) (pattern (holds S1 6-6) (holds S3 6-5)))"
    assert best.separation == 1 and best.coverage == (1, 1) and (best.count_a, best.count_b) == (len(one), 0)
    assert best.prior == Fraction(7, 21) * Fraction(7, 20) and best.prior_authority == "exact"
    assert SEPARATION <= facts[0].separation < 1 and chance[1] < 1
    assert searched[1] <= searched[0]


def test_a_price_the_space_cannot_explain_says_no_reason_found():
    """A real price outside the bar whose split no fact of the space carries:
    the best single fact separates just under one half, the best pair less
    than chance on sides this small. The report says so and shows them."""
    state = _state_at(0)
    r = reason(state, 30, 6, CHEAP)
    assert r.tile == 0 and r.verdict == REASON and r.found == NO_REASON
    assert r.facts and r.pairs and r.facts[0].separation < SEPARATION and r.pairs[0].separation <= r.chance[1]
    text = render(r)
    assert "No reason found." in text and "The best candidates anyway" in text
    assert r.facts[0].candidate.gloss in text and r.pairs[0].candidate.gloss in text


def test_the_report_refuses_what_it_cannot_read():
    state = eighty_four()
    with pytest.raises(ValueError, match="make utility"):
        reason(state, 42, tile(5, 5), dataclasses.replace(CHEAP, utility="points"))
    with pytest.raises(ValueError, match="not a legal tile"):
        reason(state, 42, tile(6, 6), CHEAP)
    zeb = reason(state, 42, tile(5, 5), CHEAP, bar=[]).tile
    with pytest.raises(ValueError, match="Zeb's own"):
        reason(state, 42, zeb, CHEAP)
    assert tile_name(tile(6, 5)) == "6-5"
