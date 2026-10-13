"""The reason report: on the book's four-fives 84 the separating fact is the
book's own set condition, an opponent holding the 6-5 with another five
(exactly 33/95 over the fiber); an exact tie is indifferent; a price inside
the noise bar is indistinguishable; and the report is the referee's too."""
import dataclasses
from fractions import Fraction
from itertools import islice
from pathlib import Path

import pytest
from scenario_core import SilasAuthority, SilasBelief, SilasQuery

from scenario.atlas import _state
from scenario.compiled import CompiledWalt
from scenario.planner import Config, Rung, situation_from_state
from scenario.reason import (
    INDIFFERENT, INDISTINGUISHABLE, REASON, candidates, noise_configs, reason, render, tile_name,
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


def test_the_eighty_fours_reason_is_the_guarded_six_five():
    """Zeb's lead against the book's 84 line, the double-five: they part in
    the worlds where an opponent guards the 6-5 with another five — the 5-5
    is set in every one of them — and that fact, found by search, reads on
    Zeb's own worlds exactly as the book's query does, at 33/95 exactly
    over the fiber."""
    cfg = Config((Rung(4, None), Rung(64, None)), "make", seed=1)
    state = eighty_four()
    r = reason(state, 42, tile(5, 5), cfg, bar=[("seed 2", dataclasses.replace(cfg, seed=2))])
    assert r.tile != tile(5, 5) and r.verdict == REASON and r.compare == "better"
    assert r.authority == "sampled(64, 1)" and r.n_worlds == 64
    assert r.price > float(r.bar) > 0 and abs(float(r.share_a - r.share_b) - r.price) < 1e-12
    best = r.facts[0]
    assert best.candidate.gloss == "an opponent holds 6-5 and another of trump"
    assert (best.prior, best.prior_authority) == (Fraction(33, 95), "exact")
    assert best.on_a == 1 and best.on_b < Fraction(1, 10) and best.authority == "sampled(64, 1)"
    assert best.count_a == len(r.side_a)
    assert all(best.separation >= f.separation for f in r.facts)
    # The book's own text reads the same masses off the same worlds.
    sit = situation_from_state(state, 42)
    f = frame(sit, remaining_hand(state, 0))
    analysis = CompiledWalt(cfg).analyze(state, 42)
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
