"""Zeb's arena bidder: the package's forecast and chooseBid, and the
match-equity option (arena/match_equity.py)."""
import json
import random
from pathlib import Path

import pytest

from arena.auction import BidContext, legal_bids
from arena.match_equity import HandLaw, Linear, MatchEquity, Odds, mark_bid_value, one_mark_value
from arena.zeb_bid import Shrink, ZebBidder, choose_bid, forecast

ROOT = Path(__file__).parent.parent
PARITY = ROOT / "zeb" / "tests" / "fixtures" / "parity.json"


def _walt(worlds: int, seed: int):
    pytest.importorskip("scenario_core")
    from scenario.compiled import CompiledWalt
    from scenario.field import FieldNet
    from scenario.planner import Config, Rung

    net = FieldNet.from_bytes((ROOT / "zeb" / "net" / "margin_d250k_h128x64.fnet").read_bytes())
    return CompiledWalt(Config((Rung(worlds, None, 16),), "make", seed, field=net))


def test_forecast_is_the_packages():
    """Every pip-declaration forecast in the package's parity fixture, to the bit."""
    cases = [c for c in json.loads(PARITY.read_text())["positions"]
             if "forecast" in c["referee"] and c["request"]["decl"] < 7]
    assert cases
    for c in cases:
        r = c["request"]
        fc = forecast(_walt(r["worlds"], int(r["seed"])), tuple(r["hand"]), (r["decl"],))
        assert list(fc[0][1]) == c["referee"]["forecast"]["pmake"]


def test_forecast_reads_only_the_bidders_hand():
    walt = _walt(64, 3)
    hand = (0, 1, 6, 8, 9, 15, 20)
    assert forecast(walt, hand) == forecast(walt, tuple(reversed(hand)))


def test_choose_bid_is_index_js():
    fc = [(0, (0.95,) * 6 + (0.85,) + (0.7,) * 6), (3, (0.97,) * 13)]
    assert choose_bid(fc, 30, 0.8) == (42, 3, 0.97)
    assert choose_bid([(0, fc[0][1])], 30, 0.8) == (36, 0, 0.85)
    assert choose_bid([(0, fc[0][1])], 37, 0.8) is None


def test_one_mark_calls_are_score_blind():
    """With one mark at stake on every branch, E[V] is V(a,b+1) + π·Δ, Δ > 0:
    the call with the larger π wins at every score."""
    v = MatchEquity(HandLaw(((0.1, 0.45), (0.05, 0.4))))
    for a in range(7):
        for b in range(7):
            for nd in (0, 1):
                lo, hi = one_mark_value(v, a, b, nd, 0.4), one_mark_value(v, a, b, nd, 0.6)
                assert hi > lo


def test_eighty_four_at_their_six_is_free():
    """They need one mark: a set loses the game at 42 or 84, a make at 84 is
    two marks. 84 over 42 for any p > 0 when we need two or more."""
    v = MatchEquity(HandLaw.coin())
    for a in range(5):
        for p in (0.05, 0.5, 0.95):
            assert mark_bid_value(v, a, 6, 0, 2, p) > one_mark_value(v, a, 6, 0, p)
    # one mark from game ourselves: 42 dominates 84 (the book's rule)
    for b in range(6):
        assert mark_bid_value(v, 6, b, 0, 2, 0.7) < one_mark_value(v, 6, b, 0, 0.7)
    assert mark_bid_value(v, 6, 6, 0, 2, 0.7) == one_mark_value(v, 6, 6, 0, 0.7)  # 6-6: one hand decides


def test_coin_is_race_wp():
    from champion.utility import race_wp

    v = MatchEquity(HandLaw.coin())
    for a in range(7):
        for b in range(7):
            assert v(a, b, 0) == pytest.approx(race_wp(7 - a, 7 - b))


class _Fixed(ZebBidder):
    def __init__(self, fc, mode, odds):
        super().__init__(walt=None, mode=mode, law=HandLaw.coin(), odds=odds)
        self._fixed = fc

    def forecast(self, hand):
        return self._fixed


def _ctx(marks, high_bid=0, high_seat=-1, bids=(-1, -1, -1, -1), seat=1, dealer=0):
    return BidContext(hand=tuple(range(7)), seat=seat, dealer=dealer, bids=bids, high_bid=high_bid,
                      high_seat=high_seat, legal=legal_bids(high_bid), marks=marks)


_K = [(h, k) for h in ("none", "partner", "opp") for k in range(4)]
ODDS = Odds(keep={(*k, b): 0.8 for k in _K for b in ("low", "high")}, over_set=0.3,
            pass_ours={k: 0.45 for k in _K})


def test_met_bids_84_where_they_need_one():
    # p42 = 0.5, p36 = 0.85: the hand-only call is 36
    fc = [(2, (0.95,) * 6 + (0.85,) + (0.7,) * 5 + (0.5,))]
    rng = random.Random(0)
    hand = _Fixed(fc, "hand", ODDS)
    met = _Fixed(fc, "met", ODDS)
    marks = _Fixed(fc, "marks", ODDS)
    # seat 1 is team 1: marks (6, 0) puts them at 6 and us at 0
    assert hand.bid(_ctx((6, 0)), rng) == 36
    assert met.bid(_ctx((6, 0)), rng) == 84       # they need one: the 84 is free
    assert met.bid(_ctx((0, 6)), rng) == 36       # we need one: 42 or below
    assert marks.bid(_ctx((6, 0)), rng) == 36     # score-blind: 84 is worth 2(2·0.5 − 1) = 0 marks
    hand.bid(_ctx((6, 0)), rng)
    assert hand.decisions[-1].call == 84          # the hand-only bidder records the priced call


def test_partner_is_overcalled_only_by_a_mark_bid():
    fc = [(2, (1.0,) * 12 + (0.6,))]
    ctx = dict(high_bid=30, high_seat=3, bids=(0, -1, 0, 30))
    assert _Fixed(fc, "hand", ODDS).bid(_ctx((0, 0), **ctx), random.Random(0)) == 0
    # we (team 1) at 5, they at 6: 84 at 0.6 beats leaving partner's 30
    assert _Fixed(fc, "met", ODDS).bid(_ctx((6, 5), **ctx), random.Random(0)) == 84


def test_linear_is_the_mark_margin():
    assert Linear()(5, 2, 0) == 3.0


class _Flags(_Fixed):
    def __init__(self, fc, mode, odds, **kw):
        ZebBidder.__init__(self, walt=None, mode=mode, law=HandLaw.coin(), odds=odds, **kw)
        self._fixed = fc


# an opponent (seat 0, team 0) holds 30; we are seat 1 (team 1), two seats after us
_OVER_OPP = dict(high_bid=30, high_seat=0, bids=(30, -1, -1, -1), dealer=3)
_OPP_ODDS = Odds(keep={(*k, b): 1.0 for k in _K for b in ("low", "high")}, over_set=0.2,
                 pass_ours={k: 0.45 if k[0] == "none" else 0.2 for k in _K})


def test_over_overcalls_when_the_mark_is_likelier_than_passing():
    """Over an opponent's bid, `over` bids the call with the largest π once it
    beats passing's π; the 0.8 rule passes the same hand."""
    fc = [(1, (0.6, 0.35) + (0.1,) * 11), (4, (0.5, 0.3) + (0.05,) * 11)]
    rng = random.Random(0)
    assert _Flags(fc, "hand", _OPP_ODDS).bid(_ctx((0, 0), **_OVER_OPP), rng) == 0
    assert _Flags(fc, "hand", _OPP_ODDS, over=True).bid(_ctx((0, 0), **_OVER_OPP), rng) == 31
    weak = [(1, (0.6, 0.15) + (0.1,) * 11)]  # 31 makes 15% < passing's 20%
    assert _Flags(weak, "hand", _OPP_ODDS, over=True).bid(_ctx((0, 0), **_OVER_OPP), rng) == 0
    # nobody bid: passing is worth 45%, and `over` acts only over an opponent
    assert _Flags(fc, "hand", _OPP_ODDS, over=True).bid(_ctx((0, 0)), rng) == 0


def test_over_is_score_blind():
    """One mark on every branch: the overcall is the same at every score."""
    b = _Flags([(1, (0.6, 0.35) + (0.1,) * 11)], "hand", _OPP_ODDS, over=True)
    calls = {b.bid(_ctx((x, y), **_OVER_OPP), random.Random(0)) for x in range(7) for y in range(7)}
    assert calls == {31}


def test_cal_reads_a_calibrated_forecast_only_over_an_opponent():
    cool = Shrink(-1.0, 1.0)  # every p lower
    assert Shrink(0.0, 1.0)(0.3) == pytest.approx(0.3)
    assert cool(0.3) < 0.3 and cool(0.9) < 0.9 and cool(0.3) < cool(0.31)
    fc = [(1, (0.9,) * 12 + (0.85,))]
    rng = random.Random(0)
    # over an opponent the 0.8 rule reads p̃: 0.9 becomes 0.77, and every bid is gone
    assert _Flags(fc, "hand", _OPP_ODDS).bid(_ctx((0, 0), **_OVER_OPP), rng) == 42
    assert _Flags(fc, "hand", _OPP_ODDS, cal=True, shrink=cool).bid(_ctx((0, 0), **_OVER_OPP), rng) == 0
    # nobody bid: the raw forecast
    assert _Flags(fc, "hand", _OPP_ODDS, cal=True, shrink=cool).bid(_ctx((0, 0)), rng) == 42
    # the mark bid is priced with p̃42 over an opponent
    met = _Flags(fc, "met", _OPP_ODDS, cal=True, shrink=cool)
    met.bid(_ctx((6, 0), **_OVER_OPP), rng)
    assert met.decisions[-1].p42 == pytest.approx(cool(0.85))


def test_flags_are_off_by_default():
    b = ZebBidder(walt=None)
    assert (b.mode, b.over, b.cal) == ("hand", False, False)
