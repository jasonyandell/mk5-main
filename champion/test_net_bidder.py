"""Net-backed bidder: the rung-#22 distilled net wired into the policy via
GusBidder's `pmake_fn` path."""
import random

import pytest

from arena.auction import PASS, BidContext, legal_bids
from champion.bidder import GusBidder, NetPointsEvaluator
from champion.utility import MarkEV, MarksToSeven
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import DOMINO_IS_DOUBLE

EVAL_DECLS = {0, 1, 2, 3, 4, 5, 6, 7, 9}
THRESHOLDS = set(range(30, 43))


def _ctx(hand, marks=(0, 0)):
    return BidContext(
        hand=hand, seat=0, dealer=3, bids=(-1, -1, -1, -1),
        high_bid=0, high_seat=-1, legal=legal_bids(0), marks=marks, marks_to_win=7,
    )


def _hand_with_three_doubles():
    doubles = [d for d in range(28) if DOMINO_IS_DOUBLE[d]]
    others = [d for d in range(28) if not DOMINO_IS_DOUBLE[d]]
    return tuple(doubles[:3] + others[:4])  # passes the doubles prefilter


# --- the pmake_fn path (no model, pure) ---------------------------------------

def test_pmake_fn_needs_no_evaluator():
    GusBidder(pmake_fn=lambda h: {0: {t: 0.9 for t in THRESHOLDS}})  # no raise


def test_no_evaluator_and_no_pmake_raises():
    with pytest.raises(ValueError):
        GusBidder()


def test_pmake_fn_bids_minimum_positive():
    hand = _hand_with_three_doubles()
    pm = lambda h: {0: {t: 0.9 for t in THRESHOLDS}}  # always-make decl 0
    bidder = GusBidder(pmake_fn=pm, utility=MarkEV())
    assert bidder.bid(_ctx(hand), random.Random(0)) == 30  # cheapest positive
    assert bidder.declare(hand, 30, random.Random(0)) == 0


def test_pmake_fn_trash_passes():
    hand = _hand_with_three_doubles()
    pm = lambda h: {d: {t: 0.05 for t in THRESHOLDS} for d in EVAL_DECLS}  # nothing makes
    bidder = GusBidder(pmake_fn=pm, utility=MarkEV())
    assert bidder.bid(_ctx(hand), random.Random(0)) == PASS


# --- the real distilled net end to end (loads champion/bid_net.pt, CPU) --------

def test_net_evaluator_table_shape_and_range():
    ev = NetPointsEvaluator()
    table = ev(tuple(deal_from_seed(1234)[0]))
    assert set(table.keys()) == EVAL_DECLS
    for row in table.values():
        assert set(row.keys()) == THRESHOLDS
        assert all(0.0 <= p <= 1.0 for p in row.values())


def test_net_bidder_produces_legal_decisions():
    ev = NetPointsEvaluator()
    bidder = GusBidder(pmake_fn=ev, utility=MarksToSeven())
    rng = random.Random(0)
    for seed in range(25):
        hand = tuple(deal_from_seed(seed)[0])
        ctx = _ctx(hand)
        b = bidder.bid(ctx, rng)
        assert b == PASS or b in ctx.legal
        if b != PASS:
            assert bidder.declare(hand, b, rng) in EVAL_DECLS
