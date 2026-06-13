"""Auction v0 tests: GusBidder policy over a stubbed P(make) table."""
import random

from arena.auction import PASS, BidContext, legal_bids
from champion.bidder import GusBidder
from champion.utility import MarksToSeven

_NAMES = [f"{a}-{b}" for a in range(7) for b in range(a + 1)]


def did(name: str) -> int:
    a, b = (int(x) for x in name.split("-"))
    return _NAMES.index(f"{max(a, b)}-{min(a, b)}")


# Three fives plus the 5-5: best_trump(hand, 3) exists, so no prefilter pass.
STRONG_HAND = tuple(did(n) for n in ("5-5", "5-4", "5-2", "5-0", "6-4", "3-2", "1-0"))
# Max pip count 2 and no three doubles: statically hopeless, prefiltered.
TRASH_HAND = tuple(did(n) for n in ("6-5", "6-4", "3-2", "3-1", "2-1", "5-0", "4-0"))


class StubEvaluator:
    """Fixed points table for every hand; counts invocations."""

    def __init__(self, table: dict[int, list[int]]):
        self.table = table
        self.calls = 0

    def __call__(self, hand: tuple[int, ...]) -> dict[int, list[int]]:
        self.calls += 1
        return self.table


def points(p30: float, p42: float, n: int = 100) -> list[int]:
    """n outcomes with P(>=30) = p30 and P(>=42) = p42."""
    n42 = round(n * p42)
    n30 = round(n * p30) - n42
    return [42] * n42 + [35] * n30 + [10] * (n - n42 - n30)


def ctx(hand=STRONG_HAND, high_bid=0, marks=(0, 0)) -> BidContext:
    return BidContext(
        hand=hand,
        seat=0,
        dealer=3,
        bids=(-1, 0, 0, -1),
        high_bid=high_bid,
        high_seat=-1 if high_bid == 0 else 1,
        legal=legal_bids(high_bid),
        marks=marks,
    )


def test_bids_minimum_positive_bid():
    ev = StubEvaluator({5: points(p30=0.8, p42=0.0)})
    bidder = GusBidder(ev)
    assert bidder.bid(ctx(), random.Random(0)) == 30


def test_overcalls_only_while_positive():
    ev = StubEvaluator({5: [36] * 70 + [10] * 30})  # P(>=36) = 0.7, P(>=37) = 0
    bidder = GusBidder(ev)
    assert bidder.bid(ctx(high_bid=33), random.Random(0)) == 34
    assert bidder.bid(ctx(high_bid=36), random.Random(0)) == PASS


def test_trash_passes_without_simulation():
    ev = StubEvaluator({5: points(p30=1.0, p42=1.0)})
    bidder = GusBidder(ev)
    assert bidder.bid(ctx(hand=TRASH_HAND), random.Random(0)) == PASS
    assert ev.calls == 0


def test_marginal_hand_passes():
    ev = StubEvaluator({5: points(p30=0.45, p42=0.0)})
    bidder = GusBidder(ev)
    assert bidder.bid(ctx(), random.Random(0)) == PASS


def test_hand_evaluated_once_across_bid_and_declare():
    ev = StubEvaluator({5: points(p30=0.8, p42=0.0)})
    bidder = GusBidder(ev)
    value = bidder.bid(ctx(), random.Random(0))
    assert bidder.declare(STRONG_HAND, value, random.Random(0)) == 5
    assert ev.calls == 1


def test_declare_picks_best_trump_at_contract_threshold():
    ev = StubEvaluator({
        2: points(p30=0.9, p42=0.0),
        5: points(p30=0.6, p42=0.0),
    })
    bidder = GusBidder(ev)
    assert bidder.declare(STRONG_HAND, 30, random.Random(0)) == 2


def test_score_conditioned_desperation_84():
    """Behind 0-6, MarksToSeven steps past negative point bids onto a
    positive two-mark gamble; MarkEV passes the same hand at every score."""
    table = {5: points(p30=0.45, p42=0.30)}
    behind = GusBidder(StubEvaluator(table), MarksToSeven())
    assert behind.bid(ctx(marks=(0, 6)), random.Random(0)) == 84
    assert behind.bid(ctx(marks=(0, 0)), random.Random(0)) == PASS
    score_blind = GusBidder(StubEvaluator(table))
    assert score_blind.bid(ctx(marks=(0, 6)), random.Random(0)) == PASS
