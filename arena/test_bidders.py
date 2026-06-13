"""Bid policy behavior tests."""
import random

from forge.oracle.tables import DOMINOES

from arena.auction import PASS, BidContext, legal_bids
from arena.bidders import Bid30Bidder, HeuristicBidder, RandomBidder
from arena.hand_metrics import best_trump

NAME_TO_ID = {f"{h}-{l}": i for i, (h, l) in enumerate(DOMINOES)}


def hand(*names: str) -> tuple[int, ...]:
    return tuple(NAME_TO_ID[n] for n in names)


def ctx(h: tuple[int, ...], high_bid: int = 0, seat: int = 1) -> BidContext:
    return BidContext(
        hand=h, seat=seat, dealer=0,
        bids=(-1, -1, -1, -1), high_bid=high_bid,
        high_seat=-1 if high_bid == 0 else 0,
        legal=legal_bids(high_bid),
    )


MONSTER = ("5-0", "5-1", "5-2", "5-3", "5-4", "5-5", "6-5")
# A 7-cycle: every pip appears exactly twice, so no suit reaches 3 tiles
GARBAGE = ("1-0", "2-1", "3-2", "4-3", "5-4", "6-5", "6-0")


def test_heuristic_opens_only_enough():
    rng = random.Random(0)
    bidder = HeuristicBidder()
    assert bidder.bid(ctx(hand(*MONSTER)), rng) == 30  # ceiling 42, bids minimum
    assert bidder.bid(ctx(hand(*MONSTER), high_bid=41), rng) == 42


def test_heuristic_passes_garbage_and_mark_bids():
    rng = random.Random(0)
    bidder = HeuristicBidder()
    assert best_trump(hand(*GARBAGE), min_trumps=3) is None
    assert bidder.bid(ctx(hand(*GARBAGE)), rng) == PASS
    # Even a monster never bids marks beyond 42
    assert bidder.bid(ctx(hand(*MONSTER), high_bid=42), rng) == PASS


def test_heuristic_never_exceeds_ceiling():
    rng = random.Random(7)
    bidder = HeuristicBidder()
    for _ in range(300):
        h = tuple(sorted(rng.sample(range(28), 7)))
        for high in (0, 30, 33, 38, 41, 42):
            value = bidder.bid(ctx(h, high_bid=high), rng)
            if value == PASS:
                continue
            assert value in legal_bids(high)
            best = best_trump(h, bidder.min_trumps)
            assert best is not None
            assert value <= best.bid_ceiling


def test_heuristic_declares_a_pip_trump():
    rng = random.Random(0)
    bidder = HeuristicBidder()
    assert bidder.declare(hand(*MONSTER), 30, rng) == 5
    assert 0 <= bidder.declare(hand(*GARBAGE), 30, rng) <= 6


def test_bid30_opens_then_passes():
    rng = random.Random(0)
    bidder = Bid30Bidder()
    assert bidder.bid(ctx(hand(*GARBAGE)), rng) == 30
    assert bidder.bid(ctx(hand(*GARBAGE), high_bid=30), rng) == PASS


def test_random_bidder_stays_legal():
    bidder = RandomBidder(p_bid=1.0)
    rng = random.Random(3)
    for high in (0, 35, 41, 42, 84):
        for _ in range(50):
            value = bidder.bid(ctx(hand(*GARBAGE), high_bid=high), rng)
            assert value == PASS or (value in legal_bids(high) and value <= 42)
