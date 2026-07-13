"""Auction rules and marks scoring tests."""
import random

import pytest

from arena.auction import (
    PASS,
    AuctionResult,
    BidContext,
    bidding_order,
    contract_points,
    legal_bids,
    marks_for_bid,
    run_auction,
    score_hand,
)


class ScriptedBidder:
    """Bids a fixed value at its turn; declares a fixed decl."""

    def __init__(self, value: int, decl: int = 5):
        self.value = value
        self.decl = decl

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        return self.value

    def declare(self, hand, bid, rng) -> int:
        return self.decl


def _deal() -> tuple[tuple[int, ...], ...]:
    ids = list(range(28))
    return tuple(tuple(ids[i * 7:(i + 1) * 7]) for i in range(4))


def test_legal_bids_progression():
    assert legal_bids(0) == tuple(range(30, 43)) + (84,)
    assert legal_bids(30) == tuple(range(31, 43)) + (84,)
    assert legal_bids(41) == (42, 84)
    assert legal_bids(42) == (84,)
    assert legal_bids(84) == (126,)
    assert legal_bids(126) == (168,)


def test_marks_and_contract_points():
    assert [marks_for_bid(b) for b in (30, 41, 42, 84, 126)] == [1, 1, 1, 2, 3]
    assert [contract_points(b) for b in (30, 41, 42, 84)] == [30, 41, 42, 42]


def test_bidding_order_left_of_shaker_first():
    assert bidding_order(0) == (1, 2, 3, 0)
    assert bidding_order(2) == (3, 0, 1, 2)


def test_score_hand():
    made = score_hand(30, 0, (30, 12))
    assert made.made and made.marks == (1, 0)
    edge = score_hand(35, 1, (7, 35))
    assert edge.made and edge.marks == (0, 1)
    setback = score_hand(35, 1, (8, 34))
    assert not setback.made and setback.marks == (1, 0)
    # A two-mark contract needs all 42 and swings 2 marks either way
    assert score_hand(84, 0, (42, 0)).marks == (2, 0)
    assert score_hand(84, 0, (41, 1)).marks == (0, 2)


def test_run_auction_highest_bid_wins():
    # dealer=0: order 1,2,3,0 — seat 1 opens 30, seat 2 raises to 31
    policies = (
        ScriptedBidder(PASS),
        ScriptedBidder(30),
        ScriptedBidder(31, decl=6),
        ScriptedBidder(PASS),
    )
    result = run_auction(_deal(), 0, policies, random.Random(0))
    assert isinstance(result, AuctionResult)
    assert result.winner == 2
    assert result.high_bid == 31
    assert result.decl_id == 6
    assert result.bids == (0, 30, 31, 0)
    assert not result.forced


def test_run_auction_rejects_illegal_bid():
    policies = (
        ScriptedBidder(PASS),
        ScriptedBidder(35),
        ScriptedBidder(32),  # below the standing 35
        ScriptedBidder(PASS),
    )
    with pytest.raises(ValueError, match="Illegal bid"):
        run_auction(_deal(), 0, policies, random.Random(0))


def test_run_auction_all_pass_returns_none():
    policies = tuple(ScriptedBidder(PASS) for _ in range(4))
    assert run_auction(_deal(), 0, policies, random.Random(0)) is None


def test_run_auction_force_shaker():
    policies = tuple(ScriptedBidder(PASS, decl=3) for _ in range(4))
    result = run_auction(_deal(), 2, policies, random.Random(0), force_shaker=True)
    assert result is not None
    assert result.forced
    assert result.winner == 2  # the shaker
    assert result.high_bid == 30
    assert result.decl_id == 3


def test_run_auction_force_shaker_inert_when_someone_bids():
    policies = (
        ScriptedBidder(PASS),
        ScriptedBidder(40),
        ScriptedBidder(PASS),
        ScriptedBidder(PASS),
    )
    result = run_auction(_deal(), 0, policies, random.Random(0), force_shaker=True)
    assert result.winner == 1 and result.high_bid == 40 and not result.forced
