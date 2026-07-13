"""Bid policies for the arena.

All policies declare a pip trump (decl_id 0-6); doubles/no-trump contracts
await a bidder that can evaluate them (Champion rungs #21-#22).
"""
from __future__ import annotations

import random

from .auction import ONE_MARK, PASS, BidContext
from .hand_metrics import PIP_TRUMPS, best_trump


def _declare_best(hand: tuple[int, ...]) -> int:
    """Best pip trump with no length floor — total, for forced bids too."""
    e = best_trump(hand, min_trumps=0)
    assert e is not None  # min_trumps=0 always yields a candidate
    return e.trump


class HeuristicBidder:
    """Roberson risk-budget bidder: bid only enough, never above ceiling.

    A hand is biddable if some pip trump has at least `min_trumps` tiles;
    its willingness is that trump's bid ceiling (42 minus unique exposed
    count points) less `caution`, capped at one mark. At its turn the
    policy bids the minimum legal raise if that raise is within its
    willingness — the ch02 "bid only enough" discipline (`supported` at
    wave 2.B.2) — and passes otherwise. It never bids marks beyond 42.
    """

    def __init__(self, min_trumps: int = 3, caution: int = 0):
        self.min_trumps = min_trumps
        self.caution = caution

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        best = best_trump(ctx.hand, self.min_trumps)
        if best is None:
            return PASS
        willing = min(ONE_MARK, best.bid_ceiling - self.caution)
        target = min((b for b in ctx.legal if b <= ONE_MARK), default=None)
        if target is None or target > willing:
            return PASS
        return target

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        best = best_trump(hand, self.min_trumps) or best_trump(hand)
        assert best is not None
        return best.trump

    def __repr__(self) -> str:
        return f"HeuristicBidder(min_trumps={self.min_trumps}, caution={self.caution})"


class Bid30Bidder:
    """Opens 30 if nothing has been bid, otherwise passes.

    The arena analogue of the historical forced-bid-30 evals: every hand is
    a 30 contract won by the seat left of the shaker — but with a sane
    declaration instead of a random one.
    """

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        return 30 if ctx.high_bid == 0 else PASS

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        return _declare_best(hand)

    def __repr__(self) -> str:
        return "Bid30Bidder()"


class RandomBidder:
    """Bids a uniform legal point bid with probability `p_bid`, else passes.

    A noise floor for auction comparisons. Declares a uniform pip trump.
    """

    def __init__(self, p_bid: float = 0.3):
        self.p_bid = p_bid

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        choices = [b for b in ctx.legal if b <= ONE_MARK]
        if not choices or rng.random() >= self.p_bid:
            return PASS
        return rng.choice(choices)

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        return rng.choice(PIP_TRUMPS)

    def __repr__(self) -> str:
        return f"RandomBidder(p_bid={self.p_bid})"
