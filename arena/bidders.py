# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Bid policies for the arena.

Every policy declares a pip trump (decl_id 0-6) except `MenuBidder`, the
heuristic bidder with the book's full menu (#155), which may also declare
doubles as trumps (7) or no trump (9). A bid policy that may declare beyond
the pip trumps says so in `declarations`; the engine refuses a table where
such a bidder sits with a play policy that cannot play what it may declare
(`arena.play.check_menu`).
"""
from __future__ import annotations

import random

from forge.oracle.declarations import GAME_DECL_IDS

from .auction import ONE_MARK, PASS, BidContext
from .hand_metrics import PIP_TRUMPS, best_trump, menu_choice


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


class MenuBidder(HeuristicBidder):
    """The heuristic bidder over all nine game declarations (#155): the
    book's Ch. 9 choice (`arena.hand_metrics.menu_choice`) of doubles as
    trumps or no trump where the hand holds four doubles or more and the
    book's condition holds, else exactly `HeuristicBidder` (the pip trump,
    its ceiling, its declaration). Opt-in: `HeuristicBidder` itself never
    leaves the pip trumps, and every registered study bids with it."""

    declarations = GAME_DECL_IDS

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        choice = menu_choice(ctx.hand, self.min_trumps)
        if choice is None:
            return PASS
        willing = min(ONE_MARK, choice.bid_ceiling - self.caution)
        target = min((b for b in ctx.legal if b <= ONE_MARK), default=None)
        if target is None or target > willing:
            return PASS
        return target

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        choice = menu_choice(hand, self.min_trumps)
        return choice.decl if choice is not None else _declare_best(hand)

    def __repr__(self) -> str:
        return f"MenuBidder(min_trumps={self.min_trumps}, caution={self.caution})"


class VariedBidder(HeuristicBidder):
    """The heuristic bidder with its caution drawn afresh at every bid,
    uniformly from `cautions` (#149's varied Zeb): the auction's own rng
    (keyed on the deal) draws it, so a hand's caution is the same in every
    arm that deals that hand. Ceilings move in steps of 5, so the default
    {-5, 0, +5} is three behaviours."""

    def __init__(self, cautions: tuple[int, ...] = (-5, 0, 5), min_trumps: int = 3):
        super().__init__(min_trumps=min_trumps)
        self.cautions = tuple(cautions)

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        self.caution = rng.choice(self.cautions)
        return super().bid(ctx, rng)

    def __repr__(self) -> str:
        return f"VariedBidder(cautions={self.cautions}, min_trumps={self.min_trumps})"


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
