"""Auction rules and marks scoring (wiki/topics/rules-of-42.md §Bidding, §Scoring).

One round of bidding: the player left of the shaker bids first, clockwise,
each player bidding exactly once — pass, or a value above the current high
bid. Point bids run 30–41; 42 is one mark; 84 (two marks) may be bid any
time two marks has not already been bid (it is also the maximum opening);
above two marks, bids rise one mark at a time. Plunge and special contracts
(nello, sevens) are out of scope for v0.

If all four players pass, the hand is reshaken with the next player as
shaker (tournament rule). The engine bounds reshakes and then applies the
common variation: the shaker must open at 30.

Marks: a successful bid awards max(1, bid // 42) marks to the bidding team;
a set awards the same marks to the defenders. A mark bid (42 or above)
requires all 42 points.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

PASS = 0
MIN_BID = 30
ONE_MARK = 42
TWO_MARKS = 84


def marks_for_bid(bid: int) -> int:
    """Marks at stake for a contract: 1 for point bids, bid // 42 above."""
    return max(1, bid // ONE_MARK)


def contract_points(bid: int) -> int:
    """Points the bidding team must take: the bid, capped at all 42."""
    return min(bid, ONE_MARK)


def legal_bids(high_bid: int) -> tuple[int, ...]:
    """Legal bid values above `high_bid` (0 = nothing bid yet).

    Below two marks, any remaining point bid plus 84 is legal ("any player
    may bid up to 2 marks when 2 marks has not already been bid"). At or
    above two marks, only one additional mark.
    """
    if high_bid >= TWO_MARKS:
        return (high_bid + ONE_MARK,)
    point_bids = tuple(range(max(MIN_BID, high_bid + 1), ONE_MARK + 1))
    return point_bids + (TWO_MARKS,)


def bidding_order(dealer: int) -> tuple[int, int, int, int]:
    """Seats in bid order: left of the shaker first, shaker last."""
    return tuple((dealer + i) % 4 for i in (1, 2, 3, 0))


@dataclass(frozen=True)
class BidContext:
    """Everything a bid policy may condition on at its one turn to speak."""

    hand: tuple[int, ...]
    seat: int
    dealer: int
    bids: tuple[int, ...]  # seat-order; -1 = has not bid yet, 0 = passed
    high_bid: int  # 0 if nothing bid yet
    high_seat: int  # -1 if nothing bid yet
    legal: tuple[int, ...]  # legal bid values; PASS is always allowed
    marks: tuple[int, int] = (0, 0)  # absolute (team0, team1) game score
    marks_to_win: int = 7

    @property
    def team(self) -> int:
        return self.seat % 2


class BidPolicy(Protocol):
    """A bidder: one bid decision per auction, plus trump declaration."""

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        """Return PASS or a value from ctx.legal."""
        ...

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        """Return the decl_id (0-9) to play the won contract under."""
        ...


@dataclass(frozen=True)
class AuctionResult:
    bids: tuple[int, int, int, int]  # seat-order bid values; PASS = 0
    winner: int
    high_bid: int
    decl_id: int
    forced: bool  # shaker was forced to open 30


def run_auction(
    hands: tuple[tuple[int, ...], ...],
    dealer: int,
    policies: tuple[BidPolicy, BidPolicy, BidPolicy, BidPolicy],
    rng: random.Random,
    *,
    force_shaker: bool = False,
    marks: tuple[int, int] = (0, 0),
    marks_to_win: int = 7,
) -> AuctionResult | None:
    """Run one round of bidding. Returns None if all four players pass.

    With force_shaker=True, a shaker who would pass into a dead auction is
    forced to open at 30 (the rules.md "common variation"), making the
    auction total.
    """
    bids = [-1, -1, -1, -1]
    high_bid, high_seat = 0, -1
    forced = False

    for seat in bidding_order(dealer):
        legal = legal_bids(high_bid)
        ctx = BidContext(
            hand=hands[seat],
            seat=seat,
            dealer=dealer,
            bids=tuple(bids),
            high_bid=high_bid,
            high_seat=high_seat,
            legal=legal,
            marks=marks,
            marks_to_win=marks_to_win,
        )
        value = policies[seat].bid(ctx, rng)
        if value == PASS and force_shaker and seat == dealer and high_bid == 0:
            value = MIN_BID
            forced = True
        if value != PASS:
            if value not in legal:
                raise ValueError(
                    f"Illegal bid {value} from seat {seat} "
                    f"(high {high_bid}, legal {legal})"
                )
            high_bid, high_seat = value, seat
        bids[seat] = value

    if high_seat == -1:
        return None

    decl_id = policies[high_seat].declare(hands[high_seat], high_bid, rng)
    if not 0 <= decl_id <= 9:
        raise ValueError(f"Illegal declaration {decl_id} from seat {high_seat}")

    return AuctionResult(
        bids=tuple(bids),
        winner=high_seat,
        high_bid=high_bid,
        decl_id=decl_id,
        forced=forced,
    )


@dataclass(frozen=True)
class HandScore:
    """Marks awarded for one completed hand."""

    marks: tuple[int, int]  # (team0, team1)
    made: bool


def score_hand(bid: int, bidder_team: int, team_points: tuple[int, int]) -> HandScore:
    """Award marks: the bid's marks go to the bidders if they took the
    contract points, otherwise to the defenders (the set)."""
    made = team_points[bidder_team] >= contract_points(bid)
    m = marks_for_bid(bid)
    marks = [0, 0]
    marks[bidder_team if made else 1 - bidder_team] = m
    return HandScore(marks=(marks[0], marks[1]), made=made)
