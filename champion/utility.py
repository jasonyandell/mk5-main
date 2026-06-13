"""Marks-to-7 utility: score-conditioned value of a contract (rung #27, v1).

The ICM analogue for Texas 42: at 6-6 the right utility is not the 0-0
utility. The primitive is a win-probability lookup table over game scores,
`race_wp`, built from the simplest defensible model of the rest of the game:
every future hand awards one mark to one team, each with probability 1/2.
Under that race model WP(a, b) — a marks still needed by us, b by them —
satisfies WP(a, b) = (WP(a-1, b) + WP(a, b-1)) / 2 with WP(0, ·) = 1 and
WP(·, 0) = 0, which is Pascal's identity; the recursion *is* the lookup
table the issue asks for.

A bid utility scores one contract decision against the status quo:

- `MarkEV` — score-blind expected mark swing, (2p - 1) * marks. Positive
  iff p > 1/2 at every score. This is Auction v0's default criterion.
- `MarksToSeven` — expected WP after the hand minus WP now. The Pascal
  identity gives U = (p - 1/2) * (WP(a-1, b) - WP(a, b-1)) for 1-mark
  contracts, so on those the score changes the stakes but never the sign:
  bid iff p > 1/2, same as MarkEV. The conditioning bites exactly on
  multi-mark contracts, where the threshold moves with the score — an 84
  needs p > 3/4 when ahead 6-0, p > 1/2 at even, and only p > 1/4 when
  behind 0-6. Prudence and desperation emerge instead of being authored.

v1 limitation, stated honestly: the pass baseline is WP at the current
score under the neutral race model. It does not yet model who bids if we
pass — the defensive value of passing and the cost of handing the auction
to opponents at 6-x is rung #26's equilibrium to find.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from arena.auction import contract_points, marks_for_bid  # noqa: F401 (re-export for callers)


@lru_cache(maxsize=None)
def race_wp(my_needed: int, opp_needed: int, p_hand: float = 0.5) -> float:
    """P(we reach 0 needed before they do) when each hand is a p_hand coin."""
    if my_needed <= 0:
        return 1.0
    if opp_needed <= 0:
        return 0.0
    return (
        p_hand * race_wp(my_needed - 1, opp_needed, p_hand)
        + (1.0 - p_hand) * race_wp(my_needed, opp_needed - 1, p_hand)
    )


class BidUtility(Protocol):
    """Value of bidding a contract, relative to passing; bid iff > 0."""

    def value(
        self,
        p_make: float,
        bid: int,
        *,
        team: int,
        marks: tuple[int, int],
        marks_to_win: int,
    ) -> float:
        ...


class MarkEV:
    """Score-blind expected mark swing: (2p - 1) * marks at stake."""

    def value(
        self,
        p_make: float,
        bid: int,
        *,
        team: int,
        marks: tuple[int, int],
        marks_to_win: int,
    ) -> float:
        return (2.0 * p_make - 1.0) * marks_for_bid(bid)

    def __repr__(self) -> str:
        return "MarkEV()"


class MarksToSeven:
    """Win-probability gain of bidding, under the race-model WP table."""

    def value(
        self,
        p_make: float,
        bid: int,
        *,
        team: int,
        marks: tuple[int, int],
        marks_to_win: int,
    ) -> float:
        my_needed = marks_to_win - marks[team]
        opp_needed = marks_to_win - marks[1 - team]
        m = marks_for_bid(bid)
        wp_make = race_wp(my_needed - m, opp_needed)
        wp_set = race_wp(my_needed, opp_needed - m)
        wp_now = race_wp(my_needed, opp_needed)
        return p_make * wp_make + (1.0 - p_make) * wp_set - wp_now

    def __repr__(self) -> str:
        return "MarksToSeven()"
