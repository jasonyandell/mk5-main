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

v2 (2026-06-12): the pass baseline is parameterizable. Default is still WP at
the current score (the neutral race model); with ``pass_q_opp > 0`` it credits
the defensive cost of handing the auction to opponents — a first, calibrated
step toward rung #26's full equilibrium, which would derive who bids from
self-play rather than a scalar. ``score_to_utility`` extends the same
race-model conditioning to play-phase risk.
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
    """Win-probability gain of bidding, under the race-model WP table.

    The pass alternative is valued at ``wp_pass``. By default (``pass_q_opp=0``)
    that is the status-quo WP at the current score — passing freezes the race,
    the v1 baseline. With ``pass_q_opp > 0`` it becomes equilibrium-aware:
    passing hands the auction on, an opponent declares a one-mark contract with
    probability ``pass_q_opp`` and makes it with probability ``pass_make_rate``
    (they gain a mark) or is set (we gain a mark). Crediting that defensive cost
    is rung #27's v2 step toward the #26 equilibrium — passing into a strong
    field at 6-x becomes correctly worse than the neutral 0, so the bidder
    fights harder to deny opponents the auction when behind.
    """

    def __init__(self, pass_q_opp: float = 0.0, pass_make_rate: float = 0.5):
        self.pass_q_opp = pass_q_opp
        self.pass_make_rate = pass_make_rate

    def _wp_pass(self, my_needed: int, opp_needed: int) -> float:
        """WP if we pass: status quo, unless an opponent takes the contract."""
        q, r = self.pass_q_opp, self.pass_make_rate
        if q <= 0.0:
            return race_wp(my_needed, opp_needed)
        opp_takes = q * (
            r * race_wp(my_needed, opp_needed - 1)            # they make → they gain
            + (1.0 - r) * race_wp(my_needed - 1, opp_needed)  # they're set → we gain
        )
        no_take = (1.0 - q) * race_wp(my_needed, opp_needed)
        return opp_takes + no_take

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
        wp_pass = self._wp_pass(my_needed, opp_needed)
        return p_make * wp_make + (1.0 - p_make) * wp_set - wp_pass

    def __repr__(self) -> str:
        if self.pass_q_opp <= 0.0:
            return "MarksToSeven()"
        return (
            f"MarksToSeven(pass_q_opp={self.pass_q_opp}, "
            f"pass_make_rate={self.pass_make_rate})"
        )


def score_to_utility(
    marks: tuple[int, int],
    team: int,
    marks_to_win: int = 7,
    *,
    band: float = 0.15,
    ahead: str = "cvar_10",
    even: str = "ev",
    behind: str = "upside_10",
) -> str:
    """Pick a play-risk lens by score: protect a lead, chase from behind.

    The acting team's win probability under the neutral race model decides the
    risk posture. Comfortably ahead (WP >= 1/2 + band) the right move is to
    protect the lead — minimize the downside that could squander it — so play
    the lower-tail-averse ``cvar_10``. Comfortably behind (WP <= 1/2 - band)
    only the rare big outcome changes the game, so chase variance with
    ``upside_10``. Near even, maximize expected value (``ev``). The band keeps
    the default ev posture across the broad middle where risk-shaping is noise.

    This is the play-phase analogue of MarksToSeven's bid-phase conditioning:
    both read the same ``race_wp`` table, so the champion's play risk and bid
    risk move together rather than as two ad-hoc policies.
    """
    my_needed = marks_to_win - marks[team]
    opp_needed = marks_to_win - marks[1 - team]
    wp = race_wp(my_needed, opp_needed)
    if wp >= 0.5 + band:
        return ahead
    if wp <= 0.5 - band:
        return behind
    return even
