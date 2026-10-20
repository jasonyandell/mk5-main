# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Static hand metrics for bid evaluation.

A module-grade port of the wave-2.B validated arithmetic in
`w42/bidding_risk_budget_claim_validation/validate_bidding_risk_budget.py`
(hand_eval). In suit-algebra terms (wiki/topics/suit-algebra-spec.md §2): for a
candidate pip trump t, every non-trump non-double tile in hand is an "off"
whose two sides each expose the unheld count tiles of that side's natural
suit σ_p (excluding tiles called into trump). The bid ceiling proxy is 42
minus the unique exposed count points — Roberson's risk budget.

`test_hand_metrics.py` checks this port against the original hand_eval over
randomized hands.
"""
from __future__ import annotations

from dataclasses import dataclass

from forge.oracle.declarations import DOUBLES_TRUMP, NOTRUMP
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_IS_DOUBLE,
    DOMINOES,
    can_follow,
    domino_contains_pip,
    is_in_called_suit,
    trick_rank,
)

COUNT_TILE_IDS = tuple(i for i, p in enumerate(DOMINO_COUNT_POINTS) if p > 0)
PIP_TRUMPS = tuple(range(7))


@dataclass(frozen=True)
class TrumpEval:
    """Static evaluation of one hand under one candidate pip trump."""

    trump: int
    trump_count: int
    has_trump_double: bool
    held_count_points: int
    unique_exposed_points: int

    @property
    def bid_ceiling(self) -> int:
        """42 minus the count points our offs expose — the risk budget."""
        return 42 - self.unique_exposed_points


def evaluate_trump(hand: tuple[int, ...], trump: int) -> TrumpEval:
    """Evaluate `hand` as if `trump` (a pip 0-6) were declared."""
    hand_set = frozenset(hand)
    trump_double = trump * (trump + 3) // 2  # id of the (trump, trump) tile

    trump_count = sum(1 for d in hand if domino_contains_pip(d, trump))
    offs = [
        d for d in hand
        if not domino_contains_pip(d, trump) and not DOMINO_IS_DOUBLE[d]
    ]

    exposed: set[int] = set()
    for off in offs:
        for side in DOMINOES[off]:
            for count_id in COUNT_TILE_IDS:
                if count_id in hand_set:
                    continue
                if domino_contains_pip(count_id, trump):
                    continue
                if domino_contains_pip(count_id, side):
                    exposed.add(count_id)

    return TrumpEval(
        trump=trump,
        trump_count=trump_count,
        has_trump_double=trump_double in hand_set,
        held_count_points=sum(DOMINO_COUNT_POINTS[d] for d in hand),
        unique_exposed_points=sum(DOMINO_COUNT_POINTS[d] for d in exposed),
    )


def best_trump(hand: tuple[int, ...], min_trumps: int = 0) -> TrumpEval | None:
    """The strongest pip-trump evaluation, or None if no trump reaches
    `min_trumps` tiles. Ordered by risk budget, then trump length, then the
    trump double, then held count; pip value breaks remaining ties."""
    candidates = [
        e for e in (evaluate_trump(hand, t) for t in PIP_TRUMPS)
        if e.trump_count >= min_trumps
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda e: (
            e.bid_ceiling,
            e.trump_count,
            e.has_trump_double,
            e.held_count_points,
            e.trump,
        ),
    )


# -- The book's Ch. 9 menu: doubles as trumps and no trumps (#155).
#
# The pip-trump arithmetic above, read through a non-pip declaration's own
# suit algebra (forge.oracle.tables, the rules authority). Under doubles
# trump the doubles leave their suits, so every non-double is an off and the
# six-end tops each other suit (W42-p075-b); under no trump nothing is trump
# and each double tops its own suit (W42-p079-d). The book prices these hands
# by what the declaration protects (Hands 23-25: an off that tops a suit is
# "as good as a double"), so an off's side p exposes nothing when the viewer
# holds the top tile of suit p under the declaration (the off itself may be
# it). That protection is our reading, applied only to the Ch. 9
# declarations: the pip arithmetic is the arena's unchanged.

SIX_SIX = DOMINOES.index((6, 6))
FIVE_FIVE = DOMINOES.index((5, 5))
# An off high in a four, five or six suit (W42-p078-b): both pips 4 or more.
HIGH_OFFS = frozenset(i for i, (h, l) in enumerate(DOMINOES) if h != l and l >= 4)
CH9_MIN_DOUBLES = 4  # "usually four or five" doubles (W42-p075-b); evaluate at four (W42-p079-a)


def suit_top(pip: int, decl: int) -> int:
    """The highest tile that follows suit `pip` under declaration `decl`."""
    return max(
        (t for t in range(len(DOMINOES)) if can_follow(t, pip, decl)),
        key=lambda t: trick_rank(t, pip, decl),
    )


def ch9_ceiling(hand: tuple[int, ...], decl: int) -> int:
    """42 minus the count points exposed under doubles trump or no trump:
    each off's side p exposes the unheld, non-trump count tiles of suit p,
    unless the viewer holds suit p's top tile under `decl`."""
    if decl not in (DOUBLES_TRUMP, NOTRUMP):
        raise ValueError(f"ch9_ceiling prices doubles trump or no trump, not {decl}")
    hand_set = frozenset(hand)
    exposed: set[int] = set()
    for off in (d for d in hand if not DOMINO_IS_DOUBLE[d]):
        for side in DOMINOES[off]:
            if suit_top(side, decl) in hand_set:
                continue
            exposed.update(
                c for c in COUNT_TILE_IDS
                if c not in hand_set
                and not is_in_called_suit(c, decl)
                and domino_contains_pip(c, side)
            )
    return 42 - sum(DOMINO_COUNT_POINTS[d] for d in exposed)


def ch9_doubles_loss(hand: tuple[int, ...]) -> int:
    """The book's surface count of what a doubles-trump hand can lose, in
    points, before it plays the line in its head (W42 pp. 76-81, its loss
    tallies on Hands 23-25):

    - trumps: nothing with the 6-6 and 5-5; with the 6-6 alone, the 5-5's
      trick and its ten; with no 6-6, the first trick (the book adds a count
      it cannot name, left out here);
    - each off not its suit's top on some side: one trick plus the most count
      it can lose there, its own count or an unheld count tile of that side.

    The book's second step, the play-out that can overrule this (Hand 23's
    22 points of risk become a cinch 31, W42-p077), is a line, not a formula.
    No player reads this yet (#230)."""
    hand_set = frozenset(hand)
    if SIX_SIX not in hand_set:
        trumps = 1
    else:
        trumps = 0 if FIVE_FIVE in hand_set else 1 + DOMINO_COUNT_POINTS[FIVE_FIVE]
    offs = 0
    for off in (d for d in hand if not DOMINO_IS_DOUBLE[d]):
        open_sides = [s for s in DOMINOES[off] if suit_top(s, DOUBLES_TRUMP) not in hand_set]
        if open_sides:
            offs += 1 + max([DOMINO_COUNT_POINTS[off]] + [
                DOMINO_COUNT_POINTS[c] for c in COUNT_TILE_IDS
                if c not in hand_set
                and not is_in_called_suit(c, DOUBLES_TRUMP)
                and any(domino_contains_pip(c, s) for s in open_sides)
            ])
    return trumps + offs


@dataclass(frozen=True)
class MenuEval:
    """The declaration a hand calls from the full menu, and its risk budget."""

    decl: int
    bid_ceiling: int


def menu_choice(hand: tuple[int, ...], min_trumps: int = 3) -> MenuEval | None:
    """The book's Ch. 9 choice over all nine declarations, or None where
    the hand would bid no suit (a pip suit short of `min_trumps` and no Ch.
    9 declaration). Below four doubles it is exactly the arena's pip choice
    (`best_trump`). From four doubles (W42-p075-b, W42-p079-a):

    - doubles trump when the pip trump (one reaching `min_trumps`) leaves a
      high off, a four, five or six (W42-p078-b: the off that makes normal
      trumps too risky), its ceiling the better of the two prices, since
      the option only adds (Hands 23, 24); a high off that is a trump of
      the suit is no off (the book's Hands 26 and 27 call fours over their
      6-4);
    - no trump when the hand holds neither the highest double nor a high
      off (W42-p080-a) and no trump prices above every pip suit that
      reaches `min_trumps` (the book bids such a hand in a suit when it
      has one: Hands 2, 13);
    - else the pip trump.
    """
    pip = best_trump(hand, min_trumps)
    pip_ceiling = pip.bid_ceiling if pip is not None else None
    doubles = sum(1 for d in hand if DOMINO_IS_DOUBLE[d])
    if doubles >= CH9_MIN_DOUBLES:
        offs = () if pip is None else [
            d for d in hand if not domino_contains_pip(d, pip.trump) and not DOMINO_IS_DOUBLE[d]
        ]
        if any(d in HIGH_OFFS for d in offs):
            return MenuEval(DOUBLES_TRUMP, max(ch9_ceiling(hand, DOUBLES_TRUMP), pip_ceiling))
        if SIX_SIX not in hand and not any(d in HIGH_OFFS for d in hand):
            nt = ch9_ceiling(hand, NOTRUMP)
            if pip_ceiling is None or nt > pip_ceiling:
                return MenuEval(NOTRUMP, nt)
    return None if pip is None else MenuEval(pip.trump, pip.bid_ceiling)
