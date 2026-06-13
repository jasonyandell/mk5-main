"""Static hand metrics for bid evaluation.

A module-grade port of the wave-2.B validated arithmetic in
`w42/bidding_risk_budget_claim_validation/validate_bidding_risk_budget.py`
(hand_eval). In suit-algebra terms (docs/theory/SUIT_ALGEBRA.md §2): for a
candidate pip trump t, every non-trump non-double tile in hand is an "off"
whose two sides each expose the unheld count tiles of that side's natural
suit σ_p (excluding tiles called into trump). The bid ceiling proxy is 42
minus the unique exposed count points — Roberson's risk budget.

`test_hand_metrics.py` checks this port against the original hand_eval over
randomized hands.
"""
from __future__ import annotations

from dataclasses import dataclass

from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_IS_DOUBLE,
    DOMINOES,
    domino_contains_pip,
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
