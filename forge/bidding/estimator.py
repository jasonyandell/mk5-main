"""Bid evaluation and output formatting.

Computes P(make) for various bid thresholds and formats results.
Uses Wilson score interval for confidence bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from forge.oracle.declarations import DECL_ID_TO_NAME


# All bid thresholds from 30 to 42 (standard bids)
BID_THRESHOLDS = list(range(30, 43))  # 30, 31, 32, ... 42

# Bid thresholds with mark values for expected value calculations
# (threshold, marks_at_stake)
BID_LEVELS = [(t, 1) for t in BID_THRESHOLDS] + [(84, 2)]  # 84 = 2-mark bid


@dataclass
class BidResult:
    """Result for a single trump/bid combination."""

    trump_name: str
    decl_id: int
    bid: int
    marks: int
    p_make: float
    mark_swing: float  # Expected marks: (2*p - 1) * marks
    ci_low: float
    ci_high: float
    n_samples: int


@dataclass
class TrumpResult:
    """Results for all bids with a single trump."""

    trump_name: str
    decl_id: int
    points: List[int]  # Raw point outcomes
    bid_results: List[BidResult]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score confidence interval.

    Args:
        successes: Number of successes
        n: Total trials
        z: Z-score (1.96 for 95% CI)

    Returns:
        (lower, upper) bounds
    """
    if n == 0:
        return (0.0, 1.0)

    p_hat = successes / n
    denom = 1 + z * z / n

    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom

    return (max(0.0, center - margin), min(1.0, center + margin))


def evaluate_bids(points: List[int], decl_id: int) -> TrumpResult:
    """Evaluate all bid levels for a given trump's simulation results.

    Args:
        points: List of team 0's final points from simulations
        decl_id: The declaration/trump ID

    Returns:
        TrumpResult with bid evaluations
    """
    trump_name = DECL_ID_TO_NAME.get(decl_id, f"trump_{decl_id}")
    n = len(points)

    bid_results = []
    for threshold, marks in BID_LEVELS:
        successes = sum(1 for p in points if p >= threshold)
        p_make = successes / n if n > 0 else 0.0
        ci_low, ci_high = wilson_ci(successes, n)

        # Expected mark swing: winning gives +marks, losing gives -marks
        # E[marks] = p_make * marks + (1 - p_make) * (-marks) = (2*p - 1) * marks
        mark_swing = (2 * p_make - 1) * marks

        bid_results.append(
            BidResult(
                trump_name=trump_name,
                decl_id=decl_id,
                bid=threshold,
                marks=marks,
                p_make=p_make,
                mark_swing=mark_swing,
                ci_low=ci_low,
                ci_high=ci_high,
                n_samples=n,
            )
        )

    return TrumpResult(
        trump_name=trump_name,
        decl_id=decl_id,
        points=points,
        bid_results=bid_results,
    )


def find_best_bid(results: List[TrumpResult]) -> Tuple[str, int, float]:
    """Find the best trump/bid combination by mark swing.

    Returns:
        (trump_name, bid, mark_swing)
    """
    best = None
    best_swing = float("-inf")
    best_trump = ""
    best_bid = 30

    for trump_result in results:
        for bid_result in trump_result.bid_results:
            if bid_result.mark_swing > best_swing:
                best_swing = bid_result.mark_swing
                best_trump = bid_result.trump_name
                best_bid = bid_result.bid

    return (best_trump, best_bid, best_swing)


def format_results_table(
    results: List[TrumpResult],
    show_all_bids: bool = False,
) -> str:
    """Format results as a text table.

    Args:
        results: List of TrumpResult from all trumps
        show_all_bids: If True, show all bid levels; else just 30, 36, 42

    Returns:
        Formatted table string
    """
    lines = []

    # Header
    lines.append("Trump       Bid   P(make)   Mark Swing   CI")
    lines.append("─" * 50)

    # Filter and sort by mark swing
    all_bids = []
    for trump_result in results:
        for bid_result in trump_result.bid_results:
            if show_all_bids or bid_result.bid in (30, 36, 42):
                all_bids.append(bid_result)

    # Sort by mark swing descending
    all_bids.sort(key=lambda x: -x.mark_swing)

    for bid in all_bids:
        ci_str = f"[{bid.ci_low:.2f}, {bid.ci_high:.2f}]"
        swing_str = f"{bid.mark_swing:+.2f}" if bid.mark_swing >= 0 else f"{bid.mark_swing:.2f}"
        lines.append(
            f"{bid.trump_name:11} {bid.bid:3}   {bid.p_make:.2f}      {swing_str:>6}       {ci_str}"
        )

    return "\n".join(lines)


def format_matrix(results: List[TrumpResult]) -> str:
    """Format results as a matrix: rows=trumps, columns=bids 30-42.

    Args:
        results: List of TrumpResult from all trumps

    Returns:
        Formatted matrix string with P(make) percentages
    """
    lines = []
    bids = BID_THRESHOLDS  # 30-42

    # Header
    header = f"{'Trump':13}"
    for b in bids:
        header += f" {b:>3}"
    lines.append(header)
    lines.append("-" * len(header))

    # Data rows - compute P(make) directly from points
    for trump_result in results:
        row = f"{trump_result.trump_name:13}"
        n = len(trump_result.points)

        for bid in bids:
            if n == 0:
                row += "   ."
            else:
                successes = sum(1 for p in trump_result.points if p >= bid)
                p = successes / n
                if p >= 0.995:
                    row += " 100"
                elif p < 0.005:
                    row += "   ."
                else:
                    row += f" {p*100:>3.0f}"
        lines.append(row)

    return "\n".join(lines)


def format_json(results: List[TrumpResult]) -> Dict:
    """Format results as JSON-serializable dict."""
    output = {
        "trumps": [],
        "best": None,
    }

    for trump_result in results:
        trump_data = {
            "trump": trump_result.trump_name,
            "decl_id": trump_result.decl_id,
            "mean_points": sum(trump_result.points) / len(trump_result.points) if trump_result.points else 0,
            "bids": [],
        }

        for bid_result in trump_result.bid_results:
            trump_data["bids"].append(
                {
                    "bid": bid_result.bid,
                    "marks": bid_result.marks,
                    "p_make": bid_result.p_make,
                    "mark_swing": bid_result.mark_swing,
                    "ci_low": bid_result.ci_low,
                    "ci_high": bid_result.ci_high,
                    "n_samples": bid_result.n_samples,
                }
            )

        output["trumps"].append(trump_data)

    best_trump, best_bid, best_swing = find_best_bid(results)
    output["best"] = {
        "trump": best_trump,
        "bid": best_bid,
        "mark_swing": best_swing,
    }

    return output
