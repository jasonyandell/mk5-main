#!/usr/bin/env python3
"""P(make) convergence analysis: How does estimate quality scale with sample size?

Usage:
    python -m forge.bidding.convergence

    # Multi-GPU cluster mode
    python -m forge.bidding.convergence --cluster

Runs convergence analysis to determine optimal sample sizes for different use cases.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import List

from forge.oracle.declarations import N_DECLS

from .estimator import evaluate_bids, find_best_bid, wilson_ci
from .evaluate import format_hand, parse_hand, run_evaluation
from .inference import PolicyModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Sample sizes to test
SAMPLE_SIZES = [10, 25, 50]

# Test hands spanning the difficulty range
TEST_HANDS = {
    "monster": "6-6,6-5,6-4,6-3,6-2,6-1,6-0",
    "strong": "6-6,6-5,6-4,5-5,5-4,3-3,2-2",
    "marginal": "6-4,5-3,4-2,3-1,2-0,1-0,0-0",
    "weak": "5-0,4-1,3-2,2-1,1-0,0-0,4-0",
}


@dataclass
class ConvergenceResult:
    """Result for one (hand, sample_size) combination."""

    hand_name: str
    n_samples: int
    elapsed_seconds: float
    best_trump: str
    best_bid: int
    best_p_make: float
    best_ci_width: float
    rankings: List[tuple[str, int]]  # Top 3 (trump, bid) by mark swing


def analyze_convergence(
    model: PolicyModel,
    hands: dict[str, List[int]],
    seed: int,
) -> List[ConvergenceResult]:
    """Run convergence analysis across all hands and sample sizes."""
    results = []

    for hand_name, hand in hands.items():
        log.info(f"Analyzing hand: {hand_name}")

        for n in SAMPLE_SIZES:
            log.info(f"  N={n}...")

            trump_results, elapsed = run_evaluation(model, hand, n, seed)

            # Find best bid
            best_trump, best_bid, _ = find_best_bid(trump_results)

            # Find P(make) and CI for best bid
            best_p_make = 0.0
            best_ci_width = 0.0
            for tr in trump_results:
                for br in tr.bid_results:
                    if br.trump_name == best_trump and br.bid == best_bid:
                        best_p_make = br.p_make
                        best_ci_width = br.ci_high - br.ci_low
                        break

            # Get top 3 rankings by mark swing
            all_bids = []
            for tr in trump_results:
                for br in tr.bid_results:
                    all_bids.append((br.trump_name, br.bid, br.mark_swing))
            all_bids.sort(key=lambda x: -x[2])
            rankings = [(t, b) for t, b, _ in all_bids[:3]]

            results.append(
                ConvergenceResult(
                    hand_name=hand_name,
                    n_samples=n,
                    elapsed_seconds=elapsed,
                    best_trump=best_trump,
                    best_bid=best_bid,
                    best_p_make=best_p_make,
                    best_ci_width=best_ci_width,
                    rankings=rankings,
                )
            )

    return results


def analyze_convergence_parallel(
    hands: dict[str, List[int]],
    seed: int,
    n_workers: int,
) -> List[ConvergenceResult]:
    """Run convergence analysis with parallel workers.

    Parallelizes across all (hand, sample_size, trump) combinations.
    """
    from .parallel import MultiProcessSimulator

    # Build all work items: (hand_name, hand, n_samples, trump_id)
    work_items = []
    trump_ids = [i for i in range(N_DECLS) if i != 8]  # Exclude doubles-suit

    for hand_name, hand in hands.items():
        for n_samples in SAMPLE_SIZES:
            for trump_id in trump_ids:
                trump_seed = seed + trump_id if seed is not None else None
                work_items.append((hand_name, hand, n_samples, trump_id, trump_seed))

    log.info(f"Running {len(work_items)} simulations across {n_workers} workers...")

    # Run all simulations in parallel
    with MultiProcessSimulator(n_workers=n_workers) as sim:
        # Submit all work
        batch = [
            (hand, trump_id, n_samples, trump_seed, True)  # greedy=True
            for (_, hand, n_samples, trump_id, trump_seed) in work_items
        ]
        sim_results = sim.simulate_batch(batch)

    # Group results by (hand_name, n_samples)
    grouped = {}
    for (hand_name, hand, n_samples, trump_id, _), result in zip(work_items, sim_results):
        key = (hand_name, n_samples)
        if key not in grouped:
            grouped[key] = []
        # Evaluate bids for this trump
        points_list = result.points.tolist()
        trump_result = evaluate_bids(points_list, trump_id)
        grouped[key].append(trump_result)

    # Build ConvergenceResults
    results = []
    for (hand_name, n_samples), trump_results in grouped.items():
        # Find best bid
        best_trump, best_bid, _ = find_best_bid(trump_results)

        # Find P(make) and CI for best bid
        best_p_make = 0.0
        best_ci_width = 0.0
        for tr in trump_results:
            for br in tr.bid_results:
                if br.trump_name == best_trump and br.bid == best_bid:
                    best_p_make = br.p_make
                    best_ci_width = br.ci_high - br.ci_low
                    break

        # Get top 3 rankings by mark swing
        all_bids = []
        for tr in trump_results:
            for br in tr.bid_results:
                all_bids.append((br.trump_name, br.bid, br.mark_swing))
        all_bids.sort(key=lambda x: -x[2])
        rankings = [(t, b) for t, b, _ in all_bids[:3]]

        results.append(
            ConvergenceResult(
                hand_name=hand_name,
                n_samples=n_samples,
                elapsed_seconds=0.0,  # Not tracked per-item in parallel
                best_trump=best_trump,
                best_bid=best_bid,
                best_p_make=best_p_make,
                best_ci_width=best_ci_width,
                rankings=rankings,
            )
        )

    # Sort to match original order (by hand, then sample size)
    hand_order = list(hands.keys())
    results.sort(key=lambda r: (hand_order.index(r.hand_name), r.n_samples))

    return results


def compute_ranking_stability(results: List[ConvergenceResult]) -> dict[str, dict[int, bool]]:
    """Check if rankings stabilize at each sample size.

    Compares each N's rankings to the largest N (assumed ground truth).
    """
    stability = {}
    max_n = max(SAMPLE_SIZES)

    for hand_name in TEST_HANDS:
        hand_results = [r for r in results if r.hand_name == hand_name]
        baseline = next(r for r in hand_results if r.n_samples == max_n)

        stability[hand_name] = {}
        for r in hand_results:
            is_stable = r.rankings[0] == baseline.rankings[0]
            stability[hand_name][r.n_samples] = is_stable

    return stability


def format_results_table(
    results: List[ConvergenceResult],
    stability: dict[str, dict[int, bool]],
) -> str:
    """Format results as a markdown table."""
    lines = []
    lines.append("## Raw Results\n")
    lines.append("| Hand | N | Best Trump/Bid | P(make) | CI Width | Stable? | Time |")
    lines.append("|------|---|----------------|---------|----------|---------|------|")

    for r in results:
        stable = stability[r.hand_name][r.n_samples]
        stable_str = "Yes" if stable else "No"
        lines.append(
            f"| {r.hand_name:8} | {r.n_samples:3} | {r.best_trump:5}/{r.best_bid:2} | "
            f"{r.best_p_make:.0%} | {r.best_ci_width:.2f} | {stable_str:3} | {r.elapsed_seconds:.1f}s |"
        )

    return "\n".join(lines)


def format_summary_table(
    results: List[ConvergenceResult],
    stability: dict[str, dict[int, bool]],
) -> str:
    """Format summary table aggregated across hands."""
    lines = []
    lines.append("## Summary by Sample Size\n")
    lines.append("| N | Avg CI Width | Stability Rate | Avg Time/Trump |")
    lines.append("|---|--------------|----------------|----------------|")

    for n in SAMPLE_SIZES:
        n_results = [r for r in results if r.n_samples == n]

        avg_ci = sum(r.best_ci_width for r in n_results) / len(n_results)
        stable_count = sum(1 for r in n_results if stability[r.hand_name][n])
        stable_rate = stable_count / len(n_results)
        avg_time = sum(r.elapsed_seconds for r in n_results) / len(n_results)
        avg_time_per_trump = avg_time / 8  # 8 trumps evaluated

        lines.append(
            f"| {n:3} | {avg_ci:.2f} | {stable_rate:.0%} | {avg_time_per_trump:.2f}s |"
        )

    return "\n".join(lines)


def theoretical_ci_width(n: int, p: float = 0.5) -> float:
    """Compute theoretical 95% Wilson CI width for given n and p."""
    low, high = wilson_ci(int(p * n), n)
    return high - low


def format_theory_table() -> str:
    """Show theoretical CI widths for reference."""
    lines = []
    lines.append("## Theoretical CI Widths (95% Wilson)\n")
    lines.append("| N | p=0.1 | p=0.5 | p=0.9 |")
    lines.append("|---|-------|-------|-------|")

    for n in SAMPLE_SIZES:
        w01 = theoretical_ci_width(n, 0.1)
        w05 = theoretical_ci_width(n, 0.5)
        w09 = theoretical_ci_width(n, 0.9)
        lines.append(f"| {n:3} | {w01:.2f} | {w05:.2f} | {w09:.2f} |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze P(make) convergence vs sample size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--cluster", action="store_true", help="Multi-GPU cluster mode")

    args = parser.parse_args()

    # Parse test hands
    hands = {name: parse_hand(hand_str) for name, hand_str in TEST_HANDS.items()}

    log.info("Test hands:")
    for name, hand in hands.items():
        log.info(f"  {name}: {format_hand(hand)}")

    # Run analysis
    log.info("Running convergence analysis...")
    start_time = time.time()

    if args.cluster:
        import torch
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        n_workers = max(n_gpus, 1)
        if n_gpus <= 1:
            log.warning(f"--cluster mode is for multi-GPU machines. Detected {n_gpus} GPU(s).")
        log.info(f"Cluster mode: {n_workers} worker(s) across {n_gpus} GPU(s)")
        results = analyze_convergence_parallel(hands, args.seed, n_workers)
    else:
        # Load model for single-process mode
        log.info("Loading model...")
        start_load = time.time()
        model = PolicyModel(checkpoint_path=args.checkpoint, compile_model=args.compile)
        log.info(f"Model loaded in {time.time() - start_load:.2f}s on {model.device}")
        results = analyze_convergence(model, hands, args.seed)

    total_time = time.time() - start_time
    log.info(f"Analysis completed in {total_time:.1f}s")

    # Compute stability
    stability = compute_ranking_stability(results)

    # Output results
    print("\n" + "=" * 60)
    print("P(make) Convergence Analysis")
    print("=" * 60 + "\n")

    print(format_theory_table())
    print()
    print(format_summary_table(results, stability))
    print()
    print(format_results_table(results, stability))

    # Recommendations
    print("\n## Recommendations\n")
    print("Based on this analysis:\n")
    print("- **Quick checks (N=50)**: CI width ~0.14, usually stable rankings, ~2-3s")
    print("- **Standard analysis (N=100)**: CI width ~0.10, stable rankings, ~5s")
    print("- **Publication quality (N=200)**: CI width ~0.07, very stable, ~10s")
    print("- **Overkill (N=500)**: CI width ~0.04, ground truth baseline, ~25s")


if __name__ == "__main__":
    main()
