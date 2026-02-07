#!/usr/bin/env python3
"""Stability experiment: Compare N=200 vs N=500 vs N=1000 across random hands.

Hypothesis: N=200 is sufficient - rankings match N=1000 baseline.

Usage:
    python -m forge.bidding.stability_experiment --hands 100 --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .estimator import find_best_bid
from .evaluate import format_hand, run_evaluation
from .inference import PolicyModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAMPLE_SIZES = [200, 500, 1000]


@dataclass
class HandResult:
    hand_id: int
    hand_str: str
    n_samples: int
    best_trump: str
    best_bid: int
    p_make: float
    ci_width: float
    elapsed: float


def generate_random_hand(rng: random.Random) -> list[int]:
    """Generate a random 7-domino hand."""
    all_dominoes = list(range(28))
    return sorted(rng.sample(all_dominoes, 7))


def run_experiment(
    n_hands: int,
    output_path: Path,
    seed: int,
) -> None:
    """Run stability experiment across random hands."""

    log.info(f"Running stability experiment: {n_hands} hands × {len(SAMPLE_SIZES)} sample sizes")
    log.info(f"Sample sizes: {SAMPLE_SIZES}")
    log.info(f"Output: {output_path}")

    # Load model once
    log.info("Loading model...")
    model = PolicyModel(compile_model=False)
    log.info(f"Model loaded on {model.device}")

    # Set up RNG for hand generation
    rng = random.Random(seed)

    # Results storage
    results: list[HandResult] = []

    # Track stability
    matches_1000 = {200: 0, 500: 0}

    start_time = time.time()

    for hand_id in range(n_hands):
        hand = generate_random_hand(rng)
        hand_str = format_hand(hand)

        hand_results = {}

        for n in SAMPLE_SIZES:
            trump_results, elapsed = run_evaluation(
                model, hand, n_samples=n, seed=seed + hand_id
            )

            best_trump, best_bid, _ = find_best_bid(trump_results)

            # Find P(make) and CI for best bid
            p_make = 0.0
            ci_width = 0.0
            for tr in trump_results:
                for br in tr.bid_results:
                    if br.trump_name == best_trump and br.bid == best_bid:
                        p_make = br.p_make
                        ci_width = br.ci_high - br.ci_low
                        break

            result = HandResult(
                hand_id=hand_id,
                hand_str=hand_str,
                n_samples=n,
                best_trump=best_trump,
                best_bid=best_bid,
                p_make=p_make,
                ci_width=ci_width,
                elapsed=elapsed,
            )
            results.append(result)
            hand_results[n] = (best_trump, best_bid)

        # Check stability vs N=1000
        baseline = hand_results[1000]
        for n in [200, 500]:
            if hand_results[n] == baseline:
                matches_1000[n] += 1

        # Progress logging
        elapsed_total = time.time() - start_time
        hands_done = hand_id + 1
        rate = hands_done / elapsed_total * 60  # hands per minute
        eta = (n_hands - hands_done) / (hands_done / elapsed_total)

        if hands_done % 10 == 0 or hands_done == n_hands:
            log.info(
                f"Progress: {hands_done}/{n_hands} hands "
                f"({rate:.1f}/min, ETA {eta/60:.1f}min) | "
                f"Stability: N=200 {matches_1000[200]/hands_done:.0%}, "
                f"N=500 {matches_1000[500]/hands_done:.0%}"
            )

    total_time = time.time() - start_time

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'hand_id', 'hand', 'n_samples', 'best_trump', 'best_bid',
            'p_make', 'ci_width', 'elapsed'
        ])
        for r in results:
            writer.writerow([
                r.hand_id, r.hand_str, r.n_samples, r.best_trump, r.best_bid,
                f"{r.p_make:.4f}", f"{r.ci_width:.4f}", f"{r.elapsed:.2f}"
            ])

    # Summary
    log.info("=" * 60)
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 60)
    log.info(f"Total hands: {n_hands}")
    log.info(f"Total time: {total_time/60:.1f} minutes")
    log.info(f"Rate: {n_hands/total_time*60:.1f} hands/minute")
    log.info("")
    log.info("STABILITY (match N=1000 baseline):")
    log.info(f"  N=200:  {matches_1000[200]}/{n_hands} = {matches_1000[200]/n_hands:.1%}")
    log.info(f"  N=500:  {matches_1000[500]}/{n_hands} = {matches_1000[500]/n_hands:.1%}")
    log.info("")
    log.info(f"Results saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stability experiment comparing sample sizes"
    )
    parser.add_argument(
        "--hands", type=int, default=100,
        help="Number of random hands to test (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default="stability_results.csv",
        help="Output CSV path (default: stability_results.csv)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    run_experiment(
        n_hands=args.hands,
        output_path=Path(args.output),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
