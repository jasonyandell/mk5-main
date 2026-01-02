#!/usr/bin/env python3
"""CLI for bidding evaluation via game simulation.

Usage:
    python -m forge.bidding.evaluate --hand "6-4,5-5,4-2,3-1,2-0,1-1,0-0" --samples 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import List

import torch

from forge.oracle.declarations import DECL_ID_TO_NAME, N_DECLS
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW

from .estimator import evaluate_bids, find_best_bid, format_json, format_matrix, format_results_table
from .inference import PolicyModel
from .simulator import simulate_games


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_hand(hand_str: str) -> List[int]:
    """Parse hand string like '6-4,5-5,4-2,3-1,2-0,1-1,0-0' to domino IDs.

    Args:
        hand_str: Comma-separated high-low pairs

    Returns:
        List of 7 global domino IDs
    """
    # Build lookup from (high, low) to domino ID
    dom_lookup = {}
    for dom_id in range(28):
        high = DOMINO_HIGH[dom_id]
        low = DOMINO_LOW[dom_id]
        dom_lookup[(high, low)] = dom_id

    parts = hand_str.strip().split(",")
    if len(parts) != 7:
        raise ValueError(f"Expected 7 dominoes, got {len(parts)}")

    hand = []
    for part in parts:
        part = part.strip()
        if "-" in part:
            high, low = part.split("-")
            high, low = int(high), int(low)
        else:
            # Handle format like "64" -> (6, 4)
            if len(part) == 2:
                high, low = int(part[0]), int(part[1])
            else:
                raise ValueError(f"Cannot parse domino: {part}")

        # Normalize order (high >= low)
        if high < low:
            high, low = low, high

        key = (high, low)
        if key not in dom_lookup:
            raise ValueError(f"Invalid domino: {high}-{low}")

        dom_id = dom_lookup[key]
        if dom_id in hand:
            raise ValueError(f"Duplicate domino: {high}-{low}")

        hand.append(dom_id)

    return hand


def format_hand(hand: List[int]) -> str:
    """Format hand for display."""
    parts = []
    for dom_id in hand:
        high = DOMINO_HIGH[dom_id]
        low = DOMINO_LOW[dom_id]
        parts.append(f"{high}-{low}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bidding strength via game simulation"
    )
    parser.add_argument(
        "--hand",
        required=True,
        help='7 dominoes as "high-low" pairs, comma-separated (e.g., "6-4,5-5,4-2,3-1,2-0,1-1,0-0")',
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Simulations per trump (default 100, so 900 total games)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (uses default if not specified)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample from policy distribution instead of greedy (introduces blunders)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show sorted list format instead of matrix",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (experimental, may not work)",
    )
    parser.add_argument(
        "--trumps",
        type=str,
        default=None,
        help="Comma-separated trump IDs to evaluate (default: all 9, excluding doubles-suit)",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse hand
    try:
        hand = parse_hand(args.hand)
    except ValueError as e:
        log.error(f"Invalid hand: {e}")
        sys.exit(1)

    log.info(f"Hand: {format_hand(hand)}")
    log.info(f"Samples per trump: {args.samples}")

    # Determine which trumps to evaluate
    if args.trumps:
        trump_ids = [int(t.strip()) for t in args.trumps.split(",")]
    else:
        # All except doubles-suit (8) which is rarely used
        trump_ids = [i for i in range(N_DECLS) if i != 8]

    log.info(f"Trumps: {[DECL_ID_TO_NAME[t] for t in trump_ids]}")

    # Load model
    log.info("Loading model...")
    start_load = time.time()
    model = PolicyModel(checkpoint_path=args.checkpoint, compile_model=args.compile)
    log.info(f"Model loaded in {time.time() - start_load:.2f}s on {model.device}")

    # Run simulations
    results = []
    total_games = 0
    start_sim = time.time()

    for decl_id in trump_ids:
        trump_name = DECL_ID_TO_NAME.get(decl_id, f"trump_{decl_id}")
        log.debug(f"Simulating {trump_name}...")

        # Use different seed per trump for variety but still reproducible
        trump_seed = args.seed + decl_id if args.seed is not None else None

        points = simulate_games(
            model=model,
            bidder_hand=hand,
            decl_id=decl_id,
            n_games=args.samples,
            seed=trump_seed,
            greedy=not args.sample,
        )

        points_list = points.cpu().tolist()
        trump_result = evaluate_bids(points_list, decl_id)
        results.append(trump_result)
        total_games += args.samples

        mean_pts = sum(points_list) / len(points_list)
        log.debug(f"  {trump_name}: mean={mean_pts:.1f} pts")

    elapsed = time.time() - start_sim
    log.info(f"Simulated {total_games} games in {elapsed:.2f}s ({total_games/elapsed:.0f} games/s)")

    # Output results
    if args.json:
        output = format_json(results)
        output["timing"] = {
            "total_games": total_games,
            "elapsed_seconds": elapsed,
            "games_per_second": total_games / elapsed,
        }
        print(json.dumps(output, indent=2))
    elif args.list:
        # Sorted list format (old default)
        print()
        print(format_results_table(results, show_all_bids=True))
        print()

        best_trump, best_bid, best_swing = find_best_bid(results)
        p_make = None
        for r in results:
            for b in r.bid_results:
                if b.trump_name == best_trump and b.bid == best_bid:
                    p_make = b.p_make
                    break

        print(f"Best: {best_bid} on {best_trump} ({p_make*100:.0f}% make rate, {best_swing:+.2f} expected marks)")
        print(f"Time: {elapsed:.1f}s ({total_games} games)")
    else:
        # Matrix format (default)
        print()
        print("P(make) matrix - percentages (100 = always, . = never)")
        print()
        print(format_matrix(results))
        print()
        print(f"Time: {elapsed:.1f}s ({total_games} games)")


if __name__ == "__main__":
    main()
