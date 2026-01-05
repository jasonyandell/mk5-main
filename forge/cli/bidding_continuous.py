#!/usr/bin/env python3
"""Continuous bidding evaluation CLI: evaluates P(make) for hands non-stop.

Designed to run unattended for days/weeks, generating bidding evaluation data.

For each seed:
1. Generate P0's hand using deal_from_seed(seed)
2. Simulate N games for each of 9 declarations (0-7, 9; skip 8=doubles-suit)
3. Store raw points + P(make) for bids 30-42 + Wilson CI bounds

Usage:
    python -m forge.cli.bidding_continuous                    # Run forever
    python -m forge.cli.bidding_continuous --limit 1          # Single seed test
    python -m forge.cli.bidding_continuous --dry-run          # Preview gaps
    python -m forge.cli.bidding_continuous --start-seed 500   # Start at seed 500
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from forge.bidding.estimator import wilson_ci
from forge.bidding.inference import PolicyModel
from forge.bidding.schema import BID_THRESHOLDS, EVAL_DECLS
from forge.bidding.simulator import simulate_games
from forge.oracle.declarations import DECL_ID_TO_NAME
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW


def get_output_dir(base_dir: Path, seed: int) -> Path:
    """Route seed to train/val/test subdirectory."""
    bucket = seed % 1000
    if bucket < 900:
        return base_dir / "train"
    elif bucket < 950:
        return base_dir / "val"
    else:
        return base_dir / "test"


def format_hand(hand: list[int]) -> str:
    """Format hand as comma-separated high-low pairs."""
    parts = []
    for dom_id in hand:
        high = DOMINO_HIGH[dom_id]
        low = DOMINO_LOW[dom_id]
        parts.append(f"{high}-{low}")
    return ",".join(parts)


def find_missing_seeds(base_dir: Path, start_seed: int) -> Iterator[tuple[int, Path]]:
    """Yield (seed, output_path) for missing evaluation files."""
    seed = start_seed
    while True:
        output_dir = get_output_dir(base_dir, seed)
        path = output_dir / f"seed_{seed:08d}.parquet"
        if not path.exists():
            yield (seed, path)
        seed += 1


def evaluate_seed(
    model: PolicyModel,
    seed: int,
    n_samples: int,
    checkpoint_path: str,
) -> dict:
    """Evaluate a single seed across all declarations.

    Returns dict with all data for parquet storage.
    """
    # Get P0's hand from seed
    hands = deal_from_seed(seed)
    p0_hand = list(hands[0])
    hand_str = format_hand(p0_hand)

    result = {
        "seed": seed,
        "hand": hand_str,
        "n_samples": n_samples,
        "model_checkpoint": checkpoint_path,
        "timestamp": datetime.now().isoformat(),
    }

    # Evaluate each declaration
    for decl_id in EVAL_DECLS:
        # Run simulations
        points = simulate_games(
            model=model,
            bidder_hand=p0_hand,
            decl_id=decl_id,
            n_games=n_samples,
            seed=seed * 1000 + decl_id,  # Unique seed per decl
            greedy=True,
        )

        # Store raw points as numpy array
        points_np = points.cpu().numpy().astype(np.int8)
        result[f"points_{decl_id}"] = points_np

        # Compute P(make) and CI for each bid threshold
        n = len(points_np)
        for bid in BID_THRESHOLDS:
            successes = int((points_np >= bid).sum())
            p_make = successes / n
            ci_low, ci_high = wilson_ci(successes, n)

            result[f"pmake_{decl_id}_{bid}"] = float(p_make)
            result[f"ci_low_{decl_id}_{bid}"] = float(ci_low)
            result[f"ci_high_{decl_id}_{bid}"] = float(ci_high)

    return result


def write_result(output_path: Path, result: dict) -> None:
    """Write evaluation result to parquet file."""
    # Build schema: metadata as single-value columns, points as list columns
    fields = [
        pa.field("seed", pa.int64()),
        pa.field("hand", pa.string()),
        pa.field("n_samples", pa.int32()),
        pa.field("model_checkpoint", pa.string()),
        pa.field("timestamp", pa.string()),
    ]

    # Add points columns (list of int8)
    for decl_id in EVAL_DECLS:
        fields.append(pa.field(f"points_{decl_id}", pa.list_(pa.int8())))

    # Add P(make) and CI columns
    for decl_id in EVAL_DECLS:
        for bid in BID_THRESHOLDS:
            fields.append(pa.field(f"pmake_{decl_id}_{bid}", pa.float32()))
            fields.append(pa.field(f"ci_low_{decl_id}_{bid}", pa.float32()))
            fields.append(pa.field(f"ci_high_{decl_id}_{bid}", pa.float32()))

    schema = pa.schema(fields)

    # Build single-row table
    arrays = {
        "seed": pa.array([result["seed"]], type=pa.int64()),
        "hand": pa.array([result["hand"]], type=pa.string()),
        "n_samples": pa.array([result["n_samples"]], type=pa.int32()),
        "model_checkpoint": pa.array([result["model_checkpoint"]], type=pa.string()),
        "timestamp": pa.array([result["timestamp"]], type=pa.string()),
    }

    # Points arrays
    for decl_id in EVAL_DECLS:
        points = result[f"points_{decl_id}"]
        arrays[f"points_{decl_id}"] = pa.array([points.tolist()], type=pa.list_(pa.int8()))

    # P(make) and CI arrays
    for decl_id in EVAL_DECLS:
        for bid in BID_THRESHOLDS:
            arrays[f"pmake_{decl_id}_{bid}"] = pa.array(
                [result[f"pmake_{decl_id}_{bid}"]], type=pa.float32()
            )
            arrays[f"ci_low_{decl_id}_{bid}"] = pa.array(
                [result[f"ci_low_{decl_id}_{bid}"]], type=pa.float32()
            )
            arrays[f"ci_high_{decl_id}_{bid}"] = pa.array(
                [result[f"ci_high_{decl_id}_{bid}"]], type=pa.float32()
            )

    table = pa.table(arrays, schema=schema)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pq.write_table(table, output_path)


def count_gaps(base_dir: Path, start_seed: int, limit: int = 20) -> list[str]:
    """Count and describe gaps up to limit."""
    gaps = []
    for seed, path in find_missing_seeds(base_dir, start_seed):
        hands = deal_from_seed(seed)
        hand_str = format_hand(hands[0])
        gaps.append(f"seed={seed} hand={hand_str}")
        if len(gaps) >= limit:
            break
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous bidding evaluation: runs indefinitely, fills gaps"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Start at this seed (no backfill before this value)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of simulations per declaration (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/bidding-results"),
        help="Output directory (default: data/bidding-results)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (uses default if not specified)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (default: cuda)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview gaps without evaluating",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after evaluating this many seeds (default: run forever)",
    )
    args = parser.parse_args()

    base_dir = args.output

    print(f"=== Continuous Bidding Evaluation ===")
    print(f"Output: {base_dir}")
    print(f"Start seed: {args.start_seed}")
    print(f"Samples per declaration: {args.samples}")
    print(f"Declarations: {len(EVAL_DECLS)} ({', '.join(DECL_ID_TO_NAME[d] for d in EVAL_DECLS)})")
    print(f"Device: {args.device}")
    if args.limit:
        print(f"Limit: {args.limit} seeds")
    print()

    # Dry run: just show gaps
    if args.dry_run:
        gaps = count_gaps(base_dir, args.start_seed, limit=20)
        print(f"First {len(gaps)} gaps:")
        for gap in gaps:
            print(f"  {gap}")
        if len(gaps) == 20:
            print("  ... (more gaps exist)")
        return

    # Check device
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Use --device cpu or install CUDA-enabled PyTorch.")

    # Load model
    print("Loading model...")
    start_load = time.time()
    model = PolicyModel(checkpoint_path=args.checkpoint, device=args.device)
    checkpoint_path = args.checkpoint or "default"
    print(f"Model loaded in {time.time() - start_load:.2f}s on {model.device}")
    print()

    # Warmup model
    model.warmup(args.samples)

    evaluated = 0
    iterator = find_missing_seeds(base_dir, args.start_seed)

    for seed, output_path in iterator:
        start_time = time.time()

        # Get hand for logging
        hands = deal_from_seed(seed)
        hand_str = format_hand(hands[0])
        split = output_path.parent.name

        print(f"[{split}] seed={seed} hand={hand_str}", end=" ", flush=True)

        while True:  # Retry loop
            try:
                result = evaluate_seed(model, seed, args.samples, checkpoint_path)
                write_result(output_path, result)
                elapsed = time.time() - start_time
                print(f"done ({elapsed:.1f}s)")
                evaluated += 1
                break  # Success
            except Exception as e:
                print(f"\nERROR: {e}")
                if "out of memory" in str(e).lower():
                    print("HINT: Try reducing --samples or use a smaller batch size")
                print("Retrying in 5 seconds...")
                time.sleep(5)
                continue

        if args.limit and evaluated >= args.limit:
            print(f"\nReached limit of {args.limit} seeds.")
            break

    print(f"\n=== Complete: {evaluated} seeds evaluated ===")


if __name__ == "__main__":
    main()
