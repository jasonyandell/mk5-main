#!/usr/bin/env python3
"""Campaign script for generating marginalized Q-value training data.

Generates 3 shards per P0 hand by sampling different opponent distributions.
This allows the model to learn robust strategies through implicit averaging.

Usage:
    python -m forge.scripts.campaign_marginalized --base-seed-range 0:200 --batch-size 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW


def format_hand_for_cli(hand: list[int]) -> str:
    """Format hand as comma-separated high-low pairs for --p0-hand flag."""
    parts = []
    for dom_id in hand:
        high = DOMINO_HIGH[dom_id]
        low = DOMINO_LOW[dom_id]
        parts.append(f"{high}-{low}")
    return ",".join(parts)


def check_existing(out_dir: Path, base_seed: int, decl_id: int, n_opp_seeds: int) -> bool:
    """Check if all shards for this base_seed already exist."""
    for opp_seed in range(n_opp_seeds):
        path = out_dir / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return False
    return True


def generate_shard(
    base_seed: int,
    opp_seed: int,
    decl_id: int,
    p0_hand: str,
    out_dir: Path,
    device: str,
    wandb_group: str | None,
) -> subprocess.Popen:
    """Spawn oracle generation for a single marginalized shard."""
    cmd = [
        sys.executable, "-m", "forge.oracle.generate",
        "--seed", str(opp_seed),
        "--base-seed", str(base_seed),
        "--opp-seed", str(opp_seed),
        "--decl", str(decl_id),
        "--p0-hand", p0_hand,
        "--out", str(out_dir),
        "--device", device,
    ]
    if wandb_group:
        cmd.extend(["--wandb", "--wandb-group", wandb_group])
    return subprocess.Popen(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate marginalized Q-value training data")
    parser.add_argument(
        "--base-seed-range",
        type=str,
        required=True,
        help="Range of base seeds (e.g., 0:200 for seeds 0-199)",
    )
    parser.add_argument(
        "--n-opp-seeds",
        type=int,
        default=3,
        help="Number of opponent distributions per P0 hand (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of parallel oracle processes (default: 5 for A100 40GB)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/shards"),
        help="Output directory for shards",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for oracle computation",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Wandb group name for organizing runs",
    )
    args = parser.parse_args()

    # Parse seed range
    parts = args.base_seed_range.split(":")
    if len(parts) != 2:
        raise ValueError("base-seed-range must be START:END")
    start, end = int(parts[0]), int(parts[1])
    if end <= start:
        raise ValueError("END must be > START")

    base_seeds = list(range(start, end))
    n_opp_seeds = args.n_opp_seeds
    total_shards = len(base_seeds) * n_opp_seeds

    print(f"=== Marginalized Campaign ===")
    print(f"Base seeds: {start}-{end - 1} ({len(base_seeds)} seeds)")
    print(f"Opponent seeds per P0 hand: {n_opp_seeds}")
    print(f"Total shards to generate: {total_shards}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.out}")
    print()

    # Ensure output directory exists
    args.out.mkdir(parents=True, exist_ok=True)

    # Build work queue: (base_seed, opp_seed, decl_id, p0_hand)
    work_queue: list[tuple[int, int, int, str]] = []
    skipped = 0

    for base_seed in base_seeds:
        decl_id = base_seed % 10

        # Check if all shards for this base_seed exist
        if check_existing(args.out, base_seed, decl_id, n_opp_seeds):
            skipped += n_opp_seeds
            continue

        # Get P0's hand from base seed
        hands = deal_from_seed(base_seed)
        p0_hand = format_hand_for_cli(hands[0])

        for opp_seed in range(n_opp_seeds):
            # Check individual shard
            path = args.out / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
            if path.exists():
                skipped += 1
                continue
            work_queue.append((base_seed, opp_seed, decl_id, p0_hand))

    if skipped > 0:
        print(f"Skipping {skipped} existing shards")
    print(f"Work queue: {len(work_queue)} shards")
    print()

    if not work_queue:
        print("Nothing to do!")
        return

    # Process in batches
    completed = 0
    processes: list[tuple[subprocess.Popen, int, int, int]] = []

    def drain_batch() -> None:
        """Wait for all processes in the current batch to complete."""
        nonlocal completed
        for proc, bs, os, did in processes:
            ret = proc.wait()
            if ret != 0:
                print(f"ERROR: base_seed={bs} opp_seed={os} decl={did} failed with code {ret}")
            else:
                completed += 1
                if completed % 10 == 0 or completed == len(work_queue):
                    print(f"Progress: {completed}/{len(work_queue)} shards ({100*completed/len(work_queue):.1f}%)")
        processes.clear()

    for base_seed, opp_seed, decl_id, p0_hand in work_queue:
        proc = generate_shard(
            base_seed, opp_seed, decl_id, p0_hand,
            args.out, args.device, args.wandb_group,
        )
        processes.append((proc, base_seed, opp_seed, decl_id))

        if len(processes) >= args.batch_size:
            drain_batch()

    # Drain remaining
    if processes:
        drain_batch()

    print()
    print(f"=== Campaign Complete: {completed} shards generated ===")


if __name__ == "__main__":
    main()
