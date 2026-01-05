#!/usr/bin/env python3
"""Continuous seed generation CLI: generates oracle seeds non-stop, fills gaps.

Designed to run unattended for days/weeks, generating training data continuously.

Standard vs Marginalized (separate experiments):
- Standard: 1 decl per seed, trains on single-deal perfect play
- Marginalized: N opp seeds per P0 hand, trains on averaged/robust play

Usage:
    python -m forge.cli.generate_continuous              # Standard mode
    python -m forge.cli.generate_continuous --marginalized  # Marginalized mode
    python -m forge.cli.generate_continuous --dry-run    # Preview gaps
    python -m forge.cli.generate_continuous --start-seed 500  # Start at seed 500
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterator

import torch

from forge.oracle.context import build_context
from forge.oracle.declarations import DECL_ID_TO_NAME
from forge.oracle.output import output_path_for, write_result
from forge.oracle.rng import deal_from_seed
from forge.oracle.solve import SolveConfig, build_child_index, enumerate_gpu, solve_gpu
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW
from forge.oracle.timer import SeedTimer


def get_output_dir(base_dir: Path, seed: int) -> Path:
    """Route seed to train/val/test subdirectory."""
    bucket = seed % 1000
    if bucket < 900:
        return base_dir / "train"
    elif bucket < 950:
        return base_dir / "val"
    else:
        return base_dir / "test"


def format_hand_for_context(hand: list[int]) -> str:
    """Format hand as comma-separated high-low pairs for --p0-hand."""
    parts = []
    for dom_id in hand:
        high = DOMINO_HIGH[dom_id]
        low = DOMINO_LOW[dom_id]
        parts.append(f"{high}-{low}")
    return ",".join(parts)


def find_missing_standard(
    base_dir: Path,
    start_seed: int,
) -> Iterator[tuple[int, int, Path]]:
    """Yield (seed, decl_id, output_path) for missing standard shards."""
    seed = start_seed
    while True:
        output_dir = get_output_dir(base_dir, seed)
        decl_id = seed % 10
        path = output_dir / f"seed_{seed:08d}_decl_{decl_id}.parquet"
        if not path.exists():
            yield (seed, decl_id, path)
        seed += 1


def find_missing_marginalized(
    base_dir: Path,
    start_seed: int,
    n_opp_seeds: int,
) -> Iterator[tuple[int, int, int, Path]]:
    """Yield (seed, opp_seed, decl_id, output_path) for missing marginalized shards."""
    seed = start_seed
    while True:
        output_dir = get_output_dir(base_dir, seed)
        decl_id = seed % 10
        for opp_seed in range(n_opp_seeds):
            path = output_dir / f"seed_{seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
            if not path.exists():
                yield (seed, opp_seed, decl_id, path)
        seed += 1


def generate_standard_shard(
    seed: int,
    decl_id: int,
    output_path: Path,
    device: torch.device,
    config: SolveConfig,
) -> None:
    """Generate a single standard shard."""
    timer = SeedTimer(seed=seed, decl_id=decl_id)
    timer.phase("start", extra=f"decl={DECL_ID_TO_NAME.get(decl_id, str(decl_id))} device={device}")

    ctx = build_context(seed=seed, decl_id=decl_id, device=device)
    timer.phase("setup")

    all_states = enumerate_gpu(ctx, config=config)
    timer.phase("enumerate", extra=f"states={int(all_states.shape[0]):,}")

    child_idx = build_child_index(all_states, ctx, config=config)
    timer.phase("child_index")

    v, move_values = solve_gpu(all_states, child_idx, ctx, config=config)
    root_value = int(v[0])
    timer.phase("solve", extra=f"root={root_value:+d}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_result(output_path, seed, decl_id, all_states, v, move_values, fmt="parquet")
    timer.phase("write", extra=str(output_path))
    timer.done(root_value=root_value)


def generate_marginalized_shard(
    base_seed: int,
    opp_seed: int,
    decl_id: int,
    output_path: Path,
    device: torch.device,
    config: SolveConfig,
) -> None:
    """Generate a single marginalized shard.

    Uses base_seed to determine P0's hand, opp_seed to shuffle opponent cards.
    """
    timer = SeedTimer(seed=base_seed, decl_id=decl_id)
    timer.phase("start", extra=f"opp={opp_seed} decl={DECL_ID_TO_NAME.get(decl_id, str(decl_id))} device={device}")

    # Get P0's hand from base seed
    hands = deal_from_seed(base_seed)
    p0_hand = list(hands[0])

    # Build context with fixed P0 hand, opp_seed for opponent distribution
    ctx = build_context(seed=opp_seed, decl_id=decl_id, device=device, p0_hand=p0_hand)
    timer.phase("setup")

    all_states = enumerate_gpu(ctx, config=config)
    timer.phase("enumerate", extra=f"states={int(all_states.shape[0]):,}")

    child_idx = build_child_index(all_states, ctx, config=config)
    timer.phase("child_index")

    v, move_values = solve_gpu(all_states, child_idx, ctx, config=config)
    root_value = int(v[0])
    timer.phase("solve", extra=f"root={root_value:+d}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use base_seed for metadata (determines train/val/test split in tokenizer)
    write_result(output_path, base_seed, decl_id, all_states, v, move_values, fmt="parquet")
    timer.phase("write", extra=str(output_path))
    timer.done(root_value=root_value)


def count_gaps(
    base_dir: Path,
    start_seed: int,
    marginalized: bool,
    n_opp_seeds: int,
    limit: int = 100,
) -> list[str]:
    """Count and describe gaps up to limit."""
    gaps = []
    if marginalized:
        for seed, opp_seed, decl_id, path in find_missing_marginalized(base_dir, start_seed, n_opp_seeds):
            gaps.append(f"seed={seed} opp={opp_seed} decl={decl_id}")
            if len(gaps) >= limit:
                break
    else:
        for seed, decl_id, path in find_missing_standard(base_dir, start_seed):
            gaps.append(f"seed={seed} decl={decl_id}")
            if len(gaps) >= limit:
                break
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous seed generation: fills gaps and runs indefinitely"
    )
    parser.add_argument(
        "--marginalized",
        action="store_true",
        help="Generate marginalized data (multiple opp seeds per P0 hand)",
    )
    parser.add_argument(
        "--n-opp-seeds",
        type=int,
        default=3,
        help="Number of opponent seeds per P0 hand in marginalized mode (default: 3)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="Start at this seed (no backfill before this value)",
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
        help="Preview gaps without generating",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after generating this many shards (default: run forever)",
    )
    parser.add_argument(
        "--child-index-chunk",
        type=int,
        default=1_000_000,
        help="Chunk size for child index build",
    )
    parser.add_argument(
        "--solve-chunk",
        type=int,
        default=1_000_000,
        help="Chunk size for backward induction",
    )
    parser.add_argument(
        "--enum-chunk",
        type=int,
        default=100_000,
        help="Chunk size for enumeration",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.marginalized:
        base_dir = Path("data/shards-marginalized")
        mode_name = "marginalized"
    else:
        base_dir = Path("data/shards-standard")
        mode_name = "standard"

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Use --device cpu or install CUDA-enabled PyTorch.")

    print(f"=== Continuous Generation ({mode_name}) ===")
    print(f"Output: {base_dir}")
    print(f"Start seed: {args.start_seed}")
    print(f"Device: {device}")
    if args.marginalized:
        print(f"Opponent seeds per P0 hand: {args.n_opp_seeds}")
    if args.limit:
        print(f"Limit: {args.limit} shards")
    print()

    # Dry run: just show gaps
    if args.dry_run:
        gaps = count_gaps(base_dir, args.start_seed, args.marginalized, args.n_opp_seeds, limit=20)
        print(f"First {len(gaps)} gaps:")
        for gap in gaps:
            print(f"  {gap}")
        if len(gaps) == 20:
            print("  ... (more gaps exist)")
        return

    config = SolveConfig(
        child_index_chunk_size=args.child_index_chunk,
        solve_chunk_size=args.solve_chunk,
        enum_chunk_size=args.enum_chunk,
    )

    generated = 0

    if args.marginalized:
        iterator = find_missing_marginalized(base_dir, args.start_seed, args.n_opp_seeds)
        for base_seed, opp_seed, decl_id, output_path in iterator:
            while True:  # Retry loop
                try:
                    generate_marginalized_shard(
                        base_seed, opp_seed, decl_id, output_path, device, config
                    )
                    generated += 1
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"ERROR: {e}")
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

            if args.limit and generated >= args.limit:
                print(f"\nReached limit of {args.limit} shards.")
                break
    else:
        iterator = find_missing_standard(base_dir, args.start_seed)
        for seed, decl_id, output_path in iterator:
            while True:  # Retry loop
                try:
                    generate_standard_shard(seed, decl_id, output_path, device, config)
                    generated += 1
                    break  # Success, exit retry loop
                except Exception as e:
                    print(f"ERROR: {e}")
                    print("Retrying in 5 seconds...")
                    time.sleep(5)
                    continue

            if args.limit and generated >= args.limit:
                print(f"\nReached limit of {args.limit} shards.")
                break

    print(f"\n=== Complete: {generated} shards generated ===")


if __name__ == "__main__":
    main()
