#!/usr/bin/env python3
"""
GPU Solver CLI - Solve Texas 42 games and write training data.

Usage:
    python -m main --seed 0 --decl 0               # Solve single seed
    python -m main --seed-range 0 100 --decl 0     # Solve range of seeds
    python -m main --seed 0 --decl 0 --json        # Output as JSON
"""

import argparse
import sys
import time
from pathlib import Path

import torch

from output import solve_and_save


def get_device(use_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if use_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(
        description='GPU solver for Texas 42',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m main --seed 0 --decl 0
    python -m main --seed-range 0 100 --decl 0 --output-dir ./solved
    python -m main --seed 42 --decl 3 --json --cpu
        """
    )

    # Seed selection (mutually exclusive)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument('--seed', type=int, help='Single seed to solve')
    seed_group.add_argument('--seed-range', type=int, nargs=2, metavar=('START', 'END'),
                           help='Range of seeds to solve [start, end)')

    # Declaration
    parser.add_argument('--decl', type=int, required=True, choices=range(7),
                       help='Declaration ID (0-6 for pip trump)')

    # Output options
    parser.add_argument('--output-dir', type=Path, default=Path('./output'),
                       help='Output directory (default: ./output)')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON instead of Parquet')

    # Device options
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU mode (no GPU)')

    args = parser.parse_args()

    # Setup
    device = get_device(use_cuda=not args.cpu)
    print(f"Using device: {device}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine seeds to solve
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(range(args.seed_range[0], args.seed_range[1]))

    print(f"Seeds to solve: {len(seeds)}")
    print(f"Declaration: {args.decl} ({'blanks ones twos threes fours fives sixes'.split()[args.decl]} trump)")
    print()

    # Solve
    start_time = time.time()
    solved_count = 0
    skipped_count = 0

    for seed in seeds:
        try:
            if solve_and_save(
                seed=seed,
                decl_id=args.decl,
                output_dir=output_dir,
                device=device,
                use_json=args.json,
            ):
                solved_count += 1
            else:
                skipped_count += 1
                print(f"Skipping seed={seed} (already exists)")

        except Exception as e:
            print(f"ERROR solving seed={seed}: {e}")
            raise

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 50)
    print(f"Completed: {solved_count} solved, {skipped_count} skipped")
    print(f"Total time: {elapsed:.1f}s")
    if solved_count > 0:
        print(f"Average: {elapsed / solved_count:.2f}s per seed")


if __name__ == '__main__':
    main()
