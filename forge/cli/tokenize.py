#!/usr/bin/env python3
"""
CLI for tokenizing oracle parquet files.

Converts oracle parquet files to pre-tokenized numpy arrays with:
- Per-shard deterministic RNG (keyed by global_seed, shard_seed, decl_id)
- Train/val/test split based on seed bucket (90/5/5)
- Output format matching DominoDataset expectations

Usage:
    # Default: read from data/shards/, write to data/tokenized/
    python -m forge.cli.tokenize

    # Explicit paths
    python -m forge.cli.tokenize \\
      --input data/shards \\
      --output data/tokenized \\
      --max-samples-per-shard 50000 \\
      --seed 42

    # During transition: read from legacy location
    python -m forge.cli.tokenize \\
      --input data/solver2 \\
      --output data/tokenized \\
      --seed 42

    # Quick test with small subset
    python -m forge.cli.tokenize \\
      --input data/solver2 \\
      --output scratch/tok_test \\
      --max-files 5 \\
      --max-samples-per-shard 1000 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tokenize oracle parquet files for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default paths
  python -m forge.cli.tokenize

  # Custom paths
  python -m forge.cli.tokenize --input data/solver2 --output data/tokenized

  # Quick test
  python -m forge.cli.tokenize --input data/solver2 --output scratch/tok_test --max-files 5 --max-samples-per-shard 1000
        """,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/shards",
        help="Input directory containing seed_*.parquet files (default: data/shards)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/tokenized",
        help="Output directory for tokenized data (default: data/tokenized)",
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=50000,
        help="Maximum samples per shard (default: 50000)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum files to process (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for sampling RNG (default: 42)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input directory
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    # Check for parquet files
    parquet_files = list(input_dir.glob("seed_*.parquet"))
    if not parquet_files:
        print(f"ERROR: No seed_*.parquet files found in {input_dir}", file=sys.stderr)
        return 1

    # Import here to avoid slow startup for --help
    from forge.ml.tokenize import tokenize_shards

    try:
        manifest = tokenize_shards(
            input_dir=input_dir,
            output_dir=output_dir,
            global_seed=args.seed,
            max_samples_per_shard=args.max_samples_per_shard,
            max_files=args.max_files,
            verbose=not args.quiet,
        )

        # Summary
        if not args.quiet:
            print("\n=== Summary ===")
            for split, stats in manifest.splits.items():
                print(f"  {split}: {stats['samples']:,} samples from {stats['files']} files")

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
