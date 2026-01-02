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
import time
from pathlib import Path

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Wandb group name for organizing runs",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-tokenization even if output exists",
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
    from forge.ml.tokenize import ShardProgress, tokenize_shards

    # Initialize wandb if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: --wandb requested but wandb not installed. Run: pip install wandb")

    if use_wandb:
        # Build tags: always include tokenize/preprocessing, plus group root if provided
        tags = ["tokenize", "preprocessing"]
        if args.wandb_group:
            group_root = args.wandb_group.split("/")[0]
            tags.append(group_root)
        wandb.init(
            project="crystal-forge",
            job_type="tokenize",
            name=f"tokenize-{output_dir.name}",
            group=args.wandb_group,
            dir="runs",  # Consolidate all wandb logs in runs/wandb/
            config={
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "global_seed": args.seed,
                "max_samples_per_shard": args.max_samples_per_shard,
                "max_files": args.max_files,
            },
            tags=tags,
        )

    # Create progress callback for wandb
    start_time = time.time()

    def progress_callback(progress: ShardProgress) -> None:
        if not use_wandb:
            return
        wandb.log({
            "shard/seed": progress.seed,
            "shard/decl_id": progress.decl_id,
            "shard/split": progress.split,
            "shard/samples": progress.samples,
            "shard/elapsed_time": progress.elapsed_time,
            "progress/completed": progress.file_index + 1,
            "progress/total": progress.total_files,
            "progress/percent": 100.0 * (progress.file_index + 1) / progress.total_files,
            "cumulative/train_samples": progress.cumulative_samples["train"],
            "cumulative/val_samples": progress.cumulative_samples["val"],
            "cumulative/test_samples": progress.cumulative_samples["test"],
            "cumulative/total_samples": sum(progress.cumulative_samples.values()),
        })

    try:
        manifest = tokenize_shards(
            input_dir=input_dir,
            output_dir=output_dir,
            global_seed=args.seed,
            max_samples_per_shard=args.max_samples_per_shard,
            max_files=args.max_files,
            verbose=not args.quiet,
            progress_callback=progress_callback if use_wandb else None,
            force=args.force,
        )

        # Log final summary to wandb
        if use_wandb:
            total_time = time.time() - start_time
            total_samples = sum(s["samples"] for s in manifest.splits.values())
            wandb.run.summary["total_time"] = total_time
            wandb.run.summary["total_samples"] = total_samples
            wandb.run.summary["samples_per_second"] = total_samples / total_time if total_time > 0 else 0
            for split, stats in manifest.splits.items():
                wandb.run.summary[f"{split}_samples"] = stats["samples"]
                wandb.run.summary[f"{split}_files"] = stats["files"]
            wandb.finish()

        # Summary
        if not args.quiet:
            print("\n=== Summary ===")
            for split, stats in manifest.splits.items():
                print(f"  {split}: {stats['samples']:,} samples from {stats['files']} files")

        return 0

    except Exception as e:
        if use_wandb:
            wandb.finish(exit_code=1)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
