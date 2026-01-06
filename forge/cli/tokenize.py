#!/usr/bin/env python3
"""
CLI for tokenizing oracle parquet files.

Converts oracle parquet files to pre-tokenized numpy arrays with:
- Per-shard deterministic RNG (keyed by global_seed, shard_seed, decl_id)
- Train/val/test split based on seed bucket (90/5/5)
- Output format matching DominoDataset expectations

Usage:
    # Default: read from data/shards-standard/, write to data/tokenized/
    python -m forge.cli.tokenize

    # Explicit paths (e.g., external drive)
    python -m forge.cli.tokenize \\
      --input-dir /mnt/d/shards-standard \\
      --output-dir /mnt/d/tokenized

    # Preview without processing
    python -m forge.cli.tokenize --dry-run

    # Quick test with small subset
    python -m forge.cli.tokenize \\
      --output-dir scratch/tok_test \\
      --max-files 5 \\
      --max-samples-per-shard 1000
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


def _dry_run(input_dir: Path, output_dir: Path, files: list[Path], args: argparse.Namespace) -> int:
    """Preview tokenization without processing."""
    # Count files by source location
    subdir_counts = {'train': 0, 'val': 0, 'test': 0, 'flat': 0}
    total_bytes = 0

    for f in files:
        # Check if file is in a subdir
        parent_name = f.parent.name
        if parent_name in ('train', 'val', 'test'):
            subdir_counts[parent_name] += 1
        else:
            subdir_counts['flat'] += 1
        total_bytes += f.stat().st_size

    # Estimate output size (roughly 1:8 compression from parquet to tokenized numpy)
    # Based on empirical data: 15GB tokenized from ~120GB parquet
    estimated_output_gb = total_bytes / (1024**3) * 0.125

    print("=== Tokenization Preview (dry-run) ===\n")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Found {len(files)} parquet files ({total_bytes / (1024**3):.1f} GB)\n")

    # Show structure
    has_subdirs = any(subdir_counts[s] > 0 for s in ('train', 'val', 'test'))
    if has_subdirs:
        print("Source structure: {train,val,test}/ subdirectories")
        for split in ('train', 'val', 'test'):
            if subdir_counts[split] > 0:
                print(f"  {split}/: {subdir_counts[split]} files")
    else:
        print(f"Source structure: flat directory ({subdir_counts['flat']} files)")

    print(f"\nEstimated output: ~{estimated_output_gb:.1f} GB")
    print(f"Max samples/shard: {args.max_samples_per_shard}")
    print(f"Global seed: {args.seed}")

    if args.max_files:
        print(f"\nNote: Limited to first {args.max_files} files (--max-files)")

    print("\nTo run tokenization, remove --dry-run flag")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tokenize oracle parquet files for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default paths (local)
  python -m forge.cli.tokenize

  # External drive
  python -m forge.cli.tokenize --input-dir /mnt/d/shards-standard --output-dir /mnt/d/tokenized

  # Quick test
  python -m forge.cli.tokenize --output-dir scratch/tok_test --max-files 5 --max-samples-per-shard 1000
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Input directory containing seed_*.parquet files (default: data/shards-standard)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview file counts and estimated output size without processing",
    )

    args = parser.parse_args()

    # Apply defaults (matches generate_continuous.py pattern)
    input_dir = args.input_dir if args.input_dir else Path("data/shards-standard")
    output_dir = args.output_dir if args.output_dir else Path("data/tokenized")

    # Validate input directory
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    # Import here to avoid slow startup for --help
    from forge.ml.tokenize import ShardProgress, find_parquet_files, tokenize_shards

    # Check for parquet files (supports both subdir and flat structures)
    parquet_files = find_parquet_files(input_dir, max_files=args.max_files)
    if not parquet_files:
        print(f"ERROR: No seed_*.parquet files found in {input_dir}", file=sys.stderr)
        print("  Checked: {train,val,test}/*.parquet and *.parquet", file=sys.stderr)
        return 1

    # Dry-run mode: preview without processing
    if args.dry_run:
        return _dry_run(input_dir, output_dir, parquet_files, args)

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
