#!/usr/bin/env python3
"""One-time migration script: reorganize data/shards/ into split directories.

Migrates existing shards from flat data/shards/ to:
- data/shards-standard/{train,val,test}/
- data/shards-marginalized/{train,val,test}/

Usage:
    python -m forge.scripts.migrate_shards --dry-run  # Preview
    python -m forge.scripts.migrate_shards            # Execute
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def get_split(seed: int) -> str:
    """Route seed to train/val/test subdirectory."""
    bucket = seed % 1000
    if bucket < 900:
        return "train"
    elif bucket < 950:
        return "val"
    else:
        return "test"


def parse_filename(filename: str) -> tuple[int, int | None, int] | None:
    """Parse shard filename to extract (seed, opp_seed, decl_id).

    Returns None if filename doesn't match expected patterns.

    Patterns:
        seed_00000000_decl_0.parquet -> (0, None, 0)
        seed_00000000_opp0_decl_0.parquet -> (0, 0, 0)
    """
    # Marginalized pattern: seed_XXXXXXXX_oppY_decl_Z.parquet
    m = re.match(r"seed_(\d+)_opp(\d+)_decl_(\d+)\.parquet$", filename)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Standard pattern: seed_XXXXXXXX_decl_Z.parquet
    m = re.match(r"seed_(\d+)_decl_(\d+)\.parquet$", filename)
    if m:
        return int(m.group(1)), None, int(m.group(2))

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate shards to new directory structure")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without moving files",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/shards"),
        help="Source directory (default: data/shards)",
    )
    args = parser.parse_args()

    source_dir = args.source
    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    # Scan source for parquet files
    source_files = list(source_dir.glob("*.parquet"))
    if not source_files:
        raise SystemExit(f"No parquet files found in {source_dir}")

    print(f"=== Shard Migration {'(DRY RUN)' if args.dry_run else ''} ===")
    print(f"Source: {source_dir}")
    print(f"Files found: {len(source_files)}")
    print()

    # Build migration plan
    plan: list[tuple[Path, Path, str]] = []  # (src, dst, category)
    errors: list[str] = []

    standard_counts = {"train": 0, "val": 0, "test": 0}
    marginalized_counts = {"train": 0, "val": 0, "test": 0}

    for src_path in source_files:
        parsed = parse_filename(src_path.name)
        if parsed is None:
            errors.append(f"Cannot parse filename: {src_path.name}")
            continue

        seed, opp_seed, decl_id = parsed
        split = get_split(seed)

        if opp_seed is not None:
            # Marginalized
            dest_dir = source_dir.parent / "shards-marginalized" / split
            category = "marginalized"
            marginalized_counts[split] += 1
        else:
            # Standard
            dest_dir = source_dir.parent / "shards-standard" / split
            category = "standard"
            standard_counts[split] += 1

        dest_path = dest_dir / src_path.name
        plan.append((src_path, dest_path, category))

    # Report any parse errors
    if errors:
        print("ERRORS parsing filenames:")
        for err in errors:
            print(f"  {err}")
        raise SystemExit("Aborting due to parse errors")

    # Summary
    print("Migration plan:")
    print(f"  Standard:     train={standard_counts['train']:4d}  val={standard_counts['val']:4d}  test={standard_counts['test']:4d}")
    print(f"  Marginalized: train={marginalized_counts['train']:4d}  val={marginalized_counts['val']:4d}  test={marginalized_counts['test']:4d}")
    print(f"  Total: {len(plan)} files")
    print()

    # Verify counts match source
    if len(plan) != len(source_files):
        raise SystemExit(f"Count mismatch: planned {len(plan)} != source {len(source_files)}")

    if args.dry_run:
        print("Sample moves (first 10):")
        for src, dst, cat in plan[:10]:
            print(f"  {src.name} -> {dst.relative_to(source_dir.parent)}")
        if len(plan) > 10:
            print(f"  ... and {len(plan) - 10} more")
        print()
        print("Run without --dry-run to execute migration.")
        return

    # Execute migration
    print("Executing migration...")

    # Create destination directories
    for category in ["standard", "marginalized"]:
        for split in ["train", "val", "test"]:
            dest_dir = source_dir.parent / f"shards-{category}" / split
            dest_dir.mkdir(parents=True, exist_ok=True)

    # Move files
    moved = 0
    for src_path, dest_path, category in plan:
        shutil.move(str(src_path), str(dest_path))
        moved += 1
        if moved % 100 == 0:
            print(f"  Moved {moved}/{len(plan)} files...")

    print(f"  Moved {moved}/{len(plan)} files")
    print()

    # Verify destination counts
    print("Verifying destination...")
    dest_total = 0
    for category in ["standard", "marginalized"]:
        for split in ["train", "val", "test"]:
            dest_dir = source_dir.parent / f"shards-{category}" / split
            count = len(list(dest_dir.glob("*.parquet")))
            dest_total += count
            expected = standard_counts[split] if category == "standard" else marginalized_counts[split]
            status = "✓" if count == expected else "✗ MISMATCH"
            print(f"  shards-{category}/{split}: {count} files {status}")

    if dest_total != len(plan):
        raise SystemExit(f"VERIFICATION FAILED: destination has {dest_total} files, expected {len(plan)}")

    # Check if source is now empty (only should have subdirs or be empty)
    remaining = list(source_dir.glob("*.parquet"))
    if remaining:
        raise SystemExit(f"VERIFICATION FAILED: {len(remaining)} files still in source directory")

    print()
    print(f"=== Migration Complete: {moved} files moved ===")
    print()
    print("You may now remove the empty source directory:")
    print(f"  rmdir {source_dir}")


if __name__ == "__main__":
    main()
