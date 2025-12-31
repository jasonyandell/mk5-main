#!/usr/bin/env python3
"""
Preprocess solver2 data: convert local indices to global domino IDs.

Step 2 of MLP confidence ladder: Cross-Seed Generalization.
- Load all parquet files
- Convert local indices → global domino IDs (28 bits per player)
- Encode to ~240 features
- Split by SEED: seeds 0-89 → train, seeds 90-99 → test
- Save as train_global.parquet and test_global.parquet

Usage:
    python scripts/solver2/preprocess_global.py [--max-states-per-file 100000]
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def deal_from_seed(seed: int) -> list[list[int]]:
    """Return 4 hands of 7 unique domino IDs, deterministically from seed."""
    import random
    rng = random.Random(seed)
    dominos = list(range(28))
    rng.shuffle(dominos)
    hands = [sorted(dominos[i * 7 : (i + 1) * 7]) for i in range(4)]
    return hands


def encode_global_features(
    states: np.ndarray,
    seed: int,
    decl_id: int,
    max_states: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Encode packed states into global feature vectors.

    Features (240 total):
    - remaining[0-3]: 4 × 28 = 112 (global domino binary masks)
    - score: 1 (normalized, computed from played count dominoes)
    - leader: 4 (one-hot)
    - decl_id: 10 (one-hot)
    - trick plays: 4 × 28 = 112 (global domino one-hot per position)
    - tricks_completed: 1 (normalized 0-7)

    Total: 112 + 1 + 4 + 10 + 112 + 1 = 240 features

    Returns: (features, sample_indices) where sample_indices is None if no sampling.
    """
    sample_indices = None

    # Optionally sample
    if max_states is not None and len(states) > max_states:
        if rng is None:
            rng = np.random.default_rng(seed * 1000 + decl_id)
        sample_indices = rng.choice(len(states), size=max_states, replace=False)
        states = states[sample_indices]

    n = len(states)
    features = np.zeros((n, 240), dtype=np.float32)

    # Get local→global mapping from seed
    hands = deal_from_seed(seed)

    # Precompute count points for all 28 dominoes
    count_points = np.zeros(28, dtype=np.int32)
    DOMINOES = [(h, l) for h in range(7) for l in range(h + 1)]
    for d_id, (h, l) in enumerate(DOMINOES):
        if (h, l) in ((5, 5), (6, 4)):
            count_points[d_id] = 10
        elif (h, l) in ((5, 0), (4, 1), (3, 2)):
            count_points[d_id] = 5

    # Build local→global lookup tables for each player
    local_to_global = np.zeros((4, 8), dtype=np.int32)  # 8th slot for "none" indicator
    for p in range(4):
        for local_idx in range(7):
            local_to_global[p, local_idx] = hands[p][local_idx]
        local_to_global[p, 7] = -1  # "none" marker

    # Extract packed fields once (vectorized)
    remaining_local = np.zeros((n, 4), dtype=np.uint8)
    for p in range(4):
        remaining_local[:, p] = (states >> (p * 7)) & 0x7F

    leader = (states >> 28) & 0x3
    trick_len = (states >> 30) & 0x3
    p0 = (states >> 32) & 0x7
    p1 = (states >> 35) & 0x7
    p2 = (states >> 38) & 0x7

    # Feature 0-111: Global remaining masks (4 players × 28 dominoes)
    for p in range(4):
        for local_idx in range(7):
            global_id = hands[p][local_idx]
            # bit is set if player still has this domino
            has_domino = (remaining_local[:, p] >> local_idx) & 1
            features[:, p * 28 + global_id] = has_domino.astype(np.float32)

    # Feature 112: Score (normalized)
    # Compute from played dominoes - dominoes not in any remaining set
    all_remaining = np.zeros((n, 28), dtype=np.uint8)
    for p in range(4):
        for local_idx in range(7):
            global_id = hands[p][local_idx]
            has_domino = (remaining_local[:, p] >> local_idx) & 1
            all_remaining[:, global_id] = has_domino
    # Score is sum of count points for dominoes that are played (not remaining)
    played = 1 - all_remaining
    score = np.sum(played * count_points[np.newaxis, :], axis=1)
    features[:, 112] = score / 42.0  # Max possible score from count dominoes is 42

    # Feature 113-116: Leader (one-hot)
    row_idx = np.arange(n)
    features[row_idx, 113 + leader] = 1.0

    # Feature 117-126: Declaration (one-hot)
    features[:, 117 + decl_id] = 1.0

    # Feature 127-238: Trick plays (4 positions × 28 dominoes)
    # Each position: one-hot for global domino, all zeros if no play
    for trick_pos, (play_local, seat_offset) in enumerate([(p0, 0), (p1, 1), (p2, 2)]):
        # For 4th position (p3), it's only used when trick_len == 3 but p3 isn't stored
        # since the trick resolves immediately. We leave p3 as zeros.
        seat = (leader + seat_offset) % 4
        mask = play_local < 7  # Valid play (not "none" indicator)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) > 0:
            local_plays = play_local[valid_indices]
            global_plays = local_to_global[seat[valid_indices], local_plays]
            feature_base = 127 + trick_pos * 28
            features[valid_indices, feature_base + global_plays] = 1.0

    # 4th trick position (127 + 3*28 = 211-238): always zeros since p3 not stored

    # Feature 239: Tricks completed (normalized)
    # Total dominoes = 28, each trick uses 4 dominoes
    # remaining_count = sum of popcount of all 4 remaining masks
    remaining_count = np.zeros(n, dtype=np.int32)
    for p in range(4):
        for bit in range(7):
            remaining_count += (remaining_local[:, p] >> bit) & 1
    # Dominoes played = 28 - remaining_count
    # But during a trick, we also have partial plays in p0/p1/p2
    # Let's count actual completed tricks = (28 - remaining_count - trick_len) / 4
    dominoes_played = 28 - remaining_count
    # Actually trick_len indicates how many plays in current trick
    # Completed tricks = (dominoes_played - trick_len) / 4
    tricks_completed = (dominoes_played - trick_len) // 4
    features[:, 239] = tricks_completed / 7.0  # Max 7 tricks

    return features, sample_indices


def process_file(args: tuple) -> tuple[int, int, np.ndarray, np.ndarray] | None:
    """Process a single parquet file. Returns (seed, decl_id, features, values)."""
    file_path, max_states = args

    try:
        pf = pq.ParquetFile(file_path)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())

        df = pd.read_parquet(file_path)
        states = df["state"].values
        values = df["V"].values.astype(np.float32) / 42.0  # Normalize to [-1, 1]

        # Encode features with optional sampling (returns sample_indices for consistency)
        features, sample_indices = encode_global_features(states, seed, decl_id, max_states)

        # Apply same sampling to values
        if sample_indices is not None:
            values = values[sample_indices]

        return (seed, decl_id, features, values)

    except Exception as e:
        log(f"  ERROR processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_batch_to_file(path: Path, features: np.ndarray, values: np.ndarray, batch_idx: int) -> Path:
    """Save a batch to a numbered parquet file."""
    batch_path = path.parent / f"{path.stem}_batch{batch_idx:04d}.parquet"
    df = pd.DataFrame({
        "features": list(features),
        "value": values,
    })
    df.to_parquet(batch_path, index=False)
    return batch_path


def merge_batch_files(output_path: Path, batch_paths: list[Path]) -> None:
    """Merge batch files into a single output file."""
    if not batch_paths:
        return

    log(f"  Merging {len(batch_paths)} batch files...")
    dfs = []
    for bp in batch_paths:
        dfs.append(pd.read_parquet(bp))

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(output_path, index=False)

    # Clean up batch files
    for bp in batch_paths:
        bp.unlink()
    log(f"  Merged {len(combined):,} samples into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess solver2 data to global encoding")
    parser.add_argument(
        "--max-states-per-file",
        type=int,
        default=100000,
        help="Max states to sample per file (default: 100000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/solver2",
        help="Directory containing parquet files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--test-seeds",
        type=str,
        default="90-99",
        help="Seed range for test set (default: 90-99)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of files to process before saving (default: 50)",
    )
    args = parser.parse_args()

    # Parse test seed range
    test_start, test_end = map(int, args.test_seeds.split("-"))
    test_seeds = set(range(test_start, test_end + 1))

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    parquet_files = sorted(data_dir.glob("seed_*.parquet"))
    log(f"Found {len(parquet_files)} parquet files")
    log(f"Test seeds: {test_start}-{test_end}")
    log(f"Max states per file: {args.max_states_per_file:,}")
    log(f"Batch size: {args.batch_size} files")
    log(f"Memory: {get_memory_mb():.1f} MB")

    # Output paths
    train_path = data_dir / "train_global.parquet"
    test_path = data_dir / "test_global.parquet"

    # Remove existing files to start fresh
    if train_path.exists():
        train_path.unlink()
        log(f"Removed existing {train_path}")
    if test_path.exists():
        test_path.unlink()
        log(f"Removed existing {test_path}")

    # Process files in batches
    n_workers = args.workers or 1  # Default to 1 for memory safety
    log(f"\n=== Processing files ({n_workers} workers, batch={args.batch_size}) ===")

    train_features = []
    train_values = []
    test_features = []
    test_values = []
    train_batch_paths = []
    test_batch_paths = []
    train_batch_idx = 0
    test_batch_idx = 0
    train_total = 0
    test_total = 0

    file_args = [(str(f), args.max_states_per_file) for f in parquet_files]
    completed = 0
    t0 = time.time()

    def save_batch():
        """Save accumulated data to batch files and clear buffers."""
        nonlocal train_features, train_values, test_features, test_values
        nonlocal train_batch_idx, test_batch_idx, train_total, test_total
        nonlocal train_batch_paths, test_batch_paths

        if train_features:
            X = np.concatenate(train_features, axis=0)
            y = np.concatenate(train_values, axis=0)
            bp = save_batch_to_file(train_path, X, y, train_batch_idx)
            train_batch_paths.append(bp)
            train_batch_idx += 1
            train_total += len(X)
            log(f"    Saved train batch {train_batch_idx}: {len(X):,} samples (total: {train_total:,})")
            train_features.clear()
            train_values.clear()
            del X, y

        if test_features:
            X = np.concatenate(test_features, axis=0)
            y = np.concatenate(test_values, axis=0)
            bp = save_batch_to_file(test_path, X, y, test_batch_idx)
            test_batch_paths.append(bp)
            test_batch_idx += 1
            test_total += len(X)
            log(f"    Saved test batch {test_batch_idx}: {len(X):,} samples (total: {test_total:,})")
            test_features.clear()
            test_values.clear()
            del X, y

        gc.collect()

    # Process sequentially for memory safety
    for arg in file_args:
        result = process_file(arg)
        completed += 1

        if result is not None:
            seed, decl_id, features, values = result

            if seed in test_seeds:
                test_features.append(features)
                test_values.append(values)
            else:
                train_features.append(features)
                train_values.append(values)

        if completed % 20 == 0 or completed == len(parquet_files):
            elapsed = time.time() - t0
            rate = completed / elapsed
            log(
                f"  Processed {completed}/{len(parquet_files)} files "
                f"({elapsed:.1f}s, {rate:.1f} files/s, {get_memory_mb():.0f} MB)"
            )

        # Save batch periodically
        if completed % args.batch_size == 0:
            save_batch()

    # Save final batch
    save_batch()

    # Merge batch files
    log(f"\n=== Merging batch files ===")
    merge_batch_files(train_path, train_batch_paths)
    merge_batch_files(test_path, test_batch_paths)

    # Summary
    log(f"\n=== Summary ===")
    log(f"Train: {train_total:,} states from seeds 0-{test_start-1}")
    log(f"Test: {test_total:,} states from seeds {test_start}-{test_end}")
    log(f"Feature dimension: 240")

    if train_path.exists():
        log(f"Train file: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    if test_path.exists():
        log(f"Test file: {test_path} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
