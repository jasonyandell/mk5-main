#!/usr/bin/env python3
"""
Enhanced preprocessing with strategic derived features.

Features include:
- Raw remaining dominoes (112)
- Trump count per player (4)
- Count points remaining per player (4)
- Hand strength per player (4)
- Player void masks (4 x 7 = 28)
- Current trick info (leader, trick_len, trick points)
- Declaration and game phase info

Usage:
    python scripts/solver2/preprocess_enhanced.py [--max-states-per-file 10000]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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


# Precompute domino properties
def _create_domino_tables():
    """Create lookup tables for domino properties."""
    DOMINOES = [(h, l) for h in range(7) for l in range(h + 1)]

    high = np.array([h for h, _ in DOMINOES], dtype=np.int8)
    low = np.array([l for _, l in DOMINOES], dtype=np.int8)
    is_double = high == low
    pip_sum = high + low

    count_points = np.zeros(28, dtype=np.int8)
    for d_id, (h, l) in enumerate(DOMINOES):
        if (h, l) in ((5, 5), (6, 4)):
            count_points[d_id] = 10
        elif (h, l) in ((5, 0), (4, 1), (3, 2)):
            count_points[d_id] = 5

    # is_trump[decl_id, domino_id] -> bool
    is_trump = np.zeros((10, 28), dtype=np.bool_)
    for decl in range(7):  # pip declarations
        for d_id, (h, l) in enumerate(DOMINOES):
            if h == decl or l == decl:
                is_trump[decl, d_id] = True
    for d_id, (h, l) in enumerate(DOMINOES):
        if h == l:  # doubles
            is_trump[7, d_id] = True  # doubles-trump
            is_trump[8, d_id] = True  # doubles-suit (nello)
    # decl 9 (no-trump): no trumps

    # Suit membership: suits[pip, domino_id] -> bool (does domino have this pip?)
    contains_pip = np.zeros((7, 28), dtype=np.bool_)
    for pip in range(7):
        for d_id, (h, l) in enumerate(DOMINOES):
            contains_pip[pip, d_id] = (h == pip or l == pip)

    return high, low, is_double, pip_sum, count_points, is_trump, contains_pip


HIGH, LOW, IS_DOUBLE, PIP_SUM, COUNT_POINTS, IS_TRUMP, CONTAINS_PIP = _create_domino_tables()


def encode_enhanced_features(
    states: np.ndarray,
    seed: int,
    decl_id: int,
    max_states: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Encode packed states into enhanced feature vectors.

    Features:
    - remaining[0-3]: 4 × 28 = 112 (global domino binary masks)
    - trump_count[0-3]: 4 (normalized 0-1)
    - count_points[0-3]: 4 (normalized 0-42)
    - hand_strength[0-3]: 4 (normalized pip sum)
    - voids[0-3][0-6]: 28 (player is void in suit)
    - score: 1 (normalized)
    - leader: 4 (one-hot)
    - trick_len: 4 (one-hot)
    - current_player_team: 1 (0=team0, 1=team1)
    - decl_id: 10 (one-hot)
    - tricks_completed: 1 (normalized 0-7)
    - team0_trump_advantage: 1 (normalized -7 to +7)
    - team0_count_advantage: 1 (normalized -42 to +42)

    Total: 112 + 4 + 4 + 4 + 28 + 1 + 4 + 4 + 1 + 10 + 1 + 1 + 1 = 175 features
    """
    sample_indices = None

    # Optionally sample
    if max_states is not None and len(states) > max_states:
        if rng is None:
            rng = np.random.default_rng(seed * 1000 + decl_id)
        sample_indices = rng.choice(len(states), size=max_states, replace=False)
        states = states[sample_indices]

    n = len(states)
    N_FEATURES = 175
    features = np.zeros((n, N_FEATURES), dtype=np.float32)

    # Get local→global mapping from seed
    hands = deal_from_seed(seed)

    # Build local→global lookup table
    local_to_global = np.zeros((4, 8), dtype=np.int32)
    for p in range(4):
        for local_idx in range(7):
            local_to_global[p, local_idx] = hands[p][local_idx]
        local_to_global[p, 7] = -1

    # Extract packed fields
    remaining_local = np.zeros((n, 4), dtype=np.uint8)
    for p in range(4):
        remaining_local[:, p] = (states >> (p * 7)) & 0x7F

    leader = (states >> 28) & 0x3
    trick_len = (states >> 30) & 0x3

    # Convert remaining to global domino masks
    remaining_global = np.zeros((n, 4, 28), dtype=np.bool_)
    for p in range(4):
        for local_idx in range(7):
            global_id = hands[p][local_idx]
            has_domino = ((remaining_local[:, p] >> local_idx) & 1).astype(np.bool_)
            remaining_global[:, p, global_id] = has_domino

    # Feature 0-111: Global remaining masks (4 players × 28 dominoes)
    offset = 0
    for p in range(4):
        features[:, offset:offset+28] = remaining_global[:, p, :].astype(np.float32)
        offset += 28  # offset = 112

    # Feature 112-115: Trump count per player
    is_trump_arr = IS_TRUMP[decl_id]  # (28,) bool
    for p in range(4):
        trump_count = np.sum(remaining_global[:, p, :] & is_trump_arr, axis=1)
        features[:, offset + p] = trump_count / 7.0
    offset += 4  # offset = 116

    # Feature 116-119: Count points remaining per player
    for p in range(4):
        points = np.sum(remaining_global[:, p, :] * COUNT_POINTS[np.newaxis, :], axis=1)
        features[:, offset + p] = points / 42.0
    offset += 4  # offset = 120

    # Feature 120-123: Hand strength per player (sum of pip values)
    for p in range(4):
        strength = np.sum(remaining_global[:, p, :] * PIP_SUM[np.newaxis, :], axis=1)
        features[:, offset + p] = strength / 84.0  # Max is ~84 for 7 dominoes
    offset += 4  # offset = 124

    # Feature 124-151: Void masks per player per suit (4 players × 7 suits)
    for p in range(4):
        for suit in range(7):
            # Player is void if they have no domino with this pip
            has_suit = np.any(remaining_global[:, p, :] & CONTAINS_PIP[suit], axis=1)
            features[:, offset] = (~has_suit).astype(np.float32)
            offset += 1  # offset = 152

    # Feature 152: Score (normalized)
    all_remaining = np.any(remaining_global, axis=1)  # (n, 28)
    played = ~all_remaining
    score = np.sum(played * COUNT_POINTS[np.newaxis, :], axis=1)
    features[:, offset] = score / 42.0
    offset += 1  # offset = 153

    # Feature 153-156: Leader (one-hot)
    row_idx = np.arange(n)
    features[row_idx, offset + leader] = 1.0
    offset += 4  # offset = 157

    # Feature 157-160: Trick length (one-hot)
    features[row_idx, offset + trick_len] = 1.0
    offset += 4  # offset = 161

    # Feature 161: Current player team (0=team0, 1=team1)
    current_player = (leader + trick_len) & 0x3
    features[:, offset] = (current_player & 1).astype(np.float32)
    offset += 1  # offset = 162

    # Feature 162-171: Declaration (one-hot)
    features[:, offset + decl_id] = 1.0
    offset += 10  # offset = 172

    # Feature 172: Tricks completed
    remaining_count = np.sum(remaining_global, axis=(1, 2))  # Total dominoes remaining
    tricks_completed = (28 - remaining_count - trick_len) // 4
    features[:, offset] = tricks_completed / 7.0
    offset += 1  # offset = 173

    # Feature 173: Team 0 trump advantage
    team0_trumps = np.sum(
        (remaining_global[:, 0, :] | remaining_global[:, 2, :]) & is_trump_arr,
        axis=1
    )
    team1_trumps = np.sum(
        (remaining_global[:, 1, :] | remaining_global[:, 3, :]) & is_trump_arr,
        axis=1
    )
    trump_advantage = team0_trumps.astype(np.float32) - team1_trumps.astype(np.float32)
    features[:, offset] = trump_advantage / 14.0  # Range [-7, +7] -> [-0.5, +0.5]
    offset += 1  # offset = 174

    # Feature 174: Team 0 count advantage
    team0_counts = np.sum(
        (remaining_global[:, 0, :] | remaining_global[:, 2, :]) * COUNT_POINTS[np.newaxis, :],
        axis=1
    )
    team1_counts = np.sum(
        (remaining_global[:, 1, :] | remaining_global[:, 3, :]) * COUNT_POINTS[np.newaxis, :],
        axis=1
    )
    count_advantage = team0_counts.astype(np.float32) - team1_counts.astype(np.float32)
    features[:, offset] = count_advantage / 84.0  # Range [-42, +42] -> [-0.5, +0.5]
    offset += 1  # offset = 175

    assert offset == N_FEATURES, f"Expected {N_FEATURES} features, got {offset}"

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
        values = df["V"].values.astype(np.float32) / 42.0

        features, sample_indices = encode_enhanced_features(states, seed, decl_id, max_states)

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

    for bp in batch_paths:
        bp.unlink()
    log(f"  Merged {len(combined):,} samples into {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess with enhanced features")
    parser.add_argument("--max-states-per-file", type=int, default=10000)
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--test-seeds", type=str, default="90-99")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--output-prefix", type=str, default="enhanced")
    args = parser.parse_args()

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
    log(f"Memory: {get_memory_mb():.1f} MB")

    train_path = data_dir / f"train_{args.output_prefix}.parquet"
    test_path = data_dir / f"test_{args.output_prefix}.parquet"

    if train_path.exists():
        train_path.unlink()
    if test_path.exists():
        test_path.unlink()

    log(f"\n=== Processing files ===")

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
            log(f"  Processed {completed}/{len(parquet_files)} files ({elapsed:.1f}s, {rate:.1f} files/s)")

        if completed % args.batch_size == 0:
            save_batch()

    save_batch()

    log(f"\n=== Merging batch files ===")
    merge_batch_files(train_path, train_batch_paths)
    merge_batch_files(test_path, test_batch_paths)

    log(f"\n=== Summary ===")
    log(f"Train: {train_total:,} states")
    log(f"Test: {test_total:,} states")
    log(f"Feature dimension: 175")

    if train_path.exists():
        log(f"Train file: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    if test_path.exists():
        log(f"Test file: {test_path} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
