#!/usr/bin/env python3
"""
Pre-tokenize parquet data for fast training.

One-time preprocessing step that converts parquet files to numpy memmap format.
Training then reads from memmap files, which is much faster than re-tokenizing.

Usage:
    # Pre-tokenize training data (seeds 0-89)
    python scripts/solver2/pretokenize.py --samples-per-file 50000 --output data/solver2/tokenized

    # Quick test with small subset
    python scripts/solver2/pretokenize.py --samples-per-file 10000 --max-files 10 --output scratch/tokenized
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.solver2.declarations import N_DECLS, NOTRUMP
from scripts.solver2.rng import deal_from_seed
from scripts.solver2.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    is_in_called_suit,
    trick_rank,
)


def log(msg: str) -> None:
    elapsed = time.time() - LOG_START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


LOG_START_TIME = time.time()


# =============================================================================
# Trump Rank Table
# =============================================================================

def get_trump_rank(domino_id: int, decl_id: int) -> int:
    if decl_id == NOTRUMP:
        return 7
    if not is_in_called_suit(domino_id, decl_id):
        return 7
    trumps = []
    for d in range(28):
        if is_in_called_suit(d, decl_id):
            tau = trick_rank(d, 7, decl_id)
            trumps.append((d, tau))
    trumps.sort(key=lambda x: -x[1])
    for rank, (d, _) in enumerate(trumps):
        if d == domino_id:
            return rank
    return 7


TRUMP_RANK_TABLE = {}
for _decl in range(N_DECLS):
    for _dom in range(28):
        TRUMP_RANK_TABLE[(_dom, _decl)] = get_trump_rank(_dom, _decl)


TOKEN_TYPE_CONTEXT = 0
TOKEN_TYPE_PLAYER0 = 1
TOKEN_TYPE_TRICK_P0 = 5
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}


# =============================================================================
# Tokenization (same as train_transformer.py but returns int8 for efficiency)
# =============================================================================

def process_file_vectorized(
    file_path: Path,
    max_samples: int | None,
    rng: np.random.Generator,
) -> tuple | None:
    """Process one parquet file with vectorized operations.

    Returns int8 arrays for efficiency (all features fit in 0-255).
    """
    try:
        pf = pq.ParquetFile(file_path)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())

        df = pd.read_parquet(file_path)
        states = df["state"].values.astype(np.int64)

        mv_cols = [f"mv{i}" for i in range(7)]
        q_values_all = np.stack([df[c].values for c in mv_cols], axis=1)

        hands = deal_from_seed(seed)
        hands_flat = np.array([hands[p][i] for p in range(4) for i in range(7)])

        n_states = len(states)

        if max_samples and n_states > max_samples:
            indices = rng.choice(n_states, size=max_samples, replace=False)
            states = states[indices]
            q_values_all = q_values_all[indices]

        legal_masks = (q_values_all != -128).astype(np.float32)
        has_legal = legal_masks.any(axis=1)
        if not has_legal.any():
            return None

        states = states[has_legal]
        q_values_all = q_values_all[has_legal]
        legal_masks = legal_masks[has_legal]
        n_samples = len(states)

        remaining = np.zeros((n_samples, 4), dtype=np.int64)
        for p in range(4):
            remaining[:, p] = (states >> (p * 7)) & 0x7F

        leader = ((states >> 28) & 0x3).astype(np.int64)
        trick_len = ((states >> 30) & 0x3).astype(np.int64)
        p0_local = ((states >> 32) & 0x7).astype(np.int64)
        p1_local = ((states >> 35) & 0x7).astype(np.int64)
        p2_local = ((states >> 38) & 0x7).astype(np.int64)

        current_player = ((leader + trick_len) % 4).astype(np.int64)

        # Compute targets in int32 to avoid overflow, then store as int8
        team = (current_player % 2).astype(np.int64)
        q_int32 = q_values_all.astype(np.int32)
        q_for_argmax = np.where(team[:, np.newaxis] == 0, q_int32, -q_int32)
        q_masked = np.where(legal_masks > 0, q_for_argmax, -129)
        targets = q_masked.argmax(axis=1).astype(np.int8)

        # Store Q-values and team as int8 for output
        q_int = q_values_all.astype(np.int8)
        team_int8 = team.astype(np.int8)

        domino_features = np.zeros((28, 5), dtype=np.int8)
        for i, global_id in enumerate(hands_flat):
            domino_features[i, 0] = DOMINO_HIGH[global_id]
            domino_features[i, 1] = DOMINO_LOW[global_id]
            domino_features[i, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
            domino_features[i, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
            domino_features[i, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]

        # Build tokens as int8 (all values fit in 0-255)
        MAX_TOKENS = 32
        N_FEATURES = 12
        tokens = np.zeros((n_samples, MAX_TOKENS, N_FEATURES), dtype=np.int8)
        masks = np.zeros((n_samples, MAX_TOKENS), dtype=np.int8)

        normalized_leader = ((leader - current_player + 4) % 4).astype(np.int8)

        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_id
        tokens[:, 0, 11] = normalized_leader
        masks[:, 0] = 1

        for p in range(4):
            normalized_player = ((p - current_player + 4) % 4).astype(np.int8)

            for local_idx in range(7):
                flat_idx = p * 7 + local_idx
                token_idx = 1 + flat_idx

                tokens[:, token_idx, 0] = domino_features[flat_idx, 0]
                tokens[:, token_idx, 1] = domino_features[flat_idx, 1]
                tokens[:, token_idx, 2] = domino_features[flat_idx, 2]
                tokens[:, token_idx, 3] = domino_features[flat_idx, 3]
                tokens[:, token_idx, 4] = domino_features[flat_idx, 4]
                tokens[:, token_idx, 5] = normalized_player
                tokens[:, token_idx, 6] = (normalized_player == 0).astype(np.int8)
                tokens[:, token_idx, 7] = (normalized_player == 2).astype(np.int8)
                tokens[:, token_idx, 8] = ((remaining[:, p] >> local_idx) & 1).astype(np.int8)
                tokens[:, token_idx, 9] = TOKEN_TYPE_PLAYER0 + p
                tokens[:, token_idx, 10] = decl_id
                tokens[:, token_idx, 11] = normalized_leader

                masks[:, token_idx] = 1

        trick_plays = [p0_local, p1_local, p2_local]
        trick_token_types = [TOKEN_TYPE_TRICK_P0, TOKEN_TYPE_TRICK_P0 + 1, TOKEN_TYPE_TRICK_P0 + 2]

        for trick_pos in range(3):
            has_play = (trick_len > trick_pos) & (trick_plays[trick_pos] < 7)
            if not has_play.any():
                continue

            token_idx = 29 + trick_pos
            local_idx = trick_plays[trick_pos]
            play_player = (leader + trick_pos) % 4

            for sample_idx in np.where(has_play)[0]:
                pp = int(play_player[sample_idx])
                li = int(local_idx[sample_idx])
                global_id = hands[pp][li]

                cp = int(current_player[sample_idx])
                normalized_pp = (pp - cp + 4) % 4

                tokens[sample_idx, token_idx, 0] = DOMINO_HIGH[global_id]
                tokens[sample_idx, token_idx, 1] = DOMINO_LOW[global_id]
                tokens[sample_idx, token_idx, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
                tokens[sample_idx, token_idx, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
                tokens[sample_idx, token_idx, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]
                tokens[sample_idx, token_idx, 5] = normalized_pp
                tokens[sample_idx, token_idx, 6] = 1 if normalized_pp == 0 else 0
                tokens[sample_idx, token_idx, 7] = 1 if normalized_pp == 2 else 0
                tokens[sample_idx, token_idx, 8] = 0
                tokens[sample_idx, token_idx, 9] = trick_token_types[trick_pos]
                tokens[sample_idx, token_idx, 10] = decl_id
                tokens[sample_idx, token_idx, 11] = int(normalized_leader[sample_idx])

                masks[sample_idx, token_idx] = 1

        # Return int8 arrays
        return (
            tokens,                         # (N, 32, 12) int8
            masks,                          # (N, 32) int8
            current_player.astype(np.int8), # (N,) int8
            targets,                        # (N,) int8
            legal_masks.astype(np.int8),    # (N, 7) int8
            q_int,                          # (N, 7) int8
            team_int8,                      # (N,) int8
        )

    except Exception as e:
        log(f"ERROR processing {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize parquet data")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for tokenized data")
    parser.add_argument("--samples-per-file", type=int, default=50000,
                        help="Max samples per parquet file")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max files to process (for testing)")
    parser.add_argument("--train-seeds", type=str, default="0-89",
                        help="Seed range for training data (e.g., '0-89')")
    parser.add_argument("--test-seeds", type=str, default="90-99",
                        help="Seed range for test data (e.g., '90-99')")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse seed ranges
    train_start, train_end = map(int, args.train_seeds.split("-"))
    test_start, test_end = map(int, args.test_seeds.split("-"))

    # Find all parquet files
    parquet_files = sorted(data_dir.glob("seed_*.parquet"))
    log(f"Found {len(parquet_files)} parquet files")

    # Split by seed
    train_files = []
    test_files = []
    for f in parquet_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                seed = int(parts[1])
                if train_start <= seed <= train_end:
                    train_files.append(f)
                elif test_start <= seed <= test_end:
                    test_files.append(f)
            except ValueError:
                pass

    if args.max_files:
        train_files = train_files[:args.max_files]
        test_files = test_files[:min(args.max_files // 9 + 1, len(test_files))]

    log(f"Train files: {len(train_files)}")
    log(f"Test files: {len(test_files)}")

    # Process train and test separately
    for split_name, files in [("train", train_files), ("test", test_files)]:
        if not files:
            continue

        log(f"\n=== Processing {split_name} split ({len(files)} files) ===")

        all_tokens = []
        all_masks = []
        all_players = []
        all_targets = []
        all_legal = []
        all_qvals = []
        all_teams = []

        for i, f in enumerate(files):
            result = process_file_vectorized(f, args.samples_per_file, rng)
            if result is not None:
                tokens, masks, players, targets, legal, qvals, teams = result
                all_tokens.append(tokens)
                all_masks.append(masks)
                all_players.append(players)
                all_targets.append(targets)
                all_legal.append(legal)
                all_qvals.append(qvals)
                all_teams.append(teams)

            if (i + 1) % 10 == 0 or i == len(files) - 1:
                total = sum(len(t) for t in all_targets)
                log(f"  [{i+1}/{len(files)}] Total samples: {total:,}")

        if not all_tokens:
            log(f"WARNING: No data for {split_name} split!")
            continue

        # Concatenate
        tokens = np.concatenate(all_tokens, axis=0)
        masks = np.concatenate(all_masks, axis=0)
        players = np.concatenate(all_players, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        legal = np.concatenate(all_legal, axis=0)
        qvals = np.concatenate(all_qvals, axis=0)
        teams = np.concatenate(all_teams, axis=0)

        n_samples = len(targets)
        log(f"\n{split_name}: {n_samples:,} samples")

        # Save to numpy files
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        log(f"Saving to {split_dir}...")

        np.save(split_dir / "tokens.npy", tokens)
        np.save(split_dir / "masks.npy", masks)
        np.save(split_dir / "players.npy", players)
        np.save(split_dir / "targets.npy", targets)
        np.save(split_dir / "legal.npy", legal)
        np.save(split_dir / "qvals.npy", qvals)
        np.save(split_dir / "teams.npy", teams)

        # Save metadata
        with open(split_dir / "meta.txt", "w") as f:
            f.write(f"samples={n_samples}\n")
            f.write(f"tokens_shape={tokens.shape}\n")
            f.write(f"samples_per_file={args.samples_per_file}\n")
            f.write(f"files={len(files)}\n")

        # Size report
        tokens_mb = tokens.nbytes / 1024 / 1024
        total_mb = sum(arr.nbytes for arr in [tokens, masks, players, targets, legal, qvals, teams]) / 1024 / 1024
        log(f"  Tokens: {tokens_mb:.1f} MB")
        log(f"  Total: {total_mb:.1f} MB")

        # Free memory
        del all_tokens, all_masks, all_players, all_targets, all_legal, all_qvals, all_teams
        del tokens, masks, players, targets, legal, qvals, teams
        gc.collect()

    log(f"\nDone! Tokenized data saved to {output_dir}")


if __name__ == "__main__":
    main()
