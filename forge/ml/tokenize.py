"""
Tokenization pipeline for Domino training data.

Converts oracle parquet files to pre-tokenized numpy arrays with:
- Per-shard deterministic RNG (keyed by global_seed, shard_seed, decl_id)
- Train/val/test split based on seed bucket
- Output format matching DominoDataset expectations

Usage:
    from forge.ml.tokenize import tokenize_shards, get_split

    # Process all shards in a directory
    tokenize_shards(
        input_dir=Path("data/solver2"),
        output_dir=Path("data/tokenized"),
        global_seed=42,
        max_samples_per_shard=50000,
    )
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

# Import from forge.oracle
from forge.oracle.declarations import N_DECLS, NOTRUMP
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    is_in_called_suit,
    trick_rank,
)


# =============================================================================
# Constants
# =============================================================================

TOKEN_TYPE_CONTEXT = 0
TOKEN_TYPE_PLAYER0 = 1
TOKEN_TYPE_TRICK_P0 = 5
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}
MAX_TOKENS = 32
N_FEATURES = 12


# =============================================================================
# Trump Rank Table (precomputed)
# =============================================================================

def _get_trump_rank(domino_id: int, decl_id: int) -> int:
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
        TRUMP_RANK_TABLE[(_dom, _decl)] = _get_trump_rank(_dom, _decl)


# =============================================================================
# Split Assignment
# =============================================================================

def get_split(seed: int) -> str:
    """Assign a seed to a split based on deterministic bucketing.

    Args:
        seed: The deal seed from the parquet file

    Returns:
        One of 'train', 'val', or 'test'

    Split distribution:
        - train: 90% (buckets 0-899)
        - val: 5% (buckets 900-949)
        - test: 5% (buckets 950-999) - sacred, never touched during development
    """
    bucket = seed % 1000
    if bucket >= 950:
        return 'test'   # 5% - sacred
    elif bucket >= 900:
        return 'val'    # 5% - model selection, early stopping
    else:
        return 'train'  # 90%


# =============================================================================
# Per-Shard Tokenization
# =============================================================================

@dataclass
class ShardResult:
    """Result from processing a single shard."""
    tokens: np.ndarray     # (N, 32, 12) int8
    masks: np.ndarray      # (N, 32) int8
    targets: np.ndarray    # (N,) int8
    legal: np.ndarray      # (N, 7) int8
    qvals: np.ndarray      # (N, 7) int8
    teams: np.ndarray      # (N,) int8
    players: np.ndarray    # (N,) int8
    split: str             # 'train', 'val', or 'test'
    seed: int              # deal seed
    decl_id: int           # declaration id


@dataclass
class ShardProgress:
    """Progress info for a single shard, passed to progress callback."""
    file_index: int        # 0-based index of current file
    total_files: int       # total files to process
    file_path: str         # path to the shard file
    seed: int              # deal seed
    decl_id: int           # declaration id
    split: str             # 'train', 'val', or 'test'
    samples: int           # samples produced from this shard
    elapsed_time: float    # time to process this shard (seconds)
    cumulative_samples: dict[str, int]  # running totals by split


def process_shard(
    path: Path,
    global_seed: int,
    max_samples: int | None = None,
) -> ShardResult | None:
    """Process one parquet file with per-shard deterministic RNG.

    This is the **critical fix** for reproducibility:
    - Each shard's sampling is deterministic and independent of other shards
    - Same shard -> same samples, regardless of what else is in the dataset
    - Keyed by (global_seed, shard_seed, decl_id)

    Args:
        path: Path to parquet file
        global_seed: Global seed for the entire tokenization run
        max_samples: Maximum samples to take from this shard (None = all)

    Returns:
        ShardResult with int8 arrays, or None if processing failed
    """
    try:
        # Parse metadata from parquet
        pf = pq.ParquetFile(path)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())

        # Per-shard RNG keyed by (global_seed, seed, decl_id)
        # This ensures determinism regardless of processing order or other shards
        shard_rng = np.random.default_rng((global_seed, seed, decl_id))

        # Determine split from seed
        split = get_split(seed)

        # Load data
        df = pd.read_parquet(path)
        states = df["state"].values.astype(np.int64)

        q_cols = [f"q{i}" for i in range(7)]
        q_values_all = np.stack([df[c].values for c in q_cols], axis=1)

        # Deal hands
        hands = deal_from_seed(seed)
        hands_flat = np.array([hands[p][i] for p in range(4) for i in range(7)])

        n_states = len(states)

        # Sample with per-shard RNG
        if max_samples and n_states > max_samples:
            indices = shard_rng.choice(n_states, size=max_samples, replace=False)
            states = states[indices]
            q_values_all = q_values_all[indices]

        # Filter invalid states
        legal_masks = (q_values_all != -128).astype(np.float32)
        has_legal = legal_masks.any(axis=1)
        if not has_legal.any():
            return None

        states = states[has_legal]
        q_values_all = q_values_all[has_legal]
        legal_masks = legal_masks[has_legal]
        n_samples = len(states)

        # Parse state bits
        remaining = np.zeros((n_samples, 4), dtype=np.int64)
        for p in range(4):
            remaining[:, p] = (states >> (p * 7)) & 0x7F

        leader = ((states >> 28) & 0x3).astype(np.int64)
        trick_len = ((states >> 30) & 0x3).astype(np.int64)
        p0_local = ((states >> 32) & 0x7).astype(np.int64)
        p1_local = ((states >> 35) & 0x7).astype(np.int64)
        p2_local = ((states >> 38) & 0x7).astype(np.int64)

        current_player = ((leader + trick_len) % 4).astype(np.int64)

        # Compute targets
        team = (current_player % 2).astype(np.int64)
        q_int32 = q_values_all.astype(np.int32)
        q_for_argmax = np.where(team[:, np.newaxis] == 0, q_int32, -q_int32)
        q_masked = np.where(legal_masks > 0, q_for_argmax, -129)
        targets = q_masked.argmax(axis=1).astype(np.int8)

        # Store as int8
        q_int = q_values_all.astype(np.int8)
        team_int8 = team.astype(np.int8)

        # Build domino features
        domino_features = np.zeros((28, 5), dtype=np.int8)
        for i, global_id in enumerate(hands_flat):
            domino_features[i, 0] = DOMINO_HIGH[global_id]
            domino_features[i, 1] = DOMINO_LOW[global_id]
            domino_features[i, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
            domino_features[i, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
            domino_features[i, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]

        # Build tokens
        tokens = np.zeros((n_samples, MAX_TOKENS, N_FEATURES), dtype=np.int8)
        masks = np.zeros((n_samples, MAX_TOKENS), dtype=np.int8)

        normalized_leader = ((leader - current_player + 4) % 4).astype(np.int8)

        # Context token
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_id
        tokens[:, 0, 11] = normalized_leader
        masks[:, 0] = 1

        # Hand tokens (28 total: 4 players x 7 dominoes)
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

        # Trick tokens (up to 3)
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

        return ShardResult(
            tokens=tokens,
            masks=masks,
            targets=targets,
            legal=legal_masks.astype(np.int8),
            qvals=q_int,
            teams=team_int8,
            players=current_player.astype(np.int8),
            split=split,
            seed=seed,
            decl_id=decl_id,
        )

    except Exception as e:
        print(f"ERROR processing {path}: {e}", flush=True)
        return None


# =============================================================================
# Manifest
# =============================================================================

@dataclass
class SplitStats:
    """Statistics for a single split."""
    samples: int = 0
    files: int = 0


@dataclass
class TokenizationManifest:
    """Manifest for tokenized data."""
    version: int = 1
    created: str = ""
    generator: str = "forge.cli.tokenize"
    git_hash: str = ""
    source: str = ""
    global_seed: int = 42
    max_samples_per_shard: int | None = None
    splits: dict = field(default_factory=dict)
    token_format: str = "int8"
    token_shape: str = "[N, 32, 12]"

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        data = {
            "version": self.version,
            "created": self.created,
            "generator": self.generator,
            "git_hash": self.git_hash,
            "source": self.source,
            "sampling": {
                "global_seed": self.global_seed,
                "max_samples_per_shard": self.max_samples_per_shard,
            },
            "splits": self.splits,
            "token_format": self.token_format,
            "token_shape": self.token_shape,
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)


# =============================================================================
# Main Tokenization Pipeline
# =============================================================================

def _get_git_hash() -> str:
    """Get current git commit hash, or empty string if not in a git repo."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def find_parquet_files(input_dir: Path, max_files: int | None = None) -> list[Path]:
    """Find all parquet files in input directory.

    Args:
        input_dir: Directory to search
        max_files: Maximum files to return (for testing)

    Returns:
        Sorted list of parquet file paths
    """
    files = sorted(input_dir.glob("seed_*.parquet"))
    if max_files:
        files = files[:max_files]
    return files


def tokenize_shards(
    input_dir: Path,
    output_dir: Path,
    global_seed: int = 42,
    max_samples_per_shard: int | None = 50000,
    max_files: int | None = None,
    verbose: bool = True,
    progress_callback: Callable[[ShardProgress], None] | None = None,
) -> TokenizationManifest:
    """Tokenize all shards in input_dir to output_dir.

    This is the main entry point for tokenization.

    Args:
        input_dir: Directory containing seed_*.parquet files
        output_dir: Output directory for tokenized data
        global_seed: Global seed for sampling RNG
        max_samples_per_shard: Max samples per shard (None = all)
        max_files: Max files to process (for testing)
        verbose: Print progress
        progress_callback: Optional callback for per-shard progress (for wandb logging)

    Returns:
        TokenizationManifest with statistics
    """
    start_time = time.time()

    def log(msg: str) -> None:
        if verbose:
            elapsed = time.time() - start_time
            print(f"[{elapsed:7.1f}s] {msg}", flush=True)

    # Find files
    files = find_parquet_files(input_dir, max_files)
    log(f"Found {len(files)} parquet files")

    if not files:
        raise ValueError(f"No parquet files found in {input_dir}")

    # Collect by split
    split_data: dict[str, dict[str, list]] = {
        'train': {'tokens': [], 'masks': [], 'targets': [], 'legal': [], 'qvals': [], 'teams': [], 'players': []},
        'val': {'tokens': [], 'masks': [], 'targets': [], 'legal': [], 'qvals': [], 'teams': [], 'players': []},
        'test': {'tokens': [], 'masks': [], 'targets': [], 'legal': [], 'qvals': [], 'teams': [], 'players': []},
    }
    split_file_counts = {'train': 0, 'val': 0, 'test': 0}

    # Track cumulative samples for progress callback
    cumulative_samples = {'train': 0, 'val': 0, 'test': 0}

    # Process each file
    for i, f in enumerate(files):
        shard_start = time.time()
        result = process_shard(f, global_seed, max_samples_per_shard)
        shard_elapsed = time.time() - shard_start

        if result is not None:
            split = result.split
            split_data[split]['tokens'].append(result.tokens)
            split_data[split]['masks'].append(result.masks)
            split_data[split]['targets'].append(result.targets)
            split_data[split]['legal'].append(result.legal)
            split_data[split]['qvals'].append(result.qvals)
            split_data[split]['teams'].append(result.teams)
            split_data[split]['players'].append(result.players)
            split_file_counts[split] += 1

            shard_samples = len(result.targets)
            cumulative_samples[split] += shard_samples

            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(ShardProgress(
                    file_index=i,
                    total_files=len(files),
                    file_path=str(f),
                    seed=result.seed,
                    decl_id=result.decl_id,
                    split=split,
                    samples=shard_samples,
                    elapsed_time=shard_elapsed,
                    cumulative_samples=cumulative_samples.copy(),
                ))

        if verbose and ((i + 1) % 50 == 0 or i == len(files) - 1):
            totals = {s: sum(len(t) for t in split_data[s]['targets']) for s in split_data}
            log(f"  [{i+1}/{len(files)}] train={totals['train']:,} val={totals['val']:,} test={totals['test']:,}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest = TokenizationManifest(
        created=datetime.now(timezone.utc).isoformat(),
        git_hash=_get_git_hash(),
        source=str(input_dir),
        global_seed=global_seed,
        max_samples_per_shard=max_samples_per_shard,
    )

    # Save each split
    for split_name in ['train', 'val', 'test']:
        data = split_data[split_name]

        if not data['tokens']:
            log(f"WARNING: No data for {split_name} split!")
            continue

        # Concatenate
        tokens = np.concatenate(data['tokens'], axis=0)
        masks = np.concatenate(data['masks'], axis=0)
        targets = np.concatenate(data['targets'], axis=0)
        legal = np.concatenate(data['legal'], axis=0)
        qvals = np.concatenate(data['qvals'], axis=0)
        teams = np.concatenate(data['teams'], axis=0)
        players = np.concatenate(data['players'], axis=0)

        n_samples = len(targets)
        log(f"\n{split_name}: {n_samples:,} samples from {split_file_counts[split_name]} files")

        # Save to split directory
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        np.save(split_dir / "tokens.npy", tokens)
        np.save(split_dir / "masks.npy", masks)
        np.save(split_dir / "targets.npy", targets)
        np.save(split_dir / "legal.npy", legal)
        np.save(split_dir / "qvals.npy", qvals)
        np.save(split_dir / "teams.npy", teams)
        np.save(split_dir / "players.npy", players)

        # Update manifest
        manifest.splits[split_name] = {
            'samples': int(n_samples),
            'files': split_file_counts[split_name],
        }

        # Size report
        total_mb = sum(arr.nbytes for arr in [tokens, masks, targets, legal, qvals, teams, players]) / 1024 / 1024
        log(f"  Total size: {total_mb:.1f} MB")

        # Free memory
        del tokens, masks, targets, legal, qvals, teams, players
        gc.collect()

    # Save manifest
    manifest_path = output_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        f.write(manifest.to_yaml())
    log(f"\nManifest saved to {manifest_path}")

    log(f"\nDone! Tokenized data saved to {output_dir}")

    return manifest
