"""Shard loading utilities with caching and parallel loading.

This module wraps forge.oracle.schema with conveniences for analysis notebooks:
- load_seed(): Load specific (seed, decl_id) pair
- load_seeds(): Parallel loading of multiple seeds
- iterate_shards(): Memory-efficient iteration over all shards
- find_shard_files(): Discover shard files in a directory
- ShardCache: LRU cache for frequently accessed shards
"""

from __future__ import annotations

import re
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

from forge.oracle import schema

if TYPE_CHECKING:
    import pandas as pd

# Default data directory
DEFAULT_SHARD_DIR = Path("data/shards-standard")

# Shard filename pattern: seed_XXXXXXXX_decl_Y.parquet
SHARD_PATTERN = re.compile(r"seed_(\d{8})_decl_(\d)\.parquet")


def find_shard_files(
    base_dir: Path | str | None = None,
    seed: int | None = None,
    decl_id: int | None = None,
) -> list[Path]:
    """
    Find shard files matching criteria.

    Args:
        base_dir: Directory to search (default: data/shards-standard)
        seed: Filter to specific seed (optional)
        decl_id: Filter to specific declaration (optional)

    Returns:
        List of matching parquet file paths, sorted by (seed, decl_id)
    """
    if base_dir is None:
        base_dir = DEFAULT_SHARD_DIR
    base_dir = Path(base_dir)

    # Search in base_dir and subdirectories (train/val/test)
    paths = []
    for pattern_dir in [base_dir, base_dir / "train", base_dir / "val", base_dir / "test"]:
        if pattern_dir.exists():
            paths.extend(pattern_dir.glob("seed_*.parquet"))

    # Filter and sort
    results = []
    for path in paths:
        match = SHARD_PATTERN.match(path.name)
        if match:
            file_seed = int(match.group(1))
            file_decl = int(match.group(2))
            if seed is not None and file_seed != seed:
                continue
            if decl_id is not None and file_decl != decl_id:
                continue
            results.append((file_seed, file_decl, path))

    results.sort(key=lambda x: (x[0], x[1]))
    return [r[2] for r in results]


def load_seed(
    seed: int,
    decl_id: int | None = None,
    base_dir: Path | str | None = None,
) -> tuple["pd.DataFrame", int, int]:
    """
    Load shard(s) for a specific seed.

    Args:
        seed: Deal seed to load
        decl_id: Specific declaration (optional, loads first found if None)
        base_dir: Directory to search

    Returns:
        (df, seed, decl_id) - DataFrame with state, V, q0-q6 columns

    Raises:
        FileNotFoundError: If no matching shard found
    """
    paths = find_shard_files(base_dir, seed=seed, decl_id=decl_id)
    if not paths:
        raise FileNotFoundError(f"No shard found for seed={seed}, decl_id={decl_id}")
    return schema.load_file(paths[0])


def load_seeds(
    seeds: list[int],
    decl_id: int | None = None,
    base_dir: Path | str | None = None,
    parallel: bool = True,
    max_workers: int | None = None,
) -> "pd.DataFrame":
    """
    Load multiple seeds and concatenate into single DataFrame.

    Args:
        seeds: List of seeds to load
        decl_id: Specific declaration (optional)
        base_dir: Directory to search
        parallel: Use parallel loading (default True)
        max_workers: Number of workers for parallel loading

    Returns:
        Combined DataFrame with added 'seed' and 'decl_id' columns
    """
    import pandas as pd

    def load_one(seed: int) -> "pd.DataFrame":
        df, s, d = load_seed(seed, decl_id, base_dir)
        df["seed"] = s
        df["decl_id"] = d
        return df

    if parallel and len(seeds) > 1:
        dfs = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_one, s): s for s in seeds}
            for future in as_completed(futures):
                dfs.append(future.result())
        return pd.concat(dfs, ignore_index=True)
    else:
        dfs = [load_one(s) for s in seeds]
        return pd.concat(dfs, ignore_index=True)


def iterate_shards(
    base_dir: Path | str | None = None,
    max_shards: int | None = None,
) -> Iterator[tuple["pd.DataFrame", int, int]]:
    """
    Memory-efficient iteration over all shards.

    Yields:
        (df, seed, decl_id) for each shard file

    Args:
        base_dir: Directory to search
        max_shards: Maximum number of shards to yield (optional)
    """
    paths = find_shard_files(base_dir)
    for i, path in enumerate(paths):
        if max_shards is not None and i >= max_shards:
            break
        yield schema.load_file(path)


class ShardCache:
    """
    LRU cache for recently loaded shards.

    Usage:
        cache = ShardCache(max_entries=100)
        df = cache.get(seed=0, decl_id=5, base_dir="data/shards-standard")
    """

    def __init__(self, max_entries: int = 100):
        """
        Initialize cache.

        Args:
            max_entries: Maximum number of shards to cache
        """
        self.max_entries = max_entries
        self._cache: OrderedDict[tuple[int, int, str], "pd.DataFrame"] = OrderedDict()

    def get(
        self,
        seed: int,
        decl_id: int,
        base_dir: Path | str | None = None,
    ) -> "pd.DataFrame":
        """
        Get shard from cache, loading if necessary.

        Args:
            seed: Deal seed
            decl_id: Declaration ID
            base_dir: Directory to search

        Returns:
            DataFrame with state, V, q0-q6 columns
        """
        base_dir_str = str(base_dir or DEFAULT_SHARD_DIR)
        key = (seed, decl_id, base_dir_str)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        # Load and cache
        df, _, _ = load_seed(seed, decl_id, base_dir)
        self._cache[key] = df

        # Evict oldest if over capacity
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

        return df

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """Number of cached shards."""
        return len(self._cache)


def get_split(seed: int) -> str:
    """
    Determine train/val/test split for a seed.

    Uses deterministic bucketing: seed % 1000
    - 0-899: train (90%)
    - 900-949: val (5%)
    - 950-999: test (5%)

    Args:
        seed: Deal seed

    Returns:
        "train", "val", or "test"
    """
    bucket = seed % 1000
    if bucket < 900:
        return "train"
    elif bucket < 950:
        return "val"
    else:
        return "test"


def count_shards(base_dir: Path | str | None = None) -> dict[str, int]:
    """
    Count shards by split.

    Returns:
        Dict with keys "train", "val", "test", "total"
    """
    paths = find_shard_files(base_dir)
    counts = {"train": 0, "val": 0, "test": 0}
    for path in paths:
        match = SHARD_PATTERN.match(path.name)
        if match:
            seed = int(match.group(1))
            split = get_split(seed)
            counts[split] += 1
    counts["total"] = sum(counts.values())
    return counts
