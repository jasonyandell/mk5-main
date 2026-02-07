"""
In-place slot shuffling for tokenized data.

Fixes the slot 0 bias caused by sorted hand ordering in deal_from_seed().
Shuffles within-hand domino ordering while preserving game semantics.

Usage:
    python -m forge.ml.shuffle_tokenized \
        --input data/tokenized-full \
        --output data/tokenized-shuffled

The shuffle is deterministic (seeded by sample index) and reversible.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np


def create_permutation(rng: np.random.Generator) -> np.ndarray:
    """Create a random permutation of 7 elements."""
    perm = np.arange(7)
    rng.shuffle(perm)
    return perm


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """Compute inverse permutation: inv[perm[i]] = i."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def shuffle_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    global_seed: int = 42,
    chunk_size: int = 100_000,
    verbose: bool = True,
) -> int:
    """Shuffle a single split (train/val/test).

    Args:
        input_dir: Directory containing the original tokenized data
        output_dir: Directory for shuffled output
        split: Split name ('train', 'val', 'test')
        global_seed: Seed for reproducibility
        chunk_size: Process this many samples at a time
        verbose: Print progress

    Returns:
        Number of samples processed
    """
    split_in = input_dir / split
    split_out = output_dir / split

    if not split_in.exists():
        if verbose:
            print(f"  Skipping {split} (directory not found)")
        return 0

    split_out.mkdir(parents=True, exist_ok=True)

    # Load arrays (memory-mapped for large files)
    tokens = np.load(split_in / "tokens.npy", mmap_mode='r')
    qvals = np.load(split_in / "qvals.npy", mmap_mode='r')
    legal = np.load(split_in / "legal.npy", mmap_mode='r')
    targets = np.load(split_in / "targets.npy", mmap_mode='r')
    players = np.load(split_in / "players.npy", mmap_mode='r')

    n_samples = len(tokens)
    if verbose:
        print(f"  {split}: {n_samples:,} samples")

    # Pre-allocate output arrays
    tokens_out = np.empty_like(tokens)
    qvals_out = np.empty_like(qvals)
    legal_out = np.empty_like(legal)
    targets_out = np.empty_like(targets)

    # Process in chunks to manage memory
    n_chunks = (n_samples + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_samples)
        chunk_len = end - start

        if verbose and (chunk_idx % 10 == 0 or chunk_idx == n_chunks - 1):
            print(f"    Chunk {chunk_idx+1}/{n_chunks} ({start:,}-{end:,})")

        # Load chunk data
        tokens_chunk = np.array(tokens[start:end])
        qvals_chunk = np.array(qvals[start:end])
        legal_chunk = np.array(legal[start:end])
        targets_chunk = np.array(targets[start:end])
        players_chunk = np.array(players[start:end])

        # Generate permutations for each sample and player
        # Shape: (chunk_len, 4, 7) - 4 players, 7 slots each
        perms = np.empty((chunk_len, 4, 7), dtype=np.int64)
        inv_perms = np.empty((chunk_len, 4, 7), dtype=np.int64)

        for i in range(chunk_len):
            sample_idx = start + i
            # Seed by (global_seed, sample_idx) for reproducibility
            rng = np.random.default_rng((global_seed, sample_idx))
            for p in range(4):
                perm = create_permutation(rng)
                perms[i, p] = perm
                inv_perms[i, p] = inverse_permutation(perm)

        # Shuffle tokens for each player's hand block
        # Token positions: 0=context, 1-7=P0, 8-14=P1, 15-21=P2, 22-28=P3, 29-31=tricks
        tokens_shuffled = tokens_chunk.copy()
        for p in range(4):
            hand_start = 1 + p * 7
            hand_end = hand_start + 7
            for i in range(chunk_len):
                # Reorder: new_token[j] = old_token[perm[j]]
                tokens_shuffled[i, hand_start:hand_end] = tokens_chunk[i, hand_start:hand_end][perms[i, p]]

        # Remap qvals, legal, and targets using current player's permutation
        qvals_shuffled = np.empty_like(qvals_chunk)
        legal_shuffled = np.empty_like(legal_chunk)
        targets_shuffled = np.empty_like(targets_chunk)

        for i in range(chunk_len):
            cp = players_chunk[i]  # Current player (0-3)
            perm = perms[i, cp]
            inv_perm = inv_perms[i, cp]

            # qvals: new_qvals[j] = old_qvals[perm[j]]
            # This puts Q(old domino at perm[j]) at new position j
            qvals_shuffled[i] = qvals_chunk[i][perm]

            # legal: same remapping
            legal_shuffled[i] = legal_chunk[i][perm]

            # target: old target pointed to old slot, need to find new slot
            # new_target = where the old target's domino ended up = inv_perm[old_target]
            old_target = targets_chunk[i]
            targets_shuffled[i] = inv_perm[old_target]

        # Store results
        tokens_out[start:end] = tokens_shuffled
        qvals_out[start:end] = qvals_shuffled
        legal_out[start:end] = legal_shuffled
        targets_out[start:end] = targets_shuffled

        # Free chunk memory
        del tokens_chunk, qvals_chunk, legal_chunk, targets_chunk, players_chunk
        del tokens_shuffled, qvals_shuffled, legal_shuffled, targets_shuffled
        del perms, inv_perms
        gc.collect()

    # Save shuffled arrays
    if verbose:
        print(f"    Saving shuffled arrays...")

    np.save(split_out / "tokens.npy", tokens_out)
    np.save(split_out / "qvals.npy", qvals_out)
    np.save(split_out / "legal.npy", legal_out)
    np.save(split_out / "targets.npy", targets_out)

    # Copy unchanged arrays
    for arr_name in ["masks", "teams", "players", "values"]:
        arr_path = split_in / f"{arr_name}.npy"
        if arr_path.exists():
            arr = np.load(arr_path)
            np.save(split_out / f"{arr_name}.npy", arr)
            del arr

    gc.collect()
    return n_samples


def shuffle_tokenized_data(
    input_dir: Path,
    output_dir: Path,
    global_seed: int = 42,
    verbose: bool = True,
) -> dict[str, int]:
    """Shuffle all tokenized data.

    Args:
        input_dir: Directory containing tokenized data (with train/val/test subdirs)
        output_dir: Output directory for shuffled data
        global_seed: Seed for reproducibility
        verbose: Print progress

    Returns:
        Dict mapping split names to sample counts
    """
    start_time = time.time()

    if verbose:
        print(f"Shuffling tokenized data:")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Seed:   {global_seed}")
        print()

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split in ["train", "val", "test"]:
        n = shuffle_split(input_dir, output_dir, split, global_seed, verbose=verbose)
        if n > 0:
            results[split] = n

    # Copy manifest and update it
    manifest_in = input_dir / "manifest.yaml"
    if manifest_in.exists():
        import yaml
        with open(manifest_in) as f:
            manifest = yaml.safe_load(f)

        manifest["shuffle_hand_order"] = True
        manifest["shuffle_seed"] = global_seed
        manifest["source_dir"] = str(input_dir)

        with open(output_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nDone! Elapsed: {elapsed:.1f}s")
        for split, n in results.items():
            print(f"  {split}: {n:,} samples")

    return results


def verify_shuffle(
    input_dir: Path,
    output_dir: Path,
    n_samples: int = 100,
    global_seed: int = 42,
) -> bool:
    """Verify shuffle by checking that reverse shuffle recovers original.

    Returns True if verification passes.
    """
    print(f"\nVerifying shuffle (checking {n_samples} samples)...")

    # Load original and shuffled data
    split = "train"
    tokens_orig = np.load(input_dir / split / "tokens.npy", mmap_mode='r')[:n_samples]
    qvals_orig = np.load(input_dir / split / "qvals.npy", mmap_mode='r')[:n_samples]
    targets_orig = np.load(input_dir / split / "targets.npy", mmap_mode='r')[:n_samples]
    players_orig = np.load(input_dir / split / "players.npy", mmap_mode='r')[:n_samples]

    tokens_shuf = np.load(output_dir / split / "tokens.npy", mmap_mode='r')[:n_samples]
    qvals_shuf = np.load(output_dir / split / "qvals.npy", mmap_mode='r')[:n_samples]
    targets_shuf = np.load(output_dir / split / "targets.npy", mmap_mode='r')[:n_samples]

    # Regenerate permutations and verify reverse mapping
    all_ok = True
    for i in range(n_samples):
        rng = np.random.default_rng((global_seed, i))
        perms = [create_permutation(rng) for _ in range(4)]

        # Check token shuffle
        for p in range(4):
            hand_start = 1 + p * 7
            hand_end = hand_start + 7
            inv = inverse_permutation(perms[p])
            recovered = tokens_shuf[i, hand_start:hand_end][inv]
            if not np.array_equal(recovered, tokens_orig[i, hand_start:hand_end]):
                print(f"  FAIL: Token recovery failed for sample {i}, player {p}")
                all_ok = False

        # Check qvals/targets using current player's permutation
        cp = players_orig[i]
        inv = inverse_permutation(perms[cp])

        recovered_qvals = qvals_shuf[i][inv]
        if not np.array_equal(recovered_qvals, qvals_orig[i]):
            print(f"  FAIL: Q-value recovery failed for sample {i}")
            all_ok = False

        recovered_target = perms[cp][targets_shuf[i]]
        if recovered_target != targets_orig[i]:
            print(f"  FAIL: Target recovery failed for sample {i}")
            all_ok = False

    if all_ok:
        print("  All checks passed!")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Shuffle tokenized data to fix slot 0 bias")
    parser.add_argument("--input", type=Path, required=True, help="Input tokenized data directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for shuffled data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verify", action="store_true", help="Run verification after shuffle")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    results = shuffle_tokenized_data(
        input_dir=args.input,
        output_dir=args.output,
        global_seed=args.seed,
        verbose=not args.quiet,
    )

    if args.verify:
        verify_shuffle(args.input, args.output, global_seed=args.seed)


if __name__ == "__main__":
    main()
