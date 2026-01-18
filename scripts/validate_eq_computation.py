#!/usr/bin/env python3
"""Validate E[Q] computation by sampling worlds and querying oracle.

This script verifies that the E[Q] values in a dataset match fresh computation.
It was created to debug a suspicious [5:4] lead over [5:5] (trump) in doubles-trump.

Finding: The E[Q] computation is correct. The oracle genuinely rates high trump
leads poorly at opening, which may or may not be correct strategy.

Usage:
    python scripts/validate_eq_computation.py [dataset_path]
"""

import sys
from pathlib import Path

import numpy as np
import torch

from forge.eq import Stage1Oracle
from forge.eq.sampling import sample_consistent_worlds
from forge.oracle.declarations import DOUBLES_TRUMP
from forge.oracle.rng import deal_from_seed


def domino_name(d_id: int) -> str:
    """Convert domino ID to [high:low] notation."""
    high = 0
    while (high + 1) * (high + 2) // 2 <= d_id:
        high += 1
    low = d_id - high * (high + 1) // 2
    return f"[{high}:{low}]"


def validate_game_0(dataset_path: str, n_samples: int = 100):
    """Validate E[Q] for game 0, decision 0 (opening lead)."""
    print(f"Loading dataset: {dataset_path}")
    dataset = torch.load(dataset_path, weights_only=False)
    meta = dataset["metadata"]

    print(f"Loading oracle: {meta['checkpoint']}")
    oracle = Stage1Oracle(meta["checkpoint"], device="cuda")

    # Reconstruct game 0's setup (accounting for RNG shuffle in generate_dataset)
    seed = meta["seed"]
    rng = np.random.default_rng(seed)

    # Replicate the train/val split shuffle that consumes RNG state
    n_games = meta["n_games"]
    n_val = int(n_games * 0.1)
    game_is_val = [False] * (n_games - n_val) + [True] * n_val
    rng.shuffle(game_is_val)  # This consumes RNG state!

    # Now get game 0's seed and decl
    game_seed = int(rng.integers(0, 2**31))
    decl_id = int(rng.integers(0, 10))

    hands = deal_from_seed(game_seed)
    p0_hand = sorted(hands[0])

    print(f"\nGame 0: seed={game_seed}, decl_id={decl_id}")
    print(f"P0 hand: {[domino_name(d) for d in p0_hand]}")

    # Sample consistent worlds
    print(f"\nSampling {n_samples} consistent worlds...")
    sampled = sample_consistent_worlds(
        my_player=0,
        my_hand=set(p0_hand),
        played=set(),
        hand_sizes=[7, 7, 7, 7],
        voids={0: set(), 1: set(), 2: set(), 3: set()},
        decl_id=decl_id,
        n_samples=n_samples,
    )

    # Query oracle on each world
    all_q = []
    for world in sampled:
        full_hands = [sorted(list(h)) for h in world]

        # CORRECT: Use local_idx (0-6) for remaining bitmask, not domino_id
        remaining = np.zeros((1, 4), dtype=np.int32)
        for p in range(4):
            for local_idx in range(7):  # All 7 dominoes remaining at opening
                remaining[0, p] |= (1 << local_idx)

        q = oracle.query_batch(
            worlds=[full_hands],
            game_state_info={
                "decl_id": DOUBLES_TRUMP if decl_id == 7 else decl_id,
                "leader": 0,
                "trick_plays": [],
                "remaining": remaining,
            },
            current_player=0,
        )[0]
        all_q.append(q)

    # Compute E[Q]
    all_q = torch.stack(all_q)
    fresh_e_q = all_q.mean(dim=0)
    fresh_std = all_q.std(dim=0)

    # Get dataset E[Q] (t42-d6y1: Q-values in points, NOT logits)
    e_q_mean = dataset["e_q_mean"]
    e_q_var = dataset.get("e_q_var")

    idx = (dataset["game_idx"] == 0).nonzero()[0][0].item()
    ds_e_q = e_q_mean[idx][:7]
    ds_var = e_q_var[idx][:7] if e_q_var is not None else torch.zeros(7)

    # Compare
    print("\n" + "=" * 60)
    print("Comparison: Fresh E[Q] vs Dataset E[Q]")
    print("=" * 60)
    print(f"{'Domino':<10} {'Fresh E[Q]':>12} {'Dataset E[Q]':>14} {'Diff':>10}")
    print("-" * 60)

    for i, d in enumerate(p0_hand):
        fresh = fresh_e_q[i].item()
        ds = ds_e_q[i].item()
        diff = ds - fresh
        fresh_s = fresh_std[i].item()
        ds_s = np.sqrt(max(0, ds_var[i].item()))
        print(f"{domino_name(d):<10} {fresh:+6.2f}±{fresh_s:4.1f}   {ds:+6.2f}±{ds_s:4.1f}   {diff:+6.2f}")

    max_diff = (ds_e_q[:7] - fresh_e_q).abs().max().item()
    print("-" * 60)
    print(f"Max absolute difference: {max_diff:.2f}")

    if max_diff < 5.0:
        print("\n✓ E[Q] computation verified - differences within sampling variance")
    else:
        print("\n⚠ Large differences detected - may indicate a bug")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # Find most recent eq_v2 dataset
        data_dir = Path("forge/data")
        datasets = sorted(data_dir.glob("eq_v2_*.pt"))
        if not datasets:
            print("No eq_v2 datasets found in forge/data/")
            sys.exit(1)
        dataset_path = str(datasets[-1])

    validate_game_0(dataset_path)
