#!/usr/bin/env python3
"""GPU-native E[Q] training data generation with per-seed files.

Generates one .pt file per game seed for maximum parallelism and resumability.
Uses GPU pipeline for ~100x speedup over CPU version.

Usage:
    python -m forge.cli.generate_eq_continuous --checkpoint model.ckpt
    python -m forge.cli.generate_eq_continuous --checkpoint model.ckpt --start-seed 1000
    python -m forge.cli.generate_eq_continuous --dry-run --start-seed 0
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from forge.eq import Stage1Oracle
from forge.eq.generate_gpu import generate_eq_games_gpu, PosteriorConfig
from forge.eq.collate import collate_game_record
from forge.eq.types import ExplorationPolicy
from forge.eq.transcript_tokenize import MAX_TOKENS, N_FEATURES
from forge.oracle.rng import deal_from_seed


class GracefulExit(Exception):
    """Raised when Ctrl+C received."""
    pass


def signal_handler(signum, frame):
    raise GracefulExit()


def get_split(seed: int) -> str:
    """Determine train/val/test split from seed.

    Uses seed % 1000 (matches Stage 1 tokenize.py):
    - 0-899: train (90%)
    - 900-949: val (5%)
    - 950-999: test (5%)
    """
    bucket = seed % 1000
    if bucket < 900:
        return "train"
    elif bucket < 950:
        return "val"
    else:
        return "test"


def find_missing_seeds(base_dir: Path, start_seed: int, limit: int | None = None) -> Iterator[tuple[int, Path]]:
    """Yield (seed, output_path) for missing seeds."""
    seed = start_seed
    count = 0
    while limit is None or count < limit:
        split = get_split(seed)
        path = base_dir / split / f"seed_{seed:08d}.pt"
        if not path.exists():
            yield (seed, path)
            count += 1
        seed += 1


def write_game_pt(examples: list[dict], seed: int, path: Path, metadata: dict) -> None:
    """Write single game to .pt with atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Pad and stack tensors
    padded_tokens = []
    for ex in examples:
        tokens = ex['transcript_tokens']
        seq_len = tokens.shape[0]
        if seq_len < MAX_TOKENS:
            padding = torch.zeros((MAX_TOKENS - seq_len, N_FEATURES), dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding], dim=0)
        padded_tokens.append(tokens)

    data = {
        'transcript_tokens': torch.stack(padded_tokens),
        'transcript_lengths': torch.tensor([ex['transcript_length'] for ex in examples], dtype=torch.long),
        'e_q_mean': torch.stack([ex['e_q_mean'] for ex in examples]),
        'e_q_var': torch.stack([ex['e_q_var'] if ex['e_q_var'] is not None else torch.zeros(7) for ex in examples]),
        'legal_mask': torch.stack([ex['legal_mask'] for ex in examples]),
        'action_taken': torch.tensor([ex['action_taken'] for ex in examples], dtype=torch.long),
        'game_idx': torch.tensor([seed] * len(examples), dtype=torch.long),
        'decision_idx': torch.tensor([ex['decision_idx'] for ex in examples], dtype=torch.long),
        'train_mask': torch.tensor([get_split(seed) == "train"] * len(examples), dtype=torch.bool),
        'u_mean': torch.tensor([ex.get('u_mean', 0.0) for ex in examples], dtype=torch.float32),
        'u_max': torch.tensor([ex.get('u_max', 0.0) for ex in examples], dtype=torch.float32),
        'ess': torch.tensor([ex.get('ess') or 0.0 for ex in examples], dtype=torch.float32),
        'max_w': torch.tensor([ex.get('max_w') or 0.0 for ex in examples], dtype=torch.float32),
        'exploration_mode': torch.tensor([ex.get('exploration_mode') or 0 for ex in examples], dtype=torch.int8),
        'q_gap': torch.tensor([ex.get('q_gap') or 0.0 for ex in examples], dtype=torch.float32),
        'metadata': {
            'seed': seed,
            'decl_id': seed % 10,
            **metadata,
        }
    }

    tmp_path = path.with_suffix('.tmp')
    torch.save(data, tmp_path)
    tmp_path.rename(path)  # Atomic


def main():
    parser = argparse.ArgumentParser(description="GPU E[Q] training data generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--start-seed", type=int, default=0, help="Start gap-filling from this seed")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU batch size (games per batch)")
    parser.add_argument("--n-samples", type=int, default=50, help="Worlds sampled per decision")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="data/eq-games", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Show missing seeds without generating")
    parser.add_argument("--limit", type=int, default=None, help="Stop after generating N games")

    # Posterior args
    posterior_group = parser.add_argument_group("Posterior weighting")
    posterior_group.add_argument("--posterior", action="store_true", help="Enable posterior weighting")
    posterior_group.add_argument("--posterior-window", type=int, default=4, help="Window K for posterior")
    posterior_group.add_argument("--posterior-tau", type=float, default=0.1, help="Temperature tau")
    posterior_group.add_argument("--posterior-mix", type=float, default=0.1, help="Uniform mix coefficient")

    # Exploration args
    explore_group = parser.add_argument_group("Exploration policy")
    explore_group.add_argument("--exploration", type=str, choices=["greedy", "epsilon_greedy", "boltzmann"],
                              default="greedy", help="Exploration policy")
    explore_group.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for epsilon_greedy")
    explore_group.add_argument("--temperature", type=float, default=2.0, help="Temperature for boltzmann")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("GPU E[Q] Training Data Generation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size} games")
    print(f"Samples/decision: {args.n_samples}")
    print(f"Start seed: {args.start_seed}")
    if args.limit:
        print(f"Limit: {args.limit} games")
    print()

    # Dry run mode
    if args.dry_run:
        print("DRY RUN - showing first 20 missing seeds:")
        for i, (seed, path) in enumerate(find_missing_seeds(output_dir, args.start_seed)):
            if i >= 20:
                print("...")
                break
            print(f"  {seed}: {path}")
        return

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available")
        return

    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load oracle
    print("\nLoading Stage 1 oracle...", end=" ", flush=True)
    oracle = Stage1Oracle(args.checkpoint, device=args.device)
    print("done")

    # Build configs
    posterior_config = None
    if args.posterior:
        posterior_config = PosteriorConfig(
            enabled=True,
            window_k=args.posterior_window,
            tau=args.posterior_tau,
            uniform_mix=args.posterior_mix,
        )

    exploration_policy = None
    if args.exploration == "epsilon_greedy":
        exploration_policy = ExplorationPolicy.epsilon_greedy(epsilon=args.epsilon)
    elif args.exploration == "boltzmann":
        exploration_policy = ExplorationPolicy.boltzmann(temperature=args.temperature)

    # Generation loop
    print("\n" + "─" * 60)
    session_start = time.time()
    games_generated = 0

    # Collect seeds for batching
    seed_iter = find_missing_seeds(output_dir, args.start_seed, args.limit)

    try:
        while True:
            # Collect batch of seeds
            batch_seeds = []
            batch_paths = []
            for seed, path in seed_iter:
                batch_seeds.append(seed)
                batch_paths.append(path)
                if len(batch_seeds) >= args.batch_size:
                    break

            if not batch_seeds:
                print("\nNo more missing seeds!")
                break

            batch_start = time.time()

            # Prepare batch inputs
            batch_hands = []
            batch_decl_ids = []
            for seed in batch_seeds:
                hands = deal_from_seed(seed)
                decl_id = seed % 10
                batch_hands.append(hands)
                batch_decl_ids.append(decl_id)

            # Generate games on GPU
            game_records = generate_eq_games_gpu(
                model=oracle.model,
                hands=batch_hands,
                decl_ids=batch_decl_ids,
                n_samples=args.n_samples,
                device=args.device,
                exploration_policy=exploration_policy,
                posterior_config=posterior_config,
            )

            # Collate and save each game
            for i, (seed, path, record) in enumerate(zip(batch_seeds, batch_paths, game_records)):
                examples = collate_game_record(record)
                metadata = {
                    'n_samples': args.n_samples,
                    'checkpoint': args.checkpoint,
                    'version': '2.1',
                    'schema': {
                        'q_semantics': 'minimax_value_to_go',
                        'q_units': 'points',
                    },
                    'posterior': {
                        'enabled': args.posterior,
                        'window_k': args.posterior_window if args.posterior else 0,
                        'tau': args.posterior_tau if args.posterior else 0.0,
                    },
                    'exploration': {
                        'policy': args.exploration,
                    },
                    'generated_at': datetime.now().isoformat(),
                }
                write_game_pt(examples, seed, path, metadata)

            games_generated += len(batch_seeds)
            batch_time = time.time() - batch_start
            games_per_sec = len(batch_seeds) / batch_time

            # Progress
            elapsed = time.time() - session_start
            overall_gps = games_generated / elapsed

            print(f"Batch: {len(batch_seeds)} games | Seeds {batch_seeds[0]}-{batch_seeds[-1]} | "
                  f"{games_per_sec:.1f} g/s | Total: {games_generated}")

    except GracefulExit:
        print("\n" + "─" * 60)
        print("Ctrl+C received, exiting...")

    # Summary
    print("\n" + "=" * 60)
    print("Session Complete")
    print("=" * 60)
    elapsed = time.time() - session_start
    print(f"Duration: {elapsed:.1f}s")
    print(f"Games generated: {games_generated}")
    if elapsed > 0:
        print(f"Average: {games_generated / elapsed:.1f} games/sec")


if __name__ == "__main__":
    main()
