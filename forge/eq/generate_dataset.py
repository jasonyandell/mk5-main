#!/usr/bin/env python3
"""Generate E[Q] training dataset.

Usage:
    python -m forge.eq.generate_dataset --n-games 1000 --output forge/data/eq_train.pt

Generates training data for Stage 2 model:
- Each game produces 28 DecisionRecords
- Output is a dict with train/val split (90/10)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from forge.eq import Stage1Oracle, generate_eq_game
from forge.eq.transcript_tokenize import MAX_TOKENS, N_FEATURES
from forge.oracle.rng import deal_from_seed


def generate_dataset(
    oracle: Stage1Oracle,
    n_games: int,
    n_samples: int = 100,
    seed: int = 42,
    val_fraction: float = 0.1,
    progress_interval: int = 10,
) -> dict:
    """Generate E[Q] training dataset.

    Args:
        oracle: Stage 1 oracle for Q-value queries
        n_games: Number of games to generate
        n_samples: Samples per decision for E[Q] marginalization
        seed: Random seed for reproducibility
        val_fraction: Fraction of games for validation (default 0.1)
        progress_interval: Print progress every N games

    Returns:
        Dict with keys:
            - transcript_tokens: (N, seq_len, feat_dim) int tensor
            - e_logits: (N, 7) float tensor
            - legal_mask: (N, 7) bool tensor
            - action_taken: (N,) int tensor
            - game_idx: (N,) int tensor (which game each example came from)
            - decision_idx: (N,) int tensor (0-27 within game)
            - train_mask: (N,) bool tensor (True for train, False for val)
            - metadata: dict with generation info
    """
    rng = np.random.default_rng(seed)

    # Collect all decisions
    all_transcript_tokens = []
    all_e_logits = []
    all_legal_mask = []
    all_action_taken = []
    all_game_idx = []
    all_decision_idx = []

    # Determine train/val split by game
    n_val_games = int(n_games * val_fraction)
    n_train_games = n_games - n_val_games
    game_is_val = [False] * n_train_games + [True] * n_val_games
    rng.shuffle(game_is_val)

    all_is_val = []

    start_time = time.time()
    game_times = []

    for game_idx in range(n_games):
        # Generate random seed for this game (convert to Python int for Random)
        game_seed = int(rng.integers(0, 2**31))
        hands = deal_from_seed(game_seed)
        decl_id = rng.integers(0, 10)  # Random declaration

        game_start = time.time()
        record = generate_eq_game(oracle, hands, decl_id, n_samples=n_samples)
        game_time = time.time() - game_start
        game_times.append(game_time)

        # Collect decisions
        for decision_idx, decision in enumerate(record.decisions):
            all_transcript_tokens.append(decision.transcript_tokens)
            all_e_logits.append(decision.e_logits)
            all_legal_mask.append(decision.legal_mask)
            all_action_taken.append(decision.action_taken)
            all_game_idx.append(game_idx)
            all_decision_idx.append(decision_idx)
            all_is_val.append(game_is_val[game_idx])

        # Progress
        if (game_idx + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            games_per_sec = (game_idx + 1) / elapsed
            eta = (n_games - game_idx - 1) / games_per_sec
            print(
                f"  Game {game_idx + 1}/{n_games} | "
                f"{games_per_sec:.1f} games/s | "
                f"ETA: {eta:.0f}s"
            )

    total_time = time.time() - start_time

    # Pad transcript tokens to MAX_TOKENS and stack
    padded_transcripts = []
    transcript_lengths = []
    for tokens in all_transcript_tokens:
        seq_len = tokens.shape[0]
        transcript_lengths.append(seq_len)
        if seq_len < MAX_TOKENS:
            padding = torch.zeros((MAX_TOKENS - seq_len, N_FEATURES), dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding], dim=0)
        padded_transcripts.append(tokens)
    transcript_tokens = torch.stack(padded_transcripts)
    transcript_lengths = torch.tensor(transcript_lengths, dtype=torch.long)
    e_logits = torch.stack(all_e_logits)
    legal_mask = torch.stack(all_legal_mask)
    action_taken = torch.tensor(all_action_taken, dtype=torch.long)
    game_idx_tensor = torch.tensor(all_game_idx, dtype=torch.long)
    decision_idx_tensor = torch.tensor(all_decision_idx, dtype=torch.long)
    train_mask = torch.tensor([not v for v in all_is_val], dtype=torch.bool)

    # Metadata
    metadata = {
        "n_games": n_games,
        "n_samples": n_samples,
        "n_examples": len(all_action_taken),
        "n_train": train_mask.sum().item(),
        "n_val": (~train_mask).sum().item(),
        "seed": seed,
        "total_time_seconds": total_time,
        "avg_game_time": np.mean(game_times),
    }

    return {
        "transcript_tokens": transcript_tokens,
        "transcript_lengths": transcript_lengths,
        "e_logits": e_logits,
        "legal_mask": legal_mask,
        "action_taken": action_taken,
        "game_idx": game_idx_tensor,
        "decision_idx": decision_idx_tensor,
        "train_mask": train_mask,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate E[Q] training dataset")
    parser.add_argument(
        "--n-games", type=int, default=1000, help="Number of games to generate"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Samples per decision for E[Q] marginalization",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
        help="Path to Stage 1 Q-value checkpoint (outputs expected points)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="forge/data/eq_dataset.pt",
        help="Output path for dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Fraction for validation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda/cpu)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("E[Q] Training Data Generation")
    print("=" * 60)
    print(f"Games: {args.n_games}")
    print(f"Samples/decision: {args.n_samples}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"Val fraction: {args.val_fraction}")
    print()

    # Check GPU
    if args.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    elif args.device == "cuda":
        print("WARNING: CUDA requested but not available, using CPU")
        args.device = "cpu"

    # Load oracle
    print("\nLoading Stage 1 oracle...", end=" ", flush=True)
    load_start = time.time()
    oracle = Stage1Oracle(args.checkpoint, device=args.device)
    print(f"done ({time.time() - load_start:.2f}s)")

    # Generate dataset
    print(f"\nGenerating {args.n_games} games...")
    dataset = generate_dataset(
        oracle,
        n_games=args.n_games,
        n_samples=args.n_samples,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )

    # Print summary
    meta = dataset["metadata"]
    print()
    print("=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Total examples: {meta['n_examples']}")
    print(f"  Train: {meta['n_train']}")
    print(f"  Val: {meta['n_val']}")
    print(f"Total time: {meta['total_time_seconds']:.1f}s")
    print(f"Avg time/game: {meta['avg_game_time']:.2f}s")
    print()

    # Validate (Q-values should be expected points in ~[-42, 42] range)
    print("Validation:")
    print(f"  transcript_tokens shape: {dataset['transcript_tokens'].shape}")
    print(f"  e_logits shape: {dataset['e_logits'].shape}")

    # Only check legal actions (padded/illegal slots are -inf by design)
    legal_q = dataset['e_logits'][dataset['legal_mask']]
    q_min = legal_q.min().item()
    q_max = legal_q.max().item()
    print(f"  E[Q] range (legal only): [{q_min:.1f}, {q_max:.1f}] pts")
    print(f"  E[Q] mean: {legal_q.mean().item():.1f} pts")
    print(f"  legal_mask sum (avg): {dataset['legal_mask'].float().sum(dim=1).mean():.1f} legal/decision")

    # Check for issues in legal actions
    if torch.isnan(legal_q).any():
        print("  WARNING: E[Q] contains NaN in legal actions!")
    if torch.isinf(legal_q).any():
        print("  WARNING: E[Q] contains Inf in legal actions!")
    if (legal_q == 0).all():
        print("  WARNING: E[Q] are all zeros!")
    # Sanity check for point-valued Q (expected game outcome is roughly -42 to +42)
    if q_min < -50 or q_max > 50:
        print(f"  WARNING: E[Q] range [{q_min:.1f}, {q_max:.1f}] outside expected [-50, 50] pts")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    print(f"\nSaved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
