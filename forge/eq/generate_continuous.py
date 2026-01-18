#!/usr/bin/env python3
"""Continuous E[Q] training data generation with monitoring.

Runs indefinitely, generating batches of games and appending to dataset.
Handles Ctrl+C gracefully - saves current state before exit.

Usage:
    python -m forge.eq.generate_continuous

    # Custom batch size and target
    python -m forge.eq.generate_continuous --batch-size 100 --target-games 100000

    # Resume from existing dataset
    python -m forge.eq.generate_continuous  # Automatically resumes if dataset exists
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from forge.eq import Stage1Oracle, generate_eq_game
from forge.eq.transcript_tokenize import MAX_TOKENS, N_FEATURES
from forge.oracle.rng import deal_from_seed


class GracefulExit(Exception):
    """Raised when Ctrl+C received."""
    pass


def signal_handler(signum, frame):
    """Handle Ctrl+C by raising GracefulExit."""
    raise GracefulExit()


def load_existing_dataset(path: Path) -> dict | None:
    """Load existing dataset if present, return None if not found."""
    if not path.exists():
        return None

    print(f"Loading existing dataset from {path}...", end=" ", flush=True)
    data = torch.load(path, weights_only=False)
    meta = data["metadata"]
    print(f"done ({meta['n_games']} games, {meta['n_examples']} examples)")
    return data


def save_dataset(data: dict, path: Path) -> None:
    """Save dataset with atomic write (temp file + rename)."""
    temp_path = path.with_suffix(".tmp")
    torch.save(data, temp_path)
    temp_path.rename(path)


def append_batch(
    existing: dict | None,
    batch_tokens: list[torch.Tensor],
    batch_e_logits: list[torch.Tensor],
    batch_legal_mask: list[torch.Tensor],
    batch_action_taken: list[int],
    batch_game_idx: list[int],
    batch_decision_idx: list[int],
    batch_is_val: list[bool],
    batch_lengths: list[int],
    games_generated: int,
    n_samples: int,
    start_time: float,
) -> dict:
    """Append new batch to existing dataset (or create new one)."""
    # Pad and stack new transcripts
    padded_transcripts = []
    for tokens in batch_tokens:
        seq_len = tokens.shape[0]
        if seq_len < MAX_TOKENS:
            padding = torch.zeros((MAX_TOKENS - seq_len, N_FEATURES), dtype=tokens.dtype)
            tokens = torch.cat([tokens, padding], dim=0)
        padded_transcripts.append(tokens)

    new_transcript_tokens = torch.stack(padded_transcripts)
    new_transcript_lengths = torch.tensor(batch_lengths, dtype=torch.long)
    new_e_logits = torch.stack(batch_e_logits)
    new_legal_mask = torch.stack(batch_legal_mask)
    new_action_taken = torch.tensor(batch_action_taken, dtype=torch.long)
    new_game_idx = torch.tensor(batch_game_idx, dtype=torch.long)
    new_decision_idx = torch.tensor(batch_decision_idx, dtype=torch.long)
    new_train_mask = torch.tensor([not v for v in batch_is_val], dtype=torch.bool)

    if existing is None:
        # Create new dataset
        return {
            "transcript_tokens": new_transcript_tokens,
            "transcript_lengths": new_transcript_lengths,
            "e_logits": new_e_logits,
            "legal_mask": new_legal_mask,
            "action_taken": new_action_taken,
            "game_idx": new_game_idx,
            "decision_idx": new_decision_idx,
            "train_mask": new_train_mask,
            "metadata": {
                "n_games": games_generated,
                "n_samples": n_samples,
                "n_examples": len(batch_action_taken),
                "n_train": new_train_mask.sum().item(),
                "n_val": (~new_train_mask).sum().item(),
                "seed": 42,
                "total_time_seconds": time.time() - start_time,
                "avg_game_time": (time.time() - start_time) / games_generated,
                "generation_started": datetime.now().isoformat(),
            },
        }

    # Append to existing
    return {
        "transcript_tokens": torch.cat([existing["transcript_tokens"], new_transcript_tokens]),
        "transcript_lengths": torch.cat([existing["transcript_lengths"], new_transcript_lengths]),
        "e_logits": torch.cat([existing["e_logits"], new_e_logits]),
        "legal_mask": torch.cat([existing["legal_mask"], new_legal_mask]),
        "action_taken": torch.cat([existing["action_taken"], new_action_taken]),
        "game_idx": torch.cat([existing["game_idx"], new_game_idx]),
        "decision_idx": torch.cat([existing["decision_idx"], new_decision_idx]),
        "train_mask": torch.cat([existing["train_mask"], new_train_mask]),
        "metadata": {
            "n_games": existing["metadata"]["n_games"] + games_generated,
            "n_samples": n_samples,
            "n_examples": existing["metadata"]["n_examples"] + len(batch_action_taken),
            "n_train": existing["metadata"]["n_train"] + new_train_mask.sum().item(),
            "n_val": existing["metadata"]["n_val"] + (~new_train_mask).sum().item(),
            "seed": existing["metadata"].get("seed", 42),
            "total_time_seconds": existing["metadata"]["total_time_seconds"] + (time.time() - start_time),
            "avg_game_time": (
                existing["metadata"]["total_time_seconds"] + (time.time() - start_time)
            ) / (existing["metadata"]["n_games"] + games_generated),
            "generation_started": existing["metadata"].get("generation_started", datetime.now().isoformat()),
            "last_updated": datetime.now().isoformat(),
        },
    }


def format_time(seconds: float) -> str:
    """Format seconds as human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def main():
    parser = argparse.ArgumentParser(description="Continuous E[Q] training data generation")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Games per batch (saves after each batch)",
    )
    parser.add_argument(
        "--target-games",
        type=int,
        default=100_000,
        help="Stop after reaching this many games (0 = run forever)",
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
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction for validation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()
    output_path = Path(args.output)

    # Set up signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("E[Q] Continuous Training Data Generation")
    print("=" * 60)
    print(f"Batch size: {args.batch_size} games")
    print(f"Target: {args.target_games if args.target_games > 0 else '∞'} games")
    print(f"Samples/decision: {args.n_samples}")
    print(f"Output: {args.output}")
    print(f"Press Ctrl+C to stop (saves before exit)")
    print()

    # Check GPU
    if args.device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    elif args.device == "cuda":
        print("WARNING: CUDA requested but not available, using CPU")
        args.device = "cpu"

    # Load existing dataset
    existing = load_existing_dataset(output_path)
    current_n_games = existing["metadata"]["n_games"] if existing else 0

    if current_n_games > 0:
        print(f"\nResuming from {current_n_games} games...")
    else:
        print("\nStarting fresh...")

    # Check if target already reached
    if args.target_games > 0 and current_n_games >= args.target_games:
        print(f"\nTarget already reached ({current_n_games} >= {args.target_games})")
        return

    # Load oracle
    print("\nLoading Stage 1 oracle...", end=" ", flush=True)
    load_start = time.time()
    oracle = Stage1Oracle(args.checkpoint, device=args.device)
    print(f"done ({time.time() - load_start:.2f}s)")

    # Initialize RNG with deterministic seed based on current game count
    # This ensures different games even when resuming
    rng = np.random.default_rng(42 + current_n_games)

    # Generation loop
    print("\n" + "─" * 60)
    session_start = time.time()
    session_games = 0
    batch_num = 0

    try:
        while args.target_games == 0 or current_n_games < args.target_games:
            batch_start = time.time()
            batch_num += 1

            # Determine batch size (may be smaller for final batch)
            games_remaining = args.target_games - current_n_games if args.target_games > 0 else args.batch_size
            batch_size = min(args.batch_size, games_remaining)

            # Collect batch
            batch_tokens = []
            batch_e_logits = []
            batch_legal_mask = []
            batch_action_taken = []
            batch_game_idx = []
            batch_decision_idx = []
            batch_is_val = []
            batch_lengths = []

            for i in range(batch_size):
                game_idx = current_n_games + i
                game_seed = int(rng.integers(0, 2**31))
                hands = deal_from_seed(game_seed)
                decl_id = int(rng.integers(0, 10))
                is_val = rng.random() < args.val_fraction

                record = generate_eq_game(oracle, hands, decl_id, n_samples=args.n_samples)

                for decision_idx, decision in enumerate(record.decisions):
                    batch_tokens.append(decision.transcript_tokens)
                    batch_lengths.append(decision.transcript_tokens.shape[0])
                    batch_e_logits.append(decision.e_logits)
                    batch_legal_mask.append(decision.legal_mask)
                    batch_action_taken.append(decision.action_taken)
                    batch_game_idx.append(game_idx)
                    batch_decision_idx.append(decision_idx)
                    batch_is_val.append(is_val)

            # Update dataset
            existing = append_batch(
                existing,
                batch_tokens,
                batch_e_logits,
                batch_legal_mask,
                batch_action_taken,
                batch_game_idx,
                batch_decision_idx,
                batch_is_val,
                batch_lengths,
                batch_size,
                args.n_samples,
                batch_start,
            )

            # Save
            save_dataset(existing, output_path)

            # Update counters
            current_n_games += batch_size
            session_games += batch_size
            batch_time = time.time() - batch_start
            games_per_sec = batch_size / batch_time

            # Progress report
            session_elapsed = time.time() - session_start
            session_gps = session_games / session_elapsed

            if args.target_games > 0:
                remaining = args.target_games - current_n_games
                eta = remaining / session_gps if session_gps > 0 else 0
                eta_str = f"ETA: {format_time(eta)}"
                progress = f"{current_n_games}/{args.target_games}"
            else:
                eta_str = "∞"
                progress = f"{current_n_games}"

            file_size = output_path.stat().st_size / 1024 / 1024

            print(
                f"Batch {batch_num:4d} | "
                f"Games: {progress:>12} | "
                f"{session_gps:5.1f} g/s | "
                f"{file_size:6.1f} MB | "
                f"{eta_str}"
            )

    except GracefulExit:
        print("\n" + "─" * 60)
        print("Ctrl+C received, saving and exiting...")

        # Dataset should already be saved after last batch
        # but save again just in case there's partial state
        if existing:
            save_dataset(existing, output_path)

    # Final summary
    print("\n" + "=" * 60)
    print("Session Complete")
    print("=" * 60)
    session_elapsed = time.time() - session_start
    print(f"Session duration: {format_time(session_elapsed)}")
    print(f"Games generated this session: {session_games}")
    print(f"Total games: {current_n_games}")
    print(f"Total examples: {existing['metadata']['n_examples']}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    if args.target_games > 0:
        if current_n_games >= args.target_games:
            print(f"\n✓ Target reached ({args.target_games} games)")
        else:
            print(f"\n⚠ Stopped early ({args.target_games - current_n_games} games remaining)")


if __name__ == "__main__":
    main()
