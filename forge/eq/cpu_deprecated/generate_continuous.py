from __future__ import annotations
"""
======================================================================
DEPRECATED CPU PIPELINE - DO NOT USE
======================================================================
This module contains KNOWN BUGS (E[Q] collapse with high sample counts).
It is kept temporarily for reference only and will be deleted soon.

Use the GPU pipeline instead: forge/eq/generate_gpu.py
======================================================================
"""
import sys as _sys
if not _sys.flags.interactive:  # Allow interactive inspection
    raise RuntimeError(
        "\n" + "=" * 70 + "\n"
        "DEPRECATED CPU PIPELINE - DO NOT USE\n"
        + "=" * 70 + "\n"
        "This module contains KNOWN BUGS (E[Q] collapse with high sample counts).\n"
        "It is kept temporarily for reference only and will be deleted soon.\n"
        "\n"
        "Use the GPU pipeline instead: forge/eq/generate_gpu.py\n"
        + "=" * 70
    )
del _sys

"""Continuous Stage 2 E[Q] training data generation with monitoring.

Runs indefinitely, generating batches of games and appending to dataset.
Handles Ctrl+C gracefully - saves current state before exit.

Usage:
    python -m forge.eq.generate_continuous

    # Custom batch size and target
    python -m forge.eq.generate_continuous --batch-size 100 --target-games 100000

    # Resume from existing dataset (default output path)
    python -m forge.eq.generate_continuous
"""


import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from forge.eq import Stage1Oracle, generate_eq_game
from forge.eq.generate import ExplorationPolicy, PosteriorConfig
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
    batch_e_q_mean: list[torch.Tensor],
    batch_e_q_var: list[torch.Tensor],
    batch_legal_mask: list[torch.Tensor],
    batch_action_taken: list[int],
    batch_game_idx: list[int],
    batch_decision_idx: list[int],
    batch_is_val: list[bool],
    batch_lengths: list[int],
    batch_u_mean: list[float],
    batch_u_max: list[float],
    batch_ess: list[float],
    batch_max_w: list[float],
    batch_exploration_mode: list[int],
    batch_q_gap: list[float],
    games_generated: int,
    n_samples: int,
    seed: int,
    checkpoint: str,
    posterior_config: PosteriorConfig | None,
    exploration_policy: ExplorationPolicy | None,
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
    new_e_q_mean = torch.stack(batch_e_q_mean)
    new_e_q_var = torch.stack(batch_e_q_var)
    new_legal_mask = torch.stack(batch_legal_mask)
    new_action_taken = torch.tensor(batch_action_taken, dtype=torch.long)
    new_game_idx = torch.tensor(batch_game_idx, dtype=torch.long)
    new_decision_idx = torch.tensor(batch_decision_idx, dtype=torch.long)
    new_train_mask = torch.tensor([not v for v in batch_is_val], dtype=torch.bool)
    new_u_mean = torch.tensor(batch_u_mean, dtype=torch.float32)
    new_u_max = torch.tensor(batch_u_max, dtype=torch.float32)
    new_ess = torch.tensor(batch_ess, dtype=torch.float32)
    new_max_w = torch.tensor(batch_max_w, dtype=torch.float32)
    new_exploration_mode = torch.tensor(batch_exploration_mode, dtype=torch.int8)
    new_q_gap = torch.tensor(batch_q_gap, dtype=torch.float32)

    if existing is None:
        # Create new dataset
        return {
            "transcript_tokens": new_transcript_tokens,
            "transcript_lengths": new_transcript_lengths,
            "e_q_mean": new_e_q_mean,
            "e_q_var": new_e_q_var,
            "legal_mask": new_legal_mask,
            "action_taken": new_action_taken,
            "game_idx": new_game_idx,
            "decision_idx": new_decision_idx,
            "train_mask": new_train_mask,
            "u_mean": new_u_mean,
            "u_max": new_u_max,
            "ess": new_ess,
            "max_w": new_max_w,
            "exploration_mode": new_exploration_mode,
            "q_gap": new_q_gap,
            "metadata": {
                "version": "2.1",
                "n_games": games_generated,
                "n_samples": n_samples,
                "n_examples": len(batch_action_taken),
                "n_train": new_train_mask.sum().item(),
                "n_val": (~new_train_mask).sum().item(),
                "seed": seed,
                "checkpoint": checkpoint,
                "schema": {
                    "q_semantics": "minimax_value_to_go",
                    "q_units": "points",
                    "q_normalization": "raw",
                    "tokenizer_version": "transcript_v1",
                    "warning": "Do NOT apply softmax to e_q_mean - values are already point estimates",
                },
                "posterior": {
                    "enabled": bool(posterior_config and posterior_config.enabled),
                    "tau": posterior_config.tau if posterior_config else 10.0,
                    "beta": posterior_config.beta if posterior_config else 0.10,
                    "window_k": posterior_config.window_k if posterior_config else 8,
                    "delta": posterior_config.delta if posterior_config else 30.0,
                    "adaptive_k_enabled": posterior_config.adaptive_k_enabled if posterior_config else False,
                    "rejuvenation_enabled": posterior_config.rejuvenation_enabled if posterior_config else False,
                },
                "exploration": {
                    "enabled": exploration_policy is not None,
                    "temperature": exploration_policy.temperature if exploration_policy else 1.0,
                    "use_boltzmann": exploration_policy.use_boltzmann if exploration_policy else False,
                    "epsilon": exploration_policy.epsilon if exploration_policy else 0.0,
                    "blunder_rate": exploration_policy.blunder_rate if exploration_policy else 0.0,
                    "blunder_max_regret": exploration_policy.blunder_max_regret if exploration_policy else 5.0,
                },
                "total_time_seconds": time.time() - start_time,
                "avg_game_time": (time.time() - start_time) / games_generated,
                "generation_started": datetime.now().isoformat(),
            },
        }

    # Append to existing
    return {
        "transcript_tokens": torch.cat([existing["transcript_tokens"], new_transcript_tokens]),
        "transcript_lengths": torch.cat([existing["transcript_lengths"], new_transcript_lengths]),
        "e_q_mean": torch.cat([existing["e_q_mean"], new_e_q_mean]),
        "e_q_var": torch.cat([existing["e_q_var"], new_e_q_var]),
        "legal_mask": torch.cat([existing["legal_mask"], new_legal_mask]),
        "action_taken": torch.cat([existing["action_taken"], new_action_taken]),
        "game_idx": torch.cat([existing["game_idx"], new_game_idx]),
        "decision_idx": torch.cat([existing["decision_idx"], new_decision_idx]),
        "train_mask": torch.cat([existing["train_mask"], new_train_mask]),
        "u_mean": torch.cat([existing["u_mean"], new_u_mean]),
        "u_max": torch.cat([existing["u_max"], new_u_max]),
        "ess": torch.cat([existing["ess"], new_ess]),
        "max_w": torch.cat([existing["max_w"], new_max_w]),
        "exploration_mode": torch.cat([existing["exploration_mode"], new_exploration_mode]),
        "q_gap": torch.cat([existing["q_gap"], new_q_gap]),
        "metadata": {
            "n_games": existing["metadata"]["n_games"] + games_generated,
            "n_samples": n_samples,
            "n_examples": existing["metadata"]["n_examples"] + len(batch_action_taken),
            "n_train": existing["metadata"]["n_train"] + new_train_mask.sum().item(),
            "n_val": existing["metadata"]["n_val"] + (~new_train_mask).sum().item(),
            "seed": existing["metadata"].get("seed", seed),
            "checkpoint": existing["metadata"].get("checkpoint", checkpoint),
            "version": existing["metadata"].get("version", "2.1"),
            "schema": existing["metadata"].get(
                "schema",
                {
                    "q_semantics": "minimax_value_to_go",
                    "q_units": "points",
                    "q_normalization": "raw",
                    "tokenizer_version": "transcript_v1",
                    "warning": "Do NOT apply softmax to e_q_mean - values are already point estimates",
                },
            ),
            "posterior": existing["metadata"].get(
                "posterior",
                {
                    "enabled": bool(posterior_config and posterior_config.enabled),
                    "tau": posterior_config.tau if posterior_config else 10.0,
                    "beta": posterior_config.beta if posterior_config else 0.10,
                    "window_k": posterior_config.window_k if posterior_config else 8,
                    "delta": posterior_config.delta if posterior_config else 30.0,
                    "adaptive_k_enabled": posterior_config.adaptive_k_enabled if posterior_config else False,
                    "rejuvenation_enabled": posterior_config.rejuvenation_enabled if posterior_config else False,
                },
            ),
            "exploration": existing["metadata"].get(
                "exploration",
                {
                    "enabled": exploration_policy is not None,
                    "temperature": exploration_policy.temperature if exploration_policy else 1.0,
                    "use_boltzmann": exploration_policy.use_boltzmann if exploration_policy else False,
                    "epsilon": exploration_policy.epsilon if exploration_policy else 0.0,
                    "blunder_rate": exploration_policy.blunder_rate if exploration_policy else 0.0,
                    "blunder_max_regret": exploration_policy.blunder_max_regret if exploration_policy else 5.0,
                },
            ),
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
        help="Output path for dataset (resumes/appends if exists)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
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

    # Posterior weighting args (matches forge/eq/generate_dataset.py)
    posterior_group = parser.add_argument_group("Posterior weighting")
    posterior_group.add_argument(
        "--posterior",
        action="store_true",
        help="Enable posterior-weighted E[Q] marginalization",
    )
    posterior_group.add_argument(
        "--tau", type=float, default=10.0, help="Posterior temperature (default: 10.0)"
    )
    posterior_group.add_argument(
        "--beta", type=float, default=0.10, help="Uniform mix coefficient (default: 0.10)"
    )
    posterior_group.add_argument(
        "--window-k", type=int, default=8, help="Transcript scoring window (default: 8)"
    )
    posterior_group.add_argument(
        "--delta", type=float, default=30.0, help="Log-weight clipping threshold (default: 30.0)"
    )
    posterior_group.add_argument(
        "--adaptive-k",
        action="store_true",
        help="Enable adaptive window expansion when ESS low",
    )
    posterior_group.add_argument(
        "--rejuvenation",
        action="store_true",
        help="Enable particle rejuvenation when ESS critical",
    )

    # Exploration policy args (matches forge/eq/generate_dataset.py)
    explore_group = parser.add_argument_group("Exploration policy")
    explore_group.add_argument(
        "--explore",
        action="store_true",
        help="Enable exploration policy (mixed by default)",
    )
    explore_group.add_argument(
        "--explore-temp",
        type=float,
        default=3.0,
        help="Boltzmann temperature (default: 3.0)",
    )
    explore_group.add_argument(
        "--explore-eps",
        type=float,
        default=0.05,
        help="Epsilon-random rate (default: 0.05)",
    )
    explore_group.add_argument(
        "--explore-blunder",
        type=float,
        default=0.02,
        help="Blunder rate (default: 0.02)",
    )
    explore_group.add_argument(
        "--explore-blunder-regret",
        type=float,
        default=3.0,
        help="Max blunder regret in points (default: 3.0)",
    )
    explore_group.add_argument(
        "--explore-greedy",
        action="store_true",
        help="Use pure greedy exploration (overrides other explore options)",
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

    # Check GPU (WSL note: if CUDA is broken, stop and let the user fix it).
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA requested but not available.")
            print("  This is usually a WSL/CUDA/driver issue that requires user intervention.")
            print("  Do NOT fall back to CPU for anything performance-related; fix CUDA first.")
            return
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    # Load existing dataset
    existing = load_existing_dataset(output_path)
    current_n_games = existing["metadata"]["n_games"] if existing else 0
    if existing is not None and ("e_q_mean" not in existing or "e_q_var" not in existing):
        raise ValueError(
            f"Existing dataset at {output_path} does not match schema v2.1 "
            f"(expected keys 'e_q_mean'/'e_q_var'). Regenerate into a new file."
        )

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
    rng = np.random.default_rng(args.seed + current_n_games)

    # Build posterior config
    posterior_config = None
    if args.posterior:
        posterior_config = PosteriorConfig(
            enabled=True,
            tau=args.tau,
            beta=args.beta,
            window_k=args.window_k,
            delta=args.delta,
            adaptive_k_enabled=args.adaptive_k,
            rejuvenation_enabled=args.rejuvenation,
        )

    # Build exploration policy
    exploration_policy = None
    if args.explore:
        if args.explore_greedy:
            exploration_policy = ExplorationPolicy.greedy()
        else:
            exploration_policy = ExplorationPolicy.mixed_exploration(
                temperature=args.explore_temp,
                epsilon=args.explore_eps,
                blunder_rate=args.explore_blunder,
                blunder_max_regret=args.explore_blunder_regret,
                seed=args.seed,
            )

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
            batch_e_q_mean = []
            batch_e_q_var = []
            batch_legal_mask = []
            batch_action_taken = []
            batch_game_idx = []
            batch_decision_idx = []
            batch_is_val = []
            batch_lengths = []
            batch_u_mean = []
            batch_u_max = []
            batch_ess = []
            batch_max_w = []
            batch_exploration_mode = []
            batch_q_gap = []

            mode_to_int = {"greedy": 0, "boltzmann": 1, "epsilon": 2, "blunder": 3}

            for i in range(batch_size):
                game_idx = current_n_games + i
                game_seed = int(rng.integers(0, 2**31))
                # Derive independent RNG streams from the base game seed for determinism.
                seed_seq = np.random.SeedSequence(game_seed)
                world_ss, explore_ss = seed_seq.spawn(2)
                world_rng = np.random.default_rng(world_ss)
                explore_seed = int(explore_ss.generate_state(1)[0])
                hands = deal_from_seed(game_seed)
                decl_id = int(rng.integers(0, 10))
                is_val = rng.random() < args.val_fraction

                game_exploration_policy = exploration_policy
                if exploration_policy is not None and exploration_policy.seed is not None:
                    game_exploration_policy = ExplorationPolicy(
                        temperature=exploration_policy.temperature,
                        use_boltzmann=exploration_policy.use_boltzmann,
                        epsilon=exploration_policy.epsilon,
                        blunder_rate=exploration_policy.blunder_rate,
                        blunder_max_regret=exploration_policy.blunder_max_regret,
                        seed=explore_seed,
                    )

                record = generate_eq_game(
                    oracle,
                    hands,
                    decl_id,
                    n_samples=args.n_samples,
                    posterior_config=posterior_config,
                    exploration_policy=game_exploration_policy,
                    world_rng=world_rng,
                )

                for decision_idx, decision in enumerate(record.decisions):
                    batch_tokens.append(decision.transcript_tokens)
                    batch_lengths.append(decision.transcript_tokens.shape[0])
                    batch_e_q_mean.append(decision.e_q_mean)
                    # Optional fields (filled with zeros if missing)
                    if hasattr(decision, "e_q_var") and decision.e_q_var is not None:
                        batch_e_q_var.append(decision.e_q_var)
                        batch_u_mean.append(float(getattr(decision, "u_mean", 0.0)))
                        batch_u_max.append(float(getattr(decision, "u_max", 0.0)))
                    else:
                        batch_e_q_var.append(torch.zeros_like(decision.e_q_mean))
                        batch_u_mean.append(0.0)
                        batch_u_max.append(0.0)

                    diagnostics = getattr(decision, "diagnostics", None)
                    if diagnostics is not None:
                        batch_ess.append(float(diagnostics.ess))
                        batch_max_w.append(float(diagnostics.max_w))
                    else:
                        batch_ess.append(float(args.n_samples))
                        batch_max_w.append(1.0 / float(args.n_samples))

                    exploration = getattr(decision, "exploration", None)
                    if exploration is not None:
                        batch_exploration_mode.append(
                            mode_to_int.get(getattr(exploration, "selection_mode", "greedy"), 0)
                        )
                        batch_q_gap.append(float(getattr(exploration, "q_gap", 0.0)))
                    else:
                        batch_exploration_mode.append(0)
                        batch_q_gap.append(0.0)
                    batch_legal_mask.append(decision.legal_mask)
                    batch_action_taken.append(decision.action_taken)
                    batch_game_idx.append(game_idx)
                    batch_decision_idx.append(decision_idx)
                    batch_is_val.append(is_val)

            # Update dataset
            existing = append_batch(
                existing,
                batch_tokens,
                batch_e_q_mean,
                batch_e_q_var,
                batch_legal_mask,
                batch_action_taken,
                batch_game_idx,
                batch_decision_idx,
                batch_is_val,
                batch_lengths,
                batch_u_mean,
                batch_u_max,
                batch_ess,
                batch_max_w,
                batch_exploration_mode,
                batch_q_gap,
                batch_size,
                args.n_samples,
                args.seed,
                args.checkpoint,
                posterior_config,
                exploration_policy,
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
