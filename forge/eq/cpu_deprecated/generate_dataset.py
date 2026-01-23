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

"""Generate Stage 2 E[Q] training dataset.

Usage:
    # Basic generation (uniform averaging, greedy action selection)
    python -m forge.eq.generate_dataset --n-games 1000

    # Posterior weighting + exploration (recommended for production)
    python -m forge.eq.generate_dataset --n-games 1000 --posterior --explore

    # Custom posterior/exploration params
    python -m forge.eq.generate_dataset --n-games 1000 \
        --posterior --tau 8.0 --beta 0.07 --window-k 12 \
        --explore --explore-temp 3.0 --explore-eps 0.05

Generates training data for Stage 2 model:
- Each game produces 28 DecisionRecords
- Output is a dict with train/val split (90/10)
- Versioned output filenames prevent overwriting
"""


import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from forge.eq import Stage1Oracle, generate_eq_game


def log(msg: str, *, flush: bool = True) -> None:
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=flush)
from forge.eq.generate import (
    DecisionRecordV2,
    ExplorationPolicy,
    GameRecordV2,
    PosteriorConfig,
    generate_eq_games_batched,
)
from forge.eq.transcript_tokenize import MAX_TOKENS, N_FEATURES
from forge.oracle.rng import deal_from_seed


def generate_dataset(
    oracle: Stage1Oracle,
    n_games: int,
    n_samples: int = 100,
    seed: int = 42,
    val_fraction: float = 0.1,
    progress_interval: int = 10,
    posterior_config: PosteriorConfig | None = None,
    exploration_policy: ExplorationPolicy | None = None,
    checkpoint_path: str = "",
    game_batch_size: int = 1,
) -> dict:
    """Generate E[Q] training dataset.

    Args:
        oracle: Stage 1 oracle for Q-value queries
        n_games: Number of games to generate
        n_samples: Samples per decision for E[Q] marginalization
        seed: Random seed for reproducibility
        val_fraction: Fraction of games for validation (default 0.1)
        progress_interval: Print progress every N games
        posterior_config: Config for posterior-weighted marginalization (None = uniform)
        exploration_policy: Policy for action selection (None = greedy)
        checkpoint_path: Path to checkpoint (for metadata)
        game_batch_size: Number of games to generate per batched oracle step (default 1)

    Returns:
        Dict with keys:
            - transcript_tokens: (N, MAX_TOKENS, N_FEATURES) int tensor (padded)
            - transcript_lengths: (N,) int tensor (unpadded lengths)
            - e_q_mean: (N, 7) float tensor - E[Q] in POINTS (primary target)
            - e_q_var: (N, 7) float tensor - Var[Q] in points² (uncertainty target)
            - legal_mask: (N, 7) bool tensor
            - action_taken: (N,) int tensor (index into current-hand order at that step)
            - game_idx: (N,) int tensor (which game each example came from)
            - decision_idx: (N,) int tensor (0-27 within game)
            - train_mask: (N,) bool tensor (True for train, False for val)
            - player: (N,) int8 tensor (who made this decision, 0-3)
            - actual_outcome: (N,) float tensor (actual margin from here to end, for validation)
            - u_mean: (N,) float tensor (state-level mean uncertainty in points)
            - u_max: (N,) float tensor (state-level max uncertainty in points)
            - ess: (N,) float tensor (effective sample size)
            - max_w: (N,) float tensor (max weight)
            - exploration_mode: (N,) int tensor (0=greedy, 1=boltzmann, 2=epsilon, 3=blunder)
            - q_gap: (N,) float tensor (regret: Q_greedy - Q_taken, in points)
            - game_seeds: (n_games,) int64 tensor - seed used for deal_from_seed()
            - game_decl_ids: (n_games,) int8 tensor - declaration ID per game (0-9)
            - metadata: dict with schema + generation details

        Note:
            Q-values are in POINTS (roughly [-42, +42] range), NOT logits.
            Do NOT apply softmax to e_q_mean - it's already an interpretable point estimate.
            actual_outcome is the actual margin (my team - opp team) from decision to game end.
    """
    rng = np.random.default_rng(seed)

    # Collect all decisions
    all_transcript_tokens = []
    all_e_q_mean = []  # E[Q] in points (NOT logits)
    all_legal_mask = []
    all_action_taken = []
    all_game_idx = []
    all_decision_idx = []
    all_player = []  # Who made this decision (0-3)
    all_actual_outcome = []  # Actual margin from here to end (for validation)
    # Uncertainty fields (t42-64uj.6)
    all_e_q_var = []  # Var[Q] in points²
    all_u_mean = []
    all_u_max = []
    # Posterior diagnostics (t42-64uj.3)
    all_ess = []
    all_max_w = []
    # Exploration stats (t42-64uj.5)
    all_exploration_mode = []  # 0=greedy, 1=boltzmann, 2=epsilon, 3=blunder
    all_q_gap = []

    # Game metadata for shard self-description (t42-[NEW])
    all_game_seeds = []
    all_game_decl_ids = []

    # Aggregate exploration stats
    total_greedy = 0
    total_boltzmann = 0
    total_epsilon = 0
    total_blunder = 0
    total_q_gap = 0.0
    total_entropy = 0.0

    # Determine train/val split by game
    n_val_games = int(n_games * val_fraction)
    n_train_games = n_games - n_val_games
    game_is_val = [False] * n_train_games + [True] * n_val_games
    rng.shuffle(game_is_val)

    all_is_val = []

    start_time = time.time()
    game_times = []

    # Running stats for progress reporting
    running_ess_sum = 0.0
    running_ess_min = float("inf")
    running_q_min = float("inf")
    running_q_max = float("-inf")
    running_decisions = 0

    # Mode string to int mapping
    mode_to_int = {"greedy": 0, "boltzmann": 1, "epsilon": 2, "blunder": 3}

    if game_batch_size < 1:
        raise ValueError(f"game_batch_size must be >= 1, got {game_batch_size}")

    game_idx = 0
    while game_idx < n_games:
        batch_end = min(n_games, game_idx + game_batch_size)
        batch_size = batch_end - game_idx

        batch_hands: list[list[list[int]]] = []
        batch_decl_ids: list[int] = []
        batch_exploration_policies: list[ExplorationPolicy | None] = []
        batch_world_rngs: list[np.random.Generator] = []
        batch_game_indices = list(range(game_idx, batch_end))

        for gi in batch_game_indices:
            game_seed = int(rng.integers(0, 2**31))
            # Derive independent RNG streams from the base game seed for determinism.
            seed_seq = np.random.SeedSequence(game_seed)
            world_ss, explore_ss = seed_seq.spawn(2)
            explore_seed = int(explore_ss.generate_state(1)[0])
            batch_hands.append(deal_from_seed(game_seed))
            decl_id = int(rng.integers(0, 10))
            batch_decl_ids.append(decl_id)
            batch_world_rngs.append(np.random.default_rng(world_ss))

            # Track game metadata for shard self-description (t42-[NEW])
            all_game_seeds.append(game_seed)
            all_game_decl_ids.append(decl_id)

            if exploration_policy is not None:
                batch_exploration_policies.append(
                    ExplorationPolicy(
                        temperature=exploration_policy.temperature,
                        use_boltzmann=exploration_policy.use_boltzmann,
                        epsilon=exploration_policy.epsilon,
                        blunder_rate=exploration_policy.blunder_rate,
                        blunder_max_regret=exploration_policy.blunder_max_regret,
                        seed=explore_seed if exploration_policy.seed is not None else None,
                    )
                )
            else:
                batch_exploration_policies.append(None)

        batch_start_time = time.time()
        if batch_size == 1:
            records = [
                generate_eq_game(
                    oracle,
                    batch_hands[0],
                    batch_decl_ids[0],
                    n_samples=n_samples,
                    posterior_config=posterior_config,
                    exploration_policy=batch_exploration_policies[0],
                    world_rng=batch_world_rngs[0],
                )
            ]
        else:
            records = generate_eq_games_batched(
                oracle,
                batch_hands,
                batch_decl_ids,
                n_samples=n_samples,
                posterior_config=posterior_config,
                exploration_policies=batch_exploration_policies,
                world_rngs=batch_world_rngs,
            )
        batch_time = time.time() - batch_start_time
        for _ in range(batch_size):
            game_times.append(batch_time / batch_size)

        # Collect decisions
        for gi, record in zip(batch_game_indices, records):
            for decision_idx, decision in enumerate(record.decisions):
                all_transcript_tokens.append(decision.transcript_tokens)
                all_e_q_mean.append(decision.e_q_mean)
                all_legal_mask.append(decision.legal_mask)
                all_action_taken.append(decision.action_taken)
                all_game_idx.append(gi)
                all_decision_idx.append(decision_idx)
                all_is_val.append(game_is_val[gi])
                all_player.append(decision.player)
                all_actual_outcome.append(decision.actual_outcome if decision.actual_outcome is not None else 0.0)

                # Uncertainty fields (t42-64uj.6)
                if isinstance(decision, DecisionRecordV2) and decision.e_q_var is not None:
                    all_e_q_var.append(decision.e_q_var)
                    all_u_mean.append(decision.u_mean)
                    all_u_max.append(decision.u_max)
                    # Posterior diagnostics
                    if decision.diagnostics is not None:
                        all_ess.append(decision.diagnostics.ess)
                        all_max_w.append(decision.diagnostics.max_w)
                    else:
                        all_ess.append(float(n_samples))
                        all_max_w.append(1.0 / n_samples)
                    # Exploration stats
                    if decision.exploration is not None:
                        mode_int = mode_to_int.get(decision.exploration.selection_mode, 0)
                        all_exploration_mode.append(mode_int)
                        all_q_gap.append(decision.exploration.q_gap)
                        total_entropy += decision.exploration.action_entropy
                        # Count modes
                        if decision.exploration.selection_mode == "greedy":
                            total_greedy += 1
                        elif decision.exploration.selection_mode == "boltzmann":
                            total_boltzmann += 1
                        elif decision.exploration.selection_mode == "epsilon":
                            total_epsilon += 1
                        elif decision.exploration.selection_mode == "blunder":
                            total_blunder += 1
                        total_q_gap += decision.exploration.q_gap
                    else:
                        all_exploration_mode.append(0)  # greedy
                        all_q_gap.append(0.0)
                else:
                    # Fallback for non-V2 records
                    all_e_q_var.append(torch.zeros(7))
                    all_u_mean.append(0.0)
                    all_u_max.append(0.0)
                    all_ess.append(float(n_samples))
                    all_max_w.append(1.0 / n_samples)
                    all_exploration_mode.append(0)
                    all_q_gap.append(0.0)

                # Update running stats
                running_decisions += 1
                running_ess_sum += all_ess[-1]
                running_ess_min = min(running_ess_min, all_ess[-1])
                legal_q = decision.e_q_mean[decision.legal_mask]
                if len(legal_q) > 0:
                    running_q_min = min(running_q_min, legal_q.min().item())
                    running_q_max = max(running_q_max, legal_q.max().item())

        # Progress
        if batch_end % progress_interval == 0:
            elapsed = time.time() - start_time
            games_per_sec = batch_end / elapsed
            eta = (n_games - batch_end) / games_per_sec if games_per_sec > 0 else 0
            ess_mean = running_ess_sum / running_decisions if running_decisions > 0 else 0
            log(
                f"Game {batch_end:>5}/{n_games} | "
                f"{games_per_sec:.2f} g/s | "
                f"ESS: {ess_mean:.0f} (min {running_ess_min:.0f}) | "
                f"Q: [{running_q_min:.1f}, {running_q_max:.1f}] | "
                f"ETA: {eta:.0f}s"
            )

        game_idx = batch_end

    total_time = time.time() - start_time
    n_decisions = len(all_action_taken)

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
    e_q_mean = torch.stack(all_e_q_mean)  # E[Q] in points (NOT logits)
    legal_mask = torch.stack(all_legal_mask)
    action_taken = torch.tensor(all_action_taken, dtype=torch.long)
    game_idx_tensor = torch.tensor(all_game_idx, dtype=torch.long)
    decision_idx_tensor = torch.tensor(all_decision_idx, dtype=torch.long)
    train_mask = torch.tensor([not v for v in all_is_val], dtype=torch.bool)
    player = torch.tensor(all_player, dtype=torch.int8)
    actual_outcome = torch.tensor(all_actual_outcome, dtype=torch.float32)
    # Uncertainty fields (t42-64uj.6)
    e_q_var = torch.stack(all_e_q_var)  # Var[Q] in points²
    u_mean = torch.tensor(all_u_mean, dtype=torch.float32)
    u_max = torch.tensor(all_u_max, dtype=torch.float32)
    ess = torch.tensor(all_ess, dtype=torch.float32)
    max_w = torch.tensor(all_max_w, dtype=torch.float32)
    # Exploration stats
    exploration_mode = torch.tensor(all_exploration_mode, dtype=torch.int8)
    q_gap = torch.tensor(all_q_gap, dtype=torch.float32)

    # Compute summary stats (Q-values in points)
    legal_q = e_q_mean[legal_mask]
    q_min = legal_q.min().item() if len(legal_q) > 0 else 0.0
    q_max = legal_q.max().item() if len(legal_q) > 0 else 0.0
    q_mean = legal_q.mean().item() if len(legal_q) > 0 else 0.0

    # ESS distribution
    ess_sorted = torch.sort(ess).values
    n_ess = len(ess_sorted)
    ess_p10 = ess_sorted[int(n_ess * 0.1)].item() if n_ess > 0 else 0.0
    ess_p50 = ess_sorted[int(n_ess * 0.5)].item() if n_ess > 0 else 0.0

    # Metadata (schema v2.4 with decl_mode)
    metadata = {
        "version": "2.4",  # Bumped for decl_mode field
        "generated_at": datetime.now().isoformat(),
        "n_games": n_games,
        "n_samples": n_samples,
        "decl_mode": "random",  # "random" = 1 random decl per deal, "all" = all 10 decls per deal
        "n_examples": n_decisions,
        "n_train": train_mask.sum().item(),
        "n_val": (~train_mask).sum().item(),
        "seed": seed,
        "checkpoint": checkpoint_path,
        # Explicit schema semantics (t42-d6y1)
        "schema": {
            "q_semantics": "minimax_value_to_go",  # Q = expected points if both sides play optimally
            "q_units": "points",  # Values in game points, roughly [-42, +42]
            "q_normalization": "raw",  # No scaling applied (NOT normalized [-1, 1])
            "tokenizer_version": "transcript_v1",  # forge/eq/transcript_tokenize.py
            "warning": "Do NOT apply softmax to e_q_mean - values are already point estimates",
        },
        "checkpoint_type": "qval",  # Point-valued Q outputs
        "total_time_seconds": total_time,
        "avg_game_time": float(np.mean(game_times)),
        # Posterior config
        "posterior": {
            "enabled": posterior_config is not None and posterior_config.enabled,
            "tau": posterior_config.tau if posterior_config else 10.0,
            "beta": posterior_config.beta if posterior_config else 0.10,
            "window_k": posterior_config.window_k if posterior_config else 8,
            "delta": posterior_config.delta if posterior_config else 30.0,
            "adaptive_k_enabled": posterior_config.adaptive_k_enabled if posterior_config else False,
            "rejuvenation_enabled": posterior_config.rejuvenation_enabled if posterior_config else False,
        },
        # Exploration config
        "exploration": {
            "enabled": exploration_policy is not None,
            "temperature": exploration_policy.temperature if exploration_policy else 1.0,
            "use_boltzmann": exploration_policy.use_boltzmann if exploration_policy else False,
            "epsilon": exploration_policy.epsilon if exploration_policy else 0.0,
            "blunder_rate": exploration_policy.blunder_rate if exploration_policy else 0.0,
            "blunder_max_regret": exploration_policy.blunder_max_regret if exploration_policy else 5.0,
        },
        # Summary stats
        "summary": {
            "q_range": [q_min, q_max],
            "q_mean": q_mean,
            "ess_distribution": {
                "min": ess.min().item(),
                "mean": ess.mean().item(),
                "max": ess.max().item(),
                "p10": ess_p10,
                "p50": ess_p50,
            },
            "exploration_stats": {
                "greedy_rate": total_greedy / n_decisions if n_decisions > 0 else 1.0,
                "mean_q_gap": total_q_gap / n_decisions if n_decisions > 0 else 0.0,
                "mean_entropy": total_entropy / n_decisions if n_decisions > 0 else 0.0,
                "mode_counts": {
                    "greedy": total_greedy,
                    "boltzmann": total_boltzmann,
                    "epsilon": total_epsilon,
                    "blunder": total_blunder,
                },
            },
        },
    }

    # Convert game metadata to tensors
    game_seeds_tensor = torch.tensor(all_game_seeds, dtype=torch.int64)
    game_decl_ids_tensor = torch.tensor(all_game_decl_ids, dtype=torch.int8)

    return {
        "transcript_tokens": transcript_tokens,
        "transcript_lengths": transcript_lengths,
        # E[Q] targets (t42-d6y1: Q-values in POINTS, NOT logits)
        "e_q_mean": e_q_mean,  # E[Q] in points - the target for Stage 2
        "e_q_var": e_q_var,  # Var[Q] in points²
        # Other fields
        "legal_mask": legal_mask,
        "action_taken": action_taken,
        "game_idx": game_idx_tensor,
        "decision_idx": decision_idx_tensor,
        "train_mask": train_mask,
        # Player and actual outcome (t42-26dl)
        "player": player,  # Who made this decision (0-3)
        "actual_outcome": actual_outcome,  # Actual margin from here to end (for validation)
        # Uncertainty fields (t42-64uj.6)
        "u_mean": u_mean,  # State-level mean uncertainty (points)
        "u_max": u_max,  # State-level max uncertainty (points)
        # Posterior diagnostics (t42-64uj.3)
        "ess": ess,
        "max_w": max_w,
        # Exploration stats (t42-64uj.5)
        "exploration_mode": exploration_mode,
        "q_gap": q_gap,  # Regret in points
        # Game metadata for shard self-description (v2.3)
        "game_seeds": game_seeds_tensor,  # [n_games] - seed used for deal_from_seed()
        "game_decl_ids": game_decl_ids_tensor,  # [n_games] - declaration ID per game
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 2 E[Q] training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python -m forge.eq.generate_dataset --n-games 100

  # Posterior weighting + exploration (recommended)
  python -m forge.eq.generate_dataset --n-games 1000 --posterior --explore

  # Custom params
  python -m forge.eq.generate_dataset --n-games 1000 \\
      --posterior --tau 8.0 --beta 0.07 \\
      --explore --explore-temp 3.0 --explore-eps 0.05
""",
    )

    # Basic args
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
        help="Path to Stage 1 Q-value checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: auto-versioned filename)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--val-fraction", type=float, default=0.1, help="Fraction for validation"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--game-batch-size",
        type=int,
        default=1,
        help="Generate games in batches to reduce oracle overhead (default: 1)",
    )

    # Posterior weighting args (t42-64uj.3)
    posterior_group = parser.add_argument_group("Posterior weighting (t42-64uj.3)")
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

    # Exploration policy args (t42-64uj.5)
    explore_group = parser.add_argument_group("Exploration policy (t42-64uj.5)")
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

    # Build output path with versioning
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        n_games_str = f"{args.n_games // 1000}k" if args.n_games >= 1000 else str(args.n_games)
        output_path = Path(f"forge/data/eq_v2_{timestamp}_{n_games_str}g.pt")
    else:
        output_path = Path(args.output)

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
                seed=args.seed,  # Use main seed for reproducibility
            )

    log("=" * 60)
    log("Stage 2 E[Q] Training Data Generation")
    log("=" * 60)
    log(f"Games: {args.n_games} ({args.n_games * 28} decisions)")
    log(f"Samples/decision: {args.n_samples}")
    log(f"Game batch size: {args.game_batch_size}")
    log(f"Checkpoint: {args.checkpoint}")
    log(f"Output: {output_path}")
    log(f"Seed: {args.seed}")
    log(f"Val fraction: {args.val_fraction}")

    # Feature status
    if posterior_config:
        log("Posterior weighting: ENABLED")
        log(f"  τ={posterior_config.tau}, β={posterior_config.beta}, K={posterior_config.window_k}")
        if posterior_config.adaptive_k_enabled:
            log("  Adaptive K: ON")
        if posterior_config.rejuvenation_enabled:
            log("  Rejuvenation: ON")
    else:
        log("Posterior weighting: OFF (uniform averaging)")

    if exploration_policy:
        log("Exploration policy: ENABLED")
        if exploration_policy.use_boltzmann:
            log(f"  Boltzmann temp={exploration_policy.temperature}")
        if exploration_policy.epsilon > 0:
            log(f"  Epsilon={exploration_policy.epsilon}")
        if exploration_policy.blunder_rate > 0:
            log(f"  Blunder rate={exploration_policy.blunder_rate}, max_regret={exploration_policy.blunder_max_regret}")
    else:
        log("Exploration policy: OFF (greedy)")

    # Check GPU - fail fast if CUDA requested but unavailable
    if args.device == "cuda":
        if not torch.cuda.is_available():
            log("ERROR: CUDA requested but not available.")
            log("  - Check WSL2 CUDA drivers (may need reboot)")
            log("  - Or use --device cpu explicitly for slow CPU mode")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load oracle
    log("Loading Stage 1 oracle...")
    load_start = time.time()
    oracle = Stage1Oracle(args.checkpoint, device=args.device)
    log(f"Oracle loaded in {time.time() - load_start:.2f}s")

    # Generate dataset - progress every ~50 games, min 10, max 100
    progress_interval = max(10, min(100, args.n_games // 20))
    log(f"Starting generation (progress every {progress_interval} games)...")
    dataset = generate_dataset(
        oracle,
        n_games=args.n_games,
        n_samples=args.n_samples,
        seed=args.seed,
        val_fraction=args.val_fraction,
        progress_interval=progress_interval,
        posterior_config=posterior_config,
        exploration_policy=exploration_policy,
        checkpoint_path=args.checkpoint,
        game_batch_size=args.game_batch_size,
    )

    # Print summary
    meta = dataset["metadata"]
    log("=" * 60)
    log("Generation Complete")
    log("=" * 60)
    log(f"Total examples: {meta['n_examples']}")
    log(f"  Train: {meta['n_train']}")
    log(f"  Val: {meta['n_val']}")
    games_per_sec = meta['n_games'] / meta['total_time_seconds']
    log(f"Total time: {meta['total_time_seconds']:.1f}s ({games_per_sec:.2f} games/s)")
    log(f"Avg time/game: {meta['avg_game_time']:.2f}s")

    # Q-value validation
    summary = meta["summary"]
    log("Q-value stats:")
    q_range = summary["q_range"]
    log(f"  E[Q] range: [{q_range[0]:.1f}, {q_range[1]:.1f}] pts")
    log(f"  E[Q] mean: {summary['q_mean']:.1f} pts")

    # Check for issues
    legal_q = dataset["e_q_mean"][dataset["legal_mask"]]
    if torch.isnan(legal_q).any():
        log("  WARNING: E[Q] contains NaN in legal actions!")
    elif torch.isinf(legal_q).any():
        log("  WARNING: E[Q] contains Inf in legal actions!")
    else:
        log("  ✓ No NaN/Inf in legal actions")

    if q_range[0] < -50 or q_range[1] > 50:
        log(f"  WARNING: Q range outside expected [-50, 50] pts")
    else:
        log("  ✓ Q range within expected bounds")

    # ESS/posterior stats
    log("Posterior health:")
    ess_stats = summary["ess_distribution"]
    log(f"  ESS range: [{ess_stats['min']:.1f}, {ess_stats['max']:.1f}]")
    log(f"  ESS mean: {ess_stats['mean']:.1f}")
    log(f"  ESS p10/p50: {ess_stats['p10']:.1f} / {ess_stats['p50']:.1f}")

    # Uncertainty validation
    log("Uncertainty stats:")
    e_q_var = dataset["e_q_var"]
    legal_var = e_q_var[dataset["legal_mask"]]
    legal_std = torch.sqrt(torch.clamp(legal_var, min=0))

    if (legal_var < 0).any():
        log("  WARNING: Negative variance detected!")
    else:
        log("  ✓ All variances non-negative")

    log(f"  Std range: [{legal_std.min().item():.2f}, {legal_std.max().item():.2f}]")
    log(f"  U_mean range: [{dataset['u_mean'].min().item():.2f}, {dataset['u_mean'].max().item():.2f}]")

    # Exploration stats
    if exploration_policy:
        log("Exploration stats:")
        exp_stats = summary["exploration_stats"]
        log(f"  Greedy rate: {exp_stats['greedy_rate']:.1%}")
        log(f"  Mean Q-gap: {exp_stats['mean_q_gap']:.2f} pts")
        log(f"  Mean entropy: {exp_stats['mean_entropy']:.2f}")
        mode_counts = exp_stats["mode_counts"]
        log(
            f"  Modes: greedy={mode_counts['greedy']}, "
            f"boltzmann={mode_counts['boltzmann']}, "
            f"epsilon={mode_counts['epsilon']}, "
            f"blunder={mode_counts['blunder']}"
        )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Saving to {output_path}...")
    torch.save(dataset, output_path)
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    log(f"Saved: {output_path} ({file_size_mb:.1f} MB)")
    log("Done.")


if __name__ == "__main__":
    main()
