"""MCTS-based AlphaZero-style training for Zeb.

Uses MCTS to generate training targets, trains network to predict:
1. Policy: MCTS visit distribution (over 7 hand slots)
2. Value: Game outcome

This provides much more stable learning than pure REINFORCE.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .mcts_self_play import play_games_with_mcts, mcts_examples_to_zeb_tensors
from .batched_mcts import play_games_batched
from .model import ZebModel, get_model_config
from .observation import observe, N_HAND_SLOTS
from .types import ZebGameState, GamePhase, BidState


def train_epoch(
    model: ZebModel,
    optimizer: torch.optim.Optimizer,
    tokens: Tensor,
    masks: Tensor,
    hand_indices: Tensor,
    hand_masks: Tensor,
    target_policies: Tensor,
    target_values: Tensor,
    batch_size: int = 128,
) -> dict:
    """Train one epoch on MCTS-generated data using ZebModel.

    Args:
        model: ZebModel transformer
        optimizer: Optimizer
        tokens: [N, MAX_TOKENS, N_FEATURES] observation tokens
        masks: [N, MAX_TOKENS] valid token masks
        hand_indices: [N, 7] indices of hand slots in token sequence
        hand_masks: [N, 7] which hand slots have unplayed dominoes
        target_policies: [N, 7] MCTS visit distributions over hand slots
        target_values: [N] game outcomes
        batch_size: Batch size

    Returns:
        Dict with policy_loss and value_loss
    """
    model.train()

    n_samples = tokens.shape[0]
    indices = torch.randperm(n_samples, device=tokens.device)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    n_batches = 0

    for start in range(0, n_samples - batch_size + 1, batch_size):
        idx = indices[start:start + batch_size]

        batch_tokens = tokens[idx]
        batch_masks = masks[idx]
        batch_hand_indices = hand_indices[idx]
        batch_hand_masks = hand_masks[idx]
        batch_target_policy = target_policies[idx]
        batch_target_value = target_values[idx]

        # Forward: ZebModel returns (policy_logits, value)
        policy_logits, value = model(
            batch_tokens, batch_masks, batch_hand_indices, batch_hand_masks
        )

        # Mask illegal actions and compute log softmax
        policy_logits = policy_logits.masked_fill(~batch_hand_masks, float('-inf'))
        log_policy = F.log_softmax(policy_logits, dim=-1)

        # Policy loss: cross-entropy with soft targets (KL divergence)
        # Clamp to avoid -inf * 0 = NaN
        log_policy_safe = log_policy.clamp(min=-100)
        policy_loss = -(batch_target_policy * log_policy_safe).sum(dim=-1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(value, batch_target_value)

        # Combined loss
        loss = policy_loss + value_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        n_batches += 1

    return {
        'policy_loss': total_policy_loss / n_batches if n_batches > 0 else 0.0,
        'value_loss': total_value_loss / n_batches if n_batches > 0 else 0.0,
    }


def evaluate_vs_random(model: ZebModel, n_games: int = 100, device: str = 'cpu') -> float:
    """Evaluate model by playing against random opponent.

    Model plays team 0, random plays team 1.
    Returns win rate when using greedy policy.
    """
    from forge.oracle.rng import deal_from_seed
    from forge.oracle.tables import DOMINO_COUNT_POINTS
    from .game import new_game, apply_action, is_terminal, current_player, legal_actions
    import random as rand_module

    model.eval()
    model.to(device)

    wins = 0
    for game_idx in range(n_games):
        seed = 10000 + game_idx
        state = new_game(seed, dealer=0, skip_bidding=True)

        # Play game: model on team 0, random on team 1
        while not is_terminal(state):
            player = current_player(state)
            legal_slots = legal_actions(state)  # Returns slot indices

            if player % 2 == 0:
                # Model's turn - use greedy policy
                with torch.no_grad():
                    tokens, mask, hand_indices = observe(state, player)
                    hand_mask = torch.zeros(N_HAND_SLOTS, dtype=torch.bool)
                    for slot in legal_slots:
                        hand_mask[slot] = True

                    # Add batch dimension and move to device
                    tokens = tokens.unsqueeze(0).to(device)
                    mask = mask.unsqueeze(0).to(device)
                    hand_indices = hand_indices.unsqueeze(0).to(device)
                    hand_mask = hand_mask.unsqueeze(0).to(device)

                    policy_logits, _ = model(tokens, mask, hand_indices, hand_mask)
                    policy_logits = policy_logits.masked_fill(~hand_mask, float('-inf'))
                    action_slot = policy_logits[0].argmax().item()
            else:
                # Random opponent
                action_slot = rand_module.choice(legal_slots)

            state = apply_action(state, action_slot)

        # Count points from play history
        team0_pts = sum(
            DOMINO_COUNT_POINTS[d] for p, d in state.play_history if p % 2 == 0
        )
        team1_pts = sum(
            DOMINO_COUNT_POINTS[d] for p, d in state.play_history if p % 2 == 1
        )

        if team0_pts > team1_pts:
            wins += 1

    return wins / n_games


def main():
    parser = argparse.ArgumentParser(description='MCTS-based Zeb training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--games-per-epoch', type=int, default=100)
    parser.add_argument('--n-simulations', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='ZebModel size configuration')
    parser.add_argument('--use-oracle', action='store_true',
                        help='Use Stage1Oracle for MCTS leaf evaluation (requires CUDA)')
    parser.add_argument('--oracle-device', type=str, default='cuda',
                        help='Device for oracle inference')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='zeb-mcts',
                        help='W&B project name')
    parser.add_argument('--run-name', type=str, default=None,
                        help='W&B run name (auto-generated if not provided)')
    # Cross-game leaf batching parameters (for future batched oracle implementation)
    parser.add_argument('--n-parallel-games', type=int, default=16,
                        help='Number of concurrent MCTS games for cross-game batching')
    parser.add_argument('--cross-game-batch-size', type=int, default=512,
                        help='Target batch size for oracle evaluation across games')
    args = parser.parse_args()

    # Initialize W&B
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        run_name = args.run_name or f"mcts-zeb-{'oracle' if args.use_oracle else 'rollout'}-{args.model_size}-{timestamp}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            tags=['mcts', 'zeb', args.model_size, 'oracle' if args.use_oracle else 'rollout'],
            config={
                'epochs': args.epochs,
                'games_per_epoch': args.games_per_epoch,
                'n_simulations': args.n_simulations,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'temperature': args.temperature,
                'model_size': args.model_size,
                'use_oracle': args.use_oracle,
                'device': args.device,
                'n_parallel_games': args.n_parallel_games,
                'cross_game_batch_size': args.cross_game_batch_size,
            },
        )
        print(f"W&B run: {wandb.run.url}")

    print("=== MCTS AlphaZero-style Training (ZebModel) ===")
    print(f"Epochs: {args.epochs}")
    print(f"Games/epoch: {args.games_per_epoch}")
    print(f"MCTS simulations: {args.n_simulations}")
    print(f"Learning rate: {args.lr}")
    print(f"Model size: {args.model_size}")
    print(f"Use oracle: {args.use_oracle}")
    print()

    # Load oracle value function if requested
    value_fn = None
    if args.use_oracle:
        print("Loading Stage1Oracle for leaf evaluation...")
        from .oracle_value import create_oracle_value_fn
        t0 = time.time()
        value_fn = create_oracle_value_fn(device=args.oracle_device, compile=True)
        print(f"Oracle loaded in {time.time() - t0:.1f}s")
        print()

    # Model: ZebModel transformer
    model_config = get_model_config(args.model_size)
    model = ZebModel(**model_config)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ZebModel ({args.model_size}): {total_params:,} parameters")
    print(f"  embed_dim={model_config['embed_dim']}, n_layers={model_config['n_layers']}, n_heads={model_config['n_heads']}")

    # Initial evaluation
    print("\nInitial evaluation...")
    win_rate = evaluate_vs_random(model, n_games=100, device=args.device)
    print(f"  vs Random: {win_rate:.1%}")
    if use_wandb:
        wandb.log({'eval/vs_random_win_rate': win_rate, 'epoch': -1})

    # Training loop
    for epoch in range(args.epochs):
        t0 = time.time()

        # Generate games with MCTS (cross-game batched for GPU efficiency)
        games = play_games_batched(
            n_games=args.games_per_epoch,
            n_simulations=args.n_simulations,
            temperature=args.temperature,
            base_seed=epoch * args.games_per_epoch,
            value_fn=value_fn,
            n_parallel_games=args.n_parallel_games,
            target_batch_size=args.cross_game_batch_size,
        )
        gen_time = time.time() - t0

        # Convert to Zeb tensors (proper observation encoding)
        tokens, masks, hand_indices, hand_masks, target_policies, target_values = \
            mcts_examples_to_zeb_tensors(games)

        # Move to device
        tokens = tokens.to(args.device)
        masks = masks.to(args.device)
        hand_indices = hand_indices.to(args.device)
        hand_masks = hand_masks.to(args.device)
        target_policies = target_policies.to(args.device)
        target_values = target_values.to(args.device)

        # Train
        t1 = time.time()
        metrics = train_epoch(
            model, optimizer,
            tokens, masks, hand_indices, hand_masks,
            target_policies, target_values,
            batch_size=args.batch_size,
        )
        train_time = time.time() - t1

        # Compute stats
        n_examples = tokens.shape[0]
        games_per_sec = args.games_per_epoch / gen_time
        oracle_queries = 0
        if value_fn is not None and hasattr(value_fn, 'query_count'):
            oracle_queries = value_fn.query_count
            value_fn.query_count = 0  # Reset for next epoch

        # Build epoch summary
        epoch_str = f"Epoch {epoch:3d}: policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}"
        epoch_str += f" (gen={gen_time:.1f}s, train={train_time:.2f}s, {games_per_sec:.2f} games/s)"
        if oracle_queries > 0:
            epoch_str += f" [oracle: {oracle_queries:,}]"
        print(epoch_str)

        # Log to W&B
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/policy_loss': metrics['policy_loss'],
                'train/value_loss': metrics['value_loss'],
                'train/total_loss': metrics['policy_loss'] + metrics['value_loss'],
                'perf/gen_time_s': gen_time,
                'perf/train_time_s': train_time,
                'perf/games_per_sec': games_per_sec,
                'perf/examples': n_examples,
                'perf/oracle_queries': oracle_queries,
            })

        # Periodic evaluation
        if (epoch + 1) % args.eval_every == 0:
            win_rate = evaluate_vs_random(model, n_games=100, device=args.device)
            print(f"         → vs Random: {win_rate:.1%}")
            if use_wandb:
                wandb.log({'eval/vs_random_win_rate': win_rate, 'epoch': epoch})

    # Final evaluation
    print("\n=== Final Evaluation ===")
    win_rate = evaluate_vs_random(model, n_games=200, device=args.device)
    print(f"vs Random: {win_rate:.1%}")

    if use_wandb:
        wandb.log({'final/vs_random_win_rate': win_rate})
        wandb.finish()
        print("W&B run finished.")


if __name__ == '__main__':
    main()
