"""MCTS-based AlphaZero-style training for Zeb.

Uses MCTS to generate training targets, trains network to predict:
1. Policy: MCTS visit distribution (over 7 hand slots)
2. Value: Game outcome

This provides much more stable learning than pure REINFORCE.
"""
import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'


def get_rng_state() -> dict:
    """Capture all RNG states for exact reproducibility."""
    state = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    # Some environments have a CUDA-enabled PyTorch build but no usable CUDA runtime
    # (e.g., no driver access). In that case, attempting to touch torch.cuda will raise.
    try:
        state['cuda'] = torch.cuda.get_rng_state_all()
    except Exception:
        state['cuda'] = None
    return state


def set_rng_state(state: dict) -> None:
    """Restore all RNG states."""
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    cuda_state = state.get('cuda')
    if cuda_state is not None:
        try:
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception:
            pass


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    wandb_run_id: Optional[str] = None,
) -> None:
    """Save training checkpoint with full state for resumability."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': get_rng_state(),
        'config': config,
        'wandb_run_id': wandb_run_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Load checkpoint and restore model/optimizer/RNG states.

    Returns:
        Checkpoint dict with epoch, config, wandb_run_id
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    set_rng_state(checkpoint['rng_state'])
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: Path, model_size: str) -> Optional[Path]:
    """Find the most recent checkpoint for the given model size.

    Looks for files matching pattern: zeb-{model_size}-epoch*.pt
    Returns the one with the highest epoch number.
    """
    pattern = f'zeb-{model_size}-epoch*.pt'
    checkpoints = sorted(checkpoint_dir.glob(pattern))
    return checkpoints[-1] if checkpoints else None


def cleanup_old_checkpoints(checkpoint_dir: Path, model_size: str, keep: int) -> None:
    """Keep only the most recent N checkpoints for this model size."""
    pattern = f'zeb-{model_size}-epoch*.pt'
    checkpoints = sorted(checkpoint_dir.glob(pattern))
    for old_ckpt in checkpoints[:-keep]:
        old_ckpt.unlink()


def checkpoint_path(checkpoint_dir: Path, model_size: str, epoch: int) -> Path:
    """Generate checkpoint filename following naming convention."""
    return checkpoint_dir / f'zeb-{model_size}-epoch{epoch:04d}.pt'

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

        # Use correct trick-based scoring from game state
        team0_pts, team1_pts = state.team_points

        if team0_pts > team1_pts:
            wins += 1

    return wins / n_games


def main():
    # Training entrypoint: self-play (GPU MCTS).
    #
    # Keep the rest of this module intact because other scripts import checkpoint helpers
    # and evaluation utilities from here.
    from .run_selfplay_training import main as _selfplay_main

    _selfplay_main()


if __name__ == '__main__':
    main()
