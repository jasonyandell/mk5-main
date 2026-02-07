"""Self-play trajectory generation for Zeb.

This module implements batched self-play game generation using a neural network
policy. Games are played to completion and trajectories are collected for training.

Key functions:
- play_games_batched: Play N games with neural network policy
- trajectories_to_batch: Convert trajectories to training tensors
"""
import torch
from torch import Tensor
from typing import List, Tuple
from dataclasses import dataclass, field

from .types import ZebGameState, TrajectoryStep, TrajectoryGame
from .game import new_game, apply_action, is_terminal, get_outcome, current_player, legal_actions
from .observation import observe, get_legal_mask
from .model import ZebModel
from .evaluate import RuleBasedPlayer


@dataclass
class GameInstance:
    """Tracks a single game during batched self-play."""
    state: ZebGameState
    steps: List[TrajectoryStep] = field(default_factory=list)
    game_idx: int = 0


def play_games_batched(
    model: ZebModel,
    n_games: int,
    temperature: float = 1.0,
    device: str = 'cuda',
    base_seed: int = 0,
) -> List[TrajectoryGame]:
    """Play N games with neural network policy, collect trajectories.

    Uses batched inference for efficiency - all active games are processed
    together in each forward pass.

    Args:
        model: The policy/value network
        n_games: Number of games to play
        temperature: Sampling temperature (higher = more exploration)
        device: Device for model inference
        base_seed: Starting seed for game generation

    Returns:
        List of completed game trajectories with filled outcomes
    """
    model.eval()
    model.to(device)

    # Initialize all games
    games = [
        GameInstance(
            state=new_game(seed=base_seed + i, dealer=i % 4, skip_bidding=True),
            game_idx=i,
        )
        for i in range(n_games)
    ]

    completed: List[TrajectoryGame] = []

    with torch.no_grad():
        while games:
            # Collect observations for all active games
            batch_tokens: List[Tensor] = []
            batch_masks: List[Tensor] = []
            batch_hand_indices: List[Tensor] = []
            batch_hand_masks: List[Tensor] = []
            batch_players: List[int] = []

            for game in games:
                state = game.state
                player = current_player(state)
                tokens, mask, hand_indices = observe(state, player)
                legal = get_legal_mask(state, player)

                batch_tokens.append(tokens)
                batch_masks.append(mask)
                batch_hand_indices.append(hand_indices)
                batch_hand_masks.append(legal)
                batch_players.append(player)

            # Stack into batch tensors
            tokens_t = torch.stack(batch_tokens).to(device)
            masks_t = torch.stack(batch_masks).to(device)
            hand_idx_t = torch.stack(batch_hand_indices).to(device)
            hand_mask_t = torch.stack(batch_hand_masks).to(device)

            # Get actions from model
            actions, log_probs, values = model.get_action(
                tokens_t, masks_t, hand_idx_t, hand_mask_t,
                temperature=temperature,
            )

            # Apply actions and record steps
            next_games: List[GameInstance] = []
            for i, game in enumerate(games):
                action = actions[i].item()
                player = batch_players[i]

                # Record step with all observation data for training
                # Outcome will be filled after game completion
                step = TrajectoryStep(
                    tokens=batch_tokens[i],
                    mask=batch_masks[i],
                    hand_indices=batch_hand_indices[i],
                    legal_mask=batch_hand_masks[i],
                    action=action,
                    seat=player,
                    outcome=0.0,  # Placeholder, filled at game end
                )
                game.steps.append(step)

                # Apply action to game state
                game.state = apply_action(game.state, action)

                # Check if game is terminal
                if is_terminal(game.state):
                    # Fill outcomes for all steps based on final result
                    for step in game.steps:
                        step.outcome = get_outcome(game.state, step.seat)

                    final_scores = game.state.team_points
                    winner = 0 if final_scores[0] >= final_scores[1] else 1

                    completed.append(TrajectoryGame(
                        steps=game.steps,
                        final_scores=final_scores,
                        winner=winner,
                    ))
                else:
                    next_games.append(game)

            games = next_games

    return completed


def play_games_vs_heuristic(
    model: ZebModel,
    n_games: int,
    temperature: float = 1.0,
    device: str = 'cuda',
    base_seed: int = 0,
) -> List[TrajectoryGame]:
    """Play N games: model (seats 0,2) vs heuristic (seats 1,3).

    Only collects trajectories for the model's actions, providing a stable
    training target (fixed heuristic opponent doesn't change).

    Args:
        model: The policy/value network (plays seats 0, 2)
        n_games: Number of games to play
        temperature: Sampling temperature for model
        device: Device for model inference
        base_seed: Starting seed for game generation

    Returns:
        List of completed game trajectories (model steps only)
    """
    model.eval()
    model.to(device)
    heuristic = RuleBasedPlayer()

    # Initialize all games
    games = [
        GameInstance(
            state=new_game(seed=base_seed + i, dealer=i % 4, skip_bidding=True),
            game_idx=i,
        )
        for i in range(n_games)
    ]

    completed: List[TrajectoryGame] = []

    with torch.no_grad():
        while games:
            # Separate games by whether current player is model or heuristic
            model_games: List[Tuple[int, GameInstance]] = []
            heuristic_games: List[Tuple[int, GameInstance]] = []

            for idx, game in enumerate(games):
                player = current_player(game.state)
                if player in (0, 2):  # Model's turn
                    model_games.append((idx, game))
                else:  # Heuristic's turn
                    heuristic_games.append((idx, game))

            # Process heuristic moves (no batching needed, no trajectory collection)
            for idx, game in heuristic_games:
                player = current_player(game.state)
                action = heuristic.select_action(game.state, player)
                game.state = apply_action(game.state, action)

            # Process model moves in batch
            if model_games:
                batch_tokens: List[Tensor] = []
                batch_masks: List[Tensor] = []
                batch_hand_indices: List[Tensor] = []
                batch_hand_masks: List[Tensor] = []
                batch_players: List[int] = []
                batch_game_refs: List[GameInstance] = []

                for idx, game in model_games:
                    player = current_player(game.state)
                    tokens, mask, hand_indices = observe(game.state, player)
                    legal = get_legal_mask(game.state, player)

                    batch_tokens.append(tokens)
                    batch_masks.append(mask)
                    batch_hand_indices.append(hand_indices)
                    batch_hand_masks.append(legal)
                    batch_players.append(player)
                    batch_game_refs.append(game)

                # Stack into batch tensors
                tokens_t = torch.stack(batch_tokens).to(device)
                masks_t = torch.stack(batch_masks).to(device)
                hand_idx_t = torch.stack(batch_hand_indices).to(device)
                hand_mask_t = torch.stack(batch_hand_masks).to(device)

                # Get actions from model
                actions, log_probs, values = model.get_action(
                    tokens_t, masks_t, hand_idx_t, hand_mask_t,
                    temperature=temperature,
                )

                # Apply actions and record steps (model only)
                for i, game in enumerate(batch_game_refs):
                    action = actions[i].item()
                    player = batch_players[i]

                    # Record step for training
                    step = TrajectoryStep(
                        tokens=batch_tokens[i],
                        mask=batch_masks[i],
                        hand_indices=batch_hand_indices[i],
                        legal_mask=batch_hand_masks[i],
                        action=action,
                        seat=player,
                        outcome=0.0,  # Filled at game end
                    )
                    game.steps.append(step)

                    # Apply action
                    game.state = apply_action(game.state, action)

            # Check for completed games
            next_games: List[GameInstance] = []
            for game in games:
                if is_terminal(game.state):
                    # Fill outcomes for model steps only
                    for step in game.steps:
                        step.outcome = get_outcome(game.state, step.seat)

                    final_scores = game.state.team_points
                    winner = 0 if final_scores[0] >= final_scores[1] else 1

                    completed.append(TrajectoryGame(
                        steps=game.steps,
                        final_scores=final_scores,
                        winner=winner,
                    ))
                else:
                    next_games.append(game)

            games = next_games

    return completed


def trajectories_to_batch(
    trajectories: List[TrajectoryGame],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Convert trajectories to training batch tensors.

    Flattens all steps from all games into single batch tensors suitable
    for training with ZebLightningModule.

    Args:
        trajectories: List of completed game trajectories

    Returns:
        Tuple of tensors:
            tokens: [N, seq, 8] observation tokens
            masks: [N, seq] attention mask (1 = valid, 0 = padding)
            hand_indices: [N, 7] indices of player's hand tokens
            hand_masks: [N, 7] legal action mask (True = legal)
            actions: [N] action indices taken
            outcomes: [N] game outcomes for each step
    """
    all_tokens: List[Tensor] = []
    all_masks: List[Tensor] = []
    all_hand_indices: List[Tensor] = []
    all_hand_masks: List[Tensor] = []
    all_actions: List[int] = []
    all_outcomes: List[float] = []

    for game in trajectories:
        for step in game.steps:
            all_tokens.append(step.tokens)
            all_masks.append(step.mask)
            all_hand_indices.append(step.hand_indices)
            all_hand_masks.append(step.legal_mask)
            all_actions.append(step.action)
            all_outcomes.append(step.outcome)

    return (
        torch.stack(all_tokens),
        torch.stack(all_masks),
        torch.stack(all_hand_indices),
        torch.stack(all_hand_masks),
        torch.tensor(all_actions, dtype=torch.long),
        torch.tensor(all_outcomes, dtype=torch.float),
    )


def create_self_play_dataset(
    trajectories: List[TrajectoryGame],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Alias for trajectories_to_batch for clearer API.

    Same as trajectories_to_batch but with a more descriptive name
    for creating training datasets.
    """
    return trajectories_to_batch(trajectories)
