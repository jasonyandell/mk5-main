"""Evaluation baselines for Zeb self-play."""
import torch
from torch import Tensor
from typing import List, Tuple, Protocol
from abc import ABC, abstractmethod
import random

from .types import ZebGameState
from .game import new_game, apply_action, legal_actions, is_terminal, get_outcome
from .observation import observe, get_legal_mask


class Player(Protocol):
    """Protocol for game players."""
    def select_action(self, state: ZebGameState, player: int) -> int:
        """Select action (slot index) for given player."""
        ...


class RandomPlayer:
    """Uniform random from legal moves."""

    def select_action(self, state: ZebGameState, player: int) -> int:
        legal = legal_actions(state)
        return random.choice(legal)


class RuleBasedPlayer:
    """Simple heuristic player.

    Rules:
    1. If leading: play highest trump if available, else highest non-trump
    2. If following and can win: play lowest winning domino
    3. If following and can't win: play lowest domino
    4. If sloughing: play lowest point domino
    """

    def select_action(self, state: ZebGameState, player: int) -> int:
        from forge.oracle.tables import (
            DOMINO_HIGH, DOMINO_LOW, DOMINO_COUNT_POINTS, DOMINO_IS_DOUBLE,
            is_in_called_suit, trick_rank, led_suit_for_lead_domino,
        )

        legal = legal_actions(state)
        hand = state.hands[player]

        # Get legal domino IDs
        legal_dominoes = [hand[slot] for slot in legal]

        # Simple heuristic: play highest if leading, lowest otherwise
        is_leading = len(state.current_trick) == 0

        if is_leading:
            # Lead highest domino (prioritize trumps)
            decl = state.decl_id
            best_slot = max(legal, key=lambda s: (
                is_in_called_suit(hand[s], decl),  # Trumps first
                DOMINO_HIGH[hand[s]] + DOMINO_LOW[hand[s]],  # Then by pip sum
            ))
        else:
            # Follow with lowest legal
            best_slot = min(legal, key=lambda s: (
                DOMINO_COUNT_POINTS[hand[s]],  # Minimize points given up
                DOMINO_HIGH[hand[s]] + DOMINO_LOW[hand[s]],
            ))

        return best_slot


class NeuralPlayer:
    """Neural network player."""

    def __init__(self, model, device: str = 'cuda', temperature: float = 0.1):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.model.eval()
        self.model.to(device)

    def select_action(self, state: ZebGameState, player: int) -> int:
        with torch.no_grad():
            tokens, mask, hand_indices = observe(state, player)
            legal = get_legal_mask(state, player)

            tokens = tokens.unsqueeze(0).to(self.device)
            mask = mask.unsqueeze(0).to(self.device)
            hand_indices = hand_indices.unsqueeze(0).to(self.device)
            legal = legal.unsqueeze(0).to(self.device)

            action, _, _ = self.model.get_action(
                tokens, mask, hand_indices, legal,
                temperature=self.temperature,
            )
            return action.item()


def play_match(
    players: Tuple[Player, Player, Player, Player],
    n_games: int = 100,
    base_seed: int = 0,
) -> dict:
    """Play matches between players.

    Args:
        players: Tuple of 4 players (seats 0-3)
        n_games: Number of games to play
        base_seed: Starting seed

    Returns:
        Stats dict with win rates, margins, etc.
    """
    team0_wins = 0
    team1_wins = 0
    total_margin = 0

    for i in range(n_games):
        state = new_game(seed=base_seed + i, dealer=i % 4, skip_bidding=True)

        while not is_terminal(state):
            player = _get_current_player(state)
            action = players[player].select_action(state, player)
            state = apply_action(state, action)

        # Score
        if state.team_points[0] > state.team_points[1]:
            team0_wins += 1
        else:
            team1_wins += 1
        total_margin += state.team_points[0] - state.team_points[1]

    return {
        'team0_wins': team0_wins,
        'team1_wins': team1_wins,
        'team0_win_rate': team0_wins / n_games,
        'avg_margin': total_margin / n_games,
        'n_games': n_games,
    }


def _get_current_player(state: ZebGameState) -> int:
    """Get current player to act."""
    trick_len = len(state.current_trick)
    return (state.trick_leader + trick_len) % 4


def evaluate_vs_random(model, n_games: int = 100, device: str = 'cuda') -> dict:
    """Evaluate model vs random players (batched for GPU efficiency).

    Model plays seats 0, 2 (team 0), random plays seats 1, 3 (team 1).
    Runs all games in parallel with batched model inference.
    """
    return evaluate_vs_random_batched(model, n_games=n_games, device=device)


def evaluate_vs_random_batched(
    model,
    n_games: int = 100,
    device: str = 'cuda',
    temperature: float = 0.1,
    neural_team: int = 0,
) -> dict:
    """Batched evaluation - runs N games in parallel with batched inference.

    Args:
        neural_team: Which team the model plays (0 or 1). Default 0 = seats 0,2.
    """
    model.eval()
    model.to(device)

    # Initialize all games
    states = [new_game(seed=i, dealer=i % 4, skip_bidding=True) for i in range(n_games)]
    active = [True] * n_games  # Track which games are still running

    # Play until all games complete
    while any(active):
        # Collect states needing neural network (seats 0, 2)
        neural_indices = []
        neural_states = []
        neural_players = []

        # Collect states needing random action (seats 1, 3)
        random_indices = []

        for i, (state, is_active) in enumerate(zip(states, active)):
            if not is_active:
                continue

            player = _get_current_player(state)
            if player % 2 == neural_team:  # Neural player's turn
                neural_indices.append(i)
                neural_states.append(state)
                neural_players.append(player)
            else:  # Random player's turn
                random_indices.append(i)

        # Batch neural network inference
        if neural_indices:
            with torch.no_grad():
                # Build batched tensors
                batch_tokens = []
                batch_masks = []
                batch_hand_indices = []
                batch_legal = []

                for state, player in zip(neural_states, neural_players):
                    tokens, mask, hand_indices = observe(state, player)
                    legal = get_legal_mask(state, player)
                    batch_tokens.append(tokens)
                    batch_masks.append(mask)
                    batch_hand_indices.append(hand_indices)
                    batch_legal.append(legal)

                # Stack and move to device
                batch_tokens = torch.stack(batch_tokens).to(device)
                batch_masks = torch.stack(batch_masks).to(device)
                batch_hand_indices = torch.stack(batch_hand_indices).to(device)
                batch_legal = torch.stack(batch_legal).to(device)

                # Single forward pass for all neural states
                actions, _, _ = model.get_action(
                    batch_tokens, batch_masks, batch_hand_indices, batch_legal,
                    temperature=temperature,
                )

                # Apply actions
                for idx, game_idx in enumerate(neural_indices):
                    action = actions[idx].item()
                    states[game_idx] = apply_action(states[game_idx], action)
                    if is_terminal(states[game_idx]):
                        active[game_idx] = False

        # Handle random player turns
        for game_idx in random_indices:
            state = states[game_idx]
            legal = legal_actions(state)
            action = random.choice(legal)
            states[game_idx] = apply_action(state, action)
            if is_terminal(states[game_idx]):
                active[game_idx] = False

    # Count results
    team0_wins = 0
    team1_wins = 0
    total_margin = 0

    for state in states:
        if state.team_points[0] > state.team_points[1]:
            team0_wins += 1
        else:
            team1_wins += 1
        total_margin += state.team_points[0] - state.team_points[1]

    return {
        'team0_wins': team0_wins,
        'team1_wins': team1_wins,
        'team0_win_rate': team0_wins / n_games,
        'avg_margin': total_margin / n_games,
        'n_games': n_games,
    }


def evaluate_vs_heuristic(model, n_games: int = 100, device: str = 'cuda') -> dict:
    """Evaluate model vs heuristic players."""
    neural = NeuralPlayer(model, device=device)
    heuristic = RuleBasedPlayer()

    players = (neural, heuristic, neural, heuristic)
    return play_match(players, n_games=n_games)
