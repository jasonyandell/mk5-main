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
    """Evaluate model vs random players.

    Model plays seats 0, 2 (team 0), random plays seats 1, 3 (team 1).
    """
    neural = NeuralPlayer(model, device=device)
    random_player = RandomPlayer()

    players = (neural, random_player, neural, random_player)
    return play_match(players, n_games=n_games)


def evaluate_vs_heuristic(model, n_games: int = 100, device: str = 'cuda') -> dict:
    """Evaluate model vs heuristic players."""
    neural = NeuralPlayer(model, device=device)
    heuristic = RuleBasedPlayer()

    players = (neural, heuristic, neural, heuristic)
    return play_match(players, n_games=n_games)
