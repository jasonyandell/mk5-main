"""Zeb self-play data structures."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import torch
from torch import Tensor


class GamePhase(IntEnum):
    BIDDING = 0
    TRUMP_SELECTION = 1
    PLAYING = 2
    TERMINAL = 3


@dataclass(frozen=True)
class BidState:
    """Track bidding progress."""

    bids: Tuple[int, ...]  # -1=not bid yet, 0=pass, 30-42=bid amount
    high_bidder: int  # -1 if no bids yet
    high_bid: int  # 0 if no bids yet


@dataclass(frozen=True)
class ZebGameState:
    """Immutable full game state."""

    hands: Tuple[Tuple[int, ...], ...]  # 4 hands of 7 dominoes (0-27)
    dealer: int
    phase: GamePhase
    bid_state: BidState
    decl_id: int  # trump declaration (-1 until selected)
    bidder: int  # who won the bid (-1 until bidding complete)
    played: frozenset  # set of played domino IDs
    play_history: Tuple[Tuple[int, int], ...]  # (player, domino_id) tuples
    current_trick: Tuple[int, ...]  # domino_ids in current trick (0-4 items)
    trick_leader: int  # who led current trick
    team_points: Tuple[int, int]  # (team0, team1) points so far


@dataclass
class TrajectoryStep:
    """Single decision point in a trajectory.

    Contains all information needed for training: observation tokens,
    masks, the action taken, and the eventual outcome.
    """

    tokens: Tensor  # [max_tokens, n_features] observation
    mask: Tensor  # [max_tokens] valid token mask
    hand_indices: Tensor  # [7] indices of player's hand tokens in sequence
    legal_mask: Tensor  # [max_hand] legal action mask
    action: int  # chosen action (slot index 0-6)
    seat: int  # player (0-3)
    outcome: float  # filled after game ends (-1 to 1)


@dataclass
class TrajectoryGame:
    """Complete game trajectory."""

    steps: list  # List[TrajectoryStep]
    final_scores: Tuple[int, int]  # (team0, team1) final points
    winner: int  # 0 or 1 (team that won)


@dataclass
class TrainingConfig:
    """Configuration for Zeb training."""

    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    games_per_epoch: int = 500
    epochs: int = 20
    batch_size: int = 128
    temperature: float = 1.0
    entropy_weight: float = 0.01
    value_weight: float = 0.5
