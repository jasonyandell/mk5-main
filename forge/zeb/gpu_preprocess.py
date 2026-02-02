"""GPU-accelerated preprocessing for oracle queries.

Moves bitmask computation and legal mask building to GPU to eliminate CPU bottleneck.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_remaining_bitmask_gpu(
    original_hands: Tensor,  # (N, 4, 7) int32 domino IDs
    current_hands: Tensor,   # (N, 4, 7) int32 domino IDs, -1 for played
) -> Tensor:
    """Compute remaining domino bitmasks on GPU.

    Args:
        original_hands: Original 7-card hands for all 4 players (N, 4, 7)
        current_hands: Current hands with -1 padding for played dominoes (N, 4, 7)

    Returns:
        remaining: (N, 4) int64 bitmasks indicating which dominoes remain
    """
    # For each position in original hand, check if that domino is in current hand
    # original_hands: (N, 4, 7)
    # current_hands: (N, 4, 7)

    # Expand for broadcasting: compare each original to all current
    # (N, 4, 7, 1) == (N, 4, 1, 7) -> (N, 4, 7, 7) -> any along last dim -> (N, 4, 7)
    in_hand = (
        original_hands.unsqueeze(-1) == current_hands.unsqueeze(-2)
    ).any(dim=-1)  # (N, 4, 7) bool

    # Convert to bitmask: sum(in_hand[j] << j for j in range(7))
    # bit_positions: [1, 2, 4, 8, 16, 32, 64]
    bit_positions = torch.tensor(
        [1 << j for j in range(7)],
        dtype=torch.int64,
        device=original_hands.device
    )

    # in_hand * bit_positions broadcasts (N, 4, 7) * (7,) -> (N, 4, 7)
    # Sum along last dim -> (N, 4)
    remaining = (in_hand.long() * bit_positions).sum(dim=-1)

    return remaining


def compute_legal_mask_gpu(
    original_hands: Tensor,  # (N, 4, 7) int32 domino IDs
    current_hands: Tensor,   # (N, 4, 7) int32 domino IDs, -1 for played
    actors: Tensor,          # (N,) int32 current player
    lead_suit: Tensor | None = None,  # (N,) int32 lead suit, -1 if no lead
) -> Tensor:
    """Compute legal action mask on GPU.

    Legal actions are dominoes that are:
    1. In the current player's current hand (not played yet)
    2. Follow the lead suit if one is set (suit-following rule)

    Args:
        original_hands: Original 7-card hands for all 4 players (N, 4, 7)
        current_hands: Current hands with -1 padding for played dominoes (N, 4, 7)
        actors: Current player for each sample (N,)
        lead_suit: Lead suit for each sample (N,), -1 if leading. None to skip suit check.

    Returns:
        legal_mask: (N, 7) bool mask of legal actions by slot index
    """
    n = original_hands.shape[0]
    device = original_hands.device

    # Get actor's hands: (N, 7)
    # Use gather: index into dim 1 with actors
    actor_original = original_hands[torch.arange(n, device=device), actors]  # (N, 7)
    actor_current = current_hands[torch.arange(n, device=device), actors]    # (N, 7)

    # A slot is legal if the original domino is in current hand
    # Compare original to each position in current
    # (N, 7, 1) == (N, 1, 7) -> (N, 7, 7) -> any -> (N, 7)
    in_hand = (
        actor_original.unsqueeze(-1) == actor_current.unsqueeze(-2)
    ).any(dim=-1)  # (N, 7) bool

    # If no lead suit specified, all in-hand dominoes are legal
    if lead_suit is None:
        return in_hand

    # TODO: Implement suit-following logic if needed
    # For now, return the in-hand mask
    # The suit-following would require knowing the lead domino's suits
    return in_hand
