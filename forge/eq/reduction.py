"""Reduction helpers for turning per-world Q-values into E[Q] and Var[Q]."""

from __future__ import annotations

import torch
from torch import Tensor


def _reduce_world_q_values(
    *,
    all_q_values: Tensor,
    weights: Tensor,
    hypothetical_deals: list[list[list[int]]],
    player: int,
    my_hand: list[int],
) -> tuple[Tensor, Tensor]:
    """Reduce per-world Q-values into E[Q] mean/variance for my_hand (CPU tensors).

    This avoids per-scalar `.item()` calls on CUDA tensors, which force stream
    synchronization and dominate runtime at small batch sizes.

    Args:
        all_q_values: (N, 7) tensor of Q-values indexed by initial-hand local_idx
        weights: (N,) tensor of normalized world weights (uniform or posterior)
        hypothetical_deals: List of N initial deals (4 hands × 7 dominoes each)
        player: Current player ID (0-3)
        my_hand: Remaining-hand domino IDs (global), in desired output order

    Returns:
        Tuple of (e_q_mean, e_q_var), both (len(my_hand),) CPU float32 tensors.
    """
    n_worlds = len(hypothetical_deals)
    if n_worlds == 0:
        raise ValueError("hypothetical_deals cannot be empty")
    if all_q_values.shape != (n_worlds, 7):
        raise ValueError(
            f"Expected all_q_values shape ({n_worlds}, 7), got {tuple(all_q_values.shape)}"
        )
    if weights.shape != (n_worlds,):
        raise ValueError(
            f"Expected weights shape ({n_worlds},), got {tuple(weights.shape)}"
        )
    if not my_hand:
        return torch.empty((0,), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)

    # KEY INSIGHT: The acting player's initial hand is world-invariant across all N sampled worlds.
    # All worlds share the same initial hand for the actor, so gather indices are identical.
    # Compute them ONCE from hypothetical_deals[0][player] instead of building (N, K) index tensor.
    initial_hand = hypothetical_deals[0][player]  # World-invariant for actor!

    # Build gather indices: map each domino in my_hand to its position in initial_hand
    gather_idx = torch.tensor(
        [initial_hand.index(d) for d in my_hand],
        device=all_q_values.device,
        dtype=torch.long,
    )  # (K,) where K = len(my_hand)

    # Gather Q-values for remaining hand positions, staying on GPU
    q_remaining = all_q_values[:, gather_idx]  # (N, K) - GPU tensor

    # Weighted reduction on GPU
    w = weights.unsqueeze(1)  # (N, 1)
    mean = (w * q_remaining).sum(dim=0)  # (K,) - GPU
    mean2 = (w * q_remaining ** 2).sum(dim=0)  # (K,) - GPU
    var = (mean2 - mean * mean).clamp(min=0.0)  # (K,) - GPU

    # Only transfer final results (K floats, typically ≤7) to CPU
    return mean.cpu(), var.cpu()

