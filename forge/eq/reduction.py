"""Reduction helpers for turning per-world Q-values into E[Q] and Var[Q]."""

from __future__ import annotations

import numpy as np
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

    # Move once to CPU to avoid per-element CUDA synchronization.
    q_cpu = all_q_values.detach().to(device="cpu", dtype=torch.float32)
    w_cpu = weights.detach().to(device="cpu", dtype=torch.float32)

    # Build gather indices: for each world, map each remaining-hand domino to its initial local_idx.
    # Vectorized using numpy: extract player's sorted initial hands (N, 7) and use broadcasting
    # with searchsorted to find the local index of each domino in my_hand within each world.
    initial_hands_array = np.array(
        [[initial_hands[player][i] for i in range(7)] for initial_hands in hypothetical_deals],
        dtype=np.int32,
    )  # (N, 7)
    my_hand_array = np.array(my_hand, dtype=np.int32)  # (len(my_hand),)

    # Use equality comparison to find indices: for each world, compare initial_hand with my_hand.
    # Since initial hands are sorted and contain all dominoes in my_hand, argmax on equality gives local_idx.
    gather_idx = torch.from_numpy(
        np.argmax(
            initial_hands_array[:, None, :] == my_hand_array[None, :, None],
            axis=2,
        ).astype(np.int64)
    )  # (N, len(my_hand))

    q_for_my = q_cpu.gather(dim=1, index=gather_idx)  # (N, len(my_hand))
    w = w_cpu.unsqueeze(1)  # (N, 1)

    mean = (w * q_for_my).sum(dim=0)
    mean2 = (w * (q_for_my * q_for_my)).sum(dim=0)
    var = (mean2 - mean * mean).clamp(min=0.0)
    return mean, var

