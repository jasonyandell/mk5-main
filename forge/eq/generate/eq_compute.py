"""E[Q] computation functions (mean, variance, PDF)."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_eq_with_counts(
    q_values: Tensor,  # [N, M, 7]
    world_counts: Tensor,  # [N]
) -> tuple[Tensor, Tensor]:
    """Compute E[Q] with variable world counts per game.

    Args:
        q_values: [N, M, 7] Q-values padded to max world count
        world_counts: [N] actual world count per game

    Returns:
        Tuple of (e_q, e_q_var):
            - e_q: [N, 7] mean Q across valid worlds
            - e_q_var: [N, 7] variance across valid worlds
    """
    n_games, max_worlds, n_actions = q_values.shape
    device = q_values.device

    # Create mask for valid worlds: [N, M]
    world_idx = torch.arange(max_worlds, device=device).unsqueeze(0)  # [1, M]
    valid_mask = world_idx < world_counts.unsqueeze(1)  # [N, M]

    # Expand mask for actions: [N, M, 1]
    valid_mask_expanded = valid_mask.unsqueeze(2)  # [N, M, 1]

    # Masked Q-values (set invalid to 0 for sum)
    q_masked = q_values * valid_mask_expanded.float()

    # Sum and count
    q_sum = q_masked.sum(dim=1)  # [N, 7]
    counts_expanded = world_counts.unsqueeze(1).float()  # [N, 1]
    counts_expanded = counts_expanded.clamp(min=1)  # Avoid div by zero

    # Mean
    e_q = q_sum / counts_expanded  # [N, 7]

    # Variance: Var = E[X^2] - E[X]^2
    q_sq_masked = (q_values ** 2) * valid_mask_expanded.float()
    q_sq_sum = q_sq_masked.sum(dim=1)  # [N, 7]
    e_q_sq = q_sq_sum / counts_expanded
    e_q_var = e_q_sq - e_q ** 2  # [N, 7]

    # Clamp variance to avoid numerical issues
    e_q_var = e_q_var.clamp(min=0)

    return e_q, e_q_var


def compute_eq_pdf(
    q_values: Tensor,  # [N, M, 7]
    weights: Tensor | None = None,  # [N, M] optional weights
    world_counts: Tensor | None = None,  # [N] for enumeration with variable counts
) -> Tensor:
    """Compute full PDF of Q-values per action.

    Builds a histogram of Q-values across samples/worlds for each action.
    Q-values in [-42, +42] map to bins [0, 84] (85 bins total).

    Args:
        q_values: [N, M, 7] Q-values per game/sample/action
        weights: [N, M] optional weights (uniform 1/M if None)
        world_counts: [N] actual world count per game (for enumeration)

    Returns:
        pdf: [N, 7, 85] P(Q=q|action) where bin i -> Q = i - 42
    """
    N, M, n_actions = q_values.shape
    device = q_values.device

    # Bin Q-values: q in [-42, 42] -> bin in [0, 84]
    bins = (q_values + 42).round().clamp(0, 84).long()  # [N, M, 7]

    # Determine weights
    if weights is None:
        if world_counts is not None:
            # Variable counts: weight = 1/count for valid, 0 for invalid
            world_idx = torch.arange(M, device=device).unsqueeze(0)  # [1, M]
            valid_mask = world_idx < world_counts.unsqueeze(1)  # [N, M]
            weights = valid_mask.float() / world_counts.unsqueeze(1).clamp(min=1).float()
        else:
            # Uniform weights
            weights = torch.full((N, M), 1.0 / M, device=device)

    # Vectorized scatter_add: single kernel instead of N separate calls
    # Flatten to 1D and compute flat indices: flat_idx = n * (85 * 7) + bin * 7 + action
    #
    # Layout: pdf_flat[n * 85 * 7 + bin * 7 + a] for game n, bin b, action a
    pdf_flat = torch.zeros(N * 85 * n_actions, device=device, dtype=torch.float32)

    # Compute flat indices for all (n, m, a) combinations
    # game_offset[n, m, a] = n * 85 * 7
    game_idx = torch.arange(N, device=device).view(N, 1, 1)  # [N, 1, 1]
    game_offset = game_idx * (85 * n_actions)  # [N, 1, 1]

    # bin_offset[n, m, a] = bins[n, m, a] * 7
    bin_offset = bins * n_actions  # [N, M, 7]

    # action_offset[n, m, a] = a
    action_idx = torch.arange(n_actions, device=device).view(1, 1, n_actions)  # [1, 1, 7]

    # Combined flat index
    flat_indices = (game_offset + bin_offset + action_idx).view(-1)  # [N * M * 7]

    # Expand weights: [N, M] -> [N, M, 7] -> [N * M * 7]
    weights_flat = weights.unsqueeze(-1).expand(N, M, n_actions).reshape(-1)

    # Single scatter_add call
    pdf_flat.scatter_add_(0, flat_indices, weights_flat)

    # Reshape to [N, 85, 7] then transpose to [N, 7, 85]
    pdf = pdf_flat.view(N, 85, n_actions).permute(0, 2, 1).contiguous()

    return pdf
