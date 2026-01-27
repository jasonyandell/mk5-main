"""Hypothetical deal building for GPU E[Q] pipeline."""

from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor


def build_hypothetical_deals(
    states: GameStateTensor,
    worlds: Tensor,
) -> Tensor:
    """Reconstruct hypothetical full deals from current hands + sampled opponents.

    Vectorized implementation - no Python loops or .item() calls.

    Args:
        states: GameStateTensor with n_games
        worlds: [n_games, n_samples, 3, max_hand_size] opponent hands (padded with -1)

    Returns:
        [n_games, n_samples, 4, 7] full deals

    Note: Reconstructs initial hands by combining:
        - Current player's actual hand (same across all samples)
        - Sampled opponent hands (different per sample)
        - Both may have variable sizes (padded with -1)
    """
    n_games = states.n_games
    n_samples = worlds.shape[1]
    max_hand_size = worlds.shape[3]
    device = states.device
    current_players = states.current_player.long()  # [N]

    # Initialize output
    deals = torch.full((n_games, n_samples, 4, 7), -1, dtype=torch.int32, device=device)

    # === Step 1: Scatter current player's hand into correct position ===
    # my_hands[g] = states.hands[g, current_players[g], :]
    # Use gather to get each game's current player's hand
    player_idx_for_hands = current_players.view(n_games, 1, 1).expand(n_games, 1, 7)  # [N, 1, 7]
    my_hands = torch.gather(states.hands, dim=1, index=player_idx_for_hands).squeeze(1)  # [N, 7]

    # Convert to int32 to match output dtype (states.hands is int8)
    my_hands = my_hands.to(torch.int32)

    # Expand my_hands for all samples: [N, 7] -> [N, M, 7]
    my_hands_expanded = my_hands.unsqueeze(1).expand(n_games, n_samples, 7)  # [N, M, 7]

    # Scatter my_hands into deals at position current_players[g] for each game
    # deals[g, :, current_players[g], :] = my_hands_expanded[g, :, :]
    # Use scatter_ along dim=2 (player dimension)
    scatter_idx = current_players.view(n_games, 1, 1, 1).expand(n_games, n_samples, 1, 7)  # [N, M, 1, 7]
    deals.scatter_(dim=2, index=scatter_idx, src=my_hands_expanded.unsqueeze(2))  # scatter at player dim

    # === Step 2: Compute opponent player indices without loops ===
    # opp_players[g, i] = (current_players[g] + i + 1) % 4 for i in 0,1,2
    offsets = torch.arange(1, 4, device=device).unsqueeze(0)  # [1, 3]
    opp_players = (current_players.unsqueeze(1) + offsets) % 4  # [N, 3]

    # === Step 3: Scatter opponent hands into correct positions ===
    # For each opponent index (0, 1, 2), scatter worlds[:, :, opp_idx, :] to deals[:, :, opp_players[:, opp_idx], :]
    copy_size = min(max_hand_size, 7)

    # Process all 3 opponents at once using advanced indexing
    # worlds: [N, M, 3, max_hand_size]
    # We want: deals[g, s, opp_players[g, opp_idx], :copy_size] = worlds[g, s, opp_idx, :copy_size]

    # For each of the 3 opponents, scatter their hands
    for opp_idx in range(3):
        # opp_player_positions: [N] - which player slot for this opponent in each game
        opp_player_positions = opp_players[:, opp_idx]  # [N]

        # Expand for samples and hand slots: [N] -> [N, M, 1, copy_size]
        scatter_opp_idx = opp_player_positions.view(n_games, 1, 1, 1).expand(n_games, n_samples, 1, copy_size)

        # Get this opponent's hands: [N, M, copy_size]
        opp_hands = worlds[:, :, opp_idx, :copy_size]

        # Scatter into deals: [N, M, 1, copy_size] at scatter_opp_idx positions
        deals[:, :, :, :copy_size].scatter_(
            dim=2,
            index=scatter_opp_idx,
            src=opp_hands.unsqueeze(2)  # [N, M, 1, copy_size]
        )

    return deals
