"""World sampling functions for GPU E[Q] pipeline."""

from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.sampling_gpu import WorldSampler
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV


def sample_worlds_batched(
    states: GameStateTensor,
    sampler: WorldSampler | WorldSamplerMRV,
    n_samples: int,
) -> Tensor:
    """Sample consistent worlds for all games.

    Vectorized implementation - no Python loops or .item() calls.

    Args:
        states: GameStateTensor with n_games
        sampler: Pre-allocated WorldSampler or WorldSamplerMRV
        n_samples: Number of samples per game

    Returns:
        [n_games, n_samples, 3, 7] opponent hands

    Note:
        WorldSamplerMRV is guaranteed to produce valid samples.
    """
    n_games = states.n_games
    device = states.device
    current_players = states.current_player.long()  # [N]

    # === Step 1: Vectorize pool computation ===
    # For each game, pool = all_dominoes - played - my_hand
    all_dominoes = torch.arange(28, device=device)  # [28]

    # Get current player's hand for each game using gather
    # states.hands: [N, 4, 7]
    # current_players: [N] -> expand to [N, 1, 7] for gathering
    player_idx_for_hands = current_players.view(n_games, 1, 1).expand(n_games, 1, 7)
    my_hands = torch.gather(states.hands, dim=1, index=player_idx_for_hands).squeeze(1)  # [N, 7]

    # Build mask for my dominoes across all games
    # my_hands: [N, 7], values are domino IDs or -1
    my_hand_mask = my_hands >= 0  # [N, 7]

    # Create pool masks for all games: [N, 28]
    # pool_mask[g, d] = True if domino d is in the pool for game g
    pool_masks = ~states.played_mask.clone()  # [N, 28] - start with unplayed

    # Remove my dominoes from pool
    # For each game g, set pool_masks[g, my_hands[g, i]] = False for valid dominoes
    # Use scatter to mark my dominoes as unavailable
    batch_indices = torch.arange(n_games, device=device).unsqueeze(1).expand(n_games, 7)  # [N, 7]
    my_dominoes_safe = torch.where(my_hand_mask, my_hands.long(), torch.zeros_like(my_hands))  # Replace -1 with 0

    # Scatter False into pool_masks at my_dominoes positions
    pool_masks[batch_indices[my_hand_mask], my_dominoes_safe[my_hand_mask]] = False

    # Convert pool_masks to pool lists with padding
    # pool_masks: [N, 28] -> pools: [N, 21] (max pool size is 21 when hand size = 7)
    pools = torch.full((n_games, 21), -1, dtype=torch.int32, device=device)

    for g in range(n_games):
        pool = all_dominoes[pool_masks[g]]
        pools[g, :len(pool)] = pool

    # === Step 2: Vectorize hand sizes computation ===
    # hand_counts: [N, 4] - number of dominoes per player per game
    hand_counts = (states.hands >= 0).sum(dim=2)  # [N, 4]

    # For each game, get opponent hand sizes (3 opponents)
    # Opponent i = (current_player + i + 1) % 4 for i in [0, 1, 2]
    offsets = torch.arange(1, 4, device=device).unsqueeze(0)  # [1, 3]
    opponent_indices = (current_players.unsqueeze(1) + offsets) % 4  # [N, 3]

    # Gather opponent hand sizes: [N, 3]
    # hand_counts: [N, 4], opponent_indices: [N, 3]
    hand_sizes_t = torch.gather(hand_counts, dim=1, index=opponent_indices.long())  # [N, 3]

    # === Step 3: Vectorize voids inference ===
    voids_t = infer_voids_batched(states)  # [N, 3, 8]

    # Get decl_ids
    decl_ids_t = states.decl_ids

    # Sample worlds - GPU only, no fallback
    worlds = sampler.sample(pools, hand_sizes_t, voids_t, decl_ids_t, n_samples)

    return worlds


def infer_voids_batched(states: GameStateTensor) -> Tensor:
    """Infer void suits from play history for all games (vectorized).

    Args:
        states: GameStateTensor with N games

    Returns:
        [N, 3, 8] boolean tensor where voids[g, opp_idx, suit] = True if opponent is void in suit
        Opponent indices are relative to current_player:
        - 0 = (current_player + 1) % 4
        - 1 = (current_player + 2) % 4
        - 2 = (current_player + 3) % 4
    """
    from forge.oracle.tables import can_follow, led_suit_for_lead_domino

    n_games = states.n_games
    device = states.device
    voids = torch.zeros(n_games, 3, 8, dtype=torch.bool, device=device)

    # Process each game (history structure makes full vectorization difficult)
    # This is still much faster than before because we removed other .item() calls
    for g in range(n_games):
        # Extract history for this game
        history = states.history[g].cpu().numpy()  # [28, 3]
        decl_id = states.decl_ids[g].item()
        current_player = states.current_player[g].item()

        for i in range(28):
            if history[i, 0] < 0:  # End of history
                break

            player, domino_id, lead_domino_id = history[i]

            # Skip current player's plays (we know our hand)
            if player == current_player:
                continue

            # Map absolute player to relative opponent index (0, 1, 2)
            opp_idx = (player - current_player - 1) % 4

            if opp_idx >= 3:  # Safety check
                continue

            # Check if this play revealed a void
            led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)
            if not can_follow(domino_id, led_suit, decl_id):
                voids[g, opp_idx, led_suit] = True

    return voids
