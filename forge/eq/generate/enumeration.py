"""World enumeration functions for exact E[Q] computation."""

from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.enumeration_gpu import WorldEnumeratorGPU
from forge.eq.game_tensor import GameStateTensor
from forge.eq.sampling_gpu import WorldSampler
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV

from .sampling import sample_worlds_batched, infer_voids_batched


def enumerate_or_sample_worlds(
    states: GameStateTensor,
    enumerator: WorldEnumeratorGPU,
    sampler: WorldSampler | WorldSamplerMRV,
    n_samples: int,
    threshold: int,
) -> tuple[Tensor, Tensor | None, int]:
    """Enumerate or sample worlds based on estimated world count.

    Uses enumeration when the estimated world count is below threshold,
    otherwise falls back to sampling.

    Args:
        states: GameStateTensor with N games
        enumerator: WorldEnumeratorGPU instance
        sampler: WorldSampler or WorldSamplerMRV for fallback
        n_samples: Number of samples if sampling
        threshold: Maximum worlds to enumerate

    Returns:
        Tuple of (worlds, world_counts, actual_n_samples):
            - worlds: [N, M, 3, 7] opponent hands where M = max(counts) or n_samples
            - world_counts: [N] actual counts per game, or None if sampling
            - actual_n_samples: Number of worlds/samples in dimension 1
    """
    n_games = states.n_games
    device = states.device

    # Compute history length to estimate world counts
    history_len = (states.history[:, :, 0] >= 0).sum(dim=1)  # [N]

    # Check if any game should enumerate
    # Use a vectorized check: late games (history_len >= 12) usually have < 100K worlds
    # (history_len 12 = end of trick 3, upper bound ~35k worlds)
    should_try_enum = (history_len >= 12).any().item()

    if not should_try_enum:
        # All games are early - use sampling
        worlds = sample_worlds_batched(states, sampler, n_samples)
        return worlds, None, n_samples

    # Try enumeration
    voids = infer_voids_batched(states)

    # For current hand enumeration, we don't use played_by for constraints
    # (played dominoes are already out of hands). Voids are the constraint.
    # Create empty played_by tensor (no known cards in current hands)
    played_by = torch.full((n_games, 3, 7), -1, dtype=torch.int8, device=device)

    # Get opponent hand sizes (current remaining cards)
    hand_counts = (states.hands >= 0).sum(dim=2)  # [N, 4]
    offsets = torch.arange(1, 4, device=device).unsqueeze(0)
    opp_indices = (states.current_player.unsqueeze(1) + offsets) % 4
    opp_hand_sizes = torch.gather(hand_counts, 1, opp_indices.long())  # [N, 3]

    # Build pools (unplayed dominoes - my hand = available for assignment)
    pools, pool_sizes = build_enum_pools(states, played_by)

    # Enumerate worlds
    worlds_enum, counts = enumerator.enumerate(
        pools=pools,
        hand_sizes=opp_hand_sizes,
        played_by=played_by,
        voids=voids,
        decl_ids=states.decl_ids,
    )

    # Check if enumeration succeeded for most games
    max_count = counts.max().item()

    if max_count > threshold:
        # Some games exceeded threshold - need to sample those
        # For simplicity, fall back to sampling for all games
        worlds = sample_worlds_batched(states, sampler, n_samples)
        return worlds, None, n_samples

    # Return enumerated worlds
    return worlds_enum, counts, max_count


def build_enum_pools(
    states: GameStateTensor,
    played_by: Tensor,  # [N, 3, 7]
) -> tuple[Tensor, Tensor]:
    """Build domino pools for enumeration (unknowns only).

    Args:
        states: GameStateTensor
        played_by: [N, 3, 7] dominoes played by each opponent

    Returns:
        Tuple of (pools, pool_sizes):
            - pools: [N, 21] domino IDs, -1 padded
            - pool_sizes: [N] actual pool size
    """
    n_games = states.n_games
    device = states.device
    all_dominoes = torch.arange(28, device=device)

    # Start with unplayed dominoes
    pool_masks = ~states.played_mask.clone()  # [N, 28]

    # Remove my hand
    current_players = states.current_player.long()
    player_idx = current_players.view(n_games, 1, 1).expand(n_games, 1, 7)
    my_hands = torch.gather(states.hands, dim=1, index=player_idx.long()).squeeze(1)
    my_hand_mask = my_hands >= 0

    batch_idx = torch.arange(n_games, device=device).unsqueeze(1).expand(n_games, 7)
    my_dom_safe = torch.where(my_hand_mask, my_hands.long(), torch.zeros_like(my_hands))
    pool_masks[batch_idx[my_hand_mask], my_dom_safe[my_hand_mask]] = False

    # Remove known opponent plays (they're assigned, not in unknown pool)
    for opp in range(3):
        for slot in range(7):
            dom_ids = played_by[:, opp, slot]
            valid = dom_ids >= 0
            if valid.any():
                batch_valid = torch.arange(n_games, device=device)[valid]
                pool_masks[batch_valid, dom_ids[valid].long()] = False

    # Convert to pool tensors
    pools = torch.full((n_games, 21), -1, dtype=torch.int8, device=device)
    pool_sizes = torch.zeros(n_games, dtype=torch.int32, device=device)

    for g in range(n_games):
        pool = all_dominoes[pool_masks[g]]
        pool_sizes[g] = len(pool)
        pools[g, :len(pool)] = pool.to(torch.int8)

    return pools, pool_sizes
