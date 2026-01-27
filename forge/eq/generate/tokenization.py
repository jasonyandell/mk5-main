"""Tokenization functions for GPU E[Q] pipeline."""

from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.tokenize_gpu import GPUTokenizer


def tokenize_batched(
    states: GameStateTensor,
    deals: Tensor,
    tokenizer: GPUTokenizer,
) -> tuple[Tensor, Tensor]:
    """Tokenize all game states and deals.

    Vectorized implementation - no Python loops or .item() calls.

    Args:
        states: GameStateTensor with n_games
        deals: [n_games, n_samples, 4, 7] hypothetical deals
        tokenizer: Pre-allocated GPUTokenizer

    Returns:
        Tuple of (tokens, masks):
            - tokens: [n_games * n_samples, 32, 12] int8
            - masks: [n_games * n_samples, 32] int8
    """
    n_games = states.n_games
    n_samples = deals.shape[1]
    batch_size = n_games * n_samples
    device = states.device

    # Flatten deals: [batch, 4, 7]
    deals_flat = deals.reshape(batch_size, 4, 7)

    # Build remaining bitmasks
    remaining = build_remaining_bitmasks(states, deals)  # [batch, 4]

    # === Prepare batched inputs for tokenizer.tokenize_batched() ===

    # decl_ids: [n_games] -> expand to [batch]
    decl_ids_expanded = states.decl_ids.unsqueeze(1).expand(n_games, n_samples).reshape(batch_size)

    # leaders: [n_games] -> expand to [batch]
    leaders_expanded = states.leader.unsqueeze(1).expand(n_games, n_samples).reshape(batch_size)

    # current_players: [n_games] -> expand to [batch]
    current_players_expanded = states.current_player.unsqueeze(1).expand(n_games, n_samples).reshape(batch_size)

    # trick_plays: [n_games, 4] domino IDs -> need to convert to [batch, 3, 2] (player, domino) format
    # trick_lens: [n_games] number of valid trick plays

    # Count valid trick plays per game (clamp to 3 max, as tokenizer only handles 3 trick tokens)
    trick_lens = torch.clamp((states.trick_plays >= 0).sum(dim=1), max=3)  # [n_games]

    # Build trick_plays tensor: [n_games, 3, 2] where [:, :, 0] is player, [:, :, 1] is domino
    # states.trick_plays: [n_games, 4] contains domino IDs in play order
    # We need to compute player IDs based on leader + position
    trick_plays_tensor = torch.zeros(n_games, 3, 2, dtype=torch.int32, device=device)

    # For each position (0-2), compute player = (leader + position) % 4 vectorized
    positions = torch.arange(3, device=device).unsqueeze(0)  # [1, 3]
    trick_players = (states.leader.unsqueeze(1) + positions) % 4  # [n_games, 3]

    # Get domino IDs for positions 0-2
    trick_dominoes = states.trick_plays[:, :3]  # [n_games, 3]

    # Store in trick_plays_tensor
    trick_plays_tensor[:, :, 0] = trick_players
    trick_plays_tensor[:, :, 1] = trick_dominoes

    # Expand trick_plays to [batch, 3, 2]
    trick_plays_expanded = trick_plays_tensor.unsqueeze(1).expand(n_games, n_samples, 3, 2).reshape(batch_size, 3, 2)

    # Expand trick_lens to [batch]
    trick_lens_expanded = trick_lens.unsqueeze(1).expand(n_games, n_samples).reshape(batch_size)

    # Call vectorized tokenization
    tokens, masks = tokenizer.tokenize_batched(
        worlds=deals_flat,
        decl_ids=decl_ids_expanded,
        leaders=leaders_expanded,
        current_players=current_players_expanded,
        trick_plays=trick_plays_expanded,
        trick_lens=trick_lens_expanded,
        remaining=remaining,
    )

    return tokens, masks


def build_remaining_bitmasks(
    states: GameStateTensor,
    deals: Tensor,
) -> Tensor:
    """Build remaining domino bitmasks.

    Args:
        states: GameStateTensor with played history
        deals: [n_games, n_samples, 4, 7] hypothetical deals

    Returns:
        [n_games * n_samples, 4] bitmasks
    """
    n_games = states.n_games
    n_samples = deals.shape[1]
    batch_size = n_games * n_samples

    # Flatten deals
    deals_flat = deals.reshape(batch_size, 4, 7)

    # Expand played_mask: [n_games, 28] -> [batch, 28]
    played_mask = states.played_mask.unsqueeze(1).expand(-1, n_samples, -1).reshape(batch_size, 28)

    # Build bitmasks
    remaining = torch.zeros(batch_size, 4, dtype=torch.int32, device=states.device)

    for player in range(4):
        hand = deals_flat[:, player, :]  # [batch, 7]
        for local_idx in range(7):
            domino_ids = hand[:, local_idx].long()  # [batch]

            # Check if valid and not played
            valid = domino_ids >= 0
            batch_indices = torch.arange(batch_size, device=states.device)

            # Safe indexing: use 0 for invalid, then mask out
            domino_ids_safe = torch.where(valid, domino_ids, torch.zeros_like(domino_ids))
            is_played = played_mask[batch_indices, domino_ids_safe]

            is_remaining = valid & ~is_played
            remaining[:, player] |= is_remaining.int() << local_idx

    return remaining
