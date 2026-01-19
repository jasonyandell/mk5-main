"""
GPU-native tokenization for E[Q] pipeline.

Converts game states to model input format using pure tensor operations.
Pre-allocates buffers and pre-computes lookup tables for <50ms latency.

Phase 3 of GPU-native E[Q] pipeline:
- Phase 1: GameStateTensor - GPU game state representation
- Phase 2: WorldSampler - GPU world sampling
- Phase 3: GPUTokenizer - GPU tokenization (this file)
- Phase 4: End-to-end GPU pipeline integration
"""
from __future__ import annotations

import torch
from torch import Tensor

from forge.ml.tokenize import (
    COUNT_VALUE_MAP,
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    MAX_TOKENS,
    N_FEATURES,
    TOKEN_TYPE_CONTEXT,
    TOKEN_TYPE_PLAYER0,
    TOKEN_TYPE_TRICK_P0,
    TRUMP_RANK_TABLE,
)


def _build_domino_features_by_decl() -> Tensor:
    """Build per-decl domino feature tensor for GPU lookup.

    Returns:
        Tensor of shape (10, 28, 5) where features are:
        [high, low, is_double, count_value, trump_rank]
    """
    import numpy as np

    # Base features (same for all decls except trump_rank)
    base = torch.zeros(28, 4, dtype=torch.int8)
    base[:, 0] = torch.tensor(DOMINO_HIGH, dtype=torch.int8)
    base[:, 1] = torch.tensor(DOMINO_LOW, dtype=torch.int8)
    base[:, 2] = torch.tensor([1 if v else 0 for v in DOMINO_IS_DOUBLE], dtype=torch.int8)
    base[:, 3] = torch.tensor([COUNT_VALUE_MAP[p] for p in DOMINO_COUNT_POINTS], dtype=torch.int8)

    # Stack all 10 declarations
    all_features = torch.zeros(10, 28, 5, dtype=torch.int8)
    all_features[:, :, :4] = base.unsqueeze(0)  # Broadcast base to all decls

    # Add trump_rank for each decl
    for decl_id in range(10):
        for domino_id in range(28):
            trump_rank = TRUMP_RANK_TABLE[(domino_id, decl_id)]
            all_features[decl_id, domino_id, 4] = trump_rank

    return all_features


# Module-level constant (moved to GPU on first use)
_DOMINO_FEATURES_BY_DECL = _build_domino_features_by_decl()


class GPUTokenizer:
    """GPU-native tokenizer for Stage 1 oracle queries.

    Pre-allocates output buffers and pre-computes feature tables for fast
    tokenization using pure tensor indexing (no Python loops).

    Target: <50ms for 1,600 batch on RTX 3050 Ti

    Example:
        >>> tokenizer = GPUTokenizer(max_batch=1600, device='cuda')
        >>> tokens, masks = tokenizer.tokenize(
        ...     worlds=worlds,  # [N, 4, 7] sampled hands
        ...     decl_id=3,
        ...     leader=0,
        ...     trick_plays=[(0, 5), (1, 12)],
        ...     remaining=remaining,  # [N, 4] bitmasks
        ...     current_player=2,
        ... )
        >>> # tokens: [N, 32, 12], masks: [N, 32]
    """

    def __init__(self, max_batch: int, device: str = 'cuda'):
        """Initialize tokenizer with pre-allocated buffers.

        Args:
            max_batch: Maximum batch size to support
            device: Device to place tensors on ('cuda', 'cpu', 'mps')
        """
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        self.device = device
        self.max_batch = max_batch

        # Pre-allocate output buffers
        self.tokens = torch.zeros(
            max_batch, MAX_TOKENS, N_FEATURES,
            dtype=torch.int8, device=device
        )
        self.masks = torch.zeros(
            max_batch, MAX_TOKENS,
            dtype=torch.int8, device=device
        )

        # Pre-compute and move feature tables to device once
        self.domino_features = _DOMINO_FEATURES_BY_DECL.to(device)

        # Pre-compute shared lookup arrays (same pattern as oracle.py)
        # For 28 hand positions: player_ids = [0,0,0,0,0,0,0, 1,1,1,1,1,1,1, ...]
        player_ids = torch.repeat_interleave(torch.arange(4, dtype=torch.int8), 7)
        self.player_ids = player_ids.to(device)

        # Token types for hand positions
        token_types = (TOKEN_TYPE_PLAYER0 + player_ids).to(torch.int8)
        self.token_types = token_types.to(device)

        # Local indices within each hand: [0,1,2,3,4,5,6, 0,1,2,3,4,5,6, ...]
        local_indices = torch.tile(torch.arange(7, dtype=torch.int8), (4,))
        self.local_indices = local_indices.to(device)

        # Player index for each token position (same as player_ids)
        self.player_for_token = self.player_ids

    def tokenize(
        self,
        worlds: Tensor,           # [N, 4, 7] domino IDs per player
        decl_id: int,
        leader: int,
        trick_plays: list[tuple[int, int]],  # [(player, domino_id), ...]
        remaining: Tensor,        # [N, 4] bitmasks of remaining dominoes
        current_player: int,
    ) -> tuple[Tensor, Tensor]:
        """Tokenize N worlds into model input format.

        Args:
            worlds: [N, 4, 7] tensor of domino IDs for each player
            decl_id: Declaration ID (0-9)
            leader: Trick leader player ID (0-3)
            trick_plays: List of (player, domino_id) for current trick
            remaining: [N, 4] tensor of remaining domino bitmasks
            current_player: Current player ID (0-3)

        Returns:
            Tuple of (tokens, masks):
                - tokens: [N, 32, 12] int8 tensor
                - masks: [N, 32] int8 tensor
        """
        n_worlds = worlds.shape[0]

        if n_worlds > self.max_batch:
            raise ValueError(f"Batch size {n_worlds} exceeds max_batch {self.max_batch}")

        # Get views into pre-allocated buffers
        tokens = self.tokens[:n_worlds]
        masks = self.masks[:n_worlds]

        # Clear buffers (critical for correctness)
        tokens.fill_(0)
        masks.fill_(0)

        # Normalize leader relative to current player
        normalized_leader = (leader - current_player + 4) % 4

        # =====================================================================
        # Step 1: Look up domino features for all hands
        # =====================================================================
        # worlds: [N, 4, 7], flatten to [N, 28]
        flat_ids = worlds.reshape(n_worlds, 28).long()

        # Get features for all dominoes: domino_features[decl_id, domino_ids]
        # domino_features: [10, 28, 5], flat_ids: [N, 28]
        # We want features[decl_id, flat_ids[i, j]] for each i, j
        # Use advanced indexing: features[decl_id, flat_ids]
        all_features = self.domino_features[decl_id, flat_ids]  # [N, 28, 5]

        # Assign to hand token features (tokens[:, 1:29, 0:5])
        tokens[:, 1:29, 0:5] = all_features

        # =====================================================================
        # Step 2: Vectorized player normalization and flags
        # =====================================================================
        # Normalize player indices relative to current player
        # player_ids: [28] -> [0,0,0,0,0,0,0, 1,1,1,1,1,1,1, ...]
        normalized_players = (self.player_ids - current_player + 4) % 4  # [28]

        # Broadcast to all worlds: tokens[:, 1:29, 5] = [N, 28]
        tokens[:, 1:29, 5] = normalized_players.unsqueeze(0)  # Broadcast
        tokens[:, 1:29, 6] = (normalized_players == 0).to(torch.int8).unsqueeze(0)
        tokens[:, 1:29, 7] = (normalized_players == 2).to(torch.int8).unsqueeze(0)

        # =====================================================================
        # Step 3: Vectorized remaining bits
        # =====================================================================
        # For each token position, check if corresponding bit is set
        # remaining: [N, 4], player_for_token: [28], local_indices: [28]
        # remaining_bits[world, token] = (remaining[world, player] >> local_idx) & 1

        # Select player's remaining mask for each token: [N, 28]
        player_remaining = remaining[:, self.player_for_token.long()]  # [N, 28]

        # Shift by local index and mask: [N, 28]
        remaining_bits = (player_remaining >> self.local_indices.long()) & 1

        tokens[:, 1:29, 8] = remaining_bits.to(torch.int8)

        # =====================================================================
        # Step 4: Token type and context (same for all worlds)
        # =====================================================================
        tokens[:, 1:29, 9] = self.token_types.unsqueeze(0)  # Broadcast
        tokens[:, 1:29, 10] = decl_id
        tokens[:, 1:29, 11] = normalized_leader

        # Mark all hand tokens as present
        masks[:, 1:29] = 1

        # =====================================================================
        # Step 5: Context token (same for all worlds)
        # =====================================================================
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_id
        tokens[:, 0, 11] = normalized_leader
        masks[:, 0] = 1

        # =====================================================================
        # Step 6: Trick tokens (broadcast across all worlds)
        # =====================================================================
        # trick_plays contains public information, identical for all worlds
        for trick_pos, (play_player, domino_id) in enumerate(trick_plays):
            if trick_pos >= 3:  # Max 3 trick tokens
                break

            token_idx = 29 + trick_pos
            normalized_pp = (play_player - current_player + 4) % 4

            # Get domino features for this domino
            # domino_features: [10, 28, 5]
            domino_feats = self.domino_features[decl_id, domino_id]  # [5]

            # Broadcast to all worlds
            tokens[:, token_idx, 0:5] = domino_feats
            tokens[:, token_idx, 5] = normalized_pp
            tokens[:, token_idx, 6] = 1 if normalized_pp == 0 else 0
            tokens[:, token_idx, 7] = 1 if normalized_pp == 2 else 0
            tokens[:, token_idx, 8] = 0  # Not in remaining (already played)
            tokens[:, token_idx, 9] = TOKEN_TYPE_TRICK_P0 + trick_pos
            tokens[:, token_idx, 10] = decl_id
            tokens[:, token_idx, 11] = normalized_leader

            masks[:, token_idx] = 1

        return tokens, masks
