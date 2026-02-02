"""GPU-accelerated tokenizer for Stage1Oracle.

Moves tokenization to GPU to eliminate CPU bottleneck and achieve ~100% GPU utilization.
"""

from __future__ import annotations

import numpy as np
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
from forge.oracle.declarations import N_DECLS


class GPUTokenizer:
    """GPU-based tokenizer for oracle queries.

    All lookup tables are stored as GPU tensors. Tokenization happens
    entirely on GPU with no CPU-GPU data transfer during inference.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

        # Build domino features lookup table: (10, 28, 5)
        # [decl_id, domino_id] -> [high, low, is_double, count_value, trump_rank]
        domino_features = torch.zeros((N_DECLS, 28, 5), dtype=torch.int32, device=device)

        high = torch.tensor(DOMINO_HIGH, dtype=torch.int32, device=device)
        low = torch.tensor(DOMINO_LOW, dtype=torch.int32, device=device)
        is_double = torch.tensor([1 if v else 0 for v in DOMINO_IS_DOUBLE], dtype=torch.int32, device=device)
        count_val = torch.tensor([COUNT_VALUE_MAP[p] for p in DOMINO_COUNT_POINTS], dtype=torch.int32, device=device)

        for decl_id in range(N_DECLS):
            domino_features[decl_id, :, 0] = high
            domino_features[decl_id, :, 1] = low
            domino_features[decl_id, :, 2] = is_double
            domino_features[decl_id, :, 3] = count_val
            trump_ranks = torch.tensor(
                [TRUMP_RANK_TABLE[(d, decl_id)] for d in range(28)],
                dtype=torch.int32, device=device
            )
            domino_features[decl_id, :, 4] = trump_ranks

        self.domino_features = domino_features  # (10, 28, 5)

        # Player ID for each of the 28 domino tokens (4 players x 7 dominoes)
        # Tokens 0-6: player 0, tokens 7-13: player 1, etc.
        player_ids = torch.tensor([p for p in range(4) for _ in range(7)], dtype=torch.int32, device=device)
        self.player_ids = player_ids  # (28,)

        # Local index within hand (0-6) for each token
        local_indices = torch.tensor([i for _ in range(4) for i in range(7)], dtype=torch.int32, device=device)
        self.local_indices = local_indices  # (28,)

        # Player for each token position (for remaining bit extraction)
        self.player_for_token = player_ids  # Same as player_ids

        # Token types: player 0-3 have types TOKEN_TYPE_PLAYER0 + player_id
        token_types = torch.tensor(
            [TOKEN_TYPE_PLAYER0 + p for p in range(4) for _ in range(7)],
            dtype=torch.int32, device=device
        )
        self.token_types = token_types  # (28,)

        # Pre-allocate output buffers (will resize if needed)
        self._max_batch = 0
        self._tokens_buf = None
        self._masks_buf = None

    def _ensure_buffers(self, n: int):
        """Ensure output buffers are large enough."""
        if n > self._max_batch:
            self._max_batch = max(n, self._max_batch * 2, 1024)
            self._tokens_buf = torch.zeros(
                (self._max_batch, MAX_TOKENS, N_FEATURES),
                dtype=torch.int32, device=self.device
            )
            self._masks_buf = torch.zeros(
                (self._max_batch, MAX_TOKENS),
                dtype=torch.int32, device=self.device
            )

    def tokenize(
        self,
        worlds: Tensor,  # (N, 4, 7) int32 domino IDs
        decl_ids: Tensor,  # (N,) int32
        actors: Tensor,  # (N,) int32
        leaders: Tensor,  # (N,) int32
        remaining: Tensor,  # (N, 4) int64 bitmasks
        trick_players: Tensor | None = None,  # (N, max_tricks) int32, -1 for no play
        trick_dominoes: Tensor | None = None,  # (N, max_tricks) int32, -1 for no play
    ) -> tuple[Tensor, Tensor]:
        """Tokenize batch on GPU.

        Args:
            worlds: Domino IDs for each player's hand (N, 4, 7)
            decl_ids: Declaration ID per sample (N,)
            actors: Current player per sample (N,)
            leaders: Trick leader per sample (N,)
            remaining: Bitmask of remaining dominoes per player (N, 4)
            trick_players: Player who made each trick play (N, max_tricks), -1 if no play
            trick_dominoes: Domino played in each trick position (N, max_tricks), -1 if no play

        Returns:
            tokens: (N, MAX_TOKENS, N_FEATURES) int32
            masks: (N, MAX_TOKENS) int32
        """
        n = worlds.shape[0]
        self._ensure_buffers(n)

        tokens = self._tokens_buf[:n]
        masks = self._masks_buf[:n]

        # Zero out buffers
        tokens.zero_()
        masks.zero_()

        # Flatten worlds: (N, 4, 7) -> (N, 28)
        flat_ids = worlds.reshape(n, 28)

        # === Domino tokens (positions 1-28) ===

        # Lookup domino features: (N, 28, 5)
        # Use advanced indexing: domino_features[decl_ids[i], flat_ids[i, j]]
        # Expand decl_ids for broadcasting: (N, 1) -> index into first dim
        # flat_ids already (N, 28) -> index into second dim
        features = self.domino_features[decl_ids.unsqueeze(1).expand(-1, 28), flat_ids]
        tokens[:, 1:29, 0:5] = features

        # Normalized player: (player_id - actor + 4) % 4
        # player_ids is (28,), actors is (N,) -> broadcast to (N, 28)
        normalized_players = (self.player_ids.unsqueeze(0) - actors.unsqueeze(1) + 4) % 4
        tokens[:, 1:29, 5] = normalized_players
        tokens[:, 1:29, 6] = (normalized_players == 0).int()  # is_current
        tokens[:, 1:29, 7] = (normalized_players == 2).int()  # is_partner

        # Remaining bits: (remaining[player] >> local_idx) & 1
        # remaining: (N, 4), player_for_token: (28,), local_indices: (28,)
        # Gather remaining for each token's player: (N, 28)
        remaining_per_token = remaining[:, self.player_for_token]  # (N, 28)
        remaining_bits = (remaining_per_token >> self.local_indices.unsqueeze(0)) & 1
        tokens[:, 1:29, 8] = remaining_bits.int()

        # Token type
        tokens[:, 1:29, 9] = self.token_types.unsqueeze(0).expand(n, -1)

        # Decl ID (broadcast)
        tokens[:, 1:29, 10] = decl_ids.unsqueeze(1).expand(-1, 28)

        # Normalized leader
        normalized_leaders = (leaders - actors + 4) % 4
        tokens[:, 1:29, 11] = normalized_leaders.unsqueeze(1).expand(-1, 28)

        # Mask domino tokens
        masks[:, 1:29] = 1

        # === Context token (position 0) ===
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_ids
        tokens[:, 0, 11] = normalized_leaders
        masks[:, 0] = 1

        # === Trick tokens (positions 29-31) - fully vectorized ===
        if trick_players is not None and trick_dominoes is not None:
            # Process all 3 trick positions at once
            # trick_players: (N, 3), trick_dominoes: (N, 3)

            # Valid mask for each position: (N, 3)
            valid = trick_dominoes >= 0

            # Clamp domino IDs for safe indexing
            safe_domino_ids = trick_dominoes.clamp(min=0)  # (N, 3)

            # Get features for all trick positions: (N, 3, 5)
            # Index: domino_features[decl_ids[i], safe_domino_ids[i, j]]
            # Expand decl_ids: (N,) -> (N, 3)
            decl_ids_expanded = decl_ids.unsqueeze(1).expand(-1, 3)
            trick_features = self.domino_features[decl_ids_expanded, safe_domino_ids]  # (N, 3, 5)

            # Normalized players: (N, 3)
            normalized_pp = (trick_players - actors.unsqueeze(1) + 4) % 4

            # Token type for each position: TOKEN_TYPE_TRICK_P0 + [0, 1, 2]
            trick_token_types = torch.tensor(
                [TOKEN_TYPE_TRICK_P0, TOKEN_TYPE_TRICK_P0 + 1, TOKEN_TYPE_TRICK_P0 + 2],
                dtype=torch.int32, device=trick_players.device
            ).unsqueeze(0).expand(n, -1)  # (N, 3)

            # Build trick tokens: (N, 3, 12)
            trick_tokens = torch.zeros((n, 3, N_FEATURES), dtype=torch.int32, device=trick_players.device)
            trick_tokens[:, :, 0:5] = trick_features
            trick_tokens[:, :, 5] = normalized_pp
            trick_tokens[:, :, 6] = (normalized_pp == 0).int()
            trick_tokens[:, :, 7] = (normalized_pp == 2).int()
            trick_tokens[:, :, 8] = 0  # Not in hand
            trick_tokens[:, :, 9] = trick_token_types
            trick_tokens[:, :, 10] = decl_ids.unsqueeze(1).expand(-1, 3)
            trick_tokens[:, :, 11] = normalized_leaders.unsqueeze(1).expand(-1, 3)

            # Apply valid mask: only set tokens where there's a play
            valid_expanded = valid.unsqueeze(-1).expand(-1, -1, N_FEATURES)  # (N, 3, 12)
            tokens[:, 29:32, :] = torch.where(valid_expanded, trick_tokens, tokens[:, 29:32, :])

            # Set masks for valid plays
            masks[:, 29:32] = valid.int()

        return tokens, masks


def convert_trick_plays_to_tensors(
    trick_plays_list: list[list[tuple[int, int]]],
    device: str = "cuda",
) -> tuple[Tensor, Tensor]:
    """Convert trick_plays_list to GPU tensors for vectorized processing.

    Args:
        trick_plays_list: List of N trick plays, each is [(player, domino_id), ...]
        device: Target device

    Returns:
        trick_players: (N, 3) int32, -1 for no play
        trick_dominoes: (N, 3) int32, -1 for no play
    """
    n = len(trick_plays_list)

    # Pre-fill with -1 (no play)
    players = np.full((n, 3), -1, dtype=np.int32)
    dominoes = np.full((n, 3), -1, dtype=np.int32)

    for i, trick_plays in enumerate(trick_plays_list):
        for j, (player, domino_id) in enumerate(trick_plays[:3]):
            players[i, j] = player
            dominoes[i, j] = domino_id

    return (
        torch.from_numpy(players).to(device),
        torch.from_numpy(dominoes).to(device),
    )
