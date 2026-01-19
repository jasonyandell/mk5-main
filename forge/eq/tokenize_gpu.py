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

from dataclasses import dataclass

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


@dataclass
class PastStatesGPU:
    """GPU representation of past game states for posterior scoring.

    For N games and K past steps (padded to max_k).
    All tensors should be on the same device.

    Attributes:
        played_before: [N, K, 28] bool - which dominoes played before each step
        trick_plays: [N, K, 3, 2] int32 - trick plays at each step (player, domino)
        trick_lens: [N, K] int32 - number of valid trick plays (0-3)
        leaders: [N, K] int32 - trick leader for each step (0-3)
        actors: [N, K] int32 - acting player for each step (0-3)
        observed_actions: [N, K] int32 - observed domino IDs
        step_indices: [N, K] int32 - global step indices in play history
        valid_mask: [N, K] bool - which steps are valid (for padding)
    """
    played_before: Tensor      # [N, K, 28] bool
    trick_plays: Tensor        # [N, K, 3, 2] int32
    trick_lens: Tensor         # [N, K] int32
    leaders: Tensor            # [N, K] int32
    actors: Tensor             # [N, K] int32
    observed_actions: Tensor   # [N, K] int32
    step_indices: Tensor       # [N, K] int32
    valid_mask: Tensor         # [N, K] bool


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

    def tokenize_past_steps(
        self,
        worlds: Tensor,              # [N, M, 4, 7] sampled worlds per game
        past_states: PastStatesGPU,  # Reconstructed states for K past steps
        decl_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Tokenize all N×M×K combinations for posterior scoring.

        For each game g, sample m, and past step k:
        - Use worlds[g, m] as the hypothetical deal
        - Use past_states for that step's context (trick_plays, leader, actor, remaining)

        Args:
            worlds: [N, M, 4, 7] hypothetical hands per sample
            past_states: Reconstructed states for K past steps
            decl_id: Declaration ID (0-9)

        Returns:
            tokens: [N*M*K, 32, 12] int8 tokens
            masks: [N*M*K, 32] int8 attention masks
        """
        N, M = worlds.shape[:2]
        K = past_states.actors.shape[1]

        # Total number of tokenizations needed
        total_batch = N * M * K

        if total_batch > self.max_batch:
            raise ValueError(
                f"Total batch size {total_batch} (N={N}, M={M}, K={K}) exceeds max_batch {self.max_batch}"
            )

        # Pre-allocate output tensors
        all_tokens = torch.zeros(
            total_batch, MAX_TOKENS, N_FEATURES,
            dtype=torch.int8, device=self.device
        )
        all_masks = torch.zeros(
            total_batch, MAX_TOKENS,
            dtype=torch.int8, device=self.device
        )

        # Process each game and step separately (since trick_plays varies)
        # Output layout: [g0k0m0, g0k0m1, ..., g0k0m(M-1), g0k1m0, ..., g0k(K-1)m(M-1), g1k0m0, ...]
        # So index = g*M*K + k*M + m
        batch_idx = 0

        for g in range(N):
            for k in range(K):
                if not past_states.valid_mask[g, k]:
                    # Skip invalid steps (padding) - leave as zeros
                    batch_idx += M
                    continue

                # Get context for this game and step
                current_player = past_states.actors[g, k].item()
                leader = past_states.leaders[g, k].item()
                trick_len = past_states.trick_lens[g, k].item()

                # Convert trick_plays to list format
                trick_plays_list = []
                for t in range(trick_len):
                    player = past_states.trick_plays[g, k, t, 0].item()
                    domino = past_states.trick_plays[g, k, t, 1].item()
                    trick_plays_list.append((player, domino))

                # Compute remaining bits for all M samples at this game/step
                # played_before: [28] bool for this game/step
                played = past_states.played_before[g, k]  # [28] bool

                # For each sample m, compute which dominoes are remaining
                # remaining[m, p] = bitmask of which slots in worlds[g, m, p] are not played
                remaining = torch.zeros(M, 4, dtype=torch.int64, device=self.device)

                for m in range(M):
                    for p in range(4):
                        for slot in range(7):
                            domino_id = worlds[g, m, p, slot].item()
                            if not played[domino_id]:
                                remaining[m, p] |= (1 << slot)

                # Tokenize all M samples for this game/step
                worlds_gm = worlds[g, :, :, :]  # [M, 4, 7]

                tokens_batch, masks_batch = self.tokenize(
                    worlds=worlds_gm,
                    decl_id=decl_id,
                    leader=leader,
                    trick_plays=trick_plays_list,
                    remaining=remaining,
                    current_player=current_player,
                )

                # Store in output tensor
                all_tokens[batch_idx:batch_idx + M] = tokens_batch
                all_masks[batch_idx:batch_idx + M] = masks_batch
                batch_idx += M

        return all_tokens, all_masks

    def _compute_remaining_batched(
        self,
        worlds: Tensor,        # [batch, 4, 7] domino IDs
        played_before: Tensor, # [batch, 28] bool mask
    ) -> Tensor:
        """Compute remaining bitmasks without any Python loops.

        Returns:
            remaining: [batch, 4] int64 bitmasks where bit i is set if slot i is remaining
        """
        batch = worlds.shape[0]
        device = worlds.device

        # Step 1: Look up played status for each domino
        # worlds: [batch, 4, 7] -> flat_worlds: [batch, 28]
        flat_worlds = worlds.reshape(batch, 28).long()

        # Handle padding: -1 values indicate empty slots (no domino)
        # Clamp to valid range [0, 27] for gather - empty slots will be marked as "played"
        padding_mask = (flat_worlds < 0)  # [batch, 28] where True = padding
        flat_worlds_clamped = flat_worlds.clamp(0, 27)  # Safe indices for gather

        # Use gather to look up played_before[b, worlds[b, p, s]]
        # played_before: [batch, 28] bool
        # flat_worlds_clamped: [batch, 28] (indices in [0, 27])
        # Result: is_played[b, i] = played_before[b, flat_worlds_clamped[b, i]]
        is_played = torch.gather(played_before, 1, flat_worlds_clamped)  # [batch, 28] bool

        # Mark padding positions as "played" so they won't appear in remaining
        is_played = is_played | padding_mask

        # Step 2: Reshape to [batch, 4, 7]
        is_played = is_played.reshape(batch, 4, 7)
        is_remaining = ~is_played  # [batch, 4, 7] bool

        # Step 3: Compute bitmask using powers of 2
        # bitmask[b, p] = sum over s of (is_remaining[b, p, s] * 2^s)
        powers_of_2 = (2 ** torch.arange(7, device=device)).unsqueeze(0).unsqueeze(0)  # [1, 1, 7]

        # Multiply and sum: [batch, 4, 7] * [1, 1, 7] -> [batch, 4, 7] -> sum -> [batch, 4]
        remaining = (is_remaining.to(torch.int64) * powers_of_2).sum(dim=2)  # [batch, 4]

        return remaining

    def tokenize_batched(
        self,
        worlds: Tensor,            # [batch, 4, 7] domino IDs
        decl_ids: Tensor,          # [batch] declaration IDs
        leaders: Tensor,           # [batch] trick leaders
        current_players: Tensor,   # [batch] current players
        trick_plays: Tensor,       # [batch, 3, 2] (player, domino) per position
        trick_lens: Tensor,        # [batch] number of valid trick plays (0-3)
        remaining: Tensor,         # [batch, 4] bitmasks
    ) -> tuple[Tensor, Tensor]:
        """Fully vectorized tokenization - no Python loops, no .item() calls.

        Args:
            worlds: [batch, 4, 7] tensor of domino IDs for each player
            decl_ids: [batch] declaration IDs (0-9)
            leaders: [batch] trick leader player IDs (0-3)
            current_players: [batch] current player IDs (0-3)
            trick_plays: [batch, 3, 2] (player, domino) pairs for current trick
            trick_lens: [batch] number of valid trick plays (0-3) for each batch
            remaining: [batch, 4] tensor of remaining domino bitmasks

        Returns:
            Tuple of (tokens, masks):
                - tokens: [batch, 32, 12] int8 tensor
                - masks: [batch, 32] int8 tensor
        """
        batch = worlds.shape[0]
        device = worlds.device

        if batch > self.max_batch:
            raise ValueError(f"Batch size {batch} exceeds max_batch {self.max_batch}")

        # Get views into pre-allocated buffers
        tokens = self.tokens[:batch]
        masks = self.masks[:batch]

        # Clear buffers (critical for correctness)
        tokens.fill_(0)
        masks.fill_(0)

        # Normalize leader relative to current player: [batch]
        normalized_leaders = (leaders - current_players + 4) % 4

        # =====================================================================
        # Step 1: Look up domino features for all hands (per-batch decl lookup)
        # =====================================================================
        # worlds: [batch, 4, 7], flatten to [batch, 28]
        flat_ids = worlds.reshape(batch, 28).long()

        # Handle padding: -1 values indicate empty slots (no domino)
        # Clamp to valid range [0, 27] for gather
        flat_ids_clamped = flat_ids.clamp(0, 27)

        # Get features for all dominoes with per-batch decl_id:
        # self.domino_features: [10, 28, 5]
        # decl_ids: [batch] -> [batch, 1, 1] for indexing
        # First select decl features: [batch, 28, 5]
        decl_features = self.domino_features[decl_ids.long()]  # [batch, 28, 5]

        # Now gather based on flat_ids_clamped: for each batch b and position i,
        # we want decl_features[b, flat_ids_clamped[b, i], :]
        # Expand flat_ids_clamped for gather: [batch, 28] -> [batch, 28, 5]
        flat_ids_expanded = flat_ids_clamped.unsqueeze(-1).expand(-1, -1, 5)
        all_features = torch.gather(decl_features, 1, flat_ids_expanded)  # [batch, 28, 5]

        # Assign to hand token features (tokens[:, 1:29, 0:5])
        tokens[:, 1:29, 0:5] = all_features

        # =====================================================================
        # Step 2: Vectorized per-batch player normalization and flags
        # =====================================================================
        # Normalize player indices relative to current player
        # self.player_ids: [28] -> [0,0,0,0,0,0,0, 1,1,1,1,1,1,1, ...]
        # current_players: [batch]
        # normalized_players[b, i] = (player_ids[i] - current_players[b] + 4) % 4
        normalized_players = (self.player_ids.unsqueeze(0) - current_players.unsqueeze(1) + 4) % 4  # [batch, 28]

        # Assign to tokens
        tokens[:, 1:29, 5] = normalized_players.to(torch.int8)
        tokens[:, 1:29, 6] = (normalized_players == 0).to(torch.int8)
        tokens[:, 1:29, 7] = (normalized_players == 2).to(torch.int8)

        # =====================================================================
        # Step 3: Vectorized remaining bits
        # =====================================================================
        # For each token position, check if corresponding bit is set
        # remaining: [batch, 4], player_for_token: [28], local_indices: [28]
        # remaining_bits[b, token] = (remaining[b, player] >> local_idx) & 1

        # Select player's remaining mask for each token: [batch, 28]
        player_remaining = remaining[:, self.player_for_token.long()]  # [batch, 28]

        # Shift by local index and mask: [batch, 28]
        remaining_bits = (player_remaining >> self.local_indices.long()) & 1

        tokens[:, 1:29, 8] = remaining_bits.to(torch.int8)

        # =====================================================================
        # Step 4: Token type and context (broadcast scalar, per-batch decl/leader)
        # =====================================================================
        tokens[:, 1:29, 9] = self.token_types.unsqueeze(0)  # Broadcast [28] -> [batch, 28]
        # Per-batch decl_id and normalized_leader
        tokens[:, 1:29, 10] = decl_ids.unsqueeze(1).to(torch.int8)  # [batch, 1] -> [batch, 28]
        tokens[:, 1:29, 11] = normalized_leaders.unsqueeze(1).to(torch.int8)  # [batch, 1] -> [batch, 28]

        # Mark all hand tokens as present
        masks[:, 1:29] = 1

        # =====================================================================
        # Step 5: Context token (per-batch values)
        # =====================================================================
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_ids.to(torch.int8)
        tokens[:, 0, 11] = normalized_leaders.to(torch.int8)
        masks[:, 0] = 1

        # =====================================================================
        # Step 6: Trick tokens (vectorized with variable length)
        # =====================================================================
        # trick_plays: [batch, 3, 2] where [:, :, 0] is player, [:, :, 1] is domino
        # trick_lens: [batch] number of valid plays (0-3)

        # Create position indices: [1, 3]
        trick_positions = torch.arange(3, device=device).unsqueeze(0)  # [1, 3]

        # Valid mask: [batch, 3] - which trick positions are valid
        valid_trick_mask = trick_positions < trick_lens.unsqueeze(1)  # [batch, 3]

        # Extract player and domino from trick_plays
        trick_players = trick_plays[:, :, 0]  # [batch, 3]
        trick_dominoes = trick_plays[:, :, 1].long()  # [batch, 3]

        # Normalize trick players relative to current_players: [batch, 3]
        normalized_trick_players = (trick_players - current_players.unsqueeze(1) + 4) % 4

        # Get domino features for trick dominoes
        # Need per-batch decl features for trick dominoes
        # decl_features: [batch, 28, 5] (already computed above)
        # Clamp domino indices to valid range (invalid slots have 0, which is still valid)
        trick_dominoes_clamped = trick_dominoes.clamp(0, 27)  # [batch, 3]
        trick_dominoes_expanded = trick_dominoes_clamped.unsqueeze(-1).expand(-1, -1, 5)
        trick_domino_features = torch.gather(decl_features, 1, trick_dominoes_expanded)  # [batch, 3, 5]

        # Token indices for tricks: 29, 30, 31
        # tokens[:, 29:32, :] are trick tokens

        # Fill in trick token features where valid
        # Use broadcasting with mask: [batch, 3, 1] mask -> expand as needed

        # Features 0:5 - domino features
        tokens[:, 29:32, 0:5] = (trick_domino_features * valid_trick_mask.unsqueeze(-1)).to(torch.int8)

        # Feature 5 - normalized player
        tokens[:, 29:32, 5] = (normalized_trick_players * valid_trick_mask).to(torch.int8)

        # Feature 6 - is_self (normalized_player == 0)
        is_self_trick = (normalized_trick_players == 0).to(torch.int8)
        tokens[:, 29:32, 6] = is_self_trick * valid_trick_mask.to(torch.int8)

        # Feature 7 - is_partner (normalized_player == 2)
        is_partner_trick = (normalized_trick_players == 2).to(torch.int8)
        tokens[:, 29:32, 7] = is_partner_trick * valid_trick_mask.to(torch.int8)

        # Feature 8 - remaining (always 0 for trick tokens - already played)
        # Already 0 from fill_

        # Feature 9 - token type: TOKEN_TYPE_TRICK_P0 + position
        # [batch, 3] where value is TOKEN_TYPE_TRICK_P0 + position
        trick_token_types = TOKEN_TYPE_TRICK_P0 + trick_positions  # [1, 3]
        tokens[:, 29:32, 9] = (trick_token_types * valid_trick_mask).to(torch.int8)

        # Feature 10 - decl_id (per-batch)
        decl_broadcast = decl_ids.unsqueeze(1).expand(-1, 3)  # [batch, 3]
        tokens[:, 29:32, 10] = (decl_broadcast * valid_trick_mask).to(torch.int8)

        # Feature 11 - normalized leader (per-batch)
        leader_broadcast = normalized_leaders.unsqueeze(1).expand(-1, 3)  # [batch, 3]
        tokens[:, 29:32, 11] = (leader_broadcast * valid_trick_mask).to(torch.int8)

        # Set masks for valid trick tokens
        masks[:, 29:32] = valid_trick_mask.to(torch.int8)

        return tokens, masks

    def tokenize_past_steps_batched(
        self,
        worlds: Tensor,              # [N, M, 4, 7] sampled worlds per game
        past_states: PastStatesGPU,  # Reconstructed states for K past steps
        decl_id: int,
    ) -> tuple[Tensor, Tensor]:
        """Fully vectorized tokenization of N×M×K combinations.

        VECTORIZED IMPLEMENTATION - no Python loops or .item() calls.

        For each game g, sample m, and past step k:
        - Use worlds[g, m] as the hypothetical deal
        - Use past_states for context (trick_plays, leader, actor, remaining)

        Output layout: [g0k0m0, g0k0m1, ..., g0k(K-1)m(M-1), g1k0m0, ...]
        i.e., index = g*K*M + k*M + m

        Args:
            worlds: [N, M, 4, 7] hypothetical hands per sample
            past_states: Reconstructed states for K past steps
            decl_id: Declaration ID (0-9)

        Returns:
            tokens: [N*K*M, 32, 12] int8 tokens
            masks: [N*K*M, 32] int8 attention masks
        """
        N, M = worlds.shape[:2]
        K = past_states.actors.shape[1]
        device = worlds.device

        total_batch = N * K * M

        if total_batch > self.max_batch:
            raise ValueError(
                f"Total batch size {total_batch} (N={N}, M={M}, K={K}) exceeds max_batch {self.max_batch}"
            )

        # =====================================================================
        # Step 1: Expand worlds to [N, K, M, 4, 7] then flatten to [N*K*M, 4, 7]
        # Same world is used for all K steps within a game
        # =====================================================================
        # worlds: [N, M, 4, 7] -> [N, 1, M, 4, 7] -> [N, K, M, 4, 7]
        worlds_expanded = worlds.unsqueeze(1).expand(N, K, M, 4, 7)
        # Flatten to [N*K*M, 4, 7]
        flat_worlds = worlds_expanded.reshape(total_batch, 4, 7)

        # =====================================================================
        # Step 2: Expand past_states to match [N, K, M] and flatten
        # =====================================================================
        # For each field with shape [N, K, ...], expand to [N, K, M, ...] then flatten

        # decl_ids: all same value, just broadcast
        flat_decl_ids = torch.full((total_batch,), decl_id, dtype=torch.int32, device=device)

        # leaders: [N, K] -> [N, K, M] -> [N*K*M]
        flat_leaders = past_states.leaders.unsqueeze(-1).expand(N, K, M).reshape(total_batch)

        # actors (current_players): [N, K] -> [N, K, M] -> [N*K*M]
        flat_current_players = past_states.actors.unsqueeze(-1).expand(N, K, M).reshape(total_batch)

        # trick_plays: [N, K, 3, 2] -> [N, K, M, 3, 2] -> [N*K*M, 3, 2]
        flat_trick_plays = past_states.trick_plays.unsqueeze(2).expand(N, K, M, 3, 2).reshape(total_batch, 3, 2)

        # trick_lens: [N, K] -> [N, K, M] -> [N*K*M]
        flat_trick_lens = past_states.trick_lens.unsqueeze(-1).expand(N, K, M).reshape(total_batch)

        # valid_mask: [N, K] -> [N, K, M] -> [N*K*M]
        flat_valid_mask = past_states.valid_mask.unsqueeze(-1).expand(N, K, M).reshape(total_batch)

        # =====================================================================
        # Step 3: Compute remaining bits for all N×K×M combinations
        # =====================================================================
        # played_before: [N, K, 28] -> [N, K, M, 28] -> [N*K*M, 28]
        flat_played_before = past_states.played_before.unsqueeze(2).expand(N, K, M, 28).reshape(total_batch, 28)

        # Use _compute_remaining_batched
        flat_remaining = self._compute_remaining_batched(flat_worlds, flat_played_before)

        # =====================================================================
        # Step 4: Call tokenize_batched with all inputs
        # =====================================================================
        tokens, masks = self.tokenize_batched(
            worlds=flat_worlds,
            decl_ids=flat_decl_ids,
            leaders=flat_leaders,
            current_players=flat_current_players,
            trick_plays=flat_trick_plays,
            trick_lens=flat_trick_lens,
            remaining=flat_remaining,
        )

        # =====================================================================
        # Step 5: Zero out invalid entries based on valid_mask
        # =====================================================================
        # Where valid_mask is False, set tokens and masks to 0
        invalid_mask = ~flat_valid_mask  # [N*K*M]
        tokens[invalid_mask] = 0
        masks[invalid_mask] = 0

        return tokens, masks
