"""
Stage 1 Oracle for E[Q] marginalization.

Wraps trained Stage 1 model for efficient batch querying of Q-values
across multiple sampled world states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from forge.ml.module import DominoLightningModule
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


def _build_domino_features_by_decl() -> list[np.ndarray]:
    """Precompute per-decl domino features for fast vectorized tokenization.

    Shape per decl: (28, 5) columns = [high, low, is_double, count_value, trump_rank]
    """
    base = np.zeros((28, 4), dtype=np.int32)
    base[:, 0] = np.asarray(DOMINO_HIGH, dtype=np.int32)
    base[:, 1] = np.asarray(DOMINO_LOW, dtype=np.int32)
    base[:, 2] = np.asarray([1 if v else 0 for v in DOMINO_IS_DOUBLE], dtype=np.int32)
    base[:, 3] = np.asarray([COUNT_VALUE_MAP[p] for p in DOMINO_COUNT_POINTS], dtype=np.int32)

    by_decl: list[np.ndarray] = []
    for decl_id in range(N_DECLS):
        feats = np.zeros((28, 5), dtype=np.int32)
        feats[:, :4] = base
        feats[:, 4] = np.asarray([TRUMP_RANK_TABLE[(d, decl_id)] for d in range(28)], dtype=np.int32)
        by_decl.append(feats)
    return by_decl


class Stage1Oracle:
    """Wrapper around trained Stage 1 model for querying Q-values.

    Design:
    - Load checkpoint once on initialization
    - Tokenize N sampled worlds in batch
    - Run single forward pass for efficiency
    - Return Q-values (expected points) for caller to process

    IMPORTANT: This oracle returns Q-values in POINTS (roughly [-42, +42] range),
    NOT logits. Do NOT apply softmax to the output - these are already interpretable
    point estimates of expected game outcome for each action.

    Example:
        oracle = Stage1Oracle("checkpoints/qval.ckpt")

        # Query 100 sampled worlds
        worlds = [sample_world() for _ in range(100)]
        q_values = oracle.query_batch(
            worlds=worlds,
            game_state_info={
                'decl_id': 3,
                'leader': 0,
                'trick_plays': [],
                'current_player': 0,
            },
            current_player=0
        )

        # Average Q-values across worlds: shape (7,)
        e_q = q_values.mean(dim=0)  # E[Q] in points
        best_action = e_q.argmax()  # Pick action with highest expected points
    """

    # Shared lookup tables (allow Stage1Oracle.__new__ usage in tests).
    _domino_features_by_decl = _build_domino_features_by_decl()
    _player_ids = np.repeat(np.arange(4, dtype=np.int32), 7)  # (28,)
    _token_types = (TOKEN_TYPE_PLAYER0 + _player_ids).astype(np.int32)  # (28,)
    _local_indices = np.tile(np.arange(7, dtype=np.int32), 4)  # (28,)
    _player_for_token = _player_ids  # (28,)

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda", compile: bool = True, use_async: bool = True):
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to Lightning checkpoint (.ckpt file)
            device: Device to load model on ("cuda", "cpu", "mps")
            compile: Whether to apply torch.compile optimization (default True).
                     Only applied on CUDA devices. Disable for testing/debugging.
            use_async: Whether to use async CUDA streams for overlapped execution (default True).
                       Only applies to CUDA devices. Requires pin_memory for host tensors.
        """
        # Load checkpoint directly, bypassing Lightning's on_load_checkpoint
        # which has RNG state compatibility issues. For inference we only need weights.
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

        # Extract hyperparameters and create model
        hparams = checkpoint['hyper_parameters']
        self.model = DominoLightningModule(
            embed_dim=hparams.get('embed_dim', 64),
            n_heads=hparams.get('n_heads', 4),
            n_layers=hparams.get('n_layers', 2),
            ff_dim=hparams.get('ff_dim', 128),
            dropout=hparams.get('dropout', 0.1),
            lr=hparams.get('lr', 1e-3),
        )

        # Load just the model weights (skip optimizer, RNG state, etc.)
        # Handle torch.compile() checkpoints which have "_orig_mod." prefix
        state_dict = checkpoint['state_dict']
        if any(k.startswith('model._orig_mod.') for k in state_dict.keys()):
            state_dict = {
                k.replace('._orig_mod.', '.'): v for k, v in state_dict.items()
            }
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.use_async = use_async and device == "cuda"

        # Pre-allocated buffers for tokenization (reused across calls)
        self._token_buffer: np.ndarray | None = None
        self._mask_buffer: np.ndarray | None = None
        self._buffer_size: int = 0

        # CUDA async pipeline (Phase 4: t42-tg2r)
        # Use separate streams for H2D transfers and inference to overlap work
        if self.use_async:
            self.stream_h2d = torch.cuda.Stream()  # Host-to-device transfers
            self.stream_compute = torch.cuda.Stream()  # Inference computation

            # Double buffering: two sets of GPU tensors, alternate between them
            # While GPU processes batch N, CPU prepares batch N+1
            self._gpu_buffers: list[dict[str, Tensor]] = [
                {
                    'tokens': torch.empty((1, MAX_TOKENS, N_FEATURES), dtype=torch.int32, device=device),
                    'masks': torch.empty((1, MAX_TOKENS), dtype=torch.int8, device=device),
                    'current_player': torch.empty((1,), dtype=torch.long, device=device),
                }
                for _ in range(2)
            ]
            self._buffer_idx = 0  # Which buffer to use next

            # Pinned host memory for fast H2D transfers
            # Pre-allocate with reasonable default size (will grow if needed)
            self._pinned_tokens: Tensor | None = None
            self._pinned_masks: Tensor | None = None
            self._pinned_size = 0

        # Apply torch.compile for GPU optimization (CUDA graph + reduced overhead)
        # Only compile on CUDA devices - CPU/MPS have compatibility issues
        if compile and device == "cuda":
            # Use fullgraph=False to handle TransformerEncoder's boolean src_key_padding_mask
            # which can cause graph breaks with strict fullgraph=True
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",  # Optimize for repeated calls with CUDA graphs
                fullgraph=False,  # Allow graph breaks for TransformerEncoder compatibility
            )

            # Warmup: Run a single forward pass to avoid JIT overhead at first inference
            # This compiles the execution graph ahead of time
            dummy_tokens = torch.zeros((1, MAX_TOKENS, N_FEATURES), dtype=torch.int32, device=device)
            dummy_masks = torch.ones((1, MAX_TOKENS), dtype=torch.int8, device=device)
            dummy_player = torch.zeros((1,), dtype=torch.long, device=device)
            with torch.inference_mode():
                _ = self.model(dummy_tokens, dummy_masks, dummy_player)

    def _get_pinned_buffers(self, n_worlds: int) -> tuple[Tensor, Tensor]:
        """Get pinned host memory buffers for fast async H2D transfers.

        Args:
            n_worlds: Number of worlds to transfer

        Returns:
            Tuple of (pinned_tokens, pinned_masks) views into pinned memory

        Design:
            - Uses torch pinned memory instead of numpy for async transfers
            - Allocates with headroom to avoid frequent reallocs
            - Lazy initialization for compatibility with non-async mode
        """
        if not self.use_async:
            raise RuntimeError("Pinned buffers only available in async mode")

        if n_worlds > self._pinned_size:
            # Allocate with headroom (2x or min 256) to avoid frequent reallocs
            new_size = max(n_worlds, self._pinned_size * 2, 256)
            # pin_memory=True enables fast DMA transfers, bypassing CPU caching
            self._pinned_tokens = torch.empty(
                (new_size, MAX_TOKENS, N_FEATURES),
                dtype=torch.int32,
                pin_memory=True
            )
            self._pinned_masks = torch.empty(
                (new_size, MAX_TOKENS),
                dtype=torch.int8,
                pin_memory=True
            )
            self._pinned_size = new_size

        # Return views into the pinned buffers
        return (
            self._pinned_tokens[:n_worlds],
            self._pinned_masks[:n_worlds]
        )

    def _ensure_gpu_buffer_size(self, n_worlds: int, buffer_idx: int):
        """Ensure GPU buffer can hold n_worlds.

        Args:
            n_worlds: Required capacity
            buffer_idx: Which buffer (0 or 1) to resize
        """
        if not self.use_async:
            raise RuntimeError("GPU buffers only available in async mode")

        buf = self._gpu_buffers[buffer_idx]
        current_size = buf['tokens'].shape[0]

        if n_worlds > current_size:
            # Allocate with headroom to avoid frequent reallocs
            new_size = max(n_worlds, current_size * 2, 256)
            buf['tokens'] = torch.empty(
                (new_size, MAX_TOKENS, N_FEATURES),
                dtype=torch.int32,
                device=self.device
            )
            buf['masks'] = torch.empty(
                (new_size, MAX_TOKENS),
                dtype=torch.int8,
                device=self.device
            )
            buf['current_player'] = torch.empty(
                (new_size,),
                dtype=torch.long,
                device=self.device
            )

    def query_batch(
        self,
        worlds: list[list[list[int]]],
        game_state_info: dict[str, Any],
        current_player: int,
    ) -> Tensor:
        """Query N sampled worlds in batch.

        Args:
            worlds: List of N possible world states, each is 4 hands of 7 domino IDs
                    Example: [[[0,1,2,3,4,5,6], [7,8,...], ...], ...]
            game_state_info: Dict with keys:
                - decl_id: Declaration ID (0-9)
                - leader: Leader player ID (0-3)
                - trick_plays: List of (player, domino_id) tuples for current trick
                  NOTE: Uses domino_id (not local_idx) for world-invariant encoding
                - remaining: 2D array (N, 4) of remaining domino bitmasks
            current_player: Who is deciding (0-3)

        Returns:
            Tensor of shape (N, 7) with Q-values (expected points) for each action

        Notes:
            - Returns Q-values in POINTS (roughly [-42, +42] range), NOT logits
            - Do NOT apply softmax - these are already interpretable point estimates
            - Caller should mask illegal moves with -inf before argmax
            - All N worlds must have same game state (decl, trick, etc.)
            - trick_plays uses domino_id (public info) for efficient batching
            - With use_async=True, uses CUDA streams for overlapped execution
        """
        if not worlds:
            raise ValueError("worlds cannot be empty")

        n_worlds = len(worlds)

        # Extract game state
        decl_id = game_state_info['decl_id']
        leader = game_state_info['leader']
        trick_plays = game_state_info.get('trick_plays', [])
        remaining = game_state_info['remaining']  # (N, 4) array

        # Tokenize all worlds (CPU work)
        tokens, masks = self._tokenize_worlds(
            worlds=worlds,
            decl_id=decl_id,
            leader=leader,
            trick_plays=trick_plays,
            remaining=remaining,
            current_player=current_player,
        )

        # Fast path: async mode with CUDA streams
        if self.use_async:
            return self._query_batch_async(tokens, masks, current_player, n_worlds)

        # Fallback: synchronous mode
        # Move to device and run forward pass.
        # Keep tokens as int32 (Embedding accepts int32/int64), avoiding per-call dtype expansion on CPU.
        tokens = torch.from_numpy(tokens).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)
        current_player_tensor = torch.full((n_worlds,), current_player, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            q_values, _ = self.model(tokens, masks, current_player_tensor)

        # Clone output to avoid CUDA graph tensor reuse issues with mode="reduce-overhead"
        return q_values.clone()

    def _query_batch_async(
        self,
        tokens_np: np.ndarray,
        masks_np: np.ndarray,
        current_player: int,
        n_worlds: int,
    ) -> Tensor:
        """Async variant using CUDA streams for overlapped execution.

        Args:
            tokens_np: Tokenized input (N, MAX_TOKENS, N_FEATURES) numpy array
            masks_np: Attention masks (N, MAX_TOKENS) numpy array
            current_player: Current player ID
            n_worlds: Number of worlds

        Returns:
            Q-values tensor (N, 7) on GPU

        Design (Phase 4 async pipeline):
            1. Copy numpy → pinned host memory (no GPU wait)
            2. Start async H2D transfer on stream_h2d
            3. Wait for transfer, then run inference on stream_compute
            4. Return result (caller decides when to sync)

        Note: Full double-buffering requires pipelining multiple calls,
        which would need API changes to expose "submit batch" + "wait batch".
        For now, we get async H2D transfers with non_blocking=True.
        """
        # Get pinned host buffers (fast to access from both CPU and GPU)
        pinned_tokens, pinned_masks = self._get_pinned_buffers(n_worlds)

        # Copy from numpy to pinned memory (CPU-only, no GPU wait)
        # This is fast because pinned memory is page-locked
        pinned_tokens.copy_(torch.from_numpy(tokens_np))
        pinned_masks.copy_(torch.from_numpy(masks_np))

        # Get current GPU buffer (for double buffering in future)
        buf_idx = self._buffer_idx
        self._ensure_gpu_buffer_size(n_worlds, buf_idx)
        buf = self._gpu_buffers[buf_idx]

        # Slice to actual size
        tokens_gpu = buf['tokens'][:n_worlds]
        masks_gpu = buf['masks'][:n_worlds]
        current_player_gpu = buf['current_player'][:n_worlds]

        # Fill current_player (scalar broadcast, very cheap)
        current_player_gpu.fill_(current_player)

        # Use H2D stream for async transfers
        with torch.cuda.stream(self.stream_h2d):
            # non_blocking=True: Don't wait for previous ops to finish
            # Returns immediately, GPU DMA runs in background
            tokens_gpu.copy_(pinned_tokens, non_blocking=True)
            masks_gpu.copy_(pinned_masks, non_blocking=True)

        # Wait for H2D transfer to complete before inference
        # This is necessary because inference depends on the data
        self.stream_compute.wait_stream(self.stream_h2d)

        # Run inference on compute stream
        with torch.cuda.stream(self.stream_compute):
            with torch.inference_mode():
                q_values, _ = self.model(tokens_gpu, masks_gpu, current_player_gpu)

        # Swap buffers for next call (enables future pipelining)
        self._buffer_idx = 1 - self._buffer_idx

        # Clone to avoid CUDA graph tensor reuse issues
        # This implicitly syncs the compute stream (clone needs the data)
        return q_values.clone()

    def _get_buffers(self, n_worlds: int) -> tuple[np.ndarray, np.ndarray]:
        """Get pre-allocated buffers, expanding if needed.

        Args:
            n_worlds: Number of worlds to tokenize

        Returns:
            Tuple of (tokens, masks) views into pre-allocated buffers

        Design:
            - Allocates buffers with headroom to avoid frequent reallocs
            - Returns views into the buffer for the requested size
            - Zeros out the portion being used for correctness
            - Lazy initialization for compatibility with __new__ usage in tests
        """
        # Lazy initialization (for __new__ usage in tests)
        if not hasattr(self, '_buffer_size'):
            self._token_buffer = None
            self._mask_buffer = None
            self._buffer_size = 0

        if n_worlds > self._buffer_size:
            # Allocate with headroom (2x or min 128) to avoid frequent reallocs
            new_size = max(n_worlds, self._buffer_size * 2, 128)
            self._token_buffer = np.zeros((new_size, MAX_TOKENS, N_FEATURES), dtype=np.int32)
            self._mask_buffer = np.zeros((new_size, MAX_TOKENS), dtype=np.int8)
            self._buffer_size = new_size

        # Return views into the pre-allocated buffers
        tokens = self._token_buffer[:n_worlds]
        masks = self._mask_buffer[:n_worlds]

        # Zero out the portion we're using (critical for correctness)
        tokens.fill(0)
        masks.fill(0)

        return tokens, masks

    def _tokenize_worlds(
        self,
        worlds: list[list[list[int]]],
        decl_id: int,
        leader: int,
        trick_plays: list[tuple[int, int]],
        remaining: np.ndarray,
        current_player: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize N worlds into model input format (vectorized).

        This is adapted from forge.ml.tokenize.process_shard but operates on
        in-memory world states instead of parquet data.

        Optimization: Vectorized implementation using numpy broadcasting to avoid
        triple-nested Python loops (N × 4 × 7 iterations).

        Args:
            worlds: List of N world states (each is 4 hands)
            decl_id: Declaration ID
            leader: Leader player ID
            trick_plays: List of (player, domino_id) for current trick (public ids)
            remaining: (N, 4) array of remaining domino bitmasks
            current_player: Current player ID

        Returns:
            Tuple of (tokens, masks):
                - tokens: (N, 32, 12) int32 array
                - masks: (N, 32) int8 array
        """
        n_worlds = len(worlds)

        # Normalize leader relative to current player
        normalized_leader = (leader - current_player + 4) % 4

        # Get pre-allocated buffers (reused across calls)
        tokens, masks = self._get_buffers(n_worlds)

        # =====================================================================
        # Step 1: Per-decl domino feature lookup (28 dominoes × 5 features)
        # =====================================================================
        domino_features = self._domino_features_by_decl[decl_id]

        # =====================================================================
        # Step 2: Convert worlds to numpy array for vectorized indexing
        # =====================================================================
        # Shape: (N, 4, 7) - each world has 4 players with 7 dominoes
        worlds_array = np.array(worlds, dtype=np.int32)
        # Flatten to (N, 28) for feature lookup
        flat_ids = worlds_array.reshape(n_worlds, 28)

        # =====================================================================
        # Step 3: Vectorized feature assignment for all hand tokens
        # =====================================================================
        # Get features for all dominoes in all worlds at once
        # Shape: (N, 28, 5)
        all_features = domino_features[flat_ids]
        # Assign to tokens[:, 1:29, 0:5]
        tokens[:, 1:29, 0:5] = all_features

        # =====================================================================
        # Step 4: Vectorize player normalization and flags
        # =====================================================================
        # Player indices: [0,0,0,0,0,0,0, 1,1,1,1,1,1,1, 2,2,2,2,2,2,2, 3,3,3,3,3,3,3]
        player_ids = self._player_ids  # Shape: (28,)
        normalized_players = (player_ids - current_player + 4) % 4  # Shape: (28,)

        # Broadcast to all worlds - same for every world
        tokens[:, 1:29, 5] = normalized_players  # normalized player
        tokens[:, 1:29, 6] = (normalized_players == 0).astype(np.int32)  # is_current
        tokens[:, 1:29, 7] = (normalized_players == 2).astype(np.int32)  # is_partner

        # =====================================================================
        # Step 5: Vectorize remaining bits
        # =====================================================================
        # For each of 28 hand positions, check if corresponding bit is set
        local_indices = self._local_indices
        player_for_token = self._player_for_token

        # remaining_bits[world, token] = (remaining[world, player] >> local_idx) & 1
        # Shape: (N, 28)
        remaining_bits = (remaining[:, player_for_token] >> local_indices) & 1
        tokens[:, 1:29, 8] = remaining_bits.astype(np.int32)

        # =====================================================================
        # Step 6: Vectorize token type and context (same for all worlds)
        # =====================================================================
        tokens[:, 1:29, 9] = self._token_types
        tokens[:, 1:29, 10] = decl_id
        tokens[:, 1:29, 11] = normalized_leader

        # Mark all hand tokens as present
        masks[:, 1:29] = 1

        # =====================================================================
        # Step 7: Context token (same for all worlds)
        # =====================================================================
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_id
        tokens[:, 0, 11] = normalized_leader
        masks[:, 0] = 1

        # =====================================================================
        # Step 8: Trick tokens (broadcast across all worlds - domino_id is public info)
        # =====================================================================
        # trick_plays is now [(player, domino_id), ...] - world-invariant public info
        # We can broadcast identical trick tokens to all N worlds
        for trick_pos, (play_player, domino_id) in enumerate(trick_plays):
            if trick_pos >= 3:  # Max 3 trick tokens
                break

            token_idx = 29 + trick_pos
            normalized_pp = (play_player - current_player + 4) % 4

            # Broadcast to ALL worlds - these are identical (public information)
            tokens[:, token_idx, 0:5] = domino_features[domino_id]
            tokens[:, token_idx, 5] = normalized_pp
            tokens[:, token_idx, 6] = 1 if normalized_pp == 0 else 0
            tokens[:, token_idx, 7] = 1 if normalized_pp == 2 else 0
            tokens[:, token_idx, 8] = 0  # Not in remaining (already played)
            tokens[:, token_idx, 9] = TOKEN_TYPE_TRICK_P0 + trick_pos
            tokens[:, token_idx, 10] = decl_id
            tokens[:, token_idx, 11] = normalized_leader

            masks[:, token_idx] = 1

        return tokens, masks

    def query_batch_multi_state(
        self,
        worlds: list[list[list[int]]],
        decl_ids: np.ndarray | int,
        actors: np.ndarray,
        leaders: np.ndarray,
        trick_plays_list: list[list[tuple[int, int]]],
        remaining: np.ndarray,
    ) -> Tensor:
        """Query N samples with DIFFERENT game states per sample.

        This is optimized for batched posterior scoring where each sample
        may have a different actor, leader, and trick state.

        Args:
            worlds: List of N world states (4 hands of 7 dominoes each)
            decl_ids: Declaration IDs - either int (same for all) or (N,) array (per-sample)
            actors: (N,) array of current player IDs
            leaders: (N,) array of leader player IDs
            trick_plays_list: List of N trick_plays, each is [(player, domino_id), ...]
            remaining: (N, 4) array of remaining domino bitmasks

        Returns:
            Tensor of shape (N, 7) with Q-values for each sample
        """
        if not worlds:
            raise ValueError("worlds cannot be empty")

        n_samples = len(worlds)

        # Convert scalar decl_id to array for uniform handling
        if isinstance(decl_ids, int):
            decl_ids_array = np.full(n_samples, decl_ids, dtype=np.int32)
        else:
            decl_ids_array = decl_ids.astype(np.int32)

        # Tokenize with per-sample states
        tokens, masks = self._tokenize_worlds_multi_state(
            worlds=worlds,
            decl_ids=decl_ids_array,
            actors=actors,
            leaders=leaders,
            trick_plays_list=trick_plays_list,
            remaining=remaining,
        )

        # Fast path: async mode with CUDA streams
        if self.use_async:
            return self._query_batch_multi_state_async(tokens, masks, actors, n_samples)

        # Fallback: synchronous mode
        # Move to device and run forward pass (keep tokens as int32).
        tokens = torch.from_numpy(tokens).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)
        actors_tensor = torch.from_numpy(actors).to(self.device, dtype=torch.long)

        with torch.inference_mode():
            q_values, _ = self.model(tokens, masks, actors_tensor)

        # Clone output to avoid CUDA graph tensor reuse issues with mode="reduce-overhead"
        return q_values.clone()

    def _query_batch_multi_state_async(
        self,
        tokens_np: np.ndarray,
        masks_np: np.ndarray,
        actors_np: np.ndarray,
        n_samples: int,
    ) -> Tensor:
        """Async variant of multi-state query using CUDA streams.

        Args:
            tokens_np: Tokenized input (N, MAX_TOKENS, N_FEATURES) numpy array
            masks_np: Attention masks (N, MAX_TOKENS) numpy array
            actors_np: Actor IDs (N,) numpy array
            n_samples: Number of samples

        Returns:
            Q-values tensor (N, 7) on GPU
        """
        # Get pinned host buffers
        pinned_tokens, pinned_masks = self._get_pinned_buffers(n_samples)

        # Copy numpy → pinned memory (CPU-only)
        pinned_tokens.copy_(torch.from_numpy(tokens_np))
        pinned_masks.copy_(torch.from_numpy(masks_np))

        # Actors need pinned memory too
        # Create pinned tensor directly from numpy (small array, ok to allocate)
        pinned_actors = torch.from_numpy(actors_np).pin_memory()

        # Get GPU buffer
        buf_idx = self._buffer_idx
        self._ensure_gpu_buffer_size(n_samples, buf_idx)
        buf = self._gpu_buffers[buf_idx]

        tokens_gpu = buf['tokens'][:n_samples]
        masks_gpu = buf['masks'][:n_samples]
        actors_gpu = buf['current_player'][:n_samples]  # Reuse current_player buffer for actors

        # Async H2D transfers
        with torch.cuda.stream(self.stream_h2d):
            tokens_gpu.copy_(pinned_tokens, non_blocking=True)
            masks_gpu.copy_(pinned_masks, non_blocking=True)
            actors_gpu.copy_(pinned_actors.to(dtype=torch.long), non_blocking=True)

        # Wait for transfers, then compute
        self.stream_compute.wait_stream(self.stream_h2d)

        with torch.cuda.stream(self.stream_compute):
            with torch.inference_mode():
                q_values, _ = self.model(tokens_gpu, masks_gpu, actors_gpu)

        # Swap buffers
        self._buffer_idx = 1 - self._buffer_idx

        return q_values.clone()

    def _tokenize_worlds_multi_state(
        self,
        worlds: list[list[list[int]]],
        decl_ids: np.ndarray,
        actors: np.ndarray,
        leaders: np.ndarray,
        trick_plays_list: list[list[tuple[int, int]]],
        remaining: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize N samples with per-sample game states.

        Unlike _tokenize_worlds which broadcasts same state to all samples,
        this handles different actors, leaders, tricks, and decl_ids per sample.

        Args:
            decl_ids: (N,) array of declaration IDs (per-sample)
        """
        n_samples = len(worlds)

        # Get pre-allocated buffers (reused across calls)
        tokens, masks = self._get_buffers(n_samples)

        # Stack all domino features: (10, 28, 5)
        all_domino_features = np.stack(self._domino_features_by_decl)

        # Convert worlds to numpy array
        worlds_array = np.array(worlds, dtype=np.int32)
        flat_ids = worlds_array.reshape(n_samples, 28)

        # Per-sample domino feature lookup: (N, 28, 5)
        # all_domino_features[decl_ids[:, None], flat_ids] uses advanced indexing
        # to select features for each sample's decl_id and domino IDs
        all_features = all_domino_features[decl_ids[:, None], flat_ids]
        tokens[:, 1:29, 0:5] = all_features

        # Per-sample player normalization
        # For each sample i, normalize player p as (p - actors[i] + 4) % 4
        player_ids = self._player_ids  # Shape: (28,)
        # Expand for broadcasting: (N, 28)
        actors_expanded = actors[:, np.newaxis]  # (N, 1)
        normalized_players = (player_ids - actors_expanded + 4) % 4  # (N, 28)

        tokens[:, 1:29, 5] = normalized_players.astype(np.int32)
        tokens[:, 1:29, 6] = (normalized_players == 0).astype(np.int32)  # is_current
        tokens[:, 1:29, 7] = (normalized_players == 2).astype(np.int32)  # is_partner

        # Remaining bits (already per-sample)
        local_indices = self._local_indices
        player_for_token = self._player_for_token
        remaining_bits = (remaining[:, player_for_token] >> local_indices) & 1
        tokens[:, 1:29, 8] = remaining_bits.astype(np.int32)

        # Token type (same for all)
        tokens[:, 1:29, 9] = self._token_types

        # Per-sample decl_id and normalized_leader
        tokens[:, 1:29, 10] = decl_ids[:, np.newaxis]  # Broadcast to 28 cols
        # normalized_leader per sample
        normalized_leaders = (leaders - actors + 4) % 4  # (N,)
        tokens[:, 1:29, 11] = normalized_leaders[:, np.newaxis]  # Broadcast to 28 cols

        masks[:, 1:29] = 1

        # Context token (per-sample leader and decl_id)
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_ids
        tokens[:, 0, 11] = normalized_leaders
        masks[:, 0] = 1

        # Trick tokens - vectorized by grouping samples with identical trick patterns
        # Key insight: Many samples share the same trick_plays (especially in batched posterior scoring)
        # We group by (trick_plays, actor, decl_id) pattern and broadcast to all matching samples.

        # Step 1: Build pattern keys and group samples
        # Pattern key: (trick_plays_tuple, actor, decl_id) - actor affects normalized_pp, decl_id affects features
        pattern_to_samples: dict[tuple, list[int]] = {}
        for i in range(n_samples):
            # Convert list to tuple for hashability
            trick_tuple = tuple(trick_plays_list[i])
            pattern_key = (trick_tuple, actors[i], decl_ids[i])
            if pattern_key not in pattern_to_samples:
                pattern_to_samples[pattern_key] = []
            pattern_to_samples[pattern_key].append(i)

        # Step 2: For each unique pattern, compute tokens once and broadcast to all samples
        for (trick_tuple, actor, decl_id), sample_indices in pattern_to_samples.items():
            trick_plays = list(trick_tuple)  # Convert back to list

            # Get domino features for this decl_id
            domino_features = self._domino_features_by_decl[decl_id]

            # Compute trick tokens for this pattern
            for trick_pos, (play_player, domino_id) in enumerate(trick_plays):
                if trick_pos >= 3:
                    break

                token_idx = 29 + trick_pos
                normalized_pp = (play_player - actor + 4) % 4

                # Build the token features once
                trick_token = np.zeros(N_FEATURES, dtype=np.int32)
                trick_token[0:5] = domino_features[domino_id]
                trick_token[5] = normalized_pp
                trick_token[6] = 1 if normalized_pp == 0 else 0
                trick_token[7] = 1 if normalized_pp == 2 else 0
                trick_token[8] = 0
                trick_token[9] = TOKEN_TYPE_TRICK_P0 + trick_pos
                trick_token[10] = decl_id
                # Note: normalized_leader differs per sample, will be set below

                # Broadcast to all samples with this pattern
                sample_indices_array = np.array(sample_indices, dtype=np.int32)
                tokens[sample_indices_array, token_idx, :11] = trick_token[:11]
                # Set per-sample normalized_leader
                tokens[sample_indices_array, token_idx, 11] = normalized_leaders[sample_indices_array]
                masks[sample_indices_array, token_idx] = 1

        return tokens, masks
