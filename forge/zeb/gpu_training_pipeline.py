"""Zero-copy GPU training pipeline for MCTS self-play.

Integrates GPUGameState and GPUMCTSForest to keep all data on GPU:
- Deal generation on GPU
- MCTS search with GPU tree operations
- Oracle evaluation with GPU tensors
- Training with GPU-native data

No CPU<->GPU transfers in the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from forge.zeb.cuda_only import require_cuda
from forge.zeb.gpu_game_state import (
    GPUGameState,
    apply_action_gpu,
    current_player_gpu,
    deal_random_gpu,
    get_domino_tables,
    is_terminal_gpu,
    legal_actions_gpu,
    DominoTables,
)
from forge.zeb.gpu_mcts import (
    GPUMCTSForest,
    backprop_gpu,
    create_forest,
    expand_gpu,
    get_leaf_states,
    get_root_policy_gpu,
    get_terminal_values,
    select_leaves_gpu,
)
from forge.zeb.gpu_game_state import current_player_gpu

if TYPE_CHECKING:
    from forge.zeb.oracle_value import OracleValueFunction
    from forge.zeb.model import ZebModel


def _index_state_batch(states: GPUGameState, batch_idx: Tensor) -> GPUGameState:
    """Index a GPUGameState along the batch dimension."""
    idx = batch_idx.long()
    return GPUGameState(
        hands=states.hands.index_select(0, idx),
        played_mask=states.played_mask.index_select(0, idx),
        play_history=states.play_history.index_select(0, idx),
        n_plays=states.n_plays.index_select(0, idx),
        current_trick=states.current_trick.index_select(0, idx),
        trick_len=states.trick_len.index_select(0, idx),
        leader=states.leader.index_select(0, idx),
        decl_id=states.decl_id.index_select(0, idx),
        scores=states.scores.index_select(0, idx),
    )


def _concat_state_batches(states_list: list[GPUGameState]) -> GPUGameState:
    """Concatenate multiple GPUGameState batches along the batch dimension."""
    return GPUGameState(
        hands=torch.cat([s.hands for s in states_list], dim=0),
        played_mask=torch.cat([s.played_mask for s in states_list], dim=0),
        play_history=torch.cat([s.play_history for s in states_list], dim=0),
        n_plays=torch.cat([s.n_plays for s in states_list], dim=0),
        current_trick=torch.cat([s.current_trick for s in states_list], dim=0),
        trick_len=torch.cat([s.trick_len for s in states_list], dim=0),
        leader=torch.cat([s.leader for s in states_list], dim=0),
        decl_id=torch.cat([s.decl_id for s in states_list], dim=0),
        scores=torch.cat([s.scores for s in states_list], dim=0),
    )


@dataclass
class GPUTrainingExample:
    """Training examples stored entirely on GPU.

    All tensors are on the same device (GPU).
    """
    observations: Tensor    # (N, MAX_TOKENS, N_FEATURES) int32
    masks: Tensor          # (N, MAX_TOKENS) bool
    hand_indices: Tensor   # (N, 7) int64
    hand_masks: Tensor     # (N, 7) bool - which slots have unplayed dominoes
    policy_targets: Tensor # (N, 7) float32 - visit distributions
    value_targets: Tensor  # (N,) float32 - game outcomes

    @property
    def device(self) -> torch.device:
        return self.observations.device

    @property
    def n_examples(self) -> int:
        return self.observations.shape[0]


class GPUReplayBuffer:
    """Fixed-capacity replay buffer stored entirely on GPU.

    Eliminates CPU<->GPU transfers during training by keeping all data on GPU.
    Uses circular buffer semantics for O(1) add operations.

    Memory usage for 50k examples: ~65 MB
    """

    MAX_TOKENS = 36
    N_FEATURES = 8
    N_HAND_SLOTS = 7

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.write_idx = 0

        # Pre-allocate GPU tensors
        self.observations = torch.zeros(
            (capacity, self.MAX_TOKENS, self.N_FEATURES),
            dtype=torch.int32, device=device
        )
        self.masks = torch.zeros(
            (capacity, self.MAX_TOKENS),
            dtype=torch.bool, device=device
        )
        self.hand_indices = torch.zeros(
            (capacity, self.N_HAND_SLOTS),
            dtype=torch.int64, device=device
        )
        self.hand_masks = torch.zeros(
            (capacity, self.N_HAND_SLOTS),
            dtype=torch.bool, device=device
        )
        self.policy_targets = torch.zeros(
            (capacity, self.N_HAND_SLOTS),
            dtype=torch.float32, device=device
        )
        self.value_targets = torch.zeros(
            capacity,
            dtype=torch.float32, device=device
        )

    def __len__(self) -> int:
        return self.size

    def add_batch(self, examples: GPUTrainingExample) -> None:
        """Add batch of examples (already on GPU). O(1) amortized."""
        n = examples.n_examples
        if n > self.capacity:
            # If batch larger than capacity, only keep last `capacity` examples
            examples = GPUTrainingExample(
                observations=examples.observations[-self.capacity:],
                masks=examples.masks[-self.capacity:],
                hand_indices=examples.hand_indices[-self.capacity:],
                hand_masks=examples.hand_masks[-self.capacity:],
                policy_targets=examples.policy_targets[-self.capacity:],
                value_targets=examples.value_targets[-self.capacity:],
            )
            n = self.capacity

        end_idx = self.write_idx + n

        if end_idx <= self.capacity:
            # Simple case: no wraparound
            self.observations[self.write_idx:end_idx] = examples.observations
            self.masks[self.write_idx:end_idx] = examples.masks
            self.hand_indices[self.write_idx:end_idx] = examples.hand_indices
            self.hand_masks[self.write_idx:end_idx] = examples.hand_masks
            self.policy_targets[self.write_idx:end_idx] = examples.policy_targets
            self.value_targets[self.write_idx:end_idx] = examples.value_targets
        else:
            # Wraparound: split into two copies
            first_part = self.capacity - self.write_idx
            second_part = n - first_part

            self.observations[self.write_idx:] = examples.observations[:first_part]
            self.observations[:second_part] = examples.observations[first_part:]

            self.masks[self.write_idx:] = examples.masks[:first_part]
            self.masks[:second_part] = examples.masks[first_part:]

            self.hand_indices[self.write_idx:] = examples.hand_indices[:first_part]
            self.hand_indices[:second_part] = examples.hand_indices[first_part:]

            self.hand_masks[self.write_idx:] = examples.hand_masks[:first_part]
            self.hand_masks[:second_part] = examples.hand_masks[first_part:]

            self.policy_targets[self.write_idx:] = examples.policy_targets[:first_part]
            self.policy_targets[:second_part] = examples.policy_targets[first_part:]

            self.value_targets[self.write_idx:] = examples.value_targets[:first_part]
            self.value_targets[:second_part] = examples.value_targets[first_part:]

        self.write_idx = end_idx % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> GPUTrainingExample:
        """Sample random batch entirely on GPU. No CPU involvement."""
        if batch_size > self.size:
            raise ValueError(f"batch_size {batch_size} > buffer size {self.size}")

        # Generate random indices on GPU
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return GPUTrainingExample(
            observations=self.observations[indices],
            masks=self.masks[indices],
            hand_indices=self.hand_indices[indices],
            hand_masks=self.hand_masks[indices],
            policy_targets=self.policy_targets[indices],
            value_targets=self.value_targets[indices],
        )

    def to_list(self) -> list[dict]:
        """Convert to CPU list of dicts for checkpoint serialization.

        Optimized: 6 bulk GPU→CPU transfers instead of 300k individual ones.
        """
        # Bulk transfer to CPU (6 operations instead of 50k × 6)
        obs_cpu = self.observations[:self.size].cpu()
        masks_cpu = self.masks[:self.size].cpu()
        hand_idx_cpu = self.hand_indices[:self.size].cpu()
        hand_masks_cpu = self.hand_masks[:self.size].cpu()
        policy_cpu = self.policy_targets[:self.size].cpu()
        value_cpu = self.value_targets[:self.size].cpu()

        # Build list from CPU tensors (no GPU involvement)
        result = []
        for i in range(self.size):
            result.append({
                'obs': obs_cpu[i],
                'mask': masks_cpu[i],
                'hand_idx': hand_idx_cpu[i],
                'hand_mask': hand_masks_cpu[i],
                'policy': policy_cpu[i],
                'value': value_cpu[i],
            })
        return result

    @classmethod
    def from_list(
        cls,
        data: list[dict],
        capacity: int,
        device: torch.device,
    ) -> "GPUReplayBuffer":
        """Restore from checkpoint list."""
        buffer = cls(capacity, device)
        if not data:
            return buffer

        # Stack all data and add as single batch
        n = len(data)
        examples = GPUTrainingExample(
            observations=torch.stack([d['obs'] for d in data]).to(device),
            masks=torch.stack([d['mask'] for d in data]).to(device),
            hand_indices=torch.stack([d['hand_idx'] for d in data]).to(device),
            hand_masks=torch.stack([d['hand_mask'] for d in data]).to(device),
            policy_targets=torch.stack([d['policy'] for d in data]).to(device),
            value_targets=torch.stack([d['value'] for d in data]).to(device),
        )
        buffer.add_batch(examples)
        return buffer


class GPUObservationTokenizer:
    """GPU-native observation tokenizer for Zeb training.

    Converts GPUGameState to observation tokens entirely on GPU.
    Matches the format expected by ZebModel (36 tokens x 8 features).
    """

    # Feature indices (match observation.py)
    FEAT_HIGH = 0
    FEAT_LOW = 1
    FEAT_IS_DOUBLE = 2
    FEAT_COUNT = 3
    FEAT_PLAYER = 4
    FEAT_IS_IN_HAND = 5
    FEAT_DECL = 6
    FEAT_TOKEN_TYPE = 7

    # Token types
    TOKEN_TYPE_DECL = 0
    TOKEN_TYPE_HAND = 1
    TOKEN_TYPE_PLAY = 2

    # Layout constants
    N_FEATURES = 8
    MAX_TOKENS = 36  # 1 decl + 7 hand + 28 plays
    N_HAND_SLOTS = 7

    def __init__(self, device: torch.device):
        device = require_cuda(device, where="GPUObservationTokenizer.__init__")
        self.device = device
        self.tables = get_domino_tables(device)

        # Build count value lookup table
        # Points: 0->0, 5->1, 10->2
        self.count_values = self._build_count_values(device)
        self._hand_indices_1x7 = torch.arange(1, 8, dtype=torch.int64, device=device).unsqueeze(0)

    def _build_count_values(self, device: torch.device) -> Tensor:
        """Build count value lookup (domino_id -> count category)."""
        count_vals = []
        for high in range(7):
            for low in range(high + 1):
                if (high, low) in ((5, 5), (6, 4)):
                    count_vals.append(2)  # 10 points
                elif (high, low) in ((5, 0), (4, 1), (3, 2)):
                    count_vals.append(1)  # 5 points
                else:
                    count_vals.append(0)  # 0 points
        return torch.tensor(count_vals, dtype=torch.int32, device=device)

    def tokenize_batch_into(
        self,
        states: GPUGameState,
        original_hands: Tensor,  # (N, 4, 7)
        perspective_players: Tensor,  # (N,)
        *,
        out_tokens: Tensor,      # (N, 36, 8) int32
        out_masks: Tensor,       # (N, 36) bool
        out_hand_masks: Tensor,  # (N, 7) bool
        batch_idx: Tensor | None = None,  # (N,) int64
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Allocation-free tokenizer variant (for CUDA graph capture)."""
        n = states.batch_size
        device = states.device

        if out_tokens.shape != (n, self.MAX_TOKENS, self.N_FEATURES):
            raise ValueError(f"out_tokens must have shape {(n, self.MAX_TOKENS, self.N_FEATURES)}")
        if out_masks.shape != (n, self.MAX_TOKENS):
            raise ValueError(f"out_masks must have shape {(n, self.MAX_TOKENS)}")
        if out_hand_masks.shape != (n, 7):
            raise ValueError("out_hand_masks must have shape (N, 7)")

        if batch_idx is None:
            batch_idx = torch.arange(n, device=device, dtype=torch.int64)

        out_tokens.zero_()
        out_masks.zero_()

        players_i64 = perspective_players.to(torch.int64)

        # === Token 0: Declaration ===
        out_tokens[:, 0, self.FEAT_DECL] = states.decl_id
        out_tokens[:, 0, self.FEAT_TOKEN_TYPE] = self.TOKEN_TYPE_DECL
        out_masks[:, 0] = True

        # === Tokens 1-7: Hand slots ===
        my_hand = original_hands[batch_idx, players_i64]  # (N, 7)
        played_lookup = states.played_mask[batch_idx.unsqueeze(1), my_hand.long()]  # (N, 7)
        in_hand = ~played_lookup  # (N, 7) bool

        safe_ids = my_hand.clamp(min=0).long()
        out_tokens[:, 1:8, self.FEAT_HIGH] = self.tables.high[safe_ids]
        out_tokens[:, 1:8, self.FEAT_LOW] = self.tables.low[safe_ids]
        out_tokens[:, 1:8, self.FEAT_IS_DOUBLE] = self.tables.is_double[safe_ids].int()
        out_tokens[:, 1:8, self.FEAT_COUNT] = self.count_values[safe_ids]
        out_tokens[:, 1:8, self.FEAT_PLAYER] = 0
        out_tokens[:, 1:8, self.FEAT_IS_IN_HAND] = in_hand.int()
        out_tokens[:, 1:8, self.FEAT_DECL] = states.decl_id.unsqueeze(1).expand(-1, 7)
        out_tokens[:, 1:8, self.FEAT_TOKEN_TYPE] = self.TOKEN_TYPE_HAND
        out_masks[:, 1:8] = in_hand

        # === Tokens 8-35: Play history ===
        n_plays = states.n_plays  # (N,)
        for play_idx in range(28):
            token_pos = 8 + play_idx
            has_play = n_plays > play_idx

            play_player = states.play_history[:, play_idx, 0]
            play_domino = states.play_history[:, play_idx, 1]
            rel_player = (play_player - perspective_players + 4) % 4

            safe_domino = play_domino.clamp(min=0).long()
            out_tokens[:, token_pos, self.FEAT_HIGH] = torch.where(has_play, self.tables.high[safe_domino], 0)
            out_tokens[:, token_pos, self.FEAT_LOW] = torch.where(has_play, self.tables.low[safe_domino], 0)
            out_tokens[:, token_pos, self.FEAT_IS_DOUBLE] = torch.where(
                has_play, self.tables.is_double[safe_domino].int(), 0
            )
            out_tokens[:, token_pos, self.FEAT_COUNT] = torch.where(has_play, self.count_values[safe_domino], 0)
            out_tokens[:, token_pos, self.FEAT_PLAYER] = torch.where(has_play, rel_player, 0)
            out_tokens[:, token_pos, self.FEAT_IS_IN_HAND] = 0
            out_tokens[:, token_pos, self.FEAT_DECL] = torch.where(has_play, states.decl_id, 0)
            out_tokens[:, token_pos, self.FEAT_TOKEN_TYPE] = torch.where(has_play, self.TOKEN_TYPE_PLAY, 0)
            out_masks[:, token_pos] = has_play

        out_hand_masks.copy_(in_hand)
        hand_indices = self._hand_indices_1x7.expand(n, -1)
        return out_tokens, out_masks, hand_indices, out_hand_masks

    def tokenize_batch(
        self,
        states: GPUGameState,
        original_hands: Tensor,  # (N, 4, 7) - original deal
        perspective_players: Tensor,  # (N,) - which player's view
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Tokenize batch of game states.

        Args:
            states: GPUGameState with N games
            original_hands: Original 7-card hands (N, 4, 7)
            perspective_players: Player perspective for each game (N,)

        Returns:
            tokens: (N, 36, 8) int32 observation tokens
            masks: (N, 36) bool valid token mask
            hand_indices: (N, 7) int64 indices of hand slots
            hand_masks: (N, 7) bool which slots have unplayed dominoes
        """
        n = states.batch_size
        device = states.device

        tokens = torch.empty((n, self.MAX_TOKENS, self.N_FEATURES), dtype=torch.int32, device=device)
        masks = torch.empty((n, self.MAX_TOKENS), dtype=torch.bool, device=device)
        hand_masks = torch.empty((n, 7), dtype=torch.bool, device=device)
        return self.tokenize_batch_into(
            states,
            original_hands,
            perspective_players,
            out_tokens=tokens,
            out_masks=masks,
            out_hand_masks=hand_masks,
        )


class GPUTrainingPipeline:
    """Zero-copy training pipeline - all data stays on GPU.

    Generates training examples through MCTS self-play entirely on GPU:
    - Deals random games on GPU
    - Runs MCTS with GPU tree operations
    - Evaluates leaves with oracle OR model's value head (self-play mode)
    - Collects training examples without CPU transfer
    - Trains model on GPU-native tensors

    Two modes:
    - Oracle mode: Use perfect oracle for leaf evaluation (distillation)
    - Self-play mode: Use model's value head for leaf evaluation (true AlphaZero)
    """

    def __init__(
        self,
        oracle: "OracleValueFunction | None",
        device: torch.device,
        n_parallel_games: int = 16,
        n_simulations: int = 100,
        wave_size: int = 1,
        max_mcts_nodes: int = 1024,
        c_puct: float = 1.414,
        temperature: float = 1.0,
        model: "ZebModel | None" = None,
        use_cudagraph_mcts: bool = True,
        use_fullstep_eval: bool = True,
    ):
        """Initialize pipeline.

        Args:
            oracle: OracleValueFunction for leaf evaluation (oracle mode).
                    Pass None for self-play mode (requires model).
            device: GPU device
            n_parallel_games: Number of concurrent games per batch
            n_simulations: MCTS simulations per move
            wave_size: Number of leaf batches to collect before evaluation.
                       wave_size=1 matches sequential MCTS (evaluate each sim).
            max_mcts_nodes: Maximum nodes per MCTS tree
            c_puct: UCB exploration constant
            temperature: Action sampling temperature (1.0 = proportional to visits)
            model: ZebModel for self-play mode leaf evaluation.
                   Pass None for oracle mode (requires oracle).

        Exactly one of oracle/model must be provided.
        """
        device = require_cuda(device, where="GPUTrainingPipeline.__init__")
        if oracle is None and model is None:
            raise ValueError("Must provide either oracle or model")
        if oracle is not None and model is not None:
            raise ValueError("Cannot provide both oracle and model - choose one mode")

        self.oracle = oracle
        self.model = model
        self.self_play_mode = model is not None
        self.device = device
        self.n_parallel_games = n_parallel_games
        self.n_simulations = n_simulations
        self.wave_size = max(1, int(wave_size))
        self.max_mcts_nodes = max_mcts_nodes
        self.c_puct = c_puct
        self.temperature = temperature
        self.use_cudagraph_mcts = bool(use_cudagraph_mcts) and device.type == "cuda"
        self.use_fullstep_eval = bool(use_fullstep_eval)

        # Initialize tokenizer
        self.tokenizer = GPUObservationTokenizer(device)
        self._oracle_position_idx = torch.arange(3, device=device, dtype=torch.int64).unsqueeze(0)  # (1, 3)

        # Stats for monitoring
        self.total_games_generated = 0
        self.total_oracle_queries = 0
        self.total_model_queries = 0  # For self-play mode

    def generate_games_gpu(
        self,
        n_games: int,
        *,
        max_moves: int = 28,
    ) -> GPUTrainingExample:
        """Generate training examples entirely on GPU.

        Plays n_games of self-play using MCTS, collecting training examples
        at each move. All computation stays on GPU.

        Args:
            n_games: Number of games to generate

        Returns:
            GPUTrainingExample with all training data on GPU
        """
        # No gradients needed during game generation - only during training
        with torch.no_grad():
            return self._generate_games_gpu_impl(n_games, max_moves=max_moves)

    def _generate_games_gpu_impl(self, n_games: int, *, max_moves: int) -> GPUTrainingExample:
        """Implementation of generate_games_gpu (called within no_grad context)."""
        all_obs = []
        all_masks = []
        all_hand_indices = []
        all_hand_masks = []
        all_policies = []
        all_values = []
        all_players = []  # Track which player for outcome assignment

        games_completed = 0
        batch_idx = 0

        while games_completed < n_games:
            # How many games to run this batch
            batch_games = min(self.n_parallel_games, n_games - games_completed)

            # Deal random games
            deals = deal_random_gpu(batch_games, self.device)
            original_hands = deals.hands.clone()  # Save for observation encoding

            # Play games to completion
            game_examples = self._play_games_mcts(deals, original_hands, max_moves=max_moves)

            all_obs.append(game_examples['observations'])
            all_masks.append(game_examples['masks'])
            all_hand_indices.append(game_examples['hand_indices'])
            all_hand_masks.append(game_examples['hand_masks'])
            all_policies.append(game_examples['policies'])
            all_values.append(game_examples['values'])

            games_completed += batch_games
            batch_idx += 1
            self.total_games_generated += batch_games

        # Concatenate all examples
        return GPUTrainingExample(
            observations=torch.cat(all_obs, dim=0),
            masks=torch.cat(all_masks, dim=0),
            hand_indices=torch.cat(all_hand_indices, dim=0),
            hand_masks=torch.cat(all_hand_masks, dim=0),
            policy_targets=torch.cat(all_policies, dim=0),
            value_targets=torch.cat(all_values, dim=0),
        )

    def _play_games_mcts(
        self,
        initial_states: GPUGameState,
        original_hands: Tensor,
        *,
        max_moves: int = 28,
    ) -> dict[str, Tensor]:
        """Play games using MCTS (up to max_moves).

        Args:
            initial_states: Starting game states
            original_hands: Original hands for observation encoding (N, 4, 7)

        Returns:
            Dict with training tensors
        """
        n = initial_states.batch_size
        device = initial_states.device

        # Track examples per game: list of (obs, mask, hand_idx, hand_mask, policy, player)
        # Structure: [move_0_data, move_1_data, ...]
        # Each move_data is a dict with tensors for all n games
        move_examples = []

        # Track which game index each example belongs to
        game_indices_per_move = []

        # Current states (will be modified as games progress)
        states = initial_states

        # Reuse a single forest across all moves for this batch (enables CUDA graphs).
        forest = create_forest(
            n_trees=n,
            max_nodes=self.max_mcts_nodes,
            initial_states=states,
            device=device,
            c_puct=self.c_puct,
            original_hands=original_hands,
        )

        # Graph pool shared across all MCTS CUDA graphs for this forest (reduces allocator pressure
        # and keeps memory stable when capturing multiple depth variants).
        graph_pool = getattr(forest, "_mcts_graph_pool", None)
        if graph_pool is None and self.use_cudagraph_mcts:
            graph_pool = torch.cuda.graphs.graph_pool_handle()
            setattr(forest, "_mcts_graph_pool", graph_pool)

        for move_idx in range(int(max_moves)):
            # Check which games are still active (not terminal)
            terminal = is_terminal_gpu(states)
            active = ~terminal

            if not active.any():
                break

            # Get active game indices
            active_idx = active.nonzero(as_tuple=True)[0]
            game_indices_per_move.append(active_idx)

            # Get current player for active games
            current_players = current_player_gpu(states)  # (N,)

            # Record pre-move observations for active games only
            obs, masks, hand_idx, hand_masks = self.tokenizer.tokenize_batch(
                states, original_hands, current_players
            )

            # Reset forest in-place for this move (keep allocations stable).
            from forge.zeb.gpu_mcts import reset_forest_inplace
            reset_forest_inplace(forest, states, original_hands=original_hands)

            # Depth cap: only traverse as deep as there are remaining moves.
            # This is purely a runtime optimization: deeper traversal would become inactive anyway.
            remaining_depth = int(max_moves) - int(move_idx)
            if remaining_depth < 1:
                remaining_depth = 1

            if self.wave_size <= 1:
                if self.use_cudagraph_mcts:
                    if self.self_play_mode and self.use_fullstep_eval:
                        # Phase 5: single-graph sim step including self-play model eval.
                        from forge.zeb.gpu_mcts import MCTSSelfPlayFullStepCUDAGraphRunner

                        runner: MCTSSelfPlayFullStepCUDAGraphRunner | None = getattr(
                            forest, "_mcts_selfplay_fullstep_runner", None
                        )
                        if runner is None or getattr(runner, "model", None) is not self.model:
                            runner = MCTSSelfPlayFullStepCUDAGraphRunner(
                                forest,
                                model=self.model,
                                tokenizer=self.tokenizer,
                                pool=graph_pool,
                            )
                            setattr(forest, "_mcts_selfplay_fullstep_runner", runner)

                        if not runner.captured:
                            runner.capture()
                            reset_forest_inplace(forest, states, original_hands=original_hands)

                        for _ in range(self.n_simulations):
                            runner.step()
                        self.total_model_queries += n * self.n_simulations
                    else:
                        # Prefer Phase 4 full-step capture only when oracle is compatible.
                        oracle_impl = getattr(self.oracle, "oracle", None)
                        oracle_capturable = (
                            (not self.self_play_mode)
                            and oracle_impl is not None
                            and hasattr(oracle_impl, "gpu_tokenizer")
                            and self.use_fullstep_eval
                        )

                        if oracle_capturable:
                            # Phase 4: single-graph sim step including oracle eval (fixed-shape).
                            from forge.zeb.gpu_mcts import MCTSFullStepCUDAGraphRunner

                            capture_depth_max = max(28, int(max_moves))
                            depth_variants = [capture_depth_max, 16, 8, 4, 2, 1]
                            depth_variants = sorted(
                                {d for d in depth_variants if d >= 1 and d <= capture_depth_max}, reverse=True
                            )

                            runners: dict[int, MCTSFullStepCUDAGraphRunner] | None = getattr(
                                forest, "_mcts_fullstep_runners", None
                            )
                            if runners is None:
                                runners = {}
                                setattr(forest, "_mcts_fullstep_runners", runners)

                            if not getattr(forest, "_mcts_fullstep_runners_captured", False):
                                for d in depth_variants:
                                    if d not in runners:
                                        runners[d] = MCTSFullStepCUDAGraphRunner(
                                            forest, self.oracle, max_depth=d, pool=graph_pool
                                        )
                                for d in depth_variants:
                                    runner_d = runners[d]
                                    if not runner_d.captured:
                                        runner_d.capture()
                                        reset_forest_inplace(forest, states, original_hands=original_hands)
                                setattr(forest, "_mcts_fullstep_runners_captured", True)

                            # Pick the smallest captured depth that still covers the remaining moves.
                            runner_depth = capture_depth_max
                            for d in reversed(depth_variants):
                                if d >= remaining_depth:
                                    runner_depth = d
                                    break
                            runner = runners[runner_depth]
                            for _ in range(self.n_simulations):
                                runner.step()
                        else:
                            # Phase 3 runner: model/oracle eval outside graphs (works with mocks too).
                            from forge.zeb.gpu_mcts import MCTSCUDAGraphRunner

                            capture_depth_max = max(28, int(max_moves))
                            depth_variants = [capture_depth_max, 16, 8, 4, 2, 1]
                            depth_variants = sorted(
                                {d for d in depth_variants if d >= 1 and d <= capture_depth_max}, reverse=True
                            )

                            runners: dict[int, MCTSCUDAGraphRunner] | None = getattr(
                                forest, "_mcts_cudagraph_runners", None
                            )
                            if runners is None:
                                runners = {}
                                setattr(forest, "_mcts_cudagraph_runners", runners)

                            if not getattr(forest, "_mcts_cudagraph_runners_captured", False):
                                for d in depth_variants:
                                    if d not in runners:
                                        runners[d] = MCTSCUDAGraphRunner(
                                            forest, max_depth=d, pool=graph_pool
                                        )
                                for d in depth_variants:
                                    runner_d = runners[d]
                                    if not runner_d._captured:
                                        runner_d.capture()
                                        reset_forest_inplace(forest, states, original_hands=original_hands)
                                setattr(forest, "_mcts_cudagraph_runners_captured", True)

                            def _oracle(leaf_states: GPUGameState, leaf_indices: Tensor, tree_indices: Tensor) -> Tensor:
                                return self._evaluate_leaves_gpu(
                                    forest,
                                    leaf_states,
                                    leaf_indices,
                                    tree_indices=tree_indices,
                                )

                            runner_depth = capture_depth_max
                            for d in reversed(depth_variants):
                                if d >= remaining_depth:
                                    runner_depth = d
                                    break
                            runner = runners[runner_depth]
                            for _ in range(self.n_simulations):
                                runner.step(_oracle)
                else:
                    # Sequential MCTS: evaluate every simulation (but avoid tensor.any() sync).
                    for _ in range(self.n_simulations):
                        leaf_indices, paths = select_leaves_gpu(forest, max_depth=remaining_depth)
                        leaf_states = get_leaf_states(forest, leaf_indices)
                        terminal_values, is_terminal = get_terminal_values(
                            forest, leaf_indices, leaf_states
                        )

                        leaf_values = torch.zeros(n, dtype=torch.float32, device=device)
                        nonterm_idx = (~is_terminal).nonzero(as_tuple=True)[0]
                        if nonterm_idx.numel() > 0:
                            leaf_values_nonterm = self._evaluate_leaves_gpu(
                                forest,
                                _index_state_batch(leaf_states, nonterm_idx),
                                leaf_indices[nonterm_idx],
                                tree_indices=nonterm_idx,
                            )
                            leaf_values[nonterm_idx] = leaf_values_nonterm

                        values = torch.where(is_terminal, terminal_values, leaf_values)
                        expand_gpu(forest, leaf_indices, leaf_states=leaf_states)
                        backprop_gpu(forest, leaf_indices, values, paths)
            else:
                # Wave batching: collect W selections, evaluate a single (W*N) batch,
                # then backprop W times.
                sims_remaining = self.n_simulations
                while sims_remaining > 0:
                    wave = min(self.wave_size, sims_remaining)

                    leaf_indices_steps: list[Tensor] = []
                    paths_steps: list[Tensor] = []
                    leaf_states_steps: list[GPUGameState] = []
                    terminal_values_steps: list[Tensor] = []
                    is_terminal_steps: list[Tensor] = []

                    for _ in range(wave):
                        leaf_indices, paths = select_leaves_gpu(forest, max_depth=remaining_depth)
                        leaf_states = get_leaf_states(forest, leaf_indices)
                        terminal_values, is_terminal = get_terminal_values(
                            forest, leaf_indices, leaf_states
                        )

                        leaf_indices_steps.append(leaf_indices)
                        paths_steps.append(paths)
                        leaf_states_steps.append(leaf_states)
                        terminal_values_steps.append(terminal_values)
                        is_terminal_steps.append(is_terminal)

                    # Flatten (wave, N) -> (wave*N)
                    flat_leaf_states = _concat_state_batches(leaf_states_steps)
                    flat_leaf_indices = torch.stack(leaf_indices_steps, dim=0).reshape(-1)
                    flat_is_terminal = torch.stack(is_terminal_steps, dim=0).reshape(-1)

                    tree_idx = torch.arange(n, device=device, dtype=torch.int64).repeat(wave)  # (wave*N,)

                    # Evaluate only non-terminals in one big batch.
                    flat_leaf_values = torch.zeros(wave * n, dtype=torch.float32, device=device)
                    nonterm_flat_idx = (~flat_is_terminal).nonzero(as_tuple=True)[0]
                    if nonterm_flat_idx.numel() > 0:
                        leaf_values_nonterm = self._evaluate_leaves_gpu(
                            forest,
                            _index_state_batch(flat_leaf_states, nonterm_flat_idx),
                            flat_leaf_indices[nonterm_flat_idx],
                            tree_indices=tree_idx[nonterm_flat_idx],
                        )
                        flat_leaf_values[nonterm_flat_idx] = leaf_values_nonterm

                    # Backprop each step in order (keeping semantics close to sequential MCTS).
                    for step in range(wave):
                        start = step * n
                        end = (step + 1) * n

                        leaf_indices = leaf_indices_steps[step]
                        paths = paths_steps[step]
                        terminal_values = terminal_values_steps[step]
                        is_terminal = is_terminal_steps[step]

                        leaf_values = flat_leaf_values[start:end]
                        values = torch.where(is_terminal, terminal_values, leaf_values)

                        expand_gpu(forest, leaf_indices)
                        backprop_gpu(forest, leaf_indices, values, paths)

                    sims_remaining -= wave

            # Get visit distributions as policies
            policies = get_root_policy_gpu(forest)  # (N, 7)

            # Store examples for active games only
            move_examples.append({
                'obs': obs[active_idx],
                'masks': masks[active_idx],
                'hand_idx': hand_idx[active_idx],
                'hand_masks': hand_masks[active_idx],
                'policies': policies[active_idx],
                'players': current_players[active_idx],
            })

            # Sample actions from policy
            actions = self._sample_actions_gpu(policies, hand_masks)

            # Apply actions to advance games
            states = apply_action_gpu(states, actions)

        # Compute final outcomes from final scores
        # Team with more points wins: +1, loses: -1
        team0_score = states.scores[:, 0]
        team1_score = states.scores[:, 1]

        # For each game, compute outcome from team 0's perspective
        # +1 if team 0 wins, -1 if team 0 loses, 0 if tie
        game_outcomes_team0 = torch.zeros(n, dtype=torch.float32, device=device)
        game_outcomes_team0 = torch.where(
            team0_score > team1_score,
            torch.ones_like(game_outcomes_team0),
            game_outcomes_team0
        )
        game_outcomes_team0 = torch.where(
            team0_score < team1_score,
            -torch.ones_like(game_outcomes_team0),
            game_outcomes_team0
        )

        # Collate all examples and assign values
        all_obs = []
        all_masks = []
        all_hand_idx = []
        all_hand_masks = []
        all_policies = []
        all_values = []

        for move_idx, (game_indices, examples) in enumerate(
            zip(game_indices_per_move, move_examples)
        ):
            all_obs.append(examples['obs'])
            all_masks.append(examples['masks'])
            all_hand_idx.append(examples['hand_idx'])
            all_hand_masks.append(examples['hand_masks'])
            all_policies.append(examples['policies'])

            # Compute values for each example
            players = examples['players']  # (n_active,)
            player_teams = players % 2  # 0 or 1

            # Get game outcomes for these examples
            example_outcomes = game_outcomes_team0[game_indices]  # From team 0's view

            # Flip sign for team 1 players
            values = torch.where(
                player_teams == 0,
                example_outcomes,
                -example_outcomes
            )
            all_values.append(values)

        return {
            'observations': torch.cat(all_obs, dim=0),
            'masks': torch.cat(all_masks, dim=0),
            'hand_indices': torch.cat(all_hand_idx, dim=0),
            'hand_masks': torch.cat(all_hand_masks, dim=0),
            'policies': torch.cat(all_policies, dim=0),
            'values': torch.cat(all_values, dim=0),
        }

    def set_model(self, model: "ZebModel") -> None:
        """Update model reference for self-play.

        Call this to use a newly trained model for game generation.
        Only valid in self-play mode.
        """
        if not self.self_play_mode:
            raise ValueError("set_model only valid in self-play mode")
        self.model = model

    def _evaluate_leaves_gpu(
        self,
        forest: GPUMCTSForest,
        leaf_states: GPUGameState,
        leaf_indices: Tensor,
        tree_indices: Tensor | None = None,
    ) -> Tensor:
        """Evaluate leaf states - dispatches to oracle or model.

        Args:
            forest: MCTS forest (provides original_hands)
            leaf_states: States to evaluate
            leaf_indices: (N,) node indices of leaves (needed for storing priors)

        Returns:
            values: (N,) float32 values from root player's perspective
        """
        if self.self_play_mode:
            return self._evaluate_model_gpu(forest, leaf_states, leaf_indices, tree_indices=tree_indices)
        else:
            return self._evaluate_oracle_gpu(forest, leaf_states, tree_indices=tree_indices)

    def _evaluate_oracle_gpu(
        self,
        forest: GPUMCTSForest,
        leaf_states: GPUGameState,
        tree_indices: Tensor | None = None,
    ) -> Tensor:
        """Evaluate leaf states with oracle - all on GPU."""
        device = leaf_states.device
        n = leaf_states.batch_size

        if tree_indices is None:
            original_hands = forest.original_hands
            root_players = forest.to_play[:, 0]
        else:
            idx = tree_indices.to(device=device, dtype=torch.int64)
            original_hands = forest.original_hands.index_select(0, idx)
            root_players = forest.to_play[:, 0].index_select(0, idx)

        actors = current_player_gpu(leaf_states)

        # Extract trick plays from current_trick tensor: (N, 4, 2) -> (N, 3)
        trick_players = leaf_states.current_trick[:, :3, 0]
        trick_dominoes = leaf_states.current_trick[:, :3, 1]

        # Mark empty trick positions as -1 based on trick_len
        trick_len = leaf_states.trick_len.unsqueeze(1)  # (N, 1)
        empty_mask = self._oracle_position_idx >= trick_len  # (N, 3)
        trick_players = torch.where(empty_mask, torch.full_like(trick_players, -1), trick_players)
        trick_dominoes = torch.where(empty_mask, torch.full_like(trick_dominoes, -1), trick_dominoes)

        values = self.oracle.batch_evaluate_gpu(
            original_hands=original_hands,
            current_hands=leaf_states.hands,
            decl_ids=leaf_states.decl_id,
            actors=actors,
            leaders=leaf_states.leader,
            trick_players=trick_players,
            trick_dominoes=trick_dominoes,
            players=root_players,
        )

        self.total_oracle_queries += n
        return values

    def _evaluate_model_gpu(
        self,
        forest: GPUMCTSForest,
        leaf_states: GPUGameState,
        leaf_indices: Tensor,
        tree_indices: Tensor | None = None,
    ) -> Tensor:
        """Evaluate leaf states with model's value AND policy heads (self-play mode).

        This is the key AlphaZero component: both policy and value come from the
        neural network. The policy becomes the prior for MCTS, guiding which
        actions to explore.

        Args:
            forest: MCTS forest (provides original_hands for tokenization)
            leaf_states: States to evaluate
            leaf_indices: (N,) node indices of leaves being evaluated

        Returns:
            values: (N,) float32 values from root player's perspective

        Side effect:
            Updates forest.priors for the leaf nodes with policy probabilities
        """
        n = leaf_states.batch_size
        device = forest.device

        # Get current player at each leaf (for observation perspective)
        current_players = current_player_gpu(leaf_states)

        # Tokenize states from current player's perspective
        if tree_indices is None:
            original_hands = forest.original_hands
            root_players = forest.to_play[:, 0]
            batch_tree_indices = None
        else:
            idx = tree_indices.to(device=device, dtype=torch.int64)
            original_hands = forest.original_hands.index_select(0, idx)
            root_players = forest.to_play[:, 0].index_select(0, idx)
            batch_tree_indices = idx

        tokens, masks, hand_idx, hand_masks = self.tokenizer.tokenize_batch(
            leaf_states,
            original_hands,
            current_players,
        )

        # Model forward pass (no gradients during game generation)
        self.model.eval()
        with torch.no_grad():
            policy_logits, values = self.model(tokens.long(), masks, hand_idx, hand_masks)

        # Convert policy logits to probabilities, masking illegal actions
        policy_logits = policy_logits.masked_fill(~hand_masks, float('-inf'))
        # Avoid GPU->CPU sync in Python control flow (and avoid NaNs from softmax(-inf)).
        all_masked = hand_masks.sum(dim=-1) == 0  # (N,)
        safe_logits = torch.where(all_masked.unsqueeze(1), torch.zeros_like(policy_logits), policy_logits)
        policy_probs = F.softmax(safe_logits, dim=-1)
        # If all actions are masked, softmax(zeros) already yields uniform.

        # Store policy as priors for these leaf nodes
        # This is the key AlphaZero insight: policy guides MCTS exploration
        from forge.zeb.gpu_mcts import set_node_priors
        set_node_priors(forest, leaf_indices, policy_probs, tree_indices=batch_tree_indices)

        # Values are from current player's perspective
        # Need to convert to root player's perspective for MCTS backprop
        # Check if same team as root
        same_team = (current_players % 2) == (root_players % 2)

        # Flip value for opponent team
        values = torch.where(same_team, values, -values)

        self.total_model_queries += n

        return values

    def _sample_actions_gpu(
        self,
        policies: Tensor,  # (N, 7) visit probabilities
        hand_masks: Tensor,  # (N, 7) which slots are valid
    ) -> Tensor:
        """Sample actions from MCTS policies on GPU.

        Args:
            policies: Visit distributions from MCTS
            hand_masks: Which slots contain unplayed dominoes

        Returns:
            actions: (N,) int32 sampled slot indices
        """
        # Apply temperature
        if self.temperature != 1.0:
            policies = policies.pow(1.0 / self.temperature)

        # Mask and renormalize
        masked = policies * hand_masks.float()
        total = masked.sum(dim=1, keepdim=True)
        policies = masked / total.clamp(min=1e-8)

        # Handle degenerate cases:
        # - total==0 (e.g., no child visits yet) -> uniform over hand_masks
        # - hand_masks sums to 0 (shouldn't happen) -> uniform over all 7
        zero_policy = total.squeeze(1) <= 0
        hand_counts = hand_masks.sum(dim=1, keepdim=True)
        uniform = hand_masks.float() / hand_counts.clamp(min=1)
        uniform = torch.where(
            (hand_counts == 0),
            torch.full_like(uniform, 1.0 / 7.0),
            uniform,
        )
        policies = torch.where(zero_policy.unsqueeze(1), uniform, policies)

        # Sample
        actions = torch.multinomial(policies, 1).squeeze(1)

        return actions.int()

    def train_batch_gpu(
        self,
        model: ZebModel,
        optimizer: torch.optim.Optimizer,
        observations: Tensor,
        masks: Tensor,
        hand_indices: Tensor,
        hand_masks: Tensor,
        policy_targets: Tensor,
        value_targets: Tensor,
    ) -> dict[str, float]:
        """Train on GPU tensors directly.

        No DataLoader needed - data is already on GPU in optimal format.

        Args:
            model: ZebModel to train
            optimizer: Optimizer
            observations: (batch, 36, 8) int32 tokens
            masks: (batch, 36) bool attention masks
            hand_indices: (batch, 7) int64 hand slot indices
            hand_masks: (batch, 7) bool legal action masks
            policy_targets: (batch, 7) float32 MCTS visit distributions
            value_targets: (batch,) float32 game outcomes

        Returns:
            Dict with policy_loss and value_loss
        """
        model.train()

        # Forward pass
        policy_logits, value = model(
            observations.long(),  # ZebModel expects long
            masks,
            hand_indices,
            hand_masks,
        )

        # Mask illegal actions for loss computation
        policy_logits = policy_logits.masked_fill(~hand_masks, float('-inf'))
        log_policy = F.log_softmax(policy_logits, dim=-1)

        # Policy loss: cross-entropy with soft targets
        log_policy_safe = log_policy.clamp(min=-100)
        policy_loss = -(policy_targets * log_policy_safe).sum(dim=-1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(value, value_targets)

        # Combined loss
        loss = policy_loss + value_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

    def train_n_steps_from_buffer(
        self,
        model: ZebModel,
        optimizer: torch.optim.Optimizer,
        replay_buffer: GPUReplayBuffer,
        n_steps: int,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Train N steps sampling from GPU replay buffer - fused loop.

        Minimizes Python overhead by keeping everything in a tight loop with
        direct tensor indexing. No intermediate GPUTrainingExample objects.

        Args:
            model: ZebModel to train
            optimizer: Optimizer
            replay_buffer: GPUReplayBuffer with training data
            n_steps: Number of gradient steps
            batch_size: Samples per step

        Returns:
            Dict with average policy_loss and value_loss
        """
        model.train()
        device = replay_buffer.device
        buffer_size = len(replay_buffer)

        # Accumulate losses as GPU tensors (avoid .item() sync per step)
        total_policy_loss = torch.tensor(0.0, device=device)
        total_value_loss = torch.tensor(0.0, device=device)

        # Pre-fetch buffer tensors (avoid attribute lookup in loop)
        buf_obs = replay_buffer.observations
        buf_masks = replay_buffer.masks
        buf_hand_idx = replay_buffer.hand_indices
        buf_hand_masks = replay_buffer.hand_masks
        buf_policy = replay_buffer.policy_targets
        buf_value = replay_buffer.value_targets

        for _ in range(n_steps):
            # Sample indices on GPU
            indices = torch.randint(0, buffer_size, (batch_size,), device=device)

            # Index directly into buffer tensors (fast GPU ops)
            obs = buf_obs[indices]
            masks = buf_masks[indices]
            hand_idx = buf_hand_idx[indices]
            hand_masks = buf_hand_masks[indices]
            policy_targets = buf_policy[indices]
            value_targets = buf_value[indices]

            # Forward pass
            policy_logits, value = model(obs.long(), masks, hand_idx, hand_masks)

            # Mask illegal actions
            policy_logits = policy_logits.masked_fill(~hand_masks, float('-inf'))
            log_policy = F.log_softmax(policy_logits, dim=-1)

            # Policy loss: cross-entropy with soft targets
            log_policy_safe = log_policy.clamp(min=-100)
            policy_loss = -(policy_targets * log_policy_safe).sum(dim=-1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(value, value_targets)

            # Combined loss
            loss = policy_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate on GPU (no sync)
            total_policy_loss = total_policy_loss + policy_loss.detach()
            total_value_loss = total_value_loss + value_loss.detach()

        # Single sync at end
        return {
            'policy_loss': (total_policy_loss / n_steps).item(),
            'value_loss': (total_value_loss / n_steps).item(),
        }

    def train_epoch_gpu(
        self,
        model: ZebModel,
        optimizer: torch.optim.Optimizer,
        examples: GPUTrainingExample,
        batch_size: int = 128,
    ) -> dict[str, float]:
        """Train one epoch on GPU-native examples.

        Shuffles and batches data entirely on GPU.

        Args:
            model: ZebModel to train
            optimizer: Optimizer
            examples: GPUTrainingExample with all data
            batch_size: Mini-batch size

        Returns:
            Dict with average policy_loss and value_loss
        """
        model.train()
        n_samples = examples.n_examples

        # Shuffle on GPU
        perm = torch.randperm(n_samples, device=examples.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples - batch_size + 1, batch_size):
            idx = perm[start:start + batch_size]

            losses = self.train_batch_gpu(
                model=model,
                optimizer=optimizer,
                observations=examples.observations[idx],
                masks=examples.masks[idx],
                hand_indices=examples.hand_indices[idx],
                hand_masks=examples.hand_masks[idx],
                policy_targets=examples.policy_targets[idx],
                value_targets=examples.value_targets[idx],
            )

            total_policy_loss += losses['policy_loss']
            total_value_loss += losses['value_loss']
            n_batches += 1

        return {
            'policy_loss': total_policy_loss / max(n_batches, 1),
            'value_loss': total_value_loss / max(n_batches, 1),
        }


def create_gpu_pipeline(
    oracle_device: str = 'cuda',
    n_parallel_games: int = 16,
    n_simulations: int = 100,
    max_mcts_nodes: int = 1024,
    **kwargs,
) -> GPUTrainingPipeline:
    """Factory function to create GPU training pipeline.

    Args:
        oracle_device: Device for oracle (default 'cuda')
        n_parallel_games: Concurrent games per batch
        n_simulations: MCTS simulations per move
        max_mcts_nodes: Max nodes per MCTS tree
        **kwargs: Additional args for GPUTrainingPipeline

    Returns:
        Configured GPUTrainingPipeline
    """
    from forge.zeb.oracle_value import create_oracle_value_fn

    device = require_cuda(oracle_device, where="create_gpu_pipeline")
    # For CUDA-graph MCTS, prefer oracle compile disabled for capture stability.
    use_cudagraph_mcts = bool(kwargs.get("use_cudagraph_mcts", True))
    oracle = create_oracle_value_fn(
        device=oracle_device,
        compile=not use_cudagraph_mcts,
        use_async=False,
        use_gpu_tokenizer=True,
    )

    return GPUTrainingPipeline(
        oracle=oracle,
        device=device,
        n_parallel_games=n_parallel_games,
        n_simulations=n_simulations,
        max_mcts_nodes=max_mcts_nodes,
        **kwargs,
    )


def create_selfplay_pipeline(
    model: "ZebModel",
    device: str = 'cuda',
    n_parallel_games: int = 16,
    n_simulations: int = 100,
    max_mcts_nodes: int = 1024,
    **kwargs,
) -> GPUTrainingPipeline:
    """Factory function to create self-play training pipeline.

    Uses model's value head for MCTS leaf evaluation instead of oracle.
    This is the true AlphaZero approach - the model bootstraps from itself.

    Args:
        model: ZebModel to use for leaf evaluation (must be on device)
        device: Device for pipeline (default 'cuda')
        n_parallel_games: Concurrent games per batch
        n_simulations: MCTS simulations per move
        max_mcts_nodes: Max nodes per MCTS tree
        **kwargs: Additional args for GPUTrainingPipeline

    Returns:
        Configured GPUTrainingPipeline in self-play mode
    """
    torch_device = require_cuda(device, where="create_selfplay_pipeline")

    return GPUTrainingPipeline(
        oracle=None,  # No oracle in self-play mode
        device=torch_device,
        n_parallel_games=n_parallel_games,
        n_simulations=n_simulations,
        max_mcts_nodes=max_mcts_nodes,
        model=model,
        **kwargs,
    )
