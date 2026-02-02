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
    prepare_oracle_inputs,
    select_leaves_gpu,
)

if TYPE_CHECKING:
    from forge.zeb.oracle_value import OracleValueFunction
    from forge.zeb.model import ZebModel


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
        self.device = device
        self.tables = get_domino_tables(device)

        # Build count value lookup table
        # Points: 0->0, 5->1, 10->2
        self.count_values = self._build_count_values(device)

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

        # Initialize output tensors
        tokens = torch.zeros(
            (n, self.MAX_TOKENS, self.N_FEATURES),
            dtype=torch.int32,
            device=device
        )
        masks = torch.zeros((n, self.MAX_TOKENS), dtype=torch.bool, device=device)

        batch_idx = torch.arange(n, device=device, dtype=torch.int64)
        players_i64 = perspective_players.to(torch.int64)

        # === Token 0: Declaration ===
        tokens[:, 0, self.FEAT_DECL] = states.decl_id
        tokens[:, 0, self.FEAT_TOKEN_TYPE] = self.TOKEN_TYPE_DECL
        masks[:, 0] = True

        # === Tokens 1-7: Hand slots ===
        # Get perspective player's hand from original hands
        my_hand = original_hands[batch_idx, players_i64]  # (N, 7) domino IDs

        # Check which slots still have their domino (not in played)
        # played_mask is (N, 28) bool - True if domino played
        # For each hand slot, check if that domino is in played
        played_lookup = states.played_mask[batch_idx.unsqueeze(1), my_hand.long()]  # (N, 7)
        in_hand = ~played_lookup  # True if still in hand

        # Encode domino features for hand slots
        safe_ids = my_hand.clamp(min=0).long()  # Safe for lookup
        tokens[:, 1:8, self.FEAT_HIGH] = self.tables.high[safe_ids]
        tokens[:, 1:8, self.FEAT_LOW] = self.tables.low[safe_ids]
        tokens[:, 1:8, self.FEAT_IS_DOUBLE] = self.tables.is_double[safe_ids].int()
        tokens[:, 1:8, self.FEAT_COUNT] = self.count_values[safe_ids]
        tokens[:, 1:8, self.FEAT_PLAYER] = 0  # Always "me" for hand tokens
        tokens[:, 1:8, self.FEAT_IS_IN_HAND] = in_hand.int()
        tokens[:, 1:8, self.FEAT_DECL] = states.decl_id.unsqueeze(1).expand(-1, 7)
        tokens[:, 1:8, self.FEAT_TOKEN_TYPE] = self.TOKEN_TYPE_HAND
        masks[:, 1:8] = in_hand  # Only valid if domino still in hand

        # === Tokens 8-35: Play history ===
        # play_history is (N, 28, 3) with (player, domino, lead_domino)
        n_plays = states.n_plays  # (N,)

        for play_idx in range(28):
            token_pos = 8 + play_idx

            # Only process games that have this many plays
            has_play = n_plays > play_idx

            if not has_play.any():
                break

            # Get play info
            play_player = states.play_history[:, play_idx, 0]  # (N,)
            play_domino = states.play_history[:, play_idx, 1]  # (N,)

            # Compute relative player
            rel_player = (play_player - perspective_players + 4) % 4  # (N,)

            # Encode domino
            safe_domino = play_domino.clamp(min=0).long()
            tokens[:, token_pos, self.FEAT_HIGH] = torch.where(
                has_play, self.tables.high[safe_domino], 0
            )
            tokens[:, token_pos, self.FEAT_LOW] = torch.where(
                has_play, self.tables.low[safe_domino], 0
            )
            tokens[:, token_pos, self.FEAT_IS_DOUBLE] = torch.where(
                has_play, self.tables.is_double[safe_domino].int(), 0
            )
            tokens[:, token_pos, self.FEAT_COUNT] = torch.where(
                has_play, self.count_values[safe_domino], 0
            )
            tokens[:, token_pos, self.FEAT_PLAYER] = torch.where(
                has_play, rel_player, 0
            )
            tokens[:, token_pos, self.FEAT_IS_IN_HAND] = 0  # Played dominoes not in hand
            tokens[:, token_pos, self.FEAT_DECL] = torch.where(
                has_play, states.decl_id, 0
            )
            tokens[:, token_pos, self.FEAT_TOKEN_TYPE] = torch.where(
                has_play, self.TOKEN_TYPE_PLAY, 0
            )
            masks[:, token_pos] = has_play

        # Hand indices always point to positions 1-7
        hand_indices = torch.arange(1, 8, dtype=torch.int64, device=device)
        hand_indices = hand_indices.unsqueeze(0).expand(n, -1)  # (N, 7)

        # Hand mask = which slots have unplayed dominoes
        hand_masks = in_hand  # (N, 7)

        return tokens, masks, hand_indices, hand_masks


class GPUTrainingPipeline:
    """Zero-copy training pipeline - all data stays on GPU.

    Generates training examples through MCTS self-play entirely on GPU:
    - Deals random games on GPU
    - Runs MCTS with GPU tree operations
    - Evaluates leaves with oracle (GPU tensors)
    - Collects training examples without CPU transfer
    - Trains model on GPU-native tensors
    """

    def __init__(
        self,
        oracle: OracleValueFunction,
        device: torch.device,
        n_parallel_games: int = 16,
        n_simulations: int = 100,
        max_mcts_nodes: int = 1024,
        c_puct: float = 1.414,
        temperature: float = 1.0,
    ):
        """Initialize pipeline.

        Args:
            oracle: OracleValueFunction for leaf evaluation (must support batch_evaluate_gpu)
            device: GPU device
            n_parallel_games: Number of concurrent games per batch
            n_simulations: MCTS simulations per move
            max_mcts_nodes: Maximum nodes per MCTS tree
            c_puct: UCB exploration constant
            temperature: Action sampling temperature (1.0 = proportional to visits)
        """
        self.oracle = oracle
        self.device = device
        self.n_parallel_games = n_parallel_games
        self.n_simulations = n_simulations
        self.max_mcts_nodes = max_mcts_nodes
        self.c_puct = c_puct
        self.temperature = temperature

        # Initialize tokenizer
        self.tokenizer = GPUObservationTokenizer(device)

        # Stats for monitoring
        self.total_games_generated = 0
        self.total_oracle_queries = 0

    def generate_games_gpu(
        self,
        n_games: int,
    ) -> GPUTrainingExample:
        """Generate training examples entirely on GPU.

        Plays n_games of self-play using MCTS, collecting training examples
        at each move. All computation stays on GPU.

        Args:
            n_games: Number of games to generate

        Returns:
            GPUTrainingExample with all training data on GPU
        """
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
            game_examples = self._play_games_mcts(deals, original_hands)

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
    ) -> dict[str, Tensor]:
        """Play games to completion using MCTS.

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

        # Play all 28 moves
        for move_idx in range(28):
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

            # Create MCTS forest
            # Pass original_hands for oracle queries (states.hands has -1 for played)
            forest = create_forest(
                n_trees=n,
                max_nodes=self.max_mcts_nodes,
                initial_states=states,
                device=device,
                c_puct=self.c_puct,
                original_hands=original_hands,
            )

            # Run MCTS simulations
            for _ in range(self.n_simulations):
                # Select leaves
                leaf_indices, paths = select_leaves_gpu(forest)

                # Get leaf states
                leaf_states = get_leaf_states(forest, leaf_indices)

                # Check for terminals
                terminal_values, is_terminal = get_terminal_values(
                    forest, leaf_indices, leaf_states
                )

                # Evaluate non-terminals with oracle
                if (~is_terminal).any():
                    oracle_values = self._evaluate_oracle_gpu(forest, leaf_states)
                else:
                    oracle_values = torch.zeros(n, dtype=torch.float32, device=device)

                # Combine terminal and oracle values
                values = torch.where(is_terminal, terminal_values, oracle_values)

                # Expand and backpropagate
                expand_gpu(forest, leaf_indices)
                backprop_gpu(forest, leaf_indices, values, paths)

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

    def _evaluate_oracle_gpu(
        self,
        forest: GPUMCTSForest,
        leaf_states: GPUGameState,
    ) -> Tensor:
        """Evaluate leaf states with oracle - all on GPU.

        Args:
            forest: MCTS forest (provides original_hands)
            leaf_states: States to evaluate

        Returns:
            values: (N,) float32 values from root player's perspective
        """
        # Prepare oracle inputs
        oracle_inputs = prepare_oracle_inputs(forest, leaf_states)

        # Call oracle's GPU-native method
        values = self.oracle.batch_evaluate_gpu(**oracle_inputs)

        self.total_oracle_queries += forest.n_trees

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
        policies = policies * hand_masks.float()
        total = policies.sum(dim=1, keepdim=True).clamp(min=1e-8)
        policies = policies / total

        # Handle zero-probability case (fallback to uniform)
        zero_policy = (total.squeeze(1) < 1e-8)
        if zero_policy.any():
            uniform = hand_masks.float() / hand_masks.sum(dim=1, keepdim=True).clamp(min=1)
            policies = torch.where(
                zero_policy.unsqueeze(1),
                uniform,
                policies
            )

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

    device = torch.device(oracle_device)
    oracle = create_oracle_value_fn(device=oracle_device, compile=True)

    return GPUTrainingPipeline(
        oracle=oracle,
        device=device,
        n_parallel_games=n_parallel_games,
        n_simulations=n_simulations,
        max_mcts_nodes=max_mcts_nodes,
        **kwargs,
    )
