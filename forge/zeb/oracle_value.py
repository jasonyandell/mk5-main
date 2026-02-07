"""Oracle-based value function for MCTS leaf evaluation.

Wraps Stage1Oracle to provide a value_fn compatible with MCTS.
Uses max(Q-values) as state value, normalized to [-1, 1].
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from forge.eq.game import GameState
from forge.eq.oracle import Stage1Oracle
from forge.zeb.gpu_preprocess import compute_remaining_bitmask_gpu, compute_legal_mask_gpu


# Default checkpoint path
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "models" / "domino-qval-3.3M-shuffle-qgap0.074-qmae0.96.ckpt"


class OracleValueFunction:
    """Wraps Stage1Oracle to provide MCTS-compatible value function.

    The oracle returns Q-values (expected points) for each action.
    We use max(Q-values) as the state value, normalized to [-1, 1].

    Q-values are from Team 0's perspective, so we flip sign for Team 1.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        device: str = "cuda",
        compile: bool = True,
        use_async: bool = True,
        use_gpu_tokenizer: bool = True,
    ):
        """Initialize oracle.

        Args:
            checkpoint_path: Path to Stage1Oracle checkpoint.
                           Defaults to the 3.3M shuffle checkpoint.
            device: Device to run oracle on.
            compile: Whether to torch.compile the model.
        """
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT

        self.oracle = Stage1Oracle(
            checkpoint_path=checkpoint_path,
            device=device,
            compile=compile,
            use_async=use_async,
            use_gpu_tokenizer=use_gpu_tokenizer,
        )
        self.device = device

        # Stats for debugging
        self.query_count = 0

    def __call__(self, state: GameState, player: int) -> float:
        """Evaluate state from player's perspective.

        Args:
            state: Current game state (with full hand information)
            player: Player whose perspective to evaluate from

        Returns:
            Value in [-1, 1] range. Positive means player's team is ahead.
        """
        self.query_count += 1

        # Convert GameState to oracle format
        worlds, game_state_info, current_player, original_hands = self._convert_state(state)

        # Query oracle
        q_values = self.oracle.query_batch(
            worlds=worlds,
            game_state_info=game_state_info,
            current_player=current_player,
        )

        # Get best Q-value (expected points with optimal play)
        # Only consider legal actions (dominoes still in hand that can be played)
        legal_actions = set(state.legal_actions())
        legal_mask = torch.zeros(7, dtype=torch.bool, device=q_values.device)

        # Map domino IDs to local indices (0-6) in the ORIGINAL sorted hand
        # Oracle Q-values are indexed by position in the original 7-card hand
        original_hand = original_hands[current_player]
        for local_idx, domino_id in enumerate(original_hand):
            if domino_id in legal_actions:
                legal_mask[local_idx] = True

        # Mask illegal actions with -inf
        masked_q = q_values[0].clone()
        masked_q[~legal_mask] = float('-inf')

        # Best Q-value (from Team 0's perspective)
        best_q = masked_q.max().item()

        # Flip sign if player is on Team 1
        # Q-values are from Team 0's perspective (positive = Team 0 ahead)
        if player % 2 == 1:
            best_q = -best_q

        # Normalize to [-1, 1] (Q-values are in [-42, +42] points range)
        return best_q / 42.0

    def batch_evaluate(
        self,
        states: list[GameState],
        players: list[int],
    ) -> list[float]:
        """Batch evaluate multiple states.

        Uses query_batch_multi_state() for efficient GPU utilization.

        Args:
            states: List of game states to evaluate
            players: List of players whose perspective to evaluate from

        Returns:
            List of values in [-1, 1] range
        """
        if not states:
            return []

        n = len(states)
        self.query_count += n

        # Prepare batch inputs
        decl_ids = np.zeros(n, dtype=np.int32)
        actors = np.zeros(n, dtype=np.int32)
        leaders = np.zeros(n, dtype=np.int32)
        trick_plays_list = []

        # Also track original hands and legal actions for post-processing
        original_hands_list = []
        legal_actions_list = []

        for i, state in enumerate(states):
            # Reconstruct original hands
            original_hands = self._reconstruct_original_hands(state)
            original_hands_list.append(original_hands)

            # Game state info
            current_player = state.current_player()
            decl_ids[i] = state.decl_id
            actors[i] = current_player
            leaders[i] = state.leader

            # Trick plays
            trick_plays = [(p, d) for p, d in state.current_trick]
            trick_plays_list.append(trick_plays)

            # Legal actions for post-processing
            legal_actions_list.append(set(state.legal_actions()))

        # Vectorized remaining bitmask computation
        # Convert original_hands_list to numpy array (n, 4, 7)
        original_hands_arr = np.array(original_hands_list, dtype=np.int64)

        # Build current hands array with -1 padding for empty slots
        # Max hand size is 7, but current hands may have fewer dominoes
        current_hands_arr = np.full((n, 4, 7), -1, dtype=np.int64)
        for i, state in enumerate(states):
            for p in range(4):
                hand = state.hands[p]
                current_hands_arr[i, p, :len(hand)] = hand

        # For each position in original hand, check if that domino is in current hand
        # original_hands_arr: (n, 4, 7) - the domino IDs
        # current_hands_arr: (n, 4, 7) - current dominoes with -1 padding
        # We need to check if original_hands_arr[i, p, j] is in current_hands_arr[i, p, :]

        # Expand dimensions for broadcasting:
        # original_hands_arr[:, :, :, np.newaxis] -> (n, 4, 7, 1)
        # current_hands_arr[:, :, np.newaxis, :] -> (n, 4, 1, 7)
        # Compare: (n, 4, 7, 7) -> any match along last axis -> (n, 4, 7) bool
        in_hand = np.any(
            original_hands_arr[:, :, :, np.newaxis] == current_hands_arr[:, :, np.newaxis, :],
            axis=3
        )  # Shape: (n, 4, 7)

        # Convert boolean mask to bitmask
        # in_hand[i, p, j] is True if original hand position j is still in current hand
        # We need remaining[i, p] = sum over j of (in_hand[i, p, j] << j)
        bit_positions = np.array([1 << j for j in range(7)], dtype=np.int64)  # [1, 2, 4, 8, 16, 32, 64]
        remaining = np.sum(in_hand * bit_positions, axis=2)  # Shape: (n, 4)

        # Query oracle in batch
        q_values = self.oracle.query_batch_multi_state(
            worlds=original_hands_list,
            decl_ids=decl_ids,
            actors=actors,
            leaders=leaders,
            trick_plays_list=trick_plays_list,
            remaining=remaining,
        )

        # Vectorized post-processing: extract max legal Q-value for each state
        # Build legal mask on CPU with numpy, then transfer once to GPU
        legal_mask_np = np.zeros((n, 7), dtype=bool)
        for i in range(n):
            original_hand = original_hands_list[i][actors[i]]
            legal_actions = legal_actions_list[i]
            for local_idx, domino_id in enumerate(original_hand):
                if domino_id in legal_actions:
                    legal_mask_np[i, local_idx] = True

        legal_mask = torch.from_numpy(legal_mask_np).to(q_values.device)

        # Mask illegal actions with -inf (vectorized)
        masked_q = q_values.clone()
        masked_q[~legal_mask] = float('-inf')

        # Get best Q-value per state (vectorized)
        best_q = masked_q.max(dim=1).values  # (n,)

        # Flip sign for Team 1 players (vectorized)
        players_arr = np.array(players)
        team1_mask = torch.from_numpy(players_arr % 2 == 1).to(q_values.device)
        best_q[team1_mask] = -best_q[team1_mask]

        # Normalize and convert to list
        return (best_q / 42.0).cpu().tolist()

    def batch_evaluate_with_originals(
        self,
        states: list[GameState],
        players: list[int],
        original_hands_list: list[list[list[int]]],
    ) -> list[float]:
        """Batch evaluate multiple states with pre-computed original hands.

        This is an optimized version of batch_evaluate() that accepts original
        hands as a parameter instead of reconstructing them. Use this when the
        caller already has access to original hands (e.g., from ActiveGame).

        Args:
            states: List of game states to evaluate
            players: List of players whose perspective to evaluate from
            original_hands_list: List of original hands for each state.
                Each element is a list of 4 hands (one per player),
                where each hand is a list of 7 domino IDs.

        Returns:
            List of values in [-1, 1] range
        """
        if not states:
            return []

        n = len(states)
        self.query_count += n

        # Prepare batch inputs
        worlds = original_hands_list  # Already in correct format
        decl_ids = np.zeros(n, dtype=np.int32)
        actors = np.zeros(n, dtype=np.int32)
        leaders = np.zeros(n, dtype=np.int32)
        trick_plays_list = []

        # Track legal actions for post-processing
        legal_actions_list = []

        for i, state in enumerate(states):
            # Game state info
            current_player = state.current_player()
            decl_ids[i] = state.decl_id
            actors[i] = current_player
            leaders[i] = state.leader

            # Trick plays
            trick_plays = [(p, d) for p, d in state.current_trick]
            trick_plays_list.append(trick_plays)

            # Legal actions for post-processing
            legal_actions_list.append(set(state.legal_actions()))

        # Vectorized remaining bitmask computation
        # Convert original_hands_list to numpy array (n, 4, 7)
        original_hands_arr = np.array(original_hands_list, dtype=np.int64)

        # Build current hands array with -1 padding for empty slots
        # Max hand size is 7, but current hands may have fewer dominoes
        current_hands_arr = np.full((n, 4, 7), -1, dtype=np.int64)
        for i, state in enumerate(states):
            for p in range(4):
                hand = state.hands[p]
                current_hands_arr[i, p, :len(hand)] = hand

        # For each position in original hand, check if that domino is in current hand
        # original_hands_arr: (n, 4, 7) - the domino IDs
        # current_hands_arr: (n, 4, 7) - current dominoes with -1 padding
        # We need to check if original_hands_arr[i, p, j] is in current_hands_arr[i, p, :]

        # Expand dimensions for broadcasting:
        # original_hands_arr[:, :, :, np.newaxis] -> (n, 4, 7, 1)
        # current_hands_arr[:, :, np.newaxis, :] -> (n, 4, 1, 7)
        # Compare: (n, 4, 7, 7) -> any match along last axis -> (n, 4, 7) bool
        in_hand = np.any(
            original_hands_arr[:, :, :, np.newaxis] == current_hands_arr[:, :, np.newaxis, :],
            axis=3
        )  # Shape: (n, 4, 7)

        # Convert boolean mask to bitmask
        # in_hand[i, p, j] is True if original hand position j is still in current hand
        # We need remaining[i, p] = sum over j of (in_hand[i, p, j] << j)
        bit_positions = np.array([1 << j for j in range(7)], dtype=np.int64)  # [1, 2, 4, 8, 16, 32, 64]
        remaining = np.sum(in_hand * bit_positions, axis=2)  # Shape: (n, 4)

        # Query oracle in batch
        q_values = self.oracle.query_batch_multi_state(
            worlds=worlds,
            decl_ids=decl_ids,
            actors=actors,
            leaders=leaders,
            trick_plays_list=trick_plays_list,
            remaining=remaining,
        )

        # Vectorized post-processing: extract max legal Q-value for each state
        # Build legal mask on CPU with numpy, then transfer once to GPU
        legal_mask_np = np.zeros((n, 7), dtype=bool)
        for i in range(n):
            original_hand = original_hands_list[i][actors[i]]
            legal_actions = legal_actions_list[i]
            for local_idx, domino_id in enumerate(original_hand):
                if domino_id in legal_actions:
                    legal_mask_np[i, local_idx] = True

        legal_mask = torch.from_numpy(legal_mask_np).to(q_values.device)

        # Mask illegal actions with -inf (vectorized)
        masked_q = q_values.clone()
        masked_q[~legal_mask] = float('-inf')

        # Get best Q-value per state (vectorized)
        best_q = masked_q.max(dim=1).values  # (n,)

        # Flip sign for Team 1 players (vectorized)
        players_arr = np.array(players)
        team1_mask = torch.from_numpy(players_arr % 2 == 1).to(q_values.device)
        best_q[team1_mask] = -best_q[team1_mask]

        # Normalize and convert to list
        return (best_q / 42.0).cpu().tolist()

    def batch_evaluate_gpu(
        self,
        original_hands: Tensor,  # (N, 4, 7) int32 - original deal
        current_hands: Tensor,   # (N, 4, 7) int32 - current hands, -1 for played
        decl_ids: Tensor,        # (N,) int32
        actors: Tensor,          # (N,) int32 - current player
        leaders: Tensor,         # (N,) int32 - trick leader
        trick_players: Tensor,   # (N, 3) int32 - player for each trick play, -1 if none
        trick_dominoes: Tensor,  # (N, 3) int32 - domino for each trick play, -1 if none
        players: Tensor,         # (N,) int32 - perspective player for value
    ) -> Tensor:
        """Fully GPU-native batch evaluation.

        All inputs are GPU tensors. No CPU work in the hot path.

        Args:
            original_hands: Original 7-card hands (N, 4, 7)
            current_hands: Current hands with -1 for played dominoes (N, 4, 7)
            decl_ids: Declaration ID per sample (N,)
            actors: Current player per sample (N,)
            leaders: Trick leader per sample (N,)
            trick_players: Player who made each trick play (N, 3), -1 if no play
            trick_dominoes: Domino played in each trick position (N, 3), -1 if no play
            players: Player perspective for value computation (N,)

        Returns:
            values: (N,) tensor of values in [-1, 1]
        """
        n = original_hands.shape[0]
        device = original_hands.device

        # If we're not on CUDA (or Stage1Oracle was created without GPU tokenizer),
        # fall back to Stage1Oracle.query_batch_multi_state() + GPU-style postprocessing.
        # This keeps the API usable for CPU tests and non-CUDA environments.
        if device.type != "cuda" or not hasattr(self.oracle, "gpu_tokenizer"):
            self.query_count += n

            worlds = original_hands.to("cpu").tolist()
            decl_ids_np = decl_ids.to("cpu").numpy().astype(np.int32)
            actors_np = actors.to("cpu").numpy().astype(np.int32)
            leaders_np = leaders.to("cpu").numpy().astype(np.int32)

            # Remaining bitmask (N, 4) int64
            remaining = compute_remaining_bitmask_gpu(original_hands, current_hands)
            remaining_np = remaining.to("cpu").numpy().astype(np.int64)

            # Build trick plays list expected by Stage1Oracle: List[List[(player, domino)]]
            tp = trick_players.to("cpu").numpy().astype(np.int32)
            td = trick_dominoes.to("cpu").numpy().astype(np.int32)
            trick_plays_list: list[list[tuple[int, int]]] = []
            for i in range(n):
                plays: list[tuple[int, int]] = []
                for j in range(3):
                    p = int(tp[i, j])
                    if p < 0:
                        continue
                    plays.append((p, int(td[i, j])))
                trick_plays_list.append(plays)

            q_values = self.oracle.query_batch_multi_state(
                worlds=worlds,
                decl_ids=decl_ids_np,
                actors=actors_np,
                leaders=leaders_np,
                trick_plays_list=trick_plays_list,
                remaining=remaining_np,
            )  # (N, 7) on oracle.device

            legal_mask = compute_legal_mask_gpu(original_hands, current_hands, actors)
            masked_q = q_values.clone()
            masked_q[~legal_mask] = float("-inf")
            best_q = masked_q.max(dim=1).values

            team1_mask = (players % 2) == 1
            best_q = torch.where(team1_mask, -best_q, best_q)
            return best_q / 42.0

        # Compute remaining bitmask on GPU
        remaining = compute_remaining_bitmask_gpu(original_hands, current_hands)

        # GPU tokenization
        tokens, masks = self.oracle.gpu_tokenizer.tokenize(
            worlds=original_hands,
            decl_ids=decl_ids,
            actors=actors,
            leaders=leaders,
            remaining=remaining,
            trick_players=trick_players,
            trick_dominoes=trick_dominoes,
        )

        # Forward pass
        with torch.inference_mode():
            q_values, _ = self.oracle.model(tokens, masks, actors.long())

        # Compute legal mask on GPU
        legal_mask = compute_legal_mask_gpu(original_hands, current_hands, actors)

        # Mask illegal actions and get max
        masked_q = q_values.clone()
        masked_q[~legal_mask] = float('-inf')
        best_q = masked_q.max(dim=1).values

        # Flip sign for Team 1 players
        team1_mask = (players % 2) == 1
        best_q = torch.where(team1_mask, -best_q, best_q)

        # Normalize to [-1, 1]
        return best_q / 42.0

    def _convert_state(self, state: GameState) -> tuple:
        """Convert GameState to oracle query format.

        Returns:
            (worlds, game_state_info, current_player, original_hands) tuple
        """
        current_player = state.current_player()

        # Worlds: list of 4 hands (each is list of 7 domino IDs)
        # Need to include played dominoes to maintain 7-domino structure
        # Oracle expects full original hands, with 'remaining' mask indicating what's left

        # Reconstruct original hands from play_history
        original_hands = self._reconstruct_original_hands(state)
        worlds = [original_hands]

        # Compute remaining bitmask: bit i set if local_idx i is still in hand
        remaining = np.zeros((1, 4), dtype=np.int64)
        for p in range(4):
            original_hand = original_hands[p]
            current_hand = set(state.hands[p])
            for local_idx, d in enumerate(original_hand):
                if d in current_hand:
                    remaining[0, p] |= (1 << local_idx)

        # Trick plays: (player, domino_id) format
        trick_plays = [(p, d) for p, d in state.current_trick]

        game_state_info = {
            'decl_id': state.decl_id,
            'leader': state.leader,
            'trick_plays': trick_plays,
            'remaining': remaining,
        }

        return worlds, game_state_info, current_player, original_hands

    def _reconstruct_original_hands(self, state: GameState) -> list[list[int]]:
        """Reconstruct original 7-card hands from current state.

        The oracle expects the original deal, with 'remaining' indicating
        which dominoes are still in hand.
        """
        # Start with current hands
        hands = [list(h) for h in state.hands]

        # Add back played dominoes in order they were played
        for player, domino_id, _ in state.play_history:
            hands[player].append(domino_id)

        # Sort each hand to get consistent ordering
        # (Original deal order doesn't matter for oracle, just need 7 dominoes)
        for h in hands:
            h.sort()

        return hands


def create_oracle_value_fn(
    checkpoint_path: Optional[str | Path] = None,
    device: str = "cuda",
    compile: bool = True,
    use_async: bool = True,
    use_gpu_tokenizer: bool = True,
) -> OracleValueFunction:
    """Factory function to create oracle value function.

    Args:
        checkpoint_path: Path to checkpoint. Defaults to 3.3M shuffle model.
        device: Device to run on.
        compile: Whether to torch.compile.

    Returns:
        Callable value function for MCTS.
    """
    return OracleValueFunction(
        checkpoint_path=checkpoint_path,
        device=device,
        compile=compile,
        use_async=use_async,
        use_gpu_tokenizer=use_gpu_tokenizer,
    )
