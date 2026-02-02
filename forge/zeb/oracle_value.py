"""Oracle-based value function for MCTS leaf evaluation.

Wraps Stage1Oracle to provide a value_fn compatible with MCTS.
Uses max(Q-values) as state value, normalized to [-1, 1].
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from forge.eq.game import GameState
from forge.eq.oracle import Stage1Oracle


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
        worlds = []
        decl_ids = np.zeros(n, dtype=np.int32)
        actors = np.zeros(n, dtype=np.int32)
        leaders = np.zeros(n, dtype=np.int32)
        trick_plays_list = []
        remaining = np.zeros((n, 4), dtype=np.int64)

        # Also track original hands and legal actions for post-processing
        original_hands_list = []
        legal_actions_list = []

        for i, state in enumerate(states):
            # Reconstruct original hands
            original_hands = self._reconstruct_original_hands(state)
            original_hands_list.append(original_hands)
            worlds.append(original_hands)

            # Game state info
            current_player = state.current_player()
            decl_ids[i] = state.decl_id
            actors[i] = current_player
            leaders[i] = state.leader

            # Trick plays
            trick_plays = [(p, d) for p, d in state.current_trick]
            trick_plays_list.append(trick_plays)

            # Remaining bitmask
            for p in range(4):
                original_hand = original_hands[p]
                current_hand = set(state.hands[p])
                for local_idx, d in enumerate(original_hand):
                    if d in current_hand:
                        remaining[i, p] |= (1 << local_idx)

            # Legal actions for post-processing
            legal_actions_list.append(set(state.legal_actions()))

        # Query oracle in batch
        q_values = self.oracle.query_batch_multi_state(
            worlds=worlds,
            decl_ids=decl_ids,
            actors=actors,
            leaders=leaders,
            trick_plays_list=trick_plays_list,
            remaining=remaining,
        )

        # Post-process: extract max legal Q-value for each state
        values = []
        for i in range(n):
            original_hand = original_hands_list[i][actors[i]]
            legal_actions = legal_actions_list[i]

            # Build legal mask
            legal_mask = torch.zeros(7, dtype=torch.bool, device=q_values.device)
            for local_idx, domino_id in enumerate(original_hand):
                if domino_id in legal_actions:
                    legal_mask[local_idx] = True

            # Mask illegal actions
            masked_q = q_values[i].clone()
            masked_q[~legal_mask] = float('-inf')

            # Best Q-value (from Team 0's perspective)
            best_q = masked_q.max().item()

            # Flip sign if player is on Team 1
            if players[i] % 2 == 1:
                best_q = -best_q

            # Normalize to [-1, 1]
            values.append(best_q / 42.0)

        return values

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
    )
