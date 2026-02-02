"""Monte Carlo Tree Search for Zeb.

Implements determinized UCT with root sampling for imperfect information games.
At root, samples N opponent hand configurations, runs UCT on each, aggregates.

Uses eq/ infrastructure:
- forge/eq/game.py: GameState for game mechanics
- forge/eq/sampling_mrv_gpu.py: WorldSamplerMRV for hand sampling (future)
- forge/eq/oracle.py: Stage1Oracle for leaf evaluation (optional)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from forge.eq.game import GameState
from forge.oracle.rng import deal_from_seed


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""

    state: GameState
    parent: Optional[MCTSNode] = None
    action: Optional[int] = None  # Domino ID that led here

    # Statistics
    visits: int = 0
    value_sum: float = 0.0

    # Children: domino_id -> MCTSNode
    children: dict = field(default_factory=dict)

    # Prior probability from policy network (if available)
    prior: float = 1.0

    @property
    def value(self) -> float:
        """Mean value estimate."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, c_puct: float = 1.414) -> float:
        """UCB1 score for selection."""
        if self.visits == 0:
            return float('inf')

        parent_visits = self.parent.visits if self.parent else 1
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.value + exploration

    def is_terminal(self) -> bool:
        return self.state.is_complete()

    def is_expanded(self) -> bool:
        """Check if all legal actions have been expanded."""
        if self.is_terminal():
            return True
        return len(self.children) == len(self.state.legal_actions())


class MCTS:
    """Monte Carlo Tree Search with UCT selection."""

    def __init__(
        self,
        c_puct: float = 1.414,
        n_simulations: int = 100,
        value_fn: Optional[callable] = None,
        wave_size: int = 32,
    ):
        """
        Args:
            c_puct: Exploration constant for UCB
            n_simulations: Number of simulations per search
            value_fn: Optional function(GameState, player) -> value in [-1, 1]
                     If None, uses random rollout.
                     If value_fn has batch_evaluate(), will use batched search.
            wave_size: Number of leaves to collect before batch evaluation
        """
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.value_fn = value_fn
        self.wave_size = wave_size

    def search(self, root_state: GameState, player: int) -> dict[int, int]:
        """Run MCTS from root_state, return action visit counts.

        Automatically uses batched search if value_fn supports batch_evaluate().

        Args:
            root_state: Current game state
            player: Player to move (for value perspective)

        Returns:
            Dict mapping domino_id -> visit count
        """
        # Use batched search if value_fn supports it
        if self.value_fn is not None and hasattr(self.value_fn, 'batch_evaluate'):
            return self._search_batched(root_state, player)

        return self._search_sequential(root_state, player)

    def _search_sequential(self, root_state: GameState, player: int) -> dict[int, int]:
        """Sequential MCTS search (original algorithm)."""
        root = MCTSNode(state=root_state)

        for _ in range(self.n_simulations):
            node = root

            # Selection: traverse tree using UCB
            while node.is_expanded() and not node.is_terminal():
                node = self._select_child(node)

            # Expansion: add one child
            if not node.is_terminal():
                node = self._expand(node)

            # Evaluation: get value estimate
            value = self._evaluate(node, player)

            # Backpropagation: update statistics
            self._backpropagate(node, value, player)

        # Return visit counts for root's children
        return {action: child.visits for action, child in root.children.items()}

    def _search_batched(self, root_state: GameState, player: int) -> dict[int, int]:
        """Batched MCTS search with virtual loss.

        Collects multiple leaves in a "wave", batch evaluates them,
        then backpropagates all values. Uses virtual loss to encourage
        exploration diversity within each wave.
        """
        root = MCTSNode(state=root_state)
        sims_remaining = self.n_simulations

        while sims_remaining > 0:
            wave_count = min(self.wave_size, sims_remaining)
            pending: list[tuple[MCTSNode, list[MCTSNode]]] = []

            # Collect wave of leaves with virtual loss
            for _ in range(wave_count):
                node = root
                path: list[MCTSNode] = []

                # Selection with virtual loss
                while node.is_expanded() and not node.is_terminal():
                    node.visits += 1  # Virtual loss
                    path.append(node)
                    node = self._select_child(node)

                # Expansion
                if not node.is_terminal():
                    path.append(node)
                    node.visits += 1  # Virtual loss on parent
                    node = self._expand(node)

                # Add leaf to pending
                node.visits += 1  # Virtual loss on leaf
                path.append(node)
                pending.append((node, path))

            # Separate terminal vs non-terminal leaves
            terminal_pending = [(leaf, path) for leaf, path in pending if leaf.is_terminal()]
            eval_pending = [(leaf, path) for leaf, path in pending if not leaf.is_terminal()]

            # Evaluate terminals immediately
            for leaf, path in terminal_pending:
                value = self._terminal_value(leaf.state, player)
                self._backpropagate_value_only(path, value, player)

            # Batch evaluate non-terminals
            if eval_pending:
                states = [leaf.state for leaf, _ in eval_pending]
                players = [player] * len(states)
                values = self.value_fn.batch_evaluate(states, players)

                for (leaf, path), value in zip(eval_pending, values):
                    self._backpropagate_value_only(path, value, player)

            sims_remaining -= wave_count

        return {action: child.visits for action, child in root.children.items()}

    def _backpropagate_value_only(self, path: list[MCTSNode], value: float, root_player: int):
        """Update value_sum only (visits were incremented during virtual loss)."""
        for node in path:
            if node.parent is not None:
                parent_player = node.parent.state.current_player()
                # If parent's player is same team as root_player, use value; else negate
                if parent_player % 2 == root_player % 2:
                    node.value_sum += value
                else:
                    node.value_sum -= value
            else:
                # Root node
                node.value_sum += value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = float('-inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding one unexplored child."""
        legal = node.state.legal_actions()

        # Find first unexplored action
        for action in legal:
            if action not in node.children:
                new_state = node.state.apply_action(action)
                child = MCTSNode(
                    state=new_state,
                    parent=node,
                    action=action,
                    prior=1.0 / len(legal),  # Uniform prior
                )
                node.children[action] = child
                return child

        # All expanded (shouldn't reach here if is_expanded() is correct)
        return node

    def _evaluate(self, node: MCTSNode, root_player: int) -> float:
        """Evaluate leaf node."""
        if node.is_terminal():
            # Game over - compute actual outcome
            return self._terminal_value(node.state, root_player)

        if self.value_fn is not None:
            # Use provided value function
            return self.value_fn(node.state, root_player)

        # Random rollout
        return self._random_rollout(node.state, root_player)

    def _terminal_value(self, state: GameState, player: int) -> float:
        """Compute value for terminal state."""
        # Count points for each team
        team0_points = 0
        team1_points = 0

        for p, domino_id, _ in state.play_history:
            from forge.oracle.tables import DOMINO_COUNT_POINTS
            points = DOMINO_COUNT_POINTS[domino_id]
            if p % 2 == 0:
                team0_points += points
            else:
                team1_points += points

        # Player's team
        player_team = player % 2
        my_points = team0_points if player_team == 0 else team1_points
        opp_points = team1_points if player_team == 0 else team0_points

        # Simple win/loss
        if my_points > opp_points:
            return 1.0
        elif my_points < opp_points:
            return -1.0
        return 0.0

    def _random_rollout(self, state: GameState, player: int) -> float:
        """Random playout to terminal state."""
        import random

        current = state
        while not current.is_complete():
            legal = current.legal_actions()
            action = random.choice(legal)
            current = current.apply_action(action)

        return self._terminal_value(current, player)

    def _backpropagate(self, node: MCTSNode, value: float, root_player: int):
        """Update statistics from leaf to root."""
        current = node
        while current is not None:
            current.visits += 1
            # Flip value for opponent's perspective
            current_player = current.state.current_player() if not current.is_terminal() else -1
            if current.parent is not None:
                parent_player = current.parent.state.current_player()
                # If parent's player is same team as root_player, use value; else negate
                if parent_player % 2 == root_player % 2:
                    current.value_sum += value
                else:
                    current.value_sum -= value
            else:
                # Root node
                current.value_sum += value
            current = current.parent


class DeterminizedMCTS:
    """MCTS with root sampling for imperfect information.

    Samples multiple consistent opponent hands, runs UCT on each,
    aggregates action counts.
    """

    def __init__(
        self,
        n_determinizations: int = 10,
        n_simulations: int = 50,
        c_puct: float = 1.414,
        value_fn: Optional[callable] = None,
        wave_size: int = 32,
    ):
        """
        Args:
            n_determinizations: Number of opponent hand samples
            n_simulations: Simulations per determinization
            c_puct: Exploration constant
            value_fn: Optional value function for leaf evaluation
            wave_size: Batch size for leaf evaluation (if value_fn supports it)
        """
        self.n_determinizations = n_determinizations
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.value_fn = value_fn
        self.wave_size = wave_size

        self.mcts = MCTS(
            c_puct=c_puct,
            n_simulations=n_simulations,
            value_fn=value_fn,
            wave_size=wave_size,
        )

    def search(
        self,
        known_hand: tuple[int, ...],
        played: frozenset[int],
        decl_id: int,
        leader: int,
        current_trick: tuple[tuple[int, int], ...],
        play_history: tuple[tuple[int, int, int], ...],
        player: int,
        seed: int = 0,
    ) -> dict[int, float]:
        """Run determinized MCTS, return action probabilities.

        Args:
            known_hand: Player's own hand (domino IDs)
            played: Set of played domino IDs
            decl_id: Declaration ID
            leader: Current trick leader
            current_trick: Current trick state
            play_history: History of plays
            player: Current player (whose hand we know)
            seed: Random seed for sampling

        Returns:
            Dict mapping domino_id -> probability (normalized visit counts)
        """
        import random
        rng = random.Random(seed)

        # Aggregate visit counts across determinizations
        total_visits: dict[int, int] = {}

        for det_idx in range(self.n_determinizations):
            # Sample consistent opponent hands
            sampled_hands = self._sample_hands(
                known_hand, played, player, decl_id, play_history,
                seed=rng.randint(0, 2**31),
            )

            # Build complete game state
            state = GameState(
                hands=sampled_hands,
                played=played,
                play_history=play_history,
                current_trick=current_trick,
                leader=leader,
                decl_id=decl_id,
            )

            # Run MCTS
            visits = self.mcts.search(state, player)

            # Aggregate
            for action, count in visits.items():
                total_visits[action] = total_visits.get(action, 0) + count

        # Normalize to probabilities
        total = sum(total_visits.values())
        if total == 0:
            # Fallback to uniform
            legal = tuple(d for d in known_hand if d not in played)
            return {d: 1.0 / len(legal) for d in legal}

        return {action: count / total for action, count in total_visits.items()}

    def _sample_hands(
        self,
        known_hand: tuple[int, ...],
        played: frozenset[int],
        player: int,
        decl_id: int,
        play_history: tuple,
        seed: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Sample consistent opponent hands.

        For now, uses simple random sampling. Can be upgraded to use
        forge/eq/sampling_mrv_gpu.py for void-aware sampling.
        """
        import random
        rng = random.Random(seed)

        # Available dominoes (not in player's hand, not played)
        all_dominoes = set(range(28))
        available = all_dominoes - set(known_hand) - played

        # Compute remaining hand sizes
        n_played_per_player = [0, 0, 0, 0]
        for p, _, _ in play_history:
            n_played_per_player[p] += 1

        hand_sizes = [7 - n_played_per_player[p] for p in range(4)]

        # Sample hands for opponents
        available_list = list(available)
        rng.shuffle(available_list)

        hands = [None, None, None, None]
        hands[player] = tuple(d for d in known_hand if d not in played)

        idx = 0
        for p in range(4):
            if p == player:
                continue
            size = hand_sizes[p]
            hands[p] = tuple(available_list[idx:idx + size])
            idx += size

        return tuple(hands)


def select_action_mcts(
    state: GameState,
    player: int,
    n_determinizations: int = 10,
    n_simulations: int = 50,
    temperature: float = 1.0,
    seed: int = 0,
) -> tuple[int, dict[int, float]]:
    """Select action using MCTS with temperature-based sampling.

    Args:
        state: Current game state (with known hands - for self-play)
        player: Current player
        n_determinizations: Hand samples (1 for perfect info)
        n_simulations: MCTS iterations per sample
        temperature: Action sampling temperature (0 = greedy)
        seed: Random seed

    Returns:
        (selected_action, action_probabilities)
    """
    import random
    rng = random.Random(seed)

    mcts = MCTS(n_simulations=n_simulations)
    visits = mcts.search(state, player)

    # Convert to probabilities with temperature
    if temperature == 0:
        # Greedy
        best_action = max(visits.keys(), key=lambda a: visits[a])
        probs = {a: 1.0 if a == best_action else 0.0 for a in visits}
        return best_action, probs

    # Temperature-scaled sampling
    actions = list(visits.keys())
    counts = [visits[a] for a in actions]

    # Apply temperature
    if temperature != 1.0:
        counts = [c ** (1.0 / temperature) for c in counts]

    total = sum(counts)
    probs = {a: c / total for a, c in zip(actions, counts)}

    # Sample
    r = rng.random() * total
    cumsum = 0
    for a, c in zip(actions, counts):
        cumsum += c
        if r <= cumsum:
            return a, probs

    return actions[-1], probs
