"""Cross-game batched MCTS for improved GPU utilization.

Instead of running one game at a time (sequential), this maintains N active
MCTS searches and collects leaves from ALL of them together for batch
evaluation. This dramatically improves GPU utilization:

- Before: 32 leaves/batch from 1 game → 40% GPU utilization
- After: 500+ leaves/batch from 16 games → 80%+ GPU utilization

The key insight is that leaf collection is CPU-bound (tree traversal, state
cloning), so by running multiple games in parallel we can overlap CPU and
GPU work more effectively.

Additional optimization: async double-buffering overlaps CPU preprocessing
(leaf collection) with GPU evaluation using separate CUDA streams. While
GPU evaluates batch N, CPU prepares batch N+1.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

import torch

from forge.eq.game import GameState
from forge.oracle.rng import deal_from_seed

from .mcts import MCTS, MCTSNode
from .mcts_self_play import MCTSGame, MCTSTrainingExample, _select_from_visits


@dataclass
class PendingLeaf:
    """A leaf node pending evaluation, with its backpropagation path."""

    game_idx: int  # Which game this leaf belongs to
    node: MCTSNode  # The leaf node
    path: List[MCTSNode]  # Path from root to leaf (for backprop)
    is_terminal: bool  # Whether this is a terminal node


@dataclass
class _PendingGPUBatch:
    """State for a GPU batch that's being evaluated asynchronously.

    Used for double-buffering: while GPU processes this batch, CPU prepares
    the next batch. After GPU sync, backpropagation uses these stored values.
    """

    eval_leaves: List[PendingLeaf]  # Leaves being evaluated
    values: List[float]  # GPU-computed values (valid after stream sync)
    pending_leaves: List[PendingLeaf]  # All leaves for sim count tracking
    active_games_snapshot: List[ActiveGame]  # Game references at batch creation


@dataclass
class ActiveGame:
    """An in-progress game being played with MCTS."""

    seed: int
    original_hands: tuple  # Fixed original hands for training examples
    state: GameState  # Current game state
    mcts_root: MCTSNode  # Current MCTS root
    player: int  # Current player (for value perspective)
    sims_done: int = 0  # Simulations completed for current move
    move_num: int = 0  # Current move number
    examples: List[MCTSTrainingExample] = field(default_factory=list)

    # MCTS config
    n_simulations: int = 100
    c_puct: float = 1.414


def _select_child(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Select child with highest UCB score."""
    best_score = float('-inf')
    best_child = None

    for child in node.children.values():
        score = child.ucb_score(c_puct)
        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def _expand(node: MCTSNode) -> MCTSNode:
    """Expand node by adding one unexplored child."""
    legal = node.state.legal_actions()

    for action in legal:
        if action not in node.children:
            new_state = node.state.apply_action(action)
            child = MCTSNode(
                state=new_state,
                parent=node,
                action=action,
                prior=1.0 / len(legal),
            )
            node.children[action] = child
            return child

    return node


def _terminal_value(state: GameState, player: int) -> float:
    """Compute value for terminal state."""
    from forge.oracle.tables import DOMINO_COUNT_POINTS

    team0_points = 0
    team1_points = 0

    for p, domino_id, _ in state.play_history:
        points = DOMINO_COUNT_POINTS[domino_id]
        if p % 2 == 0:
            team0_points += points
        else:
            team1_points += points

    player_team = player % 2
    my_points = team0_points if player_team == 0 else team1_points
    opp_points = team1_points if player_team == 0 else team0_points

    if my_points > opp_points:
        return 1.0
    elif my_points < opp_points:
        return -1.0
    return 0.0


def _backpropagate_value_only(
    path: List[MCTSNode],
    value: float,
    root_player: int,
):
    """Update value_sum only (visits were incremented during virtual loss)."""
    for node in path:
        if node.parent is not None:
            parent_player = node.parent.state.current_player()
            if parent_player % 2 == root_player % 2:
                node.value_sum += value
            else:
                node.value_sum -= value
        else:
            node.value_sum += value


class BatchedMCTSCoordinator:
    """Coordinates multiple MCTS games for cross-game leaf batching.

    Maintains N active games simultaneously, collecting leaves from all
    of them together before sending to the oracle for batch evaluation.
    This maximizes GPU utilization by sending larger batches.
    """

    def __init__(
        self,
        n_parallel_games: int = 16,
        target_batch_size: int = 512,
        n_simulations: int = 100,
        temperature: float = 1.0,
        c_puct: float = 1.414,
        value_fn: Optional[Callable] = None,
    ):
        """
        Args:
            n_parallel_games: Number of concurrent MCTS games
            target_batch_size: Target batch size for oracle evaluation
            n_simulations: MCTS simulations per move
            temperature: Action sampling temperature
            c_puct: Exploration constant for UCB
            value_fn: Value function with batch_evaluate() method
        """
        self.n_parallel_games = n_parallel_games
        self.target_batch_size = target_batch_size
        self.n_simulations = n_simulations
        self.temperature = temperature
        self.c_puct = c_puct
        self.value_fn = value_fn

        # Derived: leaves per game per batch
        self.wave_size = max(1, target_batch_size // n_parallel_games)

    def play_games(
        self,
        n_games: int,
        base_seed: int = 0,
    ) -> List[MCTSGame]:
        """Play n_games using cross-game batched MCTS with async double-buffering.

        Uses CUDA streams to overlap CPU preprocessing (leaf collection) with
        GPU evaluation. While GPU processes batch N, CPU prepares batch N+1.

        Double-buffering pattern:
        1. Collect batch 0 (CPU)
        2. Launch batch 0 on GPU (async)
        3. While GPU works: Collect batch 1 (CPU) <-- OVERLAP
        4. Sync batch 0, backprop, process completions
        5. Launch batch 1 on GPU (async)
        6. While GPU works: Collect batch 2 (CPU) <-- OVERLAP
        7. ... repeat

        Args:
            n_games: Total number of games to play
            base_seed: Starting random seed

        Returns:
            List of completed MCTSGame with training examples
        """
        completed_games: List[MCTSGame] = []
        active_games: List[ActiveGame] = []
        next_seed = base_seed
        games_started = 0

        # Initialize first batch of games
        n_initial = min(self.n_parallel_games, n_games)
        for _ in range(n_initial):
            game = self._create_game(next_seed)
            active_games.append(game)
            next_seed += 1
            games_started += 1

        # Create CUDA stream for async GPU execution
        use_cuda = torch.cuda.is_available() and self.value_fn is not None
        gpu_stream = torch.cuda.Stream() if use_cuda else None

        # Double-buffer state: info about the in-flight GPU batch
        inflight_batch: Optional[_PendingGPUBatch] = None

        # Main loop: process until all games complete
        while active_games:
            # Collect leaves from all active games (CPU work)
            pending_leaves = self._collect_leaves(active_games)

            # Separate terminal vs non-terminal leaves
            terminal_leaves = [p for p in pending_leaves if p.is_terminal]
            eval_leaves = [p for p in pending_leaves if not p.is_terminal]

            # Evaluate terminals immediately (CPU only)
            for pending in terminal_leaves:
                game = active_games[pending.game_idx]
                value = _terminal_value(pending.node.state, game.player)
                _backpropagate_value_only(pending.path, value, game.player)

            # Prepare GPU batch data (CPU work - can overlap with previous GPU)
            gpu_batch_data = None
            if eval_leaves and self.value_fn is not None:
                states = [p.node.state for p in eval_leaves]
                players = [active_games[p.game_idx].player for p in eval_leaves]
                original_hands_list = [
                    list(list(h) for h in active_games[p.game_idx].original_hands)
                    for p in eval_leaves
                ]
                gpu_batch_data = (states, players, original_hands_list)

            # If there's an in-flight GPU batch, wait for it and backprop
            if inflight_batch is not None:
                gpu_stream.synchronize()

                # Backpropagate results from completed GPU batch
                for pending, value in zip(inflight_batch.eval_leaves, inflight_batch.values):
                    game = inflight_batch.active_games_snapshot[pending.game_idx]
                    _backpropagate_value_only(pending.path, value, game.player)

                # Process completions for the previous batch
                self._process_completions(
                    active_games,
                    inflight_batch.pending_leaves,
                    completed_games,
                )

                # Spawn new games to maintain parallelism
                while len(active_games) < self.n_parallel_games and games_started < n_games:
                    game = self._create_game(next_seed)
                    active_games.append(game)
                    next_seed += 1
                    games_started += 1

                inflight_batch = None

            # Launch GPU evaluation for current batch
            if gpu_batch_data is not None:
                states, players, original_hands_list = gpu_batch_data

                if gpu_stream is not None:
                    # Launch GPU work asynchronously
                    with torch.cuda.stream(gpu_stream):
                        if hasattr(self.value_fn, 'batch_evaluate_with_originals'):
                            values = self.value_fn.batch_evaluate_with_originals(
                                states, players, original_hands_list
                            )
                        else:
                            values = self.value_fn.batch_evaluate(states, players)

                    # Store as in-flight batch (will be processed next iteration)
                    inflight_batch = _PendingGPUBatch(
                        eval_leaves=eval_leaves,
                        values=values,
                        pending_leaves=pending_leaves,
                        active_games_snapshot=list(active_games),
                    )
                    # Continue to next iteration - GPU works while we collect more leaves
                else:
                    # No CUDA - synchronous evaluation
                    if hasattr(self.value_fn, 'batch_evaluate_with_originals'):
                        values = self.value_fn.batch_evaluate_with_originals(
                            states, players, original_hands_list
                        )
                    else:
                        values = self.value_fn.batch_evaluate(states, players)

                    for pending, value in zip(eval_leaves, values):
                        game = active_games[pending.game_idx]
                        _backpropagate_value_only(pending.path, value, game.player)

                    # Process completions immediately
                    self._process_completions(
                        active_games, pending_leaves, completed_games
                    )

                    # Spawn new games
                    while len(active_games) < self.n_parallel_games and games_started < n_games:
                        game = self._create_game(next_seed)
                        active_games.append(game)
                        next_seed += 1
                        games_started += 1
            else:
                # No GPU eval needed - just process completions for terminal-only batch
                self._process_completions(
                    active_games, pending_leaves, completed_games
                )

                # Spawn new games
                while len(active_games) < self.n_parallel_games and games_started < n_games:
                    game = self._create_game(next_seed)
                    active_games.append(game)
                    next_seed += 1
                    games_started += 1

        # Handle final in-flight batch if any
        if inflight_batch is not None and gpu_stream is not None:
            gpu_stream.synchronize()
            for pending, value in zip(inflight_batch.eval_leaves, inflight_batch.values):
                game = inflight_batch.active_games_snapshot[pending.game_idx]
                _backpropagate_value_only(pending.path, value, game.player)

            # Process final completions (active_games may be empty now)
            # Use snapshot since active_games might have been cleared
            self._process_completions(
                list(inflight_batch.active_games_snapshot),
                inflight_batch.pending_leaves,
                completed_games,
            )

        return completed_games

    def _process_completions(
        self,
        active_games: List[ActiveGame],
        pending_leaves: List[PendingLeaf],
        completed_games: List[MCTSGame],
    ) -> None:
        """Update simulation counts and handle move/game completions."""
        games_to_remove = []
        for idx, game in enumerate(active_games):
            # Count how many leaves were processed for this game
            n_processed = sum(
                1 for p in pending_leaves if p.game_idx == idx
            )
            game.sims_done += n_processed

            # Check if current move's MCTS is done
            if game.sims_done >= self.n_simulations:
                self._complete_move(game)

                # Check if game is complete
                if game.state.is_complete():
                    completed_game = self._finalize_game(game)
                    completed_games.append(completed_game)
                    games_to_remove.append(idx)
                else:
                    # Start next move
                    game.player = game.state.current_player()
                    game.mcts_root = MCTSNode(state=game.state)
                    game.sims_done = 0
                    game.move_num += 1

        # Remove completed games (reverse order to preserve indices)
        for idx in reversed(games_to_remove):
            active_games.pop(idx)

    def _create_game(self, seed: int) -> ActiveGame:
        """Create a new active game."""
        rng = random.Random(seed)
        hands = deal_from_seed(seed)
        original_hands = tuple(tuple(h) for h in hands)
        decl_id = seed % 10
        leader = rng.randint(0, 3)

        state = GameState.from_hands(hands, decl_id=decl_id, leader=leader)
        player = state.current_player()

        return ActiveGame(
            seed=seed,
            original_hands=original_hands,
            state=state,
            mcts_root=MCTSNode(state=state),
            player=player,
            n_simulations=self.n_simulations,
            c_puct=self.c_puct,
        )

    def _collect_leaves(self, active_games: List[ActiveGame]) -> List[PendingLeaf]:
        """Collect leaves from all active games using virtual loss.

        Each game contributes up to wave_size leaves, for a total batch
        size of approximately n_parallel_games * wave_size.
        """
        pending: List[PendingLeaf] = []

        for game_idx, game in enumerate(active_games):
            # How many more simulations does this game need?
            sims_remaining = self.n_simulations - game.sims_done
            wave_count = min(self.wave_size, sims_remaining)

            for _ in range(wave_count):
                node = game.mcts_root
                path: List[MCTSNode] = []

                # Selection with virtual loss
                while node.is_expanded() and not node.is_terminal():
                    node.visits += 1  # Virtual loss
                    path.append(node)
                    node = _select_child(node, game.c_puct)

                # Expansion
                if not node.is_terminal():
                    path.append(node)
                    node.visits += 1  # Virtual loss on parent
                    node = _expand(node)

                # Virtual loss on leaf
                node.visits += 1
                path.append(node)

                pending.append(PendingLeaf(
                    game_idx=game_idx,
                    node=node,
                    path=path,
                    is_terminal=node.is_terminal(),
                ))

        return pending

    def _complete_move(self, game: ActiveGame):
        """Complete the current move: record example, apply action."""
        # Get visit counts from root's children
        visits = {
            action: child.visits
            for action, child in game.mcts_root.children.items()
        }

        # Sample action with temperature
        rng = random.Random(game.seed + game.move_num)
        action, probs = _select_from_visits(visits, self.temperature, rng)

        # Record training example
        play_history = tuple((p, d) for p, d, _ in game.state.play_history)
        example = MCTSTrainingExample(
            original_hands=game.original_hands,
            played=game.state.played,
            play_history=play_history,
            current_trick=game.state.current_trick,
            trick_leader=game.state.leader,
            decl_id=game.state.decl_id,
            player=game.player,
            action_probs=probs,
        )
        game.examples.append(example)

        # Apply action
        game.state = game.state.apply_action(action)

    def _finalize_game(self, game: ActiveGame) -> MCTSGame:
        """Finalize a completed game: compute outcomes, create MCTSGame."""
        from forge.oracle.tables import DOMINO_COUNT_POINTS

        team0_points = 0
        team1_points = 0

        for p, domino_id, _ in game.state.play_history:
            points = DOMINO_COUNT_POINTS[domino_id]
            if p % 2 == 0:
                team0_points += points
            else:
                team1_points += points

        winner = 0 if team0_points > team1_points else 1

        # Fill outcomes for all examples
        for ex in game.examples:
            player_team = ex.player % 2
            if player_team == winner:
                ex.outcome = 1.0
            else:
                ex.outcome = -1.0

        return MCTSGame(
            examples=game.examples,
            final_points=(team0_points, team1_points),
            winner=winner,
        )


def play_games_batched(
    n_games: int,
    n_simulations: int = 100,
    temperature: float = 1.0,
    base_seed: int = 0,
    value_fn: Optional[Callable] = None,
    n_parallel_games: int = 16,
    target_batch_size: int = 512,
) -> List[MCTSGame]:
    """Play multiple games with cross-game batched MCTS.

    Drop-in replacement for play_games_with_mcts() that uses cross-game
    batching for improved GPU utilization.

    Args:
        n_games: Number of games to play
        n_simulations: MCTS simulations per move
        temperature: Action sampling temperature
        base_seed: Starting random seed
        value_fn: Value function with batch_evaluate() method
        n_parallel_games: Number of concurrent MCTS games
        target_batch_size: Target batch size for oracle evaluation

    Returns:
        List of MCTSGame with training examples
    """
    coordinator = BatchedMCTSCoordinator(
        n_parallel_games=n_parallel_games,
        target_batch_size=target_batch_size,
        n_simulations=n_simulations,
        temperature=temperature,
        value_fn=value_fn,
    )
    return coordinator.play_games(n_games, base_seed)
