"""MCTS-based self-play for AlphaZero-style training.

Instead of pure policy sampling (REINFORCE), uses MCTS to generate
training targets. The network learns to predict:
1. Policy: MCTS visit distribution
2. Value: Game outcome

This provides much lower variance than REINFORCE since MCTS
does the hard work of credit assignment via lookahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional

import torch
from torch import Tensor

from forge.eq.game import GameState
from forge.oracle.rng import deal_from_seed

from .mcts import MCTS, select_action_mcts
from .observation import observe, N_FEATURES, MAX_TOKENS, N_HAND_SLOTS
from .types import ZebGameState, GamePhase, BidState

# Type alias for value function
ValueFn = Callable[[GameState, int], float]


def _select_from_visits(
    visits: dict[int, int],
    temperature: float,
    rng,
) -> tuple[int, dict[int, float]]:
    """Convert visit counts to probabilities and sample an action.

    Args:
        visits: Dict of action -> visit count
        temperature: Sampling temperature (0 = greedy, 1 = proportional)
        rng: Random number generator

    Returns:
        (selected_action, action_probabilities)
    """
    actions = list(visits.keys())
    counts = [visits[a] for a in actions]

    # Apply temperature
    if temperature == 0:
        # Greedy
        best_idx = max(range(len(counts)), key=lambda i: counts[i])
        probs = {a: (1.0 if i == best_idx else 0.0) for i, a in enumerate(actions)}
        return actions[best_idx], probs

    # Temperature-scaled sampling
    if temperature != 1.0:
        counts = [c ** (1.0 / temperature) for c in counts]

    total = sum(counts)
    if total == 0:
        # Uniform fallback
        probs = {a: 1.0 / len(actions) for a in actions}
        return rng.choice(actions), probs

    probs = {a: c / total for a, c in zip(actions, counts)}

    # Sample
    r = rng.random()
    cumsum = 0
    for a, c in zip(actions, counts):
        cumsum += c / total
        if r <= cumsum:
            return a, probs

    return actions[-1], probs


@dataclass
class MCTSTrainingExample:
    """Single training example from MCTS self-play."""

    # Original dealt hands (fixed 7-tuple per player, never shrinks)
    original_hands: Tuple[Tuple[int, ...], ...]

    # Game state at this decision point
    played: frozenset
    play_history: Tuple[Tuple[int, int], ...]  # (player, domino_id) for observe()
    current_trick: Tuple[Tuple[int, int], ...]
    trick_leader: int
    decl_id: int

    # Current player
    player: int

    # MCTS output: target policy (visit distribution)
    action_probs: dict  # domino_id -> probability

    # Filled after game ends
    outcome: float = 0.0  # -1 to 1 for this player


@dataclass
class MCTSGame:
    """Complete game with MCTS training examples."""

    examples: List[MCTSTrainingExample] = field(default_factory=list)
    final_points: Tuple[int, int] = (0, 0)
    winner: int = 0


def play_game_with_mcts(
    seed: int,
    n_simulations: int = 100,
    temperature: float = 1.0,
    decl_id: Optional[int] = None,
    value_fn: Optional[ValueFn] = None,
) -> MCTSGame:
    """Play a single game using MCTS, collect training examples.

    Args:
        seed: Random seed for dealing
        n_simulations: MCTS simulations per move
        temperature: Action sampling temperature
        decl_id: Declaration ID (random if None)
        value_fn: Optional value function for leaf evaluation.
                 If None, uses random rollout.

    Returns:
        MCTSGame with examples and outcome
    """
    import random
    rng = random.Random(seed)

    # Deal hands - store original for observation encoding
    hands = deal_from_seed(seed)
    original_hands = tuple(tuple(h) for h in hands)

    if decl_id is None:
        decl_id = seed % 10

    # Random leader (normally bid winner, but we skip bidding)
    leader = rng.randint(0, 3)

    # Create initial state
    state = GameState.from_hands(hands, decl_id=decl_id, leader=leader)

    # Create MCTS instance (reused for all moves in this game)
    mcts = MCTS(n_simulations=n_simulations, value_fn=value_fn)

    game = MCTSGame()
    move_num = 0

    while not state.is_complete():
        player = state.current_player()

        # Run MCTS search
        visits = mcts.search(state, player)

        # Convert visits to action probabilities with temperature
        action, probs = _select_from_visits(visits, temperature, rng)

        # Convert play_history: (player, domino, lead) -> (player, domino)
        play_history = tuple((p, d) for p, d, _ in state.play_history)

        # Record training example with fixed original hands
        example = MCTSTrainingExample(
            original_hands=original_hands,
            played=state.played,
            play_history=play_history,
            current_trick=state.current_trick,
            trick_leader=state.leader,
            decl_id=state.decl_id,
            player=player,
            action_probs=probs,
        )
        game.examples.append(example)

        # Apply action
        state = state.apply_action(action)
        move_num += 1

    # Compute final points
    team0_points = 0
    team1_points = 0
    from forge.oracle.tables import DOMINO_COUNT_POINTS
    for p, domino_id, _ in state.play_history:
        points = DOMINO_COUNT_POINTS[domino_id]
        if p % 2 == 0:
            team0_points += points
        else:
            team1_points += points

    game.final_points = (team0_points, team1_points)
    game.winner = 0 if team0_points > team1_points else 1

    # Fill outcomes for all examples
    for ex in game.examples:
        player_team = ex.player % 2
        if player_team == game.winner:
            ex.outcome = 1.0
        else:
            ex.outcome = -1.0

    return game


def play_games_with_mcts(
    n_games: int,
    n_simulations: int = 100,
    temperature: float = 1.0,
    base_seed: int = 0,
    value_fn: Optional[ValueFn] = None,
) -> List[MCTSGame]:
    """Play multiple games with MCTS, collect training data.

    Args:
        n_games: Number of games to play
        n_simulations: MCTS simulations per move
        temperature: Action sampling temperature
        base_seed: Starting random seed
        value_fn: Optional value function for leaf evaluation.
                 If None, uses random rollout.

    Returns:
        List of MCTSGame with training examples
    """
    games = []
    for i in range(n_games):
        game = play_game_with_mcts(
            seed=base_seed + i,
            n_simulations=n_simulations,
            temperature=temperature,
            value_fn=value_fn,
        )
        games.append(game)
    return games


def mcts_examples_to_tensors(
    games: List[MCTSGame],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Convert MCTS games to training tensors.

    Returns:
        states: Tensor representation of game states
        legal_masks: Which actions are legal
        target_policies: MCTS visit distributions
        target_values: Game outcomes
        players: Which player is to move
    """
    # For now, use a simple state representation
    # This can be upgraded to match Zeb's observation format

    all_states = []
    all_legal_masks = []
    all_policies = []
    all_values = []
    all_players = []

    for game in games:
        for ex in game.examples:
            # Simple state encoding: flatten relevant info
            # In full implementation, use Zeb's observation encoding
            state_vec = _encode_state(ex)
            all_states.append(state_vec)

            # Legal actions mask (over 28 dominoes)
            legal_mask = torch.zeros(28, dtype=torch.bool)
            for d in ex.action_probs.keys():
                legal_mask[d] = True
            all_legal_masks.append(legal_mask)

            # Target policy (over 28 dominoes)
            policy = torch.zeros(28, dtype=torch.float32)
            for d, p in ex.action_probs.items():
                policy[d] = p
            all_policies.append(policy)

            # Target value
            all_values.append(ex.outcome)

            # Player
            all_players.append(ex.player)

    return (
        torch.stack(all_states),
        torch.stack(all_legal_masks),
        torch.stack(all_policies),
        torch.tensor(all_values, dtype=torch.float32),
        torch.tensor(all_players, dtype=torch.long),
    )


def _encode_state(ex: MCTSTrainingExample) -> Tensor:
    """Simple state encoding for initial experiments (DEPRECATED - use mcts_examples_to_zeb_tensors).

    Encodes:
    - Player's hand (28-dim binary)
    - Played dominoes (28-dim binary)
    - Current trick (28-dim binary + who led)
    - Declaration (10-dim one-hot)
    - Current player (4-dim one-hot)
    """
    state = torch.zeros(28 + 28 + 28 + 4 + 10 + 4, dtype=torch.float32)

    # Player's hand
    player = ex.player
    hand = [d for d in ex.original_hands[player] if d not in ex.played]
    for d in hand:
        state[d] = 1.0

    # Played dominoes
    for d in ex.played:
        state[28 + d] = 1.0

    # Current trick
    for p, d in ex.current_trick:
        state[56 + d] = 1.0
        state[84 + p] = 1.0  # Who played in trick

    # Declaration one-hot
    state[88 + ex.decl_id] = 1.0

    # Current player one-hot
    state[98 + player] = 1.0

    return state


def _example_to_zeb_state(ex: MCTSTrainingExample) -> ZebGameState:
    """Convert MCTSTrainingExample to ZebGameState for observation encoding."""
    # Minimal bid state (not used by observe, but needed for ZebGameState)
    bid_state = BidState(bids=(30, 0, 0, 0), high_bidder=0, high_bid=30)

    return ZebGameState(
        hands=ex.original_hands,
        dealer=0,  # Not used by observe
        phase=GamePhase.PLAYING,
        bid_state=bid_state,
        decl_id=ex.decl_id,
        bidder=0,  # Not used by observe
        played=ex.played,
        play_history=ex.play_history,
        current_trick=tuple(d for _, d in ex.current_trick),  # Extract domino IDs only
        trick_leader=ex.trick_leader,
        team_points=(0, 0),  # Not used by observe
    )


def mcts_examples_to_zeb_tensors(
    games: List[MCTSGame],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Convert MCTS games to Zeb-compatible training tensors.

    Uses the proper observation encoding with play history and slot-based actions.

    Returns:
        tokens: [N, MAX_TOKENS, N_FEATURES] observation tokens
        masks: [N, MAX_TOKENS] valid token masks
        hand_indices: [N, 7] indices of hand slots in token sequence
        hand_masks: [N, 7] which hand slots have unplayed dominoes
        target_policies: [N, 7] MCTS visit distributions over hand slots
        target_values: [N] game outcomes
    """
    all_tokens = []
    all_masks = []
    all_hand_indices = []
    all_hand_masks = []
    all_policies = []
    all_values = []

    for game in games:
        for ex in game.examples:
            # Convert to ZebGameState for observation
            zeb_state = _example_to_zeb_state(ex)
            player = ex.player

            # Get observation tensors
            tokens, mask, hand_indices = observe(zeb_state, player)
            all_tokens.append(tokens)
            all_masks.append(mask)
            all_hand_indices.append(hand_indices)

            # Build hand mask (which slots have unplayed dominoes)
            hand = ex.original_hands[player]
            hand_mask = torch.tensor(
                [d not in ex.played for d in hand],
                dtype=torch.bool,
            )
            all_hand_masks.append(hand_mask)

            # Convert domino_id probs to slot probs
            # Map: domino_id -> slot index in original_hands
            domino_to_slot = {d: slot for slot, d in enumerate(hand)}
            slot_probs = torch.zeros(N_HAND_SLOTS, dtype=torch.float32)
            for domino_id, prob in ex.action_probs.items():
                slot = domino_to_slot[domino_id]
                slot_probs[slot] = prob
            all_policies.append(slot_probs)

            # Target value
            all_values.append(ex.outcome)

    return (
        torch.stack(all_tokens),
        torch.stack(all_masks),
        torch.stack(all_hand_indices),
        torch.stack(all_hand_masks),
        torch.stack(all_policies),
        torch.tensor(all_values, dtype=torch.float32),
    )


def play_games_with_oracle(
    n_games: int,
    n_simulations: int = 50,
    temperature: float = 1.0,
    base_seed: int = 0,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
) -> List[MCTSGame]:
    """Play games using oracle-based MCTS for high-quality training data.

    This is a convenience function that loads the oracle and runs games.
    For many games, consider loading the oracle once and passing to
    play_games_with_mcts directly.

    Args:
        n_games: Number of games to play
        n_simulations: MCTS simulations per move
        temperature: Action sampling temperature
        base_seed: Starting random seed
        device: Device for oracle inference
        checkpoint_path: Path to oracle checkpoint (uses default if None)

    Returns:
        List of MCTSGame with training examples
    """
    from .oracle_value import create_oracle_value_fn

    value_fn = create_oracle_value_fn(
        checkpoint_path=checkpoint_path,
        device=device,
        compile=True,
    )

    return play_games_with_mcts(
        n_games=n_games,
        n_simulations=n_simulations,
        temperature=temperature,
        base_seed=base_seed,
        value_fn=value_fn,
    )
