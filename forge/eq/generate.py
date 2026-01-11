"""
E[Q] game generation for Stage 2 training.

Plays complete games using Stage 1 oracle with world sampling,
recording all decision points as training examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from forge.eq.game import GameState
from forge.eq.oracle import Stage1Oracle
from forge.eq.sampling import sample_consistent_worlds
from forge.eq.transcript_tokenize import tokenize_transcript
from forge.oracle.tables import can_follow, led_suit_for_lead_domino


@dataclass
class DecisionRecord:
    """One training example for Stage 2."""

    transcript_tokens: Tensor  # Stage 2 input (from tokenize_transcript)
    e_logits: Tensor  # (7,) target - averaged logits from oracle
    legal_mask: Tensor  # (7,) which actions were legal
    action_taken: int  # Which action was actually played


@dataclass
class GameRecord:
    """All decisions from one game."""

    decisions: list[DecisionRecord]
    # 28 decisions per game (4 players × 7 tricks)


def generate_eq_game(
    oracle: Stage1Oracle,
    hands: list[list[int]],  # Initial deal [hand0, hand1, hand2, hand3]
    decl_id: int,
    n_samples: int = 100,
) -> GameRecord:
    """Play one game, record all 28 decisions.

    For each decision point:
    1. Get current player's perspective
    2. Infer voids from play history (incrementally)
    3. Sample N consistent worlds
    4. Query oracle for Q-values on each world
    5. Average logits to get E[Q]
    6. Record decision with transcript tokens
    7. Play the best action (argmax of E[Q] over legal actions)

    Args:
        oracle: Stage 1 oracle for querying Q-values
        hands: Initial deal as [hand0, hand1, hand2, hand3]
        decl_id: Declaration ID (0-9)
        n_samples: Number of worlds to sample per decision (default 100)

    Returns:
        GameRecord with 28 DecisionRecords
    """
    # Store initial deal - oracle needs this for tokenization
    initial_deal = [list(hand) for hand in hands]

    # Precompute: which domino is at which (player, local_idx) in initial deal
    domino_to_initial_pos: dict[int, tuple[int, int]] = {}
    for p in range(4):
        for local_idx, domino_id in enumerate(initial_deal[p]):
            domino_to_initial_pos[domino_id] = (p, local_idx)

    game = GameState.from_hands(hands, decl_id, leader=0)
    decisions = []

    # Incremental void tracking
    voids: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    plays_processed = 0

    while not game.is_complete():
        player = game.current_player()
        my_hand = list(game.hands[player])

        # 1. Incrementally update voids from new plays only
        for play_idx in range(plays_processed, len(game.play_history)):
            play_player, domino_id, lead_domino_id = game.play_history[play_idx]
            led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)
            if not can_follow(domino_id, led_suit, decl_id):
                voids[play_player].add(led_suit)
        plays_processed = len(game.play_history)

        # 2. Sample consistent worlds (remaining hands only)
        remaining_worlds = sample_consistent_worlds(
            my_player=player,
            my_hand=my_hand,
            played=game.played,
            hand_sizes=game.hand_sizes(),
            voids=voids,
            decl_id=decl_id,
            n_samples=n_samples,
        )

        # 3. Query oracle for each world (pass initial_deal directly)
        # Build game_state_info efficiently using precomputed lookup
        game_state_info = _build_game_state_info_fast(
            game, remaining_worlds, initial_deal, domino_to_initial_pos, player
        )
        # Pass same initial_deal for all worlds (current player's hand is constant)
        logits = oracle.query_batch(
            [initial_deal] * n_samples, game_state_info, player
        )  # (N, 7)

        # 4. Average to get E[Q]
        e_logits = logits.mean(dim=0)  # (7,)

        # 5. Build transcript tokens for Stage 2 training
        plays_for_transcript = [(p, d) for p, d, _ in game.play_history]
        transcript_tokens = tokenize_transcript(my_hand, plays_for_transcript, decl_id, player)

        # 6. Determine legal actions and select best
        legal_actions = game.legal_actions()
        legal_mask = _build_legal_mask(legal_actions, my_hand)  # (7,) boolean mask

        # Mask illegal actions and select
        masked_logits = e_logits.clone()
        masked_logits[~legal_mask] = float("-inf")
        action_idx = masked_logits.argmax().item()
        action_domino = my_hand[action_idx]

        # 7. Record decision
        decisions.append(
            DecisionRecord(
                transcript_tokens=transcript_tokens,
                e_logits=e_logits,
                legal_mask=legal_mask,
                action_taken=action_idx,
            )
        )

        # 8. Apply action
        game = game.apply_action(action_domino)

    return GameRecord(decisions=decisions)


def _build_game_state_info_fast(
    game: GameState,
    remaining_worlds: list[list[list[int]]],
    initial_deal: list[list[int]],
    domino_to_initial_pos: dict[int, tuple[int, int]],
    current_player: int,
) -> dict:
    """Extract game state information for oracle query (optimized version).

    Key optimizations:
    1. Uses precomputed domino_to_initial_pos lookup instead of list.index()
    2. Builds remaining bitmasks directly from remaining_worlds (no reconstruction)
    3. For current player, bitmask uses actual initial_deal (constant across worlds)

    Args:
        game: Current game state
        remaining_worlds: List of N worlds, each with 4 hands of remaining dominoes
        initial_deal: The actual initial deal (constant, known for current player)
        domino_to_initial_pos: Precomputed {domino_id: (player, local_idx)} lookup
        current_player: Current player making the decision

    Returns:
        Dict with keys: decl_id, leader, trick_plays, remaining
    """
    # Extract current trick as (player, local_idx) tuples
    # local_idx refers to position in the INITIAL hand
    trick_plays = []
    for play_player, domino_id in game.current_trick:
        # Use precomputed lookup instead of list.index()
        _, local_idx = domino_to_initial_pos[domino_id]
        trick_plays.append((play_player, local_idx))

    # Build remaining bitmasks for each world
    # Bitmask indicates which dominoes from initial hand are still available
    n_worlds = len(remaining_worlds)
    remaining = np.zeros((n_worlds, 4), dtype=np.int32)

    for world_idx, remaining_hands in enumerate(remaining_worlds):
        # For current player: use actual initial_deal to build bitmask
        for local_idx, domino_id in enumerate(initial_deal[current_player]):
            if domino_id not in game.played:
                remaining[world_idx, current_player] |= 1 << local_idx

        # For opponents: build bitmask from their remaining dominoes
        for player_idx in range(4):
            if player_idx == current_player:
                continue

            # Get opponent's remaining dominoes from sampled world
            for domino_id in remaining_hands[player_idx]:
                # Find where this domino was in their initial hand
                initial_player, local_idx = domino_to_initial_pos[domino_id]
                # Sanity check - should match player_idx
                if initial_player == player_idx:
                    remaining[world_idx, player_idx] |= 1 << local_idx

    return {
        "decl_id": game.decl_id,
        "leader": game.leader,
        "trick_plays": trick_plays,
        "remaining": remaining,
    }


def _build_legal_mask(legal_actions: tuple[int, ...], hand: list[int]) -> Tensor:
    """Build boolean mask of legal actions.

    Args:
        legal_actions: Tuple of legal domino IDs
        hand: Current hand as list of domino IDs

    Returns:
        Boolean tensor of shape (7,) where True means legal
    """
    # Pad hand to length 7 if needed
    padded_hand = hand + [-1] * (7 - len(hand))

    legal_set = set(legal_actions)
    mask = torch.tensor([domino in legal_set for domino in padded_hand], dtype=torch.bool)

    return mask
