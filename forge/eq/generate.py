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
    4. Reconstruct hypothetical initial hands for each world
    5. Query oracle for Q-values on each hypothetical world
    6. Average logits to get E[Q]
    7. Record decision with transcript tokens
    8. Play the best action (argmax of E[Q] over legal actions)

    Args:
        oracle: Stage 1 oracle for querying Q-values
        hands: Initial deal as [hand0, hand1, hand2, hand3]
        decl_id: Declaration ID (0-9)
        n_samples: Number of worlds to sample per decision (default 100)

    Returns:
        GameRecord with 28 DecisionRecords
    """
    game = GameState.from_hands(hands, decl_id, leader=0)
    decisions = []

    # Incremental void tracking
    voids: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    plays_processed = 0

    # Track which dominoes each player has played (for reconstructing initial hands)
    played_by: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}

    while not game.is_complete():
        player = game.current_player()
        my_hand = list(game.hands[player])

        # 1. Incrementally update voids and played_by from new plays
        for play_idx in range(plays_processed, len(game.play_history)):
            play_player, domino_id, lead_domino_id = game.play_history[play_idx]
            played_by[play_player].append(domino_id)
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

        # 3. Reconstruct hypothetical initial hands and query oracle for each world
        hypothetical_deals, game_state_info = _build_hypothetical_worlds(
            game, remaining_worlds, played_by, player
        )
        logits = oracle.query_batch(
            hypothetical_deals, game_state_info, player
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


def _build_hypothetical_worlds(
    game: GameState,
    remaining_worlds: list[list[list[int]]],
    played_by: dict[int, list[int]],
    current_player: int,
) -> tuple[list[list[list[int]]], dict]:
    """Reconstruct hypothetical initial hands for each sampled world.

    For proper E[Q] marginalization, we need to query the oracle on HYPOTHETICAL
    worlds, not the true game state. Each sampled world gives us remaining hands
    for opponents. We reconstruct initial hands as:
        initial[p] = remaining[p] + played_by[p]

    Args:
        game: Current game state
        remaining_worlds: List of N worlds, each with 4 hands of remaining dominoes
        played_by: Dict mapping player -> list of dominoes they've played
        current_player: Current player making the decision

    Returns:
        Tuple of (hypothetical_deals, game_state_info) where:
        - hypothetical_deals: List of N initial deals, one per world
        - game_state_info: Dict with decl_id, leader, trick_plays, remaining
    """
    n_worlds = len(remaining_worlds)
    hypothetical_deals = []
    remaining_bitmasks = np.zeros((n_worlds, 4), dtype=np.int32)

    # We need trick_plays indexed by local position in each world's initial hand
    # Since initial hands vary per world, we compute trick_plays per world
    # But oracle expects a single trick_plays list, so we use the first world's indexing
    # (All worlds have same current_player hand, so trick_plays for current_player are consistent)

    first_world_trick_plays = None

    for world_idx, remaining_hands in enumerate(remaining_worlds):
        # Reconstruct initial hands: initial[p] = remaining[p] + played_by[p]
        initial_hands = []
        for p in range(4):
            initial = list(remaining_hands[p]) + list(played_by[p])
            initial.sort()  # Sort for consistent ordering
            initial_hands.append(initial)

        hypothetical_deals.append(initial_hands)

        # Build domino -> local_idx lookup for this world
        domino_to_local: dict[int, int] = {}
        for p in range(4):
            for local_idx, domino in enumerate(initial_hands[p]):
                domino_to_local[(p, domino)] = local_idx

        # Build remaining bitmask for this world
        for p in range(4):
            for local_idx, domino in enumerate(initial_hands[p]):
                if domino not in game.played:
                    remaining_bitmasks[world_idx, p] |= 1 << local_idx

        # Build trick_plays for this world (use first world's as representative)
        if first_world_trick_plays is None:
            first_world_trick_plays = []
            for play_player, domino_id in game.current_trick:
                local_idx = domino_to_local.get((play_player, domino_id))
                if local_idx is not None:
                    first_world_trick_plays.append((play_player, local_idx))

    game_state_info = {
        "decl_id": game.decl_id,
        "leader": game.leader,
        "trick_plays": first_world_trick_plays or [],
        "remaining": remaining_bitmasks,
    }

    return hypothetical_deals, game_state_info


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
