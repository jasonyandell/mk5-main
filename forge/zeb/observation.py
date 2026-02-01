"""
Observation encoding for Zeb self-play system.

Reuses the Stage 2 tokenization format from forge/eq/transcript_tokenize.py.
Encodes game state from a player's perspective with imperfect information:
- Player sees own hand + play history
- Does NOT see opponent hands

Token structure:
    Position 0: Declaration token
    Positions 1-7: Player's remaining hand (7 slots, padded for played)
    Positions 8-35: Play history (0-28 plays)

8 features per token (same as Stage 2):
    FEAT_HIGH = 0        # High pip (0-6)
    FEAT_LOW = 1         # Low pip (0-6)
    FEAT_IS_DOUBLE = 2   # 1 if double
    FEAT_COUNT = 3       # 0=0pts, 1=5pts, 2=10pts
    FEAT_PLAYER = 4      # Relative player (0-3)
    FEAT_IS_IN_HAND = 5  # 1 if in hand, 0 if played
    FEAT_DECL = 6        # Declaration ID (0-9)
    FEAT_TOKEN_TYPE = 7  # 0=decl, 1=hand, 2=play
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
)

from .types import ZebGameState
from .game import legal_actions as game_legal_actions, current_player

# Re-export feature constants from transcript_tokenize for consistency
N_FEATURES = 8
MAX_TOKENS = 36  # 1 decl + 7 hand + 28 plays

# Feature indices (match Stage 2)
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

# Padding value for empty hand slots (must be 0 for valid embedding indices)
PAD_VALUE = 0

# Count value mapping
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}

# Number of hand slots (always 7, padded for played dominoes)
N_HAND_SLOTS = 7


def observe(state: ZebGameState, player: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Create observation from player's perspective.

    Uses fixed-size output with padding for consistent tensor shapes:
    - Always 36 tokens (1 decl + 7 hand slots + 28 play slots)
    - Hand slots preserve original slot positions (7 fixed slots)
    - Played dominoes are masked out but slot position is preserved
    - Play slots padded with zeros, mask indicates valid tokens

    All player IDs in the output are RELATIVE to the observing player:
        0 = me (observing player)
        1 = left opponent
        2 = partner
        3 = right opponent

    Args:
        state: Current game state (ZebGameState)
        player: Which player's perspective (0-3)

    Returns:
        tokens: [MAX_TOKENS, N_FEATURES] feature tensor (int64)
        mask: [MAX_TOKENS] valid token mask (bool, True for valid tokens)
        hand_indices: [N_HAND_SLOTS] indices into tokens for hand slots
    """
    if not (0 <= player < 4):
        raise ValueError(f"player must be in [0, 3], got {player}")

    # Initialize output tensors
    tokens = torch.zeros((MAX_TOKENS, N_FEATURES), dtype=torch.long)
    mask = torch.zeros(MAX_TOKENS, dtype=torch.bool)

    # Token 0: Declaration
    tokens[0, FEAT_DECL] = state.decl_id
    tokens[0, FEAT_TOKEN_TYPE] = TOKEN_TYPE_DECL
    mask[0] = True

    # Tokens 1-7: Hand slots (fixed 7 slots, preserving original positions)
    # In ZebGameState, hands are fixed-size tuples where we track played via state.played
    my_hand = state.hands[player]
    for slot in range(N_HAND_SLOTS):
        token_pos = 1 + slot
        domino_id = my_hand[slot]

        if domino_id not in state.played:
            # Domino still in hand - encode it
            _encode_domino(tokens, token_pos, domino_id, player=0, decl_id=state.decl_id)
            tokens[token_pos, FEAT_IS_IN_HAND] = 1
            tokens[token_pos, FEAT_TOKEN_TYPE] = TOKEN_TYPE_HAND
            mask[token_pos] = True
        else:
            # Domino already played - pad this slot
            tokens[token_pos] = PAD_VALUE
            mask[token_pos] = False

    # Tokens 8+: Play history
    play_offset = 1 + N_HAND_SLOTS  # Position 8

    # play_history already contains ALL plays including current trick
    # (each play is added to play_history when it's made)
    # play_history contains (player, domino_id) tuples
    for i, (abs_player, domino_id) in enumerate(state.play_history):
        token_pos = play_offset + i
        rel_player = (abs_player - player + 4) % 4
        _encode_domino(tokens, token_pos, domino_id, player=rel_player, decl_id=state.decl_id)
        tokens[token_pos, FEAT_IS_IN_HAND] = 0
        tokens[token_pos, FEAT_TOKEN_TYPE] = TOKEN_TYPE_PLAY
        mask[token_pos] = True

    # Hand indices: positions 1-7 in token array
    hand_indices = torch.arange(1, 1 + N_HAND_SLOTS, dtype=torch.long)

    return tokens, mask, hand_indices


def _encode_domino(
    tokens: Tensor,
    idx: int,
    domino_id: int,
    player: int,
    decl_id: int,
) -> None:
    """Encode a single domino into the token array at position idx.

    Args:
        tokens: Token array to write into
        idx: Position in sequence
        domino_id: Global domino ID (0-27)
        player: Relative player ID (0-3)
        decl_id: Declaration ID (0-9)
    """
    tokens[idx, FEAT_HIGH] = DOMINO_HIGH[domino_id]
    tokens[idx, FEAT_LOW] = DOMINO_LOW[domino_id]
    tokens[idx, FEAT_IS_DOUBLE] = 1 if DOMINO_IS_DOUBLE[domino_id] else 0
    tokens[idx, FEAT_COUNT] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[domino_id]]
    tokens[idx, FEAT_PLAYER] = player
    tokens[idx, FEAT_DECL] = decl_id


def get_legal_mask(state: ZebGameState, player: int) -> Tensor:
    """Get legal action mask for current player's hand slots.

    The mask corresponds to the 7 fixed hand slot positions. A slot is legal
    if it contains a domino that can be legally played according to follow-suit
    rules.

    Note: This only returns valid actions when it's the player's turn.
    When it's not the player's turn, all slots are False.

    Args:
        state: Current game state
        player: Which player's perspective (0-3)

    Returns:
        legal: [N_HAND_SLOTS] bool tensor, True for legal slot indices
    """
    if not (0 <= player < 4):
        raise ValueError(f"player must be in [0, 3], got {player}")

    # Build mask for each hand slot
    legal = torch.zeros(N_HAND_SLOTS, dtype=torch.bool)

    # Only current player has legal actions
    if current_player(state) != player:
        return legal

    # Get legal slot indices from game module
    legal_slots = set(game_legal_actions(state))

    for slot in range(N_HAND_SLOTS):
        if slot in legal_slots:
            legal[slot] = True

    return legal


def slot_to_domino(state: ZebGameState, player: int, slot: int) -> int:
    """Convert a hand slot index to a domino ID.

    Args:
        state: Current game state
        player: Which player (0-3)
        slot: Hand slot index (0-6)

    Returns:
        domino_id: The domino ID at that slot

    Raises:
        ValueError: If slot is out of range or domino already played
    """
    if not (0 <= slot < N_HAND_SLOTS):
        raise ValueError(f"Slot {slot} out of range [0, {N_HAND_SLOTS})")

    my_hand = state.hands[player]
    domino_id = my_hand[slot]

    if domino_id in state.played:
        raise ValueError(f"Domino at slot {slot} already played")

    return domino_id
