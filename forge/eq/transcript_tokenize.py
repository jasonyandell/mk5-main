"""
Tokenizer for Stage 2 (E[Q] training) - PUBLIC INFORMATION ONLY

Stage 2 sees only what's visible during actual play:
- Current player's hand (7 or fewer dominoes remaining)
- Complete play transcript (player, domino) pairs
- Declaration (trump type)

NO opponent hands - this is the key difference from Stage 1.

Token format:
    [decl_token, hand_tokens..., play_tokens...]

Each token is a feature vector with position-specific information.
"""

from __future__ import annotations

import torch
from torch import Tensor

from forge.oracle.declarations import N_DECLS
from forge.oracle.tables import DOMINO_COUNT_POINTS, DOMINO_HIGH, DOMINO_IS_DOUBLE, DOMINO_LOW


# =============================================================================
# Token Feature Dimensions
# =============================================================================

N_FEATURES = 8  # Feature vector size per token

# Feature indices
FEAT_HIGH = 0        # High pip (0-6)
FEAT_LOW = 1         # Low pip (0-6)
FEAT_IS_DOUBLE = 2   # 1 if double, 0 otherwise
FEAT_COUNT = 3       # Count value: 0=0pts, 1=5pts, 2=10pts
FEAT_PLAYER = 4      # Relative player ID (0=me, 1=left, 2=partner, 3=right)
FEAT_IS_IN_HAND = 5  # 1 if in current player's hand, 0 if played
FEAT_DECL = 6        # Declaration ID (0-9)
FEAT_TOKEN_TYPE = 7  # Token type: 0=decl, 1=hand, 2=play

# Count value mapping
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}

# Token types
TOKEN_TYPE_DECL = 0
TOKEN_TYPE_HAND = 1
TOKEN_TYPE_PLAY = 2

# Maximum sequence length (1 decl + 7 hand + 28 plays = 36)
MAX_TOKENS = 36


# =============================================================================
# Main Tokenization Function
# =============================================================================

def tokenize_transcript(
    my_hand: list[int],
    plays: list[tuple[int, int]],
    decl_id: int,
    current_player: int,
) -> Tensor:
    """Encode game state from current player's perspective.

    All player IDs in the output are RELATIVE to current_player:
        0 = me (current player)
        1 = left opponent
        2 = partner
        3 = right opponent

    Args:
        my_hand: Current player's remaining dominoes (global IDs 0-27)
        plays: All plays so far as (absolute_player, domino_id) tuples
        decl_id: Declaration ID (0-9)
        current_player: Absolute player ID (0-3) for perspective transformation

    Returns:
        Tensor of shape (seq_len, n_features) where:
            - seq_len = 1 + len(my_hand) + len(plays)
            - First token is declaration
            - Next tokens are current hand
            - Remaining tokens are plays in chronological order
    """
    if not (0 <= current_player < 4):
        raise ValueError(f"current_player must be in [0, 3], got {current_player}")
    if not (0 <= decl_id < N_DECLS):
        raise ValueError(f"decl_id must be in [0, {N_DECLS}), got {decl_id}")
    if not all(0 <= d < 28 for d in my_hand):
        raise ValueError(f"my_hand contains invalid domino IDs: {my_hand}")
    if len(my_hand) > 7:
        raise ValueError(f"my_hand has too many dominoes: {len(my_hand)}")

    # Calculate sequence length
    seq_len = 1 + len(my_hand) + len(plays)
    if seq_len > MAX_TOKENS:
        raise ValueError(f"Sequence too long: {seq_len} > {MAX_TOKENS}")

    # Initialize token array
    tokens = torch.zeros((seq_len, N_FEATURES), dtype=torch.long)
    idx = 0

    # Token 0: Declaration
    tokens[idx, FEAT_DECL] = decl_id
    tokens[idx, FEAT_TOKEN_TYPE] = TOKEN_TYPE_DECL
    idx += 1

    # Tokens 1..len(my_hand): Current hand
    for domino_id in my_hand:
        _encode_domino(tokens, idx, domino_id, player=0, decl_id=decl_id, is_in_hand=True)
        tokens[idx, FEAT_TOKEN_TYPE] = TOKEN_TYPE_HAND
        idx += 1

    # Remaining tokens: Plays
    for abs_player, domino_id in plays:
        # Convert absolute player to relative
        rel_player = (abs_player - current_player + 4) % 4

        _encode_domino(tokens, idx, domino_id, player=rel_player, decl_id=decl_id, is_in_hand=False)
        tokens[idx, FEAT_TOKEN_TYPE] = TOKEN_TYPE_PLAY
        idx += 1

    return tokens


def _encode_domino(
    tokens: Tensor,
    idx: int,
    domino_id: int,
    player: int,
    decl_id: int,
    is_in_hand: bool,
) -> None:
    """Encode a single domino into the token array at position idx.

    Args:
        tokens: Token array to write into
        idx: Position in sequence
        domino_id: Global domino ID (0-27)
        player: Relative player ID (0-3)
        decl_id: Declaration ID (0-9)
        is_in_hand: True if domino is in current player's hand
    """
    tokens[idx, FEAT_HIGH] = DOMINO_HIGH[domino_id]
    tokens[idx, FEAT_LOW] = DOMINO_LOW[domino_id]
    tokens[idx, FEAT_IS_DOUBLE] = 1 if DOMINO_IS_DOUBLE[domino_id] else 0
    tokens[idx, FEAT_COUNT] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[domino_id]]
    tokens[idx, FEAT_PLAYER] = player
    tokens[idx, FEAT_IS_IN_HAND] = 1 if is_in_hand else 0
    tokens[idx, FEAT_DECL] = decl_id
