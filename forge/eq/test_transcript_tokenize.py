"""
Tests for Stage 2 transcript tokenizer.

CRITICAL: Tests use TINY inputs (1-3 plays) with SHORT timeouts (< 5s).
This prevents hours wasted on slow tests.
"""

from __future__ import annotations

import pytest
import torch

from forge.eq.transcript_tokenize import (
    FEAT_COUNT,
    FEAT_DECL,
    FEAT_HIGH,
    FEAT_IS_DOUBLE,
    FEAT_IS_IN_HAND,
    FEAT_LOW,
    FEAT_PLAYER,
    FEAT_TOKEN_TYPE,
    TOKEN_TYPE_DECL,
    TOKEN_TYPE_HAND,
    TOKEN_TYPE_PLAY,
    tokenize_transcript,
)
from forge.oracle.declarations import DOUBLES_TRUMP, NOTRUMP


def test_empty_plays_just_hand_and_declaration():
    """Minimal case: declaration + hand, no plays yet."""
    my_hand = [0, 1, 2]  # Three dominoes: (0,0), (1,0), (1,1)
    plays = []
    decl_id = 5  # Fives
    current_player = 0

    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    # Should have: 1 decl token + 3 hand tokens = 4 total
    assert tokens.shape == (4, 8)

    # Token 0: Declaration
    assert tokens[0, FEAT_DECL] == 5
    assert tokens[0, FEAT_TOKEN_TYPE] == TOKEN_TYPE_DECL

    # Tokens 1-3: Hand
    for i in range(1, 4):
        assert tokens[i, FEAT_TOKEN_TYPE] == TOKEN_TYPE_HAND
        assert tokens[i, FEAT_PLAYER] == 0  # Me
        assert tokens[i, FEAT_IS_IN_HAND] == 1
        assert tokens[i, FEAT_DECL] == 5

    # Check first domino (0,0)
    assert tokens[1, FEAT_HIGH] == 0
    assert tokens[1, FEAT_LOW] == 0
    assert tokens[1, FEAT_IS_DOUBLE] == 1
    assert tokens[1, FEAT_COUNT] == 0  # No points

    # Check second domino (1,0)
    assert tokens[2, FEAT_HIGH] == 1
    assert tokens[2, FEAT_LOW] == 0
    assert tokens[2, FEAT_IS_DOUBLE] == 0

    # Check third domino (1,1)
    assert tokens[3, FEAT_HIGH] == 1
    assert tokens[3, FEAT_LOW] == 1
    assert tokens[3, FEAT_IS_DOUBLE] == 1


def test_one_play():
    """One play added to transcript."""
    my_hand = [0, 1]  # (0,0), (1,0)
    plays = [(2, 25)]  # Player 2 plays (6,4) = 10 points
    decl_id = DOUBLES_TRUMP
    current_player = 0

    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    # Should have: 1 decl + 2 hand + 1 play = 4 tokens
    assert tokens.shape == (4, 8)

    # Declaration
    assert tokens[0, FEAT_TOKEN_TYPE] == TOKEN_TYPE_DECL
    assert tokens[0, FEAT_DECL] == DOUBLES_TRUMP

    # Hand tokens
    assert tokens[1, FEAT_TOKEN_TYPE] == TOKEN_TYPE_HAND
    assert tokens[2, FEAT_TOKEN_TYPE] == TOKEN_TYPE_HAND

    # Play token
    assert tokens[3, FEAT_TOKEN_TYPE] == TOKEN_TYPE_PLAY
    assert tokens[3, FEAT_PLAYER] == 2  # Partner (relative from player 0)
    assert tokens[3, FEAT_IS_IN_HAND] == 0  # Not in hand
    assert tokens[3, FEAT_HIGH] == 6
    assert tokens[3, FEAT_LOW] == 4
    assert tokens[3, FEAT_IS_DOUBLE] == 0
    assert tokens[3, FEAT_COUNT] == 2  # (6,4) is 10 points


def test_relative_player_ids():
    """Player IDs should be relative to current_player."""
    my_hand = [0]  # Just one domino
    plays = [
        (1, 1),  # Absolute player 1 plays
        (2, 2),  # Absolute player 2 plays
        (3, 3),  # Absolute player 3 plays
        (0, 4),  # Absolute player 0 plays
    ]
    decl_id = 0
    current_player = 1  # We are player 1

    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    # Should have: 1 decl + 1 hand + 4 plays = 6 tokens
    assert tokens.shape == (6, 8)

    # From player 1's perspective:
    # Absolute 1 -> Relative 0 (me)
    # Absolute 2 -> Relative 1 (left opponent)
    # Absolute 3 -> Relative 2 (partner)
    # Absolute 0 -> Relative 3 (right opponent)

    play_tokens = tokens[2:]  # Skip decl and hand
    assert play_tokens[0, FEAT_PLAYER] == 0  # Abs 1 -> Rel 0
    assert play_tokens[1, FEAT_PLAYER] == 1  # Abs 2 -> Rel 1
    assert play_tokens[2, FEAT_PLAYER] == 2  # Abs 3 -> Rel 2
    assert play_tokens[3, FEAT_PLAYER] == 3  # Abs 0 -> Rel 3


def test_count_points():
    """Test point value encoding."""
    # Domino 20 is (5,5) = 10 points
    # Domino 25 is (6,4) = 10 points
    # Domino 15 is (5,0) = 5 points
    # Domino 0 is (0,0) = 0 points

    my_hand = [20, 25, 15, 0]
    plays = []
    decl_id = NOTRUMP
    current_player = 0

    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    # Count values should be encoded as: 0->0, 5->1, 10->2
    assert tokens[1, FEAT_COUNT] == 2  # (5,5) = 10 points
    assert tokens[2, FEAT_COUNT] == 2  # (6,4) = 10 points
    assert tokens[3, FEAT_COUNT] == 1  # (5,0) = 5 points
    assert tokens[4, FEAT_COUNT] == 0  # (0,0) = 0 points


def test_full_hand_no_plays():
    """Test with full 7-domino hand."""
    my_hand = [0, 1, 2, 3, 4, 5, 6]  # 7 dominoes
    plays = []
    decl_id = 3
    current_player = 2

    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    # 1 decl + 7 hand = 8 tokens
    assert tokens.shape == (8, 8)

    # All hand tokens should be from player 0 (me)
    for i in range(1, 8):
        assert tokens[i, FEAT_TOKEN_TYPE] == TOKEN_TYPE_HAND
        assert tokens[i, FEAT_PLAYER] == 0
        assert tokens[i, FEAT_IS_IN_HAND] == 1


def test_multiple_plays_different_players():
    """Test transcript with multiple plays from different players."""
    my_hand = [0, 1, 2]
    plays = [
        (0, 10),  # Player 0 leads
        (1, 11),  # Player 1 follows
        (2, 12),  # Player 2 follows
    ]
    decl_id = 4
    current_player = 0

    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    # 1 decl + 3 hand + 3 plays = 7 tokens
    assert tokens.shape == (7, 8)

    # Check play tokens (indices 4, 5, 6)
    play_tokens = tokens[4:]

    # All plays should have is_in_hand=0
    assert all(play_tokens[:, FEAT_IS_IN_HAND] == 0)

    # All plays should be PLAY type
    assert all(play_tokens[:, FEAT_TOKEN_TYPE] == TOKEN_TYPE_PLAY)

    # Check relative players (from player 0's perspective)
    assert play_tokens[0, FEAT_PLAYER] == 0  # Me
    assert play_tokens[1, FEAT_PLAYER] == 1  # Left
    assert play_tokens[2, FEAT_PLAYER] == 2  # Partner


def test_invalid_current_player():
    """Test validation of current_player parameter."""
    with pytest.raises(ValueError, match="current_player must be in"):
        tokenize_transcript([0], [], 0, -1)

    with pytest.raises(ValueError, match="current_player must be in"):
        tokenize_transcript([0], [], 0, 4)


def test_invalid_decl_id():
    """Test validation of decl_id parameter."""
    with pytest.raises(ValueError, match="decl_id must be in"):
        tokenize_transcript([0], [], -1, 0)

    with pytest.raises(ValueError, match="decl_id must be in"):
        tokenize_transcript([0], [], 10, 0)


def test_invalid_hand_dominoes():
    """Test validation of hand contents."""
    with pytest.raises(ValueError, match="invalid domino IDs"):
        tokenize_transcript([28], [], 0, 0)  # ID too high

    with pytest.raises(ValueError, match="invalid domino IDs"):
        tokenize_transcript([-1], [], 0, 0)  # Negative ID


def test_hand_too_large():
    """Test that hand can't exceed 7 dominoes."""
    my_hand = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 dominoes
    with pytest.raises(ValueError, match="too many dominoes"):
        tokenize_transcript(my_hand, [], 0, 0)


def test_tensor_dtype():
    """Ensure output is torch.long for embedding layers."""
    my_hand = [0, 1]
    plays = [(0, 2)]
    decl_id = 0
    current_player = 0

    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    assert tokens.dtype == torch.long


def test_all_declarations():
    """Smoke test: ensure all declaration IDs work."""
    my_hand = [0]
    plays = []
    current_player = 0

    for decl_id in range(10):  # 0-9 are valid
        tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)
        assert tokens[0, FEAT_DECL] == decl_id
        assert tokens[1, FEAT_DECL] == decl_id  # Propagated to hand token


def test_relative_player_rotation_player_2():
    """Test player ID rotation from player 2's perspective."""
    my_hand = [0]
    plays = [
        (0, 1),  # Absolute 0
        (1, 2),  # Absolute 1
        (2, 3),  # Absolute 2 (current)
        (3, 4),  # Absolute 3
    ]
    decl_id = 0
    current_player = 2  # We are player 2

    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    # From player 2's perspective:
    # Absolute 0 -> Relative 2 (partner)
    # Absolute 1 -> Relative 3 (right opponent)
    # Absolute 2 -> Relative 0 (me)
    # Absolute 3 -> Relative 1 (left opponent)

    play_tokens = tokens[2:]  # Skip decl and hand
    assert play_tokens[0, FEAT_PLAYER] == 2  # Abs 0 -> Rel 2
    assert play_tokens[1, FEAT_PLAYER] == 3  # Abs 1 -> Rel 3
    assert play_tokens[2, FEAT_PLAYER] == 0  # Abs 2 -> Rel 0
    assert play_tokens[3, FEAT_PLAYER] == 1  # Abs 3 -> Rel 1


def test_empty_hand():
    """Test with empty hand (all dominoes played)."""
    my_hand = []
    plays = [(0, 1), (1, 2)]
    decl_id = 5
    current_player = 1

    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    # 1 decl + 0 hand + 2 plays = 3 tokens
    assert tokens.shape == (3, 8)
    assert tokens[0, FEAT_TOKEN_TYPE] == TOKEN_TYPE_DECL
    assert tokens[1, FEAT_TOKEN_TYPE] == TOKEN_TYPE_PLAY
    assert tokens[2, FEAT_TOKEN_TYPE] == TOKEN_TYPE_PLAY
