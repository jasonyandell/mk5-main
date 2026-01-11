"""
Tests for Stage 1 Oracle.

Testing strategy:
1. Mock-based unit tests (fast, < 5s)
2. Tokenization correctness tests
3. Integration test with real checkpoint (optional, slower)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from forge.eq.oracle import Stage1Oracle
from forge.oracle.rng import deal_from_seed


# =============================================================================
# Mock helpers
# =============================================================================

class MockDominoLightningModule:
    """Mock Lightning module for testing."""

    def __init__(self):
        self.eval_called = False
        self.to_called = False
        self.device_arg = None

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.to_called = True
        self.device_arg = device
        return self

    def __call__(self, tokens, masks, current_player):
        """Return dummy logits and values."""
        batch_size = tokens.shape[0]
        device = tokens.device

        # Return logits shape (batch, 7) and value shape (batch,)
        logits = torch.randn(batch_size, 7, device=device)
        value = torch.randn(batch_size, device=device)
        return logits, value


@pytest.fixture
def mock_checkpoint():
    """Mock checkpoint loading to avoid filesystem dependency."""
    mock_module = MockDominoLightningModule()

    with patch('forge.eq.oracle.DominoLightningModule') as mock_class:
        mock_class.load_from_checkpoint.return_value = mock_module
        yield mock_class, mock_module


# =============================================================================
# Unit Tests (mocked, fast)
# =============================================================================

def test_oracle_initialization(mock_checkpoint):
    """Test oracle loads checkpoint and sets eval mode."""
    mock_class, mock_module = mock_checkpoint

    oracle = Stage1Oracle("fake_checkpoint.ckpt", device="cpu")

    # Verify checkpoint was loaded
    mock_class.load_from_checkpoint.assert_called_once_with(
        "fake_checkpoint.ckpt",
        map_location="cpu"
    )

    # Verify eval() and to() were called
    assert mock_module.eval_called
    assert mock_module.to_called
    assert mock_module.device_arg == "cpu"


def test_oracle_query_batch_shape(mock_checkpoint):
    """Test query_batch returns correct shape."""
    _, _ = mock_checkpoint

    oracle = Stage1Oracle("fake_checkpoint.ckpt", device="cpu")

    # Create 10 sampled worlds
    worlds = [deal_from_seed(i) for i in range(10)]

    # Create game state info
    remaining = np.ones((10, 4), dtype=np.int64) * 0x7F  # All 7 dominoes remaining
    game_state_info = {
        'decl_id': 3,
        'leader': 0,
        'trick_plays': [],
        'remaining': remaining,
    }

    # Query
    logits = oracle.query_batch(
        worlds=worlds,
        game_state_info=game_state_info,
        current_player=0,
    )

    # Verify shape
    assert logits.shape == (10, 7), f"Expected (10, 7), got {logits.shape}"
    assert logits.dtype == torch.float32


def test_oracle_empty_worlds_raises(mock_checkpoint):
    """Test that empty worlds list raises ValueError."""
    _, _ = mock_checkpoint

    oracle = Stage1Oracle("fake_checkpoint.ckpt", device="cpu")

    with pytest.raises(ValueError, match="worlds cannot be empty"):
        oracle.query_batch(
            worlds=[],
            game_state_info={'decl_id': 0, 'leader': 0, 'remaining': np.array([])},
            current_player=0,
        )


# =============================================================================
# Tokenization Tests
# =============================================================================

def test_tokenize_worlds_shapes():
    """Test tokenization produces correct shapes."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)  # Create without __init__
    oracle.device = "cpu"

    # Create 2 worlds
    worlds = [deal_from_seed(0), deal_from_seed(1)]
    remaining = np.ones((2, 4), dtype=np.int64) * 0x7F

    tokens, masks = oracle._tokenize_worlds(
        worlds=worlds,
        decl_id=3,
        leader=0,
        trick_plays=[],
        remaining=remaining,
        current_player=0,
    )

    # Verify shapes
    assert tokens.shape == (2, 32, 12), f"Expected (2, 32, 12), got {tokens.shape}"
    assert masks.shape == (2, 32), f"Expected (2, 32), got {masks.shape}"
    assert tokens.dtype == np.int8
    assert masks.dtype == np.int8


def test_tokenize_context_token():
    """Test context token is populated correctly."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    worlds = [deal_from_seed(0)]
    remaining = np.ones((1, 4), dtype=np.int64) * 0x7F

    tokens, masks = oracle._tokenize_worlds(
        worlds=worlds,
        decl_id=5,
        leader=2,
        trick_plays=[],
        remaining=remaining,
        current_player=1,
    )

    # Context token is at index 0
    # TOKEN_TYPE_CONTEXT = 0, decl_id in col 10, normalized_leader in col 11
    assert tokens[0, 0, 9] == 0, "Token type should be CONTEXT (0)"
    assert tokens[0, 0, 10] == 5, "Declaration should be 5"

    # Leader normalized: (2 - 1 + 4) % 4 = 1
    assert tokens[0, 0, 11] == 1, "Normalized leader should be 1"
    assert masks[0, 0] == 1, "Context token should be masked in"


def test_tokenize_hand_tokens():
    """Test hand tokens are populated correctly."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    # Use a known deal
    worlds = [[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]]
    remaining = np.ones((1, 4), dtype=np.int64) * 0x7F

    tokens, masks = oracle._tokenize_worlds(
        worlds=worlds,
        decl_id=0,
        leader=0,
        trick_plays=[],
        remaining=remaining,
        current_player=0,
    )

    # Check first hand token (player 0, domino 0)
    # Token index = 1 + (0 * 7 + 0) = 1
    from forge.ml.tokenize import DOMINO_HIGH, DOMINO_LOW, TOKEN_TYPE_PLAYER0

    assert tokens[0, 1, 0] == DOMINO_HIGH[0]
    assert tokens[0, 1, 1] == DOMINO_LOW[0]
    assert tokens[0, 1, 9] == TOKEN_TYPE_PLAYER0  # Player 0 token type

    # Check that all 28 hand tokens are masked in
    assert np.all(masks[0, 1:29] == 1), "All 28 hand tokens should be masked in"


def test_tokenize_trick_tokens():
    """Test trick tokens are populated correctly."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    hands = [
        [0, 1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27]
    ]
    worlds = [hands]
    remaining = np.ones((1, 4), dtype=np.int64) * 0x7F

    # Leader is player 1, they played domino at local index 2 (global ID 9)
    trick_plays = [(1, 2)]

    tokens, masks = oracle._tokenize_worlds(
        worlds=worlds,
        decl_id=0,
        leader=1,
        trick_plays=trick_plays,
        remaining=remaining,
        current_player=1,
    )

    # Trick token should be at index 29
    from forge.ml.tokenize import DOMINO_HIGH, DOMINO_LOW, TOKEN_TYPE_TRICK_P0

    global_id = hands[1][2]  # Should be 9
    assert global_id == 9

    assert tokens[0, 29, 0] == DOMINO_HIGH[9]
    assert tokens[0, 29, 1] == DOMINO_LOW[9]
    assert tokens[0, 29, 9] == TOKEN_TYPE_TRICK_P0
    assert masks[0, 29] == 1, "Trick token should be masked in"


def test_tokenize_remaining_bits():
    """Test remaining domino bits are encoded correctly."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    worlds = [deal_from_seed(0)]

    # Player 0 has dominoes 0, 1, 2 remaining (bitmask 0b0000111 = 0x07)
    # Player 1 has all dominoes (0x7F)
    remaining = np.array([[0x07, 0x7F, 0x7F, 0x7F]], dtype=np.int64)

    tokens, _ = oracle._tokenize_worlds(
        worlds=worlds,
        decl_id=0,
        leader=0,
        trick_plays=[],
        remaining=remaining,
        current_player=0,
    )

    # Check player 0's tokens (indices 1-7)
    # Dominoes 0, 1, 2 should have is_remaining=1, others should be 0
    for local_idx in range(7):
        token_idx = 1 + local_idx
        expected = 1 if local_idx < 3 else 0
        actual = tokens[0, token_idx, 8]
        assert actual == expected, f"Player 0 domino {local_idx}: expected {expected}, got {actual}"


def test_tokenize_normalized_players():
    """Test player IDs are normalized relative to current player."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    worlds = [deal_from_seed(0)]
    remaining = np.ones((1, 4), dtype=np.int64) * 0x7F

    # Current player is 2
    tokens, _ = oracle._tokenize_worlds(
        worlds=worlds,
        decl_id=0,
        leader=0,
        trick_plays=[],
        remaining=remaining,
        current_player=2,
    )

    # Player 0's normalized ID: (0 - 2 + 4) % 4 = 2
    # Player 1's normalized ID: (1 - 2 + 4) % 4 = 3
    # Player 2's normalized ID: (2 - 2 + 4) % 4 = 0 (current player)
    # Player 3's normalized ID: (3 - 2 + 4) % 4 = 1

    # Check player 2's first token (index 1 + 2*7 = 15)
    assert tokens[0, 15, 5] == 0, "Player 2 should be normalized to 0 (current player)"
    assert tokens[0, 15, 6] == 1, "is_current should be 1"

    # Check player 0's first token (index 1)
    assert tokens[0, 1, 5] == 2, "Player 0 should be normalized to 2"
    assert tokens[0, 1, 7] == 1, "is_partner should be 1"


# =============================================================================
# Integration Test (optional, slower)
# =============================================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not Path("checkpoints/stage1-latest.ckpt").exists(),
    reason="No checkpoint found at checkpoints/stage1-latest.ckpt"
)
def test_oracle_with_real_checkpoint():
    """Integration test with real checkpoint (slower, requires checkpoint file)."""
    checkpoint_path = "checkpoints/stage1-latest.ckpt"
    oracle = Stage1Oracle(checkpoint_path, device="cpu")

    # Create sample worlds
    worlds = [deal_from_seed(i) for i in range(5)]
    remaining = np.ones((5, 4), dtype=np.int64) * 0x7F

    game_state_info = {
        'decl_id': 3,
        'leader': 0,
        'trick_plays': [],
        'remaining': remaining,
    }

    # Query
    logits = oracle.query_batch(
        worlds=worlds,
        game_state_info=game_state_info,
        current_player=0,
    )

    # Verify output
    assert logits.shape == (5, 7)
    assert torch.isfinite(logits).all(), "Logits should be finite"

    # Verify logits are reasonable (not all zeros/same)
    assert logits.std() > 0.01, "Logits should have variation"


# =============================================================================
# Performance Tests
# =============================================================================

def test_batch_query_performance(mock_checkpoint):
    """Test that batch querying is efficient (should complete quickly)."""
    import time

    _, _ = mock_checkpoint
    oracle = Stage1Oracle("fake_checkpoint.ckpt", device="cpu")

    # Create 100 worlds
    worlds = [deal_from_seed(i) for i in range(100)]
    remaining = np.ones((100, 4), dtype=np.int64) * 0x7F

    game_state_info = {
        'decl_id': 3,
        'leader': 0,
        'trick_plays': [],
        'remaining': remaining,
    }

    # Time the query
    start = time.time()
    logits = oracle.query_batch(
        worlds=worlds,
        game_state_info=game_state_info,
        current_player=0,
    )
    elapsed = time.time() - start

    # Should complete in < 2 seconds (very generous, mock should be instant)
    assert elapsed < 2.0, f"Batch query took {elapsed:.3f}s (expected < 2s)"
    assert logits.shape == (100, 7)


# =============================================================================
# Edge Cases
# =============================================================================

def test_single_world_query(mock_checkpoint):
    """Test querying a single world works correctly."""
    _, _ = mock_checkpoint
    oracle = Stage1Oracle("fake_checkpoint.ckpt", device="cpu")

    worlds = [deal_from_seed(0)]
    remaining = np.ones((1, 4), dtype=np.int64) * 0x7F

    game_state_info = {
        'decl_id': 0,
        'leader': 0,
        'trick_plays': [],
        'remaining': remaining,
    }

    logits = oracle.query_batch(
        worlds=worlds,
        game_state_info=game_state_info,
        current_player=0,
    )

    assert logits.shape == (1, 7)


def test_different_declarations(mock_checkpoint):
    """Test querying with different declarations."""
    _, _ = mock_checkpoint
    oracle = Stage1Oracle("fake_checkpoint.ckpt", device="cpu")

    worlds = [deal_from_seed(0), deal_from_seed(1)]
    remaining = np.ones((2, 4), dtype=np.int64) * 0x7F

    # Try different declarations
    for decl_id in [0, 5, 7, 9]:  # blanks, fives, doubles-trump, notrump
        game_state_info = {
            'decl_id': decl_id,
            'leader': 0,
            'trick_plays': [],
            'remaining': remaining,
        }

        logits = oracle.query_batch(
            worlds=worlds,
            game_state_info=game_state_info,
            current_player=0,
        )

        assert logits.shape == (2, 7)


def test_mid_trick_state(mock_checkpoint):
    """Test querying in the middle of a trick."""
    _, _ = mock_checkpoint
    oracle = Stage1Oracle("fake_checkpoint.ckpt", device="cpu")

    worlds = [deal_from_seed(0)]
    remaining = np.ones((1, 4), dtype=np.int64) * 0x7F

    # Leader (player 1) played domino at local index 2
    # Next player (player 2) played domino at local index 0
    game_state_info = {
        'decl_id': 3,
        'leader': 1,
        'trick_plays': [(1, 2), (2, 0)],
        'remaining': remaining,
    }

    # Current player is 3 (third to play)
    logits = oracle.query_batch(
        worlds=worlds,
        game_state_info=game_state_info,
        current_player=3,
    )

    assert logits.shape == (1, 7)
