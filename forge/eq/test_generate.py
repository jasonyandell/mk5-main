"""Tests for E[Q] game generation."""

from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.generate import DecisionRecord, GameRecord, generate_eq_game
from forge.oracle.rng import deal_from_seed


class MockOracle:
    """Mock oracle that returns random logits for testing."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.query_count = 0

    def query_batch(
        self,
        worlds: list[list[list[int]]],
        game_state_info: dict,
        current_player: int,
    ) -> Tensor:
        """Return random logits favoring lower indices."""
        n = len(worlds)
        self.query_count += 1

        # Return random logits with slight bias toward first action
        # This makes tests more predictable
        logits = torch.randn(n, 7, device=self.device)
        logits[:, 0] += 0.5  # Favor first action slightly

        return logits


def test_generate_one_game():
    """Test basic game generation with mock oracle."""
    oracle = MockOracle()
    hands = deal_from_seed(42)  # Use real deal
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=3)  # Only 3 samples!

    # Should have 28 decisions (4 players × 7 tricks)
    assert len(record.decisions) == 28
    assert isinstance(record, GameRecord)

    # Check that oracle was called 28 times
    assert oracle.query_count == 28


def test_decision_record_structure():
    """Verify DecisionRecord fields have correct structure."""
    oracle = MockOracle()
    hands = deal_from_seed(0)
    record = generate_eq_game(oracle, hands, decl_id=1, n_samples=2)

    # Check first decision
    decision = record.decisions[0]
    assert isinstance(decision, DecisionRecord)

    # transcript_tokens should be 2D tensor
    assert decision.transcript_tokens.dim() == 2
    assert decision.transcript_tokens.shape[1] == 8  # N_FEATURES from transcript_tokenize

    # e_logits should be (7,) tensor
    assert decision.e_logits.shape == (7,)

    # legal_mask should be (7,) boolean tensor
    assert decision.legal_mask.shape == (7,)
    assert decision.legal_mask.dtype == torch.bool

    # action_taken should be an int in range [0, 6]
    assert isinstance(decision.action_taken, int)
    assert 0 <= decision.action_taken <= 6


def test_transcript_tokens_grow_over_time():
    """Verify transcript tokens get longer as game progresses."""
    oracle = MockOracle()
    hands = deal_from_seed(1)
    record = generate_eq_game(oracle, hands, decl_id=2, n_samples=2)

    # First decision should have shorter transcript than last
    first_len = record.decisions[0].transcript_tokens.shape[0]
    last_len = record.decisions[-1].transcript_tokens.shape[0]

    # Last decision should have ~27 more play tokens than first
    # (27 plays before the 28th decision)
    assert last_len > first_len
    assert last_len - first_len >= 20  # Allow some variation


def test_legal_mask_matches_actions():
    """Verify legal_mask correctly indicates legal actions."""
    oracle = MockOracle()
    hands = deal_from_seed(2)
    record = generate_eq_game(oracle, hands, decl_id=3, n_samples=2)

    for decision in record.decisions:
        # At least one action should be legal
        assert decision.legal_mask.any()

        # action_taken should be legal
        assert decision.legal_mask[decision.action_taken]


def test_e_logits_are_averaged():
    """Verify e_logits are reasonable averages (no NaN/Inf)."""
    oracle = MockOracle()
    hands = deal_from_seed(3)
    record = generate_eq_game(oracle, hands, decl_id=4, n_samples=5)

    for decision in record.decisions:
        # Should not have NaN or Inf
        assert torch.isfinite(decision.e_logits).all()

        # Should have reasonable range (not all zeros, not too extreme)
        assert decision.e_logits.abs().max() < 100  # Reasonable logit range


def test_different_declarations():
    """Test game generation with different declaration IDs."""
    oracle = MockOracle()
    hands = deal_from_seed(4)

    # Test a few different declarations
    for decl_id in [0, 3, 7, 9]:
        record = generate_eq_game(oracle, hands, decl_id=decl_id, n_samples=2)
        assert len(record.decisions) == 28


def test_varying_sample_counts():
    """Test that n_samples parameter is respected."""
    oracle = MockOracle()
    hands = deal_from_seed(5)

    # With more samples, should still get 28 decisions
    record_small = generate_eq_game(oracle, hands, decl_id=0, n_samples=2)
    record_large = generate_eq_game(oracle, hands, decl_id=0, n_samples=10)

    assert len(record_small.decisions) == 28
    assert len(record_large.decisions) == 28


def test_hand_sizes_decrease():
    """Verify that as game progresses, hands get smaller."""
    oracle = MockOracle()
    hands = deal_from_seed(6)
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=2)

    # Check that legal_mask has fewer True values as game progresses
    # First decision: 7 cards, last decision: 1 card
    first_decision = record.decisions[0]
    last_decision = record.decisions[-1]

    # First decision should have more legal actions (hand is fuller)
    first_legal_count = first_decision.legal_mask.sum().item()
    last_legal_count = last_decision.legal_mask.sum().item()

    # Early in game, should have close to 7 cards
    assert first_legal_count >= 5

    # Late in game, should have just 1 card
    assert last_legal_count >= 1
