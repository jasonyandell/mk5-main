"""Tests for E[Q] viewer decode paths and rendering (t42-64uj.7).

These tests ensure the viewer can handle v1, v2, and v2.2 datasets gracefully.
"""

import pytest
import torch
from torch import Tensor

from forge.eq.viewer import (
    DECLARATION_NAMES,
    EXPLORATION_MODE_NAMES,
    decode_transcript,
    domino_id_to_pips,
    domino_str,
    find_current_winner,
    infer_voids_from_plays,
    pips_to_domino_id,
    player_name,
    plays_to_tricks,
    render_default_mode,
    render_diagnostics_mode,
)
from forge.eq.transcript_tokenize import (
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
)


# =============================================================================
# Domino ID conversion tests
# =============================================================================


class TestDominoIdConversion:
    """Test triangular encoding for domino IDs."""

    def test_domino_id_to_pips_0_0(self):
        """[0:0] should be ID 0."""
        assert domino_id_to_pips(0) == (0, 0)

    def test_domino_id_to_pips_1_0(self):
        """[1:0] should be ID 1."""
        assert domino_id_to_pips(1) == (1, 0)

    def test_domino_id_to_pips_1_1(self):
        """[1:1] should be ID 2."""
        assert domino_id_to_pips(2) == (1, 1)

    def test_domino_id_to_pips_6_6(self):
        """[6:6] should be ID 27."""
        assert domino_id_to_pips(27) == (6, 6)

    def test_pips_to_domino_id_roundtrip(self):
        """All 28 domino IDs should roundtrip correctly."""
        for domino_id in range(28):
            high, low = domino_id_to_pips(domino_id)
            assert pips_to_domino_id(high, low) == domino_id

    def test_pips_to_domino_id_canonical_ordering(self):
        """pips_to_domino_id should handle non-canonical (low, high) order."""
        # [5:3] and [3:5] should both map to same ID
        assert pips_to_domino_id(5, 3) == pips_to_domino_id(3, 5)

    def test_domino_str_format(self):
        """domino_str should format as [H:L]."""
        assert domino_str(5, 3) == "[5:3]"
        assert domino_str(0, 0) == "[0:0]"


# =============================================================================
# Transcript decoding tests
# =============================================================================


def make_test_tokens(decl_id: int, hand: list[tuple[int, int]], plays: list[tuple[int, int, int]]) -> tuple[Tensor, int]:
    """Create test tokens tensor.

    Args:
        decl_id: Declaration ID (0-9)
        hand: List of (high, low) tuples for current hand
        plays: List of (rel_player, high, low) tuples for play history

    Returns:
        (tokens tensor, length)
    """
    max_tokens = 36
    n_features = 8
    tokens = torch.zeros(max_tokens, n_features, dtype=torch.int64)

    idx = 0

    # Declaration token
    tokens[idx, FEAT_DECL] = decl_id
    tokens[idx, FEAT_TOKEN_TYPE] = TOKEN_TYPE_DECL
    idx += 1

    # Hand tokens
    for high, low in hand:
        tokens[idx, FEAT_HIGH] = high
        tokens[idx, FEAT_LOW] = low
        tokens[idx, FEAT_IS_DOUBLE] = 1 if high == low else 0
        tokens[idx, FEAT_IS_IN_HAND] = 1
        tokens[idx, FEAT_PLAYER] = 0  # Always "me"
        tokens[idx, FEAT_DECL] = decl_id
        tokens[idx, FEAT_TOKEN_TYPE] = TOKEN_TYPE_HAND
        idx += 1

    # Play tokens
    for rel_player, high, low in plays:
        tokens[idx, FEAT_HIGH] = high
        tokens[idx, FEAT_LOW] = low
        tokens[idx, FEAT_IS_DOUBLE] = 1 if high == low else 0
        tokens[idx, FEAT_IS_IN_HAND] = 0
        tokens[idx, FEAT_PLAYER] = rel_player
        tokens[idx, FEAT_DECL] = decl_id
        tokens[idx, FEAT_TOKEN_TYPE] = TOKEN_TYPE_PLAY
        idx += 1

    return tokens, idx


class TestDecodeTranscript:
    """Test transcript token decoding."""

    def test_decode_empty_plays(self):
        """Decode transcript with no plays (opening lead)."""
        hand = [(5, 4), (5, 3), (4, 4), (3, 2), (2, 1), (1, 0), (0, 0)]
        tokens, length = make_test_tokens(decl_id=5, hand=hand, plays=[])

        decoded = decode_transcript(tokens, length)

        assert decoded["decl_id"] == 5
        assert decoded["hand"] == hand
        assert decoded["plays"] == []

    def test_decode_with_plays(self):
        """Decode transcript with play history."""
        hand = [(5, 4), (5, 3)]
        plays = [(0, 5, 5), (1, 5, 0), (2, 4, 3), (3, 4, 2)]  # One complete trick
        tokens, length = make_test_tokens(decl_id=5, hand=hand, plays=plays)

        decoded = decode_transcript(tokens, length)

        assert decoded["decl_id"] == 5
        assert decoded["hand"] == hand
        assert decoded["plays"] == plays

    def test_decode_extracts_correct_player_ids(self):
        """Verify relative player IDs are preserved."""
        hand = [(5, 4)]
        plays = [(0, 5, 5), (1, 4, 4), (2, 3, 3), (3, 2, 2)]
        tokens, length = make_test_tokens(decl_id=0, hand=hand, plays=plays)

        decoded = decode_transcript(tokens, length)

        players = [p for p, h, l in decoded["plays"]]
        assert players == [0, 1, 2, 3]


class TestPlaysToTricks:
    """Test trick grouping."""

    def test_empty_plays(self):
        """Empty plays should return empty tricks."""
        assert plays_to_tricks([]) == []

    def test_one_complete_trick(self):
        """Four plays should form one trick."""
        plays = [(0, 5, 5), (1, 5, 0), (2, 4, 4), (3, 4, 0)]
        tricks = plays_to_tricks(plays)
        assert len(tricks) == 1
        assert len(tricks[0]) == 4

    def test_incomplete_trick(self):
        """Less than 4 plays should form incomplete trick."""
        plays = [(0, 5, 5), (1, 5, 0)]
        tricks = plays_to_tricks(plays)
        assert len(tricks) == 1
        assert len(tricks[0]) == 2


# =============================================================================
# Render function tests (graceful degradation)
# =============================================================================


class TestRenderDefaultMode:
    """Test default mode rendering handles missing fields gracefully."""

    def test_render_without_uncertainty(self):
        """Render should work without uncertainty fields (v1 dataset)."""
        hand = [(5, 4), (5, 3), (4, 4)]
        tokens, length = make_test_tokens(decl_id=5, hand=hand, plays=[])
        e_q_mean = torch.tensor([1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
        legal_mask = torch.tensor([True, True, True, False, False, False, False])

        text = render_default_mode(
            idx=0, total=100, tokens=tokens, length=length,
            e_q_mean=e_q_mean, legal_mask=legal_mask,
            action_taken=0, game_idx=0, decision_idx=0,
            # No uncertainty fields
        )

        assert "Example 1/100" in text
        assert "[5:4]" in text
        assert "E[Q]" in text

    def test_render_with_uncertainty(self):
        """Render should show uncertainty when available (v2 dataset)."""
        hand = [(5, 4), (5, 3)]
        tokens, length = make_test_tokens(decl_id=5, hand=hand, plays=[])
        e_q_mean = torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        e_q_var = torch.tensor([4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # σ = 2.0, 1.0
        legal_mask = torch.tensor([True, True, False, False, False, False, False])

        text = render_default_mode(
            idx=0, total=100, tokens=tokens, length=length,
            e_q_mean=e_q_mean, legal_mask=legal_mask,
            action_taken=0, game_idx=0, decision_idx=0,
            e_q_var=e_q_var, u_mean=1.5, u_max=2.0,
        )

        assert "±" in text  # Uncertainty notation
        assert "U_mean" in text

    def test_render_with_posterior_diagnostics(self):
        """Render should show posterior health when available (v2.1+ dataset)."""
        hand = [(5, 4)]
        tokens, length = make_test_tokens(decl_id=5, hand=hand, plays=[])
        e_q_mean = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        legal_mask = torch.tensor([True, False, False, False, False, False, False])

        text = render_default_mode(
            idx=0, total=100, tokens=tokens, length=length,
            e_q_mean=e_q_mean, legal_mask=legal_mask,
            action_taken=0, game_idx=0, decision_idx=0,
            ess=85.3, max_w=0.045,
            exploration_mode=0, q_gap=0.0,
        )

        assert "ESS=" in text
        assert "max_w=" in text
        assert "greedy" in text

    def test_render_with_low_ess_warning(self):
        """Render should warn on low ESS."""
        hand = [(5, 4)]
        tokens, length = make_test_tokens(decl_id=5, hand=hand, plays=[])
        e_q_mean = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        legal_mask = torch.tensor([True, False, False, False, False, False, False])

        text = render_default_mode(
            idx=0, total=100, tokens=tokens, length=length,
            e_q_mean=e_q_mean, legal_mask=legal_mask,
            action_taken=0, game_idx=0, decision_idx=0,
            ess=5.0,  # Critical low ESS
            max_w=0.3,
        )

        assert "⚠️" in text  # Warning for critical ESS


class TestRenderDiagnosticsMode:
    """Test diagnostics mode rendering."""

    def test_render_with_full_metadata(self):
        """Render diagnostics with full v2.2 metadata."""
        hand = [(5, 4)]
        tokens, length = make_test_tokens(decl_id=5, hand=hand, plays=[])
        e_q_mean = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        legal_mask = torch.tensor([True, False, False, False, False, False, False])

        metadata = {
            "version": "2.2",
            "posterior": {
                "enabled": True,
                "tau": 10.0,
                "beta": 0.10,
                "window_k": 8,
                "delta": 30.0,
                "adaptive_k_enabled": False,
                "rejuvenation_enabled": False,
            },
            "exploration": {
                "enabled": True,
                "temperature": 3.0,
                "use_boltzmann": True,
                "epsilon": 0.05,
                "blunder_rate": 0.02,
                "blunder_max_regret": 3.0,
            },
            "summary": {
                "q_range": [-35.0, 35.0],
                "ess_distribution": {"min": 5.0, "p50": 80.0, "mean": 75.0},
                "exploration_stats": {"greedy_rate": 0.85, "mean_q_gap": 0.5},
            },
        }

        text = render_diagnostics_mode(
            idx=0, total=100, tokens=tokens, length=length,
            e_q_mean=e_q_mean, legal_mask=legal_mask,
            action_taken=0, game_idx=0, decision_idx=0,
            ess=85.0, max_w=0.03,
            exploration_mode=1, q_gap=0.5,
            player=0, actual_outcome=5.0,
            metadata=metadata,
        )

        assert "[DIAGNOSTICS]" in text
        assert "Schema version: 2.2" in text
        assert "τ (tau): 10.0" in text
        assert "β (beta): 0.1" in text
        assert "ESS = 85.0" in text
        assert "boltzmann" in text

    def test_render_without_metadata(self):
        """Render diagnostics gracefully without metadata (v1 dataset)."""
        hand = [(5, 4)]
        tokens, length = make_test_tokens(decl_id=5, hand=hand, plays=[])
        e_q_mean = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        legal_mask = torch.tensor([True, False, False, False, False, False, False])

        text = render_diagnostics_mode(
            idx=0, total=100, tokens=tokens, length=length,
            e_q_mean=e_q_mean, legal_mask=legal_mask,
            action_taken=0, game_idx=0, decision_idx=0,
            # No posterior/exploration/metadata
        )

        assert "[DIAGNOSTICS]" in text
        assert "not available" in text


# =============================================================================
# Void inference tests
# =============================================================================


class TestInferVoids:
    """Test void inference from play history."""

    def test_no_voids_first_trick(self):
        """First trick, all follow suit - no voids."""
        plays = [(0, 5, 5), (1, 5, 0), (2, 5, 1), (3, 5, 2)]  # All play 5s (trump in fives)
        voids = infer_voids_from_plays(plays, decl_id=5)

        for p in range(4):
            assert len(voids[p]) == 0

    def test_void_detected(self):
        """Player who doesn't follow suit should be marked void."""
        # Fives declaration: [5:5] leads (trump), player 1 plays [0:0] (not trump)
        plays = [(0, 5, 5), (1, 0, 0), (2, 5, 1), (3, 5, 2)]
        voids = infer_voids_from_plays(plays, decl_id=5)

        # Player 1 is void in the suit that was led
        assert 7 in voids[1]  # 7 = trump suit in standard encoding


# =============================================================================
# Constants tests
# =============================================================================


class TestConstants:
    """Test that constants are properly defined."""

    def test_declaration_names_complete(self):
        """All 10 declarations should have names."""
        for i in range(10):
            assert i in DECLARATION_NAMES

    def test_exploration_mode_names_complete(self):
        """All 4 exploration modes should have names."""
        for i in range(4):
            assert i in EXPLORATION_MODE_NAMES

    def test_player_names(self):
        """Player names should be ME, L, P, R."""
        assert player_name(0) == "ME"
        assert player_name(1) == "L"
        assert player_name(2) == "P"
        assert player_name(3) == "R"
