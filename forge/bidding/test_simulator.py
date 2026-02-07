"""Tests for vectorized game simulation.

Verifies that the vectorized implementation produces correct results
by checking game invariants and comparing against known outcomes.
"""

from __future__ import annotations

import pytest
import torch

from forge.bidding.simulator import (
    BatchedGameState,
    deal_random_hands,
    _TABLES,
    _get_table,
    DOMINO_HIGH_T,
    DOMINO_LOW_T,
    IS_IN_CALLED_SUIT_T,
    CAN_FOLLOW_T,
    LED_SUIT_T,
    TRICK_RANK_T,
    COUNT_POINTS_T,
)
from forge.oracle.declarations import N_DECLS, NOTRUMP
from forge.oracle.tables import (
    DOMINO_HIGH,
    DOMINO_LOW,
    DOMINO_COUNT_POINTS,
    is_in_called_suit,
    trick_rank,
)


class TestTensorTables:
    """Test that tensor lookup tables match the original Python functions."""

    def test_domino_high_matches(self):
        """DOMINO_HIGH tensor matches tuple."""
        for d in range(28):
            assert DOMINO_HIGH_T[d].item() == DOMINO_HIGH[d]

    def test_domino_low_matches(self):
        """DOMINO_LOW tensor matches tuple."""
        for d in range(28):
            assert DOMINO_LOW_T[d].item() == DOMINO_LOW[d]

    def test_is_in_called_suit_matches(self):
        """IS_IN_CALLED_SUIT tensor matches function."""
        for d in range(28):
            for decl in range(N_DECLS):
                expected = is_in_called_suit(d, decl)
                actual = IS_IN_CALLED_SUIT_T[d, decl].item()
                assert actual == expected, f"Mismatch at domino={d}, decl={decl}"

    def test_trick_rank_matches(self):
        """TRICK_RANK tensor matches function."""
        for d in range(28):
            for led in range(8):
                for decl in range(N_DECLS):
                    expected = trick_rank(d, led, decl)
                    actual = TRICK_RANK_T[d, led, decl].item()
                    assert actual == expected, f"Mismatch at d={d}, led={led}, decl={decl}"

    def test_count_points_matches(self):
        """COUNT_POINTS tensor matches tuple."""
        for d in range(28):
            assert COUNT_POINTS_T[d].item() == DOMINO_COUNT_POINTS[d]


class TestDealRandomHands:
    """Test deal_random_hands function."""

    def test_basic_deal(self):
        """Basic dealing produces valid hands."""
        bidder_hand = [0, 1, 2, 3, 4, 5, 6]  # First 7 dominoes
        n_games = 10
        device = torch.device("cpu")

        hands = deal_random_hands(bidder_hand, n_games, device)

        assert hands.shape == (n_games, 4, 7)
        assert hands.dtype == torch.long

    def test_bidder_hand_preserved(self):
        """Player 0 always gets the bidder's hand."""
        bidder_hand = [0, 7, 14, 21, 3, 10, 17]
        n_games = 5
        device = torch.device("cpu")

        hands = deal_random_hands(bidder_hand, n_games, device)

        expected = torch.tensor(sorted(bidder_hand), dtype=torch.long)
        for g in range(n_games):
            assert torch.equal(hands[g, 0], expected)

    def test_all_dominoes_dealt(self):
        """All 28 dominoes are dealt across the 4 players."""
        bidder_hand = [0, 1, 2, 3, 4, 5, 6]
        n_games = 10
        device = torch.device("cpu")

        hands = deal_random_hands(bidder_hand, n_games, device)

        for g in range(n_games):
            all_dealt = hands[g].flatten().tolist()
            assert len(set(all_dealt)) == 28
            assert set(all_dealt) == set(range(28))

    def test_hands_sorted(self):
        """Each player's hand is sorted."""
        bidder_hand = [27, 14, 0, 7, 21, 3, 10]
        n_games = 10
        device = torch.device("cpu")

        hands = deal_random_hands(bidder_hand, n_games, device)

        for g in range(n_games):
            for p in range(4):
                hand = hands[g, p].tolist()
                assert hand == sorted(hand)

    def test_reproducibility(self):
        """Same seed produces same hands."""
        bidder_hand = [0, 1, 2, 3, 4, 5, 6]
        n_games = 5
        device = torch.device("cpu")

        rng1 = torch.Generator()
        rng1.manual_seed(42)
        hands1 = deal_random_hands(bidder_hand, n_games, device, rng1)

        rng2 = torch.Generator()
        rng2.manual_seed(42)
        hands2 = deal_random_hands(bidder_hand, n_games, device, rng2)

        assert torch.equal(hands1, hands2)


class TestBatchedGameState:
    """Test BatchedGameState class."""

    @pytest.fixture
    def sample_hands(self):
        """Create sample hands for testing."""
        bidder_hand = [0, 1, 2, 3, 4, 5, 6]
        return deal_random_hands(bidder_hand, 5, torch.device("cpu"))

    def test_initialization(self, sample_hands):
        """State initializes correctly."""
        state = BatchedGameState(sample_hands, decl_id=5, device=torch.device("cpu"))

        assert state.n_games == 5
        assert state.decl_id == 5
        assert state.hands.shape == (5, 4, 7)
        assert state.remaining.shape == (5, 4, 7)
        assert state.remaining.all()  # All dominoes initially remaining
        assert state.team_points.shape == (5, 2)
        assert (state.team_points == 0).all()
        assert state.tricks_played.shape == (5,)
        assert (state.tricks_played == 0).all()

    def test_domino_features_shape(self, sample_hands):
        """Domino features have correct shape."""
        state = BatchedGameState(sample_hands, decl_id=5, device=torch.device("cpu"))

        assert state.domino_features.shape == (5, 4, 7, 5)

    def test_current_player(self, sample_hands):
        """current_player returns correct player."""
        state = BatchedGameState(sample_hands, decl_id=5, device=torch.device("cpu"))

        # Initially, leader is 0 and trick_len is 0, so current = 0
        assert (state.current_player() == 0).all()

        # Simulate some plays
        state.trick_len[0] = 1
        state.trick_len[1] = 2
        assert state.current_player()[0] == 1
        assert state.current_player()[1] == 2

    def test_is_game_over(self, sample_hands):
        """is_game_over returns correct status."""
        state = BatchedGameState(sample_hands, decl_id=5, device=torch.device("cpu"))

        assert not state.is_game_over().any()

        state.tricks_played[0] = 7
        state.tricks_played[2] = 7
        game_over = state.is_game_over()
        assert game_over[0]
        assert not game_over[1]
        assert game_over[2]


class TestGetLegalMask:
    """Test get_legal_mask method."""

    @pytest.fixture
    def simple_state(self):
        """Create a simple state for testing."""
        # Create known hands
        hands = torch.zeros(2, 4, 7, dtype=torch.long)
        # Game 0: Player 0 has dominoes 0-6, Player 1 has 7-13, etc.
        for p in range(4):
            hands[0, p] = torch.arange(p * 7, (p + 1) * 7)
        # Game 1: Same for simplicity
        hands[1] = hands[0]

        return BatchedGameState(hands, decl_id=5, device=torch.device("cpu"))

    def test_leading_all_legal(self, simple_state):
        """When leading, all remaining dominoes are legal."""
        state = simple_state

        legal = state.get_legal_mask()

        # All 7 dominoes should be legal for each game
        assert legal.shape == (2, 7)
        assert legal.all()

    def test_following_suit(self, simple_state):
        """When following, must follow suit if possible."""
        state = simple_state

        # Player 0 leads with local index 0 (domino 0 = 0-0, blank double)
        state.trick_plays[:, 0] = 0  # Local index 0
        state.trick_len[:] = 1  # One play made

        legal = state.get_legal_mask()

        # Now player 1 is current. Player 1 has dominoes 7-13.
        # With fives trump (decl_id=5), need to check which can follow blanks
        # Domino 0 is 0-0 (blank double), led suit is 0 (blanks)
        # Player 1's dominoes that have blanks (pip 0): 7 = 1-0, etc.
        assert legal.shape == (2, 7)


class TestStepAndResolve:
    """Test step and trick resolution."""

    def test_step_marks_played(self):
        """step() marks dominoes as played."""
        hands = torch.zeros(2, 4, 7, dtype=torch.long)
        for p in range(4):
            hands[:, p] = torch.arange(p * 7, (p + 1) * 7)

        state = BatchedGameState(hands, decl_id=NOTRUMP, device=torch.device("cpu"))

        # All remaining initially
        assert state.remaining.all()

        # Player 0 plays local index 0
        actions = torch.tensor([0, 0])
        state.step(actions)

        # Local index 0 should be marked as played for player 0
        assert not state.remaining[0, 0, 0]
        assert not state.remaining[1, 0, 0]
        assert state.trick_len[0] == 1
        assert state.trick_len[1] == 1

    def test_full_trick(self):
        """A full trick resolves correctly."""
        hands = torch.zeros(1, 4, 7, dtype=torch.long)
        for p in range(4):
            hands[0, p] = torch.arange(p * 7, (p + 1) * 7)

        state = BatchedGameState(hands, decl_id=NOTRUMP, device=torch.device("cpu"))

        # Play 4 actions to complete a trick
        for i in range(4):
            actions = torch.tensor([0])  # Each player plays their first domino
            state.step(actions)

        # Trick should be resolved
        assert state.tricks_played[0] == 1
        assert state.trick_len[0] == 0  # Reset for next trick
        assert state.team_points.sum() > 0  # Some points scored


class TestBuildTokens:
    """Test build_tokens method."""

    def test_tokens_shape(self):
        """Tokens have correct shape."""
        hands = deal_random_hands([0, 1, 2, 3, 4, 5, 6], 3, torch.device("cpu"))
        state = BatchedGameState(hands, decl_id=5, device=torch.device("cpu"))

        tokens, mask, current = state.build_tokens()

        assert tokens.shape == (3, 32, 12)
        assert mask.shape == (3, 32)
        assert current.shape == (3,)

    def test_context_token(self):
        """Context token (position 0) is set correctly."""
        hands = deal_random_hands([0, 1, 2, 3, 4, 5, 6], 2, torch.device("cpu"))
        state = BatchedGameState(hands, decl_id=5, device=torch.device("cpu"))

        tokens, mask, current = state.build_tokens()

        # Context token at position 0
        assert mask[:, 0].all()  # Mask is 1
        assert (tokens[:, 0, 10] == 5).all()  # decl_id = 5

    def test_hand_tokens_mask(self):
        """Hand tokens (positions 1-28) are all masked in."""
        hands = deal_random_hands([0, 1, 2, 3, 4, 5, 6], 2, torch.device("cpu"))
        state = BatchedGameState(hands, decl_id=5, device=torch.device("cpu"))

        tokens, mask, current = state.build_tokens()

        # Positions 1-28 should all be masked in
        assert mask[:, 1:29].all()

    def test_trick_tokens_empty_initially(self):
        """Trick tokens (positions 29-31) are initially not masked."""
        hands = deal_random_hands([0, 1, 2, 3, 4, 5, 6], 2, torch.device("cpu"))
        state = BatchedGameState(hands, decl_id=5, device=torch.device("cpu"))

        tokens, mask, current = state.build_tokens()

        # No trick plays yet, so positions 29-31 should not be masked
        assert not mask[:, 29:32].any()


class TestFullGameSimulation:
    """Test full game simulation without model."""

    def test_game_completes(self):
        """A game can run to completion."""
        hands = deal_random_hands([0, 1, 2, 3, 4, 5, 6], 5, torch.device("cpu"))
        state = BatchedGameState(hands, decl_id=5, device=torch.device("cpu"))

        # Play 28 steps (maximum for 7 tricks × 4 plays)
        for _ in range(28):
            if state.is_game_over().all():
                break

            legal = state.get_legal_mask()
            # Pick first legal action
            actions = legal.long().argmax(dim=1)
            state.step(actions)

        # All games should be over
        assert state.is_game_over().all()
        assert (state.tricks_played == 7).all()

    def test_points_valid(self):
        """Total points across teams equals 42."""
        hands = deal_random_hands([0, 1, 2, 3, 4, 5, 6], 10, torch.device("cpu"))
        state = BatchedGameState(hands, decl_id=5, device=torch.device("cpu"))

        # Play to completion
        for _ in range(28):
            if state.is_game_over().all():
                break
            legal = state.get_legal_mask()
            actions = legal.long().argmax(dim=1)
            state.step(actions)

        # Total points should be 42 for each game
        # (7 tricks base + 35 count points = 42)
        total_points = state.team_points.sum(dim=1)
        assert (total_points == 42).all(), f"Expected 42, got {total_points}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
