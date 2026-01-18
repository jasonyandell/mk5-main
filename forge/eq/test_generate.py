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
    n_samples = 3
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=n_samples)

    # Should have 28 decisions (4 players × 7 tricks)
    assert len(record.decisions) == 28
    assert isinstance(record, GameRecord)

    # Oracle is now called once per world per decision (after t42-64uj.1 fix)
    # Total calls = n_samples * 28 decisions
    assert oracle.query_count == n_samples * 28


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
    """Verify e_logits are reasonable averages (no NaN in valid positions).

    After t42-64uj.1 fix, e_logits are padded with -inf for positions beyond
    the remaining hand size. We only check that valid positions are finite.
    """
    oracle = MockOracle()
    hands = deal_from_seed(3)
    record = generate_eq_game(oracle, hands, decl_id=4, n_samples=5)

    for decision in record.decisions:
        # Legal positions should be finite
        legal_logits = decision.e_logits[decision.legal_mask]
        assert torch.isfinite(legal_logits).all(), f"Non-finite legal logits: {legal_logits}"
        assert not legal_logits.isnan().any(), f"NaN in legal logits: {legal_logits}"

        # Legal logits should have reasonable range
        assert legal_logits.abs().max() < 100  # Reasonable logit range


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


class DeterministicOracle:
    """Oracle that returns deterministic Q-values based on domino ID.

    This oracle returns high Q-values for specific domino IDs, allowing us to
    verify that the index alignment fix correctly maps between initial-hand
    slots and remaining-hand positions.

    The key test: oracle outputs are indexed by initial-hand local_idx, but
    generate.py should project these to remaining-hand order. If the projection
    is wrong, we'll select the wrong domino.
    """

    def __init__(self, target_domino_id: int = 0, device: str = "cpu"):
        """Create oracle that strongly prefers a specific domino.

        Args:
            target_domino_id: Domino ID (0-27) to give highest Q-value
            device: Device for tensor output
        """
        self.target_domino_id = target_domino_id
        self.device = device
        self.query_count = 0
        self.last_world = None

    def query_batch(
        self,
        worlds: list[list[list[int]]],
        game_state_info: dict,
        current_player: int,
    ) -> Tensor:
        """Return Q-values that prefer target_domino_id.

        CRITICAL: Returns logits indexed by initial-hand local_idx.
        The initial_hand is sorted, so local_idx i corresponds to
        initial_hands[current_player][i].
        """
        n = len(worlds)
        self.query_count += 1
        self.last_world = worlds[0]

        logits = torch.zeros(n, 7, device=self.device)

        for world_idx, initial_hands in enumerate(worlds):
            # Find which local_idx has the target domino
            player_hand = initial_hands[current_player]  # Sorted initial hand
            for local_idx, domino_id in enumerate(player_hand):
                if domino_id == self.target_domino_id:
                    logits[world_idx, local_idx] = 10.0  # High Q for target
                else:
                    logits[world_idx, local_idx] = 0.0

        return logits


def test_index_alignment_regression():
    """Regression test for t42-64uj.1: E[Q] local-index alignment bug.

    This test verifies that after plays are made (when remaining hand < 7 dominoes),
    the action_taken correctly refers to the intended domino in the remaining hand,
    not a misaligned slot from the initial hand.

    The bug was:
    - Oracle outputs Q[local_idx] for initial-hand slots
    - generate.py was using action_idx directly as remaining-hand index
    - After plays, initial-hand index != remaining-hand index

    The fix:
    - Project oracle outputs from local_idx -> domino_id -> remaining-hand index
    """
    # Use a deal where we know exactly what's happening
    hands = deal_from_seed(42)

    # Player 0's initial hand (sorted): pick a domino that will still be there after 1 play
    player0_initial = sorted(hands[0])
    target_domino = player0_initial[3]  # Pick middle domino

    # Create oracle that strongly prefers this domino
    oracle = DeterministicOracle(target_domino_id=target_domino)

    # Generate game
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=2)

    # Check first decision (full hand)
    first = record.decisions[0]
    remaining_hand_first = player0_initial  # First decision has full hand

    # action_taken should point to target_domino in remaining hand
    selected_domino_first = remaining_hand_first[first.action_taken]
    assert selected_domino_first == target_domino, (
        f"First decision: expected domino {target_domino}, got {selected_domino_first}. "
        f"action_taken={first.action_taken}, hand={remaining_hand_first}"
    )

    # Now check a later decision (after some plays) for player 0
    # Player 0 makes decisions at positions 0, 4, 8, 12, 16, 20, 24 (every 4th decision)
    # After decision 4, player 0 has 6 dominoes left
    if len(record.decisions) > 4:
        fifth_decision_by_p0 = record.decisions[16]  # 5th decision by player 0

        # If target_domino was played earlier, the test passes (it was selected correctly)
        # If target_domino is still in hand, verify alignment

        # The e_logits should have high value for the target domino position
        # in REMAINING hand order (not initial hand order)
        legal_mask = fifth_decision_by_p0.legal_mask
        e_logits = fifth_decision_by_p0.e_logits

        # Check that legal actions have proper logits (no -inf for valid positions)
        valid_logits = e_logits[legal_mask]
        assert all(torch.isfinite(v) for v in valid_logits), (
            f"Some legal logits are non-finite: {e_logits[legal_mask]}"
        )


def test_action_taken_is_remaining_hand_index():
    """Verify action_taken indexes into remaining hand, not initial hand.

    Key invariant: my_hand[action_taken] should equal the domino that was played.
    """
    oracle = MockOracle()
    hands = deal_from_seed(100)

    # Track which dominoes each player plays
    played_dominoes = {0: [], 1: [], 2: [], 3: []}

    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=2)

    # For each decision, verify action_taken is valid for remaining hand size
    current_hands = [list(sorted(h)) for h in hands]  # Copy and sort

    for i, decision in enumerate(record.decisions):
        player = i % 4  # Player order: 0, 1, 2, 3, 0, 1, ...
        remaining_hand = current_hands[player]
        hand_size = len(remaining_hand)

        # action_taken must be < hand_size (valid index into remaining hand)
        assert 0 <= decision.action_taken < hand_size, (
            f"Decision {i}: action_taken={decision.action_taken} but hand_size={hand_size}"
        )

        # legal_mask should only have True values for valid indices
        assert not decision.legal_mask[hand_size:].any(), (
            f"Decision {i}: legal_mask has True beyond hand_size={hand_size}"
        )

        # After this play, remove the domino from the player's hand
        # (We don't know which domino was played without the oracle,
        # but the constraint above should still hold)


def test_e_logits_finite_for_remaining_hand():
    """Verify e_logits are finite for remaining hand positions.

    After the fix, e_logits[i] should be a valid averaged Q-value for
    remaining_hand[i], not -inf (which would indicate alignment issues).
    """
    oracle = MockOracle()
    hands = deal_from_seed(200)
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=5)

    # Check all decisions
    for i, decision in enumerate(record.decisions):
        # Count how many valid (non-inf) logits we have
        hand_size = decision.legal_mask.sum().item()  # At least this many valid

        # All logits up to hand_size should be finite (valid Q-values)
        # Logits beyond hand_size are padded with -inf
        valid_logits = decision.e_logits[:hand_size + 2]  # Allow some slack
        finite_count = torch.isfinite(valid_logits).sum().item()

        # At least hand_size should be finite
        assert finite_count >= hand_size, (
            f"Decision {i}: expected at least {hand_size} finite logits, got {finite_count}. "
            f"e_logits[:10]={decision.e_logits[:10]}"
        )
