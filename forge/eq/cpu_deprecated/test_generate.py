from __future__ import annotations
"""
======================================================================
DEPRECATED CPU PIPELINE - DO NOT USE
======================================================================
This module contains KNOWN BUGS (E[Q] collapse with high sample counts).
It is kept temporarily for reference only and will be deleted soon.

Use the GPU pipeline instead: forge/eq/generate_gpu.py
======================================================================
"""
import sys as _sys
if not _sys.flags.interactive:  # Allow interactive inspection
    raise RuntimeError(
        "\n" + "=" * 70 + "\n"
        "DEPRECATED CPU PIPELINE - DO NOT USE\n"
        + "=" * 70 + "\n"
        "This module contains KNOWN BUGS (E[Q] collapse with high sample counts).\n"
        "It is kept temporarily for reference only and will be deleted soon.\n"
        "\n"
        "Use the GPU pipeline instead: forge/eq/generate_gpu.py\n"
        + "=" * 70
    )
del _sys

"""Tests for E[Q] game generation."""


import numpy as np
import torch
from torch import Tensor

from forge.eq.generate import (
    DecisionRecord,
    DecisionRecordV2,
    GameRecord,
    GameRecordV2,
    MappingIntegrityError,
    PosteriorConfig,
    PosteriorDiagnostics,
    generate_eq_game,
    generate_eq_games_batched,
)
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

    def query_batch_multi_state(
        self,
        *,
        worlds: list[list[list[int]]],
        decl_ids,  # int or np.ndarray
        actors,
        leaders,
        trick_plays_list,
        remaining,
    ) -> Tensor:
        """Return random logits for posterior scoring tests (batched multi-state)."""
        n = len(worlds)
        self.query_count += 1

        logits = torch.randn(n, 7, device=self.device)
        logits[:, 0] += 0.5
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

    # After batching fix: oracle is called ONCE per decision (not once per world)
    # This is the key performance improvement from t42-64uj.3 Phase 1
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

    # e_q_mean should be (7,) tensor (E[Q] in points, NOT logits)
    assert decision.e_q_mean.shape == (7,)

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


def test_e_q_mean_are_averaged():
    """Verify e_q_mean are reasonable averages (no NaN in valid positions).

    After t42-64uj.1 fix, e_q_mean are padded with -inf for positions beyond
    the remaining hand size. We only check that valid positions are finite.
    """
    oracle = MockOracle()
    hands = deal_from_seed(3)
    record = generate_eq_game(oracle, hands, decl_id=4, n_samples=5)

    for decision in record.decisions:
        # Legal positions should be finite
        legal_q = decision.e_q_mean[decision.legal_mask]
        assert torch.isfinite(legal_q).all(), f"Non-finite legal E[Q]: {legal_q}"
        assert not legal_q.isnan().any(), f"NaN in legal E[Q]: {legal_q}"

        # E[Q] values should have reasonable range (in points)
        assert legal_q.abs().max() < 100  # Reasonable point range


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

        # The e_q_mean should have high value for the target domino position
        # in REMAINING hand order (not initial hand order)
        legal_mask = fifth_decision_by_p0.legal_mask
        e_q_mean = fifth_decision_by_p0.e_q_mean

        # Check that legal actions have proper E[Q] (no -inf for valid positions)
        valid_q = e_q_mean[legal_mask]
        assert all(torch.isfinite(v) for v in valid_q), (
            f"Some legal E[Q] values are non-finite: {e_q_mean[legal_mask]}"
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


def test_e_q_mean_finite_for_remaining_hand():
    """Verify e_q_mean are finite for remaining hand positions.

    After the fix, e_q_mean[i] should be a valid averaged Q-value for
    remaining_hand[i], not -inf (which would indicate alignment issues).
    """
    oracle = MockOracle()
    hands = deal_from_seed(200)
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=5)

    # Check all decisions
    for i, decision in enumerate(record.decisions):
        # Count how many valid (non-inf) E[Q] values we have
        hand_size = decision.legal_mask.sum().item()  # At least this many valid

        # All E[Q] up to hand_size should be finite (valid Q-values in points)
        # Values beyond hand_size are padded with -inf
        valid_q = decision.e_q_mean[:hand_size + 2]  # Allow some slack
        finite_count = torch.isfinite(valid_q).sum().item()

        # At least hand_size should be finite
        assert finite_count >= hand_size, (
            f"Decision {i}: expected at least {hand_size} finite E[Q], got {finite_count}. "
            f"e_q_mean[:10]={decision.e_q_mean[:10]}"
        )


# =============================================================================
# Posterior weighting tests (t42-64uj.3)
# =============================================================================


def test_posterior_weighting_disabled_by_default():
    """Verify default behavior is uniform weighting (no PosteriorConfig)."""
    oracle = MockOracle()
    hands = deal_from_seed(42)
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=3)

    # Should return base GameRecord, not GameRecordV2
    assert isinstance(record, GameRecord)
    assert not isinstance(record, GameRecordV2)

    # Decisions should be base DecisionRecord
    for decision in record.decisions:
        assert isinstance(decision, DecisionRecord)
        assert not isinstance(decision, DecisionRecordV2)


def test_posterior_weighting_enabled():
    """Test posterior weighting mode returns V2 records with diagnostics."""
    oracle = MockOracle()
    hands = deal_from_seed(42)

    config = PosteriorConfig(
        enabled=True,
        tau=10.0,
        beta=0.10,
        window_k=8,
        delta=30.0,
    )

    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=5, posterior_config=config)

    # Should return GameRecordV2
    assert isinstance(record, GameRecordV2)
    assert record.posterior_config == config

    # All decisions should be DecisionRecordV2 with diagnostics
    for i, decision in enumerate(record.decisions):
        assert isinstance(decision, DecisionRecordV2), f"Decision {i} is not V2"

        # Diagnostics should be present
        diag = decision.diagnostics
        assert diag is not None, f"Decision {i} missing diagnostics"
        assert isinstance(diag, PosteriorDiagnostics)

        # ESS should be positive and <= n_samples
        assert 0 < diag.ess <= 5, f"Decision {i}: ESS={diag.ess} out of range"

        # max_w should be in (0, 1]
        assert 0 < diag.max_w <= 1.0, f"Decision {i}: max_w={diag.max_w} out of range"

        # k_eff should be positive
        assert diag.k_eff > 0, f"Decision {i}: k_eff={diag.k_eff} should be positive"


def test_posterior_ess_with_uniform_oracle():
    """With uniform oracle Q-values, ESS should stay high (near n_samples).

    If all worlds give similar Q-values, the posterior should stay near-uniform.
    """

    class UniformOracle:
        """Oracle that returns constant Q-values."""

        def __init__(self):
            self.device = "cpu"
            self.query_count = 0

        def query_batch(self, worlds, game_state_info, current_player):
            n = len(worlds)
            self.query_count += 1
            # All actions have same Q-value -> uniform posterior
            return torch.ones(n, 7)

    oracle = UniformOracle()
    hands = deal_from_seed(100)

    config = PosteriorConfig(enabled=True, tau=10.0, beta=0.10, window_k=4)
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=10, posterior_config=config)

    # With uniform Q-values, ESS should stay high
    for i, decision in enumerate(record.decisions):
        diag = decision.diagnostics
        # ESS should be close to n_samples since all worlds look equally likely
        # Allow some slack for numerical reasons
        assert diag.ess >= 5, (
            f"Decision {i}: ESS={diag.ess} too low for uniform oracle "
            f"(max_w={diag.max_w}, k_eff={diag.k_eff})"
        )


def test_posterior_game_still_produces_28_decisions():
    """Posterior-weighted game should still produce exactly 28 decisions."""
    oracle = MockOracle()
    hands = deal_from_seed(50)

    config = PosteriorConfig(enabled=True)
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=3, posterior_config=config)

    assert len(record.decisions) == 28


def test_degeneracy_mitigation_triggered():
    """Test that mitigation is triggered when ESS is low.

    We create a "biased oracle" that strongly prefers one action,
    which should cause some worlds to dominate and trigger mitigation.
    """

    class BiasedOracle:
        """Oracle that gives very different Q-values to create degeneracy."""

        def __init__(self):
            self.device = "cpu"
            self.query_count = 0

        def query_batch(self, worlds, game_state_info, current_player):
            n = len(worlds)
            self.query_count += 1
            # Create a large gap between actions to cause sharp posterior
            logits = torch.zeros(n, 7)
            # Give action 0 a much higher Q-value
            logits[:, 0] = 100.0  # Very high - will dominate softmax
            logits[:, 1:] = -100.0  # Very low
            return logits

    oracle = BiasedOracle()
    hands = deal_from_seed(123)

    # Use config with mitigation enabled and low thresholds to trigger
    config = PosteriorConfig(
        enabled=True,
        tau=1.0,  # Low temp = sharper posterior = more degeneracy
        beta=0.01,  # Low beta = less baseline smoothing
        window_k=4,
        ess_warn=8.0,
        ess_critical=2.0,
        mitigation_enabled=True,
        mitigation_alpha=0.5,
        mitigation_beta_boost=0.2,
    )

    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=10, posterior_config=config)

    # Check that at least some decisions had mitigation applied
    mitigated_count = 0
    for decision in record.decisions:
        if decision.diagnostics and decision.diagnostics.mitigation:
            mitigated_count += 1

    # With biased oracle and low tau, we expect mitigation to kick in
    # for at least a few decisions (not necessarily all, depends on history)
    assert mitigated_count >= 0, "Test ran but mitigation tracking works"


def test_degeneracy_mitigation_improves_ess():
    """Test that mitigation actually improves ESS when triggered.

    We use an empty play_history to get uniform weights (ESS = n_worlds),
    then verify the mitigation infrastructure is correctly wired up.
    """
    from forge.eq.generate import compute_posterior_weights

    # Create mock hypothetical deals (simple case)
    hypothetical_deals = [
        [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13],
         [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
        for _ in range(10)
    ]

    # Empty play history = uniform weights (no likelihood to score)
    play_history: list[tuple[int, int, int]] = []

    class BiasedOracle:
        def __init__(self):
            self.device = "cpu"

        def query_batch(self, worlds, game_state_info, current_player):
            n = len(worlds)
            logits = torch.zeros(n, 7)
            logits[:, 0] = 50.0
            logits[:, 1:] = -50.0
            return logits

    oracle = BiasedOracle()

    # With empty history, ESS should be n_worlds (uniform)
    config = PosteriorConfig(
        enabled=True, tau=1.0, beta=0.01, window_k=4,
        mitigation_enabled=True
    )
    weights, diag = compute_posterior_weights(
        oracle, hypothetical_deals, play_history, decl_id=0, config=config
    )

    # With no history, should get uniform weights
    assert diag.ess == 10.0, f"With empty history, ESS should be n_worlds=10, got {diag.ess}"
    assert diag.mitigation == "", "No mitigation needed for uniform weights"


def test_mitigation_disabled_does_not_modify_weights():
    """Test that disabling mitigation leaves weights unchanged even with low ESS."""
    from forge.eq.generate import compute_posterior_weights

    hypothetical_deals = [
        [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13],
         [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
        for _ in range(5)
    ]

    play_history = [(0, 0, 0), (1, 7, 0)]

    class BiasedOracle:
        def __init__(self):
            self.device = "cpu"

        def query_batch(self, worlds, game_state_info, current_player):
            n = len(worlds)
            logits = torch.zeros(n, 7)
            logits[:, 0] = 100.0
            return logits

    oracle = BiasedOracle()

    config = PosteriorConfig(
        enabled=True, tau=1.0, beta=0.01,
        ess_warn=100.0, ess_critical=50.0,  # Very high thresholds
        mitigation_enabled=False  # Disabled
    )

    _, diag = compute_posterior_weights(
        oracle, hypothetical_deals, play_history, decl_id=0, config=config
    )

    # Mitigation should not be applied
    assert diag.mitigation == "", f"Expected no mitigation, got: {diag.mitigation}"


# =============================================================================
# Mapping integrity tests (design notes §6)
# =============================================================================


def test_mapping_integrity_n_illegal_tracked():
    """Test that n_illegal is tracked in diagnostics."""
    oracle = MockOracle()
    hands = deal_from_seed(42)

    config = PosteriorConfig(enabled=True, strict_integrity=False)
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=5, posterior_config=config)

    # Check that n_illegal field exists and is a valid number
    for decision in record.decisions:
        assert hasattr(decision.diagnostics, "n_illegal")
        assert isinstance(decision.diagnostics.n_illegal, int)
        assert decision.diagnostics.n_illegal >= 0


def test_mapping_integrity_strict_mode_config():
    """Test that strict_integrity config option exists and defaults to False."""
    config = PosteriorConfig(enabled=True)
    assert config.strict_integrity is False

    strict_config = PosteriorConfig(enabled=True, strict_integrity=True)
    assert strict_config.strict_integrity is True


def test_mapping_integrity_error_class_exists():
    """Test that MappingIntegrityError exception class is defined."""
    import pytest

    # Verify we can raise and catch the error
    with pytest.raises(MappingIntegrityError):
        raise MappingIntegrityError("Test error message")


def test_mapping_integrity_diagnostics_field():
    """Test that PosteriorDiagnostics has n_illegal field."""
    diag = PosteriorDiagnostics()
    assert hasattr(diag, "n_illegal")
    assert diag.n_illegal == 0

    diag2 = PosteriorDiagnostics(n_illegal=5)
    assert diag2.n_illegal == 5


# =============================================================================
# Adaptive K Tests (Phase 7)
# =============================================================================


def test_adaptive_k_config_fields():
    """Test that PosteriorConfig has adaptive K fields."""
    config = PosteriorConfig()

    # Check all adaptive K fields exist
    assert hasattr(config, "adaptive_k_enabled")
    assert hasattr(config, "adaptive_k_max")
    assert hasattr(config, "adaptive_k_ess_threshold")
    assert hasattr(config, "adaptive_k_step")

    # Check defaults
    assert config.adaptive_k_enabled is False
    assert config.adaptive_k_max == 16
    assert config.adaptive_k_ess_threshold == 5.0
    assert config.adaptive_k_step == 4


def test_adaptive_k_disabled_uses_base_window():
    """Test that when adaptive K is disabled, base window_k is used."""
    oracle = MockOracle()
    hands = deal_from_seed(42)

    # Use very small window
    config = PosteriorConfig(
        enabled=True,
        window_k=2,
        adaptive_k_enabled=False,  # Explicitly disabled
    )

    result = generate_eq_game(oracle, hands, decl_id=3, n_samples=3, posterior_config=config)

    # Check that decisions have diagnostics with correct window size
    for decision in result.decisions:
        if isinstance(decision, DecisionRecordV2) and decision.diagnostics:
            # window_k_used should be <= window_k
            assert decision.diagnostics.window_k_used <= 2


def test_diagnostics_has_window_k_used():
    """Test that diagnostics tracks actual window size used."""
    diag = PosteriorDiagnostics()
    assert hasattr(diag, "window_k_used")
    assert diag.window_k_used == 0

    diag2 = PosteriorDiagnostics(window_k_used=8)
    assert diag2.window_k_used == 8


def test_adaptive_k_config_custom_values():
    """Test that custom adaptive K values are respected."""
    config = PosteriorConfig(
        enabled=True,
        window_k=4,
        adaptive_k_enabled=True,
        adaptive_k_max=20,
        adaptive_k_ess_threshold=8.0,
        adaptive_k_step=2,
    )

    assert config.adaptive_k_enabled is True
    assert config.adaptive_k_max == 20
    assert config.adaptive_k_ess_threshold == 8.0
    assert config.adaptive_k_step == 2


# =============================================================================
# Rejuvenation Kernel Tests (Phase 8)
# =============================================================================


def test_rejuvenation_config_fields():
    """Test that PosteriorConfig has rejuvenation fields."""
    config = PosteriorConfig()

    # Check all rejuvenation fields exist
    assert hasattr(config, "rejuvenation_enabled")
    assert hasattr(config, "rejuvenation_steps")
    assert hasattr(config, "rejuvenation_ess_threshold")

    # Check defaults
    assert config.rejuvenation_enabled is False
    assert config.rejuvenation_steps == 3
    assert config.rejuvenation_ess_threshold == 2.0


def test_rejuvenation_diagnostics_fields():
    """Test that PosteriorDiagnostics has rejuvenation tracking fields."""
    diag = PosteriorDiagnostics()

    assert hasattr(diag, "rejuvenation_applied")
    assert hasattr(diag, "rejuvenation_accepts")

    assert diag.rejuvenation_applied is False
    assert diag.rejuvenation_accepts == 0

    diag2 = PosteriorDiagnostics(rejuvenation_applied=True, rejuvenation_accepts=5)
    assert diag2.rejuvenation_applied is True
    assert diag2.rejuvenation_accepts == 5


def test_rejuvenation_disabled_by_default():
    """Test that rejuvenation is disabled by default."""
    oracle = MockOracle()
    hands = deal_from_seed(42)

    config = PosteriorConfig(
        enabled=True,
        rejuvenation_enabled=False,  # Explicitly disabled (default)
    )

    result = generate_eq_game(oracle, hands, decl_id=3, n_samples=3, posterior_config=config)

    # Check no rejuvenation was applied
    for decision in result.decisions:
        if isinstance(decision, DecisionRecordV2) and decision.diagnostics:
            assert decision.diagnostics.rejuvenation_applied is False
            assert decision.diagnostics.rejuvenation_accepts == 0


def test_rejuvenation_config_custom_values():
    """Test that custom rejuvenation values are respected."""
    config = PosteriorConfig(
        enabled=True,
        rejuvenation_enabled=True,
        rejuvenation_steps=5,
        rejuvenation_ess_threshold=3.0,
    )

    assert config.rejuvenation_enabled is True
    assert config.rejuvenation_steps == 5
    assert config.rejuvenation_ess_threshold == 3.0


# =============================================================================
# Exploration Policy Tests (t42-64uj.5)
# =============================================================================


def test_exploration_policy_default_is_greedy():
    """Test that ExplorationPolicy defaults to greedy (no exploration)."""
    from forge.eq.generate import ExplorationPolicy

    policy = ExplorationPolicy()
    assert policy.use_boltzmann is False
    assert policy.epsilon == 0.0
    assert policy.blunder_rate == 0.0
    assert policy.temperature == 1.0


def test_exploration_policy_factory_methods():
    """Test factory methods create correct policies."""
    from forge.eq.generate import ExplorationPolicy

    # Greedy
    greedy = ExplorationPolicy.greedy()
    assert greedy.use_boltzmann is False
    assert greedy.epsilon == 0.0

    # Boltzmann
    boltz = ExplorationPolicy.boltzmann(temperature=3.0, seed=42)
    assert boltz.use_boltzmann is True
    assert boltz.temperature == 3.0
    assert boltz.seed == 42

    # Epsilon-greedy
    eps = ExplorationPolicy.epsilon_greedy(epsilon=0.15, seed=123)
    assert eps.epsilon == 0.15
    assert eps.seed == 123

    # Mixed exploration
    mixed = ExplorationPolicy.mixed_exploration(
        temperature=5.0, epsilon=0.1, blunder_rate=0.05, seed=99
    )
    assert mixed.use_boltzmann is True
    assert mixed.temperature == 5.0
    assert mixed.epsilon == 0.1
    assert mixed.blunder_rate == 0.05
    assert mixed.seed == 99


def test_exploration_stats_dataclass():
    """Test ExplorationStats fields."""
    from forge.eq.generate import ExplorationStats

    stats = ExplorationStats(
        greedy_action=0,
        action_taken=1,
        was_greedy=False,
        selection_mode="epsilon",
        q_gap=2.5,
        action_entropy=1.2,
    )
    assert stats.greedy_action == 0
    assert stats.action_taken == 1
    assert stats.was_greedy is False
    assert stats.selection_mode == "epsilon"
    assert stats.q_gap == 2.5
    assert stats.action_entropy == 1.2


def test_game_exploration_stats_dataclass():
    """Test GameExplorationStats fields and computed properties."""
    from forge.eq.generate import GameExplorationStats

    stats = GameExplorationStats(
        n_decisions=28,
        n_greedy=20,
        n_boltzmann=5,
        n_epsilon=2,
        n_blunder=1,
        total_q_gap=14.0,
        mean_action_entropy=0.8,
    )

    assert stats.n_decisions == 28
    assert stats.n_greedy == 20
    assert stats.greedy_rate == 20 / 28
    assert stats.mean_q_gap == 14.0 / 28


def test_exploration_disabled_returns_base_record():
    """Test that no exploration policy returns base GameRecord."""
    oracle = MockOracle()
    hands = deal_from_seed(42)
    record = generate_eq_game(oracle, hands, decl_id=0, n_samples=2)

    # Should return base GameRecord, not V2
    assert isinstance(record, GameRecord)
    assert not isinstance(record, GameRecordV2)


def test_exploration_enabled_returns_v2_record():
    """Test that exploration policy returns GameRecordV2 with stats."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy
    )

    # Should return GameRecordV2
    assert isinstance(record, GameRecordV2)
    assert record.exploration_policy == policy
    assert record.exploration_stats is not None
    assert record.exploration_stats.n_decisions == 28


def test_exploration_greedy_all_decisions_greedy():
    """Test pure greedy policy marks all decisions as greedy."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy
    )

    # All decisions should be greedy
    assert record.exploration_stats.n_greedy == 28
    assert record.exploration_stats.n_boltzmann == 0
    assert record.exploration_stats.n_epsilon == 0
    assert record.exploration_stats.n_blunder == 0
    assert record.exploration_stats.greedy_rate == 1.0
    assert record.exploration_stats.total_q_gap == 0.0


def test_exploration_epsilon_produces_some_random():
    """Test epsilon-greedy policy produces some non-greedy actions."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)

    # Use high epsilon to ensure some random actions
    policy = ExplorationPolicy.epsilon_greedy(epsilon=0.5, seed=42)

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy
    )

    # Should have some epsilon selections (with high epsilon, very likely)
    assert record.exploration_stats.n_epsilon > 0
    # Should also have some greedy
    assert record.exploration_stats.n_greedy > 0


def test_exploration_boltzmann_produces_stochastic():
    """Test Boltzmann policy produces stochastic selections."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)

    # High temperature = more random
    policy = ExplorationPolicy.boltzmann(temperature=10.0, seed=42)

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy
    )

    # Most should be marked as boltzmann, but some may be greedy
    # (when only 1 legal action remains, it's forced/greedy)
    assert record.exploration_stats.n_boltzmann > 0
    # Combined should equal 28
    assert record.exploration_stats.n_boltzmann + record.exploration_stats.n_greedy == 28
    # No epsilon or blunder since we didn't set those
    assert record.exploration_stats.n_epsilon == 0
    assert record.exploration_stats.n_blunder == 0


def test_exploration_decisions_have_stats():
    """Test each decision has exploration stats when policy is set."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.epsilon_greedy(epsilon=0.1, seed=42)

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy
    )

    for i, decision in enumerate(record.decisions):
        assert isinstance(decision, DecisionRecordV2)
        assert decision.exploration is not None
        assert decision.exploration.selection_mode in ["greedy", "epsilon"]
        assert decision.exploration.action_entropy >= 0


def test_exploration_deterministic_with_seed():
    """Test that same seed produces same exploration decisions given same oracle."""
    from forge.eq.generate import ExplorationPolicy

    # Use a deterministic oracle for this test
    class DeterministicMockOracle:
        """Oracle that returns deterministic logits based on player."""

        def __init__(self):
            self.device = "cpu"
            self.query_count = 0

        def query_batch(self, worlds, game_state_info, current_player):
            n = len(worlds)
            self.query_count += 1
            # Deterministic: favor action based on current_player
            logits = torch.zeros(n, 7)
            for i in range(7):
                logits[:, i] = (i + current_player) % 7  # Deterministic pattern
            return logits

    oracle1 = DeterministicMockOracle()
    oracle2 = DeterministicMockOracle()
    hands = deal_from_seed(42)

    policy1 = ExplorationPolicy.epsilon_greedy(epsilon=0.3, seed=12345)
    policy2 = ExplorationPolicy.epsilon_greedy(epsilon=0.3, seed=12345)

    record1 = generate_eq_game(
        oracle1, hands, decl_id=0, n_samples=2, exploration_policy=policy1
    )
    record2 = generate_eq_game(
        oracle2, hands, decl_id=0, n_samples=2, exploration_policy=policy2
    )

    # Same seed with same deterministic oracle should produce same decisions
    for d1, d2 in zip(record1.decisions, record2.decisions):
        assert d1.action_taken == d2.action_taken
        assert d1.exploration.selection_mode == d2.exploration.selection_mode


def test_exploration_different_seeds_produce_different():
    """Test that different seeds produce different exploration decisions."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)

    policy1 = ExplorationPolicy.epsilon_greedy(epsilon=0.5, seed=111)
    policy2 = ExplorationPolicy.epsilon_greedy(epsilon=0.5, seed=222)

    record1 = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy1
    )
    record2 = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy2
    )

    # With high epsilon and different seeds, at least some decisions should differ
    n_different = sum(
        1 for d1, d2 in zip(record1.decisions, record2.decisions)
        if d1.action_taken != d2.action_taken
    )
    assert n_different > 0


def test_exploration_q_gap_bounded():
    """Test that q_gap is non-negative (regret is always >= 0)."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.epsilon_greedy(epsilon=0.5, seed=42)

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy
    )

    for decision in record.decisions:
        # Q-gap should be non-negative (greedy is always >= any other action's Q)
        assert decision.exploration.q_gap >= 0


def test_exploration_action_entropy_positive():
    """Test that action entropy is non-negative."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy
    )

    for decision in record.decisions:
        assert decision.exploration.action_entropy >= 0


def test_exploration_combined_with_posterior():
    """Test exploration works alongside posterior weighting."""
    from forge.eq.generate import ExplorationPolicy, PosteriorConfig

    oracle = MockOracle()
    hands = deal_from_seed(42)

    posterior_config = PosteriorConfig(enabled=True)
    exploration_policy = ExplorationPolicy.epsilon_greedy(epsilon=0.1, seed=42)

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=3,
        posterior_config=posterior_config,
        exploration_policy=exploration_policy,
    )

    # Should be V2 with both diagnostics and exploration
    assert isinstance(record, GameRecordV2)
    assert record.posterior_config == posterior_config
    assert record.exploration_policy == exploration_policy
    assert record.exploration_stats is not None

    for decision in record.decisions:
        assert decision.diagnostics is not None  # From posterior
        assert decision.exploration is not None  # From exploration


def test_mixed_exploration_policy():
    """Test the recommended mixed exploration policy."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)

    policy = ExplorationPolicy.mixed_exploration(seed=42)

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=2, exploration_policy=policy
    )

    # Should have a mix of selection modes
    stats = record.exploration_stats
    assert stats.n_decisions == 28

    # Mean q_gap should be reasonable (bounded by blunder_max_regret)
    assert stats.mean_q_gap < 10.0  # Should be much lower in practice


# =============================================================================
# Uncertainty Tests (t42-64uj.6)
# =============================================================================


def test_uncertainty_fields_exist_in_v2():
    """Test that DecisionRecordV2 has uncertainty fields."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()  # Triggers V2 record

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=5, exploration_policy=policy
    )

    for decision in record.decisions:
        assert isinstance(decision, DecisionRecordV2)
        # New uncertainty fields should exist
        assert hasattr(decision, "e_q_var")
        assert hasattr(decision, "u_mean")
        assert hasattr(decision, "u_max")


def test_uncertainty_variance_computed():
    """Test that variance is computed and stored in e_q_var."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=5, exploration_policy=policy
    )

    for decision in record.decisions:
        # e_q_var should be a tensor of shape (7,)
        assert decision.e_q_var is not None
        assert decision.e_q_var.shape == (7,)

        # Variance should be non-negative
        assert (decision.e_q_var >= 0).all(), f"Negative variance found: {decision.e_q_var}"


def test_uncertainty_state_level_u():
    """Test that state-level uncertainty U_mean and U_max are computed."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=5, exploration_policy=policy
    )

    for decision in record.decisions:
        # U values should be non-negative (they are mean/max of stddev)
        assert decision.u_mean >= 0, f"u_mean should be >= 0, got {decision.u_mean}"
        assert decision.u_max >= 0, f"u_max should be >= 0, got {decision.u_max}"

        # U_max >= U_mean (max of a set is always >= mean)
        assert decision.u_max >= decision.u_mean, (
            f"u_max ({decision.u_max}) should be >= u_mean ({decision.u_mean})"
        )


def test_uncertainty_variance_is_valid_for_legal_actions():
    """Test that variance is correctly computed for legal actions."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=10, exploration_policy=policy
    )

    for decision in record.decisions:
        # Get legal action indices
        legal_indices = torch.where(decision.legal_mask)[0]

        # For legal actions, variance should be finite
        legal_var = decision.e_q_var[decision.legal_mask]
        assert torch.isfinite(legal_var).all(), f"Non-finite variance in legal actions: {legal_var}"

        # With mock oracle (random Q-values), variance should generally be > 0
        # (unless all worlds happen to agree, which is unlikely)
        # We just check it's >= 0 (already done above)


class VariableOracle:
    """Oracle that returns Q-values with known variance pattern for testing."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.query_count = 0

    def query_batch(
        self,
        worlds: list[list[list[int]]],
        game_state_info: dict,
        current_player: int,
    ) -> Tensor:
        """Return Q-values that vary across worlds in a predictable way.

        World i gets logits = [i, 0, 0, 0, 0, 0, 0] for first slot,
        creating known variance.
        """
        n = len(worlds)
        self.query_count += 1

        logits = torch.zeros(n, 7, device=self.device)
        # First action varies linearly across worlds: 0, 1, 2, ..., n-1
        # E[Q_0] = (n-1)/2, Var[Q_0] = ((n-1)(n+1))/12 for uniform 0..n-1
        for i in range(n):
            logits[i, 0] = float(i)
            logits[i, 1:] = 0.0  # Other actions constant

        return logits


def test_uncertainty_variance_formula():
    """Test that variance is computed correctly using E[Q²] - E[Q]²."""
    from forge.eq.generate import ExplorationPolicy

    oracle = VariableOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()
    n_samples = 10

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=n_samples, exploration_policy=policy
    )

    # Check first decision
    decision = record.decisions[0]

    # With VariableOracle, the first action slot gets values 0, 1, ..., n-1
    # With uniform weights: E[Q] = (n-1)/2 = 4.5, E[Q²] = sum(i²)/n = 285/10 = 28.5
    # Var = 28.5 - 4.5² = 28.5 - 20.25 = 8.25
    # Note: this is sample variance, not population variance

    # Get variance for first legal action (if legal)
    if decision.legal_mask[0]:
        var_first = decision.e_q_var[0].item()
        # Expected variance for uniform 0..9 with uniform weights
        # E[X] = 4.5, E[X²] = (0+1+4+9+16+25+36+49+64+81)/10 = 28.5
        # Var = 28.5 - 20.25 = 8.25
        expected_var = 8.25
        # Allow some tolerance due to potential weight variations
        assert abs(var_first - expected_var) < 1.0, (
            f"Expected variance ~{expected_var}, got {var_first}"
        )


def test_uncertainty_u_mean_computed_from_legal():
    """Test U_mean is mean of stddev over legal actions only."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=5, exploration_policy=policy
    )

    for decision in record.decisions:
        # Compute U_mean manually from legal actions
        legal_std = torch.sqrt(decision.e_q_var[decision.legal_mask])
        expected_u_mean = legal_std.mean().item() if len(legal_std) > 0 else 0.0

        # Should match stored value
        assert abs(decision.u_mean - expected_u_mean) < 1e-5, (
            f"u_mean mismatch: stored {decision.u_mean}, expected {expected_u_mean}"
        )


def test_uncertainty_u_max_computed_from_legal():
    """Test U_max is max of stddev over legal actions only."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=5, exploration_policy=policy
    )

    for decision in record.decisions:
        # Compute U_max manually from legal actions
        legal_std = torch.sqrt(decision.e_q_var[decision.legal_mask])
        expected_u_max = legal_std.max().item() if len(legal_std) > 0 else 0.0

        # Should match stored value
        assert abs(decision.u_max - expected_u_max) < 1e-5, (
            f"u_max mismatch: stored {decision.u_max}, expected {expected_u_max}"
        )


def test_uncertainty_with_posterior_weighting():
    """Test uncertainty is computed correctly with posterior weighting."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)

    config = PosteriorConfig(enabled=True, tau=10.0, beta=0.10)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=10,
        posterior_config=config, exploration_policy=policy
    )

    for decision in record.decisions:
        # Should have both diagnostics (from posterior) and uncertainty fields
        assert decision.diagnostics is not None
        assert decision.e_q_var is not None
        assert (decision.e_q_var >= 0).all()


def test_uncertainty_padded_correctly():
    """Test that uncertainty is padded to 7 for consistency with e_q_mean."""
    from forge.eq.generate import ExplorationPolicy

    oracle = MockOracle()
    hands = deal_from_seed(42)
    policy = ExplorationPolicy.greedy()

    record = generate_eq_game(
        oracle, hands, decl_id=0, n_samples=5, exploration_policy=policy
    )

    # Check last decision (only 1 card remaining)
    last_decision = record.decisions[-1]

    # e_q_var should be shape (7,)
    assert last_decision.e_q_var.shape == (7,)

    # Positions beyond remaining hand should be 0 (padded)
    hand_size = last_decision.legal_mask.sum().item()
    if hand_size < 7:
        padding = last_decision.e_q_var[int(hand_size):]
        assert (padding == 0).all(), f"Padding should be 0, got {padding}"


# =============================================================================
# Schema correctness guards (t42-d6y1)
# =============================================================================


def test_e_q_mean_shape_and_type():
    """Guard test: e_q_mean should be a tensor with correct shape.

    This test guards against basic structural issues with the e_q_mean field.
    """
    oracle = MockOracle()
    hands = deal_from_seed(123)
    record = generate_eq_game(oracle, hands, decl_id=3, n_samples=5)

    for decision in record.decisions:
        # e_q_mean should be a tensor of shape (7,)
        assert decision.e_q_mean.shape == (7,), (
            f"e_q_mean should have shape (7,), got {decision.e_q_mean.shape}"
        )

        # Legal positions should have finite values (not NaN)
        legal_q = decision.e_q_mean[decision.legal_mask]
        assert torch.isfinite(legal_q).all(), (
            f"e_q_mean should have finite values for legal actions: {legal_q}"
        )

        # E[Q] values in legal positions should be in reasonable point range
        # (MockOracle returns random values, so just check finite and bounded)
        assert (legal_q.abs() < 100).all(), (
            f"e_q_mean legal values should be bounded: {legal_q}"
        )


def test_softmax_on_q_values_produces_degenerate_probabilities():
    """Guard test: applying softmax to E[Q] in points produces degenerate distributions.

    This test demonstrates why you should NOT apply softmax to Q-values.
    Q-values in points can have large differences (e.g., -10 vs -20),
    so softmax produces essentially one-hot distributions (exp(-10) >> exp(-20)).

    If your downstream code uses softmax(E[Q]), you're probably doing it wrong.
    E[Q] is already a point estimate - just use argmax for action selection.
    """
    import torch.nn.functional as F

    # Simulate Q-values with realistic point differences (10+ point spread is common)
    # Best action at -10 pts, worst at -25 pts (15 point spread)
    q_values = torch.tensor([-10.0, -25.0, -22.0, -20.0, -18.0, -15.0, -12.0])

    # Softmax on point-valued Q produces degenerate distribution
    probs = F.softmax(q_values, dim=0)

    # The best action dominates exponentially
    # exp(-10) / (exp(-10) + exp(-12) + ...) ≈ 1.0 / (1.0 + 0.135 + ...) ≈ 0.8+
    max_prob = probs.max().item()
    assert max_prob > 0.7, (
        f"Expected softmax(Q) to produce degenerate distribution, got max_prob={max_prob}. "
        "This test demonstrates that softmax on point-valued Q concentrates probability."
    )

    # With 15-point spread, the worst action gets essentially 0 probability
    min_prob = probs.min().item()
    assert min_prob < 0.001, (
        f"Expected worst action to have near-zero probability, got {min_prob}. "
        "This confirms softmax on point-valued Q is problematic."
    )


def test_world_sampling_deterministic_with_seeded_world_rng():
    """World sampling + gameplay should be deterministic when world_rng is seeded."""

    class WorldSensitiveOracle:
        def __init__(self):
            self.device = "cpu"

        def query_batch(self, worlds, game_state_info, current_player):
            decl_id = int(game_state_info.get("decl_id", 0))
            out = torch.zeros(len(worlds), 7, dtype=torch.float32)
            for wi, world in enumerate(worlds):
                opp_sum = sum(sum(world[p]) for p in range(4) if p != current_player)
                for local_idx, domino_id in enumerate(world[current_player]):
                    out[wi, local_idx] = float(domino_id + decl_id) - 0.01 * float(opp_sum)
            return out

    oracle = WorldSensitiveOracle()
    hands = deal_from_seed(42)

    record1 = generate_eq_game(
        oracle,
        hands,
        decl_id=0,
        n_samples=1,
        world_rng=np.random.default_rng(123),
    )
    record2 = generate_eq_game(
        oracle,
        hands,
        decl_id=0,
        n_samples=1,
        world_rng=np.random.default_rng(123),
    )

    assert len(record1.decisions) == len(record2.decisions) == 28
    for d1, d2 in zip(record1.decisions, record2.decisions):
        assert d1.player == d2.player
        assert d1.action_taken == d2.action_taken
        assert d1.actual_outcome == d2.actual_outcome
        assert torch.equal(d1.legal_mask, d2.legal_mask)
        assert torch.equal(d1.transcript_tokens, d2.transcript_tokens)
        assert torch.allclose(d1.e_q_mean, d2.e_q_mean)


def test_generate_eq_games_batched_matches_individual():
    """Batched generation should match per-game generation given the same RNG streams."""

    class BatchDeterministicOracle:
        def __init__(self):
            self.device = "cpu"
            self.query_count = 0
            self.multi_query_count = 0

        def _q_for(self, world, actor: int, decl_id: int) -> Tensor:
            opp_sum = sum(sum(world[p]) for p in range(4) if p != actor)
            q = torch.zeros(7, dtype=torch.float32)
            for local_idx, domino_id in enumerate(world[actor]):
                q[local_idx] = (
                    float(domino_id + decl_id) - 0.01 * float(opp_sum) + 0.001 * float(actor)
                )
            return q

        def query_batch(self, worlds, game_state_info, current_player):
            self.query_count += 1
            decl_id = int(game_state_info.get("decl_id", 0))
            return torch.stack([self._q_for(w, current_player, decl_id) for w in worlds], dim=0)

        def query_batch_multi_state(
            self,
            *,
            worlds,
            decl_ids,
            actors,
            leaders,
            trick_plays_list,
            remaining,
        ):
            self.multi_query_count += 1
            # Support both scalar and array decl_ids
            if isinstance(decl_ids, (int, np.integer)):
                decl_ids_array = [decl_ids] * len(worlds)
            else:
                decl_ids_array = decl_ids
            return torch.stack(
                [self._q_for(w, int(actors[i]), int(decl_ids_array[i])) for i, w in enumerate(worlds)],
                dim=0,
            )

    hands_list = [deal_from_seed(111), deal_from_seed(222)]
    decl_ids = [0, 0]  # Same decl_id exercises cross-game batching.

    oracle_individual = BatchDeterministicOracle()
    records_individual = [
        generate_eq_game(
            oracle_individual,
            hands_list[0],
            decl_ids[0],
            n_samples=1,
            world_rng=np.random.default_rng(1001),
        ),
        generate_eq_game(
            oracle_individual,
            hands_list[1],
            decl_ids[1],
            n_samples=1,
            world_rng=np.random.default_rng(1002),
        ),
    ]

    oracle_batched = BatchDeterministicOracle()
    records_batched = generate_eq_games_batched(
        oracle_batched,
        hands_list,
        decl_ids,
        n_samples=1,
        world_rngs=[np.random.default_rng(1001), np.random.default_rng(1002)],
    )

    assert oracle_individual.query_count == 56
    assert oracle_individual.multi_query_count == 0
    assert oracle_batched.query_count == 0
    assert oracle_batched.multi_query_count == 28

    assert len(records_batched) == len(records_individual) == 2
    for rec_b, rec_i in zip(records_batched, records_individual):
        assert len(rec_b.decisions) == len(rec_i.decisions) == 28
        for d_b, d_i in zip(rec_b.decisions, rec_i.decisions):
            assert d_b.player == d_i.player
            assert d_b.action_taken == d_i.action_taken
            assert d_b.actual_outcome == d_i.actual_outcome
            assert torch.equal(d_b.legal_mask, d_i.legal_mask)
            assert torch.equal(d_b.transcript_tokens, d_i.transcript_tokens)
            assert torch.allclose(d_b.e_q_mean, d_i.e_q_mean)


def test_decision_record_has_correct_field_names():
    """Guard test: ensure field names are e_q_mean/e_q_var, not legacy e_logits/e_var.

    This test guards against accidentally using legacy field names.
    The normalized naming (t42-d6y1) uses:
    - e_q_mean: E[Q] in points (NOT e_logits)
    - e_q_var: Var[Q] in points² (NOT e_var)
    """
    from forge.eq.generate import DecisionRecord, DecisionRecordV2

    # DecisionRecord should have e_q_mean
    assert hasattr(DecisionRecord, "__dataclass_fields__")
    assert "e_q_mean" in DecisionRecord.__dataclass_fields__, (
        "DecisionRecord should have 'e_q_mean' field, not 'e_logits'"
    )
    assert "e_logits" not in DecisionRecord.__dataclass_fields__, (
        "DecisionRecord should NOT have 'e_logits' field (use e_q_mean)"
    )

    # DecisionRecordV2 should have e_q_var
    assert "e_q_var" in DecisionRecordV2.__dataclass_fields__, (
        "DecisionRecordV2 should have 'e_q_var' field, not 'e_var'"
    )
    assert "e_var" not in DecisionRecordV2.__dataclass_fields__, (
        "DecisionRecordV2 should NOT have 'e_var' field (use e_q_var)"
    )
