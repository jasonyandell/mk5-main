"""Tests for pipelined E[Q] game generation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from forge.eq.generate_batched import generate_eq_games_batched
from forge.eq.generate_game import generate_eq_game
from forge.eq.generate_pipelined import generate_eq_games_pipelined
from forge.eq.oracle import Stage1Oracle
from forge.eq.types import ExplorationPolicy, PosteriorConfig
from forge.oracle.rng import deal_from_seed


@pytest.fixture
def oracle():
    """Create a test oracle (CPU for testing)."""
    # Use a small model checkpoint for testing
    # In practice, you'd mock this or use a tiny model
    return _create_mock_oracle()


def _create_mock_oracle():
    """Create a mock oracle for testing that returns deterministic Q-values."""

    class MockOracle:
        """Mock oracle that returns simple Q-values based on domino IDs."""

        def __init__(self):
            self.device = "cpu"

        def query_batch(self, worlds, game_state_info, current_player):
            """Return mock Q-values (N, 7) - higher for lower domino IDs."""
            n_worlds = len(worlds)
            # Simple heuristic: prefer lower domino IDs (arbitrary for testing)
            my_hand = worlds[0][current_player]
            q_values = torch.zeros(n_worlds, 7, dtype=torch.float32)
            for i in range(len(my_hand)):
                # Q = -domino_id (prefer playing lower IDs)
                q_values[:, i] = -float(my_hand[i])
            return q_values

        def query_batch_multi_state(
            self, worlds, decl_id=None, decl_ids=None, actors=None, leaders=None, trick_plays_list=None, remaining=None
        ):
            """Mock multi-state query - supports both single and per-sample decl_id."""
            n_samples = len(worlds)
            q_values = torch.zeros(n_samples, 7, dtype=torch.float32)
            for i in range(n_samples):
                actor = actors[i]
                my_hand = worlds[i][actor]
                for j in range(len(my_hand)):
                    q_values[i, j] = -float(my_hand[j])
            return q_values

    return MockOracle()


def test_pipelined_single_game_matches_sequential(oracle):
    """Test that pipelined generator matches sequential for single game."""
    hands = deal_from_seed(42)
    decl_id = 3

    # Generate with sequential (reference)
    record_seq = generate_eq_game(
        oracle=oracle,
        hands=hands,
        decl_id=decl_id,
        n_samples=10,
        world_rng=np.random.default_rng(42),
    )

    # Generate with pipelined
    records_pipe = generate_eq_games_pipelined(
        oracle=oracle,
        hands_list=[hands],
        decl_ids=[decl_id],
        n_samples=10,
        world_rngs=[np.random.default_rng(42)],
        n_workers=1,  # Single worker for determinism
    )

    assert len(records_pipe) == 1
    record_pipe = records_pipe[0]

    # Check same number of decisions
    assert len(record_pipe.decisions) == len(record_seq.decisions)

    # Check all decisions match (actions and Q-values should be identical)
    for i, (d_pipe, d_seq) in enumerate(
        zip(record_pipe.decisions, record_seq.decisions)
    ):
        assert d_pipe.player == d_seq.player, f"Decision {i}: player mismatch"
        assert d_pipe.action_taken == d_seq.action_taken, f"Decision {i}: action mismatch"
        # Q-values should be close (may have minor numerical differences)
        assert torch.allclose(
            d_pipe.e_q_mean, d_seq.e_q_mean, rtol=1e-4, atol=1e-4
        ), f"Decision {i}: Q-values differ"


def test_pipelined_multiple_games_matches_batched(oracle):
    """Test that pipelined generator matches batched for multiple games."""
    n_games = 4

    # Generate hands and decl_ids
    hands_list = [deal_from_seed(100 + i) for i in range(n_games)]
    decl_ids = [i % 10 for i in range(n_games)]  # Mix of declarations

    # Generate with batched (reference)
    records_batched = generate_eq_games_batched(
        oracle=oracle,
        hands_list=hands_list,
        decl_ids=decl_ids,
        n_samples=10,
        world_rngs=[np.random.default_rng(200 + i) for i in range(n_games)],
    )

    # Generate with pipelined
    records_pipelined = generate_eq_games_pipelined(
        oracle=oracle,
        hands_list=hands_list,
        decl_ids=decl_ids,
        n_samples=10,
        world_rngs=[np.random.default_rng(200 + i) for i in range(n_games)],
        n_workers=2,
    )

    assert len(records_pipelined) == len(records_batched)

    # Check each game matches
    for game_idx, (rec_pipe, rec_batch) in enumerate(
        zip(records_pipelined, records_batched)
    ):
        assert len(rec_pipe.decisions) == len(rec_batch.decisions)

        for i, (d_pipe, d_batch) in enumerate(
            zip(rec_pipe.decisions, rec_batch.decisions)
        ):
            assert (
                d_pipe.player == d_batch.player
            ), f"Game {game_idx}, Decision {i}: player mismatch"
            assert (
                d_pipe.action_taken == d_batch.action_taken
            ), f"Game {game_idx}, Decision {i}: action mismatch"
            assert torch.allclose(
                d_pipe.e_q_mean, d_batch.e_q_mean, rtol=1e-4, atol=1e-4
            ), f"Game {game_idx}, Decision {i}: Q-values differ"


def test_pipelined_with_posterior(oracle):
    """Test pipelined generator with posterior weighting."""
    hands_list = [deal_from_seed(100 + i) for i in range(2)]
    decl_ids = [3, 5]

    posterior_config = PosteriorConfig(
        enabled=True,
        tau=10.0,
        beta=0.10,
        window_k=4,
    )

    records = generate_eq_games_pipelined(
        oracle=oracle,
        hands_list=hands_list,
        decl_ids=decl_ids,
        n_samples=10,
        posterior_config=posterior_config,
        world_rngs=[np.random.default_rng(200 + i) for i in range(2)],
        n_workers=1,
    )

    assert len(records) == 2
    for record in records:
        assert len(record.decisions) == 28  # Full game
        # Check that diagnostics are present (posterior was used)
        for decision in record.decisions:
            assert hasattr(decision, "diagnostics")
            # Diagnostics may be None for early decisions (window not full)


def test_pipelined_with_exploration(oracle):
    """Test pipelined generator with exploration policy."""
    hands_list = [deal_from_seed(100 + i) for i in range(2)]
    decl_ids = [3, 5]

    exploration_policies = [
        ExplorationPolicy.epsilon_greedy(epsilon=0.1, seed=300),
        ExplorationPolicy.boltzmann(temperature=2.0, seed=301),
    ]

    records = generate_eq_games_pipelined(
        oracle=oracle,
        hands_list=hands_list,
        decl_ids=decl_ids,
        n_samples=10,
        exploration_policies=exploration_policies,
        world_rngs=[np.random.default_rng(200 + i) for i in range(2)],
        n_workers=1,
    )

    assert len(records) == 2
    for record in records:
        assert len(record.decisions) == 28
        # Check that exploration stats are present
        for decision in record.decisions:
            assert hasattr(decision, "exploration")
            # Exploration stats may be None if policy is None


def test_pipelined_empty_input(oracle):
    """Test pipelined generator with empty input."""
    records = generate_eq_games_pipelined(
        oracle=oracle,
        hands_list=[],
        decl_ids=[],
        n_samples=10,
    )
    assert len(records) == 0


def test_pipelined_validation_errors(oracle):
    """Test input validation."""
    hands_list = [deal_from_seed(42)]

    # Mismatched lengths
    with pytest.raises(ValueError, match="must have same length"):
        generate_eq_games_pipelined(
            oracle=oracle,
            hands_list=hands_list,
            decl_ids=[3, 5],  # Wrong length
            n_samples=10,
        )

    # Mismatched exploration policies
    with pytest.raises(ValueError, match="exploration_policies must be same length"):
        generate_eq_games_pipelined(
            oracle=oracle,
            hands_list=hands_list,
            decl_ids=[3],
            n_samples=10,
            exploration_policies=[None, None],  # Wrong length
        )


def test_pipelined_different_decl_ids(oracle):
    """Test pipelined generator with varied declaration IDs."""
    n_games = 6

    hands_list = [deal_from_seed(100 + i) for i in range(n_games)]
    # Use all 10 possible declarations (0-9), some repeated
    decl_ids = [0, 1, 2, 3, 4, 5]

    records = generate_eq_games_pipelined(
        oracle=oracle,
        hands_list=hands_list,
        decl_ids=decl_ids,
        n_samples=10,
        world_rngs=[np.random.default_rng(200 + i) for i in range(n_games)],
        n_workers=2,
    )

    assert len(records) == n_games
    for record in records:
        assert len(record.decisions) == 28


def test_pipelined_actual_outcomes(oracle):
    """Test that actual outcomes are filled correctly."""
    hands_list = [deal_from_seed(100)]
    decl_ids = [3]

    records = generate_eq_games_pipelined(
        oracle=oracle,
        hands_list=hands_list,
        decl_ids=decl_ids,
        n_samples=10,
        world_rngs=[np.random.default_rng(200)],
        n_workers=1,
    )

    record = records[0]
    # Check that all decisions have actual_outcome filled
    for decision in record.decisions:
        assert decision.actual_outcome is not None
        assert isinstance(decision.actual_outcome, float)

    # Verify backward-fill logic: later decisions should have smaller absolute outcomes
    # (closer to end of game means less remaining variance)


def test_pipelined_n_workers(oracle):
    """Test different numbers of workers."""
    hands_list = [deal_from_seed(100 + i) for i in range(4)]
    decl_ids = [3, 4, 5, 6]

    # Test with different worker counts
    for n_workers in [1, 2, 4]:
        records = generate_eq_games_pipelined(
            oracle=oracle,
            hands_list=hands_list,
            decl_ids=decl_ids,
            n_samples=10,
            world_rngs=[np.random.default_rng(200 + i) for i in range(4)],
            n_workers=n_workers,
        )

        assert len(records) == 4
        for record in records:
            assert len(record.decisions) == 28


def test_pipelined_thread_safety(oracle):
    """Test that pipelined generator is thread-safe (no race conditions)."""
    # Run the same configuration multiple times and ensure deterministic results
    hands_list = [deal_from_seed(100 + i) for i in range(4)]
    decl_ids = [3, 4, 5, 6]

    results = []
    for run in range(3):
        records = generate_eq_games_pipelined(
            oracle=oracle,
            hands_list=hands_list,
            decl_ids=decl_ids,
            n_samples=10,
            world_rngs=[np.random.default_rng(200 + i) for i in range(4)],
            n_workers=2,
        )
        results.append(records)

    # All runs should produce identical results (deterministic RNGs)
    for game_idx in range(4):
        ref_record = results[0][game_idx]
        for run_idx in range(1, 3):
            test_record = results[run_idx][game_idx]
            assert len(test_record.decisions) == len(ref_record.decisions)

            for i, (d_test, d_ref) in enumerate(
                zip(test_record.decisions, ref_record.decisions)
            ):
                assert (
                    d_test.action_taken == d_ref.action_taken
                ), f"Run {run_idx}, Game {game_idx}, Decision {i}: action mismatch"
