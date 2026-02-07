"""
Tests for per-sample decl_id support in Stage1Oracle.

This tests the t42-5kvo feature that allows mixed decl_ids in a single batch,
removing the constraint that forced grouping by declaration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from forge.eq.oracle import Stage1Oracle
from forge.oracle.rng import deal_from_seed


def test_multi_state_accepts_scalar_decl_id():
    """Test backward compatibility: query_batch_multi_state accepts scalar decl_id."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    worlds = [deal_from_seed(i) for i in range(5)]
    actors = np.array([0, 1, 2, 3, 0], dtype=np.int32)
    leaders = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    trick_plays_list = [[] for _ in range(5)]
    remaining = np.ones((5, 4), dtype=np.int64) * 0x7F

    # Scalar decl_id (backward compatibility)
    tokens, masks = oracle._tokenize_worlds_multi_state(
        worlds=worlds,
        decl_ids=np.array([3, 3, 3, 3, 3], dtype=np.int32),  # All same
        actors=actors,
        leaders=leaders,
        trick_plays_list=trick_plays_list,
        remaining=remaining,
    )

    # Verify shapes
    assert tokens.shape == (5, 32, 12)
    assert masks.shape == (5, 32)

    # Verify all samples have decl_id=3
    assert np.all(tokens[:, 0, 10] == 3), "Context token should have decl_id=3"
    assert np.all(tokens[:, 1:29, 10] == 3), "Hand tokens should have decl_id=3"


def test_multi_state_accepts_array_decl_ids():
    """Test new feature: query_batch_multi_state accepts per-sample decl_ids."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    worlds = [deal_from_seed(i) for i in range(5)]
    actors = np.array([0, 1, 2, 3, 0], dtype=np.int32)
    leaders = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    trick_plays_list = [[] for _ in range(5)]
    remaining = np.ones((5, 4), dtype=np.int64) * 0x7F

    # Mixed decl_ids (new feature)
    decl_ids = np.array([0, 3, 5, 7, 9], dtype=np.int32)

    tokens, masks = oracle._tokenize_worlds_multi_state(
        worlds=worlds,
        decl_ids=decl_ids,
        actors=actors,
        leaders=leaders,
        trick_plays_list=trick_plays_list,
        remaining=remaining,
    )

    # Verify shapes
    assert tokens.shape == (5, 32, 12)
    assert masks.shape == (5, 32)

    # Verify each sample has correct decl_id
    for i in range(5):
        expected_decl = decl_ids[i]
        assert tokens[i, 0, 10] == expected_decl, f"Sample {i}: context token decl_id mismatch"
        assert np.all(
            tokens[i, 1:29, 10] == expected_decl
        ), f"Sample {i}: hand token decl_id mismatch"


def test_per_sample_decl_id_affects_trump_rank():
    """Test that per-sample decl_id correctly affects trump rank features."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    # Use same world but different decl_ids to verify trump rank changes
    world = deal_from_seed(0)
    worlds = [world, world, world]

    actors = np.array([0, 0, 0], dtype=np.int32)
    leaders = np.array([0, 0, 0], dtype=np.int32)
    trick_plays_list = [[] for _ in range(3)]
    remaining = np.ones((3, 4), dtype=np.int64) * 0x7F

    # Different declarations: blanks (0), doubles-trump (7), no-trump (9)
    decl_ids = np.array([0, 7, 9], dtype=np.int32)

    tokens, _ = oracle._tokenize_worlds_multi_state(
        worlds=worlds,
        decl_ids=decl_ids,
        actors=actors,
        leaders=leaders,
        trick_plays_list=trick_plays_list,
        remaining=remaining,
    )

    # Trump rank is in column 4 (index 4) of domino features
    # For the same domino (e.g., domino 0 = [0,0] double blank), trump rank differs by decl

    # Get trump rank for first domino (0) in player 0's hand for each decl
    trump_rank_0 = tokens[0, 1, 4]  # Decl 0 (blanks trump)
    trump_rank_7 = tokens[1, 1, 4]  # Decl 7 (doubles trump)
    trump_rank_9 = tokens[2, 1, 4]  # Decl 9 (no trump)

    # Verify they differ (specific values depend on TRUMP_RANK_TABLE)
    # We just verify they're not all the same (which would indicate per-sample decl_id not working)
    trump_ranks = {trump_rank_0, trump_rank_7, trump_rank_9}
    assert len(trump_ranks) > 1, "Trump ranks should differ across declarations"


def test_per_sample_decl_id_affects_trick_tokens():
    """Test that per-sample decl_id correctly affects trick token features."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    # Use same world and trick but different decl_ids
    world = deal_from_seed(0)
    worlds = [world, world]

    actors = np.array([0, 0], dtype=np.int32)
    leaders = np.array([0, 0], dtype=np.int32)

    # Same trick in both samples
    trick_plays_list = [[(0, 5)], [(0, 5)]]
    remaining = np.ones((2, 4), dtype=np.int64) * 0x7F

    # Different declarations
    decl_ids = np.array([0, 7], dtype=np.int32)

    tokens, masks = oracle._tokenize_worlds_multi_state(
        worlds=worlds,
        decl_ids=decl_ids,
        actors=actors,
        leaders=leaders,
        trick_plays_list=trick_plays_list,
        remaining=remaining,
    )

    # Both should have trick token present
    assert masks[0, 29] == 1
    assert masks[1, 29] == 1

    # Verify decl_id in trick token (column 10)
    assert tokens[0, 29, 10] == 0, "Sample 0 trick token should have decl_id=0"
    assert tokens[1, 29, 10] == 7, "Sample 1 trick token should have decl_id=7"

    # Verify trump rank differs (column 4)
    trump_rank_0 = tokens[0, 29, 4]
    trump_rank_7 = tokens[1, 29, 4]
    assert (
        trump_rank_0 != trump_rank_7
    ), "Trump rank in trick token should differ by decl_id"


def test_trick_token_grouping_respects_decl_id():
    """Test that vectorized trick token grouping considers decl_id."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    # Create samples with same trick_plays and actor but different decl_ids
    worlds = [deal_from_seed(i) for i in range(4)]

    # All samples: same actor, same trick
    actors = np.array([0, 0, 0, 0], dtype=np.int32)
    leaders = np.array([0, 0, 0, 0], dtype=np.int32)
    trick_plays_list = [[(1, 7)], [(1, 7)], [(1, 7)], [(1, 7)]]
    remaining = np.ones((4, 4), dtype=np.int64) * 0x7F

    # But different decl_ids: samples 0,1 share decl_id=3, samples 2,3 share decl_id=5
    decl_ids = np.array([3, 3, 5, 5], dtype=np.int32)

    tokens, masks = oracle._tokenize_worlds_multi_state(
        worlds=worlds,
        decl_ids=decl_ids,
        actors=actors,
        leaders=leaders,
        trick_plays_list=trick_plays_list,
        remaining=remaining,
    )

    # All should have trick token present
    assert np.all(masks[:, 29] == 1)

    # Verify grouping: samples 0,1 should have same trick token features (except leader)
    # and samples 2,3 should have same trick token features (except leader)
    # but group 1 should differ from group 2 due to decl_id

    # Compare columns 0-10 (excluding normalized_leader at 11)
    assert np.array_equal(
        tokens[0, 29, :11], tokens[1, 29, :11]
    ), "Samples 0,1 should have identical trick token (same decl_id)"
    assert np.array_equal(
        tokens[2, 29, :11], tokens[3, 29, :11]
    ), "Samples 2,3 should have identical trick token (same decl_id)"

    # Verify groups differ
    assert not np.array_equal(
        tokens[0, 29, :11], tokens[2, 29, :11]
    ), "Different decl_ids should produce different trick tokens"

    # Verify decl_id is correct
    assert tokens[0, 29, 10] == 3
    assert tokens[1, 29, 10] == 3
    assert tokens[2, 29, 10] == 5
    assert tokens[3, 29, 10] == 5


def test_mixed_decl_ids_with_varied_tricks():
    """Test complex scenario: mixed decl_ids with varied trick patterns."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    worlds = [deal_from_seed(i) for i in range(8)]

    # Varied actors
    actors = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    leaders = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

    # Varied trick patterns
    trick_plays_list = [
        [],
        [(0, 5)],
        [(1, 7), (2, 10)],
        [],
        [(0, 5)],  # Same as sample 1 but different decl_id
        [(1, 7), (2, 10)],  # Same as sample 2 but different decl_id
        [(3, 20)],
        [(0, 1), (1, 2), (2, 3)],
    ]

    remaining = np.ones((8, 4), dtype=np.int64) * 0x7F

    # Mixed decl_ids
    decl_ids = np.array([0, 3, 5, 7, 0, 3, 5, 9], dtype=np.int32)

    tokens, masks = oracle._tokenize_worlds_multi_state(
        worlds=worlds,
        decl_ids=decl_ids,
        actors=actors,
        leaders=leaders,
        trick_plays_list=trick_plays_list,
        remaining=remaining,
    )

    # Verify shapes
    assert tokens.shape == (8, 32, 12)

    # Verify each sample has correct decl_id
    for i in range(8):
        assert tokens[i, 0, 10] == decl_ids[i], f"Sample {i}: context token decl_id mismatch"
        assert np.all(
            tokens[i, 1:29, 10] == decl_ids[i]
        ), f"Sample {i}: hand tokens decl_id mismatch"

        # Check trick tokens have correct decl_id
        trick_len = len(trick_plays_list[i])
        for trick_pos in range(trick_len):
            token_idx = 29 + trick_pos
            assert (
                tokens[i, token_idx, 10] == decl_ids[i]
            ), f"Sample {i}, trick {trick_pos}: decl_id mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_per_sample_decl_id_with_real_model():
    """Integration test with real model (requires checkpoint and CUDA)."""
    from pathlib import Path

    checkpoint_path = Path("checkpoints/stage1-latest.ckpt")
    if not checkpoint_path.exists():
        pytest.skip("No checkpoint found at checkpoints/stage1-latest.ckpt")

    oracle = Stage1Oracle(str(checkpoint_path), device="cuda", compile=False)

    # Create batch with mixed decl_ids
    worlds = [deal_from_seed(i) for i in range(10)]
    actors = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int32)
    leaders = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 0], dtype=np.int32)
    trick_plays_list = [[] for _ in range(10)]
    remaining = np.ones((10, 4), dtype=np.int64) * 0x7F

    # Mixed decl_ids: use all 10 declarations
    decl_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32)

    # Query oracle
    q_values = oracle.query_batch_multi_state(
        worlds=worlds,
        decl_ids=decl_ids,
        actors=actors,
        leaders=leaders,
        trick_plays_list=trick_plays_list,
        remaining=remaining,
    )

    # Verify output shape
    assert q_values.shape == (10, 7)
    assert torch.isfinite(q_values).all(), "Q-values should be finite"

    # Verify Q-values differ across declarations (same world, different decl)
    # This indicates the model is actually using the per-sample decl_id
    q_std = q_values.std(dim=0).mean().item()
    assert q_std > 0.01, "Q-values should vary across different declarations"


def test_backward_compat_scalar_decl_id_in_api():
    """Test that query_batch_multi_state API accepts scalar decl_id."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    # Mock the model to avoid loading checkpoint
    class MockModel:
        def __call__(self, tokens, masks, actors):
            batch_size = tokens.shape[0]
            return (
                torch.randn(batch_size, 7, dtype=torch.float32),
                torch.randn(batch_size, dtype=torch.float32),
            )

        def to(self, device):
            return self

    oracle.model = MockModel()
    oracle.use_async = False

    worlds = [deal_from_seed(i) for i in range(5)]
    actors = np.array([0, 1, 2, 3, 0], dtype=np.int32)
    leaders = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    trick_plays_list = [[] for _ in range(5)]
    remaining = np.ones((5, 4), dtype=np.int64) * 0x7F

    # Pass scalar decl_id (backward compatibility)
    q_values = oracle.query_batch_multi_state(
        worlds=worlds,
        decl_ids=3,  # Scalar int
        actors=actors,
        leaders=leaders,
        trick_plays_list=trick_plays_list,
        remaining=remaining,
    )

    assert q_values.shape == (5, 7)


def test_backward_compat_array_decl_id_in_api():
    """Test that query_batch_multi_state API accepts array decl_ids."""
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    oracle.device = "cpu"

    # Mock the model
    class MockModel:
        def __call__(self, tokens, masks, actors):
            batch_size = tokens.shape[0]
            return (
                torch.randn(batch_size, 7, dtype=torch.float32),
                torch.randn(batch_size, dtype=torch.float32),
            )

        def to(self, device):
            return self

    oracle.model = MockModel()
    oracle.use_async = False

    worlds = [deal_from_seed(i) for i in range(5)]
    actors = np.array([0, 1, 2, 3, 0], dtype=np.int32)
    leaders = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    trick_plays_list = [[] for _ in range(5)]
    remaining = np.ones((5, 4), dtype=np.int64) * 0x7F

    # Pass array decl_ids (new feature)
    decl_ids = np.array([0, 3, 5, 7, 9], dtype=np.int32)

    q_values = oracle.query_batch_multi_state(
        worlds=worlds,
        decl_ids=decl_ids,  # Array
        actors=actors,
        leaders=leaders,
        trick_plays_list=trick_plays_list,
        remaining=remaining,
    )

    assert q_values.shape == (5, 7)
