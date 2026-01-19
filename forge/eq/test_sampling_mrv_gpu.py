"""Tests for MRV GPU world sampler."""

import pytest
import torch
import numpy as np

from forge.eq.sampling_mrv_gpu import (
    sample_worlds_mrv_gpu,
    WorldSamplerMRV,
    _popcount_vectorized,
    _build_void_masks_vectorized,
    _pool_to_mask,
)
from forge.eq.sampling_gpu import CAN_FOLLOW
from forge.eq.sampling import sample_consistent_worlds, hand_violates_voids


class TestPopcount:
    """Test bit counting."""

    def test_popcount_zero(self):
        x = torch.tensor([0], dtype=torch.int64)
        assert _popcount_vectorized(x)[0] == 0

    def test_popcount_one(self):
        x = torch.tensor([1], dtype=torch.int64)
        assert _popcount_vectorized(x)[0] == 1

    def test_popcount_all_bits(self):
        # 28 bits set
        x = torch.tensor([(1 << 28) - 1], dtype=torch.int64)
        assert _popcount_vectorized(x)[0] == 28

    def test_popcount_mixed(self):
        x = torch.tensor([0b1010101010101010], dtype=torch.int64)
        assert _popcount_vectorized(x)[0] == 8


class TestPoolToMask:
    """Test pool to bitmask conversion."""

    def test_simple_pool(self):
        pools = torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.int32)
        masks = _pool_to_mask(pools)
        assert masks[0] == 0b111

    def test_sparse_pool(self):
        pools = torch.tensor([[0, 5, 10, -1]], dtype=torch.int32)
        masks = _pool_to_mask(pools)
        expected = (1 << 0) | (1 << 5) | (1 << 10)
        assert masks[0] == expected


class TestVoidMasks:
    """Test void mask building."""

    def test_no_voids(self):
        voids = torch.zeros(1, 3, 8, dtype=torch.bool)
        decl_ids = torch.tensor([9])  # NOTRUMP
        masks = _build_void_masks_vectorized(voids, decl_ids, 'cpu')
        # No voids means mask should be 0
        assert masks[0, 0] == 0
        assert masks[0, 1] == 0
        assert masks[0, 2] == 0

    def test_void_in_suit_0(self):
        voids = torch.zeros(1, 3, 8, dtype=torch.bool)
        voids[0, 0, 0] = True  # Player 0 void in suit 0
        decl_ids = torch.tensor([9])  # NOTRUMP
        masks = _build_void_masks_vectorized(voids, decl_ids, 'cpu')

        # Mask should have bits set for dominoes containing pip 0
        # that can follow suit 0
        can_follow = CAN_FOLLOW.cpu()
        expected = 0
        for d in range(28):
            if can_follow[d, 0, 9]:
                expected |= (1 << d)

        assert masks[0, 0] == expected


class TestMRVSamplerBasic:
    """Test basic MRV sampling functionality."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        n_games = 2
        n_samples = 5
        pool_size = 21

        # Full starting pool (21 dominoes for 3 opponents)
        pools = torch.arange(21).unsqueeze(0).expand(n_games, -1).clone()
        hand_sizes = torch.tensor([[7, 7, 7], [7, 7, 7]], dtype=torch.int32)
        voids = torch.zeros(n_games, 3, 8, dtype=torch.bool)
        decl_ids = torch.tensor([9, 9])  # NOTRUMP

        result = sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples, device='cpu')

        assert result.shape == (n_games, n_samples, 3, 7)

    def test_no_duplicate_dominoes(self):
        """Test that each domino appears exactly once across all hands."""
        n_games = 1
        n_samples = 10

        pools = torch.arange(21).unsqueeze(0)
        hand_sizes = torch.tensor([[7, 7, 7]], dtype=torch.int32)
        voids = torch.zeros(n_games, 3, 8, dtype=torch.bool)
        decl_ids = torch.tensor([9])

        result = sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples, device='cpu')

        for s in range(n_samples):
            all_dominoes = []
            for p in range(3):
                hand = result[0, s, p].numpy()
                valid = hand[hand >= 0].tolist()
                all_dominoes.extend(valid)

            # Should have exactly 21 unique dominoes
            assert len(all_dominoes) == 21, f"Sample {s}: got {len(all_dominoes)} dominoes"
            assert len(set(all_dominoes)) == 21, f"Sample {s}: duplicates found"

    def test_correct_hand_sizes(self):
        """Test that each hand has the correct number of dominoes."""
        n_games = 1
        n_samples = 5

        pools = torch.arange(15).unsqueeze(0)  # Smaller pool
        hand_sizes = torch.tensor([[5, 5, 5]], dtype=torch.int32)
        voids = torch.zeros(n_games, 3, 8, dtype=torch.bool)
        decl_ids = torch.tensor([9])

        result = sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples, device='cpu')

        for s in range(n_samples):
            for p in range(3):
                hand = result[0, s, p].numpy()
                valid = hand[hand >= 0]
                assert len(valid) == 5, f"Sample {s}, player {p}: expected 5, got {len(valid)}"


class TestMRVSamplerConstraints:
    """Test that MRV sampler respects void constraints."""

    def test_void_constraint_respected(self):
        """Test that samples respect void constraints."""
        n_games = 1
        n_samples = 20

        pools = torch.arange(21).unsqueeze(0)
        hand_sizes = torch.tensor([[7, 7, 7]], dtype=torch.int32)

        # Player 0 is void in suit 0 (pips containing 0)
        voids = torch.zeros(n_games, 3, 8, dtype=torch.bool)
        voids[0, 0, 0] = True

        decl_ids = torch.tensor([9])  # NOTRUMP

        result = sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples, device='cpu')

        # Check that player 0 never has a domino that can follow suit 0
        can_follow = CAN_FOLLOW.cpu()

        for s in range(n_samples):
            hand = result[0, s, 0].numpy()
            valid = hand[hand >= 0]
            for d in valid:
                assert not can_follow[d, 0, 9], (
                    f"Sample {s}: player 0 has domino {d} which can follow void suit 0"
                )

    def test_multiple_voids_respected(self):
        """Test with multiple void constraints."""
        n_games = 1
        n_samples = 20

        pools = torch.arange(21).unsqueeze(0)
        hand_sizes = torch.tensor([[7, 7, 7]], dtype=torch.int32)

        # Multiple voids
        voids = torch.zeros(n_games, 3, 8, dtype=torch.bool)
        voids[0, 0, 0] = True  # Player 0 void in 0s
        voids[0, 1, 1] = True  # Player 1 void in 1s
        voids[0, 2, 2] = True  # Player 2 void in 2s

        decl_ids = torch.tensor([9])

        result = sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples, device='cpu')

        can_follow = CAN_FOLLOW.cpu()

        for s in range(n_samples):
            for p, void_suit in [(0, 0), (1, 1), (2, 2)]:
                hand = result[0, s, p].numpy()
                valid = hand[hand >= 0]
                for d in valid:
                    assert not can_follow[d, void_suit, 9], (
                        f"Sample {s}: player {p} has domino {d} which can follow void suit {void_suit}"
                    )

    def test_matches_cpu_validation(self):
        """Test that MRV samples pass CPU validation."""
        n_games = 1
        n_samples = 10

        pools = torch.arange(21).unsqueeze(0)
        hand_sizes = torch.tensor([[7, 7, 7]], dtype=torch.int32)
        voids_tensor = torch.zeros(n_games, 3, 8, dtype=torch.bool)
        voids_tensor[0, 0, 3] = True  # Player 0 void in 3s
        voids_tensor[0, 1, 5] = True  # Player 1 void in 5s

        decl_ids = torch.tensor([9])

        result = sample_worlds_mrv_gpu(pools, hand_sizes, voids_tensor, decl_ids, n_samples, device='cpu')

        # Convert voids tensor to dict format for CPU validation
        voids_dict = {0: {3}, 1: {5}}

        for s in range(n_samples):
            for p in range(3):
                hand = result[0, s, p].numpy()
                valid = hand[hand >= 0].tolist()

                void_suits = voids_dict.get(p, set())
                violates = hand_violates_voids(valid, void_suits, decl_id=9)
                assert not violates, f"Sample {s}, player {p}: hand {valid} violates voids {void_suits}"


class TestWorldSamplerMRV:
    """Test the stateful WorldSamplerMRV class."""

    def test_basic_usage(self):
        """Test basic sampler usage."""
        sampler = WorldSamplerMRV(max_games=4, max_samples=20, device='cpu')

        pools = torch.arange(21).unsqueeze(0).expand(2, -1).clone()
        hand_sizes = torch.tensor([[7, 7, 7], [7, 7, 7]], dtype=torch.int32)
        voids = torch.zeros(2, 3, 8, dtype=torch.bool)
        decl_ids = torch.tensor([9, 9])

        result = sampler.sample(pools, hand_sizes, voids, decl_ids, n_samples=10)

        assert result.shape == (2, 10, 3, 7)

    def test_exceeds_max_games(self):
        """Test that exceeding max_games raises error."""
        sampler = WorldSamplerMRV(max_games=2, max_samples=10, device='cpu')

        pools = torch.arange(21).unsqueeze(0).expand(5, -1).clone()
        hand_sizes = torch.tensor([[7, 7, 7]] * 5, dtype=torch.int32)
        voids = torch.zeros(5, 3, 8, dtype=torch.bool)
        decl_ids = torch.tensor([9] * 5)

        with pytest.raises(ValueError, match="exceeds max_games"):
            sampler.sample(pools, hand_sizes, voids, decl_ids, n_samples=5)


class TestMRVSamplerDiversity:
    """Test that MRV sampler produces diverse samples."""

    def test_samples_are_different(self):
        """Test that different samples produce different hands."""
        n_games = 1
        n_samples = 100

        pools = torch.arange(21).unsqueeze(0)
        hand_sizes = torch.tensor([[7, 7, 7]], dtype=torch.int32)
        voids = torch.zeros(n_games, 3, 8, dtype=torch.bool)
        decl_ids = torch.tensor([9])

        result = sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples, device='cpu')

        # Check that not all samples are identical
        first_sample = result[0, 0]
        different_count = 0
        for s in range(1, n_samples):
            if not torch.equal(result[0, s], first_sample):
                different_count += 1

        # Should have significant diversity
        assert different_count > n_samples * 0.9, (
            f"Expected >90% different samples, got {different_count}/{n_samples-1}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMRVSamplerGPU:
    """Test MRV sampler on GPU."""

    def test_gpu_output_shape(self):
        """Test that GPU produces correct output shape."""
        n_games = 4
        n_samples = 50

        pools = torch.arange(21).unsqueeze(0).expand(n_games, -1).clone()
        hand_sizes = torch.tensor([[7, 7, 7]] * n_games, dtype=torch.int32)
        voids = torch.zeros(n_games, 3, 8, dtype=torch.bool)
        decl_ids = torch.tensor([9] * n_games)

        result = sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples, device='cuda')

        assert result.shape == (n_games, n_samples, 3, 7)
        assert result.device.type == 'cuda'

    def test_gpu_constraints_respected(self):
        """Test that GPU respects void constraints."""
        n_games = 2
        n_samples = 30

        pools = torch.arange(21).unsqueeze(0).expand(n_games, -1).clone()
        hand_sizes = torch.tensor([[7, 7, 7]] * n_games, dtype=torch.int32)
        voids = torch.zeros(n_games, 3, 8, dtype=torch.bool)
        voids[0, 0, 4] = True  # Game 0: player 0 void in 4s
        voids[1, 2, 6] = True  # Game 1: player 2 void in 6s

        decl_ids = torch.tensor([9, 9])

        result = sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples, device='cuda')

        can_follow = CAN_FOLLOW.cpu()
        result_cpu = result.cpu()

        # Check game 0, player 0 void in 4s
        for s in range(n_samples):
            hand = result_cpu[0, s, 0].numpy()
            valid = hand[hand >= 0]
            for d in valid:
                assert not can_follow[d, 4, 9], f"Game 0, sample {s}: void 4s violated"

        # Check game 1, player 2 void in 6s
        for s in range(n_samples):
            hand = result_cpu[1, s, 2].numpy()
            valid = hand[hand >= 0]
            for d in valid:
                assert not can_follow[d, 6, 9], f"Game 1, sample {s}: void 6s violated"
