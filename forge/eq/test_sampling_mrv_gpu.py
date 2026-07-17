"""Tests for MRV GPU world sampler."""

from collections import Counter
from random import Random

import pytest
import torch

from forge.eq.sampling_mrv_gpu import (
    SAMPLER_ALGORITHM,
    sample_worlds_mrv_gpu,
    WorldSamplerMRV,
    _build_suffix_completion_counts,
    _popcount_vectorized,
    _build_void_masks_vectorized,
    _pool_to_mask,
)
from forge.eq.sampling_gpu import CAN_FOLLOW
from forge.eq.enumeration_gpu import enumerate_worlds_cpu
from forge.eq.sampling import hand_violates_voids


def _sampler_devices() -> list[str]:
    """Every device the sampler can run on here; uniformity must hold on all.

    The MPS int64-gather defect produced valid but severely non-uniform
    worlds, so accelerator coverage cannot be CPU-only.
    """
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def _fixture_historical_dead_end():
    """Exact audit fixture where greedy MRV reached a dead end with p=1/3."""

    pools = torch.tensor([[3, 5, 7, 8, 14, 24]], dtype=torch.int32)
    hand_sizes = torch.tensor([[2, 2, 2]], dtype=torch.int32)
    voids = torch.zeros(1, 3, 8, dtype=torch.bool)
    voids[0, 0, 7] = True
    voids[0, 1, 3] = True
    voids[0, 2, 7] = True
    decl_ids = torch.tensor([0], dtype=torch.int32)
    return pools, hand_sizes, voids, decl_ids


def _fixture_valid_only_bias():
    """Exact audit fixture where greedy MRV was valid but non-uniform."""

    pools = torch.tensor([[0, 2, 4, 5, 6, 25]], dtype=torch.int32)
    hand_sizes = torch.tensor([[2, 2, 2]], dtype=torch.int32)
    voids = torch.zeros(1, 3, 8, dtype=torch.bool)
    voids[0, 1, 3:6] = True
    decl_ids = torch.tensor([6], dtype=torch.int32)
    return pools, hand_sizes, voids, decl_ids


def _canonical_world(world: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sorted(int(domino) for domino in hand if int(domino) >= 0))
        for hand in world
    )


def _exact_world_keys(
    pool: list[int],
    hand_sizes: list[int],
    voids: list[set[int]],
    decl_id: int,
) -> set[tuple[tuple[int, ...], ...]]:
    worlds = enumerate_worlds_cpu(
        pool=pool,
        known=[[], [], []],
        slots=hand_sizes,
        voids=voids,
        decl_id=decl_id,
    )
    return {
        tuple(tuple(sorted(hand)) for hand in world)
        for world in worlds
    }


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


class TestCompletionCounts:
    """Test the exact int64 suffix dynamic program directly."""

    def test_unconstrained_7_7_7_root_count_does_not_overflow(self):
        pools = torch.arange(21, dtype=torch.int64).unsqueeze(0)
        pool_sizes = torch.tensor([21], dtype=torch.int64)
        void_masks = torch.zeros(1, 3, dtype=torch.int64)

        _tiles, _active, _allowed, suffix = _build_suffix_completion_counts(
            pools, pool_sizes, void_masks
        )

        assert suffix.dtype == torch.int64
        assert int(suffix[0, 0, 7, 7, 7]) == 399_072_960

    def test_padding_positions_are_identity_transitions(self):
        pools = torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.int64)
        pool_sizes = torch.tensor([3], dtype=torch.int64)
        void_masks = torch.zeros(1, 3, dtype=torch.int64)

        _tiles, active, _allowed, suffix = _build_suffix_completion_counts(
            pools, pool_sizes, void_masks
        )

        assert active.tolist() == [[True, True, True, False, False]]
        assert torch.equal(suffix[:, 3], suffix[:, 4])
        assert torch.equal(suffix[:, 4], suffix[:, 5])
        assert int(suffix[0, 0, 1, 1, 1]) == 6

    def test_randomized_small_roots_match_exact_enumeration(self):
        """Bounded property check across order, padding, capacity, and rules."""

        rng = Random(20260711)
        for case in range(48):
            pool_size = rng.randrange(1, 9)
            pool = rng.sample(range(28), pool_size)

            while True:
                owners = [rng.randrange(3) for _ in range(pool_size)]
                hand_sizes = [owners.count(player) for player in range(3)]
                if max(hand_sizes) <= 7:
                    break

            decl_id = rng.randrange(10)
            void_sets = [
                {suit for suit in range(8) if rng.random() < 0.28}
                for _ in range(3)
            ]
            voids = torch.zeros(1, 3, 8, dtype=torch.bool)
            for player, suits in enumerate(void_sets):
                for suit in suits:
                    voids[0, player, suit] = True

            width = pool_size + rng.randrange(5)
            padded_pool = pool + [-1] * (width - pool_size)
            rng.shuffle(padded_pool)
            pools = torch.tensor([padded_pool], dtype=torch.int64)
            pool_sizes = torch.tensor([pool_size], dtype=torch.int64)
            void_masks = _build_void_masks_vectorized(
                voids, torch.tensor([decl_id]), "cpu"
            )

            _tiles, _active, _allowed, suffix = _build_suffix_completion_counts(
                pools, pool_sizes, void_masks
            )
            observed = int(suffix[0, 0, *hand_sizes])
            expected = len(
                enumerate_worlds_cpu(
                    pool=pool,
                    known=[[], [], []],
                    slots=hand_sizes,
                    voids=void_sets,
                    decl_id=decl_id,
                )
            )

            assert observed == expected, (
                f"case={case} pool={pool} hand_sizes={hand_sizes} "
                f"decl_id={decl_id} voids={void_sets}: "
                f"suffix={observed}, enumeration={expected}"
            )


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


class TestUniformCompletionRegression:
    """Regression coverage for the two exact sampler-audit failures."""

    @pytest.mark.parametrize("device", _sampler_devices())
    def test_historical_dead_end_fixture_returns_only_exact_worlds(self, device):
        pools, hand_sizes, voids, decl_ids = _fixture_historical_dead_end()
        exact = _exact_world_keys(
            pool=[3, 5, 7, 8, 14, 24],
            hand_sizes=[2, 2, 2],
            voids=[{7}, {3}, {7}],
            decl_id=0,
        )
        assert len(exact) == 12

        torch.manual_seed(20260711)
        result = sample_worlds_mrv_gpu(
            pools, hand_sizes, voids, decl_ids, n_samples=2_000, device=device
        ).cpu()
        observed = {_canonical_world(world) for world in result[0]}

        # The historical implementation emitted malformed worlds one third of
        # the time by injecting domino 0.  Every result now belongs to the
        # exact valid support, and this sample reaches the whole support.
        assert observed == exact
        assert not (result == 0).any()

    @pytest.mark.parametrize("device", _sampler_devices())
    def test_valid_only_bias_fixture_is_uniform_against_enumeration(self, device):
        pools, hand_sizes, voids, decl_ids = _fixture_valid_only_bias()
        exact = _exact_world_keys(
            pool=[0, 2, 4, 5, 6, 25],
            hand_sizes=[2, 2, 2],
            voids=[set(), {3, 4, 5}, set()],
            decl_id=6,
        )
        assert len(exact) == 60

        n_samples = 60_000
        torch.manual_seed(20260711)
        result = sample_worlds_mrv_gpu(
            pools, hand_sizes, voids, decl_ids, n_samples=n_samples, device=device
        ).cpu()
        counts = Counter(_canonical_world(world) for world in result[0])

        assert set(counts) == exact
        expected = n_samples / len(exact)
        chi_square = sum(
            (counts[world] - expected) ** 2 / expected
            for world in exact
        )
        # Fixed-seed goodness-of-fit guard.  df=59 has mean 59; this generous
        # bound is stable for a uniform source, while the audited greedy MRV
        # distribution (TVD 0.0333) is far outside it at this sample count.
        assert chi_square < 130.0

    @pytest.mark.parametrize("device", _sampler_devices())
    def test_float32_unsafe_root_count_regime_valid_and_uniform(self, device):
        """21-tile unconstrained pool: root count 399,072,960 > 2^24.

        The two audit fixtures above use 6-tile pools whose counts sit inside
        float32's exact-integer range, so an accelerator int64-through-float32
        regression (the repaired MPS defect) would keep them green while
        silently biasing every real early-game pool. This test exercises the
        exact regime the 2026-07-06 corpus was contaminated in.
        """
        pools = torch.arange(21, dtype=torch.int32).unsqueeze(0)
        hand_sizes = torch.tensor([[7, 7, 7]], dtype=torch.int32)
        voids = torch.zeros(1, 3, 8, dtype=torch.bool)
        decl_ids = torch.tensor([0], dtype=torch.int32)

        n_samples = 6_000
        torch.manual_seed(20260717)
        result = sample_worlds_mrv_gpu(
            pools, hand_sizes, voids, decl_ids, n_samples=n_samples, device=device
        ).cpu()

        worlds = result[0]  # [n_samples, 3, 7]
        assert worlds.shape == (n_samples, 3, 7)
        flat = worlds.reshape(n_samples, 21)
        # Every world is a permutation of the full pool: valid, no duplicates.
        sorted_tiles = flat.sort(dim=1).values
        assert (sorted_tiles == torch.arange(21, dtype=worlds.dtype)).all()

        # Ownership marginals: each tile lands with each opponent ~1/3.
        # se ≈ 0.0061 at n=6000; 0.03 is a ~5-sigma gate against the audited
        # non-uniformity while staying flake-free.
        for tile in (0, 10, 20):
            owner_counts = (worlds == tile).sum(dim=2).float().mean(dim=0)
            assert torch.allclose(
                owner_counts, torch.full((3,), 1 / 3), atol=0.03
            ), f"tile {tile} owner marginal {owner_counts.tolist()} on {device}"

    def test_padding_variable_hand_sizes_and_batch_constraints(self):
        pools = torch.tensor(
            [
                [3, -1, 5, 7, 8, 14, 24, -1],
                [0, 1, -1, 2, 3, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        hand_sizes = torch.tensor([[2, 2, 2], [2, 1, 1]], dtype=torch.int32)
        voids = torch.zeros(2, 3, 8, dtype=torch.bool)
        voids[0, 0, 7] = True
        voids[0, 1, 3] = True
        voids[0, 2, 7] = True
        decl_ids = torch.tensor([0, 9], dtype=torch.int32)

        torch.manual_seed(17)
        result = sample_worlds_mrv_gpu(
            pools, hand_sizes, voids, decl_ids, n_samples=100, device="cpu"
        )

        assert result.shape == (2, 100, 3, 7)
        expected_pools = [{3, 5, 7, 8, 14, 24}, {0, 1, 2, 3}]
        for game in range(2):
            for world in result[game]:
                valid = [int(domino) for domino in world.flatten() if domino >= 0]
                assert set(valid) == expected_pools[game]
                assert len(valid) == len(expected_pools[game])
                for player in range(3):
                    assert int((world[player] >= 0).sum()) == int(
                        hand_sizes[game, player]
                    )

    def test_torch_seed_reproduces_exact_batch(self):
        inputs = _fixture_valid_only_bias()
        torch.manual_seed(12345)
        first = sample_worlds_mrv_gpu(
            *inputs, n_samples=256, device="cpu"
        )
        torch.manual_seed(12345)
        second = sample_worlds_mrv_gpu(
            *inputs, n_samples=256, device="cpu"
        )
        assert torch.equal(first, second)

    def test_impossible_constraints_fail_before_sampling(self):
        pools = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        hand_sizes = torch.tensor([[1, 1, 1]], dtype=torch.int32)
        voids = torch.zeros(1, 3, 8, dtype=torch.bool)
        voids[0, 0, :7] = True
        decl_ids = torch.tensor([9], dtype=torch.int32)

        with pytest.raises(ValueError, match="no void-consistent hand assignment"):
            sample_worlds_mrv_gpu(
                pools, hand_sizes, voids, decl_ids, n_samples=10, device="cpu"
            )

    def test_low_acceptance_arena_state_has_no_rejection_wall(self):
        # This real JudSearch state has exactly 924/17,153,136 = 5.39e-5 valid
        # mass under uniformly random partitions. The rejected replacement
        # exhausted 40,960 proposals with only 3/10 worlds (empirical 7.32e-5);
        # completion counting samples it directly.
        pool = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 22, 23, 24, 25, 26, 27]
        pools = torch.tensor([pool], dtype=torch.int32)
        hand_sizes = torch.tensor([[6, 6, 6]], dtype=torch.int32)
        voids = torch.zeros(1, 3, 8, dtype=torch.bool)
        voids[0, 1, 7] = True
        voids[0, 2, 7] = True
        decl_ids = torch.tensor([6], dtype=torch.int32)

        void_masks = _build_void_masks_vectorized(voids, decl_ids, "cpu")
        _tiles, _active, _allowed, suffix = _build_suffix_completion_counts(
            pools.to(torch.int64), torch.tensor([18]), void_masks
        )
        assert int(suffix[0, 0, 6, 6, 6]) == 924

        torch.manual_seed(2)
        result = sample_worlds_mrv_gpu(
            pools, hand_sizes, voids, decl_ids, n_samples=10, device="cpu"
        )

        assert result.shape == (1, 10, 3, 7)
        for world in result[0]:
            valid = [int(domino) for domino in world.flatten() if domino >= 0]
            assert len(valid) == 18 and set(valid) == set(pool)
            for player in (1, 2):
                for domino in world[player, :6]:
                    assert not CAN_FOLLOW[int(domino), 7, 6]

    def test_sampler_exposes_algorithm_fingerprint(self):
        sampler = WorldSamplerMRV(max_games=1, max_samples=2, device="cpu")
        assert SAMPLER_ALGORITHM == "uniform-completion-dp-v1"
        assert sampler.algorithm == SAMPLER_ALGORITHM

    @pytest.mark.skipif(torch.cuda.is_available(), reason="requires a host without CUDA")
    def test_cuda_request_does_not_fall_back_to_cpu(self):
        with pytest.raises(RuntimeError, match="no CPU fallback"):
            WorldSamplerMRV(max_games=1, max_samples=1, device="cuda")


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
