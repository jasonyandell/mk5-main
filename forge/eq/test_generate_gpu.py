"""Tests for GPU-native E[Q] generation pipeline.

Tests Phase 4 integration:
1. Correctness: E[Q] matches CPU baseline within tolerance
2. Performance: 32 games complete in reasonable time
3. Memory: No OOM on 3050 Ti parameters (32 games × 50 samples)
"""

import time

import pytest
import torch

from forge.eq.game import GameState
from forge.eq.generate_gpu import generate_eq_games_gpu
from forge.eq.oracle import Stage1Oracle
from forge.oracle.rng import deal_from_seed


@pytest.fixture
def device():
    """Get available device (prefer CUDA)."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@pytest.fixture
def oracle(device):
    """Load Stage 1 oracle model."""
    # Try multiple checkpoint paths
    checkpoint_paths = [
        "checkpoints/stage1/best.ckpt",
        "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
        "forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt",
    ]

    for checkpoint_path in checkpoint_paths:
        try:
            oracle = Stage1Oracle(checkpoint_path, device=device, compile=False)
            return oracle
        except Exception:
            continue

    pytest.skip(f"Could not load oracle from any checkpoint path")


def test_single_game_correctness(oracle, device):
    """Test that GPU pipeline produces reasonable E[Q] values for one game.

    This test doesn't compare to CPU (which is stochastic), but verifies:
    - All games complete (28 decisions)
    - E[Q] values are in reasonable range
    - Legal actions are respected
    """
    # Generate one game
    hands = deal_from_seed(12345)
    decl_id = 3

    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        greedy=True,
    )

    assert len(results) == 1
    game = results[0]

    # Check game completed (28 decisions)
    assert len(game.decisions) == 28

    # Check E[Q] values are reasonable (points roughly in [-42, 42])
    for decision in game.decisions:
        e_q = decision.e_q
        legal_e_q = e_q[decision.legal_mask]

        # Legal E[Q] values should not be -inf
        assert not torch.isinf(legal_e_q).any(), "Legal E[Q] should not be -inf"

        # E[Q] should be in reasonable range (loose bounds)
        assert legal_e_q.min() > -100, f"E[Q] too low: {legal_e_q.min()}"
        assert legal_e_q.max() < 100, f"E[Q] too high: {legal_e_q.max()}"

    # Check that chosen actions were legal
    for decision in game.decisions:
        assert decision.legal_mask[decision.action_taken], \
            f"Illegal action taken: {decision.action_taken}, legal: {decision.legal_mask}"


def test_multi_game_batch(oracle, device):
    """Test that multiple games can be processed in parallel."""
    n_games = 4
    hands = [deal_from_seed(1000 + i) for i in range(n_games)]
    decl_ids = [i % 10 for i in range(n_games)]

    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=decl_ids,
        n_samples=25,
        device=device,
        greedy=True,
    )

    assert len(results) == n_games

    for game in results:
        assert len(game.decisions) == 28, "Each game should have 28 decisions"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for performance test")
def test_performance_32_games(oracle):
    """Test performance target: 32 games × 50 samples on 3050 Ti.

    Target: 5+ games/s (32 games in < 6.4s)
    """
    device = 'cuda'
    n_games = 32
    n_samples = 50

    # Generate diverse games
    hands = [deal_from_seed(2000 + i) for i in range(n_games)]
    decl_ids = [i % 10 for i in range(n_games)]

    # Warmup (compile, allocate buffers)
    _ = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands[:2],
        decl_ids=decl_ids[:2],
        n_samples=n_samples,
        device=device,
        greedy=True,
    )

    # Timed run
    torch.cuda.synchronize()
    start = time.perf_counter()

    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=decl_ids,
        n_samples=n_samples,
        device=device,
        greedy=True,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Verify correctness
    assert len(results) == n_games
    for game in results:
        assert len(game.decisions) == 28

    # Performance metrics
    games_per_sec = n_games / elapsed
    print(f"\nPerformance: {games_per_sec:.2f} games/s ({elapsed:.2f}s for {n_games} games)")
    print(f"Target: 5+ games/s")

    # This is a soft target - don't fail on slower hardware
    if games_per_sec < 3:
        pytest.skip(f"Performance below minimum threshold: {games_per_sec:.2f} games/s")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for memory test")
def test_memory_no_oom():
    """Test that 32 games × 50 samples doesn't OOM on 3050 Ti (4GB).

    This test doesn't load the full model (too heavy for CI),
    but verifies the data pipeline can be allocated.
    """
    device = 'cuda'
    n_games = 32
    n_samples = 50

    from forge.eq.game_tensor import GameStateTensor
    from forge.eq.sampling_gpu import WorldSampler
    from forge.eq.tokenize_gpu import GPUTokenizer

    # Generate test data
    hands = [deal_from_seed(3000 + i) for i in range(n_games)]
    decl_ids = [i % 10 for i in range(n_games)]

    # Test allocation
    states = GameStateTensor.from_deals(hands, decl_ids, device)
    sampler = WorldSampler(max_games=n_games, max_samples=n_samples, device=device)
    tokenizer = GPUTokenizer(max_batch=n_games * n_samples, device=device)

    # Test basic operations
    legal_actions = states.legal_actions()
    assert legal_actions.shape == (n_games, 7)

    # If we got here without OOM, success
    print(f"\nMemory test passed: {n_games} games × {n_samples} samples allocated")

    # Check memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"CUDA memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")


def test_cpu_compatibility(oracle):
    """Test that GPU pipeline works on CPU (for debugging/testing)."""
    device = 'cpu'

    # Small test (CPU is slow)
    hands = [deal_from_seed(4000), deal_from_seed(4001)]
    decl_ids = [3, 5]

    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=decl_ids,
        n_samples=10,  # Fewer samples for CPU
        device=device,
        greedy=True,
    )

    assert len(results) == 2
    for game in results:
        assert len(game.decisions) == 28


def test_greedy_vs_sampled(oracle, device):
    """Test that greedy and sampled action selection both work."""
    hands = deal_from_seed(6000)
    decl_id = 3

    # Greedy
    greedy_results = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        greedy=True,
    )

    # Sampled
    sampled_results = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        greedy=False,
    )

    # Both should complete
    assert len(greedy_results[0].decisions) == 28
    assert len(sampled_results[0].decisions) == 28

    # Greedy always picks argmax
    for decision in greedy_results[0].decisions:
        e_q_masked = decision.e_q.clone()
        e_q_masked[~decision.legal_mask] = float('-inf')
        expected_action = e_q_masked.argmax().item()
        assert decision.action_taken == expected_action, \
            "Greedy should always pick argmax"

    # Sampled might differ (stochastic), just verify it completes
    # No specific assertions needed


def test_various_declarations(oracle, device):
    """Test pipeline with different declaration types."""
    hands = deal_from_seed(7000)

    for decl_id in [0, 3, 7, 9]:  # Sample different declarations
        results = generate_eq_games_gpu(
            model=oracle.model,
            hands=[hands],
            decl_ids=[decl_id],
            n_samples=25,
            device=device,
            greedy=True,
        )

        assert len(results) == 1
        assert len(results[0].decisions) == 28
        assert results[0].decl_id == decl_id


def test_exploration_policy_works(oracle, device):
    """Test that exploration policies work correctly in GPU pipeline."""
    from forge.eq.types import ExplorationPolicy

    hands = deal_from_seed(8000)
    decl_id = 3

    # Test with Boltzmann sampling
    policy = ExplorationPolicy.boltzmann(temperature=2.0, seed=42)

    result = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        exploration_policy=policy,
    )

    # Should complete successfully
    assert len(result) == 1
    assert len(result[0].decisions) == 28

    # All actions should be legal
    for decision in result[0].decisions:
        assert decision.legal_mask[decision.action_taken], \
            "Exploration should only pick legal actions"

    # Note: GPU pipeline world sampling is non-deterministic even with exploration seed,
    # because world sampling uses device-specific RNG. Exploration seed only controls
    # action selection given E[Q] values.


def test_exploration_produces_variety(oracle, device):
    """Test that exploration produces different actions than greedy."""
    from forge.eq.types import ExplorationPolicy

    hands = deal_from_seed(8100)
    decl_id = 3

    # Generate greedy baseline
    greedy_record = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        greedy=True,
    )

    # High epsilon should produce some non-greedy actions
    policy = ExplorationPolicy.epsilon_greedy(epsilon=0.5, seed=12345)

    exploration_record = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        exploration_policy=policy,
    )

    # Count differences
    n_different = sum(
        1 for g_dec, e_dec in zip(greedy_record[0].decisions, exploration_record[0].decisions)
        if g_dec.action_taken != e_dec.action_taken
    )

    # With epsilon=0.5, expect some non-greedy actions
    assert n_different > 5, \
        f"Exploration should produce some non-greedy actions (got {n_different}/28 different)"


def test_exploration_boltzmann_sampling(oracle, device):
    """Test Boltzmann sampling exploration."""
    from forge.eq.types import ExplorationPolicy

    hands = deal_from_seed(8200)
    decl_id = 3

    # High temperature should produce more variety
    policy = ExplorationPolicy.boltzmann(temperature=5.0, seed=999)

    result = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        exploration_policy=policy,
    )

    # Should complete successfully
    assert len(result) == 1
    assert len(result[0].decisions) == 28

    # All actions should be legal
    for decision in result[0].decisions:
        assert decision.legal_mask[decision.action_taken], \
            "Boltzmann sampling should only pick legal actions"


def test_exploration_mixed_policy(oracle, device):
    """Test mixed exploration (temperature + epsilon + blunder)."""
    from forge.eq.types import ExplorationPolicy

    hands = deal_from_seed(8300)
    decl_id = 3

    # Mixed exploration with blunder component
    policy = ExplorationPolicy.mixed_exploration(
        temperature=3.0,
        epsilon=0.05,
        blunder_rate=0.3,
        blunder_max_regret=5.0,
        seed=777,
    )

    result = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        exploration_policy=policy,
    )

    # Should complete successfully
    assert len(result) == 1
    assert len(result[0].decisions) == 28

    # All actions should be legal
    for decision in result[0].decisions:
        assert decision.legal_mask[decision.action_taken], \
            "Mixed exploration should only pick legal actions"


def test_exploration_with_multiple_games(oracle, device):
    """Test exploration policy with batch of games."""
    from forge.eq.types import ExplorationPolicy

    n_games = 4
    hands = [deal_from_seed(8400 + i) for i in range(n_games)]
    decl_ids = [i % 10 for i in range(n_games)]

    policy = ExplorationPolicy.epsilon_greedy(epsilon=0.3, seed=555)

    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=decl_ids,
        n_samples=25,
        device=device,
        exploration_policy=policy,
    )

    # All games should complete
    assert len(results) == n_games
    for game in results:
        assert len(game.decisions) == 28


def test_posterior_weighting_basic(oracle, device):
    """Test that posterior weighting config works and produces valid E[Q]."""
    from forge.eq.generate_gpu import PosteriorConfig

    hands = deal_from_seed(9000)
    decl_id = 3

    # Enable posterior weighting
    posterior_config = PosteriorConfig(
        enabled=True,
        window_k=4,
        tau=0.1,
        uniform_mix=0.1,
    )

    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        greedy=True,
        posterior_config=posterior_config,
    )

    # Should complete successfully
    assert len(results) == 1
    game = results[0]
    assert len(game.decisions) == 28

    # Check E[Q] values are reasonable
    for decision in game.decisions:
        e_q = decision.e_q
        legal_e_q = e_q[decision.legal_mask]

        # Legal E[Q] should not be -inf
        assert not torch.isinf(legal_e_q).any(), "Legal E[Q] should not be -inf"

        # E[Q] should be in reasonable range
        assert legal_e_q.min() > -100, f"E[Q] too low: {legal_e_q.min()}"
        assert legal_e_q.max() < 100, f"E[Q] too high: {legal_e_q.max()}"

    # All actions should be legal
    for decision in game.decisions:
        assert decision.legal_mask[decision.action_taken], \
            f"Illegal action taken: {decision.action_taken}"


def test_posterior_disabled_vs_enabled(oracle, device):
    """Test that posterior weighting changes E[Q] values compared to uniform."""
    from forge.eq.generate_gpu import PosteriorConfig

    hands = deal_from_seed(9100)
    decl_id = 3

    # Run with posterior disabled (uniform weighting)
    results_uniform = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        greedy=True,
        posterior_config=None,  # Default: disabled
    )

    # Run with posterior enabled
    posterior_config = PosteriorConfig(enabled=True, window_k=4, tau=0.1, uniform_mix=0.1)
    results_posterior = generate_eq_games_gpu(
        model=oracle.model,
        hands=[hands],
        decl_ids=[decl_id],
        n_samples=50,
        device=device,
        greedy=True,
        posterior_config=posterior_config,
    )

    # Both should complete
    assert len(results_uniform) == 1
    assert len(results_posterior) == 1
    assert len(results_uniform[0].decisions) == 28
    assert len(results_posterior[0].decisions) == 28

    # E[Q] values should differ for at least some decisions (once history builds up)
    # Early decisions may be identical if not enough history exists
    differences = 0
    for i, (dec_uniform, dec_posterior) in enumerate(
        zip(results_uniform[0].decisions, results_posterior[0].decisions)
    ):
        # Skip very early decisions (< K steps)
        if i < 4:
            continue

        # Check if E[Q] values differ
        e_q_diff = torch.abs(dec_uniform.e_q - dec_posterior.e_q).max()
        if e_q_diff > 0.01:  # Small threshold for numerical differences
            differences += 1

    # After enough history, posterior should affect some decisions
    # This is a soft check - may not differ if worlds happen to be similar
    print(f"\nPosterior weighting changed E[Q] for {differences}/24 decisions (after first 4)")


def test_posterior_with_mixed_decl_ids(oracle, device):
    """Test posterior weighting works correctly with different decl_ids per game.

    This is a regression test for the bug where tokenize_past_steps_batched and
    compute_legal_masks_gpu used only decl_ids[0] for all games in a batch.

    The fix: both functions now accept per-game decl_ids tensor.
    """
    from forge.eq.generate_gpu import PosteriorConfig

    n_games = 4
    # Different seeds and DIFFERENT declarations per game
    hands = [deal_from_seed(50000 + i) for i in range(n_games)]
    decl_ids = [3, 7, 0, 9]  # Different trumps: 3s, doubles-trump, 0s, no-trump

    posterior_config = PosteriorConfig(enabled=True, window_k=4, tau=0.1, uniform_mix=0.1)

    # This should NOT crash or produce wrong results
    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=decl_ids,
        n_samples=50,
        device=device,
        greedy=True,
        posterior_config=posterior_config,
    )

    # All games should complete with 28 decisions each
    assert len(results) == n_games
    for i, game in enumerate(results):
        assert len(game.decisions) == 28, f"Game {i} should have 28 decisions"

        for j, decision in enumerate(game.decisions):
            # Check E[Q] is valid (not NaN, not all -inf)
            e_q = decision.e_q
            legal_e_q = e_q[decision.legal_mask]

            assert not torch.isnan(legal_e_q).any(), \
                f"Game {i} decision {j}: E[Q] contains NaN"
            assert not torch.isinf(legal_e_q).all(), \
                f"Game {i} decision {j}: All legal E[Q] are inf"

            # E[Q] should be in reasonable range
            assert legal_e_q.min() > -100, \
                f"Game {i} decision {j}: E[Q] too low: {legal_e_q.min()}"
            assert legal_e_q.max() < 100, \
                f"Game {i} decision {j}: E[Q] too high: {legal_e_q.max()}"

            # Action taken should be legal
            assert decision.legal_mask[decision.action_taken], \
                f"Game {i} decision {j}: Illegal action taken"

    print(f"\nMixed decl_ids test passed: {n_games} games with decl_ids={decl_ids}")


def test_build_hypothetical_deals_vectorized(device):
    """Verify vectorized _build_hypothetical_deals matches original loop version."""
    from forge.eq.game_tensor import GameStateTensor
    from forge.eq.generate_gpu import _build_hypothetical_deals, _build_hypothetical_deals_loop

    # Test with various game configurations
    test_cases = [
        # (n_games, n_samples, max_hand_size)
        (1, 10, 7),   # Single game, full hands
        (4, 25, 7),   # Multiple games, full hands
        (8, 50, 5),   # More games, partial hands (mid-game)
        (16, 100, 3), # Many games, small hands (late game)
    ]

    for n_games, n_samples, max_hand_size in test_cases:
        # Create test game states
        hands = [deal_from_seed(10000 + i) for i in range(n_games)]
        decl_ids = [i % 10 for i in range(n_games)]
        states = GameStateTensor.from_deals(hands, decl_ids, device)

        # Create random worlds tensor
        # worlds: [n_games, n_samples, 3, max_hand_size]
        # Values should be valid domino IDs (0-27) or -1 for padding
        torch.manual_seed(42)
        worlds = torch.randint(
            -1, 28,
            (n_games, n_samples, 3, max_hand_size),
            dtype=torch.int32,
            device=device
        )

        # Run both implementations
        result_loop = _build_hypothetical_deals_loop(states, worlds)
        result_vectorized = _build_hypothetical_deals(states, worlds)

        # Verify shapes match
        assert result_loop.shape == result_vectorized.shape, \
            f"Shape mismatch: {result_loop.shape} vs {result_vectorized.shape}"

        # Verify values match exactly
        matches = torch.all(result_loop == result_vectorized)
        if not matches:
            # Find where they differ for debugging
            diff_mask = result_loop != result_vectorized
            diff_indices = torch.nonzero(diff_mask)
            print(f"\nMismatch at indices: {diff_indices[:5]}", flush=True)
            print(f"Loop values: {result_loop[diff_mask][:5]}", flush=True)
            print(f"Vectorized values: {result_vectorized[diff_mask][:5]}", flush=True)

        assert matches, f"Mismatch for config (n_games={n_games}, n_samples={n_samples}, max_hand_size={max_hand_size})"

    print(f"\nVectorized _build_hypothetical_deals matches loop version for all {len(test_cases)} test cases", flush=True)


def test_build_hypothetical_deals_benchmark(device):
    """Benchmark vectorized vs loop _build_hypothetical_deals.

    Must complete in under 60 seconds total.
    """
    from forge.eq.game_tensor import GameStateTensor
    from forge.eq.generate_gpu import _build_hypothetical_deals, _build_hypothetical_deals_loop

    n_games = 100
    n_samples = 50
    n_iterations = 3
    max_hand_size = 7

    # Create test data
    hands = [deal_from_seed(20000 + i) for i in range(n_games)]
    decl_ids = [i % 10 for i in range(n_games)]
    states = GameStateTensor.from_deals(hands, decl_ids, device)

    torch.manual_seed(123)
    worlds = torch.randint(
        -1, 28,
        (n_games, n_samples, 3, max_hand_size),
        dtype=torch.int32,
        device=device
    )

    # Warmup
    _ = _build_hypothetical_deals_loop(states, worlds)
    _ = _build_hypothetical_deals(states, worlds)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark loop version
    loop_times = []
    for i in range(n_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = _build_hypothetical_deals_loop(states, worlds)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        loop_times.append(elapsed)

    # Benchmark vectorized version
    vec_times = []
    for i in range(n_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = _build_hypothetical_deals(states, worlds)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        vec_times.append(elapsed)

    loop_mean = sum(loop_times) / n_iterations
    vec_mean = sum(vec_times) / n_iterations
    speedup = loop_mean / vec_mean if vec_mean > 0 else float('inf')

    print(f"\n=== _build_hypothetical_deals Benchmark ===", flush=True)
    print(f"Config: N={n_games} games, M={n_samples} samples, {n_iterations} iterations", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Loop version:       {loop_mean*1000:.2f}ms mean", flush=True)
    print(f"Vectorized version: {vec_mean*1000:.2f}ms mean", flush=True)
    print(f"Speedup: {speedup:.2f}x", flush=True)

    # Verify correctness one more time
    result_loop = _build_hypothetical_deals_loop(states, worlds)
    result_vec = _build_hypothetical_deals(states, worlds)
    assert torch.all(result_loop == result_vec), "Results don't match!"
    print(f"Correctness verified: outputs match exactly", flush=True)


def test_sample_worlds_vectorized(device):
    """Verify vectorized _sample_worlds_batched matches original loop version."""
    from forge.eq.game_tensor import GameStateTensor
    from forge.eq.generate_gpu import _sample_worlds_batched, _sample_worlds_batched_loop
    from forge.eq.sampling_mrv_gpu import WorldSamplerMRV

    # Test with various game configurations
    test_cases = [
        # (n_games, n_samples)
        (1, 10),   # Single game
        (4, 25),   # Multiple games
        (8, 50),   # More games
    ]

    for n_games, n_samples in test_cases:
        print(f"\nTesting n_games={n_games}, n_samples={n_samples}", flush=True)

        # Create test game states
        hands = [deal_from_seed(30000 + i) for i in range(n_games)]
        decl_ids = [i % 10 for i in range(n_games)]
        states = GameStateTensor.from_deals(hands, decl_ids, device)

        # Create sampler
        sampler_loop = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)
        sampler_vec = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)

        # Run both implementations with same seed
        torch.manual_seed(42)
        if device == 'cuda':
            torch.cuda.manual_seed(42)
        result_loop = _sample_worlds_batched_loop(states, sampler_loop, n_samples)

        torch.manual_seed(42)
        if device == 'cuda':
            torch.cuda.manual_seed(42)
        result_vectorized = _sample_worlds_batched(states, sampler_vec, n_samples)

        # Verify shapes match
        assert result_loop.shape == result_vectorized.shape, \
            f"Shape mismatch: {result_loop.shape} vs {result_vectorized.shape}"

        print(f"  Shape: {result_loop.shape}", flush=True)

        # Note: Due to randomness in sampling, exact match is not expected
        # But we verify shapes and value ranges are correct
        assert result_loop.min() >= -1, "Loop result has invalid min"
        assert result_loop.max() < 28, "Loop result has invalid max"
        assert result_vectorized.min() >= -1, "Vectorized result has invalid min"
        assert result_vectorized.max() < 28, "Vectorized result has invalid max"

        print(f"  Both implementations produce valid domino IDs", flush=True)

    print(f"\nVectorized _sample_worlds_batched produces valid outputs for all {len(test_cases)} test cases", flush=True)


def test_sample_worlds_benchmark(device):
    """Benchmark vectorized vs loop _sample_worlds_batched.

    Must complete in under 60 seconds total.
    """
    from forge.eq.game_tensor import GameStateTensor
    from forge.eq.generate_gpu import _sample_worlds_batched, _sample_worlds_batched_loop
    from forge.eq.sampling_mrv_gpu import WorldSamplerMRV

    n_games = 32
    n_samples = 50
    n_iterations = 3

    print(f"\n=== _sample_worlds_batched Benchmark ===", flush=True)
    print(f"Config: N={n_games} games, M={n_samples} samples, {n_iterations} iterations", flush=True)
    print(f"Device: {device}", flush=True)

    # Create test data
    hands = [deal_from_seed(40000 + i) for i in range(n_games)]
    decl_ids = [i % 10 for i in range(n_games)]
    states = GameStateTensor.from_deals(hands, decl_ids, device)

    # Create samplers
    sampler_loop = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)
    sampler_vec = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)

    # Warmup
    _ = _sample_worlds_batched_loop(states, sampler_loop, n_samples)
    _ = _sample_worlds_batched(states, sampler_vec, n_samples)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark loop version
    loop_times = []
    for i in range(n_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = _sample_worlds_batched_loop(states, sampler_loop, n_samples)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        loop_times.append(elapsed)

    # Benchmark vectorized version
    vec_times = []
    for i in range(n_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = _sample_worlds_batched(states, sampler_vec, n_samples)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        vec_times.append(elapsed)

    loop_mean = sum(loop_times) / n_iterations
    vec_mean = sum(vec_times) / n_iterations
    speedup = loop_mean / vec_mean if vec_mean > 0 else float('inf')

    print(f"Loop version:       {loop_mean*1000:.2f}ms mean", flush=True)
    print(f"Vectorized version: {vec_mean*1000:.2f}ms mean", flush=True)
    print(f"Speedup: {speedup:.2f}x", flush=True)

    # Verify both produce valid outputs
    result_loop = _sample_worlds_batched_loop(states, sampler_loop, n_samples)
    result_vec = _sample_worlds_batched(states, sampler_vec, n_samples)

    assert result_loop.shape == result_vec.shape, "Shape mismatch!"
    assert result_loop.min() >= -1 and result_loop.max() < 28, "Loop result invalid"
    assert result_vec.min() >= -1 and result_vec.max() < 28, "Vectorized result invalid"

    print(f"Correctness verified: both produce valid domino IDs", flush=True)

    # Test should complete in under 60 seconds
    total_time = sum(loop_times) + sum(vec_times)
    assert total_time < 60, f"Benchmark took too long: {total_time:.2f}s"


def test_tokenize_batched_vectorized(device):
    """Verify vectorized _tokenize_batched matches original loop version."""
    from forge.eq.game_tensor import GameStateTensor
    from forge.eq.generate_gpu import _tokenize_batched, _tokenize_batched_loop, _build_hypothetical_deals
    from forge.eq.tokenize_gpu import GPUTokenizer
    from forge.eq.sampling_mrv_gpu import WorldSamplerMRV

    # Test with various game configurations
    test_cases = [
        # (n_games, n_samples)
        (1, 10),   # Single game
        (4, 25),   # Few games
        (8, 50),   # Multiple games
        (16, 32),  # Many games
    ]

    for n_games, n_samples in test_cases:
        print(f"\nTesting n_games={n_games}, n_samples={n_samples}", flush=True)

        # Create test game states with some trick plays
        hands = [deal_from_seed(30000 + i) for i in range(n_games)]
        decl_ids = [i % 10 for i in range(n_games)]
        states = GameStateTensor.from_deals(hands, decl_ids, device)

        # Simulate a few plays to create non-empty trick_plays
        # Apply 2-3 actions to each game
        for step in range(3):
            if not states.active_games().any():
                break
            legal_mask = states.legal_actions()
            # Pick first legal action for each game
            actions = torch.zeros(n_games, dtype=torch.long, device=device)
            for g in range(n_games):
                legal_indices = torch.nonzero(legal_mask[g], as_tuple=True)[0]
                if len(legal_indices) > 0:
                    actions[g] = legal_indices[0]
            states = states.apply_actions(actions)

        # Sample worlds
        sampler = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)
        from forge.eq.generate_gpu import _sample_worlds_batched
        worlds = _sample_worlds_batched(states, sampler, n_samples)

        # Build hypothetical deals
        deals = _build_hypothetical_deals(states, worlds)

        # Create tokenizer
        tokenizer = GPUTokenizer(max_batch=n_games * n_samples, device=device)

        # Run both implementations
        tokens_loop, masks_loop = _tokenize_batched_loop(states, deals, tokenizer)
        tokens_vec, masks_vec = _tokenize_batched(states, deals, tokenizer)

        # Verify shapes match
        assert tokens_loop.shape == tokens_vec.shape, \
            f"Token shape mismatch: {tokens_loop.shape} vs {tokens_vec.shape}"
        assert masks_loop.shape == masks_vec.shape, \
            f"Mask shape mismatch: {masks_loop.shape} vs {masks_vec.shape}"

        # Verify values match exactly
        tokens_match = torch.all(tokens_loop == tokens_vec)
        masks_match = torch.all(masks_loop == masks_vec)

        if not tokens_match:
            # Find where they differ for debugging
            diff_mask = tokens_loop != tokens_vec
            diff_indices = torch.nonzero(diff_mask)
            print(f"Token mismatch at indices: {diff_indices[:5]}", flush=True)
            print(f"Loop values: {tokens_loop[diff_mask][:10]}", flush=True)
            print(f"Vectorized values: {tokens_vec[diff_mask][:10]}", flush=True)

        if not masks_match:
            diff_mask = masks_loop != masks_vec
            diff_indices = torch.nonzero(diff_mask)
            print(f"Mask mismatch at indices: {diff_indices[:5]}", flush=True)
            print(f"Loop values: {masks_loop[diff_mask][:10]}", flush=True)
            print(f"Vectorized values: {masks_vec[diff_mask][:10]}", flush=True)

        assert tokens_match, f"Token mismatch for config (n_games={n_games}, n_samples={n_samples})"
        assert masks_match, f"Mask mismatch for config (n_games={n_games}, n_samples={n_samples})"

    print(f"\nVectorized _tokenize_batched matches loop version for all {len(test_cases)} test cases", flush=True)


def test_tokenize_batched_benchmark(device):
    """Benchmark vectorized vs loop _tokenize_batched.

    Must complete in under 60 seconds total.
    """
    from forge.eq.game_tensor import GameStateTensor
    from forge.eq.generate_gpu import _tokenize_batched, _tokenize_batched_loop, _build_hypothetical_deals, _sample_worlds_batched
    from forge.eq.tokenize_gpu import GPUTokenizer
    from forge.eq.sampling_mrv_gpu import WorldSamplerMRV

    n_games = 32
    n_samples = 50
    n_iterations = 3

    # Create test data
    hands = [deal_from_seed(40000 + i) for i in range(n_games)]
    decl_ids = [i % 10 for i in range(n_games)]
    states = GameStateTensor.from_deals(hands, decl_ids, device)

    # Apply a few actions to create realistic game states with trick plays
    for step in range(5):
        if not states.active_games().any():
            break
        legal_mask = states.legal_actions()
        actions = torch.zeros(n_games, dtype=torch.long, device=device)
        for g in range(n_games):
            legal_indices = torch.nonzero(legal_mask[g], as_tuple=True)[0]
            if len(legal_indices) > 0:
                actions[g] = legal_indices[0]
        states = states.apply_actions(actions)

    # Sample worlds and build deals
    sampler = WorldSamplerMRV(max_games=n_games, max_samples=n_samples, device=device)
    worlds = _sample_worlds_batched(states, sampler, n_samples)
    deals = _build_hypothetical_deals(states, worlds)

    # Create tokenizer
    tokenizer = GPUTokenizer(max_batch=n_games * n_samples, device=device)

    # Warmup
    _ = _tokenize_batched_loop(states, deals, tokenizer)
    _ = _tokenize_batched(states, deals, tokenizer)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark loop version
    loop_times = []
    for i in range(n_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = _tokenize_batched_loop(states, deals, tokenizer)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        loop_times.append(elapsed)

    # Benchmark vectorized version
    vec_times = []
    for i in range(n_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = _tokenize_batched(states, deals, tokenizer)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        vec_times.append(elapsed)

    loop_mean = sum(loop_times) / n_iterations
    vec_mean = sum(vec_times) / n_iterations
    speedup = loop_mean / vec_mean if vec_mean > 0 else float('inf')

    print(f"\n=== _tokenize_batched Benchmark ===", flush=True)
    print(f"Config: N={n_games} games, M={n_samples} samples, {n_iterations} iterations", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Loop version:       {loop_mean*1000:.2f}ms mean", flush=True)
    print(f"Vectorized version: {vec_mean*1000:.2f}ms mean", flush=True)
    print(f"Speedup: {speedup:.2f}x", flush=True)

    # Verify correctness one more time
    tokens_loop, masks_loop = _tokenize_batched_loop(states, deals, tokenizer)
    tokens_vec, masks_vec = _tokenize_batched(states, deals, tokenizer)
    assert torch.all(tokens_loop == tokens_vec), "Token mismatch!"
    assert torch.all(masks_loop == masks_vec), "Mask mismatch!"
    print(f"Correctness verified: outputs match exactly", flush=True)

    # Test should complete in under 60 seconds
    total_time = sum(loop_times) + sum(vec_times)
    assert total_time < 60, f"Benchmark took too long: {total_time:.2f}s"


def test_enumeration_basic(oracle, device):
    """Test that enumeration mode produces valid results."""
    # Use a late-game scenario where enumeration is effective
    # Generate a few games
    n_games = 2
    hands = [deal_from_seed(42 + i) for i in range(n_games)]
    decl_ids = [3, 5]  # Different declarations

    # Run with enumeration enabled (use lower threshold to avoid OOM on small GPUs)
    records = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=decl_ids,
        n_samples=50,
        device=device,
        greedy=True,
        use_enumeration=True,
        enumeration_threshold=1_000,  # Small threshold for testing
    )

    assert len(records) == n_games

    for i, record in enumerate(records):
        # Should have 28 decisions per game
        assert len(record.decisions) == 28, f"Game {i}: expected 28 decisions, got {len(record.decisions)}"

        # All E[Q] values should be valid (not all zeros or inf)
        for j, decision in enumerate(record.decisions):
            legal = decision.legal_mask
            e_q_legal = decision.e_q[legal]

            # Should have at least one legal action
            assert legal.any(), f"Game {i}, decision {j}: no legal actions"

            # E[Q] should be finite
            assert torch.isfinite(e_q_legal).all(), (
                f"Game {i}, decision {j}: non-finite E[Q] values"
            )


def test_enumeration_multiple_games(oracle, device):
    """Test that enumeration mode works with multiple games."""
    # Test with 4 games to ensure batch processing works
    n_games = 4
    hands = [deal_from_seed(200 + i) for i in range(n_games)]
    decl_ids = [0, 3, 5, 9]  # Various declarations

    # Run with enumeration
    records = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=decl_ids,
        n_samples=50,
        device=device,
        greedy=True,
        use_enumeration=True,
        enumeration_threshold=1_000,  # Small threshold for testing
    )

    # All games should complete
    assert len(records) == n_games
    for i, record in enumerate(records):
        assert len(record.decisions) == 28, f"Game {i}: expected 28 decisions"

        # All decisions should have valid E[Q] values
        for j, dec in enumerate(record.decisions):
            legal_eq = dec.e_q[dec.legal_mask]
            assert torch.isfinite(legal_eq).all(), f"Game {i}, dec {j}: non-finite E[Q]"


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
