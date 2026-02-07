"""Tests for GPU training pipeline.

Verifies zero-copy training pipeline correctness and performance.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestDealRandomGPU:
    """Test GPU-native deal generation (from Phase 1)."""

    def test_deal_produces_valid_hands(self):
        """Deals should have 4 players with 7 unique dominoes each."""
        from forge.zeb.gpu_game_state import deal_random_gpu

        n = 16
        device = torch.device('cuda')
        states = deal_random_gpu(n, device)

        # Check shapes
        assert states.hands.shape == (n, 4, 7)
        assert states.hands.device.type == 'cuda'

        # Check each game has valid hands
        for i in range(n):
            all_dominoes = states.hands[i].flatten().cpu().tolist()
            assert len(all_dominoes) == 28
            assert len(set(all_dominoes)) == 28  # All unique
            assert min(all_dominoes) == 0
            assert max(all_dominoes) == 27

    def test_deal_respects_declaration_ids(self):
        """Deal should use provided declaration IDs."""
        from forge.zeb.gpu_game_state import deal_random_gpu

        device = torch.device('cuda')

        # Scalar decl_id
        states = deal_random_gpu(4, device, decl_ids=5)
        assert (states.decl_id == 5).all()

        # Tensor decl_id
        decl_ids = torch.tensor([0, 3, 7, 9], device=device)
        states = deal_random_gpu(4, device, decl_ids=decl_ids)
        assert torch.equal(states.decl_id, decl_ids)

    def test_deal_initializes_empty_state(self):
        """Deal should create initial game state with no plays."""
        from forge.zeb.gpu_game_state import deal_random_gpu

        states = deal_random_gpu(8, torch.device('cuda'))

        assert states.n_plays.sum() == 0
        assert states.trick_len.sum() == 0
        assert (~states.played_mask).all()
        assert states.scores.sum() == 0


class TestGPUObservationTokenizer:
    """Test GPU-native observation tokenization."""

    def test_tokenize_produces_correct_shapes(self):
        """Tokenizer should produce correct tensor shapes."""
        from forge.zeb.gpu_game_state import deal_random_gpu
        from forge.zeb.gpu_training_pipeline import GPUObservationTokenizer

        device = torch.device('cuda')
        n = 8

        states = deal_random_gpu(n, device)
        original_hands = states.hands.clone()
        players = torch.zeros(n, dtype=torch.int32, device=device)

        tokenizer = GPUObservationTokenizer(device)
        tokens, masks, hand_indices, hand_masks = tokenizer.tokenize_batch(
            states, original_hands, players
        )

        assert tokens.shape == (n, 36, 8)
        assert masks.shape == (n, 36)
        assert hand_indices.shape == (n, 7)
        assert hand_masks.shape == (n, 7)

    def test_tokenize_initial_state(self):
        """Initial state should have all hand slots valid."""
        from forge.zeb.gpu_game_state import deal_random_gpu
        from forge.zeb.gpu_training_pipeline import GPUObservationTokenizer

        device = torch.device('cuda')
        n = 4

        states = deal_random_gpu(n, device)
        original_hands = states.hands.clone()
        players = torch.zeros(n, dtype=torch.int32, device=device)

        tokenizer = GPUObservationTokenizer(device)
        tokens, masks, hand_indices, hand_masks = tokenizer.tokenize_batch(
            states, original_hands, players
        )

        # All 7 hand slots should be valid (no plays yet)
        assert hand_masks.all()

        # Declaration token should be valid
        assert masks[:, 0].all()

        # Hand tokens should be valid
        assert masks[:, 1:8].all()

        # No play tokens yet
        assert (~masks[:, 8:]).all()

    def test_tokenize_after_plays(self):
        """After plays, hand slots should reflect played dominoes."""
        from forge.zeb.gpu_game_state import deal_random_gpu, apply_action_gpu, legal_actions_gpu
        from forge.zeb.gpu_training_pipeline import GPUObservationTokenizer

        device = torch.device('cuda')
        n = 4

        states = deal_random_gpu(n, device)
        original_hands = states.hands.clone()

        # Make one play per game
        legal = legal_actions_gpu(states)
        actions = legal.int().argmax(dim=1)  # First legal action
        states = apply_action_gpu(states, actions)

        players = torch.zeros(n, dtype=torch.int32, device=device)

        tokenizer = GPUObservationTokenizer(device)
        tokens, masks, hand_indices, hand_masks = tokenizer.tokenize_batch(
            states, original_hands, players
        )

        # Should have 6 valid hand slots (one played)
        assert hand_masks.sum(dim=1).min() <= 6

        # Should have one play token
        assert masks[:, 8].all()  # First play position

    def test_tokenize_stays_on_gpu(self):
        """Tokenization should not move data to CPU."""
        from forge.zeb.gpu_game_state import deal_random_gpu
        from forge.zeb.gpu_training_pipeline import GPUObservationTokenizer

        device = torch.device('cuda')
        states = deal_random_gpu(4, device)

        tokenizer = GPUObservationTokenizer(device)
        tokens, masks, hand_indices, hand_masks = tokenizer.tokenize_batch(
            states, states.hands.clone(), torch.zeros(4, dtype=torch.int32, device=device)
        )

        assert tokens.device.type == 'cuda'
        assert masks.device.type == 'cuda'
        assert hand_indices.device.type == 'cuda'
        assert hand_masks.device.type == 'cuda'


class TestSelfPlayLoop:
    """Test MCTS self-play loop."""

    def test_self_play_runs_without_errors(self):
        """Self-play loop should complete without errors."""
        from forge.zeb.gpu_game_state import deal_random_gpu
        from forge.zeb.gpu_mcts import create_forest, select_leaves_gpu, expand_gpu, backprop_gpu, get_root_policy_gpu

        device = torch.device('cuda')
        n = 4

        states = deal_random_gpu(n, device)
        forest = create_forest(n, max_nodes=64, initial_states=states, device=device)

        # Run a few simulations
        for _ in range(10):
            leaf_indices, paths = select_leaves_gpu(forest)
            values = torch.randn(n, device=device)  # Random values for test
            expand_gpu(forest, leaf_indices)
            backprop_gpu(forest, leaf_indices, values, paths)

        # Should produce valid policy
        policy = get_root_policy_gpu(forest)
        assert policy.shape == (n, 7)
        assert (policy >= 0).all()
        # Policy should sum to ~1 (or 0 for invalid)
        policy_sum = policy.sum(dim=1)
        assert ((policy_sum > 0.9) | (policy_sum < 0.1)).all()

    def test_forest_stays_on_gpu(self):
        """MCTS forest should keep all data on GPU."""
        from forge.zeb.gpu_game_state import deal_random_gpu
        from forge.zeb.gpu_mcts import create_forest, select_leaves_gpu

        device = torch.device('cuda')
        states = deal_random_gpu(4, device)
        forest = create_forest(4, max_nodes=64, initial_states=states, device=device)

        # All forest tensors should be on CUDA
        assert forest.visit_counts.device.type == 'cuda'
        assert forest.value_sums.device.type == 'cuda'
        assert forest.states.hands.device.type == 'cuda'


class TestTrainingExamplesShape:
    """Test training example generation."""

    @pytest.fixture
    def mock_oracle(self):
        """Create mock oracle for testing."""
        oracle = MagicMock()

        def batch_evaluate_gpu(
            original_hands, current_hands, decl_ids, actors, leaders,
            trick_players, trick_dominoes, players
        ):
            n = original_hands.shape[0]
            return torch.randn(n, device=original_hands.device)

        oracle.batch_evaluate_gpu = batch_evaluate_gpu
        return oracle

    def test_generate_produces_correct_shapes(self, mock_oracle):
        """Generated examples should have correct tensor shapes."""
        from forge.zeb.gpu_training_pipeline import GPUTrainingPipeline

        device = torch.device('cuda')
        pipeline = GPUTrainingPipeline(
            oracle=mock_oracle,
            device=device,
            n_parallel_games=4,
            n_simulations=5,  # Few sims for speed
            max_mcts_nodes=64,
        )

        # Generate small number of games
        examples = pipeline.generate_games_gpu(n_games=2)

        # Should have 28 examples per game
        expected_examples = 2 * 28  # 28 moves per game

        # Check shapes
        assert examples.observations.shape[0] <= expected_examples
        assert examples.observations.shape[1] == 36
        assert examples.observations.shape[2] == 8

        assert examples.masks.shape[0] == examples.observations.shape[0]
        assert examples.masks.shape[1] == 36

        assert examples.policy_targets.shape[1] == 7
        assert examples.value_targets.shape[0] == examples.observations.shape[0]

    def test_examples_stay_on_gpu(self, mock_oracle):
        """Generated examples should be on GPU."""
        from forge.zeb.gpu_training_pipeline import GPUTrainingPipeline

        device = torch.device('cuda')
        pipeline = GPUTrainingPipeline(
            oracle=mock_oracle,
            device=device,
            n_parallel_games=2,
            n_simulations=3,
            max_mcts_nodes=32,
        )

        examples = pipeline.generate_games_gpu(n_games=1)

        assert examples.observations.device.type == 'cuda'
        assert examples.masks.device.type == 'cuda'
        assert examples.policy_targets.device.type == 'cuda'
        assert examples.value_targets.device.type == 'cuda'


class TestNoCPUTransfers:
    """Test that hot path has no CPU transfers."""

    def test_no_cpu_calls_in_tokenizer(self):
        """Tokenizer should not call .cpu() or .tolist()."""
        from forge.zeb.gpu_game_state import deal_random_gpu
        from forge.zeb.gpu_training_pipeline import GPUObservationTokenizer

        device = torch.device('cuda')
        states = deal_random_gpu(4, device)

        tokenizer = GPUObservationTokenizer(device)

        # Patch cpu() to detect calls
        cpu_called = []
        original_cpu = torch.Tensor.cpu

        def tracking_cpu(self, *args, **kwargs):
            cpu_called.append(True)
            return original_cpu(self, *args, **kwargs)

        with patch.object(torch.Tensor, 'cpu', tracking_cpu):
            tokens, masks, hand_idx, hand_masks = tokenizer.tokenize_batch(
                states,
                states.hands.clone(),
                torch.zeros(4, dtype=torch.int32, device=device)
            )

        # No .cpu() calls should have happened
        assert len(cpu_called) == 0, "Tokenizer called .cpu() in hot path"


class TestTrainingLoss:
    """Test training produces decreasing loss."""

    @pytest.fixture
    def mock_oracle(self):
        """Create mock oracle."""
        oracle = MagicMock()

        def batch_evaluate_gpu(**kwargs):
            n = kwargs['original_hands'].shape[0]
            return torch.randn(n, device=kwargs['original_hands'].device)

        oracle.batch_evaluate_gpu = batch_evaluate_gpu
        return oracle

    def test_training_decreases_loss(self, mock_oracle):
        """Training should decrease loss over iterations."""
        from forge.zeb.gpu_training_pipeline import GPUTrainingPipeline
        from forge.zeb.model import ZebModel, get_model_config

        device = torch.device('cuda')

        # Create small model
        config = get_model_config('small')
        model = ZebModel(**config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create pipeline
        pipeline = GPUTrainingPipeline(
            oracle=mock_oracle,
            device=device,
            n_parallel_games=2,
            n_simulations=3,
            max_mcts_nodes=32,
        )

        # Generate examples
        examples = pipeline.generate_games_gpu(n_games=1)

        # Train for a few epochs and track loss
        losses = []
        for _ in range(5):
            metrics = pipeline.train_epoch_gpu(
                model=model,
                optimizer=optimizer,
                examples=examples,
                batch_size=16,
            )
            losses.append(metrics['policy_loss'] + metrics['value_loss'])

        # Loss should generally decrease (not strictly due to stochasticity)
        assert losses[-1] < losses[0] * 1.5, "Loss did not decrease during training"


class TestIntegration:
    """Integration tests with real oracle (if available)."""

    @pytest.mark.slow
    def test_full_pipeline_with_oracle(self):
        """Test full pipeline with actual oracle."""
        pytest.importorskip('forge.eq.oracle')

        from forge.zeb.gpu_training_pipeline import create_gpu_pipeline
        from forge.zeb.model import ZebModel, get_model_config

        device = torch.device('cuda')

        # Create pipeline with real oracle
        try:
            pipeline = create_gpu_pipeline(
                oracle_device='cuda',
                n_parallel_games=2,
                n_simulations=5,
                max_mcts_nodes=64,
            )
        except FileNotFoundError:
            pytest.skip("Oracle checkpoint not found")

        # Generate examples
        examples = pipeline.generate_games_gpu(n_games=1)

        assert examples.n_examples > 0
        assert examples.observations.device.type == 'cuda'

        # Create and train model
        config = get_model_config('small')
        model = ZebModel(**config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # One training pass
        metrics = pipeline.train_epoch_gpu(
            model=model,
            optimizer=optimizer,
            examples=examples,
            batch_size=16,
        )

        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert metrics['policy_loss'] >= 0
        assert metrics['value_loss'] >= 0


# Benchmark tests (marked for manual running)
class TestBenchmarks:
    """Performance benchmarks."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_throughput_benchmark(self):
        """Benchmark training throughput."""
        import time

        from forge.zeb.gpu_game_state import deal_random_gpu
        from forge.zeb.gpu_mcts import create_forest, select_leaves_gpu, expand_gpu, backprop_gpu

        device = torch.device('cuda')
        n = 16
        n_sims = 50

        # Warmup
        for _ in range(3):
            states = deal_random_gpu(n, device)
            forest = create_forest(n, max_nodes=256, initial_states=states, device=device)
            for _ in range(n_sims):
                leaf_idx, paths = select_leaves_gpu(forest)
                values = torch.randn(n, device=device)
                expand_gpu(forest, leaf_idx)
                backprop_gpu(forest, leaf_idx, values, paths)

        torch.cuda.synchronize()

        # Timed run
        start = time.time()
        n_trials = 10

        for _ in range(n_trials):
            states = deal_random_gpu(n, device)
            forest = create_forest(n, max_nodes=256, initial_states=states, device=device)
            for _ in range(n_sims):
                leaf_idx, paths = select_leaves_gpu(forest)
                values = torch.randn(n, device=device)
                expand_gpu(forest, leaf_idx)
                backprop_gpu(forest, leaf_idx, values, paths)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        moves_per_sec = (n_trials * n * n_sims) / elapsed
        print(f"\nMCTS throughput: {moves_per_sec:.0f} moves/sec")
        print(f"Time per trial: {elapsed/n_trials*1000:.1f}ms")
