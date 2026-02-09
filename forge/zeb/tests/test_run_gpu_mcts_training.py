"""Tests for GPU MCTS training script (run_mcts_training.py with GPU pipeline).

Minimal tests to verify the training script components work.
Uses mock oracle to avoid slow model loading.
"""

import pytest
import torch
from unittest.mock import MagicMock


# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@pytest.fixture
def mock_oracle():
    """Create mock oracle for fast testing."""
    oracle = MagicMock()

    def batch_evaluate_gpu(**kwargs):
        n = kwargs['original_hands'].shape[0]
        device = kwargs['original_hands'].device
        return torch.randn(n, dtype=torch.float32, device=device)

    oracle.batch_evaluate_gpu = batch_evaluate_gpu
    return oracle


class TestGPUTrainingPipelineCreation:
    """Test pipeline can be created and configured."""

    def test_pipeline_creates_with_mock_oracle(self, mock_oracle):
        """Pipeline should initialize with mock oracle."""
        from forge.zeb.gpu_training_pipeline import GPUTrainingPipeline

        device = torch.device('cuda')
        pipeline = GPUTrainingPipeline(
            oracle=mock_oracle,
            device=device,
            n_parallel_games=1,
            n_simulations=3,
            max_mcts_nodes=32,
        )

        assert pipeline.device.type == 'cuda'
        assert pipeline.n_simulations == 3
        assert pipeline.max_mcts_nodes == 32


class TestMinimalTrainingRun:
    """Test a minimal training run completes without errors."""

    def test_generate_single_game(self, mock_oracle):
        """Generate training examples from 1 game."""
        from forge.zeb.gpu_training_pipeline import GPUTrainingPipeline

        device = torch.device('cuda')
        # Use n_simulations=10 to ensure MCTS builds valid visit distributions
        # (too few sims with random oracle can produce zero-probability policies)
        pipeline = GPUTrainingPipeline(
            oracle=mock_oracle,
            device=device,
            n_parallel_games=1,
            n_simulations=10,
            max_mcts_nodes=64,
        )

        examples = pipeline.generate_games_gpu(n_games=1)

        # Should have some examples (up to 28 per game)
        assert examples.n_examples > 0
        assert examples.n_examples <= 28

        # All on GPU
        assert examples.observations.device.type == 'cuda'
        assert examples.policy_targets.device.type == 'cuda'
        assert examples.value_targets.device.type == 'cuda'

    def test_train_single_epoch(self, mock_oracle):
        """Train for one epoch on generated examples."""
        from forge.zeb.gpu_training_pipeline import GPUTrainingPipeline
        from forge.zeb.model import ZebModel, get_model_config

        device = torch.device('cuda')

        # Create small model
        config = get_model_config('small')
        model = ZebModel(**config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create pipeline with enough simulations for valid policies
        pipeline = GPUTrainingPipeline(
            oracle=mock_oracle,
            device=device,
            n_parallel_games=1,
            n_simulations=10,
            max_mcts_nodes=64,
        )

        # Generate minimal examples
        examples = pipeline.generate_games_gpu(n_games=1)

        # Train one epoch
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


class TestTrainingScriptImports:
    """Test that training script imports work."""

    def test_imports_succeed(self):
        """All training script imports should work."""
        # These imports are used by run_mcts_training.py
        from forge.zeb.mcts_self_play import play_games_with_mcts, mcts_examples_to_zeb_tensors
        from forge.zeb.batched_mcts import play_games_batched
        from forge.zeb.model import ZebModel, get_model_config
        from forge.zeb.observation import observe, N_HAND_SLOTS
        from forge.zeb.types import ZebGameState, GamePhase, BidState

        # GPU pipeline imports
        from forge.zeb.gpu_training_pipeline import (
            GPUTrainingPipeline,
            GPUObservationTokenizer,
            TrainingExamples,
        )
        from forge.zeb.gpu_game_state import deal_random_gpu
        from forge.zeb.gpu_mcts import create_forest

        assert True  # If we get here, imports worked

    def test_model_configs_exist(self):
        """Model configs should be available."""
        from forge.zeb.model import get_model_config

        for size in ['small', 'medium', 'large']:
            config = get_model_config(size)
            assert 'embed_dim' in config
            assert 'n_layers' in config
            assert 'n_heads' in config
