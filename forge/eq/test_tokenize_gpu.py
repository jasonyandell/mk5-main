"""
Tests for GPU tokenizer.

Verifies correctness against CPU implementation and performance targets.
"""
import time

import numpy as np
import pytest
import torch

from forge.eq.tokenize_gpu import GPUTokenizer
from forge.eq.oracle import Stage1Oracle


@pytest.fixture
def tokenizer_cpu():
    """Create a CPU tokenizer for reference."""
    return GPUTokenizer(max_batch=2000, device='cpu')


@pytest.fixture
def tokenizer_gpu():
    """Create a GPU tokenizer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return GPUTokenizer(max_batch=2000, device='cuda')


def test_gpu_tokenizer_output_shape(tokenizer_cpu):
    """Test that output shapes are correct."""
    # Create sample inputs
    n_worlds = 10
    worlds = torch.randint(0, 28, (n_worlds, 4, 7), dtype=torch.int8, device='cpu')
    remaining = torch.randint(0, 128, (n_worlds, 4), dtype=torch.int64, device='cpu')

    tokens, masks = tokenizer_cpu.tokenize(
        worlds=worlds,
        decl_id=3,
        leader=0,
        trick_plays=[],
        remaining=remaining,
        current_player=0,
    )

    # Check shapes
    assert tokens.shape == (n_worlds, 32, 12)
    assert masks.shape == (n_worlds, 32)
    assert tokens.dtype == torch.int8
    assert masks.dtype == torch.int8


def test_gpu_tokenizer_matches_cpu_oracle():
    """Test that GPU tokenizer output matches CPU oracle._tokenize_worlds()."""
    # Set up deterministic random inputs
    torch.manual_seed(42)
    np.random.seed(42)

    n_worlds = 50
    decl_id = 5
    leader = 1
    current_player = 2

    # Generate random worlds (4 hands of 7 dominoes)
    worlds_list = []
    for _ in range(n_worlds):
        dominoes = np.random.permutation(28)
        world = [
            dominoes[0:7].tolist(),
            dominoes[7:14].tolist(),
            dominoes[14:21].tolist(),
            dominoes[21:28].tolist(),
        ]
        worlds_list.append(world)

    # Generate remaining bitmasks (track which dominoes are still in each hand)
    remaining_np = np.zeros((n_worlds, 4), dtype=np.int64)
    for i, world in enumerate(worlds_list):
        for player in range(4):
            # Set bits for dominoes in hand (assume all 7 are remaining)
            for local_idx in range(7):
                remaining_np[i, player] |= (1 << local_idx)

    # Empty trick for simplicity
    trick_plays = []

    # CPU reference (using Stage1Oracle's _tokenize_worlds)
    oracle = Stage1Oracle.__new__(Stage1Oracle)  # Skip __init__
    tokens_cpu, masks_cpu = oracle._tokenize_worlds(
        worlds=worlds_list,
        decl_id=decl_id,
        leader=leader,
        trick_plays=trick_plays,
        remaining=remaining_np,
        current_player=current_player,
    )

    # GPU implementation
    tokenizer = GPUTokenizer(max_batch=100, device='cpu')
    worlds_tensor = torch.tensor(worlds_list, dtype=torch.int8, device='cpu')
    remaining_tensor = torch.tensor(remaining_np, dtype=torch.int64, device='cpu')

    tokens_gpu, masks_gpu = tokenizer.tokenize(
        worlds=worlds_tensor,
        decl_id=decl_id,
        leader=leader,
        trick_plays=trick_plays,
        remaining=remaining_tensor,
        current_player=current_player,
    )

    # Convert GPU tensors to numpy for comparison
    tokens_gpu_np = tokens_gpu.cpu().numpy()
    masks_gpu_np = masks_gpu.cpu().numpy()

    # Compare
    np.testing.assert_array_equal(
        tokens_gpu_np, tokens_cpu,
        err_msg="GPU tokens don't match CPU oracle"
    )
    np.testing.assert_array_equal(
        masks_gpu_np, masks_cpu,
        err_msg="GPU masks don't match CPU oracle"
    )


def test_gpu_tokenizer_with_tricks():
    """Test tokenizer with non-empty trick_plays."""
    torch.manual_seed(42)
    np.random.seed(42)

    n_worlds = 20
    decl_id = 3
    leader = 0
    current_player = 2
    trick_plays = [(0, 5), (1, 12)]  # 2 plays in current trick

    # Generate random worlds
    worlds_list = []
    for _ in range(n_worlds):
        dominoes = np.random.permutation(28)
        world = [
            dominoes[0:7].tolist(),
            dominoes[7:14].tolist(),
            dominoes[14:21].tolist(),
            dominoes[21:28].tolist(),
        ]
        worlds_list.append(world)

    # Generate remaining bitmasks
    remaining_np = np.zeros((n_worlds, 4), dtype=np.int64)
    for i, world in enumerate(worlds_list):
        for player in range(4):
            for local_idx in range(7):
                remaining_np[i, player] |= (1 << local_idx)

    # CPU reference
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    tokens_cpu, masks_cpu = oracle._tokenize_worlds(
        worlds=worlds_list,
        decl_id=decl_id,
        leader=leader,
        trick_plays=trick_plays,
        remaining=remaining_np,
        current_player=current_player,
    )

    # GPU implementation
    tokenizer = GPUTokenizer(max_batch=100, device='cpu')
    worlds_tensor = torch.tensor(worlds_list, dtype=torch.int8, device='cpu')
    remaining_tensor = torch.tensor(remaining_np, dtype=torch.int64, device='cpu')

    tokens_gpu, masks_gpu = tokenizer.tokenize(
        worlds=worlds_tensor,
        decl_id=decl_id,
        leader=leader,
        trick_plays=trick_plays,
        remaining=remaining_tensor,
        current_player=current_player,
    )

    # Convert and compare
    tokens_gpu_np = tokens_gpu.cpu().numpy()
    masks_gpu_np = masks_gpu.cpu().numpy()

    np.testing.assert_array_equal(tokens_gpu_np, tokens_cpu)
    np.testing.assert_array_equal(masks_gpu_np, masks_cpu)


def test_gpu_tokenizer_all_declarations():
    """Test tokenizer with all 10 declaration types."""
    torch.manual_seed(42)
    np.random.seed(42)

    n_worlds = 10

    for decl_id in range(10):
        # Generate random worlds
        worlds_list = []
        for _ in range(n_worlds):
            dominoes = np.random.permutation(28)
            world = [
                dominoes[0:7].tolist(),
                dominoes[7:14].tolist(),
                dominoes[14:21].tolist(),
                dominoes[21:28].tolist(),
            ]
            worlds_list.append(world)

        remaining_np = np.ones((n_worlds, 4), dtype=np.int64) * 0x7F  # All 7 remaining

        # CPU reference
        oracle = Stage1Oracle.__new__(Stage1Oracle)
        tokens_cpu, masks_cpu = oracle._tokenize_worlds(
            worlds=worlds_list,
            decl_id=decl_id,
            leader=0,
            trick_plays=[],
            remaining=remaining_np,
            current_player=0,
        )

        # GPU implementation
        tokenizer = GPUTokenizer(max_batch=100, device='cpu')
        worlds_tensor = torch.tensor(worlds_list, dtype=torch.int8, device='cpu')
        remaining_tensor = torch.tensor(remaining_np, dtype=torch.int64, device='cpu')

        tokens_gpu, masks_gpu = tokenizer.tokenize(
            worlds=worlds_tensor,
            decl_id=decl_id,
            leader=0,
            trick_plays=[],
            remaining=remaining_tensor,
            current_player=0,
        )

        tokens_gpu_np = tokens_gpu.cpu().numpy()
        masks_gpu_np = masks_gpu.cpu().numpy()

        np.testing.assert_array_equal(
            tokens_gpu_np, tokens_cpu,
            err_msg=f"Tokens mismatch for decl_id={decl_id}"
        )
        np.testing.assert_array_equal(
            masks_gpu_np, masks_cpu,
            err_msg=f"Masks mismatch for decl_id={decl_id}"
        )


def test_gpu_tokenizer_remaining_bits():
    """Test that remaining bits are correctly encoded."""
    # Create a controlled scenario where we can verify remaining bits
    n_worlds = 5
    decl_id = 0
    worlds_list = []

    # Use fixed worlds for predictable testing
    for _ in range(n_worlds):
        worlds_list.append([
            [0, 1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13],
            [14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27],
        ])

    # Set specific remaining patterns
    remaining_np = np.zeros((n_worlds, 4), dtype=np.int64)
    remaining_np[0] = [0b0000001, 0b0000010, 0b0000100, 0b0001000]  # One per player
    remaining_np[1] = [0b1111111, 0b1111111, 0b1111111, 0b1111111]  # All remaining
    remaining_np[2] = [0b0000000, 0b0000000, 0b0000000, 0b0000000]  # None remaining
    remaining_np[3] = [0b1010101, 0b0101010, 0b1010101, 0b0101010]  # Alternating
    remaining_np[4] = [0b1110000, 0b0001110, 0b0000111, 0b1000001]  # Mixed

    # Tokenize with GPU
    tokenizer = GPUTokenizer(max_batch=100, device='cpu')
    worlds_tensor = torch.tensor(worlds_list, dtype=torch.int8, device='cpu')
    remaining_tensor = torch.tensor(remaining_np, dtype=torch.int64, device='cpu')

    tokens, masks = tokenizer.tokenize(
        worlds=worlds_tensor,
        decl_id=decl_id,
        leader=0,
        trick_plays=[],
        remaining=remaining_tensor,
        current_player=0,
    )

    # Check that remaining bits are correctly encoded in feature 8
    # For world 1 (all remaining), all hand tokens should have remaining=1
    world_1_remaining = tokens[1, 1:29, 8]  # 28 hand tokens
    assert torch.all(world_1_remaining == 1), "World 1 should have all remaining=1"

    # For world 2 (none remaining), all should be 0
    world_2_remaining = tokens[2, 1:29, 8]
    assert torch.all(world_2_remaining == 0), "World 2 should have all remaining=0"

    # For world 0, check specific bits
    # Player 0's first domino (local_idx=0) should be remaining=1
    assert tokens[0, 1, 8] == 1, "Player 0 local_idx 0 should be remaining"
    # Player 0's other dominoes should be 0
    for i in range(2, 8):
        assert tokens[0, i, 8] == 0, f"Player 0 local_idx {i-1} should not be remaining"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_tokenizer_performance():
    """Test that GPU tokenizer meets <50ms target for 1,600 batch."""
    torch.manual_seed(42)
    np.random.seed(42)

    n_worlds = 1600
    decl_id = 3

    # Generate random worlds
    worlds = torch.randint(0, 28, (n_worlds, 4, 7), dtype=torch.int8, device='cuda')
    remaining = torch.ones(n_worlds, 4, dtype=torch.int64, device='cuda') * 0x7F

    # Create tokenizer
    tokenizer = GPUTokenizer(max_batch=2000, device='cuda')

    # Warmup (JIT compilation, cache warmup)
    for _ in range(5):
        tokenizer.tokenize(
            worlds=worlds[:100],
            decl_id=decl_id,
            leader=0,
            trick_plays=[],
            remaining=remaining[:100],
            current_player=0,
        )
        torch.cuda.synchronize()

    # Benchmark
    n_runs = 10
    times = []

    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        tokenizer.tokenize(
            worlds=worlds,
            decl_id=decl_id,
            leader=0,
            trick_plays=[(0, 5)],
            remaining=remaining,
            current_player=0,
        )

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nGPU Tokenizer Performance (n=1600):")
    print(f"  Mean: {mean_time:.2f}ms ± {std_time:.2f}ms")
    print(f"  Min: {np.min(times):.2f}ms")
    print(f"  Max: {np.max(times):.2f}ms")

    # Target: <50ms on RTX 3050 Ti
    # Use 75ms as threshold to account for different GPUs
    assert mean_time < 75, f"GPU tokenizer too slow: {mean_time:.2f}ms (target: <50ms on 3050 Ti)"


def test_gpu_tokenizer_batch_size_limit():
    """Test that tokenizer enforces max_batch limit."""
    tokenizer = GPUTokenizer(max_batch=100, device='cpu')

    # Try to tokenize more than max_batch
    n_worlds = 150
    worlds = torch.randint(0, 28, (n_worlds, 4, 7), dtype=torch.int8, device='cpu')
    remaining = torch.zeros(n_worlds, 4, dtype=torch.int64, device='cpu')

    with pytest.raises(ValueError, match="exceeds max_batch"):
        tokenizer.tokenize(
            worlds=worlds,
            decl_id=0,
            leader=0,
            trick_plays=[],
            remaining=remaining,
            current_player=0,
        )


def test_gpu_tokenizer_all_players():
    """Test tokenizer with different current_player values."""
    torch.manual_seed(42)
    np.random.seed(42)

    n_worlds = 10
    decl_id = 3

    # Test all 4 current_player values
    for current_player in range(4):
        # Generate worlds
        worlds_list = []
        for _ in range(n_worlds):
            dominoes = np.random.permutation(28)
            world = [
                dominoes[0:7].tolist(),
                dominoes[7:14].tolist(),
                dominoes[14:21].tolist(),
                dominoes[21:28].tolist(),
            ]
            worlds_list.append(world)

        remaining_np = np.ones((n_worlds, 4), dtype=np.int64) * 0x7F

        # CPU reference
        oracle = Stage1Oracle.__new__(Stage1Oracle)
        tokens_cpu, masks_cpu = oracle._tokenize_worlds(
            worlds=worlds_list,
            decl_id=decl_id,
            leader=0,
            trick_plays=[],
            remaining=remaining_np,
            current_player=current_player,
        )

        # GPU implementation
        tokenizer = GPUTokenizer(max_batch=100, device='cpu')
        worlds_tensor = torch.tensor(worlds_list, dtype=torch.int8, device='cpu')
        remaining_tensor = torch.tensor(remaining_np, dtype=torch.int64, device='cpu')

        tokens_gpu, masks_gpu = tokenizer.tokenize(
            worlds=worlds_tensor,
            decl_id=decl_id,
            leader=0,
            trick_plays=[],
            remaining=remaining_tensor,
            current_player=current_player,
        )

        tokens_gpu_np = tokens_gpu.cpu().numpy()
        masks_gpu_np = masks_gpu.cpu().numpy()

        np.testing.assert_array_equal(
            tokens_gpu_np, tokens_cpu,
            err_msg=f"Tokens mismatch for current_player={current_player}"
        )
        np.testing.assert_array_equal(
            masks_gpu_np, masks_cpu,
            err_msg=f"Masks mismatch for current_player={current_player}"
        )


def test_gpu_tokenizer_max_trick_plays():
    """Test tokenizer with maximum 3 trick plays."""
    torch.manual_seed(42)
    np.random.seed(42)

    n_worlds = 10
    decl_id = 5
    trick_plays = [(0, 3), (1, 8), (2, 15)]  # 3 plays (max)

    worlds_list = []
    for _ in range(n_worlds):
        dominoes = np.random.permutation(28)
        world = [
            dominoes[0:7].tolist(),
            dominoes[7:14].tolist(),
            dominoes[14:21].tolist(),
            dominoes[21:28].tolist(),
        ]
        worlds_list.append(world)

    remaining_np = np.ones((n_worlds, 4), dtype=np.int64) * 0x7F

    # CPU reference
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    tokens_cpu, masks_cpu = oracle._tokenize_worlds(
        worlds=worlds_list,
        decl_id=decl_id,
        leader=0,
        trick_plays=trick_plays,
        remaining=remaining_np,
        current_player=3,
    )

    # GPU implementation
    tokenizer = GPUTokenizer(max_batch=100, device='cpu')
    worlds_tensor = torch.tensor(worlds_list, dtype=torch.int8, device='cpu')
    remaining_tensor = torch.tensor(remaining_np, dtype=torch.int64, device='cpu')

    tokens_gpu, masks_gpu = tokenizer.tokenize(
        worlds=worlds_tensor,
        decl_id=decl_id,
        leader=0,
        trick_plays=trick_plays,
        remaining=remaining_tensor,
        current_player=3,
    )

    tokens_gpu_np = tokens_gpu.cpu().numpy()
    masks_gpu_np = masks_gpu.cpu().numpy()

    np.testing.assert_array_equal(tokens_gpu_np, tokens_cpu)
    np.testing.assert_array_equal(masks_gpu_np, masks_cpu)

    # Verify that all 3 trick tokens are present
    # Trick tokens are at indices 29, 30, 31
    for world_idx in range(n_worlds):
        assert masks_gpu_np[world_idx, 29] == 1, "Trick token 0 should be present"
        assert masks_gpu_np[world_idx, 30] == 1, "Trick token 1 should be present"
        assert masks_gpu_np[world_idx, 31] == 1, "Trick token 2 should be present"
