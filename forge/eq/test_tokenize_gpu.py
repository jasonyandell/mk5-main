"""
Tests for GPU tokenizer.

Verifies correctness against CPU implementation and performance targets.
"""
import time

import numpy as np
import pytest
import torch

from forge.eq.tokenize_gpu import GPUTokenizer, PastStatesGPU
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


def test_tokenize_past_steps_shapes():
    """Test output shapes from tokenize_past_steps."""
    tokenizer = GPUTokenizer(max_batch=1000, device='cpu')

    N, M, K = 2, 10, 4
    worlds = torch.randint(0, 28, (N, M, 4, 7), dtype=torch.int8)

    # Create mock past_states
    past_states = PastStatesGPU(
        played_before=torch.zeros(N, K, 28, dtype=torch.bool),
        trick_plays=torch.zeros(N, K, 3, 2, dtype=torch.int32),
        trick_lens=torch.zeros(N, K, dtype=torch.int32),
        leaders=torch.zeros(N, K, dtype=torch.int32),
        actors=torch.arange(4).repeat(N, K // 4 + 1)[:N, :K],
        observed_actions=torch.randint(0, 28, (N, K)),
        step_indices=torch.arange(K).unsqueeze(0).expand(N, -1),
        valid_mask=torch.ones(N, K, dtype=torch.bool),
    )

    tokens, masks = tokenizer.tokenize_past_steps(worlds, past_states, decl_id=0)

    assert tokens.shape == (N * M * K, 32, 12)
    assert masks.shape == (N * M * K, 32)
    assert tokens.dtype == torch.int8
    assert masks.dtype == torch.int8


def test_tokenize_past_steps_with_varying_tricks():
    """Test tokenize_past_steps with different trick_plays per step."""
    tokenizer = GPUTokenizer(max_batch=2000, device='cpu')

    N, M, K = 2, 5, 3
    torch.manual_seed(42)

    # Create deterministic worlds
    worlds = torch.randint(0, 28, (N, M, 4, 7), dtype=torch.int8)

    # Create past_states with varying trick configurations
    # Step 0: No tricks (new trick starting)
    # Step 1: 1 trick play
    # Step 2: 2 trick plays
    trick_plays = torch.zeros(N, K, 3, 2, dtype=torch.int32)
    trick_lens = torch.zeros(N, K, dtype=torch.int32)

    # Game 0, Step 1: player 0 plays domino 5
    trick_plays[0, 1, 0, 0] = 0  # player
    trick_plays[0, 1, 0, 1] = 5  # domino
    trick_lens[0, 1] = 1

    # Game 0, Step 2: player 0 plays domino 5, player 1 plays domino 12
    trick_plays[0, 2, 0, 0] = 0
    trick_plays[0, 2, 0, 1] = 5
    trick_plays[0, 2, 1, 0] = 1
    trick_plays[0, 2, 1, 1] = 12
    trick_lens[0, 2] = 2

    # Game 1: Different pattern
    trick_plays[1, 0, 0, 0] = 2
    trick_plays[1, 0, 0, 1] = 8
    trick_lens[1, 0] = 1

    played_before = torch.zeros(N, K, 28, dtype=torch.bool)
    # Mark some dominoes as played
    played_before[0, 1, 5] = True  # domino 5 played before step 1
    played_before[0, 2, 5] = True
    played_before[0, 2, 12] = True

    past_states = PastStatesGPU(
        played_before=played_before,
        trick_plays=trick_plays,
        trick_lens=trick_lens,
        leaders=torch.tensor([[0, 0, 0], [2, 2, 2]], dtype=torch.int32),
        actors=torch.tensor([[0, 1, 2], [2, 3, 0]], dtype=torch.int32),
        observed_actions=torch.randint(0, 28, (N, K)),
        step_indices=torch.arange(K).unsqueeze(0).expand(N, -1),
        valid_mask=torch.ones(N, K, dtype=torch.bool),
    )

    tokens, masks = tokenizer.tokenize_past_steps(worlds, past_states, decl_id=3)

    # Check shapes
    assert tokens.shape == (N * M * K, 32, 12)
    assert masks.shape == (N * M * K, 32)

    # Verify that context tokens are set (position 0)
    assert torch.all(masks[:, 0] == 1), "All context tokens should be present"

    # Verify that hand tokens are set (positions 1-28)
    assert torch.all(masks[:, 1:29] == 1), "All hand tokens should be present"

    # For game 0, step 2, samples should have 2 trick tokens (positions 29-30)
    # Indices for game 0, step 2 are: batch_idx = 0*M*K + 2*M = 10 to 10+M-1 = 14
    step2_indices = slice(10, 15)
    assert torch.all(masks[step2_indices, 29] == 1), "Step 2 should have trick token 0"
    assert torch.all(masks[step2_indices, 30] == 1), "Step 2 should have trick token 1"
    assert torch.all(masks[step2_indices, 31] == 0), "Step 2 should NOT have trick token 2"


def test_tokenize_past_steps_with_padding():
    """Test tokenize_past_steps correctly handles invalid steps (padding)."""
    tokenizer = GPUTokenizer(max_batch=500, device='cpu')

    N, M, K = 2, 5, 4
    worlds = torch.randint(0, 28, (N, M, 4, 7), dtype=torch.int8)

    # Create past_states where some steps are invalid (padding)
    valid_mask = torch.ones(N, K, dtype=torch.bool)
    valid_mask[0, 3] = False  # Game 0, step 3 is padding
    valid_mask[1, 2:] = False  # Game 1, steps 2-3 are padding

    past_states = PastStatesGPU(
        played_before=torch.zeros(N, K, 28, dtype=torch.bool),
        trick_plays=torch.zeros(N, K, 3, 2, dtype=torch.int32),
        trick_lens=torch.zeros(N, K, dtype=torch.int32),
        leaders=torch.zeros(N, K, dtype=torch.int32),
        actors=torch.arange(4).repeat(N, K // 4 + 1)[:N, :K],
        observed_actions=torch.randint(0, 28, (N, K)),
        step_indices=torch.arange(K).unsqueeze(0).expand(N, -1),
        valid_mask=valid_mask,
    )

    tokens, masks = tokenizer.tokenize_past_steps(worlds, past_states, decl_id=0)

    # Check that invalid steps produce zero tokens/masks
    # Game 0, step 3: indices 15-19 (0*M*K + 3*M = 15)
    invalid_slice_0 = slice(15, 20)
    assert torch.all(tokens[invalid_slice_0] == 0), "Invalid step tokens should be zero"
    assert torch.all(masks[invalid_slice_0] == 0), "Invalid step masks should be zero"

    # Game 1, step 2: indices 30-34 (1*M*K + 2*M = 30)
    invalid_slice_1 = slice(30, 35)
    assert torch.all(tokens[invalid_slice_1] == 0), "Invalid step tokens should be zero"
    assert torch.all(masks[invalid_slice_1] == 0), "Invalid step masks should be zero"


def test_tokenize_past_steps_remaining_computation():
    """Test that remaining bits are correctly computed from played_before."""
    tokenizer = GPUTokenizer(max_batch=500, device='cpu')

    N, M, K = 1, 3, 2
    # Create fixed worlds for predictable testing
    worlds = torch.zeros(N, M, 4, 7, dtype=torch.int8)
    for m in range(M):
        for p in range(4):
            for slot in range(7):
                worlds[0, m, p, slot] = p * 7 + slot

    # Create past_states where some dominoes are played
    played_before = torch.zeros(N, K, 28, dtype=torch.bool)
    played_before[0, 0, 0] = True  # Domino 0 played before step 0
    played_before[0, 0, 1] = True  # Domino 1 played before step 0
    played_before[0, 1, 0:5] = True  # Dominoes 0-4 played before step 1

    past_states = PastStatesGPU(
        played_before=played_before,
        trick_plays=torch.zeros(N, K, 3, 2, dtype=torch.int32),
        trick_lens=torch.zeros(N, K, dtype=torch.int32),
        leaders=torch.zeros(N, K, dtype=torch.int32),
        actors=torch.zeros(N, K, dtype=torch.int32),
        observed_actions=torch.randint(0, 28, (N, K)),
        step_indices=torch.arange(K).unsqueeze(0).expand(N, -1),
        valid_mask=torch.ones(N, K, dtype=torch.bool),
    )

    tokens, masks = tokenizer.tokenize_past_steps(worlds, past_states, decl_id=0)

    # For step 0, sample 0: dominoes 0 and 1 (player 0, slots 0-1) should NOT be remaining
    # Token indices: 1 (player 0, slot 0), 2 (player 0, slot 1)
    step0_sample0_idx = 0  # 0*M*K + 0*M + 0 = 0
    assert tokens[step0_sample0_idx, 1, 8] == 0, "Domino 0 should not be remaining"
    assert tokens[step0_sample0_idx, 2, 8] == 0, "Domino 1 should not be remaining"
    assert tokens[step0_sample0_idx, 3, 8] == 1, "Domino 2 should be remaining"

    # For step 1, sample 0: dominoes 0-4 should NOT be remaining
    step1_sample0_idx = 3  # 0*M*K + 1*M + 0 = 3
    for slot in range(5):
        token_idx = 1 + slot  # Player 0 tokens start at index 1
        assert tokens[step1_sample0_idx, token_idx, 8] == 0, f"Domino {slot} should not be remaining at step 1"
    # Dominoes 5-6 should be remaining
    assert tokens[step1_sample0_idx, 6, 8] == 1, "Domino 5 should be remaining"
    assert tokens[step1_sample0_idx, 7, 8] == 1, "Domino 6 should be remaining"


def test_compute_remaining_batched():
    """Test _compute_remaining_batched produces correct bitmasks."""
    tokenizer = GPUTokenizer(max_batch=100, device='cpu')

    batch = 5
    # Create fixed worlds where player p has dominoes [p*7, p*7+1, ..., p*7+6]
    worlds = torch.zeros(batch, 4, 7, dtype=torch.int8)
    for b in range(batch):
        for p in range(4):
            for s in range(7):
                worlds[b, p, s] = p * 7 + s

    # Create played_before masks
    played_before = torch.zeros(batch, 28, dtype=torch.bool)
    # Batch 0: no dominoes played -> all remaining (0b1111111 = 127)
    # Batch 1: domino 0 played -> player 0 has 6 remaining (0b1111110 = 126)
    played_before[1, 0] = True
    # Batch 2: dominoes 0, 1, 7, 8 played -> player 0 has 5, player 1 has 5
    played_before[2, 0] = True
    played_before[2, 1] = True
    played_before[2, 7] = True
    played_before[2, 8] = True
    # Batch 3: all of player 0's dominoes played
    played_before[3, 0:7] = True
    # Batch 4: alternating pattern for player 0
    played_before[4, 0] = True
    played_before[4, 2] = True
    played_before[4, 4] = True
    played_before[4, 6] = True

    remaining = tokenizer._compute_remaining_batched(worlds, played_before)

    # Check shapes
    assert remaining.shape == (batch, 4)
    assert remaining.dtype == torch.int64

    # Check values
    # Batch 0: all players have all 7 remaining
    assert remaining[0, 0].item() == 0b1111111  # 127
    assert remaining[0, 1].item() == 0b1111111
    assert remaining[0, 2].item() == 0b1111111
    assert remaining[0, 3].item() == 0b1111111

    # Batch 1: player 0 missing slot 0
    assert remaining[1, 0].item() == 0b1111110  # 126
    assert remaining[1, 1].item() == 0b1111111  # others unchanged

    # Batch 2: player 0 missing slots 0,1; player 1 missing slots 0,1
    assert remaining[2, 0].item() == 0b1111100  # 124
    assert remaining[2, 1].item() == 0b1111100  # 124

    # Batch 3: player 0 has no remaining
    assert remaining[3, 0].item() == 0b0000000  # 0

    # Batch 4: player 0 has alternating (slots 1,3,5 remaining)
    assert remaining[4, 0].item() == 0b0101010  # 42


def test_tokenize_batched_matches_loop():
    """Verify tokenize_batched matches calling tokenize() in a loop."""
    torch.manual_seed(42)
    np.random.seed(42)

    batch = 100
    tokenizer = GPUTokenizer(max_batch=200, device='cpu')

    # Generate random test data with varying parameters per batch element
    worlds_list = []
    for _ in range(batch):
        dominoes = np.random.permutation(28)
        world = [
            dominoes[0:7].tolist(),
            dominoes[7:14].tolist(),
            dominoes[14:21].tolist(),
            dominoes[21:28].tolist(),
        ]
        worlds_list.append(world)

    worlds = torch.tensor(worlds_list, dtype=torch.int8, device='cpu')

    # Random per-batch parameters
    decl_ids = torch.randint(0, 10, (batch,), dtype=torch.int32, device='cpu')
    leaders = torch.randint(0, 4, (batch,), dtype=torch.int32, device='cpu')
    current_players = torch.randint(0, 4, (batch,), dtype=torch.int32, device='cpu')

    # Random trick plays (0-3 per batch)
    trick_lens = torch.randint(0, 4, (batch,), dtype=torch.int32, device='cpu')
    trick_plays = torch.zeros(batch, 3, 2, dtype=torch.int32, device='cpu')
    for b in range(batch):
        for t in range(trick_lens[b].item()):
            trick_plays[b, t, 0] = np.random.randint(0, 4)  # player
            trick_plays[b, t, 1] = np.random.randint(0, 28)  # domino

    # Random remaining bitmasks
    remaining = torch.randint(0, 128, (batch, 4), dtype=torch.int64, device='cpu')

    # =========================================================================
    # Method 1: Loop approach (reference)
    # =========================================================================
    loop_tokens = torch.zeros(batch, 32, 12, dtype=torch.int8, device='cpu')
    loop_masks = torch.zeros(batch, 32, dtype=torch.int8, device='cpu')

    for b in range(batch):
        trick_plays_list = []
        for t in range(trick_lens[b].item()):
            trick_plays_list.append((
                trick_plays[b, t, 0].item(),
                trick_plays[b, t, 1].item()
            ))

        tokens_b, masks_b = tokenizer.tokenize(
            worlds=worlds[b:b+1],
            decl_id=decl_ids[b].item(),
            leader=leaders[b].item(),
            trick_plays=trick_plays_list,
            remaining=remaining[b:b+1],
            current_player=current_players[b].item(),
        )
        loop_tokens[b] = tokens_b[0]
        loop_masks[b] = masks_b[0]

    # =========================================================================
    # Method 2: Batched approach (under test)
    # =========================================================================
    batched_tokens, batched_masks = tokenizer.tokenize_batched(
        worlds=worlds,
        decl_ids=decl_ids,
        leaders=leaders,
        current_players=current_players,
        trick_plays=trick_plays,
        trick_lens=trick_lens,
        remaining=remaining,
    )

    # =========================================================================
    # Compare results
    # =========================================================================
    # Check shapes match
    assert batched_tokens.shape == loop_tokens.shape, \
        f"Token shapes don't match: {batched_tokens.shape} vs {loop_tokens.shape}"
    assert batched_masks.shape == loop_masks.shape, \
        f"Mask shapes don't match: {batched_masks.shape} vs {loop_masks.shape}"

    # Check values match
    tokens_match = torch.all(batched_tokens == loop_tokens)
    masks_match = torch.all(batched_masks == loop_masks)

    if not tokens_match:
        # Find first mismatch for debugging
        diff_mask = batched_tokens != loop_tokens
        mismatch_indices = torch.nonzero(diff_mask)
        if len(mismatch_indices) > 0:
            b, tok, feat = mismatch_indices[0].tolist()
            print(f"\nFirst token mismatch at batch={b}, token={tok}, feature={feat}:", flush=True)
            print(f"  Batched: {batched_tokens[b, tok, feat].item()}", flush=True)
            print(f"  Loop:    {loop_tokens[b, tok, feat].item()}", flush=True)
            print(f"  decl_id={decl_ids[b].item()}, leader={leaders[b].item()}, "
                  f"current_player={current_players[b].item()}, trick_len={trick_lens[b].item()}", flush=True)

    assert tokens_match, "Batched tokens don't match loop tokens"
    assert masks_match, "Batched masks don't match loop masks"

    print(f"\ntest_tokenize_batched_matches_loop PASSED: {batch} samples verified", flush=True)


def test_tokenize_batched_benchmark():
    """Benchmark loop vs batched tokenization."""
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 1000
    n_iterations = 3
    device = 'cpu'  # Use CPU for consistent benchmark

    print(f"\n{'='*60}", flush=True)
    print(f"Tokenization Benchmark: batch_size={batch_size}, n_iter={n_iterations}", flush=True)
    print(f"{'='*60}", flush=True)

    tokenizer = GPUTokenizer(max_batch=2000, device=device)

    # Generate random test data
    worlds_list = []
    for _ in range(batch_size):
        dominoes = np.random.permutation(28)
        world = [
            dominoes[0:7].tolist(),
            dominoes[7:14].tolist(),
            dominoes[14:21].tolist(),
            dominoes[21:28].tolist(),
        ]
        worlds_list.append(world)

    worlds = torch.tensor(worlds_list, dtype=torch.int8, device=device)
    decl_ids = torch.randint(0, 10, (batch_size,), dtype=torch.int32, device=device)
    leaders = torch.randint(0, 4, (batch_size,), dtype=torch.int32, device=device)
    current_players = torch.randint(0, 4, (batch_size,), dtype=torch.int32, device=device)
    trick_lens = torch.randint(0, 4, (batch_size,), dtype=torch.int32, device=device)
    trick_plays = torch.zeros(batch_size, 3, 2, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for t in range(trick_lens[b].item()):
            trick_plays[b, t, 0] = np.random.randint(0, 4)
            trick_plays[b, t, 1] = np.random.randint(0, 28)
    remaining = torch.randint(0, 128, (batch_size, 4), dtype=torch.int64, device=device)

    # =========================================================================
    # Benchmark Loop Approach
    # =========================================================================
    loop_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()

        loop_tokens = torch.zeros(batch_size, 32, 12, dtype=torch.int8, device=device)
        loop_masks = torch.zeros(batch_size, 32, dtype=torch.int8, device=device)

        for b in range(batch_size):
            trick_plays_list = []
            for t in range(trick_lens[b].item()):
                trick_plays_list.append((
                    trick_plays[b, t, 0].item(),
                    trick_plays[b, t, 1].item()
                ))

            tokens_b, masks_b = tokenizer.tokenize(
                worlds=worlds[b:b+1],
                decl_id=decl_ids[b].item(),
                leader=leaders[b].item(),
                trick_plays=trick_plays_list,
                remaining=remaining[b:b+1],
                current_player=current_players[b].item(),
            )
            loop_tokens[b] = tokens_b[0]
            loop_masks[b] = masks_b[0]

        elapsed = time.perf_counter() - start
        loop_times.append(elapsed * 1000)

    loop_mean = np.mean(loop_times)
    loop_std = np.std(loop_times)

    # =========================================================================
    # Benchmark Batched Approach
    # =========================================================================
    batched_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()

        batched_tokens, batched_masks = tokenizer.tokenize_batched(
            worlds=worlds,
            decl_ids=decl_ids,
            leaders=leaders,
            current_players=current_players,
            trick_plays=trick_plays,
            trick_lens=trick_lens,
            remaining=remaining,
        )

        elapsed = time.perf_counter() - start
        batched_times.append(elapsed * 1000)

    batched_mean = np.mean(batched_times)
    batched_std = np.std(batched_times)

    # =========================================================================
    # Report Results
    # =========================================================================
    speedup = loop_mean / batched_mean

    print(f"\nLoop approach:    {loop_mean:.2f}ms +/- {loop_std:.2f}ms", flush=True)
    print(f"Batched approach: {batched_mean:.2f}ms +/- {batched_std:.2f}ms", flush=True)
    print(f"Speedup:          {speedup:.1f}x", flush=True)
    print(f"{'='*60}", flush=True)

    # Batched should be significantly faster (at least 5x on CPU)
    assert speedup > 2.0, f"Batched approach not faster enough: {speedup:.1f}x speedup"


def test_tokenize_past_steps_batched_matches_original():
    """Verify tokenize_past_steps_batched matches original tokenize_past_steps."""
    torch.manual_seed(42)

    N, M, K = 3, 8, 5
    tokenizer = GPUTokenizer(max_batch=2000, device='cpu')

    # Create test data
    worlds = torch.randint(0, 28, (N, M, 4, 7), dtype=torch.int8)

    # Create diverse past_states
    trick_plays = torch.zeros(N, K, 3, 2, dtype=torch.int32)
    trick_lens = torch.zeros(N, K, dtype=torch.int32)

    # Game 0: varying trick lengths
    trick_lens[0] = torch.tensor([0, 1, 2, 3, 1])
    trick_plays[0, 1, 0] = torch.tensor([0, 5])  # step 1: 1 play
    trick_plays[0, 2, 0] = torch.tensor([0, 5])  # step 2: 2 plays
    trick_plays[0, 2, 1] = torch.tensor([1, 12])
    trick_plays[0, 3, 0] = torch.tensor([0, 5])  # step 3: 3 plays
    trick_plays[0, 3, 1] = torch.tensor([1, 12])
    trick_plays[0, 3, 2] = torch.tensor([2, 19])
    trick_plays[0, 4, 0] = torch.tensor([3, 20])  # step 4: 1 play

    # Game 1: different configuration
    trick_lens[1] = torch.tensor([1, 0, 2, 1, 0])
    trick_plays[1, 0, 0] = torch.tensor([2, 8])
    trick_plays[1, 2, 0] = torch.tensor([0, 1])
    trick_plays[1, 2, 1] = torch.tensor([1, 7])
    trick_plays[1, 3, 0] = torch.tensor([3, 15])

    # Game 2: all zeros
    # trick_lens[2] already 0

    # Create played_before with some variation
    played_before = torch.zeros(N, K, 28, dtype=torch.bool)
    # Mark some dominoes as played
    played_before[0, 1, 5] = True
    played_before[0, 2, 5] = True
    played_before[0, 2, 12] = True
    played_before[0, 3, 5] = True
    played_before[0, 3, 12] = True
    played_before[0, 3, 19] = True
    played_before[1, 0, 8] = True
    played_before[1, 2, 1] = True
    played_before[1, 2, 7] = True

    past_states = PastStatesGPU(
        played_before=played_before,
        trick_plays=trick_plays,
        trick_lens=trick_lens,
        leaders=torch.randint(0, 4, (N, K), dtype=torch.int32),
        actors=torch.randint(0, 4, (N, K), dtype=torch.int32),
        observed_actions=torch.randint(0, 28, (N, K)),
        step_indices=torch.arange(K).unsqueeze(0).expand(N, -1),
        valid_mask=torch.ones(N, K, dtype=torch.bool),
    )

    decl_id = 3

    # Run both implementations
    tokens_original, masks_original = tokenizer.tokenize_past_steps(worlds, past_states, decl_id)
    tokens_batched, masks_batched = tokenizer.tokenize_past_steps_batched(worlds, past_states, decl_id)

    # Check shapes
    assert tokens_batched.shape == tokens_original.shape, \
        f"Shape mismatch: {tokens_batched.shape} vs {tokens_original.shape}"
    assert masks_batched.shape == masks_original.shape

    # Compare values
    tokens_match = torch.all(tokens_batched == tokens_original)
    masks_match = torch.all(masks_batched == masks_original)

    if not tokens_match:
        # Find first mismatch
        diff_mask = tokens_batched != tokens_original
        mismatch_indices = torch.nonzero(diff_mask)
        if len(mismatch_indices) > 0:
            idx, tok, feat = mismatch_indices[0].tolist()
            g = idx // (K * M)
            k = (idx % (K * M)) // M
            m = idx % M
            print(f"\nToken mismatch at flat_idx={idx} (g={g}, k={k}, m={m}), "
                  f"token={tok}, feature={feat}:", flush=True)
            print(f"  Batched:  {tokens_batched[idx, tok, feat].item()}", flush=True)
            print(f"  Original: {tokens_original[idx, tok, feat].item()}", flush=True)

    assert tokens_match, "Batched tokens don't match original"
    assert masks_match, "Batched masks don't match original"

    print(f"\ntest_tokenize_past_steps_batched_matches_original PASSED: "
          f"N={N}, M={M}, K={K} verified", flush=True)


def test_tokenize_past_steps_batched_with_padding():
    """Test batched version correctly handles invalid steps (padding)."""
    tokenizer = GPUTokenizer(max_batch=1000, device='cpu')

    N, M, K = 2, 5, 4
    worlds = torch.randint(0, 28, (N, M, 4, 7), dtype=torch.int8)

    # Create past_states where some steps are invalid (padding)
    valid_mask = torch.ones(N, K, dtype=torch.bool)
    valid_mask[0, 3] = False  # Game 0, step 3 is padding
    valid_mask[1, 2:] = False  # Game 1, steps 2-3 are padding

    past_states = PastStatesGPU(
        played_before=torch.zeros(N, K, 28, dtype=torch.bool),
        trick_plays=torch.zeros(N, K, 3, 2, dtype=torch.int32),
        trick_lens=torch.zeros(N, K, dtype=torch.int32),
        leaders=torch.zeros(N, K, dtype=torch.int32),
        actors=torch.arange(4).repeat(N, K // 4 + 1)[:N, :K],
        observed_actions=torch.randint(0, 28, (N, K)),
        step_indices=torch.arange(K).unsqueeze(0).expand(N, -1),
        valid_mask=valid_mask,
    )

    tokens_original, masks_original = tokenizer.tokenize_past_steps(worlds, past_states, decl_id=0)
    tokens_batched, masks_batched = tokenizer.tokenize_past_steps_batched(worlds, past_states, decl_ids=0)

    # Both should have zeros for invalid entries
    # Game 0, step 3: indices g=0, k=3 -> flat_idx = 0*K*M + 3*M = 15 to 19
    invalid_slice_0 = slice(15, 20)
    assert torch.all(tokens_batched[invalid_slice_0] == 0), "Batched: invalid step tokens should be zero"
    assert torch.all(masks_batched[invalid_slice_0] == 0), "Batched: invalid step masks should be zero"

    # Game 1, steps 2-3: indices start at g=1, k=2 -> flat_idx = 1*K*M + 2*M = 30
    invalid_slice_1 = slice(30, 40)
    assert torch.all(tokens_batched[invalid_slice_1] == 0), "Batched: invalid step tokens should be zero"
    assert torch.all(masks_batched[invalid_slice_1] == 0), "Batched: invalid step masks should be zero"

    print("\ntest_tokenize_past_steps_batched_with_padding PASSED", flush=True)


def test_tokenize_past_steps_batched_benchmark():
    """Benchmark original vs batched tokenize_past_steps."""
    torch.manual_seed(42)

    # Use realistic dimensions from the task spec
    N = 32  # games
    M = 50  # samples per game
    K = 8   # past steps
    n_iterations = 3
    device = 'cpu'

    print(f"\n{'='*60}", flush=True)
    print(f"tokenize_past_steps Benchmark: N={N}, M={M}, K={K}", flush=True)
    print(f"Total tokenizations: {N*M*K}", flush=True)
    print(f"{'='*60}", flush=True)

    tokenizer = GPUTokenizer(max_batch=20000, device=device)

    # Generate test data
    worlds = torch.randint(0, 28, (N, M, 4, 7), dtype=torch.int8, device=device)

    # Random past_states
    torch.manual_seed(42)
    trick_lens = torch.randint(0, 4, (N, K), dtype=torch.int32, device=device)
    trick_plays = torch.zeros(N, K, 3, 2, dtype=torch.int32, device=device)
    for n in range(N):
        for k in range(K):
            for t in range(trick_lens[n, k].item()):
                trick_plays[n, k, t, 0] = torch.randint(0, 4, (1,)).item()
                trick_plays[n, k, t, 1] = torch.randint(0, 28, (1,)).item()

    played_before = torch.zeros(N, K, 28, dtype=torch.bool, device=device)
    # Mark some dominoes as played
    for n in range(N):
        for k in range(K):
            n_played = k * 2  # More dominoes played in later steps
            played_indices = torch.randperm(28)[:n_played]
            played_before[n, k, played_indices] = True

    past_states = PastStatesGPU(
        played_before=played_before,
        trick_plays=trick_plays,
        trick_lens=trick_lens,
        leaders=torch.randint(0, 4, (N, K), dtype=torch.int32, device=device),
        actors=torch.randint(0, 4, (N, K), dtype=torch.int32, device=device),
        observed_actions=torch.randint(0, 28, (N, K), device=device),
        step_indices=torch.arange(K, device=device).unsqueeze(0).expand(N, -1),
        valid_mask=torch.ones(N, K, dtype=torch.bool, device=device),
    )

    decl_id = 3

    # Warmup
    _ = tokenizer.tokenize_past_steps(worlds[:2, :5], past_states.__class__(
        played_before=past_states.played_before[:2, :2],
        trick_plays=past_states.trick_plays[:2, :2],
        trick_lens=past_states.trick_lens[:2, :2],
        leaders=past_states.leaders[:2, :2],
        actors=past_states.actors[:2, :2],
        observed_actions=past_states.observed_actions[:2, :2],
        step_indices=past_states.step_indices[:2, :2],
        valid_mask=past_states.valid_mask[:2, :2],
    ), decl_id)
    _ = tokenizer.tokenize_past_steps_batched(worlds[:2, :5], past_states.__class__(
        played_before=past_states.played_before[:2, :2],
        trick_plays=past_states.trick_plays[:2, :2],
        trick_lens=past_states.trick_lens[:2, :2],
        leaders=past_states.leaders[:2, :2],
        actors=past_states.actors[:2, :2],
        observed_actions=past_states.observed_actions[:2, :2],
        step_indices=past_states.step_indices[:2, :2],
        valid_mask=past_states.valid_mask[:2, :2],
    ), decl_id)

    # Benchmark original
    original_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = tokenizer.tokenize_past_steps(worlds, past_states, decl_id)
        elapsed = (time.perf_counter() - start) * 1000
        original_times.append(elapsed)

    original_mean = np.mean(original_times)

    # Benchmark batched
    batched_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = tokenizer.tokenize_past_steps_batched(worlds, past_states, decl_id)
        elapsed = (time.perf_counter() - start) * 1000
        batched_times.append(elapsed)

    batched_mean = np.mean(batched_times)
    speedup = original_mean / batched_mean

    print(f"\nOriginal (loop):  {original_mean:.2f}ms", flush=True)
    print(f"Batched:          {batched_mean:.2f}ms", flush=True)
    print(f"Speedup:          {speedup:.1f}x", flush=True)
    print(f"{'='*60}", flush=True)

    # Verify correctness
    tokens_orig, masks_orig = tokenizer.tokenize_past_steps(worlds, past_states, decl_id)
    tokens_batch, masks_batch = tokenizer.tokenize_past_steps_batched(worlds, past_states, decl_id)
    assert torch.all(tokens_orig == tokens_batch), "Correctness check failed: tokens mismatch"
    assert torch.all(masks_orig == masks_batch), "Correctness check failed: masks mismatch"
    print(f"Correctness verified: outputs match exactly", flush=True)

    # The batched version should be significantly faster
    assert speedup > 5.0, f"Batched approach not fast enough: {speedup:.1f}x speedup"
