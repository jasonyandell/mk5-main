"""
Demo script for GPU tokenizer.

Shows usage patterns and performance characteristics.
"""
import time

import numpy as np
import torch

from forge.eq.tokenize_gpu import GPUTokenizer


def demo_basic_usage():
    """Demonstrate basic tokenizer usage."""
    print("=== Basic Usage Demo ===\n")

    # Create tokenizer with max batch size
    tokenizer = GPUTokenizer(max_batch=1000, device='cuda')
    print(f"Created GPU tokenizer with max_batch=1000 on {tokenizer.device}")

    # Generate sample worlds (4 hands of 7 dominoes)
    n_worlds = 100
    worlds = torch.randint(0, 28, (n_worlds, 4, 7), dtype=torch.int8, device='cuda')
    remaining = torch.ones(n_worlds, 4, dtype=torch.int64, device='cuda') * 0x7F  # All remaining

    print(f"\nTokenizing {n_worlds} worlds...")

    # Tokenize
    tokens, masks = tokenizer.tokenize(
        worlds=worlds,
        decl_id=3,
        leader=0,
        trick_plays=[(0, 5), (1, 12)],  # 2 plays in current trick
        remaining=remaining,
        current_player=2,
    )

    print(f"Output shapes:")
    print(f"  tokens: {tokens.shape} dtype={tokens.dtype}")
    print(f"  masks:  {masks.shape} dtype={masks.dtype}")

    # Show some token details
    print(f"\nToken 0 (context token) for world 0:")
    print(f"  Features: {tokens[0, 0].cpu().numpy()}")
    print(f"  Mask: {masks[0, 0].item()}")

    print(f"\nToken 1 (first hand token) for world 0:")
    print(f"  Features: {tokens[0, 1].cpu().numpy()}")
    print(f"  Mask: {masks[0, 1].item()}")

    print()


def demo_performance():
    """Benchmark tokenizer performance."""
    print("=== Performance Benchmark ===\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark")
        return

    batch_sizes = [100, 500, 1000, 1600, 2000]
    tokenizer = GPUTokenizer(max_batch=2500, device='cuda')

    print(f"{'Batch Size':<12} {'Mean (ms)':<12} {'Std (ms)':<12} {'Throughput (worlds/s)':<20}")
    print("-" * 60)

    for n_worlds in batch_sizes:
        # Generate data
        worlds = torch.randint(0, 28, (n_worlds, 4, 7), dtype=torch.int8, device='cuda')
        remaining = torch.ones(n_worlds, 4, dtype=torch.int64, device='cuda') * 0x7F

        # Warmup
        for _ in range(3):
            tokenizer.tokenize(
                worlds=worlds,
                decl_id=3,
                leader=0,
                trick_plays=[],
                remaining=remaining,
                current_player=0,
            )
            torch.cuda.synchronize()

        # Benchmark
        times = []
        n_runs = 20

        for _ in range(n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()

            tokenizer.tokenize(
                worlds=worlds,
                decl_id=3,
                leader=0,
                trick_plays=[(0, 5)],
                remaining=remaining,
                current_player=0,
            )

            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = (n_worlds / mean_time) * 1000  # worlds per second

        print(f"{n_worlds:<12} {mean_time:<12.2f} {std_time:<12.2f} {throughput:<20,.0f}")

    print()


def demo_correctness():
    """Verify correctness with CPU reference."""
    print("=== Correctness Check ===\n")

    from forge.eq.oracle import Stage1Oracle

    # Create deterministic test case
    torch.manual_seed(42)
    np.random.seed(42)

    n_worlds = 20
    decl_id = 5
    leader = 1
    current_player = 2
    trick_plays = [(1, 8), (2, 15)]

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
    print("Computing CPU reference (oracle._tokenize_worlds)...")
    oracle = Stage1Oracle.__new__(Stage1Oracle)
    start = time.perf_counter()
    tokens_cpu, masks_cpu = oracle._tokenize_worlds(
        worlds=worlds_list,
        decl_id=decl_id,
        leader=leader,
        trick_plays=trick_plays,
        remaining=remaining_np,
        current_player=current_player,
    )
    cpu_time = (time.perf_counter() - start) * 1000
    print(f"CPU time: {cpu_time:.2f}ms")

    # GPU implementation
    print("\nComputing GPU result (GPUTokenizer)...")
    tokenizer = GPUTokenizer(max_batch=100, device='cuda' if torch.cuda.is_available() else 'cpu')
    worlds_tensor = torch.tensor(worlds_list, dtype=torch.int8, device=tokenizer.device)
    remaining_tensor = torch.tensor(remaining_np, dtype=torch.int64, device=tokenizer.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    tokens_gpu, masks_gpu = tokenizer.tokenize(
        worlds=worlds_tensor,
        decl_id=decl_id,
        leader=leader,
        trick_plays=trick_plays,
        remaining=remaining_tensor,
        current_player=current_player,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_time = (time.perf_counter() - start) * 1000
    print(f"GPU time: {gpu_time:.2f}ms")

    # Compare
    tokens_gpu_np = tokens_gpu.cpu().numpy()
    masks_gpu_np = masks_gpu.cpu().numpy()

    tokens_match = np.array_equal(tokens_gpu_np, tokens_cpu)
    masks_match = np.array_equal(masks_gpu_np, masks_cpu)

    print(f"\nResults:")
    print(f"  Tokens match: {tokens_match}")
    print(f"  Masks match:  {masks_match}")
    if torch.cuda.is_available():
        speedup = cpu_time / gpu_time
        print(f"  Speedup: {speedup:.1f}x")

    if tokens_match and masks_match:
        print("\n✓ GPU tokenizer output exactly matches CPU reference")
    else:
        print("\n✗ Mismatch detected!")
        if not tokens_match:
            diff_count = np.sum(tokens_gpu_np != tokens_cpu)
            print(f"  Token differences: {diff_count}/{tokens_cpu.size}")
        if not masks_match:
            diff_count = np.sum(masks_gpu_np != masks_cpu)
            print(f"  Mask differences: {diff_count}/{masks_cpu.size}")

    print()


def demo_feature_extraction():
    """Show how to extract features from tokens."""
    print("=== Feature Extraction Demo ===\n")

    tokenizer = GPUTokenizer(max_batch=10, device='cpu')

    # Create a simple world
    worlds = torch.tensor([
        [
            [0, 1, 2, 3, 4, 5, 6],      # Player 0
            [7, 8, 9, 10, 11, 12, 13],  # Player 1
            [14, 15, 16, 17, 18, 19, 20],  # Player 2
            [21, 22, 23, 24, 25, 26, 27],  # Player 3
        ]
    ], dtype=torch.int8)
    remaining = torch.ones(1, 4, dtype=torch.int64) * 0x7F

    tokens, masks = tokenizer.tokenize(
        worlds=worlds,
        decl_id=3,
        leader=0,
        trick_plays=[],
        remaining=remaining,
        current_player=0,
    )

    print("Token format (12 features per token):")
    print("  [0] high pip")
    print("  [1] low pip")
    print("  [2] is_double")
    print("  [3] count_value (0=0pts, 1=5pts, 2=10pts)")
    print("  [4] trump_rank")
    print("  [5] normalized_player")
    print("  [6] is_current")
    print("  [7] is_partner")
    print("  [8] remaining")
    print("  [9] token_type")
    print("  [10] decl_id")
    print("  [11] normalized_leader")

    print("\nContext token (index 0):")
    ctx = tokens[0, 0].cpu().numpy()
    print(f"  token_type={ctx[9]}, decl_id={ctx[10]}, normalized_leader={ctx[11]}")

    print("\nFirst hand token (player 0, domino 0):")
    hand_token = tokens[0, 1].cpu().numpy()
    print(f"  Domino features: high={hand_token[0]}, low={hand_token[1]}, double={hand_token[2]}")
    print(f"  Value: count_value={hand_token[3]} (0=0pts, 1=5pts, 2=10pts)")
    print(f"  Trump rank: {hand_token[4]}")
    print(f"  Player: normalized={hand_token[5]}, is_current={hand_token[6]}, is_partner={hand_token[7]}")
    print(f"  Remaining: {hand_token[8]}")

    print()


if __name__ == "__main__":
    demo_basic_usage()
    demo_performance()
    demo_correctness()
    demo_feature_extraction()
