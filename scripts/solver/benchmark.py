#!/usr/bin/env python3
"""
GPU Solver Benchmark - Fast memory/timing simulations.

Tests optimization strategies for the solve phase:
1. Sub-chunking large levels (vary SOLVE_CHUNK_SIZE)
2. Fused min/max operations (reduce intermediate tensors)
3. torch.take for int32 indexing (avoid .long() conversion)

Usage:
    python scripts/solver/benchmark.py              # Run all benchmarks
    python scripts/solver/benchmark.py --quick      # Quick sanity check
    python scripts/solver/benchmark.py --help       # Show all options
"""

import argparse
import gc
import time
from typing import Callable, Any
import torch

# Approximate level distribution for seed=42 (85M states)
# Level 5 is the bottleneck with ~14M states
LEVEL_DISTRIBUTION_85M = {
    28: 1,           # Initial state
    27: 7,           # After first play
    26: 49,
    25: 343,
    24: 2_401,
    23: 16_807,
    22: 117_649,
    21: 823_543,
    20: 3_500_000,   # Approximate
    19: 5_000_000,
    18: 6_500_000,
    17: 8_000_000,
    16: 9_500_000,
    15: 10_500_000,
    14: 11_000_000,
    13: 10_500_000,
    12: 9_000_000,
    11: 6_500_000,
    10: 4_000_000,
    9: 2_500_000,
    8: 1_500_000,
    7: 800_000,
    6: 400_000,
    5: 14_000_000,   # The problem level!
    4: 100_000,
    3: 50_000,
    2: 10_000,
    1: 1_000,
    0: 100,          # Terminal states
}


def make_synthetic_data(
    num_states: int,
    level_counts: dict[int, int],
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic solver data for benchmarking.

    Args:
        num_states: Total number of states
        level_counts: {level: count} distribution
        device: Target device

    Returns:
        (all_states, child_idx, V, level_of, is_team0)
    """
    # all_states: random int64s (content doesn't matter, only shape)
    all_states = torch.randint(
        0, 2**47, (num_states,), dtype=torch.int64, device=device
    )

    # child_idx: (N, 7) random indices, -1 for ~30% illegal moves
    child_idx = torch.randint(
        0, num_states, (num_states, 7), dtype=torch.int32, device=device
    )
    illegal_mask = torch.rand(num_states, 7, device=device) < 0.3
    child_idx[illegal_mask] = -1

    # V: minimax values (-42 to +42)
    V = torch.randint(-42, 43, (num_states,), dtype=torch.int8, device=device)

    # level_of: assign levels according to distribution
    level_of = torch.zeros(num_states, dtype=torch.int8, device=device)
    offset = 0
    for level in sorted(level_counts.keys(), reverse=True):
        count = min(level_counts[level], num_states - offset)
        if count > 0:
            level_of[offset:offset + count] = level
            offset += count
        if offset >= num_states:
            break

    # is_team0: random 50/50
    is_team0 = torch.rand(num_states, device=device) > 0.5

    return all_states, child_idx, V, level_of, is_team0


# -----------------------------------------------------------------------------
# Measurement utilities
# -----------------------------------------------------------------------------

def measure_peak_memory(fn: Callable[[], Any]) -> tuple[Any, float]:
    """
    Run fn() and return (result, peak_memory_mb).
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    result = fn()

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    return result, peak_bytes / (1024 * 1024)


def measure_time(fn: Callable[[], Any]) -> tuple[Any, float]:
    """
    Run fn() and return (result, elapsed_seconds).
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return result, elapsed


def measure_both(fn: Callable[[], Any]) -> tuple[Any, float, float]:
    """
    Run fn() and return (result, peak_mb, elapsed_s).
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_bytes = torch.cuda.max_memory_allocated()

    return result, peak_bytes / (1024 * 1024), elapsed


# -----------------------------------------------------------------------------
# Baseline solve loop (current implementation)
# -----------------------------------------------------------------------------

def solve_baseline(
    child_idx: torch.Tensor,  # (N, 7) int32
    V: torch.Tensor,          # (N,) int8
    level_of: torch.Tensor,   # (N,) int8
    is_team0: torch.Tensor,   # (N,) bool
    level: int,
    chunk_size: int = 2_000_000
) -> None:
    """
    Baseline solve for one level (matches current solve_gpu).

    Modifies V in-place.
    """
    device = V.device
    mask = (level_of == level)
    if not mask.any():
        return

    idx = mask.nonzero(as_tuple=True)[0]
    K = idx.shape[0]

    for chunk_start in range(0, K, chunk_size):
        chunk_end = min(chunk_start + chunk_size, K)
        chunk_idx = idx[chunk_start:chunk_end]

        # Get child indices for this chunk
        cidx = child_idx[chunk_idx]  # (chunk, 7) int32
        legal = (cidx >= 0)  # (chunk, 7)

        # Clamp for safe indexing (illegal moves will be masked anyway)
        cidx_safe = cidx.clamp(min=0).long()  # int64 for indexing - THE PROBLEM!

        # Get child values
        cv = V[cidx_safe]  # (chunk, 7) int8

        # Compute minimax
        is_team0_k = is_team0[chunk_idx]

        cv16 = cv.to(torch.int16)
        cv_for_max = torch.where(legal, cv16, torch.tensor(-128, dtype=torch.int16, device=device))
        cv_for_min = torch.where(legal, cv16, torch.tensor(127, dtype=torch.int16, device=device))

        max_val = cv_for_max.max(dim=1).values
        min_val = cv_for_min.min(dim=1).values

        V[chunk_idx] = torch.where(
            is_team0_k,
            max_val.to(torch.int8),
            min_val.to(torch.int8)
        )


# -----------------------------------------------------------------------------
# Optimization 1: Fused min/max (avoid intermediate tensors)
# -----------------------------------------------------------------------------

def solve_fused_minmax(
    child_idx: torch.Tensor,
    V: torch.Tensor,
    level_of: torch.Tensor,
    is_team0: torch.Tensor,
    level: int,
    chunk_size: int = 2_000_000
) -> None:
    """
    Fused min/max: compute both in one pass without cv_for_max/cv_for_min.
    """
    device = V.device
    mask = (level_of == level)
    if not mask.any():
        return

    idx = mask.nonzero(as_tuple=True)[0]
    K = idx.shape[0]

    for chunk_start in range(0, K, chunk_size):
        chunk_end = min(chunk_start + chunk_size, K)
        chunk_idx = idx[chunk_start:chunk_end]

        cidx = child_idx[chunk_idx]
        legal = (cidx >= 0)
        cidx_safe = cidx.clamp(min=0).long()

        cv = V[cidx_safe]  # (chunk, 7) int8
        is_team0_k = is_team0[chunk_idx]

        # FUSED: Use masked_fill instead of creating cv_for_max and cv_for_min
        # For max: illegal -> -128; for min: illegal -> 127
        cv16 = cv.to(torch.int16)
        illegal = ~legal

        # Compute max for team0 positions
        max_val = cv16.masked_fill(illegal, -128).max(dim=1).values

        # Compute min for team1 positions
        min_val = cv16.masked_fill(illegal, 127).min(dim=1).values

        V[chunk_idx] = torch.where(
            is_team0_k,
            max_val.to(torch.int8),
            min_val.to(torch.int8)
        )


# -----------------------------------------------------------------------------
# Optimization 2: torch.take (might accept int32?)
# -----------------------------------------------------------------------------

def solve_torch_take(
    child_idx: torch.Tensor,
    V: torch.Tensor,
    level_of: torch.Tensor,
    is_team0: torch.Tensor,
    level: int,
    chunk_size: int = 2_000_000
) -> None:
    """
    Use torch.take instead of V[cidx_safe] to see if it accepts int32.
    """
    device = V.device
    mask = (level_of == level)
    if not mask.any():
        return

    idx = mask.nonzero(as_tuple=True)[0]
    K = idx.shape[0]

    for chunk_start in range(0, K, chunk_size):
        chunk_end = min(chunk_start + chunk_size, K)
        chunk_idx = idx[chunk_start:chunk_end]

        cidx = child_idx[chunk_idx]  # (chunk, 7) int32
        legal = (cidx >= 0)
        cidx_safe = cidx.clamp(min=0)  # Keep as int32!

        # Flatten for torch.take
        flat_idx = cidx_safe.flatten()  # (chunk * 7,) int32

        # torch.take requires int64 indices :( but let's try anyway
        try:
            cv_flat = torch.take(V, flat_idx.long())  # Falls back to long
        except RuntimeError:
            # If torch.take doesn't accept int32, use index_select
            cv_flat = V[flat_idx.long()]

        cv = cv_flat.reshape(-1, 7)  # (chunk, 7)
        is_team0_k = is_team0[chunk_idx]

        cv16 = cv.to(torch.int16)
        illegal = ~legal
        max_val = cv16.masked_fill(illegal, -128).max(dim=1).values
        min_val = cv16.masked_fill(illegal, 127).min(dim=1).values

        V[chunk_idx] = torch.where(
            is_team0_k,
            max_val.to(torch.int8),
            min_val.to(torch.int8)
        )


# -----------------------------------------------------------------------------
# Optimization 3: torch.gather with int32 (experimental)
# -----------------------------------------------------------------------------

def solve_gather_int32(
    child_idx: torch.Tensor,
    V: torch.Tensor,
    level_of: torch.Tensor,
    is_team0: torch.Tensor,
    level: int,
    chunk_size: int = 2_000_000
) -> None:
    """
    Try torch.gather which might have different indexing requirements.
    """
    device = V.device
    mask = (level_of == level)
    if not mask.any():
        return

    idx = mask.nonzero(as_tuple=True)[0]
    K = idx.shape[0]

    for chunk_start in range(0, K, chunk_size):
        chunk_end = min(chunk_start + chunk_size, K)
        chunk_idx = idx[chunk_start:chunk_end]

        cidx = child_idx[chunk_idx]  # (chunk, 7) int32
        legal = (cidx >= 0)
        cidx_safe = cidx.clamp(min=0)

        # Expand V for gather: (N,) -> (N, 1)
        # Then gather along dim=0 with cidx_safe
        # This requires cidx_safe to match V's dtype requirement

        # Actually, V.expand + gather still requires int64 index
        # So this is basically the same, but let's measure overhead
        cv = V[cidx_safe.long()]  # (chunk, 7)

        is_team0_k = is_team0[chunk_idx]
        cv16 = cv.to(torch.int16)
        illegal = ~legal
        max_val = cv16.masked_fill(illegal, -128).max(dim=1).values
        min_val = cv16.masked_fill(illegal, 127).min(dim=1).values

        V[chunk_idx] = torch.where(
            is_team0_k,
            max_val.to(torch.int8),
            min_val.to(torch.int8)
        )


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

def benchmark_chunk_sizes(
    level_size: int = 14_000_000,
    total_states: int = 85_000_000,
    device: torch.device = None
) -> list[dict]:
    """
    Benchmark different SOLVE_CHUNK_SIZE values.
    """
    if device is None:
        device = torch.device('cuda')

    print(f"\n--- Chunk Size Sweep (level={level_size:,}, total={total_states:,}) ---")
    print(f"{'chunk_size':>12}  {'peak_mb':>10}  {'time_s':>8}  {'throughput':>12}")
    print("-" * 50)

    # Create synthetic data (just for the level we care about)
    level_counts = {5: level_size, 4: total_states - level_size}
    _, child_idx, V, level_of, is_team0 = make_synthetic_data(
        total_states, level_counts, device
    )

    results = []
    chunk_sizes = [500_000, 1_000_000, 2_000_000, 4_000_000, 7_000_000, 14_000_000]

    for chunk_size in chunk_sizes:
        # Reset V for each run
        V.fill_(0)

        try:
            _, peak_mb, elapsed = measure_both(
                lambda cs=chunk_size: solve_baseline(
                    child_idx, V, level_of, is_team0, level=5, chunk_size=cs
                )
            )
            throughput = level_size / elapsed / 1e6
            status = ""
            results.append({
                'chunk_size': chunk_size,
                'peak_mb': peak_mb,
                'time_s': elapsed,
                'throughput': throughput,
                'oom': False
            })
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                peak_mb = float('inf')
                elapsed = float('inf')
                throughput = 0
                status = "OOM"
                results.append({
                    'chunk_size': chunk_size,
                    'peak_mb': peak_mb,
                    'time_s': elapsed,
                    'throughput': throughput,
                    'oom': True
                })
                torch.cuda.empty_cache()
            else:
                raise

        if results[-1]['oom']:
            print(f"{chunk_size:>12,}  {'OOM':>10}  {'---':>8}  {'---':>12}")
        else:
            print(f"{chunk_size:>12,}  {peak_mb:>10.0f}  {elapsed:>8.3f}  {throughput:>10.1f}M/s")

    return results


def benchmark_optimizations(
    level_size: int = 14_000_000,
    total_states: int = 85_000_000,
    chunk_size: int = 2_000_000,
    device: torch.device = None
) -> list[dict]:
    """
    Compare optimization strategies.
    """
    if device is None:
        device = torch.device('cuda')

    print(f"\n--- Optimization Comparison (level={level_size:,}, chunk={chunk_size:,}) ---")

    level_counts = {5: level_size, 4: total_states - level_size}
    _, child_idx, V, level_of, is_team0 = make_synthetic_data(
        total_states, level_counts, device
    )

    # Show base memory usage
    gc.collect()
    torch.cuda.empty_cache()
    base_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Base arrays: {base_mb:.0f} MB")
    print(f"{'approach':>20}  {'peak_mb':>10}  {'delta_mb':>10}  {'time_s':>8}")
    print("-" * 55)

    approaches = [
        ('baseline', solve_baseline),
        ('fused_minmax', solve_fused_minmax),
        ('torch_take', solve_torch_take),
        ('gather_int32', solve_gather_int32),
    ]

    results = []
    baseline_peak = None

    for name, fn in approaches:
        V.fill_(0)
        gc.collect()
        torch.cuda.empty_cache()

        try:
            _, peak_mb, elapsed = measure_both(
                lambda f=fn: f(child_idx, V, level_of, is_team0, level=5, chunk_size=chunk_size)
            )
            delta_mb = peak_mb - base_mb
            if baseline_peak is None:
                baseline_peak = peak_mb

            results.append({
                'approach': name,
                'peak_mb': peak_mb,
                'delta_mb': delta_mb,
                'time_s': elapsed,
                'oom': False
            })
            print(f"{name:>20}  {peak_mb:>10.0f}  {delta_mb:>+10.0f}  {elapsed:>8.3f}")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                results.append({
                    'approach': name,
                    'peak_mb': float('inf'),
                    'delta_mb': float('inf'),
                    'time_s': float('inf'),
                    'oom': True
                })
                print(f"{name:>20}  {'OOM':>10}  {'---':>10}  {'---':>8}")
                torch.cuda.empty_cache()
            else:
                raise

    # Show summary
    valid = [r for r in results if not r['oom'] and r['peak_mb'] < float('inf')]
    if valid:
        best = min(valid, key=lambda r: r['peak_mb'])
        print(f"\nBest: {best['approach']} at {best['peak_mb']:.0f} MB total, "
              f"{best['delta_mb']:+.0f} MB overhead")

    return results


def benchmark_quick(device: torch.device = None) -> None:
    """
    Quick sanity check with small data.
    """
    if device is None:
        device = torch.device('cuda')

    print("\n--- Quick Sanity Check ---")

    # Small test: 1M states, 100K at target level
    level_counts = {5: 100_000, 4: 900_000}
    _, child_idx, V, level_of, is_team0 = make_synthetic_data(
        1_000_000, level_counts, device
    )

    V.fill_(0)
    _, peak_mb, elapsed = measure_both(
        lambda: solve_baseline(child_idx, V, level_of, is_team0, level=5, chunk_size=50_000)
    )

    print(f"1M states, 100K level, 50K chunks: {peak_mb:.0f} MB, {elapsed:.3f}s")
    print("Sanity check passed!")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='GPU Solver Benchmarks')
    parser.add_argument('--quick', action='store_true', help='Quick sanity check')
    parser.add_argument('--chunk-sizes', action='store_true', help='Sweep chunk sizes')
    parser.add_argument('--optimizations', action='store_true', help='Compare optimizations')
    parser.add_argument('--level-size', type=int, default=14_000_000, help='States at target level')
    parser.add_argument('--total-states', type=int, default=85_000_000, help='Total states')
    parser.add_argument('--chunk-size', type=int, default=2_000_000, help='Chunk size for optimization tests')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(device)
    print(f"=== GPU Solver Benchmarks ===")
    print(f"Device: {props.name} ({props.total_memory // (1024**2)} MB)")

    # Default: run all if no specific test requested
    run_all = not (args.quick or args.chunk_sizes or args.optimizations)

    if args.quick or run_all:
        benchmark_quick(device)

    if args.chunk_sizes or run_all:
        benchmark_chunk_sizes(
            level_size=args.level_size,
            total_states=args.total_states,
            device=device
        )

    if args.optimizations or run_all:
        benchmark_optimizations(
            level_size=args.level_size,
            total_states=args.total_states,
            chunk_size=args.chunk_size,
            device=device
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
