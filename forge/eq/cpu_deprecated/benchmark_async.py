from __future__ import annotations
"""
======================================================================
DEPRECATED CPU PIPELINE - DO NOT USE
======================================================================
This module contains KNOWN BUGS (E[Q] collapse with high sample counts).
It is kept temporarily for reference only and will be deleted soon.

Use the GPU pipeline instead: forge/eq/generate_gpu.py
======================================================================
"""
import sys as _sys
if not _sys.flags.interactive:  # Allow interactive inspection
    raise RuntimeError(
        "\n" + "=" * 70 + "\n"
        "DEPRECATED CPU PIPELINE - DO NOT USE\n"
        + "=" * 70 + "\n"
        "This module contains KNOWN BUGS (E[Q] collapse with high sample counts).\n"
        "It is kept temporarily for reference only and will be deleted soon.\n"
        "\n"
        "Use the GPU pipeline instead: forge/eq/generate_gpu.py\n"
        + "=" * 70
    )
del _sys

"""Benchmark async vs sync oracle modes.

This script measures the performance improvement from Phase 4 async pipeline
with CUDA streams and non-blocking H2D transfers.

Usage:
    python -m forge.eq.benchmark_async --checkpoint path/to/checkpoint.ckpt
"""


import argparse
import time
from pathlib import Path

import numpy as np
import torch

from forge.eq.oracle import Stage1Oracle
from forge.oracle.rng import deal_from_seed


def benchmark_mode(
    oracle: Stage1Oracle,
    n_worlds: int,
    n_iterations: int,
    decl_id: int = 3,
) -> dict[str, float]:
    """Benchmark oracle performance for a given configuration.

    Args:
        oracle: Oracle instance to benchmark
        n_worlds: Number of worlds to sample per query
        n_iterations: Number of queries to perform
        decl_id: Declaration ID to use

    Returns:
        Dict with timing statistics
    """
    # Warmup: run once to compile CUDA graphs and allocate buffers
    worlds_warmup = [deal_from_seed(i) for i in range(n_worlds)]
    remaining = np.ones((n_worlds, 4), dtype=np.int64) * 0x7F
    game_state_info = {
        'decl_id': decl_id,
        'leader': 0,
        'trick_plays': [],
        'remaining': remaining,
    }

    _ = oracle.query_batch(worlds_warmup, game_state_info, current_player=0)

    # Ensure GPU is idle before benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark loop
    timings = []
    for i in range(n_iterations):
        worlds = [deal_from_seed(i * n_worlds + j) for j in range(n_worlds)]
        remaining = np.ones((n_worlds, 4), dtype=np.int64) * 0x7F
        game_state_info = {
            'decl_id': decl_id,
            'leader': 0,
            'trick_plays': [],
            'remaining': remaining,
        }

        start = time.perf_counter()
        _ = oracle.query_batch(worlds, game_state_info, current_player=0)

        # Force sync to measure end-to-end latency
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    timings_arr = np.array(timings)
    return {
        'mean_ms': timings_arr.mean() * 1000,
        'std_ms': timings_arr.std() * 1000,
        'median_ms': np.median(timings_arr) * 1000,
        'min_ms': timings_arr.min() * 1000,
        'max_ms': timings_arr.max() * 1000,
        'throughput_queries_per_sec': 1.0 / timings_arr.mean(),
        'throughput_worlds_per_sec': n_worlds / timings_arr.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark async vs sync oracle modes')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--n-worlds', type=int, default=100, help='Number of worlds per query')
    parser.add_argument('--n-iterations', type=int, default=50, help='Number of queries to benchmark')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--no-compile', action='store_true', help='Disable torch.compile')

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Error: CUDA not available")
        return 1

    print(f"Benchmarking oracle performance")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Worlds per query: {args.n_worlds}")
    print(f"Iterations: {args.n_iterations}")
    print(f"Compile: {not args.no_compile}")
    print()

    # Benchmark sync mode
    print("=== Synchronous Mode ===")
    oracle_sync = Stage1Oracle(
        args.checkpoint,
        device=args.device,
        compile=not args.no_compile,
        use_async=False
    )
    stats_sync = benchmark_mode(oracle_sync, args.n_worlds, args.n_iterations)

    print(f"Mean latency:    {stats_sync['mean_ms']:.2f} ± {stats_sync['std_ms']:.2f} ms")
    print(f"Median latency:  {stats_sync['median_ms']:.2f} ms")
    print(f"Min latency:     {stats_sync['min_ms']:.2f} ms")
    print(f"Max latency:     {stats_sync['max_ms']:.2f} ms")
    print(f"Throughput:      {stats_sync['throughput_queries_per_sec']:.1f} queries/s")
    print(f"                 {stats_sync['throughput_worlds_per_sec']:.0f} worlds/s")
    print()

    # Benchmark async mode (only on CUDA)
    if args.device == 'cuda':
        print("=== Asynchronous Mode (CUDA Streams) ===")
        oracle_async = Stage1Oracle(
            args.checkpoint,
            device=args.device,
            compile=not args.no_compile,
            use_async=True
        )
        stats_async = benchmark_mode(oracle_async, args.n_worlds, args.n_iterations)

        print(f"Mean latency:    {stats_async['mean_ms']:.2f} ± {stats_async['std_ms']:.2f} ms")
        print(f"Median latency:  {stats_async['median_ms']:.2f} ms")
        print(f"Min latency:     {stats_async['min_ms']:.2f} ms")
        print(f"Max latency:     {stats_async['max_ms']:.2f} ms")
        print(f"Throughput:      {stats_async['throughput_queries_per_sec']:.1f} queries/s")
        print(f"                 {stats_async['throughput_worlds_per_sec']:.0f} worlds/s")
        print()

        # Compute speedup
        speedup = stats_sync['mean_ms'] / stats_async['mean_ms']
        print(f"=== Speedup ===")
        print(f"Async vs Sync:   {speedup:.2f}x faster")
        print(f"Latency reduced: {stats_sync['mean_ms'] - stats_async['mean_ms']:.2f} ms")
        print(f"                 ({(1 - 1/speedup) * 100:.1f}% reduction)")
    else:
        print("(Async mode only available on CUDA)")

    return 0


if __name__ == '__main__':
    exit(main())
