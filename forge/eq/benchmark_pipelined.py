"""Benchmark pipelined vs batched E[Q] generation to measure CPU/GPU overlap.

This script measures the throughput improvement from pipelining:
- Baseline (batched): Sequential CPU sampling + GPU query per decision round
- Pipelined: Overlapped CPU sampling (workers) + GPU query

Expected: ~1.8x improvement (from 22.2ms → 12.2ms per decision, GPU-bound)
"""

from __future__ import annotations

import time

import numpy as np

from forge.eq.generate_batched import generate_eq_games_batched
from forge.eq.generate_pipelined import generate_eq_games_pipelined
from forge.eq.oracle import Stage1Oracle
from forge.oracle.rng import deal_from_seed


def benchmark_generator(
    name: str,
    generator_fn,
    oracle: Stage1Oracle,
    n_games: int = 8,
    n_samples: int = 100,
    n_warmup: int = 2,
    n_runs: int = 5,
):
    """Benchmark a game generator."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    print(f"Config: {n_games} games, {n_samples} samples, {n_runs} runs")

    # Prepare data
    hands_list = [deal_from_seed(1000 + i) for i in range(n_games)]
    decl_ids = [i % 10 for i in range(n_games)]
    world_rngs = [np.random.default_rng(2000 + i) for i in range(n_games)]

    # Warmup
    print(f"Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        generator_fn(
            oracle=oracle,
            hands_list=hands_list,
            decl_ids=decl_ids,
            n_samples=n_samples,
            world_rngs=[np.random.default_rng(2000 + i) for i in range(n_games)],
        )

    # Benchmark
    print(f"Running benchmark ({n_runs} runs)...")
    times = []
    for run in range(n_runs):
        start = time.perf_counter()
        records = generator_fn(
            oracle=oracle,
            hands_list=hands_list,
            decl_ids=decl_ids,
            n_samples=n_samples,
            world_rngs=[np.random.default_rng(2000 + i) for i in range(n_games)],
        )
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s ({n_games/elapsed:.2f} games/s)")

    # Stats
    mean_time = np.mean(times)
    std_time = np.std(times)
    games_per_sec = n_games / mean_time
    decisions_per_sec = (n_games * 28) / mean_time
    ms_per_decision = (mean_time * 1000) / (n_games * 28)

    print(f"\nResults:")
    print(f"  Mean time:      {mean_time:.3f} ± {std_time:.3f}s")
    print(f"  Throughput:     {games_per_sec:.2f} games/s")
    print(f"  Throughput:     {decisions_per_sec:.1f} decisions/s")
    print(f"  Per-decision:   {ms_per_decision:.1f}ms")

    return {
        "name": name,
        "mean_time": mean_time,
        "std_time": std_time,
        "games_per_sec": games_per_sec,
        "ms_per_decision": ms_per_decision,
    }


def main():
    """Run benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark pipelined generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to oracle checkpoint (required)",
    )
    parser.add_argument("--n-games", type=int, default=8, help="Number of games")
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Samples per decision"
    )
    parser.add_argument("--n-runs", type=int, default=5, help="Benchmark runs")
    parser.add_argument(
        "--n-workers", type=int, default=2, help="CPU workers for pipelined"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    if args.checkpoint is None:
        print("ERROR: --checkpoint is required")
        print("\nUsage:")
        print(
            "  python forge/eq/benchmark_pipelined.py --checkpoint path/to/oracle.ckpt"
        )
        print("\nExample:")
        print(
            "  python forge/eq/benchmark_pipelined.py --checkpoint forge/catalog/stage1/qval-v2.ckpt"
        )
        return 1

    print(f"Loading oracle from {args.checkpoint}...")
    oracle = Stage1Oracle(args.checkpoint, device=args.device)
    print(f"Oracle loaded (device={oracle.device})")

    # Benchmark batched (baseline)
    results_batched = benchmark_generator(
        name="Batched (baseline)",
        generator_fn=generate_eq_games_batched,
        oracle=oracle,
        n_games=args.n_games,
        n_samples=args.n_samples,
        n_runs=args.n_runs,
    )

    # Benchmark pipelined with different worker counts
    results_pipelined = []
    for n_workers in [1, 2, 4]:
        def pipelined_fn(**kwargs):
            return generate_eq_games_pipelined(**kwargs, n_workers=n_workers)

        result = benchmark_generator(
            name=f"Pipelined (n_workers={n_workers})",
            generator_fn=pipelined_fn,
            oracle=oracle,
            n_games=args.n_games,
            n_samples=args.n_samples,
            n_runs=args.n_runs,
        )
        results_pipelined.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Games/s':>12} {'ms/decision':>12} {'Speedup':>8}")
    print("-" * 60)

    baseline_throughput = results_batched["games_per_sec"]
    print(
        f"{results_batched['name']:<30} "
        f"{results_batched['games_per_sec']:>12.2f} "
        f"{results_batched['ms_per_decision']:>12.1f} "
        f"{'1.00x':>8}"
    )

    for result in results_pipelined:
        speedup = result["games_per_sec"] / baseline_throughput
        print(
            f"{result['name']:<30} "
            f"{result['games_per_sec']:>12.2f} "
            f"{result['ms_per_decision']:>12.1f} "
            f"{speedup:>8.2f}x"
        )

    print("\nTarget: 1.8x speedup (from CPU/GPU overlap)")
    print(
        f"Best achieved: {max(r['games_per_sec'] for r in results_pipelined) / baseline_throughput:.2f}x"
    )

    return 0


if __name__ == "__main__":
    exit(main())
