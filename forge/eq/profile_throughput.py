#!/usr/bin/env python3
"""Profile E[Q] generator throughput and GPU utilization.

This script profiles the E[Q] generation pipeline to identify bottlenecks
and measure GPU saturation. Used for t42-z4yj optimization work.

Usage:
    # Quick profiling (5 games, n_samples=100)
    python -m forge.eq.profile_throughput

    # Batch size sweep
    python -m forge.eq.profile_throughput --sweep

    # Specific batch size with detailed trace
    python -m forge.eq.profile_throughput --n-samples 256 --trace

    # Full profiling with torch.profiler trace
    python -m forge.eq.profile_throughput --n-games 10 --n-samples 128 --trace
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from forge.eq.oracle import Stage1Oracle
from forge.eq.generate_game import generate_eq_game
from forge.eq.types import PosteriorConfig
from forge.oracle.rng import deal_from_seed


@dataclass
class ProfileResult:
    """Result of a single profiling run."""
    n_games: int
    n_samples: int  # worlds per decision
    total_time_s: float
    games_per_sec: float
    decisions_per_sec: float
    avg_game_time_s: float
    peak_memory_mb: float
    gpu_name: str
    # Torch profiler stats (if available)
    cuda_time_ms: float | None = None
    cpu_time_ms: float | None = None
    self_cuda_time_ms: float | None = None


def log(msg: str, *, flush: bool = True) -> None:
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=flush)


def profile_games(
    oracle: Stage1Oracle,
    n_games: int,
    n_samples: int,
    seed: int = 42,
    warmup_games: int = 2,
    posterior: bool = False,
) -> ProfileResult:
    """Profile E[Q] generation for n_games."""
    rng = np.random.default_rng(seed)

    posterior_config = None
    if posterior:
        posterior_config = PosteriorConfig(
            enabled=True,
            tau=10.0,
            beta=0.10,
            window_k=8,
        )

    # Warmup
    log(f"Warming up ({warmup_games} games)...")
    for _ in range(warmup_games):
        game_seed = int(rng.integers(0, 2**31))
        hands = deal_from_seed(game_seed)
        decl_id = int(rng.integers(0, 10))
        world_rng = np.random.default_rng(game_seed)
        generate_eq_game(
            oracle, hands, decl_id,
            n_samples=n_samples,
            posterior_config=posterior_config,
            world_rng=world_rng,
        )

    # Clear memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    gc.collect()

    # Timed run
    log(f"Profiling {n_games} games with n_samples={n_samples}...")
    start = time.perf_counter()

    for i in range(n_games):
        game_seed = int(rng.integers(0, 2**31))
        hands = deal_from_seed(game_seed)
        decl_id = int(rng.integers(0, 10))
        world_rng = np.random.default_rng(game_seed)
        generate_eq_game(
            oracle, hands, decl_id,
            n_samples=n_samples,
            posterior_config=posterior_config,
            world_rng=world_rng,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Collect stats
    peak_memory_mb = 0.0
    gpu_name = "N/A"
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        gpu_name = torch.cuda.get_device_name(0)

    n_decisions = n_games * 28
    return ProfileResult(
        n_games=n_games,
        n_samples=n_samples,
        total_time_s=elapsed,
        games_per_sec=n_games / elapsed,
        decisions_per_sec=n_decisions / elapsed,
        avg_game_time_s=elapsed / n_games,
        peak_memory_mb=peak_memory_mb,
        gpu_name=gpu_name,
    )


def profile_with_torch_profiler(
    oracle: Stage1Oracle,
    n_games: int,
    n_samples: int,
    seed: int = 42,
    output_dir: str = "scratch/profiles",
    posterior: bool = False,
    warmup_games: int = 2,
) -> tuple[ProfileResult, Path]:
    """Profile with torch.profiler for detailed GPU analysis.

    Uses the simple context manager pattern (no schedule/step) which
    works reliably in WSL2 where CUPTI has issues.
    """
    from torch.profiler import profile, ProfilerActivity

    rng = np.random.default_rng(seed)

    posterior_config = None
    if posterior:
        posterior_config = PosteriorConfig(
            enabled=True,
            tau=10.0,
            beta=0.10,
            window_k=8,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create trace name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_name = f"eq_profile_{n_samples}w_{timestamp}"
    trace_file = output_path / f"{trace_name}.json"

    # Warmup (outside profiler)
    log(f"Warming up ({warmup_games} games)...")
    for _ in range(warmup_games):
        game_seed = int(rng.integers(0, 2**31))
        hands = deal_from_seed(game_seed)
        decl_id = int(rng.integers(0, 10))
        world_rng = np.random.default_rng(game_seed)
        generate_eq_game(
            oracle, hands, decl_id,
            n_samples=n_samples,
            posterior_config=posterior_config,
            world_rng=world_rng,
        )

    # Determine activities
    # Note: with_stack=True + CPU activity + torch.load triggers PyTorch bug #120235
    # Workaround: use CUDA-only activity with with_stack=True for full traces
    if torch.cuda.is_available():
        activities = [ProfilerActivity.CUDA]
        use_stack = True  # Works with CUDA-only
    else:
        activities = [ProfilerActivity.CPU]
        use_stack = False

    log(f"Starting torch.profiler trace ({n_games} games, activities={[a.name for a in activities]}, stack={use_stack})...")

    # Clear memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    game_times = []

    # Simple context manager pattern (no schedule/step)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=use_stack,
    ) as prof:
        for i in range(n_games):
            game_seed = int(rng.integers(0, 2**31))
            hands = deal_from_seed(game_seed)
            decl_id = int(rng.integers(0, 10))
            world_rng = np.random.default_rng(game_seed)

            game_start = time.perf_counter()
            generate_eq_game(
                oracle, hands, decl_id,
                n_samples=n_samples,
                posterior_config=posterior_config,
                world_rng=world_rng,
            )
            game_time = time.perf_counter() - game_start
            game_times.append(game_time)

    # Extract key averages
    key_averages = prof.key_averages()

    # Sum up CUDA and CPU times from key averages
    # Note: CUDA times may not be available if CUPTI failed (common in WSL2)
    total_cuda_time = 0
    total_cpu_time = 0
    total_self_cuda = 0
    for e in key_averages:
        if hasattr(e, 'cpu_time_total') and e.cpu_time_total > 0:
            total_cpu_time += e.cpu_time_total
        if hasattr(e, 'cuda_time_total') and e.cuda_time_total > 0:
            total_cuda_time += e.cuda_time_total
        if hasattr(e, 'self_cuda_time_total') and e.self_cuda_time_total > 0:
            total_self_cuda += e.self_cuda_time_total

    # Compute result
    total_time = sum(game_times)
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    result = ProfileResult(
        n_games=n_games,
        n_samples=n_samples,
        total_time_s=total_time,
        games_per_sec=n_games / total_time if total_time > 0 else 0,
        decisions_per_sec=(n_games * 28) / total_time if total_time > 0 else 0,
        avg_game_time_s=total_time / n_games if n_games > 0 else 0,
        peak_memory_mb=peak_memory_mb,
        gpu_name=gpu_name,
        cuda_time_ms=total_cuda_time / 1000,  # us to ms
        cpu_time_ms=total_cpu_time / 1000,
        self_cuda_time_ms=total_self_cuda / 1000,
    )

    # Print top operations
    log("Top 20 operations by CPU time:")
    print(key_averages.table(sort_by="cpu_time_total", row_limit=20))

    if total_cuda_time > 0:
        log("Top 20 operations by CUDA time:")
        print(key_averages.table(sort_by="cuda_time_total", row_limit=20))
    else:
        log("Note: CUDA profiling unavailable (CUPTI not initialized - common in WSL2)")

    # Export chrome trace
    log(f"Exporting trace to {trace_file}...")
    prof.export_chrome_trace(str(trace_file))

    return result, trace_file


def run_batch_size_sweep(
    oracle: Stage1Oracle,
    n_games: int = 5,
    seed: int = 42,
    posterior: bool = False,
) -> list[ProfileResult]:
    """Sweep across different n_samples (worlds per decision)."""
    batch_sizes = [32, 64, 128, 256, 512]

    # Check GPU memory to adjust max batch size
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_mem_gb < 6:  # Low memory GPU (like 3050 Ti)
            batch_sizes = [32, 64, 128, 256]
            log(f"Low memory GPU ({gpu_mem_gb:.1f}GB), limiting batch sizes to {batch_sizes}")

    results = []
    for n_samples in batch_sizes:
        log(f"\n{'='*60}")
        log(f"Testing n_samples={n_samples}")
        log(f"{'='*60}")

        try:
            result = profile_games(
                oracle, n_games, n_samples,
                seed=seed, posterior=posterior,
            )
            results.append(result)

            log(f"  Throughput: {result.games_per_sec:.2f} games/s ({result.decisions_per_sec:.0f} decisions/s)")
            log(f"  Avg game time: {result.avg_game_time_s*1000:.0f}ms")
            log(f"  Peak memory: {result.peak_memory_mb:.0f}MB")
        except torch.cuda.OutOfMemoryError:
            log(f"  OOM at n_samples={n_samples}, skipping larger sizes")
            break

        # Clear memory between runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def print_sweep_summary(results: list[ProfileResult]) -> None:
    """Print summary table of sweep results."""
    log("\n" + "="*80)
    log("BATCH SIZE SWEEP SUMMARY")
    log("="*80)
    log(f"GPU: {results[0].gpu_name if results else 'N/A'}")
    log("")
    log(f"{'n_samples':>10} {'games/s':>10} {'dec/s':>10} {'avg_ms':>10} {'peak_MB':>10}")
    log("-"*60)

    best_throughput = max(r.decisions_per_sec for r in results)
    for r in results:
        marker = " *" if r.decisions_per_sec == best_throughput else ""
        log(f"{r.n_samples:>10} {r.games_per_sec:>10.2f} {r.decisions_per_sec:>10.0f} {r.avg_game_time_s*1000:>10.0f} {r.peak_memory_mb:>10.0f}{marker}")

    log("-"*60)
    log("* = best throughput")


def main():
    parser = argparse.ArgumentParser(
        description="Profile E[Q] generator throughput and GPU utilization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--n-games", type=int, default=5, help="Games to profile (default: 5)")
    parser.add_argument("--n-samples", type=int, default=100, help="Worlds per decision (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint", type=str,
                       default="forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
                       help="Stage 1 checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--posterior", action="store_true", help="Enable posterior weighting")

    # Profiling modes
    parser.add_argument("--sweep", action="store_true", help="Run batch size sweep")
    parser.add_argument("--trace", action="store_true", help="Generate torch.profiler trace")
    parser.add_argument("--output-dir", type=str, default="scratch/profiles", help="Trace output directory")

    args = parser.parse_args()

    log("="*60)
    log("E[Q] Generator Profiling (t42-z4yj)")
    log("="*60)

    # Check GPU
    if args.device == "cuda":
        if not torch.cuda.is_available():
            log("ERROR: CUDA not available")
            return 1
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    # Load oracle
    log(f"Loading oracle from {args.checkpoint}...")
    load_start = time.perf_counter()
    oracle = Stage1Oracle(args.checkpoint, device=args.device)
    log(f"Oracle loaded in {time.perf_counter() - load_start:.2f}s")

    if args.sweep:
        # Batch size sweep
        results = run_batch_size_sweep(
            oracle, n_games=args.n_games, seed=args.seed, posterior=args.posterior
        )
        print_sweep_summary(results)
    elif args.trace:
        # Detailed torch.profiler trace
        result, trace_dir = profile_with_torch_profiler(
            oracle, args.n_games, args.n_samples,
            seed=args.seed, output_dir=args.output_dir, posterior=args.posterior,
        )
        log("\n" + "="*60)
        log("PROFILING COMPLETE")
        log("="*60)
        log(f"Throughput: {result.games_per_sec:.2f} games/s ({result.decisions_per_sec:.0f} decisions/s)")
        log(f"Avg game time: {result.avg_game_time_s*1000:.0f}ms")
        log(f"Peak memory: {result.peak_memory_mb:.0f}MB")
        if result.cuda_time_ms:
            log(f"CUDA time: {result.cuda_time_ms:.1f}ms")
            log(f"CPU time: {result.cpu_time_ms:.1f}ms")
            cuda_pct = 100 * result.cuda_time_ms / (result.cuda_time_ms + result.cpu_time_ms)
            log(f"GPU time fraction: {cuda_pct:.1f}%")
        log(f"\nTrace saved to: {trace_dir}")
        log("View in Chrome: chrome://tracing and load the .json file")
    else:
        # Simple profiling
        result = profile_games(
            oracle, args.n_games, args.n_samples,
            seed=args.seed, posterior=args.posterior,
        )
        log("\n" + "="*60)
        log("PROFILING COMPLETE")
        log("="*60)
        log(f"Games: {result.n_games}")
        log(f"Worlds/decision: {result.n_samples}")
        log(f"Throughput: {result.games_per_sec:.2f} games/s")
        log(f"Decisions/sec: {result.decisions_per_sec:.0f}")
        log(f"Avg game time: {result.avg_game_time_s:.3f}s ({result.avg_game_time_s*1000:.0f}ms)")
        log(f"Peak memory: {result.peak_memory_mb:.0f}MB")

    return 0


if __name__ == "__main__":
    exit(main())
