#!/usr/bin/env python3
"""Benchmark script for vectorized game simulation.

Usage:
    # Quick benchmark (default)
    python -m forge.bidding.benchmark

    # With profiling
    python -m forge.bidding.benchmark --profile

    # Custom batch size and iterations
    python -m forge.bidding.benchmark --n-games 500 --n-hands 10

    # Compare with baseline (if available)
    python -m forge.bidding.benchmark --compare
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch

from forge.bidding.inference import PolicyModel
from forge.bidding.simulator import (
    BatchedGameState,
    deal_random_hands,
    simulate_games,
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    n_hands: int
    n_games_per_hand: int
    total_games: int
    total_time_s: float
    hands_per_minute: float
    games_per_second: float
    avg_points: float
    device: str


@contextmanager
def timer() -> Iterator[list]:
    """Context manager for timing code blocks."""
    result = [0.0]
    start = time.perf_counter()
    try:
        yield result
    finally:
        result[0] = time.perf_counter() - start


def run_benchmark(
    model: PolicyModel,
    n_hands: int = 5,
    n_games: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run benchmark on game simulation.

    Args:
        model: The policy model for action selection
        n_hands: Number of hands to evaluate
        n_games: Number of games per hand
        seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        BenchmarkResult with timing statistics
    """
    device_name = str(model.device)

    # Sample bidder hands (use first n_hands from a shuffled deck)
    torch.manual_seed(seed)
    all_doms = list(range(28))

    total_points = 0.0
    total_games = 0

    with timer() as elapsed:
        for hand_idx in range(n_hands):
            # Generate a random hand for the bidder
            perm = torch.randperm(28).tolist()
            bidder_hand = sorted(perm[:7])

            # Use fives as trump (decl_id=5)
            decl_id = 5

            # Simulate games
            points = simulate_games(
                model=model,
                bidder_hand=bidder_hand,
                decl_id=decl_id,
                n_games=n_games,
                seed=seed + hand_idx,
                greedy=False,
            )

            total_points += points.float().mean().item()
            total_games += n_games

            if verbose:
                avg = points.float().mean().item()
                print(f"  Hand {hand_idx + 1}/{n_hands}: avg points = {avg:.2f}")

    total_time = elapsed[0]
    hands_per_min = n_hands / total_time * 60
    games_per_sec = total_games / total_time
    avg_points = total_points / n_hands

    return BenchmarkResult(
        n_hands=n_hands,
        n_games_per_hand=n_games,
        total_games=total_games,
        total_time_s=total_time,
        hands_per_minute=hands_per_min,
        games_per_second=games_per_sec,
        avg_points=avg_points,
        device=device_name,
    )


def run_profiled_benchmark(
    model: PolicyModel,
    n_hands: int = 3,
    n_games: int = 100,
    seed: int = 42,
) -> None:
    """Run benchmark with PyTorch profiler.

    Args:
        model: The policy model
        n_hands: Number of hands to profile
        n_games: Number of games per hand
        seed: Random seed
    """
    from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

    print(f"\nRunning profiled benchmark: {n_hands} hands × {n_games} games")
    print("Profiling with PyTorch profiler...")

    # Warmup
    torch.manual_seed(seed)
    perm = torch.randperm(28).tolist()
    bidder_hand = sorted(perm[:7])
    _ = simulate_games(model, bidder_hand, 5, n_games, seed=seed)

    # Profile
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for hand_idx in range(n_hands):
            perm = torch.randperm(28).tolist()
            bidder_hand = sorted(perm[:7])
            _ = simulate_games(model, bidder_hand, 5, n_games, seed=seed + hand_idx)

    # Print summary
    print("\n" + "=" * 80)
    print("CPU Time Summary (top 20 operations):")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("CUDA Time Summary (top 20 operations):")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Export trace for Chrome tracing
    trace_path = "scratch/profiler_trace.json"
    print(f"\nExporting trace to {trace_path}")
    prof.export_chrome_trace(trace_path)


def run_component_benchmark(model: PolicyModel, n_games: int = 500) -> None:
    """Benchmark individual components of the simulation."""
    print(f"\nComponent benchmark (n_games={n_games}):")
    print("-" * 60)

    device = model.device

    # Setup
    torch.manual_seed(42)
    perm = torch.randperm(28).tolist()
    bidder_hand = sorted(perm[:7])
    decl_id = 5

    # Benchmark deal_random_hands
    rng = torch.Generator()
    rng.manual_seed(42)
    with timer() as t:
        for _ in range(10):
            hands = deal_random_hands(bidder_hand, n_games, device, rng)
    print(f"  deal_random_hands: {t[0]*100:.2f}ms (10 calls)")

    # Benchmark BatchedGameState init
    with timer() as t:
        for _ in range(10):
            state = BatchedGameState(hands, decl_id, device)
    print(f"  BatchedGameState.__init__: {t[0]*100:.2f}ms (10 calls)")

    # Benchmark build_tokens
    state = BatchedGameState(hands, decl_id, device)
    with timer() as t:
        for _ in range(100):
            tokens, mask, current = state.build_tokens()
    print(f"  build_tokens: {t[0]*10:.2f}ms (100 calls)")

    # Benchmark get_legal_mask
    with timer() as t:
        for _ in range(100):
            legal = state.get_legal_mask()
    print(f"  get_legal_mask: {t[0]*10:.2f}ms (100 calls)")

    # Benchmark step (with dummy actions)
    state = BatchedGameState(hands, decl_id, device)
    actions = torch.zeros(n_games, dtype=torch.long, device=device)
    with timer() as t:
        for _ in range(28):  # Full game
            if state.is_game_over().all():
                break
            legal = state.get_legal_mask()
            # Pick first legal action
            actions = legal.long().argmax(dim=1)
            state.step(actions)
    print(f"  Full game (28 steps): {t[0]*1000:.2f}ms")

    # Benchmark model forward pass
    state = BatchedGameState(hands, decl_id, device)
    tokens, mask, current = state.build_tokens()
    legal = state.get_legal_mask()
    model.warmup(n_games)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with timer() as t:
        for _ in range(100):
            _ = model.sample_actions(tokens, mask, current, legal)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print(f"  model.sample_actions: {t[0]*10:.2f}ms (100 calls)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark game simulation")
    parser.add_argument("--n-hands", type=int, default=5, help="Number of hands to evaluate")
    parser.add_argument("--n-games", type=int, default=200, help="Games per hand")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--profile", action="store_true", help="Run with PyTorch profiler")
    parser.add_argument("--components", action="store_true", help="Benchmark individual components")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load model
    print("\nLoading model...")
    model = PolicyModel(compile_model=not args.no_compile)
    print(f"Model loaded on {model.device}")

    if args.components:
        run_component_benchmark(model, n_games=args.n_games)
        return

    if args.profile:
        run_profiled_benchmark(model, n_hands=args.n_hands, n_games=args.n_games, seed=args.seed)
        return

    # Run benchmark
    print(f"\nRunning benchmark: {args.n_hands} hands × {args.n_games} games")
    print("-" * 60)

    result = run_benchmark(
        model=model,
        n_hands=args.n_hands,
        n_games=args.n_games,
        seed=args.seed,
        verbose=True,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Device:           {result.device}")
    print(f"  Hands evaluated:  {result.n_hands}")
    print(f"  Games per hand:   {result.n_games_per_hand}")
    print(f"  Total games:      {result.total_games}")
    print(f"  Total time:       {result.total_time_s:.2f}s")
    print(f"  Hands/minute:     {result.hands_per_minute:.2f}")
    print(f"  Games/second:     {result.games_per_second:.2f}")
    print(f"  Avg points:       {result.avg_points:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
