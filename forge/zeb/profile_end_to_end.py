"""End-to-end profiling for GPU-native MCTS with real oracle execution.

This benchmarks the real GPU pipeline (deals -> MCTS -> oracle/model eval) and
compares CUDA-graph MCTS vs eager. CUDA-only by design.

Example:
  python -m forge.zeb.profile_end_to_end --n-games 64 --n-parallel-games 16 --n-simulations 100 --max-mcts-nodes 1024
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from forge.zeb.cuda_only import require_cuda
from forge.zeb.gpu_training_pipeline import GPUTrainingPipeline
from forge.zeb.oracle_value import create_oracle_value_fn


@dataclass(frozen=True)
class _BenchConfig:
    device: torch.device
    n_games: int
    n_parallel_games: int
    n_simulations: int
    wave_size: int
    max_mcts_nodes: int
    max_moves: int
    temperature: float
    repeats: int
    warmup_games: int
    compile_oracle: bool


def _run_once(pipeline: GPUTrainingPipeline, n_games: int) -> dict[str, float | int]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    oracle_before = pipeline.total_oracle_queries
    examples = pipeline.generate_games_gpu(n_games=n_games, max_moves=pipeline._profile_max_moves)
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    oracle_delta = pipeline.total_oracle_queries - oracle_before
    n_examples = int(getattr(examples, "n_examples", 0))

    return {
        "wall_time_s": float(wall),
        "oracle_queries": int(oracle_delta),
        "n_examples": int(n_examples),
    }


def _bench_mode(cfg: _BenchConfig, *, use_cudagraph_mcts: bool) -> dict:
    # Build oracle (optionally compiled). We want this to reflect real execution cost,
    # but compilation time itself is excluded by warmup.
    oracle = create_oracle_value_fn(device=str(cfg.device), compile=cfg.compile_oracle)

    pipeline = GPUTrainingPipeline(
        oracle=oracle,
        device=cfg.device,
        n_parallel_games=cfg.n_parallel_games,
        n_simulations=cfg.n_simulations,
        wave_size=cfg.wave_size,
        max_mcts_nodes=cfg.max_mcts_nodes,
        temperature=cfg.temperature,
        use_cudagraph_mcts=use_cudagraph_mcts,
    )
    pipeline._profile_max_moves = int(cfg.max_moves)

    # Warmup: triggers oracle compilation/inductor caching + CUDA-graph capture (if enabled).
    warmup = min(cfg.warmup_games, cfg.n_games)
    if warmup > 0:
        _ = _run_once(pipeline, n_games=warmup)

    results = []
    for _ in range(cfg.repeats):
        results.append(_run_once(pipeline, n_games=cfg.n_games))

    wall_times = [r["wall_time_s"] for r in results]
    oracle_q = [r["oracle_queries"] for r in results]
    n_examples = [r["n_examples"] for r in results]

    best_idx = min(range(len(results)), key=lambda i: wall_times[i])
    best = results[best_idx]

    mean_wall = float(sum(wall_times) / len(wall_times))
    mean_oracle = float(sum(oracle_q) / len(oracle_q))
    mean_examples = float(sum(n_examples) / len(n_examples))

    return {
        "use_cudagraph_mcts": bool(use_cudagraph_mcts),
        "wave_size": int(cfg.wave_size),
        "timings": results,
        "best": best,
        "mean": {
            "wall_time_s": mean_wall,
            "games_per_s": float(cfg.n_games) / mean_wall if mean_wall > 0 else 0.0,
            "examples_per_s": mean_examples / mean_wall if mean_wall > 0 else 0.0,
            "oracle_queries_per_s": mean_oracle / mean_wall if mean_wall > 0 else 0.0,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="End-to-end profiling (real oracle) for GPU MCTS.")
    parser.add_argument("--n-games", type=int, default=64)
    parser.add_argument("--n-parallel-games", type=int, default=16)
    parser.add_argument("--n-simulations", type=int, default=100)
    parser.add_argument("--wave-size", type=int, default=1)
    parser.add_argument("--max-mcts-nodes", type=int, default=1024)
    parser.add_argument("--max-moves", type=int, default=2, help="Moves per game to simulate (profiling only).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-games", type=int, default=8)
    parser.add_argument("--compile-oracle", action="store_true", help="Use torch.compile for oracle.")
    parser.add_argument("--out", type=str, default="history/profiling/gpu_mcts_end_to_end_real_oracle.json")
    args = parser.parse_args(argv)

    device = require_cuda("cuda", where="forge.zeb.profile_end_to_end")
    cfg = _BenchConfig(
        device=device,
        n_games=int(args.n_games),
        n_parallel_games=int(args.n_parallel_games),
        n_simulations=int(args.n_simulations),
        wave_size=int(args.wave_size),
        max_mcts_nodes=int(args.max_mcts_nodes),
        max_moves=int(args.max_moves),
        temperature=float(args.temperature),
        repeats=int(args.repeats),
        warmup_games=int(args.warmup_games),
        compile_oracle=bool(args.compile_oracle),
    )

    if cfg.wave_size != 1:
        print("Note: CUDA-graph MCTS currently applies to wave_size=1 (sequential MCTS).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Benchmarking eager...")
    eager = _bench_mode(cfg, use_cudagraph_mcts=False)
    print("Benchmarking cudagraph...")
    graph = _bench_mode(cfg, use_cudagraph_mcts=True)

    payload = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "config": {
            "n_games": cfg.n_games,
            "n_parallel_games": cfg.n_parallel_games,
            "n_simulations": cfg.n_simulations,
            "wave_size": cfg.wave_size,
            "max_mcts_nodes": cfg.max_mcts_nodes,
            "max_moves": cfg.max_moves,
            "temperature": cfg.temperature,
            "repeats": cfg.repeats,
            "warmup_games": cfg.warmup_games,
            "compile_oracle": cfg.compile_oracle,
        },
        "eager": eager,
        "cudagraph": graph,
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {out_path}")

    e = eager["mean"]["games_per_s"]
    g = graph["mean"]["games_per_s"]
    speedup = (g / e) if e else 0.0
    print(f"E2E games/s: eager={e:.3f}  cudagraph={g:.3f}  speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
