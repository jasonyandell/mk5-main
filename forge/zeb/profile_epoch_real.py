"""Profile one training epoch end-to-end (real oracle + training), eager vs CUDA-graph MCTS.

This is meant to answer: did CUDA-graph MCTS make real execution faster in aggregate?

Example:
  python -m forge.zeb.profile_epoch_real \\
    --games-per-epoch 128 --n-parallel-games 128 --n-simulations 50 \\
    --max-mcts-nodes 128 --lr 1e-4 --model-size small --batch-size 64
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
from forge.zeb.model import ZebModel, get_model_config
from forge.zeb.oracle_value import create_oracle_value_fn


@dataclass(frozen=True)
class _Cfg:
    device: torch.device
    games_per_epoch: int
    n_parallel_games: int
    n_simulations: int
    wave_size: int
    max_mcts_nodes: int
    model_size: str
    batch_size: int
    lr: float
    warmup_games: int
    out_path: Path
    compile_oracle: bool


def _run_epoch(
    cfg: _Cfg,
    *,
    oracle,
    use_cudagraph_mcts: bool,
) -> dict:
    pipeline = GPUTrainingPipeline(
        oracle=oracle,
        device=cfg.device,
        n_parallel_games=cfg.n_parallel_games,
        n_simulations=cfg.n_simulations,
        wave_size=cfg.wave_size,
        max_mcts_nodes=cfg.max_mcts_nodes,
        temperature=1.0,
        use_cudagraph_mcts=use_cudagraph_mcts,
    )

    model_cfg = get_model_config(cfg.model_size)
    model = ZebModel(**model_cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    # Warmup (excluded from timing): oracle compile/caches + cudagraph capture.
    if cfg.warmup_games > 0:
        _ = pipeline.generate_games_gpu(n_games=min(cfg.warmup_games, cfg.n_parallel_games), max_moves=2)
        # One tiny training step to warm up optimizer kernels.
        ex = pipeline.generate_games_gpu(n_games=min(2, cfg.n_parallel_games), max_moves=1)
        _ = pipeline.train_epoch_gpu(model=model, optimizer=optimizer, examples=ex, batch_size=min(cfg.batch_size, 16))

    torch.cuda.reset_peak_memory_stats(cfg.device)

    oracle_before = pipeline.total_oracle_queries
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    examples = pipeline.generate_games_gpu(n_games=cfg.games_per_epoch, max_moves=28)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    oracle_queries = int(pipeline.total_oracle_queries - oracle_before)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    metrics = pipeline.train_epoch_gpu(
        model=model,
        optimizer=optimizer,
        examples=examples,
        batch_size=cfg.batch_size,
    )
    torch.cuda.synchronize()
    train_s = time.perf_counter() - t1

    peak_mem_bytes = int(torch.cuda.max_memory_allocated(cfg.device))

    return {
        "use_cudagraph_mcts": bool(use_cudagraph_mcts),
        "gen_s": float(gen_s),
        "train_s": float(train_s),
        "total_s": float(gen_s + train_s),
        "games_per_s": float(cfg.games_per_epoch) / float(gen_s) if gen_s > 0 else 0.0,
        "oracle_queries": oracle_queries,
        "oracle_queries_per_s": float(oracle_queries) / float(gen_s) if gen_s > 0 else 0.0,
        "n_examples": int(examples.n_examples),
        "examples_per_s": float(examples.n_examples) / float(gen_s) if gen_s > 0 else 0.0,
        "train_metrics": {k: float(v) for k, v in metrics.items()},
        "peak_mem_bytes": peak_mem_bytes,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="End-to-end epoch timing: eager vs cudagraph MCTS (real oracle).")
    parser.add_argument("--games-per-epoch", type=int, default=128)
    parser.add_argument("--n-parallel-games", type=int, default=128)
    parser.add_argument("--n-simulations", type=int, default=50)
    parser.add_argument("--wave-size", type=int, default=1)
    parser.add_argument("--max-mcts-nodes", type=int, default=128)
    parser.add_argument("--model-size", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-games", type=int, default=8)
    parser.add_argument("--compile-oracle", action="store_true", help="Use torch.compile for oracle model.")
    parser.add_argument("--out", type=str, default="history/profiling/gpu_mcts_epoch_real_oracle.json")
    args = parser.parse_args(argv)

    device = require_cuda("cuda", where="forge.zeb.profile_epoch_real")
    cfg = _Cfg(
        device=device,
        games_per_epoch=int(args.games_per_epoch),
        n_parallel_games=int(args.n_parallel_games),
        n_simulations=int(args.n_simulations),
        wave_size=int(args.wave_size),
        max_mcts_nodes=int(args.max_mcts_nodes),
        model_size=str(args.model_size),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        warmup_games=int(args.warmup_games),
        out_path=Path(args.out),
        compile_oracle=bool(args.compile_oracle),
    )

    if cfg.wave_size != 1:
        print("Note: CUDA-graph MCTS currently applies to wave_size=1 (sequential MCTS).")

    # Share the oracle instance across both runs (so we're measuring MCTS delta, not model init).
    oracle = create_oracle_value_fn(device=str(device), compile=cfg.compile_oracle)

    print("Running eager epoch...")
    eager = _run_epoch(cfg, oracle=oracle, use_cudagraph_mcts=False)
    print("Running cudagraph epoch...")
    graph = _run_epoch(cfg, oracle=oracle, use_cudagraph_mcts=True)

    payload = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "config": {
            "games_per_epoch": cfg.games_per_epoch,
            "n_parallel_games": cfg.n_parallel_games,
            "n_simulations": cfg.n_simulations,
            "wave_size": cfg.wave_size,
            "max_mcts_nodes": cfg.max_mcts_nodes,
            "model_size": cfg.model_size,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "warmup_games": cfg.warmup_games,
            "compile_oracle": cfg.compile_oracle,
        },
        "eager": eager,
        "cudagraph": graph,
        "speedup_total": (eager["total_s"] / graph["total_s"]) if graph["total_s"] > 0 else None,
        "speedup_gen": (eager["gen_s"] / graph["gen_s"]) if graph["gen_s"] > 0 else None,
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {cfg.out_path}")
    print(
        f"Total: eager={eager['total_s']:.2f}s graph={graph['total_s']:.2f}s  speedup={payload['speedup_total']:.2f}x"
    )
    print(
        f"Gen:   eager={eager['gen_s']:.2f}s graph={graph['gen_s']:.2f}s  speedup={payload['speedup_gen']:.2f}x"
    )


if __name__ == "__main__":
    main()
