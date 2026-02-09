"""End-to-end epoch timing for self-play (model-in-the-loop), Phase 3 vs Phase 5.

This measures one training epoch (generate + train) using a self-play checkpoint:
- Baseline: CUDA-graph MCTS with model eval outside graphs (Phase 3 style)
- New:      single CUDA graph per sim-step including model eval (Phase 5)

Example:
  python -m forge.zeb.profile_epoch_selfplay \\
    --checkpoint forge/zeb/checkpoints/selfplay-epoch0349.pt \\
    --games-per-epoch 128 --n-parallel-games 128 --n-simulations 50 \\
    --max-mcts-nodes 128 --lr 1e-4 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from forge.zeb import extract_model_config, load_model
from forge.zeb.cuda_only import require_cuda
from forge.zeb.gpu_training_pipeline import create_selfplay_pipeline
from forge.zeb.model import ZebModel


@dataclass(frozen=True)
class _Cfg:
    device: torch.device
    checkpoint: Path
    games_per_epoch: int
    n_parallel_games: int
    n_simulations: int
    wave_size: int
    max_mcts_nodes: int
    batch_size: int
    lr: float
    warmup_games: int
    out_path: Path


def _load_model(checkpoint: Path, device: torch.device) -> tuple[ZebModel, dict]:
    model, ckpt = load_model(str(checkpoint), device=str(device), eval_mode=False)
    model_config = extract_model_config(ckpt)
    meta = {
        "epoch": int(ckpt.get("epoch", 0)),
        "model_config": model_config,
        "total_games": int(ckpt.get("total_games", 0)),
    }
    return model, meta


def _run_epoch(cfg: _Cfg, *, use_fullstep_eval: bool) -> dict:
    model, meta = _load_model(cfg.checkpoint, cfg.device)
    model_cfg = dict(meta.get("model_config", {}))

    pipeline = create_selfplay_pipeline(
        model=model,
        device=str(cfg.device),
        n_parallel_games=cfg.n_parallel_games,
        n_simulations=cfg.n_simulations,
        max_mcts_nodes=cfg.max_mcts_nodes,
        wave_size=cfg.wave_size,
        temperature=1.0,
        use_cudagraph_mcts=True,
        use_fullstep_eval=use_fullstep_eval,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    # Warmup (excluded): cudagraph capture + optimizer kernels.
    if cfg.warmup_games > 0:
        model.eval()
        _ = pipeline.generate_games_gpu(n_games=min(cfg.warmup_games, cfg.n_parallel_games), max_moves=2)
        ex = pipeline.generate_games_gpu(n_games=min(2, cfg.n_parallel_games), max_moves=1)
        _ = pipeline.train_epoch_gpu(model=model, optimizer=optimizer, examples=ex, batch_size=min(cfg.batch_size, 16))

    torch.cuda.reset_peak_memory_stats(cfg.device)

    model_before = pipeline.total_model_queries
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.eval()
    examples = pipeline.generate_games_gpu(n_games=cfg.games_per_epoch, max_moves=28)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0
    model_queries = int(pipeline.total_model_queries - model_before)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    model.train()
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
        "use_fullstep_eval": bool(use_fullstep_eval),
        "gen_s": float(gen_s),
        "train_s": float(train_s),
        "total_s": float(gen_s + train_s),
        "games_per_s": float(cfg.games_per_epoch) / float(gen_s) if gen_s > 0 else 0.0,
        "model_queries": model_queries,
        "model_queries_per_s": float(model_queries) / float(gen_s) if gen_s > 0 else 0.0,
        "n_examples": int(examples.n_examples),
        "examples_per_s": float(examples.n_examples) / float(gen_s) if gen_s > 0 else 0.0,
        "train_metrics": {k: float(v) for k, v in metrics.items()},
        "peak_mem_bytes": peak_mem_bytes,
        "model_config": model_cfg,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="End-to-end epoch timing: self-play Phase 3 vs Phase 5.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--games-per-epoch", type=int, default=128)
    parser.add_argument("--n-parallel-games", type=int, default=128)
    parser.add_argument("--n-simulations", type=int, default=50)
    parser.add_argument("--wave-size", type=int, default=1)
    parser.add_argument("--max-mcts-nodes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-games", type=int, default=8)
    parser.add_argument("--out", type=str, default="history/profiling/gpu_mcts_epoch_selfplay.json")
    args = parser.parse_args(argv)

    device = require_cuda("cuda", where="forge.zeb.profile_epoch_selfplay")
    cfg = _Cfg(
        device=device,
        checkpoint=Path(args.checkpoint),
        games_per_epoch=int(args.games_per_epoch),
        n_parallel_games=int(args.n_parallel_games),
        n_simulations=int(args.n_simulations),
        wave_size=int(args.wave_size),
        max_mcts_nodes=int(args.max_mcts_nodes),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        warmup_games=int(args.warmup_games),
        out_path=Path(args.out),
    )

    print("Running baseline (Phase 3: model eval outside graphs)...")
    baseline = _run_epoch(cfg, use_fullstep_eval=False)
    print("Running full-step (Phase 5: model eval inside graphs)...")
    fullstep = _run_epoch(cfg, use_fullstep_eval=True)

    payload = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "config": {
            "checkpoint": str(cfg.checkpoint),
            "games_per_epoch": cfg.games_per_epoch,
            "n_parallel_games": cfg.n_parallel_games,
            "n_simulations": cfg.n_simulations,
            "wave_size": cfg.wave_size,
            "max_mcts_nodes": cfg.max_mcts_nodes,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "warmup_games": cfg.warmup_games,
        },
        "baseline_phase3": baseline,
        "fullstep_phase5": fullstep,
        "speedup_total": (baseline["total_s"] / fullstep["total_s"]) if fullstep["total_s"] > 0 else None,
        "speedup_gen": (baseline["gen_s"] / fullstep["gen_s"]) if fullstep["gen_s"] > 0 else None,
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote: {cfg.out_path}")
    print(
        f"Gen: baseline={baseline['gen_s']:.2f}s fullstep={fullstep['gen_s']:.2f}s  speedup={payload['speedup_gen']:.2f}x"
    )
    print(
        f"Total: baseline={baseline['total_s']:.2f}s fullstep={fullstep['total_s']:.2f}s  speedup={payload['speedup_total']:.2f}x"
    )


if __name__ == "__main__":
    main()

