"""Profiling helpers for GPU-native MCTS performance.

This is CUDA-only by design. If CUDA is unavailable, this module raises.

Typical usage:
  python -m forge.zeb.profile_mcts --n-trees 16 --n-sims 100 --export /tmp/gpu_mcts_trace.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from forge.zeb.cuda_only import require_cuda
from forge.zeb.gpu_game_state import deal_random_gpu
from forge.zeb.gpu_mcts import (
    MCTSCUDAGraphRunner,
    backprop_gpu,
    create_forest,
    expand_gpu,
    get_leaf_states,
    get_root_policy_gpu,
    get_terminal_values,
    reset_forest_inplace,
    select_leaves_gpu,
)


@dataclass(frozen=True)
class _RunConfig:
    device: torch.device
    n_trees: int
    n_sims: int
    max_nodes: int
    summary_path: Path | None = None
    use_cudagraph: bool = False


def _dummy_oracle_values(states) -> Tensor:
    # Keep oracle cost near-zero so the profile reflects MCTS overhead/sync.
    return torch.zeros(states.batch_size, dtype=torch.float32, device=states.device)


def _print_tables(prof, limit: int) -> None:
    key_avgs = prof.key_averages()
    has_device_time = any(getattr(e, "device_time_total", 0) > 0 for e in key_avgs)
    sort_by = "device_time_total" if has_device_time else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=limit))

    interesting = [
        "aten::nonzero",
        "aten::index",
        "aten::index_put_",
        "aten::gather",
        "aten::where",
        "aten::argmax",
        "aten::sum",
    ]
    present = {e.key for e in key_avgs}
    subset = [k for k in interesting if k in present]
    if subset:
        print("\n-- Selected ops --")
        for k in subset:
            ev = next(e for e in key_avgs if e.key == k)
            if has_device_time:
                print(
                    f"{k:18s}  calls={ev.count:5d}  device_total={getattr(ev,'device_time_total',0)/1e3:9.3f}ms"
                )
            else:
                print(f"{k:18s}  calls={ev.count:5d}  cpu_total={ev.cpu_time_total/1e3:9.3f}ms")


def _write_summary(prof, out_path: Path, *, mode: str, wall_time_s: float | None = None) -> None:
    key_avgs = prof.key_averages()
    by_key = {e.key: e for e in key_avgs}

    step_ev = by_key.get("ProfilerStep*")
    profiled_steps = int(step_ev.count) if step_ev is not None else 0
    if profiled_steps <= 0:
        profiled_steps = 1

    def _ev_summary(key: str) -> dict:
        e = by_key.get(key)
        if e is None:
            return {"present": False}
        return {
            "present": True,
            "calls": int(e.count),
            "cpu_time_total_us": float(e.cpu_time_total),
            "cpu_time_avg_us": float(e.cpu_time_total) / max(1, int(e.count)),
            "device_time_total_us": float(getattr(e, "device_time_total", 0.0)),
        }

    cuda_launch = _ev_summary("cudaLaunchKernel")
    kernel_per_step = (
        float(cuda_launch["calls"]) / float(profiled_steps) if cuda_launch.get("present") else None
    )

    interesting = [
        "mcts/select",
        "mcts/expand",
        "mcts/backprop",
        "mcts/terminal",
        "mcts/oracle",
        "mcts/leaf_states",
        "mcts/root_policy",
        "cudaLaunchKernel",
        "cudaGraphLaunch",
        "aten::to",
        "aten::_to_copy",
        "aten::index",
        "aten::index_put_",
        "aten::gather",
        "aten::where",
        "aten::argmax",
        "aten::sum",
        "aten::nonzero",
    ]

    payload = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mode": mode,
        "profiled_steps": profiled_steps,
        "kernel_launches_per_profiled_step": kernel_per_step,
        "events": {k: _ev_summary(k) for k in interesting},
    }
    if wall_time_s is not None:
        payload["wall_time_s"] = float(wall_time_s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _profile_gpu_mcts(cfg: _RunConfig, export: Path | None, limit: int) -> None:
    # Fresh deals; use decl_id=0 for reproducibility and to reduce legal-mask variance.
    initial = deal_random_gpu(cfg.n_trees, cfg.device, decl_ids=0)
    original_hands = initial.hands.clone()

    forest = create_forest(
        n_trees=cfg.n_trees,
        max_nodes=cfg.max_nodes,
        initial_states=initial,
        device=cfg.device,
        original_hands=original_hands,
    )

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]

    # Keep profiling overhead manageable: warm up, then capture a short active window.
    warmup = min(5, max(1, cfg.n_sims // 10))
    active = min(20, max(5, cfg.n_sims // 5))
    if warmup + active > cfg.n_sims:
        warmup = max(1, cfg.n_sims // 4)
        active = max(1, cfg.n_sims - warmup)
    schedule = torch.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1)

    if cfg.use_cudagraph:
        runner = MCTSCUDAGraphRunner(forest)
        runner.capture()
        reset_forest_inplace(forest, initial, original_hands=original_hands)

        def _oracle(states) -> Tensor:
            return _dummy_oracle_values(states)

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(cfg.n_sims):
            if cfg.use_cudagraph:
                with torch.inference_mode():
                    with torch.profiler.record_function("mcts/select_graph"):
                        runner._graph_select.replay()
                    with torch.profiler.record_function("mcts/oracle"):
                        runner.oracle_values.copy_(_oracle(runner.leaf_states))
                    with torch.profiler.record_function("mcts/update_graph"):
                        runner._graph_update.replay()
            else:
                with torch.profiler.record_function("mcts/select"):
                    leaf_indices, paths = select_leaves_gpu(forest)

                with torch.profiler.record_function("mcts/leaf_states"):
                    leaf_states = get_leaf_states(forest, leaf_indices)

                with torch.profiler.record_function("mcts/terminal"):
                    terminal_values, is_terminal = get_terminal_values(
                        forest, leaf_indices, leaf_states
                    )

                with torch.profiler.record_function("mcts/oracle"):
                    oracle_values = _dummy_oracle_values(leaf_states)
                    values = torch.where(is_terminal, terminal_values, oracle_values)

                with torch.profiler.record_function("mcts/expand"):
                    expand_gpu(forest, leaf_indices, leaf_states=leaf_states)

                with torch.profiler.record_function("mcts/backprop"):
                    backprop_gpu(forest, leaf_indices, values, paths)

            prof.step()

        with torch.profiler.record_function("mcts/root_policy"):
            _ = get_root_policy_gpu(forest)

    torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - start

    if export is not None:
        export.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(export))
        print(f"\nWrote trace: {export}")

    # Optional structured summary for before/after comparisons.
    if cfg.summary_path is not None:
        _write_summary(
            prof,
            cfg.summary_path,
            mode="cudagraph" if cfg.use_cudagraph else "eager",
            wall_time_s=wall_time_s,
        )
        print(f"Wrote summary: {cfg.summary_path}")

    _print_tables(prof, limit=limit)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Profile GPU-native MCTS (CUDA-only).")
    parser.add_argument("--n-trees", type=int, default=16)
    parser.add_argument("--n-sims", type=int, default=100)
    parser.add_argument("--max-nodes", type=int, default=1024)
    parser.add_argument("--export", type=str, default=None, help="Chrome trace output path.")
    parser.add_argument("--summary", type=str, default=None, help="JSON summary output path.")
    parser.add_argument("--limit", type=int, default=50, help="Profiler row limit.")
    parser.add_argument("--use-cudagraph", action="store_true", help="Capture MCTS step with CUDA graphs.")
    args = parser.parse_args(argv)

    # Avoid implicit multi-thread noise when you're hunting kernel-launch overhead.
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = "1"

    device = require_cuda("cuda", where="forge.zeb.profile_mcts")
    cfg = _RunConfig(
        device=device,
        n_trees=args.n_trees,
        n_sims=args.n_sims,
        max_nodes=args.max_nodes,
        summary_path=Path(args.summary) if args.summary else None,
        use_cudagraph=bool(args.use_cudagraph),
    )
    export = Path(args.export) if args.export else None
    _profile_gpu_mcts(cfg, export=export, limit=args.limit)


if __name__ == "__main__":
    main()
