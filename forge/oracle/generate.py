"""Oracle data generator: generate per-seed value tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .declarations import DECL_ID_TO_NAME, ParsedDecls, parse_decl_arg
from .context import build_context
from .output import output_path_for, write_result
from .solve import SolveConfig, build_child_index, enumerate_gpu, solve_gpu
from .timer import SeedTimer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU solver: generate per-seed regret/value tables.")
    p.add_argument("--out", type=Path, default=Path("data/shards"), help="Output directory")
    p.add_argument("--format", choices=("parquet", "pt"), default="parquet", help="Output format")

    seed_group = p.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", "--seeds", type=int, dest="seed", help="Solve one deal seed")
    seed_group.add_argument("--seed-range", type=str, help="Solve a half-open range: START:END (like Python range)")

    p.add_argument(
        "--decl", "--decls",
        type=str,
        default="all",
        dest="decl",
        help="Declaration: 0..9, name (e.g. fives), or 'all'",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device (e.g. cuda, cuda:0, cpu)")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if output exists")

    p.add_argument("--child-index-chunk", type=int, default=1_000_000, help="Chunk size for child index build")
    p.add_argument("--solve-chunk", type=int, default=1_000_000, help="Chunk size for backward induction")
    p.add_argument("--enum-chunk", type=int, default=100_000, help="Chunk size for enumeration (0=disable)")
    p.add_argument("--log-memory", action="store_true", help="Log peak VRAM usage per phase")
    return p.parse_args()


def _parse_seed_range(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("seed-range must be START:END")
    start = int(parts[0])
    end = int(parts[1])
    if end < start:
        raise ValueError("seed-range END must be >= START")
    return start, end


def _log_memory(phase: str, enabled: bool) -> None:
    """Log peak VRAM since last reset, then reset for next phase."""
    if not enabled or not torch.cuda.is_available():
        return
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / (1024**3)
    print(f"  [{phase}] peak VRAM: {peak_gb:.3f} GB")
    torch.cuda.reset_peak_memory_stats()


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Install a CUDA-enabled PyTorch build or use --device cpu.")

    parsed: ParsedDecls = parse_decl_arg(args.decl)
    decl_ids = parsed.decl_ids

    if args.seed is not None:
        seeds = [args.seed]
    else:
        start, end = _parse_seed_range(args.seed_range)
        seeds = list(range(start, end))

    config = SolveConfig(
        child_index_chunk_size=args.child_index_chunk,
        solve_chunk_size=args.solve_chunk,
        enum_chunk_size=args.enum_chunk,
    )

    # Create a separate stream for I/O to overlap with next seed's computation.
    # The write stream handles GPU→CPU transfers and disk writes while the
    # main stream proceeds to the next seed's setup/enumerate phases.
    write_stream = torch.cuda.Stream() if device.type == "cuda" else None

    for seed in seeds:
        for decl_id in decl_ids:
            out_path = output_path_for(args.out, seed, decl_id, args.format)
            if out_path.exists() and not args.overwrite:
                print(f"skip existing: {out_path}")
                continue

            timer = SeedTimer(seed=seed, decl_id=decl_id)
            timer.phase("start", extra=f"decl={DECL_ID_TO_NAME.get(decl_id, str(decl_id))} device={device}")

            # Wait for previous write to complete before reusing GPU memory.
            # This ensures the previous seed's tensors have been copied to CPU.
            if write_stream is not None:
                torch.cuda.current_stream().wait_stream(write_stream)

            if args.log_memory and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            ctx = build_context(seed=seed, decl_id=decl_id, device=device)
            timer.phase("setup")
            _log_memory("setup", args.log_memory)

            all_states = enumerate_gpu(ctx, config=config)
            timer.phase("enumerate", extra=f"states={int(all_states.shape[0]):,}")
            _log_memory("enumerate", args.log_memory)

            child_idx = build_child_index(all_states, ctx, config=config)
            timer.phase("child_index")
            _log_memory("child_index", args.log_memory)

            v, move_values = solve_gpu(all_states, child_idx, ctx, config=config)
            root_value = int(v[0])
            timer.phase("solve", extra=f"root={root_value:+d}")
            _log_memory("solve", args.log_memory)

            # Write on separate stream to overlap with next seed's computation.
            if write_stream is not None:
                with torch.cuda.stream(write_stream):
                    write_result(
                        out_path, seed, decl_id, all_states, v, move_values,
                        fmt=args.format, non_blocking=True,
                    )
            else:
                write_result(out_path, seed, decl_id, all_states, v, move_values, fmt=args.format)
            timer.phase("write", extra=str(out_path))
            timer.done(root_value=root_value)

    # Ensure final write completes before exiting.
    if write_stream is not None:
        write_stream.synchronize()


if __name__ == "__main__":
    main()
