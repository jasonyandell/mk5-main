"""Campaign runner: generate diverse training data with random declaration sampling."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from .context import build_context
from .declarations import DECL_ID_TO_NAME, N_DECLS
from .output import output_path_for, write_result
from .solve import SolveConfig, build_child_index, enumerate_gpu, solve_gpu
from .timer import SeedTimer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Campaign runner: 3 random declarations per seed for diverse coverage."
    )
    p.add_argument("--out", type=Path, default=Path("data/solver2"), help="Output directory")
    p.add_argument("--format", choices=("parquet", "pt"), default="parquet", help="Output format")

    seed_group = p.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int, help="Solve one deal seed (3 random decls)")
    seed_group.add_argument(
        "--seed-range", type=str, help="Solve a half-open range: START:END (like Python range)"
    )

    p.add_argument("--decls-per-seed", type=int, default=3, help="Number of random declarations per seed")
    p.add_argument("--device", type=str, default="cuda", help="Device (e.g. cuda, cuda:0, cpu)")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if output exists")
    p.add_argument("--dry-run", action="store_true", help="Show what would be computed without running")

    p.add_argument("--child-index-chunk", type=int, default=1_000_000, help="Chunk size for child index build")
    p.add_argument("--solve-chunk", type=int, default=1_000_000, help="Chunk size for backward induction")
    p.add_argument("--enum-chunk", type=int, default=100_000, help="Chunk size for enumeration")
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


def decls_for_seed(seed: int, k: int = 3) -> list[int]:
    """Reproducibly select k random declarations for a given seed."""
    rng = random.Random(seed)
    return sorted(rng.sample(range(N_DECLS), k=k))


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Install a CUDA-enabled PyTorch build or use --device cpu.")

    if args.seed is not None:
        seeds = [args.seed]
    else:
        start, end = _parse_seed_range(args.seed_range)
        seeds = list(range(start, end))

    # Build work list
    work: list[tuple[int, int]] = []
    for seed in seeds:
        decl_ids = decls_for_seed(seed, k=args.decls_per_seed)
        for decl_id in decl_ids:
            work.append((seed, decl_id))

    # Filter to pending work
    pending = []
    for seed, decl_id in work:
        out_path = output_path_for(args.out, seed, decl_id, args.format)
        if out_path.exists() and not args.overwrite:
            continue
        pending.append((seed, decl_id, out_path))

    print(f"Campaign: {len(seeds)} seeds × {args.decls_per_seed} decls = {len(work)} total")
    print(f"Pending: {len(pending)} (skipping {len(work) - len(pending)} existing)")

    if args.dry_run:
        for seed, decl_id, out_path in pending[:20]:
            decl_name = DECL_ID_TO_NAME.get(decl_id, str(decl_id))
            print(f"  seed={seed} decl={decl_name}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    config = SolveConfig(
        child_index_chunk_size=args.child_index_chunk,
        solve_chunk_size=args.solve_chunk,
        enum_chunk_size=args.enum_chunk,
    )

    for i, (seed, decl_id, out_path) in enumerate(pending):
        timer = SeedTimer(seed=seed, decl_id=decl_id)
        decl_name = DECL_ID_TO_NAME.get(decl_id, str(decl_id))
        timer.phase("start", extra=f"[{i+1}/{len(pending)}] decl={decl_name} device={device}")

        ctx = build_context(seed=seed, decl_id=decl_id, device=device)
        timer.phase("setup")

        all_states = enumerate_gpu(ctx, config=config)
        timer.phase("enumerate", extra=f"states={int(all_states.shape[0]):,}")

        child_idx = build_child_index(all_states, ctx, config=config)
        timer.phase("child_index")

        v, move_values = solve_gpu(all_states, child_idx, ctx, config=config)
        root_value = int(v[0])
        timer.phase("solve", extra=f"root={root_value:+d}")

        write_result(out_path, seed, decl_id, all_states, v, move_values, fmt=args.format)
        timer.phase("write", extra=str(out_path))
        timer.done(root_value=root_value)


if __name__ == "__main__":
    main()
