"""Oracle data generator: generate per-seed value tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .declarations import DECL_ID_TO_NAME, ParsedDecls, parse_decl_arg
from .context import build_context
from .output import output_path_for, write_result
from .solve import SolveConfig, build_child_index, enumerate_gpu, solve_gpu
from .tables import DOMINO_HIGH, DOMINO_LOW
from .timer import SeedTimer

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_hand(hand_str: str) -> list[int]:
    """Parse hand string like '6-4,5-5,4-2,3-1,2-0,1-1,0-0' to domino IDs.

    Args:
        hand_str: Comma-separated high-low pairs

    Returns:
        List of 7 domino IDs
    """
    dom_lookup = {}
    for dom_id in range(28):
        high = DOMINO_HIGH[dom_id]
        low = DOMINO_LOW[dom_id]
        dom_lookup[(high, low)] = dom_id

    parts = hand_str.strip().split(",")
    if len(parts) != 7:
        raise ValueError(f"Expected 7 dominoes, got {len(parts)}")

    hand = []
    for part in parts:
        part = part.strip()
        if "-" in part:
            high, low = part.split("-")
            high, low = int(high), int(low)
        elif len(part) == 2:
            high, low = int(part[0]), int(part[1])
        else:
            raise ValueError(f"Cannot parse domino: {part}")

        if high < low:
            high, low = low, high

        key = (high, low)
        if key not in dom_lookup:
            raise ValueError(f"Invalid domino: {high}-{low}")

        dom_id = dom_lookup[key]
        if dom_id in hand:
            raise ValueError(f"Duplicate domino: {high}-{low}")

        hand.append(dom_id)

    return hand


def format_hand(hand: list[int]) -> str:
    """Format hand for display."""
    parts = []
    for dom_id in hand:
        high = DOMINO_HIGH[dom_id]
        low = DOMINO_LOW[dom_id]
        parts.append(f"{high}-{low}")
    return ", ".join(parts)


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
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output (show enumeration progress)")
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb-group", type=str, default=None, help="Wandb group name for organizing runs")

    # Debugging options
    p.add_argument(
        "--p0-hand",
        type=str,
        default=None,
        help='Fix P0 hand (e.g., "6-6,6-5,6-4,6-2,6-1,6-0,2-2"). Remaining dominoes dealt from --seed.',
    )
    p.add_argument(
        "--show-qvals",
        action="store_true",
        help="Print Q-values for root state (P0's opening move)",
    )
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


def _log_memory(phase: str, enabled: bool) -> float | None:
    """Log peak VRAM since last reset, then reset for next phase.

    Returns peak VRAM in GB if logging is enabled, None otherwise.
    """
    if not enabled or not torch.cuda.is_available():
        return None
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / (1024**3)
    print(f"  [{phase}] peak VRAM: {peak_gb:.3f} GB")
    torch.cuda.reset_peak_memory_stats()
    return peak_gb


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available. Install a CUDA-enabled PyTorch build or use --device cpu.")

    parsed: ParsedDecls = parse_decl_arg(args.decl)
    decl_ids = parsed.decl_ids

    # Parse fixed P0 hand if provided
    p0_hand: list[int] | None = None
    if args.p0_hand:
        try:
            p0_hand = parse_hand(args.p0_hand)
            print(f"P0 hand: {format_hand(p0_hand)}")
        except ValueError as e:
            raise SystemExit(f"Invalid --p0-hand: {e}")

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

    # Calculate total work for progress tracking
    total_shards = len(seeds) * len(decl_ids)
    completed = 0

    # Initialize wandb if requested
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: --wandb requested but wandb not installed. Run: pip install wandb")

    if use_wandb:
        run_name = f"gen-{seeds[0]}" if len(seeds) == 1 else f"gen-{seeds[0]}-{seeds[-1]}"
        # Build tags: always include oracle/generation, plus group root if provided
        tags = ["oracle", "generation"]
        if args.wandb_group:
            # Extract root from group (e.g., "cloud-run-xyz/generate" -> "cloud-run-xyz")
            group_root = args.wandb_group.split("/")[0]
            tags.append(group_root)
        wandb.init(
            project="crystal-forge",
            job_type="generate",
            name=run_name,
            group=args.wandb_group,
            dir="runs",  # Consolidate all wandb logs in runs/wandb/
            config={
                "seed_start": seeds[0],
                "seed_end": seeds[-1],
                "seed_count": len(seeds),
                "decl_ids": decl_ids,
                "decl_count": len(decl_ids),
                "total_shards": total_shards,
                "device": str(device),
                "output_dir": str(args.out),
                "format": args.format,
                "child_index_chunk": args.child_index_chunk,
                "solve_chunk": args.solve_chunk,
                "enum_chunk": args.enum_chunk,
            },
            tags=tags,
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

            ctx = build_context(seed=seed, decl_id=decl_id, device=device, p0_hand=p0_hand)
            timer.phase("setup")
            if (vram := _log_memory("setup", args.log_memory)) is not None:
                timer.record_vram("setup", vram)

            all_states = enumerate_gpu(ctx, config=config, verbose=args.verbose)
            timer.phase("enumerate", extra=f"states={int(all_states.shape[0]):,}")
            if (vram := _log_memory("enumerate", args.log_memory)) is not None:
                timer.record_vram("enumerate", vram)

            child_idx = build_child_index(all_states, ctx, config=config)
            timer.phase("child_index")
            if (vram := _log_memory("child_index", args.log_memory)) is not None:
                timer.record_vram("child_index", vram)

            v, move_values = solve_gpu(all_states, child_idx, ctx, config=config)
            root_value = int(v[0])
            timer.phase("solve", extra=f"root={root_value:+d}")
            if (vram := _log_memory("solve", args.log_memory)) is not None:
                timer.record_vram("solve", vram)

            # Show Q-values for root state if requested
            if args.show_qvals:
                # Find the initial state in all_states (they're sorted, so can't use index 0)
                initial_state = ctx.initial_state()
                root_idx = int(torch.searchsorted(all_states, initial_state))
                if root_idx >= all_states.shape[0] or all_states[root_idx].item() != initial_state.item():
                    print(f"WARNING: Could not find initial state in enumerated states")
                else:
                    root_v = int(v[root_idx])
                    root_q = move_values[root_idx].cpu().numpy()
                    print(f"\n=== Q-values for root state (seed={seed}, decl={DECL_ID_TO_NAME.get(decl_id, str(decl_id))}) ===")
                    print(f"V(root) = {root_v:+d}")
                    print(f"P0 hand: {format_hand(list(ctx.L[0].cpu().numpy()))}")
                    print()
                    q_with_idx = [(i, int(root_q[i])) for i in range(7) if root_q[i] != -128]
                    q_with_idx.sort(key=lambda x: -x[1])  # Descending by Q
                    print("Action    Domino    Q-value    Δbest")
                    print("-" * 44)
                    best_q = q_with_idx[0][1] if q_with_idx else 0
                    for local_idx, q_val in q_with_idx:
                        dom_id = int(ctx.L[0, local_idx])
                        dom_str = f"{DOMINO_HIGH[dom_id]}-{DOMINO_LOW[dom_id]}"
                        delta = q_val - best_q
                        marker = " ← BEST" if delta == 0 else ""
                        print(f"  {local_idx}        {dom_str:5}     {q_val:+3d}       {delta:+3d}{marker}")
                    print()

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
            metrics = timer.done(root_value=root_value)

            # Log to wandb
            completed += 1
            if use_wandb:
                wandb.log({
                    **metrics,
                    "progress/completed": completed,
                    "progress/total": total_shards,
                    "progress/percent": 100.0 * completed / total_shards,
                })

    # Ensure final write completes before exiting.
    if write_stream is not None:
        write_stream.synchronize()

    # Finalize wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
