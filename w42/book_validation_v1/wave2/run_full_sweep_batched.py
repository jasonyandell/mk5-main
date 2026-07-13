#!/usr/bin/env python3
"""Batched full-sweep runner for Wave 2.B.2.

Runs the forge generator in batches of --batch-seeds seeds (default 10),
then joins all per-batch .pt files into a single bid_aware_actions.csv.

This works around the MPS INT_MAX tensor-dim limit that prevents generating
500+ games (50 seeds × 10 decls) in a single forge.eq.generate call.

Usage:
  python -u w42/book_validation_v1/wave2/run_full_sweep_batched.py \
    --start-seed 9000 --n-seeds 50 --batch-seeds 10 \
    --n-decl-per-seed 10 --n-samples 200 \
    --bid-values "30,32,35,36,39,42,84" \
    --device mps \
    --output-dir w42/book_validation_v1/wave2/bid_aware_atlas
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# Re-import all helpers from the driver (avoid duplicating logic)
sys.path.insert(0, str(Path(__file__).parent))
from run_bid_aware_atlas import (
    ACTION_COLUMNS,
    DECL_NAMES,
    SEAT_ROLE,
    OFFENSE_PLAYERS,
    find_checkpoint,
    git_sha,
    mark_multiplier,
    process_pt_file,
    q_stats_from_world_slice,
    q_to_mark_utility_per_world,
    round4,
    sha256_file,
    threshold_q_for_player,
    validate_bid30_against_atlas,
    write_csv,
    write_json,
    compute_mark_ev_divergence,
    p_make_from_mark_util,
)

SCHEMA_VERSION = "w42.bookval.wave2.bid_aware_atlas.v1.batched"
BEAD_ID = "t42-7eop"


def generate_batch(
    *,
    bid_value: int,
    start_seed: int,
    n_seeds: int,
    n_decl_per_seed: int,
    n_samples: int,
    output_path: Path,
    device: str,
    checkpoint: str,
) -> dict[str, Any]:
    """Run forge generator for one (bid_value, seed_batch)."""
    n_games = n_seeds * n_decl_per_seed
    end_seed = start_seed + n_seeds - 1
    bid_str = ",".join([str(bid_value)] * n_games)

    cmd = [
        sys.executable, "-u", "-m", "forge.eq.generate",
        "--start-seed", str(start_seed),
        "--n-games", str(n_games),
        "--n-decl-per-seed", str(n_decl_per_seed),
        "--samples", str(n_samples),
        "--bid-values", bid_str,
        "--schema", "v2",
        "--save-joint-worlds",
        "--device", device,
        "--checkpoint", checkpoint,
        "--output", str(output_path),
    ]

    print(
        f"  [batch] bid={bid_value} seeds={start_seed}..{end_seed} x {n_decl_per_seed} decls = {n_games} games",
        flush=True,
    )

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=False)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        return {"status": "error", "returncode": result.returncode, "elapsed_s": round(elapsed, 2)}

    print(f"    Done in {elapsed:.1f}s", flush=True)
    return {"status": "ok", "elapsed_s": round(elapsed, 2)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched bid-aware atlas runner — Wave 2.B.2",
    )
    parser.add_argument("--start-seed", type=int, default=9000)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--batch-seeds", type=int, default=10,
                        help="Seeds per forge.eq.generate call (default 10)")
    parser.add_argument("--n-decl-per-seed", type=int, default=10)
    parser.add_argument("--bid-values", type=str, default="30,32,35,36,39,42,84")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "w42/book_validation_v1/wave2/bid_aware_atlas")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip generation if .pt file already exists (default: True)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bid_values = [int(b.strip()) for b in args.bid_values.split(",") if b.strip()]
    start_seed = args.start_seed
    n_seeds = args.n_seeds
    n_decl = args.n_decl_per_seed
    n_samples = args.n_samples
    batch_size = args.batch_seeds
    end_seed = start_seed + n_seeds - 1

    # Find checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_checkpoint(ROOT)
        if checkpoint is None:
            print("Error: No model checkpoint found. Use --checkpoint.", flush=True)
            return 1

    # Build seed batches
    seed_batches = []
    s = start_seed
    while s <= end_seed:
        batch_end = min(s + batch_size - 1, end_seed)
        seed_batches.append((s, batch_end - s + 1))
        s = batch_end + 1

    total_games = n_seeds * n_decl * len(bid_values)
    print(
        f"[batched_atlas] {n_seeds} seeds x {n_decl} decls x {len(bid_values)} bids x {n_samples} samples",
        flush=True,
    )
    print(
        f"[batched_atlas] {len(seed_batches)} batches of {batch_size} seeds each, {len(bid_values)} bids",
        flush=True,
    )
    print(f"[batched_atlas] device={args.device}, checkpoint={checkpoint}", flush=True)
    print(f"[batched_atlas] output_dir={args.output_dir}", flush=True)

    # Phase 1: Generate per-batch per-bid .pt files
    all_pt_files: list[tuple[int, Path]] = []  # (bid_value, pt_path)
    gen_log: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for batch_start, batch_n in seed_batches:
        batch_end = batch_start + batch_n - 1
        print(f"\n[batched_atlas] === Batch seeds {batch_start}..{batch_end} ===", flush=True)
        for bv in bid_values:
            pt_path = args.output_dir / f"eq_pdf_seeds{batch_start}-{batch_end}_bid{bv}_v2.pt"
            all_pt_files.append((bv, pt_path))

            if pt_path.exists() and args.skip_existing:
                print(
                    f"  [batch] bid={bv} seeds={batch_start}..{batch_end}: exists, skipping",
                    flush=True,
                )
                gen_log.append({
                    "bid": bv, "batch_start": batch_start, "batch_end": batch_end,
                    "status": "skipped", "reason": "file_exists",
                })
                continue

            meta = generate_batch(
                bid_value=bv,
                start_seed=batch_start,
                n_seeds=batch_n,
                n_decl_per_seed=n_decl,
                n_samples=n_samples,
                output_path=pt_path,
                device=args.device,
                checkpoint=checkpoint,
            )
            entry = {"bid": bv, "batch_start": batch_start, "batch_end": batch_end, **meta}
            gen_log.append(entry)

            if meta["status"] == "error":
                failures.append(entry)
                print(
                    f"  [batch] FAILURE bid={bv} seeds={batch_start}..{batch_end}: "
                    f"returncode={meta.get('returncode')}. Continuing.",
                    flush=True,
                )

    if failures:
        print(f"\n[batched_atlas] WARNING: {len(failures)} batch failures:", flush=True)
        for f in failures:
            print(f"  {f}", flush=True)

    # Phase 2: Join all .pt files into a single CSV
    print("\n[batched_atlas] === Joining per-batch .pt files ===", flush=True)
    all_action_rows: list[dict[str, Any]] = []
    join_log: list[dict[str, Any]] = []

    for bv, pt_path in all_pt_files:
        if not pt_path.exists():
            print(f"  Missing: {pt_path.name}", flush=True)
            join_log.append({"bid": bv, "file": pt_path.name, "status": "missing"})
            continue
        try:
            rows = process_pt_file(pt_path, bv)
            all_action_rows.extend(rows)
            join_log.append({
                "bid": bv, "file": pt_path.name, "status": "ok",
                "rows": len(rows),
                "sha256": sha256_file(pt_path),
            })
            print(f"  {pt_path.name}: {len(rows)} rows", flush=True)
        except Exception as exc:
            print(f"  ERROR: {pt_path.name}: {exc}", flush=True)
            join_log.append({"bid": bv, "file": pt_path.name, "status": "error", "error": str(exc)})

    # Write joined CSV
    joined_csv = args.output_dir / "bid_aware_actions.csv"
    write_csv(joined_csv, all_action_rows, fieldnames=ACTION_COLUMNS)
    print(f"\n[batched_atlas] Wrote {len(all_action_rows)} rows to {joined_csv}", flush=True)

    # Phase 3: Compute mark_ev divergence statistics
    print("[batched_atlas] Computing mark_ev divergence ...", flush=True)
    divergence = compute_mark_ev_divergence(all_action_rows) if all_action_rows else {"status": "no_data"}

    # Phase 4: Validation at bid=30 vs branch_atlas_scaled_v0
    print("[batched_atlas] Validating bid=30 against branch_atlas_scaled_v0 ...", flush=True)
    atlas_pt = ROOT / "w42/branch_atlas_scaled_v0/eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt"
    rows_bid30 = [r for r in all_action_rows if int(r["bid_value"]) == 30]
    val_csv = args.output_dir / "validation_check.csv"
    val_result = (
        validate_bid30_against_atlas(rows_bid30, atlas_pt, val_csv)
        if rows_bid30
        else {"status": "skipped", "reason": "no_bid30_rows"}
    )

    # Phase 5: Per-bid row counts
    from collections import Counter
    bid_row_counts = Counter(int(r["bid_value"]) for r in all_action_rows)

    # Phase 6: Write manifest
    elapsed = time.perf_counter() - t_start
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bead_id": BEAD_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repo_commit": git_sha(),
        "run_label": "full_sweep_wave2b2",
        "args": {
            "start_seed": start_seed,
            "n_seeds": n_seeds,
            "batch_seeds": batch_size,
            "n_decl_per_seed": n_decl,
            "bid_values": bid_values,
            "n_samples": n_samples,
            "device": args.device,
            "checkpoint": str(checkpoint),
        },
        "seed_batches": [{"start": s, "n": n, "end": s + n - 1} for s, n in seed_batches],
        "generation_log": gen_log,
        "failures": failures,
        "join_log": join_log,
        "totals": {
            "n_seeds": n_seeds,
            "n_decl_per_seed": n_decl,
            "n_bid_values": len(bid_values),
            "n_total_games": n_seeds * n_decl * len(bid_values),
            "n_action_rows": len(all_action_rows),
            "bid_row_counts": {str(k): v for k, v in sorted(bid_row_counts.items())},
        },
        "mark_ev_divergence": divergence,
        "validation_bid30": val_result,
        "artifacts": {
            "bid_aware_actions_csv": str(joined_csv),
            "validation_check_csv": str(val_csv),
        },
        "wall_seconds": round(elapsed, 2),
    }

    write_json(args.output_dir / "manifest.json", manifest)
    print(
        f"\n[batched_atlas] Done in {elapsed:.1f}s ({elapsed/60:.1f} min). "
        f"{len(all_action_rows)} total action rows across {len(bid_values)} bids.",
        flush=True,
    )

    # Summary
    print("\n=== BID ROW COUNTS ===", flush=True)
    for bv in bid_values:
        cnt = bid_row_counts.get(bv, 0)
        print(f"  bid={bv}: {cnt} rows", flush=True)

    print("\n=== VALIDATION (bid=30 vs branch_atlas_scaled_v0) ===", flush=True)
    for k, v in val_result.items():
        if k != "note":
            print(f"  {k}: {v}", flush=True)

    print("\n=== MARK EV DIVERGENCE BY BID ===", flush=True)
    for bv_str, stats in divergence.get("by_bid", {}).items():
        headline = stats.get("headline", f"bid={bv_str}: {stats}")
        print(f"  {headline}", flush=True)

    if failures:
        print(f"\n=== FAILURES ({len(failures)} batches) ===", flush=True)
        for f in failures:
            print(f"  bid={f['bid']} seeds={f['batch_start']}..{f['batch_end']}: {f}", flush=True)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
