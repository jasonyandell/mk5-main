"""metal_hoyt/bench.py — gate M5: paired same-session bench vs the CPU line.

Both arms solve IDENTICAL subgames (same worlds, same rung-0 params as the
production cascade: iters 80, gap 0.05, br_every 10, cap 256). Worlds are
enumerated + capped WITHOUT the sigma-consistency filter, so absolute
values are not comparable to the banked reference line — the pairing is
what makes the comparison fair, and the wall anatomy is the deliverable.

Respects the 10-minute bench cap: shrink --seeds, never the budget.

    python -u -m metal_hoyt.bench                # 16-seed strided subset
    python -u -m metal_hoyt.bench --seeds 555001,555090
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import hoyt
from hoyt.cfr import cfr_solve
from hoyt.refsweep import _root_of
from hoyt.toys import payoff_points
from metal_hoyt import cfr_solve_metal
from walt.grade import _world_cap_rng
from walt.worlds import enumerate_worlds

EVALSET = Path(__file__).resolve().parents[1] / "hoyt/evalset_h4_v1.jsonl"
RUNG0 = dict(iters=80, target_gap=0.05, br_every=10)
SLOT_BUDGET = 32_000_000


def make_sub(rec: dict, cap: int):
    root = _root_of(rec["root"])
    w = enumerate_worlds(root)
    if len(w) > cap:
        idx = _world_cap_rng(root, cap).choice(len(w), size=cap,
                                               replace=False)
        w = w[np.sort(idx)]
    return hoyt.build_subgame(root, w, np.full(len(w), 1.0 / len(w)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=None,
                    help="comma-separated; default 16 strided evalset seeds")
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    recs = {r["seed"]: r for r in map(json.loads, open(EVALSET))}
    if a.seeds:
        seeds = [int(s) for s in a.seeds.split(",")]
    else:
        all_seeds = sorted(recs)
        seeds = all_seeds[:: max(1, len(all_seeds) // 16)][:16]

    rows = []
    t_start = time.time()
    for seed in seeds:
        sub = make_sub(recs[seed], a.cap)
        row = {"seed": seed, "worlds": len(sub.worlds)}
        try:
            t0 = time.time()
            gpu = cfr_solve_metal(sub, payoff_points(),
                                  slot_budget=SLOT_BUDGET, **RUNG0)
            row["gpu_s"] = round(time.time() - t0, 2)
            row["gpu"] = {k: round(v, 2) for k, v in gpu.timings.items()}
            row["gpu_gap"] = round(gpu.gap, 5)
            t0 = time.time()
            cpu = cfr_solve(sub, payoff_points(), gap_exit=True,
                            threads=a.threads, slot_budget=SLOT_BUDGET,
                            **RUNG0)
            row["cpu_s"] = round(time.time() - t0, 2)
            row["cpu"] = {k: round(v, 2) for k, v in cpu.timings.items()}
            row["cpu_gap"] = round(cpu.gap, 5)
            row["dv"] = round(abs(cpu.value - gpu.value), 5)
        except hoyt.KernelMemoryError as e:
            row["verdict"] = "slot_capped"
            row["error"] = repr(e)[:120]
        rows.append(row)
        print(json.dumps(row), flush=True)

    done = [r for r in rows if "gpu_s" in r]
    if done:
        gs = sum(r["gpu_s"] for r in done)
        cs = sum(r["cpu_s"] for r in done)
        git = sum(r["gpu"]["iterate"] + r["gpu"]["gpu_gap"] for r in done)
        cit = sum(r["cpu"]["iterate"] + r["cpu"]["br"] for r in done)
        print(f"\npaired {len(done)} roots (cap {a.cap}): "
              f"gpu {gs:.1f}s vs cpu {cs:.1f}s (x{cs / gs:.2f} wall) | "
              f"searcher: gpu {git:.1f}s vs cpu {cit:.1f}s "
              f"(x{cit / max(git, 1e-9):.1f}) | "
              f"total wall {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
