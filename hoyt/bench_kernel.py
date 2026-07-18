"""hoyt/bench_kernel.py — K3: the kernel bench (≤10 min wall).

Three sections, every number printed with its comparison denominator:

1. **46-fixture suite** (walt/tests/fixtures_h4.jsonl, σ-filtered worlds):
   fresh walt.solver wall (today's net-wavefront denominator), compile_sigma
   wall (P2: ~one net solve), br_solve fast wall (P1 gate: ≤0.5s total,
   ≥17× vs the 8.5s registered wavefront; also vs the 125.4s recursion
   baseline_ms), br_solve rewalk wall (the honest no-cache re-walk), and
   the full K1 parity check (best_move identical, value + every per-move
   root value ≤1e-9 rel, node counts equal).
2. **Stochastic stress**: uniform-over-legal profile (full support — the
   worst case) on the parity-subset seeds, worlds capped at 512 (grade.py
   subsample pattern); reports the tree blowup factor vs the deterministic
   walk and the wall, including the chunked-fallback fixture if budget
   remains.
3. **H5-capped(512)**: ~20 roots via walt.bench.build_root_at_horizon
   (seed, 5, jud), u-worlds capped to 512; compile + BR walls (P5 gate:
   BR p50 < 100 ms).

Numba decision (deliverable 4): NOT used. With the σ-walk structure cached
in the SigmaTable, deterministic BR is bincount/segmented-reduceat bound at
~7 ns/node on the backward pass (~40 ns/node all-in with strategy
extraction) and beats P1 by >10×; the only python/numpy-overhead-bound
path left is the stochastic rewalk, which is a diagnostic stress case, not
a gate. Pure numpy keeps the parity story single-sourced.

Run: PYTHONPATH=. python -u hoyt/bench_kernel.py [--h5-n 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BUDGET_S = 600.0
REL_TOL = 1e-9
NET_WAVEFRONT_S = 8.5          # registered prior denominator (perf-log P1)

STRESS_SEEDS = (777005, 777003, 777004, 777010, 777016, 777017, 777022,
                777006, 777011, 777013, 777018, 777027)
STRESS_CHUNK_SEED = 777009     # all-my-lead shape: blows the slot budget,
                               # exercises the per-root-move chunk fallback


def _root_of(rd: dict):
    from walt.contracts import EndgameRoot

    return EndgameRoot(
        decl_id=rd["decl_id"], bidder=rd["bidder"], bid_value=rd["bid_value"],
        bids=tuple(rd["bids"]), dealer=rd["dealer"], me=rd["me"],
        my_hand=tuple(rd["my_hand"]),
        play_history=tuple((s, d) for s, d in rd["play_history"]),
        trick_leader=rd["trick_leader"],
        current_trick=tuple(rd["current_trick"]),
        team_points=tuple(rd["team_points"]))


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def _fixture_worlds(root, oracle):
    from walt.field import sigma_consistent
    from walt.grade import _make_moves_filter
    from walt.worlds import enumerate_worlds

    worlds = enumerate_worlds(root)
    keep = sigma_consistent(root, worlds, oracle, _make_moves_filter(root))
    filtered = worlds[keep]
    if len(filtered) > 0:
        worlds = filtered
    return worlds


def _cap(root, worlds, cap):
    from walt.grade import _world_cap_rng

    if cap and len(worlds) > cap:
        idx = _world_cap_rng(root, cap).choice(
            len(worlds), size=cap, replace=False)
        worlds = worlds[np.sort(idx)]
    return worlds


def bench_fixture_suite(oracle, t_start) -> dict:
    import hoyt as K
    from walt.solver import solve

    fix_path = Path(__file__).resolve().parents[1] / "walt" / "tests"
    recs = [json.loads(l) for l in (fix_path / "fixtures_h4.jsonl").open()]
    exp = {r["seed"]: r for r in map(
        json.loads, (fix_path / "fixtures_h4_expected.jsonl").open())}

    pay = K.payoff_points()
    t_walt = t_comp = t_br = t_rw = 0.0
    br_each, rw_each = [], []
    nodes_tot = 0
    rows_tot = 0
    baseline_tot = sum(exp[r["seed"]]["baseline_ms"] for r in recs) / 1e3
    k1_pass = 0
    fails = []
    last_log = time.time()

    for i, rec in enumerate(recs):
        root = _root_of(rec["root"])
        worlds = _fixture_worlds(root, oracle)
        weights = np.full(len(worlds), 1.0 / len(worlds), dtype=np.float64)

        t0 = time.perf_counter()
        ref = solve(root, worlds, weights, oracle, payoff="points")
        t_walt += time.perf_counter() - t0

        t0 = time.perf_counter()
        table = K.compile_sigma(root, worlds, oracle)
        t_comp += time.perf_counter() - t0
        rows_tot += table.meta["net_rows"]

        sub = K.build_subgame(root, worlds, weights)
        t0 = time.perf_counter()
        br = K.br_solve(sub, table, pay)
        dt = time.perf_counter() - t0
        t_br += dt
        br_each.append(dt * 1e3)

        t0 = time.perf_counter()
        br2 = K.br_solve(sub, table, pay, rewalk=True)
        dt = time.perf_counter() - t0
        t_rw += dt
        rw_each.append(dt * 1e3)
        nodes_tot += table.n_nodes

        ok = (br.best_move == ref.best_move
              and abs(br.value - ref.value)
              <= REL_TOL * max(1.0, abs(ref.value))
              and set(br.root_values) == set(ref.root_values)
              and all(abs(br.root_values[k] - v)
                      <= REL_TOL * max(1.0, abs(v))
                      for k, v in ref.root_values.items())
              and table.n_nodes == ref.n_nodes
              and br2.best_move == br.best_move
              and abs(br2.value - br.value) <= 1e-12)
        k1_pass += ok
        if not ok:
            fails.append(rec["seed"])
        oracle.evict_if_huge()
        if time.time() - last_log >= 30.0:
            last_log = time.time()
            print(f"  [{i + 1}/46] walt {t_walt:.1f}s compile {t_comp:.1f}s "
                  f"br {t_br * 1e3:.0f}ms wall {time.time() - t_start:.0f}s",
                  flush=True)

    print("\n=== 1. fixture suite (46 roots, σ-filtered worlds) ===")
    print(f"K1 parity: {k1_pass}/46" + (f"  FAILS: {fails}" if fails else ""))
    print(f"walt net wavefront (fresh, denominator): {t_walt:8.2f} s "
          f"(registered prior: {NET_WAVEFRONT_S:.1f} s; "
          f"old recursion baseline_ms sum: {baseline_tot:.1f} s)")
    print(f"compile_sigma total:                     {t_comp:8.2f} s "
          f"({t_comp / t_walt:.2f}x one net solve — P2; "
          f"{rows_tot / 1e6:.1f}M net rows)")
    print(f"br_solve FAST total:                     {t_br:8.3f} s "
          f"(P1 gate ≤0.5s: {'PASS' if t_br <= 0.5 else 'FAIL'}; "
          f"{t_walt / t_br:.0f}x vs fresh wavefront, "
          f"{NET_WAVEFRONT_S / t_br:.0f}x vs 8.5s prior, "
          f"{baseline_tot / t_br:.0f}x vs recursion)")
    print(f"  per fixture: p50 {_pct(br_each, 50):.2f} ms "
          f"(gate <2ms: {'PASS' if _pct(br_each, 50) < 2 else 'FAIL'})  "
          f"p95 {_pct(br_each, 95):.2f} ms  max {max(br_each):.2f} ms")
    print(f"br_solve REWALK total (no cache):        {t_rw:8.3f} s "
          f"({t_walt / t_rw:.1f}x vs fresh wavefront)")
    print(f"nodes total {nodes_tot / 1e6:.2f}M: "
          f"fast {t_br / nodes_tot * 1e9:.1f} ns/node, "
          f"rewalk {t_rw / nodes_tot * 1e9:.1f} ns/node "
          f"(net wavefront: {t_walt / nodes_tot * 1e9:.0f} ns/node ≈ "
          f"{rows_tot / t_walt / 1e3:.0f}k net rows/s)")
    return {"recs": {r["seed"]: r for r in recs}, "k1": (k1_pass, fails),
            "t_br": t_br}


def bench_stochastic(oracle, recs_by_seed, t_start) -> None:
    import hoyt as K

    pay = K.payoff_points()
    prof = K.StochasticProfile(uniform_fallback=True)
    print("\n=== 2. stochastic stress: uniform-over-legal profile "
          "(full support), worlds capped 512 ===")
    print("  (denominator per row: the same root's deterministic-σ tree)")
    walls, blows = [], []
    for seed in STRESS_SEEDS + (STRESS_CHUNK_SEED,):
        if time.time() - t_start > BUDGET_S - 180 and \
                seed == STRESS_CHUNK_SEED:
            print(f"  seed {seed}: SKIPPED (chunk case needs ~2.5 min; "
                  "budget guard)")
            continue
        root = _root_of(recs_by_seed[seed]["root"])
        worlds = _cap(root, _fixture_worlds(root, oracle), 512)
        weights = np.full(len(worlds), 1.0 / len(worlds), dtype=np.float64)
        sub = K.build_subgame(root, worlds, weights)
        table = K.compile_sigma(root, worlds, oracle)
        t0 = time.perf_counter()
        br = K.br_solve(sub, prof, pay)
        dt = time.perf_counter() - t0
        walls.append(dt)
        blows.append(br.n_nodes / table.n_nodes)
        print(f"  seed {seed}: N={len(worlds)}  det {table.n_nodes:>9,} → "
              f"stoch {br.n_nodes:>11,} nodes ({br.n_nodes / table.n_nodes:6.0f}x)  "
              f"wall {dt * 1e3:8.0f} ms  [{br.meta['mode']}]", flush=True)
        oracle.evict_if_huge()
    print(f"  blowup p50 {_pct(blows, 50):.0f}x max {max(blows):.0f}x — "
          f"walt-spec's 'stochastic fields cost x10^2-10^4' measured; "
          f"total {sum(walls):.1f}s")


def bench_h5(oracle, n_roots, t_start) -> None:
    import hoyt as K
    from arena.jud_play import JudPlay
    from champion.jud_net import load_jud_net
    from walt.bench import build_root_at_horizon
    from walt.worlds import enumerate_worlds

    jud = JudPlay(load_jud_net())
    pay = K.payoff_points()
    comp_ms, br_ms, rw_ms, nodes_l = [], [], [], []
    seed = 910000
    done = 0
    while done < n_roots and time.time() - t_start < BUDGET_S - 30:
        root = build_root_at_horizon(seed, 5, jud)
        seed += 1
        if root is None:
            continue
        worlds = enumerate_worlds(root)
        if len(worlds) == 0:
            continue
        worlds = _cap(root, worlds, 512)
        weights = np.full(len(worlds), 1.0 / len(worlds), dtype=np.float64)
        t0 = time.perf_counter()
        table = K.compile_sigma(root, worlds, oracle)
        comp_ms.append((time.perf_counter() - t0) * 1e3)
        sub = K.build_subgame(root, worlds, weights)
        t0 = time.perf_counter()
        br = K.br_solve(sub, table, pay)
        br_ms.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter()
        br2 = K.br_solve(sub, table, pay, rewalk=True)
        rw_ms.append((time.perf_counter() - t0) * 1e3)
        assert br2.best_move == br.best_move
        nodes_l.append(table.n_nodes)
        done += 1
        oracle.evict_if_huge()

    print(f"\n=== 3. H5-capped(512) BR ({done} roots, "
          "walt.bench.build_root_at_horizon(seed, 5, jud), u worlds) ===")
    p50 = _pct(br_ms, 50)
    print(f"br_solve FAST:   p50 {p50:6.2f} ms  p95 {_pct(br_ms, 95):6.2f} ms "
          f"(P5 gate <100ms: {'PASS' if p50 < 100 else 'FAIL'})")
    print(f"br_solve REWALK: p50 {_pct(rw_ms, 50):6.2f} ms  "
          f"p95 {_pct(rw_ms, 95):6.2f} ms (also "
          f"{'under' if _pct(rw_ms, 50) < 100 else 'OVER'} the P5 line "
          "without any cache)")
    print(f"compile_sigma:   p50 {_pct(comp_ms, 50):6.0f} ms  "
          f"p95 {_pct(comp_ms, 95):6.0f} ms (the one-time net pass a "
          "table amortizes)")
    print(f"nodes p50 {_pct(nodes_l, 50):,.0f}  max {max(nodes_l):,}")


def main() -> int:
    parser = argparse.ArgumentParser(description="hoyt bench (K3)")
    parser.add_argument("--h5-n", type=int, default=20)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    from walt.field import FieldOracle

    t_start = time.time()
    oracle = FieldOracle(device="cpu")
    print(f"bench_kernel: budget {BUDGET_S:.0f}s, numpy {np.__version__}",
          flush=True)

    out = bench_fixture_suite(oracle, t_start)
    bench_stochastic(oracle, out["recs"], t_start)
    bench_h5(oracle, args.h5_n, t_start)

    wall = time.time() - t_start
    k1_pass, fails = out["k1"]
    print(f"\ntotal wall {wall:.0f}s (budget {BUDGET_S:.0f}s"
          f"{' — OVER' if wall > BUDGET_S else ''});  "
          f"K1 {k1_pass}/46;  P1 {'PASS' if out['t_br'] <= 0.5 else 'FAIL'}")
    return 0 if (k1_pass == 46 and not fails) else 1


if __name__ == "__main__":
    sys.exit(main())
