"""hoyt/refsweep.py — the cap-ledger cascade for CFR reference solves.

Jason's directive (2026-07-18): perf work optimizes 4th-play eval count per
wall clock, and cap results are tolerated — early-cap games are an
interesting artifact to analyze on their own and are available in bulk. The
pattern is the gomoku VCT cascade labeler (gomoku wiki:
topics/vct-cascade-labeler.md + topics/mega-vct-solver.md): the solver
returns result-OR-cap as first-class verdicts, every item gets an explicit
ledger row (no absence-as-state), and a budget ladder deepens only the
shrinking capped-survivor tail — the deepening curve is itself the
analyzable artifact. 42's cap is STRONGER than gomoku's boolean hit_cap:
a CFR stop at any point is a QUANTIFIED verdict — the average profile plus
its exactly-measured single-seat BR gap (priced by exact BR). A gap_capped
row is a usable reference at gap g, not a failure.

Rung ladder (defaults; --rungs "iters:gap:wall:slots,..." to override):

    rung 0: iters   80, gap 0.05, wall  90 s, slots  32M
    rung 1: iters  240, gap 0.05, wall 600 s, slots  64M
    rung 2: iters 1000, gap 0.05, wall  inf, slots 128M

Each rung processes ONLY the prior rung's non-converged survivors. iters
rises with the rungs only as a backstop: measured (perf-log 18e) iteration
count to gap 0.05 is world-scale-invariant (<= 40 from 10 to 33,740 true
worlds), so wall and slot budgets are the real ladder. Wall budgets are
quiet-box-equivalent; under contention the same rung caps earlier and the
survivor simply ladders up.

Verdicts, one JSONL row per (root, rung) entered:

    converged    measured gap <= target_gap
    gap_capped   wall or iters hit; the row carries measured gap, value,
                 rent — a USABLE reference at that gap (capped_by_wall
                 distinguishes the binding constraint)
    slot_capped  a wave blew the rung's slot budget (KernelMemoryError);
                 no profile at this rung — the next rung's budget retries
    error        anything else, message kept; carried forward like a cap
                 (retried at the next rung, never silently dropped)

Resume unions every rung*_shard*.jsonl in --outdir and is verdict-aware: a
converged row retires the root; any other verdict is done for ITS rung and
a survivor for the next; a (root, rung) with a row is never re-entered.

Dispatch: one shared queue ordered by descending n_worlds_sigma, pull-based
— every worker gets the full rung queue and claims one root at a time via
atomic claim-file creation. Balance comes from the PULL, not from cost
precision: paired arms measured static partitions (the round-robin snake)
at 1.16-1.18x worse rung-0 makespan, and a sharper cost key (banked wall_s,
rank-corr 0.889) actually LOST to sigma order — exact ranking front-loads
the true monsters into simultaneous residency and pays ~14% per-seed
contention inflation, while sigma's noisy ranking decorrelates the heavy
phases (perf-log 19i; the old snake's stagger insight, reborn inside the
pull). A worker that dies mid-solve orphans that root's claim for the run;
the resume retries it, exactly as it retried a dead shard's unfinished
queue.

CLI (run from the repo root; the driver spawns per-shard subprocess
workers, torch pinned to 1 thread, heartbeat every 30 s):

    python -u -m hoyt.refsweep --workers 8 --cap 256      # full sweep
    python -u -m hoyt.refsweep --dry-run                  # plan, no solving
    python -u -m hoyt.refsweep --seeds 555038,555181 --workers 1   # smoke

--workers defaults to a memory-aware count (measured ~7 GiB peak per
cap-256 worker at 32M slots, perf-log 18g, scaled by the rung's slot
budget); an explicit --workers is honored with a warning when it exceeds
the memory cap. Outputs land in --outdir (default scratch/refsweep):
per-shard ledgers, merged reference_h4_cap<cap>.jsonl (best verdict per
seed), and the final report — deepening curve plus the headline metric,
**H4 evals/hour** (converged + gap_capped rows are evals; slot_capped and
error rows are not).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

VERDICT_RANK = {"converged": 0, "gap_capped": 1, "slot_capped": 2, "error": 3}
USABLE = ("converged", "gap_capped")
BR_EVERY = 10            # gap-measure cadence (18o: denser is affordable
#                          with gap_exit — intermediates price ~1 seat)
THREADS = 4              # numba threads per worker for the fused iterate
#                          (P13). Measured on the 20-root paired sweep,
#                          5 workers, M5 Max: rung-0 wall 279.1 (t1) ->
#                          245.9 (t2) -> 221.8 (t4) -> 215.4 (t8); t8's
#                          gain over t4 is inside the +-4% noise floor at
#                          40 threads on 18 cores, so 4 is the default.
#                          Bitwise identical at any thread count.
GIB_PER_32M = 7.0        # measured cap-256 worker peak RSS at 32M slots (18g)


@dataclass(frozen=True)
class Rung:
    iters: int
    target_gap: float
    wall_budget_s: float | None       # None = unlimited
    slot_budget: int

    def spec(self) -> str:
        w = "inf" if self.wall_budget_s is None else f"{self.wall_budget_s:g}"
        return f"{self.iters}:{self.target_gap:g}:{w}:{self.slot_budget}"


DEFAULT_RUNGS = (
    Rung(80, 0.05, 90.0, 32_000_000),
    Rung(240, 0.05, 600.0, 64_000_000),
    Rung(1000, 0.05, None, 128_000_000),
)


def parse_rungs(spec: str) -> tuple[Rung, ...]:
    """'iters:gap:wall:slots[,...]'; wall 'inf' or 'none' = unlimited."""
    out = []
    for part in spec.split(","):
        f = part.strip().split(":")
        if len(f) != 4:
            raise ValueError(
                f"rung spec must be iters:gap:wall:slots, got {part!r}")
        wall = None if f[2].lower() in ("inf", "none") else float(f[2])
        out.append(Rung(int(f[0]), float(f[1]), wall, int(float(f[3]))))
    if not out:
        raise ValueError("empty rung ladder")
    return tuple(out)


# =========================================================================== #
#  pure ladder logic (unit-tested without solving)                            #
# =========================================================================== #

def dispatch_order(sized_seeds: list[tuple[int, int]]) -> list[int]:
    """[(seed, n_worlds_sigma)] -> ONE biggest-first queue shared by every
    worker. Biggest-first because a long root dispatched late is pure
    tail: the fleet idles behind it. Deterministic; ties break by seed."""
    return [s for s, _ in sorted(sized_seeds, key=lambda t: (-t[1], t[0]))]


def claim(path: Path) -> bool:
    """Atomically claim one (root, rung) for this run; False = another
    worker got it. O_CREAT|O_EXCL on a local fs is the whole protocol."""
    try:
        with open(path, "x"):
            return True
    except FileExistsError:
        return False


def claims_dir(outdir: Path, rung_idx: int) -> Path:
    return Path(outdir) / f"rung{rung_idx}_claims"


def next_rung(rows_by_rung: dict[int, dict], n_rungs: int) -> int | None:
    """The rung this root should enter next, or None when retired.

    A converged row at any rung retires it; otherwise it enters the rung
    after the deepest rung it has a row for (a (root, rung) with a row is
    never re-entered — caps, slot blowups AND errors all ladder up);
    ladder exhausted -> None."""
    if not rows_by_rung:
        return 0
    if any(r.get("verdict") == "converged" for r in rows_by_rung.values()):
        return None
    nxt = max(rows_by_rung) + 1
    return nxt if nxt < n_rungs else None


def best_row(rows: list[dict]) -> dict:
    """Merge precedence: converged beats gap_capped beats slot_capped beats
    error; among usable rows the smaller measured gap wins (deeper rung
    breaks ties); among unusable rows the deepest attempt wins."""
    def key(r):
        rank = VERDICT_RANK.get(r.get("verdict"), 9)
        gap = r.get("final_gap")
        return (rank, gap if (rank <= 1 and gap is not None) else 0.0,
                -r.get("rung", 0))
    return min(rows, key=key)


def load_ledger(outdir: Path) -> dict[int, dict[int, dict]]:
    """seed -> rung -> row, union over every shard file. First row wins per
    (seed, rung): a shard killed mid-write can leave a torn last line
    (skipped) and a resume never duplicates a rung it can see."""
    led: dict[int, dict[int, dict]] = {}
    for f in sorted(Path(outdir).glob("rung*_shard*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                      # torn tail of a killed shard
            led.setdefault(int(row["seed"]), {}) \
               .setdefault(int(row["rung"]), row)
    return led


def default_workers(slot_budget: int = 32_000_000) -> int:
    """Memory-aware worker count: measured ~7 GiB peak per cap-256 worker
    at 32M slots (perf-log 18g), scaled linearly with the rung's slot
    budget, keeping 25% of RAM free."""
    try:
        ram_gib = (os.sysconf("SC_PAGE_SIZE")
                   * os.sysconf("SC_PHYS_PAGES")) / 2**30
    except (ValueError, OSError, AttributeError):
        return 4
    est = GIB_PER_32M * slot_budget / 32e6
    return max(1, min(os.cpu_count() or 4, int(ram_gib * 0.75 / est)))


# =========================================================================== #
#  per-root work (worker side; heavy imports stay out of the driver)          #
# =========================================================================== #

def _load_evalset(path: Path) -> dict[int, dict]:
    return {r["seed"]: r for r in map(json.loads, open(path))}


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


def solve_root(rec: dict, rung: Rung, rung_idx: int, cap: int,
               oracle, threads: int | None = None,
               engine: str = "fused") -> dict:
    """One (root, rung) -> one ledger row. The pipeline mirrors the ad-hoc
    2026-07-18 sweep worker (enumerate_worlds -> sigma_consistent ->
    deterministic world cap -> build_subgame -> compile_sigma -> walt BR ->
    cfr_solve); caps and failures are first-class rows instead of inline
    retries. cfr_reference_value = res.value (profile_value survives only
    as the deterministic 1-in-20 audit, P9); non-CFR phase walls land in
    row["phase_s"] (perf-log 18m: the oracle bucket was un-instrumented)."""
    import resource

    import numpy as np

    import hoyt as K
    from hoyt.cfr import cfr_solve
    from walt.field import sigma_consistent
    from walt.grade import _make_moves_filter, _world_cap_rng
    from walt.worlds import enumerate_worlds

    root = _root_of(rec["root"])
    row: dict = {"seed": rec["seed"], "rung": rung_idx,
                 "rung_spec": rung.spec(),
                 "me_declares": root.me % 2 == root.bidder % 2}
    t0 = time.perf_counter()
    ph: dict = {}                     # non-CFR phase walls (perf-log 18m)
    try:
        pay = K.payoff_points()
        t1 = time.perf_counter()
        worlds = enumerate_worlds(root)
        keep = sigma_consistent(root, worlds, oracle,
                                _make_moves_filter(root))
        ph["worlds"] = round(time.perf_counter() - t1, 2)
        w = worlds[keep] if keep.any() else worlds
        row["worlds_true"] = int(len(w))
        if len(w) > cap:
            idx = _world_cap_rng(root, cap).choice(len(w), size=cap,
                                                   replace=False)
            w = w[np.sort(idx)]
        row["worlds_used"] = int(len(w))
        wt = np.full(len(w), 1.0 / len(w))
        sub = K.build_subgame(root, w, wt)
        t1 = time.perf_counter()
        tab = K.compile_sigma(root, w, oracle)
        ph["compile_sigma"] = round(time.perf_counter() - t1, 2)
        t1 = time.perf_counter()
        br = K.br_solve(sub, tab, pay, slot_budget=rung.slot_budget)
        ph["walt_br"] = round(time.perf_counter() - t1, 2)
        if engine == "metal":
            # the GPU searcher (metal_hoyt/DESIGN.md): fp32 iterate +
            # steering gap on the GPU, returned gap/value certified fp64
            # by hoyt's exact BR — same verdict class as the fused lane
            from metal_hoyt import cfr_solve_metal
            res = cfr_solve_metal(sub, pay, iters=rung.iters,
                                  target_gap=rung.target_gap,
                                  br_every=BR_EVERY,
                                  wall_budget_s=rung.wall_budget_s,
                                  slot_budget=rung.slot_budget,
                                  threads=threads)
        else:
            res = cfr_solve(sub, pay, iters=rung.iters,
                            target_gap=rung.target_gap, br_every=BR_EVERY,
                            gap_exit=True,
                            wall_budget_s=rung.wall_budget_s,
                            slot_budget=rung.slot_budget,
                            threads=threads)
        # the reference value IS the solve's self-play value (measured
        # identical on all 200 banked rows, delta 0.0); profile_value's
        # full stochastic re-walk survives only as a 1-in-20 audit (P9)
        ref_val = res.value
        if rec["seed"] % 20 == 0:
            t1 = time.perf_counter()
            pv = K.profile_value(sub, res.profile, pay,
                                 slot_budget=rung.slot_budget)
            ph["audit_profile_value"] = round(time.perf_counter() - t1, 2)
            row["audit_pv_delta"] = abs(pv - res.value)
            if row["audit_pv_delta"] > 1e-9:
                raise AssertionError(
                    f"profile_value audit failed: {pv} vs res.value "
                    f"{res.value} (P9 — put per-root pricing back)")
        sign = 1.0 if row["me_declares"] else -1.0
        row.update(
            verdict=("converged" if res.gap <= rung.target_gap
                     else "gap_capped"),
            capped_by_wall=bool(res.capped),
            walt_vs_jud_value=round(br.value, 6),
            cfr_reference_value=round(ref_val, 6),
            rent_hero_orient=round(sign * (br.value - ref_val), 6),
            value_selfplay=round(res.value, 6),
            iters_run=res.iters_run,
            final_gap=round(res.gap, 6),
            trace=[(i, round(g, 6)) for i, g in res.trace],
            timings={k: round(v, 2) for k, v in res.timings.items()},
            phase_s=ph,
        )
    except K.KernelMemoryError as e:
        row.update(verdict="slot_capped", error=repr(e)[:300])
    except Exception as e:  # noqa: BLE001 — every failure is a ledger row
        row.update(verdict="error", error=repr(e)[:300])
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    row["maxrss_gib"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    row["loadavg"] = round(os.getloadavg()[0], 1)
    return row


def _worker_main(a) -> int:
    import torch
    torch.set_num_threads(1)
    from walt.field import FieldOracle

    ladder = parse_rungs(a.rungs) if a.rungs else DEFAULT_RUNGS
    rung = ladder[a.worker_rung]
    recs = _load_evalset(Path(a.evalset))
    oracle = FieldOracle(net_path=a.net, device="cpu")
    out = Path(a.outdir) / f"rung{a.worker_rung}_shard{a.shard}.jsonl"
    cdir = claims_dir(a.outdir, a.worker_rung)
    seeds = [int(s) for s in a.seeds.split(",")]
    solved = 0
    with open(out, "a") as fh:
        for seed in seeds:
            if not claim(cdir / str(seed)):
                continue
            rec = recs[seed]
            print(f"seed {seed} start "
                  f"(worlds_sigma={rec['n_worlds_sigma']})", flush=True)
            row = solve_root(rec, rung, a.worker_rung, a.cap, oracle,
                             threads=a.threads or None, engine=a.engine)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            solved += 1
            print(f"seed {seed} {row['verdict']} wall={row['wall_s']}s "
                  f"gap={row.get('final_gap')}", flush=True)
    print(f"worker done: {solved} roots -> {out}", flush=True)
    return 0


# =========================================================================== #
#  driver                                                                     #
# =========================================================================== #

def _heartbeat(procs, outdir: Path, rung_idx: int, entrants: set[int],
               t0: float) -> None:
    """>= one line per 30 s while workers live (60 s silence is a bug)."""
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        counts: dict[str, int] = {}
        for f in outdir.glob(f"rung{rung_idx}_shard*.jsonl"):
            for line in open(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("seed") in entrants:
                    v = row.get("verdict", "?")
                    counts[v] = counts.get(v, 0) + 1
        alive = sum(1 for p in procs if p.poll() is None)
        mix = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"[{time.time() - t0:6.0f}s] rung {rung_idx}: "
              f"{sum(counts.values())}/{len(entrants)} rows ({mix}) "
              f"alive {alive}/{len(procs)} "
              f"loadavg {os.getloadavg()[0]:.1f}", flush=True)


def _print_plan(recs, ledger, ladder, a) -> None:
    print("rung ladder:")
    for r, rung in enumerate(ladder):
        wall = "inf" if rung.wall_budget_s is None else f"{rung.wall_budget_s:g}s"
        print(f"  rung {r}: iters={rung.iters} gap<={rung.target_gap:g} "
              f"wall={wall} slots={rung.slot_budget:,} "
              f"(mem-cap workers <= {default_workers(rung.slot_budget)})")
    dist: dict = {}
    for s in recs:
        nr = next_rung(ledger.get(s, {}), len(ladder))
        dist[nr] = dist.get(nr, 0) + 1
    print(f"resume state ({len(recs)} roots): " + ", ".join(
        ("retired" if k is None else f"enter rung {k}") + f"={v}"
        for k, v in sorted(dist.items(), key=lambda t: (t[0] is None, t[0]))))
    for r in range(len(ladder)):
        entrants = [s for s in recs
                    if next_rung(ledger.get(s, {}), len(ladder)) == r]
        if not entrants:
            continue
        nw = _rung_workers(a, ladder[r], len(entrants))
        sized = [(s, recs[s]["n_worlds_sigma"]) for s in entrants]
        q = dispatch_order(sized)
        print(f"rung {r} dispatch queue ({len(entrants)} entrants, "
              f"{nw} workers, biggest-first by n_worlds_sigma):")
        print(f"  head {q[:6]}{'...' if len(q) > 6 else ''}")
        break                      # only the next actionable rung is plannable
    print("dry run: no solving.")


def _rung_workers(a, rung: Rung, n_entrants: int) -> int:
    mem_cap = default_workers(rung.slot_budget)
    if a.workers_explicit and a.workers > mem_cap:
        print(f"warning: --workers {a.workers} exceeds the memory-aware cap "
              f"{mem_cap} for slot_budget {rung.slot_budget:,}; honoring it",
              flush=True)
        return min(a.workers, n_entrants)
    return min(a.workers, mem_cap, n_entrants)


def _report(recs, ledger, ladder, outdir: Path, a, t0: float,
            before_keys: set) -> int:
    merged = {s: best_row(list(rows.values()))
              for s, rows in ledger.items() if s in recs and rows}
    mpath = outdir / f"reference_h4_cap{a.cap}.jsonl"
    with open(mpath, "w") as fh:
        for s in sorted(merged):
            fh.write(json.dumps(merged[s]) + "\n")

    n = len(recs)
    by_v: dict[str, int] = {}
    for r in merged.values():
        by_v[r["verdict"]] = by_v.get(r["verdict"], 0) + 1
    print(f"\nmerged {len(merged)}/{n} roots -> {mpath}")
    print("best-verdict mix: " + ", ".join(
        f"{k}={by_v.get(k, 0)}" for k in VERDICT_RANK))

    print("deepening curve:")
    cum_wall = 0.0
    for r in range(len(ladder)):
        rows_r = [rows[r] for s, rows in ledger.items()
                  if s in recs and r in rows]
        if not rows_r:
            continue
        cum_wall += sum(row.get("wall_s", 0.0) for row in rows_r)
        conv = sum(1 for s, rows in ledger.items() if s in recs and any(
            q <= r and row.get("verdict") == "converged"
            for q, row in rows.items()))
        print(f"  rung {r}: entered {len(rows_r):4d}  "
              f"converged-cum {conv:4d}/{n} ({100 * conv / n:5.1f}%)  "
              f"root-wall-cum {cum_wall:,.0f}s")

    usable = sum(1 for r in merged.values() if r["verdict"] in USABLE)
    fresh_usable = sum(
        1 for s, rows in ledger.items() if s in recs
        for q, row in rows.items()
        if (s, q) not in before_keys and row.get("verdict") in USABLE)
    elapsed = time.time() - t0
    print(f"\nH4 evals (usable references, merged): {usable}/{n} "
          f"(converged {by_v.get('converged', 0)}, "
          f"gap_capped {by_v.get('gap_capped', 0)})")
    if fresh_usable and elapsed > 1:
        print(f"H4 evals/hour (this run): "
              f"{3600 * fresh_usable / elapsed:.1f} "
              f"({fresh_usable} usable rows in {elapsed:,.0f}s wall)")
    else:
        print("H4 evals/hour: no new usable rows this run "
              "(resume with nothing to do?)")
    return 0


def _driver_main(a) -> int:
    ladder = parse_rungs(a.rungs) if a.rungs else DEFAULT_RUNGS
    evalset = Path(a.evalset)
    recs = _load_evalset(evalset)
    if a.seeds:
        want = [int(s) for s in a.seeds.split(",")]
        missing = [s for s in want if s not in recs]
        if missing:
            raise SystemExit(f"seeds not in {evalset}: {missing}")
        recs = {s: recs[s] for s in want}
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ledger = load_ledger(outdir)
    if a.dry_run:
        _print_plan(recs, ledger, ladder, a)
        return 0

    t0 = time.time()
    before_keys = {(s, r) for s, rows in ledger.items() for r in rows}
    if before_keys:
        print(f"resume: {len(before_keys)} (root, rung) rows already in "
              f"the ledger", flush=True)

    for r, rung in enumerate(ladder):
        entrants = [s for s in recs
                    if next_rung(ledger.get(s, {}), len(ladder)) == r]
        if not entrants:
            continue
        nw = _rung_workers(a, rung, len(entrants))
        sized = [(s, recs[s]["n_worlds_sigma"]) for s in entrants]
        queue = dispatch_order(sized)
        cdir = claims_dir(outdir, r)
        if cdir.exists():
            shutil.rmtree(cdir)         # claims are per-run; rows are the ledger
        cdir.mkdir(parents=True)
        print(f"rung {r} [{rung.spec()}]: {len(entrants)} entrants, "
              f"{nw} workers pulling one longest-first queue", flush=True)
        procs = []
        for w in range(nw):
            log = open(outdir / f"rung{r}_shard{w}.log", "a")
            cmd = [sys.executable, "-u", "-m", "hoyt.refsweep", "--worker",
                   "--worker-rung", str(r), "--shard", str(w),
                   "--seeds", ",".join(map(str, queue)),
                   "--cap", str(a.cap), "--outdir", str(outdir),
                   "--evalset", str(evalset), "--net", a.net,
                   "--threads", str(a.threads), "--engine", a.engine]
            if a.rungs:
                cmd += ["--rungs", a.rungs]
            procs.append(subprocess.Popen(cmd, stdout=log,
                                          stderr=subprocess.STDOUT))
        _heartbeat(procs, outdir, r, set(entrants), t0)
        bad = [p.returncode for p in procs if p.returncode != 0]
        if bad:
            print(f"rung {r}: {len(bad)} worker(s) exited nonzero {bad} — "
                  f"finished rows are in the ledger; a resume retries the "
                  f"rest", flush=True)
        ledger = load_ledger(outdir)

    return _report(recs, ledger, ladder, outdir, a, t0, before_keys)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -u -m hoyt.refsweep",
        description="cap-ledger cascade over the frozen H4 evalset")
    ap.add_argument("--workers", type=int, default=0,
                    help="worker processes (0 = memory-aware default)")
    ap.add_argument("--cap", type=int, default=256,
                    help="world cap per root (deterministic subsample)")
    ap.add_argument("--outdir", default="scratch/refsweep",
                    help="ledger + merged output directory")
    ap.add_argument("--rungs", default="",
                    help="ladder override: iters:gap:wall:slots,... "
                         "(wall 'inf' = unlimited)")
    ap.add_argument("--evalset", default="hoyt/evalset_h4_v1.jsonl")
    ap.add_argument("--net", default="champion/jud_net.pt")
    ap.add_argument("--seeds", default="",
                    help="restrict to these seeds (smoke runs)")
    ap.add_argument("--engine", choices=("fused", "metal"), default="fused",
                    help="CFR engine per root: fused (CPU numba, default) "
                         "or metal (GPU searcher, fp64-certified — "
                         "metal_hoyt/DESIGN.md)")
    ap.add_argument("--threads", type=int, default=THREADS,
                    help="numba threads per worker for the fused iterate "
                         "(P13; 0 = single-threaded kernels). Bitwise "
                         "identical either way; compose workers x threads "
                         "against the core budget")
    ap.add_argument("--dry-run", action="store_true",
                    help="print ladder + resume state + shard plan, no solving")
    # worker mode (spawned by the driver; not a user surface)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--worker-rung", type=int, default=0,
                    help=argparse.SUPPRESS)
    ap.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    a.workers_explicit = a.workers > 0
    if not a.workers_explicit:
        a.workers = default_workers()
    if a.worker:
        return _worker_main(a)
    return _driver_main(a)


if __name__ == "__main__":
    sys.exit(main())
