# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Arena batches for the Book desk's registered studies: one batch is one
arm at one base seed, written as `meta.json` and `games.jsonl`; a queue runs
every registered batch through `scripts/cpu-slot`, which starts each as cores free,
skipping batches already written and dropping (not rerunning) a batch that
exits nonzero. The estimator (`pairs`, `contrast`, `diff`) is
`scenario.small_n`'s. `scenario.small_n` and `scenario.deviant_field` keep
the copies they were registered with.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from scenario.small_n import ROOT

POINTS = 250


def write_batch(arm: str, seed: int, games: int, out: str, seats: tuple, points: bool) -> dict:
    """Play `games` game pairs (2·games games, the arena's paired halves) of
    team A against team B, `seats` = (bid_a, play_a, bid_b, play_b), and write
    them to `out`. `points`: the book's game, first to 250 points."""
    from arena import cost
    from arena.engine import ArenaConfig
    from arena.match import run_match

    bid_a, play_a, bid_b, play_b = seats
    cfg = ArenaConfig(base_seed=seed, points_to_win=POINTS if points else None)
    start = cost.snapshot()
    result = run_match(bid_a=bid_a, bid_b=bid_b, play_a=play_a, play_b=play_b,
                       n_games=2 * games, cfg=cfg, fast_batching=True)
    spent = cost.since(start)  # wall, and the load-proof CPU seconds and instructions (#173)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"arm": arm, "seed": seed, "games_per_half": games, "wall_s": round(spent.pop("wall_s"), 1),
            **spent, "load_avg": list(os.getloadavg()),
            "a": f"{bid_a!r} / {play_a!r}", "b": f"{bid_b!r} / {play_b!r}"}
    peak = getattr(play_a, "peak_work", None)
    if peak is not None:
        meta["a_peak_work"] = peak   # the most Silas work at one decision of team A's
    lapse = getattr(play_b, "lapse", None)
    if lapse is not None:
        meta["b_lapse"] = lapse      # the field's decisions, tiles forgotten and stood in for (#172)
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    with open(outdir / "games.jsonl", "w") as f:
        for g in result.games:
            f.write(json.dumps({
                "game": g.game_idx, "a_team": g.a_team, "a_marks": g.a_marks, "b_marks": g.b_marks,
                "a_won": g.a_won, "a_points": g.points[g.a_team], "b_points": g.points[1 - g.a_team],
                "hands": [{"dealer": h.dealer, "bids": list(h.bids), "bidder": h.bidder, "bid": h.bid_value,
                           "decl": h.decl_id, "forced": h.forced, "made": h.made, "a_team": h.a_team,
                           "bidder_team": h.bidder_team, "points": list(h.team_points),
                           "hands": [list(x) for x in h.hands]} for h in g.hands]}) + "\n")
    return meta


def queue(module: str, root: str, jobs: Sequence[tuple[str, int]], workers: int, games: int,
          threads: int | None = None) -> None:
    """Run `python -m <module> arena --arm A --seed S` for every job through
    scripts/cpu-slot, `workers` at a time, skipping batches whose output
    exists; a nonzero exit is logged and the batch is left out. `threads` is
    the cores a batch declares to cpu-slot; by default the game-pair batches'
    measured share."""
    todo = [(a, s) for a, s in jobs if not (Path(root) / f"{a}-s{s}" / "meta.json").exists()]
    log = open(Path(root) / "queue.log", "a")
    lock = threading.Lock()

    def say(msg: str) -> None:
        with lock:
            log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            log.flush()
            print(msg, flush=True)

    say(f"queue: {len(todo)} batches to run, {workers} workers, {games} pairs a batch")
    # The cores a batch keeps busy, measured 2026-10-16 (#173): 1.0 at 1 pair, 1.5 at 2, 4.8 at 12.
    cores = threads or max(1, min(5, round(games / 2.5)))

    def work() -> None:
        while True:
            with lock:
                if not todo:
                    return
                arm, seed = todo.pop(0)
            out = Path(root) / f"{arm}-s{seed}"
            cmd = [str(ROOT / "scripts" / "cpu-slot"), "--threads", str(cores), "--", sys.executable, "-u", "-m", module,
                   "arena", "--arm", arm, "--seed", str(seed), "--games", str(games), "--out", str(out)]
            t0 = time.time()
            r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            status = "ok" if r.returncode == 0 else f"DROPPED exit {r.returncode}"
            say(f"{arm} s{seed}: {status} ({time.time() - t0:.0f}s incl. slot wait); {len(todo)} left")
            if r.returncode != 0:
                (out.parent / f"{arm}-s{seed}.dropped.log").write_text(r.stdout)

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say("queue: done")


def main(doc: str, module: str, arms: Sequence[str], seats: Callable[[str], tuple],
         jobs: Callable[[bool], list[tuple[str, int]]], summarize: Callable[[Sequence[str]], dict],
         games: int) -> None:
    """The command line every study shares: `arena`, `queue`, `summarize`."""
    import argparse

    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arena")
    p.add_argument("--arm", choices=arms, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--games", type=int, required=True, help="game pairs")
    p.add_argument("--out", required=True)
    p = sub.add_parser("queue")
    p.add_argument("--root", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="batches offered to cpu-slot at once; it starts them as cores free (#173)")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--games", type=int, default=games, help="game pairs a batch")
    p = sub.add_parser("summarize")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "arena":
        meta = write_batch(a.arm, a.seed, a.games, a.out, seats(a.arm), a.arm.startswith("P"))
        print(json.dumps(meta, indent=1))
    elif a.cmd == "queue":
        Path(a.root).mkdir(parents=True, exist_ok=True)
        queue(module, a.root, jobs(a.pilot), a.workers, a.games)
    else:
        r = summarize(a.runs)
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r["registered"], indent=1))
