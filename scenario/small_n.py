# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The book at small n (#154): do Winning 42's rules raise a bounded Zeb?

The book is written for a reader who cannot enumerate 512 worlds a decision:
its rules are a policy for bounded computation ([[book-assumptions]] A7).
Every arena block has compared the book against deployed Zeb at full
compute, and the one place the book has won is where it was asked to do the
job a rule does for a bounded player, breaking Zeb's near-ties in the
book's own game (53.2% [51.1, 55.2] of games to 250;
[[book-investigation-2026-10-14]]). This experiment hands the book to a
Zeb that samples fewer worlds and asks whether the rules raise it, and
whether they raise it more the fewer worlds it sees.

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Team B, the opponent in every arm, is deployed Zeb 0.3.0's plan
(`k0,n160,F=<zeb/net net>,f=16,d=posterior,r=uniform:2^22:0.28,t=margin`;
[[zeb-030-confirmation-2026-10-14]]), bidding by the arena's heuristic rule
at caution 0. Team A is the bounded player, the same plan at n worlds,
n in {16, 64, 160}, with the same bidding:
- **Z<n>**: the plan alone at n worlds (`walt:k0,n<n>,...`);
- **T<n>**: Zeb at n worlds with the book as its near-tie break
  (`book:zeb,zeb,w0.02/...`): where the deciding rule allows a tile within
  0.02 of Zeb's best by Zeb's q, the best such tile, else Zeb's tile. The
  form that won the book's game at full compute;
- **F<n>**: Zeb at n worlds with the book as a filter (`book:zeb,zeb/...`,
  the investigation's variant (c)): Zeb's best tile inside the deciding
  rule's set, Zeb's own tile where no rule fires;
- **A160**: Z160 at planner seed 1, the A/A arm.
At n = 160, Z160 is deployed Zeb against itself at seed 0, which reads
exactly 0 by construction; T160 is the investigation's t1 arm against
today's deployed opponent.

*Arms.* Marks (games to 7): Z16, T16, F16, Z64, T64, Z160, T160, F160,
A160. The book's game (250 points, each hand settled by the book's rule):
PZ16, PT16, PZ160, PT160.

*Pairing and the cluster.* Every arm plays the same base seeds with the
arena's paired halves (team A as team 0 and as team 1 over the same
deals). A game pair is (base seed, game index); its value is team A's
margin summed over its two games / 2. Every contrast is taken per game
pair over the pairs all its arms hold, and its 95% interval is the
percentile bootstrap over game pairs (10,000 resamples, seed 42).

*Primary.* D16 = T16 − Z16 in marks per game: what the book's rules are
worth to a Zeb that sees 16 worlds, as its near-tie break. **The book
predicts D16 > 0.** Read against
- the A/A, A160 − Z160, whose interval should cover 0;
- the bar: D16's interval excludes 0 above = the rules raise a bounded
  player, the book is worth the most where its reader lives; excludes 0
  below = at 16 worlds the rules cost even as a tie-break; else a null at
  this power, reported with its minimum detectable effect.

*Registered secondaries* (marks per game unless stated):
- the trend: D64 = T64 − Z64 and D160 = T160 − Z160; the prediction is
  D16 > D64 > D160, read on the paired differences D16 − D160 and D64 −
  D160 (each a contrast over the pairs the four arms hold);
- the filter: E16 = F16 − Z16 and E160 = F160 − Z160, and E16 − E160; the
  prediction is that the filter's cost shrinks as n falls;
- the cost of bounded compute itself: Z16 − Z160 and Z64 − Z160 (the
  margin against deployed Zeb lost to sampling fewer worlds), and whether
  the book repays it: T16 − Z160;
- each arm's margin against deployed Zeb (0 is a reference, not a null:
  the teams are not the same player except in Z160);
- the book's game: D16 and D160 on team A's share of games won and on
  settled points per game, and PT16 − PZ160.

*Size.* From the timing pilot (`PILOT` = seed 1690, excluded): 4 game
pairs of Z16, T16, Z160 and T160 took 29-42 s a batch at load average
about 40 (two batches at once through cpu-slot), so 12 pairs sit under
the 300 s cap. The primary's per-pair SD was 0.65 over 4 pairs, too few
to trust; the sizing takes SD 1.5 (the regulars' table saw 1.86 for an
interaction, and a paired difference of two arms on the same deals is
tighter), so 288 pairs give a half-width of 1.96*1.5/sqrt(288) = 0.17
marks/game and a minimum detectable D16 of 2.8*1.5/sqrt(288) = 0.25
(80% power, two-sided 5%). So:
- marks: `SEEDS` = 1701-1724, `GAMES` = 12 pairs a batch: **288 game
  pairs per arm**, nine arms (216 batches);
- 250 points: `POINT_SEEDS` = 1801-1816, 12 pairs a batch: 192 game
  pairs per arm, four arms (64 batches);
- two batches at once through `scripts/cpu-slot`, the jobs in seed-major
  order (every marks arm of a seed before the next seed, then the points).

*Batch-drop rule.* A batch that exits nonzero (an overrun exits 124) is
dropped, not rerun, and listed; its game pairs leave every contrast that
needs them. No effect is computed until every registered batch has run:
no early stop, no top-up. Every batch runs under `scripts/cpu-slot`
(niced, one of three shared slots, killed at 300 s).

*Fence.* Every value is Zeb's plan over the mover's own cell, at fewer
worlds; the book's rules read the frame alone ([[strategy-fusion-fence]]).

    python -u -m scenario.small_n arena --arm T16 --seed S --games G --out DIR
    python -u -m scenario.small_n queue --root DIR [--workers 2] [--pilot]
    python -u -m scenario.small_n summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from scenario.planner import Reading

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / "zeb" / "net" / "margin_d250k_h128x64.fnet"
DEPLOYED = 160                # Zeb 0.3.0's worlds
NS = (16, 64, 160)
FLOOR = 16
READING = Reading("uniform", 1 << 22, 0.28)
WITHIN = 0.02                 # the near-tie band of the investigation's t1
KINDS = ("Z", "T", "F")
MARK_ARMS = ("Z16", "T16", "F16", "Z64", "T64", "Z160", "T160", "F160", "A160")
POINT_ARMS = ("PZ16", "PT16", "PZ160", "PT160")
ARMS = MARK_ARMS + POINT_ARMS
POINTS = 250
PILOT = (1690,)               # the timing pilot, excluded
GAMES = 12                    # game pairs per batch
SEEDS = tuple(range(1701, 1725))         # marks: 24 batches per arm, 288 game pairs
POINT_SEEDS = tuple(range(1801, 1817))   # 250 points: 16 batches per arm, 192 game pairs
RESAMPLES = 10_000


def player(worlds: int = DEPLOYED, seed: int = 0):
    """Zeb 0.3.0's plan at `worlds` worlds, as an arena play policy."""
    from arena.walt_play import WaltPlay
    from scenario.field import FieldNet

    net = FieldNet.from_bytes(NET.read_bytes())
    return WaltPlay(level=0, n_outer=worlds, floor=FLOOR, field=net, seed=seed,
                    draw="posterior", reading=READING, tie="margin")


def parse_arm(arm: str) -> tuple[bool, str, int]:
    """(points game, kind, worlds)."""
    if arm not in ARMS:
        raise ValueError(f"no arm {arm}")
    name = arm.removeprefix("P")
    return arm.startswith("P"), name[0], int(name[1:])


def seats(arm: str):
    """(bid_a, play_a, bid_b, play_b) for an arm."""
    from arena.bidders import HeuristicBidder
    from arena.book_play import BookPlay

    _, kind, worlds = parse_arm(arm)
    if kind == "A":
        play_a = player(worlds, seed=1)
    elif kind == "Z":
        play_a = player(worlds)
    elif kind == "T":
        play_a = BookPlay("zeb", "zeb", player(worlds), within=WITHIN)
    else:
        play_a = BookPlay("zeb", "zeb", player(worlds))
    return HeuristicBidder(), play_a, HeuristicBidder(), player()


def arena(arm: str, seed: int, games: int, out: str) -> dict:
    """One batch: `games` game pairs (2·games games) of team A against team B."""
    from arena.engine import ArenaConfig
    from arena.match import run_match

    bid_a, play_a, bid_b, play_b = seats(arm)
    cfg = ArenaConfig(base_seed=seed, points_to_win=POINTS if arm.startswith("P") else None)
    t0 = time.time()
    result = run_match(bid_a=bid_a, bid_b=bid_b, play_a=play_a, play_b=play_b,
                       n_games=2 * games, cfg=cfg, fast_batching=True)
    wall = time.time() - t0
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"arm": arm, "seed": seed, "games_per_half": games, "wall_s": round(wall, 1),
            "load_avg": list(os.getloadavg()), "a": f"{bid_a!r} / {play_a!r}", "b": f"{bid_b!r} / {play_b!r}"}
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


# -- the summary ----------------------------------------------------------------------

def _games(run: str) -> list[dict]:
    with open(Path(run) / "games.jsonl") as f:
        return [json.loads(line) for line in f]


def _meta(run: str) -> dict:
    return json.loads((Path(run) / "meta.json").read_text())


def pairs(runs: Sequence[str], key: str) -> dict[tuple[int, int], float]:
    """(base seed, game) → team A's value per game over the pair's two games:
    margin in marks ("marks") or settled points ("points"), or games won ("won", 0..1)."""
    out: dict[tuple[int, int], float] = {}
    for run in runs:
        seed = _meta(run)["seed"]
        for g in _games(run):
            k = (seed, g["game"])
            v = float(g["a_won"]) if key == "won" else g[f"a_{key}"] - g[f"b_{key}"]
            out[k] = out.get(k, 0.0) + v / 2
    return out


def contrast(arms: dict[str, dict], terms: Sequence[tuple[float, str]], seed: int = 42) -> dict:
    """Σ coef·arm per game pair over the pairs every term's arm holds, with
    its 95% percentile bootstrap over game pairs."""
    if any(a not in arms for _, a in terms):
        return {}
    common = sorted(set.intersection(*(set(arms[a]) for _, a in terms)))
    if not common:
        return {}
    d = np.array([sum(c * arms[a][k] for c, a in terms) for k in common])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sd = float(d.std(ddof=1)) if len(d) > 1 else math.nan
    return {"game_pairs": len(common), "mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "sd_per_pair": sd, "mde_80": 2.8 * sd / math.sqrt(len(d)) if len(d) > 1 else math.nan}


def diff(a: str, b: str) -> list[tuple[float, str]]:
    return [(1.0, a), (-1.0, b)]


def summarize(runs: Sequence[str]) -> dict:
    by: dict[str, list[str]] = {}
    for run in runs:
        by.setdefault(_meta(run)["arm"], []).append(run)
    marks = {a: pairs(rs, "marks") for a, rs in by.items() if not a.startswith("P")}
    won = {a: pairs(rs, "won") for a, rs in by.items() if a.startswith("P")}
    pts = {a: pairs(rs, "points") for a, rs in by.items() if a.startswith("P")}
    out: dict = {"arms": {}, "registered": {}}
    for arm, rs in sorted(by.items()):
        games = [g for r in rs for g in _games(r)]
        hands = [h for g in games for h in g["hands"]]
        made = {s: [h["made"] for h in hands if (h["bidder_team"] == h["a_team"]) == (s == "a")] for s in "ab"}
        out["arms"][arm] = {
            "batches": [{x: _meta(r)[x] for x in ("seed", "wall_s", "load_avg")} for r in rs],
            "games": len(games), "hands": len(hands),
            "a_bids": len(made["a"]) / max(len(hands), 1),
            "made_bidding": {s: sum(v) / len(v) if v else None for s, v in made.items()},
            "a_margin": contrast(marks if arm in marks else won, [(1.0, arm)]),
        }
    reg = out["registered"]
    reg["primary D16 = T16 - Z16, marks/game"] = contrast(marks, diff("T16", "Z16"))
    reg["A/A = A160 - Z160, marks/game"] = contrast(marks, diff("A160", "Z160"))
    reg["D64 = T64 - Z64, marks/game"] = contrast(marks, diff("T64", "Z64"))
    reg["D160 = T160 - Z160, marks/game"] = contrast(marks, diff("T160", "Z160"))
    reg["trend D16 - D160, marks/game"] = contrast(marks, diff("T16", "Z16") + diff("Z160", "T160"))
    reg["trend D64 - D160, marks/game"] = contrast(marks, diff("T64", "Z64") + diff("Z160", "T160"))
    reg["filter E16 = F16 - Z16, marks/game"] = contrast(marks, diff("F16", "Z16"))
    reg["filter E160 = F160 - Z160, marks/game"] = contrast(marks, diff("F160", "Z160"))
    reg["filter trend E16 - E160, marks/game"] = contrast(marks, diff("F16", "Z16") + diff("Z160", "F160"))
    reg["bounded compute Z16 - Z160, marks/game"] = contrast(marks, diff("Z16", "Z160"))
    reg["bounded compute Z64 - Z160, marks/game"] = contrast(marks, diff("Z64", "Z160"))
    reg["the book repays it? T16 - Z160, marks/game"] = contrast(marks, diff("T16", "Z160"))
    reg["250: D16 on team A's share of games won"] = contrast(won, diff("PT16", "PZ16"))
    reg["250: D160 on share of games won"] = contrast(won, diff("PT160", "PZ160"))
    reg["250: D16 on settled points/game"] = contrast(pts, diff("PT16", "PZ16"))
    reg["250: D160 on settled points/game"] = contrast(pts, diff("PT160", "PZ160"))
    reg["250: PT16 - PZ160 on share of games won"] = contrast(won, diff("PT16", "PZ160"))
    return out


# -- the queue ------------------------------------------------------------------------

def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    if pilot:
        return [(a, s) for s in PILOT for a in ("Z16", "T16", "Z160", "T160")]
    marks = [(a, s) for s in SEEDS for a in MARK_ARMS]
    points = [(a, s) for s in POINT_SEEDS for a in POINT_ARMS]
    return marks + points


def queue(root: str, workers: int, pilot: bool, games: int) -> None:
    """Run every registered batch through scripts/cpu-slot, `workers` at a time,
    skipping batches whose output exists; a nonzero exit is logged and the
    batch is left out (the batch-drop rule)."""
    todo = [(a, s) for a, s in jobs(pilot) if not (Path(root) / f"{a}-s{s}" / "meta.json").exists()]
    log = open(Path(root) / "queue.log", "a")
    lock = threading.Lock()

    def say(msg: str) -> None:
        with lock:
            log.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            log.flush()
            print(msg, flush=True)

    say(f"queue: {len(todo)} batches to run, {workers} workers, {games} pairs a batch")

    def work() -> None:
        while True:
            with lock:
                if not todo:
                    return
                arm, seed = todo.pop(0)
            out = Path(root) / f"{arm}-s{seed}"
            cmd = [str(ROOT / "scripts" / "cpu-slot"), "--", sys.executable, "-u", "-m", "scenario.small_n",
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arena")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--games", type=int, required=True, help="game pairs")
    p.add_argument("--out", required=True)
    p = sub.add_parser("queue")
    p.add_argument("--root", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--games", type=int, default=GAMES, help="game pairs a batch")
    p = sub.add_parser("summarize")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "arena":
        print(json.dumps(arena(a.arm, a.seed, a.games, a.out), indent=1))
    elif a.cmd == "queue":
        Path(a.root).mkdir(parents=True, exist_ok=True)
        queue(a.root, a.workers, a.pilot, a.games)
    else:
        r = summarize(a.runs)
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r["registered"], indent=1))


if __name__ == "__main__":
    main()
