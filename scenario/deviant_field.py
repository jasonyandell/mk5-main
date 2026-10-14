# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The named deviant field (#152): the book's bidding against the overbidders
it was written for.

Winning 42's bidding is written for a population with styles: overbidders
to set rather than outbid, wild bids that knock a partner out, a field in
which "bid only enough" is safe because raises go to 35 (S-0080-c,
S-0070-a, S-0155-a, S-0155-b, S-0114-b, S-0034-a; [[book-assumptions]]
A6). In every arena block both teams have bid by the same heuristic rule
at caution 0, a field with no style at all, and against it the book's
bidding (`risk-budget`) cost -0.71 [-1.00, -0.41] marks/game
([[book-head-to-head-2026-10-13]]). This experiment seats it against a
field that overbids by a named amount.

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Every seat plays Zeb 0.3.0's plan (`scenario.small_n.player`;
[[zeb-030-confirmation-2026-10-14]]) unless an arm says otherwise. Team B,
the field, bids by the arena's heuristic rule at caution -k for k in
{0, 5, 10}: its willingness is the ceiling plus k, so at k = 5 it bids one
step past what its hand carries and at k = 10 two (ceilings move in steps
of 5). Team A is the target:
- **H<k>**: Zeb's play, bidding by the heuristic rule at caution 0 (the
  control: the field's own rule, undeviated);
- **R<k>**: Zeb's play, bidding by the book's `risk-budget`
  (`silas/examples/book/bidding/risk-budget.silas`, through `SilasBidder`);
- **B<k>**: the book player, the head-to-head's block c (`book:zeb,zeb`:
  Zeb's best tile inside the deciding rule's set, Zeb's tile where no rule
  fires), bidding by the heuristic rule at caution 0.

*Arms.* Marks (games to 7): H0, H5, H10, R0, R5, R10, B0, B5, B10. The
book's game (250 points, each hand settled by the book's rule): PH0, PH5,
PH10, PR0, PR5, PR10 (S-0114-b's pair under both scorings).

*Pairing and the cluster.* As `scenario.small_n`: a game pair is (base
seed, game index) under the arena's paired halves; every contrast is
taken per pair over the pairs all its arms hold; 95% percentile bootstrap
over game pairs (10,000 resamples, seed 42). Duplicate deals across arms.

*Primary.* The book's bidding against each field, C<k> = R<k> - H<k> in
marks per game, for k in {0, 5, 10}. **The book predicts C<k> rises with
k**: restraint pays where the field overbids, and the claim is the k at
which the bidding stops costing (the first k whose interval covers or
lies above 0). Read on the paired differences C5 - C0 and C10 - C0, and
against the bar: C10 - C0 above 0 = the book's bidding is worth more
against its own field than against the style-free one; below 0 = worth
less; else a null at this power with its minimum detectable effect.

*Registered secondaries* (marks per game unless stated):
- the field itself: H<k> for each k, the undeviated bidder's margin
  against an overbidding field (the book predicts H5 and H10 above 0:
  overbidders set themselves);
- the book player: B<k> - H<k> for each k and B10 - B0;
- S-0114-b, marks against points: the same pair under the book's game,
  PR<k> - PH<k> on team A's share of games won and on settled points per
  game, and the sign of (PR10 - PH10) against (R10 - H10): the book
  predicts points punish the overbidders more than marks do;
- mechanism: contracts bid per hand by each team, the share made, and the
  mean bid, by arm.

*Size.* From the timing pilot (`PILOT` = seed 1890, excluded): 4 game
pairs of H0, R10, B10 and PR10 took 29-48 s a batch at load average
about 42, so 12 pairs sit under the 300 s cap. H0 against the k = 0
field read exactly 0 (the same player both sides), as it must. The
contrast of interest at k = 0 was -0.71 [-1.00, -0.41] on 198 clusters
in the head-to-head; the sizing takes SD 1.5 per pair, so 192 pairs give
a half-width of 1.96*1.5/sqrt(192) = 0.21 marks/game and a minimum
detectable C10 - C0 of 2.8*1.5/sqrt(192) = 0.30 (80% power, two-sided
5%). So:
- marks: `SEEDS` = 1901-1916, `GAMES` = 12 pairs a batch: **192 game
  pairs per arm**, nine arms (144 batches);
- 250 points: `POINT_SEEDS` = 1951-1958, 12 pairs a batch: 96 game pairs
  per arm, six arms (48 batches);
- two batches at once through `scripts/cpu-slot`, seed-major, queued
  after the book-at-small-n batches (#154) so that the two experiments
  never hold more than two slots between them.

*Batch-drop rule.* A batch that exits nonzero is dropped, not rerun, and
listed; its game pairs leave every contrast that needs them. No effect is
computed until every registered batch has run.

*Fence.* Every value is Zeb's plan over the mover's own cell; the book's
bidding reads its own hand and the auction ([[strategy-fusion-fence]]).

    python -u -m scenario.deviant_field arena --arm R5 --seed S --games G --out DIR
    python -u -m scenario.deviant_field queue --root DIR [--workers 2] [--pilot]
    python -u -m scenario.deviant_field summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from scenario.small_n import ROOT, _games, _meta, contrast, diff, pairs, player

KS = (0, 5, 10)
TARGETS = ("H", "R", "B")
MARK_ARMS = tuple(f"{t}{k}" for k in KS for t in TARGETS)
POINT_ARMS = tuple(f"P{t}{k}" for k in KS for t in ("H", "R"))
ARMS = MARK_ARMS + POINT_ARMS
POINTS = 250
PILOT = (1890,)               # the timing pilot, excluded
GAMES = 12                    # game pairs per batch
SEEDS = tuple(range(1901, 1917))         # marks: 16 batches per arm, 192 game pairs
POINT_SEEDS = tuple(range(1951, 1959))   # 250 points: 8 batches per arm, 96 game pairs


def parse_arm(arm: str) -> tuple[bool, str, int]:
    """(points game, target, k)."""
    if arm not in ARMS:
        raise ValueError(f"no arm {arm}")
    name = arm.removeprefix("P")
    return arm.startswith("P"), name[0], int(name[1:])


def seats(arm: str):
    """(bid_a, play_a, bid_b, play_b) for an arm."""
    from arena.bidders import HeuristicBidder
    from arena.book_play import BookPlay, SilasBidder

    _, target, k = parse_arm(arm)
    bid_a = SilasBidder("risk-budget") if target == "R" else HeuristicBidder()
    play_a = BookPlay("zeb", "zeb", player()) if target == "B" else player()
    return bid_a, play_a, HeuristicBidder(caution=-k), player()


def arena(arm: str, seed: int, games: int, out: str) -> dict:
    """One batch: `games` game pairs (2·games games) of team A against the field."""
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
        bids = {s: [h for h in hands if (h["bidder_team"] == h["a_team"]) == (s == "a") and not h["forced"]]
                for s in "ab"}
        out["arms"][arm] = {
            "batches": [{x: _meta(r)[x] for x in ("seed", "wall_s", "load_avg")} for r in rs],
            "games": len(games), "hands": len(hands),
            "contracts_per_hand": {s: len(v) / max(len(hands), 1) for s, v in bids.items()},
            "made": {s: sum(h["made"] for h in v) / len(v) if v else None for s, v in bids.items()},
            "mean_bid": {s: sum(h["bid"] for h in v) / len(v) if v else None for s, v in bids.items()},
            "a_margin": contrast(marks if arm in marks else won, [(1.0, arm)]),
        }
    reg = out["registered"]
    for k in KS:
        reg[f"primary C{k} = R{k} - H{k}, marks/game"] = contrast(marks, diff(f"R{k}", f"H{k}"))
    reg["C5 - C0, marks/game"] = contrast(marks, diff("R5", "H5") + diff("H0", "R0"))
    reg["C10 - C0, marks/game"] = contrast(marks, diff("R10", "H10") + diff("H0", "R0"))
    for k in KS:
        reg[f"the field: H{k} against 0, marks/game"] = contrast(marks, [(1.0, f"H{k}")])
        reg[f"the book player: B{k} - H{k}, marks/game"] = contrast(marks, diff(f"B{k}", f"H{k}"))
    reg["B10 - B0, marks/game"] = contrast(marks, diff("B10", "B0"))
    for k in KS:
        reg[f"250: PR{k} - PH{k} on share of games won"] = contrast(won, diff(f"PR{k}", f"PH{k}"))
        reg[f"250: PR{k} - PH{k} on settled points/game"] = contrast(pts, diff(f"PR{k}", f"PH{k}"))
    return out


# -- the queue ------------------------------------------------------------------------

def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    if pilot:
        return [(a, s) for s in PILOT for a in ("H0", "R10", "B10", "PR10")]
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
            cmd = [str(ROOT / "scripts" / "cpu-slot"), "--", sys.executable, "-u", "-m", "scenario.deviant_field",
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
