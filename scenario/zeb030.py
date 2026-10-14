# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Zeb 0.3.0's confirmation match: do its two new defaults pay together?

Zeb 0.3.0 is Zeb 0.2.0 with two defaults changed, each measured alone:

- **Soft play-reading** (`Config.reading`, uniform likelihood, ε = 0.28,
  margin 2^22, drawn from the posterior): +0.37 [+0.24, +0.51] marks/game
  against plain Zeb ([[reads-plays-soft-2026-10-14]] § Replication 2).
- **Make, then margin** on exact ties (`Config.tie = "margin"`, band 0):
  +0.01 [−0.04, +0.07] marks/game, +4.55 points/hand ([[tie-margin-2026-10-14]]).

They were never measured together. This match asks whether the package's
0.3.0 plan beats its 0.2.0 plan, as the package plays them.

**Fixed before any measured batch** (committed with this file; the pilot,
`PILOT`, is excluded):

- *Players*, built exactly as `zeb/src/lib.rs` builds them under a pip trump
  (the arena's `bid30` bidders only declare pip trumps), at the package's
  default 160 worlds and seed 0, over the package's own net blob
  (`zeb/net/margin_d250k_h128x64.fnet`):
  0.2.0 = `k0,n160,F=<net>,f=16`;
  0.3.0 = 0.2.0 + `d=posterior,r=uniform:4194304:0.28,t=margin` (draw budget
  2^22, the default).
- *Table.* Both seats of team A are the challenger, both of team B the
  baseline; `bid30` bidders on both teams; games to 7 marks; the arena's
  paired halves (each game seed played with team A as team 0 and as team 1
  over the same deals: duplicate deals).
- *Arms.* `v3`: 0.3.0 against 0.2.0. `aa` (the A/A control): 0.2.0 at
  planner seed 1 against 0.2.0 at seed 0, same seeds, same run.
- *Seeds.* Base seeds from `SEEDS`, fresh: none of 1–7, 101–118, 201–209,
  301–354, 401–424, 900–904, 990–996 (the reading studies) or the tie
  study's 1, 2, …
- *Primary.* `v3`'s margin in marks per game (a game pair's sum / 2); 95%
  percentile bootstrap over game pairs (the cluster: a pair shares its
  deals), 4,000 resamples, seed 42. **The bar: the interval's lower bound
  is above 0.** Below it, 0.3.0 does not ship.
- *Registered secondaries.* The A/A margin (should cover 0); `v3`'s settled
  points per game; contracts made bidding, each side.
- *Exploratory.* `v3` − A/A on shared pairs; the per-batch signs.
- *Size.* 16 batches × 24 game pairs = **384 game pairs per arm**. The
  pilot (seeds 690–691, 16 pairs per arm, excluded) put the per-pair SD at
  1.22 marks/game (`v3`) and 1.56 (`aa`), and 1.2–1.8 s per pair; the
  sizing takes SD 1.5: (1.96 · 1.5 / 0.15)² = 384 pairs for a half-width of
  0.15. A batch of 24 pairs is about a minute, well inside the 300 s cap.
- *Batch drop.* Every batch runs under `timeout 300`; one that exits
  nonzero is dropped, not rerun, and listed. No effect is computed until
  every batch has run: no early stop, no top-up.

    python -u -m scenario.zeb030 arena --arm v3 --seed S --games G --out DIR
    python -u -m scenario.zeb030 summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

from scenario.planner import Reading

NET = Path(__file__).resolve().parents[1] / "zeb" / "net" / "margin_d250k_h128x64.fnet"
WORLDS = 160                 # the package's default
FLOOR = 16
READING = Reading("uniform", 1 << 22, 0.28)
ARMS = ("v3", "aa")
PILOT = (690, 691)           # the sizing pilot, excluded
GAMES = 24                   # game pairs per batch
SEEDS = tuple(range(601, 617))   # 16 batches per arm: 384 game pairs


def player(version: str, seed: int = 0):
    """The package's plan under a pip trump, as an arena play policy."""
    from arena.walt_play import WaltPlay
    from scenario.field import FieldNet

    net = FieldNet.from_bytes(NET.read_bytes())
    if version == "0.2.0":
        return WaltPlay(level=0, n_outer=WORLDS, floor=FLOOR, field=net, seed=seed)
    if version == "0.3.0":
        return WaltPlay(level=0, n_outer=WORLDS, floor=FLOOR, field=net, seed=seed,
                        draw="posterior", reading=READING, tie="margin")
    raise ValueError(f"no zeb {version}")


def arena(arm: str, seed: int, games: int, out: str) -> dict:
    """One batch: `games` game pairs (2·games games) of team A against team B."""
    from arena.bidders import Bid30Bidder
    from arena.engine import ArenaConfig
    from arena.match import run_match

    if arm == "v3":
        play_a, play_b = player("0.3.0"), player("0.2.0")
    elif arm == "aa":
        play_a, play_b = player("0.2.0", seed=1), player("0.2.0")
    else:
        raise ValueError(f"no arm {arm}")
    t0 = time.time()
    result = run_match(bid_a=Bid30Bidder(), bid_b=Bid30Bidder(), play_a=play_a, play_b=play_b,
                       n_games=2 * games, cfg=ArenaConfig(base_seed=seed), fast_batching=True)
    wall = time.time() - t0
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"arm": arm, "seed": seed, "games_per_half": games, "wall_s": round(wall, 1),
            "load_avg": list(os.getloadavg()), "a": repr(play_a), "b": repr(play_b)}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    with open(outdir / "games.jsonl", "w") as f:
        for g in result.games:
            f.write(json.dumps({
                "game": g.game_idx, "a_team": g.a_team, "a_marks": g.a_marks, "b_marks": g.b_marks,
                "a_won": g.a_won, "a_points": g.points[g.a_team], "b_points": g.points[1 - g.a_team],
                "hands": [{"bidder_team": h.bidder_team, "made": h.made, "a_team": h.a_team,
                           "points": list(h.team_points)} for h in g.hands]}) + "\n")
    return meta


def summarize(runs: Sequence[str]) -> dict:
    """The registered effects, and the exploratory ones, from the batches."""
    from scenario.reads_soft import _diff, _games, _meta, _pairs

    by: dict[str, list[str]] = {}
    for run in runs:
        by.setdefault(_meta(run)["arm"], []).append(run)
    marks = {k: _pairs(rs, "marks") for k, rs in by.items()}
    points = {k: _pairs(rs, "points") for k, rs in by.items()}
    out: dict = {"arms": {}, "registered": {}, "exploratory": {}}
    for arm, rs in sorted(by.items()):
        games = [g for r in rs for g in _games(r)]
        hands = [h for g in games for h in g["hands"]]
        made = {side: [h["made"] for h in hands if (h["bidder_team"] == h["a_team"]) == (side == "a")]
                for side in ("a", "b")}
        per_batch = [sum(g["a_marks"] - g["b_marks"] for g in _games(r)) for r in rs]
        out["arms"][arm] = {
            "batches": [{x: _meta(r)[x] for x in ("seed", "wall_s", "load_avg")} for r in rs],
            "games": len(games), "hands": len(hands),
            "made_bidding": {s: sum(v) / len(v) if v else None for s, v in made.items()},
            "batch_signs": {"positive": sum(x > 0 for x in per_batch), "zero": sum(x == 0 for x in per_batch),
                            "negative": sum(x < 0 for x in per_batch)},
        }
    reg, exp = out["registered"], out["exploratory"]
    if "v3" in by:
        m = _diff(marks["v3"], None)
        reg["primary: 0.3.0 vs 0.2.0, marks/game"] = {**m, "success": bool(m["lo"] > 0)}
        reg["0.3.0 vs 0.2.0, settled points/game"] = _diff(points["v3"], None)
    if "aa" in by:
        reg["A/A: 0.2.0 at seed 1 vs 0.2.0, marks/game"] = _diff(marks["aa"], None)
    if "v3" in by and "aa" in by:
        exp["0.3.0 - A/A, marks/game (same pairs)"] = _diff(marks["v3"], marks["aa"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arena")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--games", type=int, required=True, help="game pairs")
    p.add_argument("--out", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "arena":
        print(json.dumps(arena(a.arm, a.seed, a.games, a.out), indent=1))
    else:
        r = summarize(a.runs)
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps({k: r[k] for k in ("registered", "exploratory")}, indent=1))


if __name__ == "__main__":
    main()
