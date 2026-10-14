"""The regulars' table (#149): does an opponent who models Zeb across hands
make Zeb's determinism cost, and the book's variation pay?

A 2 × 2 on duplicate deals. The book's lessons on style and reputation
(S-0080-b, S-0080-c, S-0081-a, S-0089-e, S-0156-b, S-0070-a, S-0189-a)
assume opponents who learn you across hands. Every arena seat forgets
between hands, so in our arena varying your play is worth exactly 0 by
construction ([[book-assumptions]] A1). This table seats one opponent that
remembers, in the one channel the pilot showed open within a game: the
auction ([[book-assumptions]] § Pilot: a deterministic bidder is identified
within one 11-hand game 83% of the time).

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Every seat plays Zeb 0.3.0's plan, the deployed default, as the
arena's native core runs it: `k0,n160,F=<zeb/net net>,f=16,d=posterior,
r=uniform:2^22:0.28,t=margin` (`player`). Team A is the target, team B the
opponents.
- **Z**: the target bids by the arena's heuristic rule at caution 0
  (`HeuristicBidder`), deterministic per coordinate.
- **Zv**: the target bids by the same rule with its caution drawn per
  seat per hand uniformly from {-5, 0, +5} (`VariedBidder`; ceilings move
  in steps of 5, so three behaviours), from the deal's own auction rng, so
  a hand's draw is the same in every arm. Its play is Z's.
- **R0**: the opponents add the auction to their reading
  (`arena.regulars.RegularsPlay`, `memory=False`): each target seat's
  bid or pass is evidence on its dealt hand under a heuristic bidder whose
  caution is unknown, flat on {-10, -5, 0, 5, 10}, trembling with
  probability 0.1; reset every hand.
- **R**: the same reader with memory within a game (`memory=True`): after
  each hand, every seat's belief over a *fixed* caution is updated by
  Bayes from its revealed hand and its action. R − R0 is exactly the
  cross-hand learning.
- The opponents bid by the heuristic rule at caution 0 in every arm. No
  seat changes its own bidding or play in response to what it learned.

*Arms* (team A vs team B): `Z.R0`, `Z.R`, `Zv.R0`, `Zv.R`, and the A/A pair
`Za.R0`, `Za.R`, where Za is Z at planner seed 1. Each in the tournament's
game (7 marks) and, for the secondaries, the book's (250 points; arm
names prefixed `P`).

*Pairing and the cluster.* Every arm plays the same base seeds with the
arena's paired halves (team A as team 0 and as team 1 over the same
deals). A game pair is (base seed, game index); its value is team A's
margin summed over its two games / 2. Every contrast is taken per game
pair over the pairs all its arms hold, and its 95% interval is the
percentile bootstrap over game pairs (10,000 resamples, seed 42).

*Primary.* The interaction, in marks per game:
    I = (Zv.R − Z.R) − (Zv.R0 − Z.R0)
the value of varying against a regular who remembers, less its value at a
table that forgets. **The book predicts I > 0.** It is read against
- the A/A interaction I_AA = (Za.R − Z.R) − (Za.R0 − Z.R0), whose interval
  should cover 0 (a re-seeded Z is not a different style);
- the bar: I's interval excludes 0 above = the book's predictability
  lesson is visible in our arena; excludes 0 below = varying costs more
  against this regular than at a memoryless table; else a null at this
  power, reported with its minimum detectable effect.

*Registered secondaries* (marks per game unless stated):
- S = Z.R − Z.R0 measured as team B's gain, i.e. −(Z.R − Z.R0): the
  regular's learning gain against a deterministic Zeb (the book predicts
  > 0).
- C = Zv.R0 − Z.R0: the price of varying at a memoryless table (the book
  predicts it small).
- V = Zv.R − Z.R: the value of varying against the regular.
- Each of the four arms' margins against 0 (by symmetry of play, but the
  teams are not the same player, so 0 is a reference, not a null).
- The 250-point interaction on the share of games won by team A, and on
  settled points per game.
- Mechanism: the regulars' final belief on each target seat per game (the
  mass on caution 0, and the entropy), by arm.

*Size.* From the timing pilot (`PILOT`, excluded; seed 1490 ran before a
slow auction slot was made table-driven, 1491 after): 8 game pairs of the
four main arms took 67-73 s with four batches at once (load average 33 on
18 cores, other sessions running). Its interaction SD per pair was 0.86
over 8 pairs, too few to trust; the sizing takes SD 1.5, so 384 pairs give
a half-width of 1.96·1.5/√384 = 0.15 marks/game and a minimum detectable
interaction of 2.8·1.5/√384 = 0.21 (80% power, two-sided 5%). So:
- marks: `SEEDS` = 1501-1532, `GAMES` = 12 pairs a batch: **384 game
  pairs per arm**, six arms;
- 250 points: `POINT_SEEDS` = 1601-1616, 12 pairs a batch: 192 game pairs
  per arm, six arms (secondaries only);
- at most 8 batches at once, every batch under `timeout 300`, the jobs in
  seed-major order (every arm of a seed before the next seed).

*Batch-drop rule.* A batch that exits nonzero (an overrun exits 124) is
dropped, not rerun, and listed; its game pairs leave every contrast that
needs them. No effect is computed until every registered batch has run:
no early stop, no top-up.

*Fence.* Every value is Zeb's plan over the reader's own cell. The auction
model is a likelihood over information public when it is used (this
hand's auction; tiles of finished hands) ([[strategy-fusion-fence]]).

    python -u -m scenario.regulars arena --arm Zv.R --seed S --games G --out DIR
    python -u -m scenario.regulars summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from scenario.planner import Reading

NET = Path(__file__).resolve().parents[1] / "zeb" / "net" / "margin_d250k_h128x64.fnet"
WORLDS = 160                  # the package's default
FLOOR = 16
READING = Reading("uniform", 1 << 22, 0.28)
TARGETS = ("Z", "Zv", "Za")
OPPONENTS = ("R0", "R")
ARMS = tuple(f"{p}{t}.{o}" for p in ("", "P") for t in TARGETS for o in OPPONENTS)
POINTS = 250
PILOT = (1490, 1491)          # the timing pilot, excluded
GAMES = 12                    # game pairs per batch
SEEDS = tuple(range(1501, 1533))         # marks: 32 batches per arm, 384 game pairs
POINT_SEEDS = tuple(range(1601, 1617))   # 250 points: 16 batches per arm, 192 game pairs
RESAMPLES = 10_000


def player(seed: int = 0):
    """Zeb 0.3.0's plan under a pip trump, as an arena play policy (`scenario.zeb030`)."""
    from arena.walt_play import WaltPlay
    from scenario.field import FieldNet

    net = FieldNet.from_bytes(NET.read_bytes())
    return WaltPlay(level=0, n_outer=WORLDS, floor=FLOOR, field=net, seed=seed,
                    draw="posterior", reading=READING, tie="margin")


def seats(arm: str):
    """(bid_a, play_a, bid_b, play_b) for an arm."""
    from arena.bidders import HeuristicBidder, VariedBidder
    from arena.regulars import RegularsPlay

    target, opp = arm.removeprefix("P").split(".")
    if target not in TARGETS or opp not in OPPONENTS:
        raise ValueError(f"no arm {arm}")
    bid_a = VariedBidder() if target == "Zv" else HeuristicBidder()
    play_a = player(seed=1 if target == "Za" else 0)
    return bid_a, play_a, HeuristicBidder(), RegularsPlay(player(), memory=opp == "R")


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
            belief = {str(s): play_b.beliefs[((g.game_idx, g.a_team), s)].pi
                      for s in range(4) if s % 2 == g.a_team and ((g.game_idx, g.a_team), s) in play_b.beliefs}
            f.write(json.dumps({
                "game": g.game_idx, "a_team": g.a_team, "a_marks": g.a_marks, "b_marks": g.b_marks,
                "a_won": g.a_won, "a_points": g.points[g.a_team], "b_points": g.points[1 - g.a_team],
                "belief": belief,
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
    return {"game_pairs": len(common), "mean": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "sd_per_pair": float(d.std(ddof=1)) if len(d) > 1 else math.nan}


def interaction(t: str, prefix: str = "") -> list[tuple[float, str]]:
    """(t.R − Z.R) − (t.R0 − Z.R0), as contrast terms."""
    return [(1.0, f"{prefix}{t}.R"), (-1.0, f"{prefix}Z.R"), (-1.0, f"{prefix}{t}.R0"), (1.0, f"{prefix}Z.R0")]


def summarize(runs: Sequence[str]) -> dict:
    by: dict[str, list[str]] = {}
    for run in runs:
        by.setdefault(_meta(run)["arm"], []).append(run)
    marks = {a: pairs(rs, "marks") for a, rs in by.items() if not a.startswith("P")}
    won = {a: pairs(rs, "won") for a, rs in by.items() if a.startswith("P")}
    pts = {a: pairs(rs, "points") for a, rs in by.items() if a.startswith("P")}
    out: dict = {"arms": {}, "registered": {}, "mechanism": {}}
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
        beliefs = [g["belief"][s] for g in games for s in g["belief"]]
        if beliefs:
            zero = [b[2] for b in beliefs]
            ent = [-sum(p * math.log2(p) for p in b if p > 0) for b in beliefs]
            out["mechanism"][arm] = {"seats": len(beliefs), "mass_on_caution_0": float(np.mean(zero)),
                                     "entropy_bits": float(np.mean(ent)), "flat_entropy_bits": math.log2(5)}
    reg = out["registered"]
    reg["primary I = (Zv.R - Z.R) - (Zv.R0 - Z.R0), marks/game"] = contrast(marks, interaction("Zv"))
    reg["A/A I_AA = (Za.R - Z.R) - (Za.R0 - Z.R0), marks/game"] = contrast(marks, interaction("Za"))
    reg["S = regular's learning gain vs Z = -(Z.R - Z.R0), marks/game"] = contrast(marks, [(-1, "Z.R"), (1, "Z.R0")])
    reg["C = Zv.R0 - Z.R0 (varying at a memoryless table), marks/game"] = contrast(marks, [(1, "Zv.R0"), (-1, "Z.R0")])
    reg["V = Zv.R - Z.R (varying against the regular), marks/game"] = contrast(marks, [(1, "Zv.R"), (-1, "Z.R")])
    reg["250: I on team A's share of games won"] = contrast(won, interaction("Zv", "P"))
    reg["250: I on settled points/game"] = contrast(pts, interaction("Zv", "P"))
    reg["250: A/A I_AA on share of games won"] = contrast(won, interaction("Za", "P"))
    reg["250: S on share of games won (team B's gain)"] = contrast(won, [(-1, "PZ.R"), (1, "PZ.R0")])
    reg["250: C on share of games won"] = contrast(won, [(1, "PZv.R0"), (-1, "PZ.R0")])
    return out


def jobs() -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    marks = [(a, s) for s in SEEDS for a in ARMS if not a.startswith("P")]
    points = [(a, s) for s in POINT_SEEDS for a in ARMS if a.startswith("P")]
    return marks + points


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
        print(json.dumps({k: r[k] for k in ("registered", "mechanism")}, indent=1))


if __name__ == "__main__":
    main()
