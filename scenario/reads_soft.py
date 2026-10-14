"""Zeb reads the other seats' plays through a soft set: does the reading pay
where its model is true? (#145)

The sharp reading ([[reads-plays-2026-10-13]], `scenario.reads_plays`) was a
null in marks against plain Zeb, and its posterior was empty at 28% of
trick-5 and 51% of trick-6 decisions, where it fell back to the uniform.
Here the set is softened (`Reading.eps`): a play of one of n legal tiles
weighs ε/n + (1 − ε)·[tile ∈ A]·w (w = 1 consistent, 1/|A| uniform), so no
world vanishes. Two tables:

- **net**: the reading seat's partner and both opponents *are* the model —
  a `NetSeat` that explores a uniform legal tile with probability ε and
  otherwise picks uniformly within the net's 2-point set A. The soft uniform
  likelihood is then the exact probability of every play the reader sees
  (given the hands), so the posterior is the true one; the consistent
  likelihood drops the set size, so uniform − consistent prices restricted
  choice where it exists by construction.
- **zeb**: the other seats are deployed Zeb (the sharp study's design: the
  reader team against plain Zeb, paired halves).

Every value is still Zeb's plan over the drawn worlds; no
perfect-information value enters (the strategy-fusion fence).

**Fixed before the measured runs** (`PREREG`, committed before any arena
batch; the wiki page repeats it):

- ε = `EPS` = 0.28, the maximum-likelihood ε of the soft uniform model on
  deployed Zeb's own choices (`fit`, the research-night pour: 40 deals of
  Zeb self-play with the same net, 714 choices: ML 0.2835, profile 95%
  [0.22, 0.36]; 7.6% of Zeb's choices fall outside the 2-point set), to 2
  significant figures. The same ε for the net seats, so the net table's
  model is true.
- Size, fixed from a timing pilot (base seeds 900–904, excluded): net
  table, 4 batches of 48 game pairs per arm at base seeds 1–4; zeb table,
  7 batches of 24 game pairs per arm at base seeds 1–7. Each batch under
  `timeout 300`; a batch that overruns is dropped, not rerun, and said so
  (the effects use the game pairs both arms hold).
- Players: plain = deployed Zeb `k0,n512,F=margin_d250k_h128x64,f=16`;
  reader = plain + `d=posterior,r=<likelihood>:2^22:EPS`.
- Arms. *net table*: team A = {X at the team's lower seat, `NetSeat` partner}
  vs team B = two `NetSeat`s, X ∈ {plain, consistent, uniform}, paired halves
  (A as team 0 and as team 1 over the same deals), `bid30` bidders, games to
  7 marks. *zeb table*: team A = two readers vs team B = two plain Zebs,
  X ∈ {consistent, uniform}. The arms share base seeds, so the game pair
  (base seed, game index) indexes the same deals in every arm (until play
  diverges the hand count), and the net seats' draws are keyed on the
  state, so identical states get identical net plays across arms.
- Effects, marks per game (pair sum / 2), 95% bootstrap over game pairs
  (4,000 resamples, seed 42; the cluster is the game pair):
  net table: uniform − plain (**primary**), consistent − plain, uniform −
  consistent (restricted choice); zeb table: uniform, consistent (the
  reader's margin against plain; chance level 0 by symmetry), uniform −
  consistent. Also settled points per game (the book's 250-point tally,
  kept on every hand) and marks per hand.
- Bars: an effect *pays* when its interval excludes 0 on the positive side,
  *costs* when it excludes 0 on the negative side, else it is a null, with
  its interval stated. The primary bar is the net-table uniform − plain.
- Diagnostics: fallback rate by trick (the sharp reader's was 28% / 51% at
  tricks 5 / 6; the soft one should be 0 off the rejection path), the draw's
  distinct worlds of 512 by trick, the importance ESS fraction of the soft
  L on plain Zeb's uniform sample (`ess`, every 50th decision), the real
  deal's likelihood rank, and the decision-change rate against plain Zeb
  at the same state beside plain Zeb re-seeded (the chance level of a fresh
  sample; 0 on an exact fiber), by trick and draw path.

    python -u -m scenario.reads_soft fit --pour DIR --net NET.npz
    python -u -m scenario.reads_soft arena --net NET.npz --table net|zeb --arm plain|consistent|uniform \\
        --seed S --games G --out DIR
    python -u -m scenario.reads_soft ess --net NET.npz --runs DIR... [--every 50] --out FILE
    python -u -m scenario.reads_soft summarize --runs DIR... --out DIR
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState

from scenario.reads_plays import MARGIN, Recorder, _decisions, clustered

EPS = 0.28                   # `fit`: ML 0.2835 [0.2195, 0.3565] on 714 choices of the research-night pour
TABLES = ("net", "zeb")
ARMS = ("plain", "consistent", "uniform")
SEAT_SALT = 0x6E657473       # "nets": the net seats' draws are SplitMix64 on _mix(SEAT_SALT, deal, record)
PREREG = __doc__


# -- the fit ------------------------------------------------------------------------

def fit(pour_dir: str, net_path: str) -> dict:
    """The maximum-likelihood ε of the soft uniform model, ε/n + (1 − ε)·[t ∈ A]/|A|,
    on the choices of a pour (each position's chosen tile, the mover's hand
    and record), A the net's set at `MARGIN`; and the share of choices
    outside A. A grid of 1,000 values of ε in (0, 1)."""
    from scenario.atlas import load
    from scenario.field import FieldNet
    from scenario.rules import rules_for

    net = FieldNet.load(net_path)
    pour = load(pour_dir)
    rows = []
    for p in range(len(pour)):
        pos = pour.position(p)
        rules = rules_for(pos.decl_id)
        n = len(pos.actions)
        if n < 2:
            continue
        a = net.allowed(rules, pos.viewer, pos.own, pos.record, pos.bid, pos.bidder % 2, MARGIN)
        rows.append((n, a.bit_count(), (a >> pos.chosen) & 1))
    n, size, inside = (np.array(c, np.float64) for c in zip(*rows))
    grid = (np.arange(1000) + 0.5) / 1000

    def ll(e: float) -> float:
        return float(np.log(e / n + (1 - e) * inside / size).sum())

    lls = np.array([ll(e) for e in grid])
    i = int(np.argmax(lls))
    within = grid[lls >= lls[i] - 1.92]            # the profile-likelihood 95% interval
    return {"choices": len(rows), "outside_set": float(1 - inside.mean()), "mean_set": float(size.mean()),
            "mean_legal": float(n.mean()), "eps_ml": float(grid[i]),
            "eps_95": [float(within.min()), float(within.max())], "loglik": float(lls[i])}


# -- the net seats --------------------------------------------------------------------

class NetSeat:
    """The model, played: explore a uniform legal tile with probability ε,
    else a uniform pick within the net's set at `MARGIN`. Its draws are
    SplitMix64 on (SEAT_SALT, the deal, the record), so a state is played
    the same way in every arm."""

    def __init__(self, net_path: str, eps: float = EPS):
        from scenario.field import FieldNet
        self.net = FieldNet.load(net_path)
        self.eps = eps

    def tile(self, state: ZebGameState, bid: int) -> int:
        from scenario.planner import SplitMix, _mix, _tape_key
        from scenario.record import record_from_state, remaining_hand
        from scenario.rules import mask_of, rules_for, tiles_of
        rec = record_from_state(state)
        seat = current_player(state)
        own = remaining_hand(state, seat)
        rules = rules_for(state.decl_id)
        legal = tiles_of(rules.legal(own, rec.led))
        rng = SplitMix(_mix(SEAT_SALT, *(mask_of(h) for h in state.hands), _tape_key(rec)))
        if len(legal) == 1:
            return legal[0]
        if rng.random() < self.eps:
            return legal[rng.next_u64() % len(legal)]
        a = tiles_of(self.net.allowed(rules, seat, own, rec, bid, state.bidder % 2, MARGIN))
        return a[rng.next_u64() % len(a)]

    def choose(self, states: Sequence[ZebGameState], bid_values: Sequence[int],
               marks=None, marks_to_win: int = 7) -> list[int]:
        from arena.auction import contract_points
        return [list(s.hands[current_player(s)]).index(self.tile(s, contract_points(int(b))))
                for s, b in zip(states, bid_values)]


class Table:
    """A team's play policy at the net table: X at the team's lower seat (0
    or 1), a `NetSeat` at its partner's."""

    def __init__(self, x, net: NetSeat):
        self.x, self.net = x, net

    def choose(self, states: Sequence[ZebGameState], bid_values: Sequence[int],
               marks=None, marks_to_win: int = 7) -> list[int]:
        mine = [i for i, s in enumerate(states) if current_player(s) < 2]
        out = self.net.choose(states, bid_values) if len(mine) < len(states) else [0] * len(states)
        if mine:
            for i, slot in zip(mine, self.x.choose([states[i] for i in mine], [bid_values[i] for i in mine])):
                out[i] = slot
        return out


class Plain:
    """Plain Zeb at the net table's X seat: no rows, no shadows."""

    def __init__(self, net_path: str):
        from arena.walt_play import parse_walt_spec
        from scenario.reads_plays import specs
        self.play = parse_walt_spec(specs(net_path, None))

    def choose(self, states, bid_values, marks=None, marks_to_win: int = 7):
        return self.play.choose(states, bid_values)


# -- the arena ----------------------------------------------------------------------

def arena(net_path: str, table: str, arm: str, seed: int, games: int, out: str) -> dict:
    """One batch: `games` game pairs (2·games games)."""
    from arena.bidders import Bid30Bidder
    from arena.engine import ArenaConfig
    from arena.match import run_match
    from arena.walt_play import parse_walt_spec
    from scenario.reads_plays import specs

    if table not in TABLES or arm not in ARMS or (table == "zeb" and arm == "plain"):
        raise ValueError(f"no arm {table}/{arm}")
    rec = None if arm == "plain" else Recorder(net_path, arm, EPS)
    if table == "net":
        net = NetSeat(net_path)
        play_a, play_b = Table(rec if rec is not None else Plain(net_path), net), net
    else:
        play_a, play_b = rec, parse_walt_spec(specs(net_path, None))
    t0 = time.time()
    result = run_match(bid_a=Bid30Bidder(), bid_b=Bid30Bidder(), play_a=play_a, play_b=play_b,
                       n_games=2 * games, cfg=ArenaConfig(base_seed=seed), fast_batching=True)
    wall = time.time() - t0
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [] if rec is None else rec.rows
    meta = {"table": table, "arm": arm, "eps": EPS, "seed": seed, "games_per_half": games,
            "wall_s": round(wall, 1), "margin": MARGIN, "decisions": len(rows),
            "x": specs(Path(net_path).name, None if arm == "plain" else arm, EPS)}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    with open(outdir / "games.jsonl", "w") as f:
        for g in result.games:
            f.write(json.dumps({
                "game": g.game_idx, "a_team": g.a_team, "a_marks": g.a_marks, "b_marks": g.b_marks,
                "a_points": g.points[g.a_team], "b_points": g.points[1 - g.a_team],
                "hands": [{"bidder_team": h.bidder_team, "made": h.made, "a_team": h.a_team,
                           "points": list(h.team_points)} for h in g.hands]}) + "\n")
    with open(outdir / "decisions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return meta


# -- the effective sample size ---------------------------------------------------------

def ess(net_path: str, runs: Sequence[str], every: int = 50) -> dict:
    """The importance ESS fraction, (Σw)² / (n·Σw²), of the soft likelihoods
    (and the sharp ones, for comparison) on the uniform sample plain Zeb
    plans over, at every `every`-th recorded decision; and the real deal's
    rank among those worlds by the soft uniform likelihood (the share of
    the sample it outweighs; a diagnostic only)."""
    from arena.walt_play import parse_walt_spec
    from scenario.atlas import _state
    from scenario.planner import Walt, situation_from_state
    from scenario.reads_plays import _rows, specs
    from scenario.record import empty_record
    from scenario.rules import rules_for

    plain = parse_walt_spec(specs(net_path, None)).walt
    refs = {f"{lk}{'' if e else '_sharp'}": Walt(parse_walt_spec(specs(net_path, lk, e)).walt.config)
            for lk in ("consistent", "uniform") for e in (EPS, 0.0)}
    out = []
    for row in (r for run in runs for i, r in enumerate(_rows(run)) if i % every == 0):
        rules = rules_for(row["decl"])
        plays = tuple(tuple(p) for p in row["plays"])
        rec = empty_record(plays[0][0] if plays else row["bidder"])
        for _, t in plays:
            rec = rec.advance(t, rules)
        state = _state(tuple(tuple(h) for h in row["hands"]), row["bidder"], row["bid"], row["decl"], plays, rec)
        worlds = plain.analyze(state, row["bid"]).worlds
        sit = situation_from_state(state, row["bid"])
        r = {"plays": len(plays), "n": len(worlds)}
        for name, ref in refs.items():
            w = np.asarray(ref.reading_likelihood(sit, worlds))
            r[name] = float(w.sum() ** 2 / (len(w) * (w ** 2).sum())) if w.any() else 0.0
        real = tuple(sum(1 << t for t in row["hands"][s] if t not in {p[1] for p in plays}) for s in range(4))
        w = np.asarray(refs["uniform"].reading_likelihood(sit, worlds + [real]))
        r["real_rank"] = float(np.mean(w[:-1] < w[-1]) + 0.5 * np.mean(w[:-1] == w[-1]))
        out.append(r)
    trick = np.array([r["plays"] // 4 + 1 for r in out])
    keys = list(refs) + ["real_rank"]
    by_trick = {int(t): {"n": int((trick == t).sum()),
                         **{k: float(np.median([r[k] for r, tt in zip(out, trick) if tt == t])) for k in keys}}
                for t in range(1, 8) if (trick == t).any()}
    return {"every": every, "eps": EPS, "by_trick": by_trick, "rows": out}


# -- the summary ----------------------------------------------------------------------

def _games(run: str) -> list[dict]:
    with open(Path(run) / "games.jsonl") as f:
        return [json.loads(line) for line in f]


def _meta(run: str) -> dict:
    return json.loads((Path(run) / "meta.json").read_text())


def _pairs(runs: Sequence[str], key: str) -> dict[tuple[int, int], float]:
    """(base seed, game) → team A's margin summed over the pair's two games,
    in marks (`key` = "marks") or settled points ("points")."""
    out: dict[tuple[int, int], float] = {}
    for run in runs:
        seed = _meta(run)["seed"]
        for g in _games(run):
            k = (seed, g["game"])
            out[k] = out.get(k, 0.0) + g[f"a_{key}"] - g[f"b_{key}"]
    return out


def _diff(a: dict, b: dict | None) -> dict:
    """Per game, a − b over the game pairs both hold (b None: a alone), clustered by pair."""
    common = sorted(set(a) if b is None else set(a) & set(b))
    d = np.array([a[k] - (0.0 if b is None else b[k]) for k in common]) / 2
    mean, lo, hi = clustered(d)
    return {"game_pairs": len(common), "mean": mean, "lo": lo, "hi": hi}


def summarize(runs: Sequence[str]) -> dict:
    by: dict[tuple[str, str], list[str]] = {}
    for run in runs:
        m = _meta(run)
        by.setdefault((m["table"], m["arm"]), []).append(run)
    out: dict = {"eps": EPS, "arms": {}, "effects": {}}
    marks = {k: _pairs(rs, "marks") for k, rs in by.items()}
    points = {k: _pairs(rs, "points") for k, rs in by.items()}
    for (table, arm), rs in sorted(by.items()):
        games = [g for r in rs for g in _games(r)]
        hands = [h for g in games for h in g["hands"]]
        rows = [json.loads(line) for r in rs for line in open(Path(r) / "decisions.jsonl")]
        made = {side: [h["made"] for h in hands if (h["bidder_team"] == h["a_team"]) == (side == "a")]
                for side in ("a", "b")}
        arm_out = {
            "batches": [_meta(r) for r in rs], "games": len(games), "hands": len(hands),
            "marks_per_game": _diff(marks[(table, arm)], None),
            "points_per_game": _diff(points[(table, arm)], None),
            "marks_per_hand": float(sum(g["a_marks"] - g["b_marks"] for g in games) / max(len(hands), 1)),
            "made": {k: {"n": len(v), "rate": float(np.mean(v)) if v else math.nan} for k, v in made.items()},
        }
        if rows:
            arm_out.update(_decisions(rows))
            arm_out.update(_draws(rows))
        out["arms"][f"{table}/{arm}"] = arm_out

    def effect(name: str, a: tuple, b: tuple | None) -> None:
        if a in marks and (b is None or b in marks):
            out["effects"][name] = {"marks_per_game": _diff(marks[a], None if b is None else marks[b]),
                                    "points_per_game": _diff(points[a], None if b is None else points[b])}

    effect("net: uniform - plain (primary)", ("net", "uniform"), ("net", "plain"))
    effect("net: consistent - plain", ("net", "consistent"), ("net", "plain"))
    effect("net: uniform - consistent (restricted choice)", ("net", "uniform"), ("net", "consistent"))
    effect("zeb: uniform vs plain", ("zeb", "uniform"), None)
    effect("zeb: consistent vs plain", ("zeb", "consistent"), None)
    effect("zeb: uniform - consistent (restricted choice)", ("zeb", "uniform"), ("zeb", "consistent"))
    return out


def _draws(rows: list[dict]) -> dict:
    """Fallbacks and the draw's distinct worlds, by trick."""
    trick = np.array([len(r["plays"]) // 4 + 1 for r in rows])
    fell = np.array([bool(r["fell_back"]) for r in rows])
    uniq = np.array([r.get("distinct", 0) for r in rows])
    return {
        "fallback_by_trick": {int(t): {"n": int((trick == t).sum()), "rate": float(fell[trick == t].mean())}
                              for t in range(1, 8) if (trick == t).any()},
        "distinct_worlds_median_by_trick": {int(t): float(np.median(uniq[trick == t]))
                                            for t in range(1, 8) if (trick == t).any()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--pour", required=True)
    p.add_argument("--net", required=True)
    p = sub.add_parser("arena")
    p.add_argument("--net", required=True)
    p.add_argument("--table", choices=TABLES, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--games", type=int, required=True, help="game pairs")
    p.add_argument("--out", required=True)
    p = sub.add_parser("ess")
    p.add_argument("--net", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--every", type=int, default=50)
    p.add_argument("--out", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        print(json.dumps(fit(a.pour, a.net), indent=1))
    elif a.cmd == "arena":
        print(json.dumps(arena(a.net, a.table, a.arm, a.seed, a.games, a.out), indent=1))
    elif a.cmd == "ess":
        r = ess(a.net, a.runs, a.every)
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r["by_trick"], indent=1))
    else:
        s = summarize(a.runs)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        print(json.dumps(s["effects"], indent=1))


if __name__ == "__main__":
    main()
