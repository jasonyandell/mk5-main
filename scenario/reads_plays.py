"""Zeb reads the other seats' plays: does restricted choice pay?

Deployed Zeb plans over worlds uniform on its cell: how the other seats
played is read only through the voids. Here the live seat's worlds are drawn
from a posterior that reads every other seat's plays (`Config.reading`):

- **The model of the other seats** is the deployed field net, the model Zeb
  already plans against. Its *set* at a decision is the legal tiles it scores
  within δ of its best. Zeb's own policy (self-minds) is the model the
  question names first; `cost` measures it, and it is too slow (the report
  says by how much), so the net stands in: a distillation of Zeb's play,
  regret 0.35 points per decision (`margin_d250k_h128x64.json`).
- **δ = 2 points**, one logit of the net's training temperature (τ = 2
  points: the target was softmax(margin/τ), so the net treats moves within
  τ as near-ties). The net's integer score unit is 2^22 per logit: `scale`
  fits the temperature that minimizes the net's own training loss on a pour
  (2^22.125 on both pours it was fitted on). So the margin is `MARGIN` = 2^22.
- **Two likelihoods,** per discretionary play of each other seat, the
  product over plays and seats (it factorizes by seat, as Silas's readings
  do): *consistent* [tile ∈ A], *uniform* [tile ∈ A]/|A| — a pick uniform
  within the set, so the set's size is evidence (restricted choice).
- **Drawn, not reweighted.** A fiber of at most 512 is weighted by L
  normalized (exact, given the model); one of at most 2^20 is enumerated and
  512 worlds drawn from it with replacement in proportion to L (exact draws);
  a larger one is 512 draws by rejection from the uniform stream (exact
  draws), the uniform sample when 2^22 proposals do not find 512. A fiber L
  vanishes on (no world explains the record under the model) plans over the
  uniform sample, counted as a fallback. The effective sample size of the
  same L as importance weights on the uniform sample is measured beside it
  (`ess`).

**Fixed before running** (and at the top of the wiki page):

- *Players.* Plain: deployed Zeb `k0,n512,F=margin_d250k_h128x64,f=16`.
  Readers: plain plus `d=posterior,r=<likelihood>:MARGIN`, one arm per
  likelihood. `bid30` bidders on both teams (the hub's dropped-30 protocol).
- *Comparison.* The arena's paired halves: a game seed is played twice, the
  reader as team 0 and then as team 1, over the same deals, games to 7
  marks. Metric: the reader's mark margin per game; 95% interval by
  bootstrap over game pairs (deal-clustered: the two games of a pair share
  their deals). Batches run under `timeout 300` with distinct base seeds
  and are pooled. Restricted choice's value is the uniform arm's margin
  less the consistent arm's, paired by game pair (the arms share seeds).
- *Decision change.* At every reading-side decision with two or more legal
  tiles, the reader's tile against the tile plain Zeb chooses at the same
  state: changed when they differ. In the uniform arm the consistent
  reader's tile is asked too (restricted choice's own change rate).
  Positions are the reader's own trajectory. *Chance baseline* (added after
  a first look at one batch, before the measured runs): plain Zeb at another
  seed (`RESEED`) at the same state, the change a fresh sample alone makes.
- *Where.* Change rates by trick, by the mover's seat relative to the
  bidder, by the tiles involved; and per hidden tile T and lump (T with the
  mover's left, partner, right) the shift in mass the reading makes,
  |P_read − P_plain|, at changed against unchanged decisions — to set beside
  lumps-voi's money (the 5-5, 6-4, 4-4; tricks 4–5; the partner's lump).
- *Diagnostics.* Fallbacks; acceptance (512/proposals, which for a 0/1
  likelihood is exactly the importance-weight ESS fraction); the importance
  ESS of each likelihood on the uniform sample (subsample, `ess`); and the
  model's fit: the share of decisions at which the real deal has L > 0.
  The real deal is used for that diagnostic only, never to play or score.

    python -u -m scenario.reads_plays scale --pour DIR --net NET.npz
    python -u -m scenario.reads_plays cost --net NET.npz [--positions 8]
    python -u -m scenario.reads_plays arena --net NET.npz --likelihood uniform --seed S --games G --out DIR
    python -u -m scenario.reads_plays ess --net NET.npz --runs DIR... [--every 25]
    python -u -m scenario.reads_plays summarize --runs DIR... --out DIR
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

MARGIN = 1 << 22             # δ = 2 points in the deployed net's integer score units
DELTA_POINTS = 2.0
TAU = 2.0                    # the net's training temperature, points per logit
SPEC = "k0,n512,F={net},f=16"
ROLES = ("bidder", "left setter", "partner", "right setter")   # (mover − bidder) mod 4
LUMPS = ("left", "partner", "right")                          # T with the mover's left, partner, right
WATCH = ("5-5", "6-4", "4-4", "6-6", "6-5")                   # lumps-voi's top five
RESEED = 1                   # plain Zeb at another seed: the change rate of a fresh sample alone


def tile_name(t: int) -> str:
    from forge.oracle.tables import DOMINOES
    hi, lo = DOMINOES[t]
    return f"{hi}-{lo}"


def tile_id(name: str) -> int:
    from forge.oracle.tables import DOMINOES
    hi, lo = map(int, name.split("-"))
    return DOMINOES.index((max(hi, lo), min(hi, lo)))


def specs(net: str, likelihood: str | None) -> str:
    base = SPEC.format(net=net)
    return base if likelihood is None else f"{base},d=posterior,r={likelihood}:{MARGIN}"


# -- the net's integer scale ------------------------------------------------------

def scale(pour_dir: str, net_path: str) -> dict:
    """The net's integer units per logit: the temperature c minimizing the
    cross-entropy of softmax(scores / c) against the net's own training
    target softmax(margin advantage / τ) on a pour, over a grid of 2^(k/8)."""
    from scenario.atlas import load
    from scenario.field import FieldNet
    from scenario.field_train import dataset, int_scores, targets

    d = dataset(load(pour_dir))
    s = int_scores(FieldNet.load(net_path), d.x).astype(np.float64)
    tgt = targets(d, "margin", TAU).astype(np.float64)

    def ce(c: float) -> float:
        z = np.where(d.legal, s / c, -np.inf)
        z = z - z.max(axis=1, keepdims=True)
        lp = np.where(d.legal, z - np.log(np.exp(z).sum(axis=1, keepdims=True)), 0.0)
        return float(-(tgt * lp).sum(axis=1).mean())

    grid = 2.0 ** np.arange(16, 28, 0.125)
    losses = [ce(c) for c in grid]
    i = int(np.argmin(losses))
    return {"rows": int(len(s)), "log2_scale": float(np.log2(grid[i])), "loss": losses[i],
            "loss_at_2^22": ce(2.0 ** 22), "margin": MARGIN, "delta_points": DELTA_POINTS}


# -- the cost of self-minds ---------------------------------------------------------

def cost(net_path: str, n_positions: int = 8, sample: int = 192, seed: int = 0) -> dict:
    """What reading with Zeb's own policy would cost per decision, at the
    cheapest form (importance weights on the 512 uniform worlds, no
    rejection): every (seat, its hand then, record) at which another seat
    made a discretionary play, over the 512 worlds, is one Zeb decision.
    Counts the distinct ones at positions of a Zeb self-play hand and times
    Zeb on a random `sample` of them, batched on every core."""
    from arena.walt_play import _declared_trump, parse_walt_spec
    from scenario.atlas import _state, deal
    from scenario.record import empty_record
    from scenario.rules import rules_for, tiles_of

    walt = parse_walt_spec(specs(net_path, None)).walt
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_positions):
        hands = deal(1000 + k)
        decl, bidder, bid = _declared_trump(hands[0]), 0, 30
        rules = rules_for(decl)
        held = tuple(tuple(tiles_of(h)) for h in hands)
        rec, plays = empty_record(bidder), ()
        stop = 4 + 4 * (k % 4)          # plays 4, 8, 12, 16, then on to a choice
        while len(plays) < stop or len(tiles_of(rules.legal(hands[rec.seat] & ~rec.played, rec.led))) < 2:
            tile = walt.decide(_state(held, bidder, bid, decl, plays, rec), bid)[0]
            plays += ((rec.seat, tile),)
            rec = rec.advance(tile, rules)
        state = _state(held, bidder, bid, decl, plays, rec)
        viewer = rec.seat
        worlds = walt.analyze(state, bid).worlds
        keys, states = set(), []
        for w in worlds:
            full = [w[s] | sum(1 << t for s2, t in plays if s2 == s) for s in range(4)]
            r = empty_record(bidder)
            for i, (s, t) in enumerate(plays):
                if s != viewer:
                    hand = full[s] & ~r.played
                    if len(tiles_of(rules.legal(hand, r.led))) > 1 and (s, hand, i) not in keys:
                        keys.add((s, hand, i))
                        states.append(_state(tuple(tuple(tiles_of(f)) for f in full), bidder, bid, decl, plays[:i], r))
                r = r.advance(t, rules)
        pick = [states[i] for i in rng.choice(len(states), size=min(sample, len(states)), replace=False)]
        t0 = time.perf_counter()
        walt.decide_batch(pick, [bid] * len(pick))
        per = (time.perf_counter() - t0) / max(len(pick), 1)
        t0 = time.perf_counter()
        walt.decide(state, bid)
        own = time.perf_counter() - t0
        rows.append({"plays": len(plays), "self_mind_decisions": len(keys), "batched_s_per_mind": per,
                     "self_minds_s": per * len(keys), "plain_decision_s": own})
        print(json.dumps(rows[-1]), flush=True)
    return {"positions": rows,
            "median_self_minds_s": float(np.median([r["self_minds_s"] for r in rows])),
            "median_plain_s": float(np.median([r["plain_decision_s"] for r in rows]))}


# -- the arena ----------------------------------------------------------------------

def _masses(worlds: Sequence[Sequence[int]], weights: Sequence[float], mover: int) -> np.ndarray:
    """[3, 28]: the belief mass in which the mover's left, partner and right
    hold each tile."""
    w = np.asarray(weights, np.float64)
    w = w / w.sum()
    m = np.asarray(worlds, np.uint32)
    bits = (m[:, :, None] >> np.arange(28, dtype=np.uint32)) & 1          # [n, 4, 28]
    out = np.einsum("n,nst->st", w, bits.astype(np.float64))
    return out[[(mover + d) % 4 for d in (1, 2, 3)]].astype(np.float32)


class Recorder:
    """The reading side's play policy: the reader decides; plain Zeb (and, in
    the uniform arm, the consistent reader) is asked at the same state; every
    decision with a choice is kept as a row."""

    def __init__(self, net_path: str, likelihood: str):
        from dataclasses import replace

        from arena.walt_play import parse_walt_spec
        from scenario.compiled import CompiledWalt
        from scenario.planner import Walt
        self.reader = parse_walt_spec(specs(net_path, likelihood)).walt
        self.plain = parse_walt_spec(specs(net_path, None)).walt
        self.shadow = parse_walt_spec(specs(net_path, "consistent")).walt if likelihood == "uniform" else None
        self.reseed = CompiledWalt(replace(self.plain.config, seed=RESEED))
        self.referee = Walt(self.reader.config)
        self.rows: list[dict] = []

    def choose(self, states: Sequence[ZebGameState], bid_values: Sequence[int],
               marks=None, marks_to_win: int = 7) -> list[int]:
        from scenario.fiber import Fiber
        from scenario.planner import situation_from_state
        from scenario.record import remaining_hand as rec_own
        bids = [int(b) for b in bid_values]
        read = self.reader.analyze_batch(states, bids)
        draws = list(self.reader.draws)
        plain = self.plain.analyze_batch(states, bids)
        shadow = self.shadow.decide_batch(states, bids) if self.shadow is not None else None
        reseed = self.reseed.decide_batch(states, bids)
        slots = []
        for i, (state, a) in enumerate(zip(states, read)):
            seat = current_player(state)
            slots.append(list(state.hands[seat]).index(a.tile))
            if not a.worlds:                     # forced
                continue
            sit = situation_from_state(state, bids[i])
            rec = sit.record
            fiber = Fiber(seat, rec_own(state, seat), rec.played, tuple(rec.hand_size(s) for s in range(4)),
                          rec.forbid).size
            real = tuple(sum(1 << t for t in state.hands[s] if t not in state.played) for s in range(4))
            like = self.referee.reading_likelihood(sit, [real])[0]
            self.rows.append({
                "hands": [list(h) for h in state.hands], "decl": state.decl_id, "bidder": state.bidder,
                "bid": bids[i], "plays": [list(p) for p in state.play_history], "mover": seat,
                "legal": len(a.q), "tile": a.tile, "plain": plain[i].tile,
                "consistent": -1 if shadow is None else shadow[i][0], "reseed": reseed[i][0],
                "fiber": fiber, "proposals": draws[i][0], "fell_back": draws[i][1], "real_like": like,
                "mass_read": _masses(a.worlds, a.weights, seat).round(5).tolist(),
                "mass_plain": _masses(plain[i].worlds, plain[i].weights, seat).round(5).tolist(),
            })
        return slots


def arena(net_path: str, likelihood: str, seed: int, games: int, out: str) -> dict:
    """One batch: `games` game pairs (2·games games), the reader vs plain Zeb."""
    from arena.bidders import Bid30Bidder
    from arena.engine import ArenaConfig
    from arena.match import run_match
    from arena.walt_play import parse_walt_spec

    rec = Recorder(net_path, likelihood)
    plain = parse_walt_spec(specs(net_path, None))
    t0 = time.time()
    result = run_match(bid_a=Bid30Bidder(), bid_b=Bid30Bidder(), play_a=rec, play_b=plain,
                       n_games=2 * games, cfg=ArenaConfig(base_seed=seed), fast_batching=True)
    wall = time.time() - t0
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    game_rows = [{
        "game": g.game_idx, "a_team": g.a_team, "a_marks": g.a_marks, "b_marks": g.b_marks,
        "hands": [{"bidder_team": h.bidder_team, "made": h.made, "a_team": h.a_team,
                   "points": list(h.team_points)} for h in g.hands],
    } for g in result.games]
    meta = {"likelihood": likelihood, "seed": seed, "games_per_half": games, "wall_s": round(wall, 1),
            "reader": specs(Path(net_path).name, likelihood), "plain": specs(Path(net_path).name, None),
            "margin": MARGIN, "delta_points": DELTA_POINTS, "decisions": len(rec.rows)}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    with open(outdir / "games.jsonl", "w") as f:
        for r in game_rows:
            f.write(json.dumps(r) + "\n")
    with open(outdir / "decisions.jsonl", "w") as f:
        for r in rec.rows:
            f.write(json.dumps(r) + "\n")
    return meta


# -- the effective sample size --------------------------------------------------------

def ess(net_path: str, runs: Sequence[str], every: int = 25) -> dict:
    """The importance ESS fraction, (Σw)² / (n·Σw²), of each likelihood on the
    uniform sample plain Zeb plans over, at every `every`-th recorded decision
    of the runs (the referee's likelihood). Both likelihoods at each."""
    from forge.oracle.declarations import PIP_TRUMP_IDS  # noqa: F401 (the net's declarations)
    from arena.walt_play import parse_walt_spec
    from scenario.atlas import _state
    from scenario.planner import Walt, situation_from_state
    from scenario.record import empty_record
    from scenario.rules import rules_for

    plain = parse_walt_spec(specs(net_path, None)).walt
    refs = {lk: Walt(parse_walt_spec(specs(net_path, lk)).walt.config) for lk in ("consistent", "uniform")}
    out = []
    for row in (r for run in runs for i, r in enumerate(_rows(run)) if i % every == 0):
        rules = rules_for(row["decl"])
        plays = tuple(tuple(p) for p in row["plays"])
        rec = empty_record(plays[0][0] if plays else (row["bidder"]))
        for _, t in plays:
            rec = rec.advance(t, rules)
        state = _state(tuple(tuple(h) for h in row["hands"]), row["bidder"], row["bid"], row["decl"], plays, rec)
        worlds = plain.analyze(state, row["bid"]).worlds
        sit = situation_from_state(state, row["bid"])
        r = {"plays": len(plays), "n": len(worlds)}
        for lk, ref in refs.items():
            w = np.asarray(ref.reading_likelihood(sit, worlds))
            r[lk] = float(w.sum() ** 2 / (len(w) * (w ** 2).sum())) if w.any() else 0.0
        out.append(r)
    trick = np.array([r["plays"] // 4 + 1 for r in out])
    by_trick = {int(t): {"n": int((trick == t).sum()),
                         **{lk: {"median": float(np.median([r[lk] for r, tt in zip(out, trick) if tt == t])),
                                 "zero": float(np.mean([r[lk] == 0 for r, tt in zip(out, trick) if tt == t]))}
                            for lk in ("consistent", "uniform")}}
                for t in range(1, 8) if (trick == t).any()}
    return {"every": every, "by_trick": by_trick, "rows": out}


# -- the summary ----------------------------------------------------------------------

def _rows(run: str):
    with open(Path(run) / "decisions.jsonl") as f:
        for line in f:
            yield json.loads(line)


def _games(run: str) -> list[dict]:
    with open(Path(run) / "games.jsonl") as f:
        return [json.loads(line) for line in f]


def pair_margins(run: str) -> dict[tuple[int, int], float]:
    """(base seed, game) → the reader's mark margin summed over the pair's two games."""
    seed = json.loads((Path(run) / "meta.json").read_text())["seed"]
    out: dict[tuple[int, int], float] = {}
    for g in _games(run):
        out[(seed, g["game"])] = out.get((seed, g["game"]), 0.0) + g["a_marks"] - g["b_marks"]
    return out


def clustered(values: np.ndarray, n_boot: int = 4000, seed: int = 42) -> tuple[float, float, float]:
    """Mean and 95% bootstrap interval of per-cluster values."""
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _rate(k: int, n: int) -> dict:
    return {"n": n, "changed": k, "rate": k / n if n else math.nan}


def summarize(runs: Sequence[str]) -> dict:
    """Pool the batches of each likelihood: marks per game (per game, from
    pair sums halved), the decision change rate and where it sits, and the
    diagnostics."""
    by: dict[str, list[str]] = {}
    for run in runs:
        by.setdefault(json.loads((Path(run) / "meta.json").read_text())["likelihood"], []).append(run)
    out: dict = {}
    pairs: dict[str, dict] = {}
    for lk, rs in sorted(by.items()):
        pm: dict = {}
        for r in rs:
            pm.update(pair_margins(r))
        pairs[lk] = pm
        games = [g for r in rs for g in _games(r)]
        hands = [h for g in games for h in g["hands"]]
        mean, lo, hi = clustered(np.array(list(pm.values())) / 2)
        made = {side: [h["made"] for h in hands if (h["bidder_team"] == h["a_team"]) == (side == "reader")]
                for side in ("reader", "plain")}
        rows = [r for run in rs for r in _rows(run)]
        out[lk] = {
            "batches": [json.loads((Path(r) / "meta.json").read_text()) for r in rs],
            "games": len(games), "game_pairs": len(pm), "hands": len(hands),
            "marks_per_game": {"mean": mean, "lo": lo, "hi": hi},
            "marks_per_hand": float(sum(g["a_marks"] - g["b_marks"] for g in games) / len(hands)),
            "made": {k: {"n": len(v), "rate": float(np.mean(v))} for k, v in made.items()},
            **_decisions(rows),
        }
    if "uniform" in pairs and "consistent" in pairs:
        common = sorted(set(pairs["uniform"]) & set(pairs["consistent"]))
        d = np.array([pairs["uniform"][k] - pairs["consistent"][k] for k in common]) / 2
        mean, lo, hi = clustered(d)
        out["restricted_choice"] = {"game_pairs": len(common), "marks_per_game": {"mean": mean, "lo": lo, "hi": hi}}
    return out


def _decisions(rows: list[dict]) -> dict:
    """The decision change rate, where it sits, and the diagnostics."""
    if not rows:
        return {"decisions": _rate(0, 0)}
    changed = np.array([r["tile"] != r["plain"] for r in rows])
    trick = np.array([len(r["plays"]) // 4 + 1 for r in rows])
    role = np.array([(r["mover"] - r["bidder"]) % 4 for r in rows])
    from scenario.planner import READ_ENUM
    fell = np.array([bool(r["fell_back"]) for r in rows])
    path = np.array(["fallback" if f else "exact" if r["fiber"] <= 512 else
                     "enumerated" if r["fiber"] <= READ_ENUM else "rejection" for r, f in zip(rows, fell)])
    noise = np.array([r["reseed"] != r["plain"] for r in rows])
    out: dict = {
        "decisions": _rate(int(changed.sum()), len(rows)),
        "reseed_baseline": _rate(int(noise.sum()), len(rows)),
        "reseed_baseline_by_trick": {int(t): _rate(int(noise[trick == t].sum()), int((trick == t).sum()))
                                     for t in range(1, 8)},
        "by_trick": {int(t): _rate(int(changed[trick == t].sum()), int((trick == t).sum())) for t in range(1, 8)},
        "by_role": {ROLES[k]: _rate(int(changed[role == k].sum()), int((role == k).sum())) for k in range(4)},
        "reseed_baseline_by_role": {ROLES[k]: _rate(int(noise[role == k].sum()), int((role == k).sum()))
                                    for k in range(4)},
        "by_path": {p: _rate(int(changed[path == p].sum()), int((path == p).sum()))
                    for p in ("exact", "enumerated", "rejection", "fallback")},
        "reseed_baseline_by_path": {p: _rate(int(noise[path == p].sum()), int((path == p).sum()))
                                    for p in ("exact", "enumerated", "rejection", "fallback")},
        "paths_by_trick": {int(t): {p: int(((trick == t) & (path == p)).sum())
                                    for p in ("exact", "enumerated", "rejection", "fallback")}
                           for t in range(1, 8) if (trick == t).any()},
        "rejection_acceptance_median_by_trick": {
            int(t): float(np.median([512 / r["proposals"] for r, p, tt in zip(rows, path, trick)
                                     if p == "rejection" and tt == t]))
            for t in range(1, 8) if ((trick == t) & (path == "rejection")).any()},
        "real_deal_kept": _rate(int(sum(r["real_like"] > 0 for r in rows)), len(rows)),
        "real_deal_kept_by_trick": {int(t): float(np.mean([r["real_like"] > 0 for r, tt in zip(rows, trick) if tt == t]))
                                    for t in range(1, 8) if (trick == t).any()},
    }
    shadow = [r for r in rows if r["consistent"] >= 0]
    if shadow:
        c = np.array([r["tile"] != r["consistent"] for r in shadow])
        tr = np.array([len(r["plays"]) // 4 + 1 for r in shadow])
        out["uniform_vs_consistent"] = {
            **_rate(int(c.sum()), len(shadow)),
            "by_trick": {int(t): _rate(int(c[tr == t].sum()), int((tr == t).sum())) for t in range(1, 8)}}
    # The tiles: which tiles the change moves between, and count.
    from collections import Counter
    pairs = Counter((tile_name(r["plain"]), tile_name(r["tile"])) for r, ch in zip(rows, changed) if ch)
    out["top_changes"] = [{"plain": a, "read": b, "n": n} for (a, b), n in pairs.most_common(15)]
    tiles_in = Counter(t for r, ch in zip(rows, changed) if ch for t in {r["plain"], r["tile"]})
    out["changed_by_tile"] = {tile_name(t): n for t, n in tiles_in.most_common()}
    # The lumps: the mass the reading moves, at changed and unchanged decisions.
    shift = np.abs(np.array([r["mass_read"] for r in rows]) - np.array([r["mass_plain"] for r in rows]))  # [n, 3, 28]
    lumps: dict = {}
    for name in WATCH:
        t = tile_id(name)
        lumps[name] = {LUMPS[j]: {"changed": float(shift[changed, j, t].mean()) if changed.any() else math.nan,
                                  "unchanged": float(shift[~changed, j, t].mean())} for j in range(3)}
    total = shift.sum(axis=2)                     # [n, 3]: summed over tiles
    lumps["every tile"] = {LUMPS[j]: {"changed": float(total[changed, j].mean()) if changed.any() else math.nan,
                                      "unchanged": float(total[~changed, j].mean())} for j in range(3)}
    per_tile = shift.sum(axis=1)                  # [n, 28]
    lift = (per_tile[changed].mean(axis=0) - per_tile[~changed].mean(axis=0)) if changed.any() else np.zeros(28)
    out["mass_shift"] = lumps
    out["mass_shift_lift_top"] = [{"tile": tile_name(int(t)), "lift": float(lift[t])} for t in np.argsort(-lift)[:8]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scale")
    p.add_argument("--pour", required=True)
    p.add_argument("--net", required=True)
    p = sub.add_parser("cost")
    p.add_argument("--net", required=True)
    p.add_argument("--positions", type=int, default=8)
    p.add_argument("--out")
    p = sub.add_parser("arena")
    p.add_argument("--net", required=True)
    p.add_argument("--likelihood", choices=("consistent", "uniform"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--games", type=int, required=True, help="game pairs")
    p.add_argument("--out", required=True)
    p = sub.add_parser("ess")
    p.add_argument("--net", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--out", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "scale":
        print(json.dumps(scale(a.pour, a.net), indent=1))
    elif a.cmd == "cost":
        r = cost(a.net, a.positions)
        print(json.dumps({k: v for k, v in r.items() if k != "positions"}, indent=1))
        if a.out:
            Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
    elif a.cmd == "arena":
        print(json.dumps(arena(a.net, a.likelihood, a.seed, a.games, a.out), indent=1))
    elif a.cmd == "ess":
        r = ess(a.net, a.runs, a.every)
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r["by_trick"], indent=1))
    else:
        s = summarize(a.runs)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        print(json.dumps({k: (v.get("marks_per_game"), v.get("decisions")) for k, v in s.items()}, indent=1))


if __name__ == "__main__":
    main()
