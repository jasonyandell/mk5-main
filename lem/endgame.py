# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""The Walt endgame probe (#244): Walt's tile against the exact optimum of Walt's own model.

At a decision with at most three own tiles, Walt plans one contingent plan over its
scenarios: uniform on the viewer's void-consistent fiber (sampled when the fiber holds
more than the live rung's n, enumerated otherwise), the other seats played by its model
(the field net at k = 0, rung-0 Walts at k = 1), its own later plays chosen once per cell
of scenarios it cannot tell apart. This module asks how far that tile is from the exact
information-state optimum of the same model on the whole fiber.

  * `course(walt, state, bid)` writes the decision as a family course
    (`silas-family-course-v1`): the viewer's decisions, keyed by what it has seen, over
    every world of the fiber, the other seats played by `walt`'s own model
    (`Walt._modeled_move`, read only), each choice's mass a count of worlds and each
    payoff a count of worlds in which the viewer's side gets its way (the bidders make,
    the defenders set). `silas family` solves it exactly; `solve` is the Python reference.
  * `probe` decides one position three ways: Walt as configured, Walt's twin whose live
    rung enumerates the fiber (Walt's own exact belief, floor off), and the course.
  * `run` / `silas` / `report`: a pour's endgame positions, resumable; Silas's solve
    joined and checked against the reference; the rates and regrets with deal-resampled
    intervals and the uniform-tile chance baseline.

STRATEGY-FUSION FENCE: nothing here is clairvoyant. Every maximum is over the viewer's
plays at one information set (one tile for every world the viewer cannot tell apart);
no per-world value is ever taken as the seat's.

Run (from the repo root, through scripts/cpu-slot):
  python -u -m lem.endgame run POUR WALT OUT.jsonl --deals A:B [--workers W] [--budget S]
  python -u -m lem.endgame silas OUT.jsonl
  python -u -m lem.endgame report OUT.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from scenario.field import FieldNet
from scenario.fiber import Fiber
from scenario.planner import HAND_TOTAL, Config, Rung, Walt, situation_from_state
from scenario.record import Record, remaining_hand
from scenario.rules import rules_for, tiles_of

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / "zeb" / "net" / "margin_d250k_h128x64.fnet"
SILAS = ROOT / "silas" / "target" / "release" / "silas"
ENUMERATE = 4096   # the exact twin's live rung: above every fiber at three tricks left (at most 1,680)
SCHEMA = "silas-family-course-v1"


def walt_config(name: str) -> Config:
    """The Walts probed. `zeb` is deployed Zeb in its arena form
    (`walt:k0,n512,F=margin_d250k_h128x64,f=16`); `champion` is `walt:k1,n512,m8`."""
    if name == "zeb":
        return Config((Rung(512, None, 16),), "make", field=FieldNet.from_bytes(NET.read_bytes()))
    if name == "champion":
        return Config((Rung(8, 4), Rung(512, None, 16)), "make")
    raise ValueError(f"no Walt named {name!r}")


def exact_twin(cfg: Config) -> Config:
    """The same Walt with its live rung enumerating every fiber of the endgame: Walt's own
    exact belief, and so, by Walt's rule, no floor."""
    live = cfg.rungs[-1]
    return replace(cfg, rungs=cfg.rungs[:-1] + (Rung(ENUMERATE, live.horizon, live.floor),))


def _modelled(cfg: Config) -> None:
    """Refused unless the other seats' play is a function of their own information: a
    field or a rung below, deterministic, the worlds uniform, no overlay."""
    if cfg.level == 0 and cfg.field is None:
        raise ValueError("at k = 0 with the random field the others are dice, not a model")
    if (cfg.field_eps, cfg.draw, cfg.reading, cfg.convention, cfg.void_miss) != (0.0, "uniform", None, None, 0.0):
        raise ValueError("the probe reads a uniform, deterministic, overlay-free Walt")
    if cfg.rungs[-1].horizon is not None:
        raise ValueError("the live rung must plan to the end of the hand")


# -- the course ------------------------------------------------------------------------


def course(walt: Walt, state, bid: int) -> dict:
    """The decision as a family course over the whole fiber, under `walt`'s model."""
    cfg = walt.config
    _modelled(cfg)
    sit = situation_from_state(state, bid)
    rec, viewer, bt = sit.record, sit.viewer, sit.bidder_team
    rules = rules_for(state.decl_id)
    own = remaining_hand(state, viewer)
    sizes = (rec.hand_size(0), rec.hand_size(1), rec.hand_size(2), rec.hand_size(3))
    worlds = Fiber(viewer, own, rec.played, sizes, rec.forbid).worlds()
    bidding = viewer % 2 == bt
    k = cfg.level

    def model(seat: int, world: int, r: Record) -> int:
        return walt._modeled_move(k - 1, seat, worlds[world][seat] & ~r.played, r, rules, bid, bt)

    def settled(r: Record) -> int | None:
        """1 or 0 for the viewer's side once the contract is decided (Walt's `make`)."""
        if r.banked[bt] >= bid:
            return 1 if bidding else 0
        if r.banked[1 - bt] > HAND_TOTAL - bid or r.finished:
            return 0 if bidding else 1
        return None

    def playout(r: Record, world: int) -> int:
        while (u := settled(r)) is None:
            r = r.advance(model(r.seat, world, r), rules)
        return u

    def observe(r: Record, alive: list[int], seen: tuple):
        """The cells the viewer can tell apart at its next turn, with what it saw."""
        if r.seat == viewer:
            yield seen, r, alive
            return
        seat, split = r.seat, {}
        for w in alive:
            split.setdefault(model(seat, w, r), []).append(w)
        for t in sorted(split):
            yield from observe(r.advance(t, rules), split[t], seen + ((seat, t),))

    choices: list[dict] = []
    memo: dict[tuple, int] = {}

    def decide(r: Record, alive: list[int]) -> int:
        key = (r.key(), tuple(alive))
        if key in memo:
            return memo[key]
        hand = own & ~r.played
        menu = tiles_of(rules.legal(hand, r.led))
        i = memo[key] = len(choices)
        choices.append({})
        u = settled(r)
        if u is not None:
            choices[i] = {"menu": menu, "mass": str(len(alive)), "payoff": [str(u * len(alive))] * len(menu),
                          "blocks": []}
            return i
        payoff, blocks = [], []
        for t in menu:
            after = r.advance(t, rules)
            if hand == 1 << t:
                payoff.append(str(sum(playout(after, w) for w in alive)))
                continue
            payoff.append("0")
            for seen, nxt, cell in observe(after, alive, ()):
                blocks.append({"play": t, "seen": [list(p) for p in seen], "next": decide(nxt, cell)})
        choices[i] = {"menu": menu, "mass": str(len(alive)), "payoff": payoff, "blocks": blocks}
        return i

    root = decide(rec, list(range(len(worlds))))
    return {"schema": SCHEMA, "root": root, "choices": choices}


def solve(c: dict) -> dict[int, Fraction]:
    """The Python reference: each root play's exact value, the viewer side's success
    probability given the root's mass, with the best play at every later decision."""
    best: dict[int, int] = {}

    def value(i: int) -> list[int]:
        ch = c["choices"][i]
        return [int(p) + sum(top(b["next"]) for b in ch["blocks"] if b["play"] == t)
                for t, p in zip(ch["menu"], ch["payoff"])]

    def top(i: int) -> int:
        if i not in best:
            best[i] = max(value(i))
        return best[i]

    root = c["choices"][c["root"]]
    mass = int(root["mass"])
    return {t: Fraction(v, mass) for t, v in zip(root["menu"], value(c["root"]))}


# -- one position ------------------------------------------------------------------------


def arms(cfg: Config) -> dict[str, Config]:
    """Walt as configured, and two descriptive arms: the floor off (f = 1), and the live
    rung at 160 worlds, the zeb package's default budget."""
    live = cfg.rungs[-1]
    return {
        "nofloor": replace(cfg, rungs=cfg.rungs[:-1] + (Rung(live.n_worlds, live.horizon, 1),)),
        "n160": replace(cfg, rungs=cfg.rungs[:-1] + (Rung(160, live.horizon, live.floor),)),
    }


def probe(cfg: Config, state, bid: int) -> tuple[dict, dict]:
    """Walt's tile, the exact twin's values, and the course; the row and the course."""
    sit = situation_from_state(state, bid)
    rec, viewer = sit.record, sit.viewer
    own = remaining_hand(state, viewer)
    sizes = tuple(rec.hand_size(s) for s in range(4))
    fiber = Fiber(viewer, own, rec.played, sizes, rec.forbid).size
    live = Walt(cfg)
    t0 = time.process_time()
    tile, q = live.decide(state, bid)
    cpu_live = time.process_time() - t0
    tiles = {name: Walt(c).decide(state, bid)[0] for name, c in arms(cfg).items()}
    twin = Walt(exact_twin(cfg))
    t1 = time.process_time()
    tile_x, q_x = twin.decide(state, bid)
    c = course(twin, state, bid)
    exact = solve(c)
    t2 = time.process_time()
    bidding = viewer % 2 == sit.bidder_team
    # Walt's q is the bidders' make probability; the course is the viewer side's success.
    as_make = {t: (v if bidding else 1 - v) for t, v in exact.items()}
    top = max(exact.values())
    row = {
        "viewer": viewer, "bidding": bidding, "own": own.bit_count(), "plays": rec.plays_so_far(),
        "fiber": fiber, "sampled": fiber > cfg.rungs[-1].n_worlds, "menu": sorted(exact),
        "best": sorted(t for t, v in exact.items() if v == top),
        "exact": {str(t): str(v) for t, v in sorted(exact.items())},
        "tile": tile, "q": {str(t): v for t, v in sorted(q.items())}, "arms": tiles,
        "twin_tile": tile_x, "twin_q": {str(t): v for t, v in sorted(q_x.items())},
        "twin_agrees": all(abs(q_x[t] - float(as_make[t])) < 1e-9 for t in exact),
        "choices": len(c["choices"]), "cpu_live_s": round(cpu_live, 3), "cpu_exact_s": round(t2 - t1, 3),
    }
    return row, c


# -- a pour's endgame -------------------------------------------------------------------


def _positions(pour_dir: str, deals: range):
    """(deal, n_plays, state, bid) for every planned position of a pour with at most
    three own tiles and more than one legal tile, in pour order."""
    from scenario import atlas

    p = atlas.load(pour_dir)
    keep = np.nonzero((p.n_plays >= 16) & (p.n_actions > 1) & np.isin(p.deal, np.array(list(deals))))[0]
    for i in keep:
        pos = p.position(int(i))
        hands = tuple(tuple(tiles_of(h)) for h in pos.hands)
        yield pos.deal, len(pos.plays), atlas._state(hands, pos.bidder, pos.bid, pos.decl_id, pos.plays,
                                                     pos.record), pos.bid, pos.chosen


def _work(args):
    name, deal, n_plays, state, bid, chosen = args
    row, c = probe(walt_config(name), state, bid)
    row.update(deal=deal, n_plays=n_plays, pour_tile=chosen, walt=name)
    c["id"] = f"{name}/{deal}/{n_plays}"
    row["id"] = c["id"]
    return row, c


def run(pour_dir: str, name: str, out: str, deals: range, workers: int, budget: float) -> None:
    """Probe every endgame position of `deals`, appending rows to `out` and courses to
    `out`.courses.jsonl; positions already in `out` are skipped; no new position starts
    after `budget` seconds."""
    from multiprocessing import Pool

    out_p, courses_p = Path(out), Path(out + ".courses.jsonl")
    done = set()
    if out_p.exists():
        done = {json.loads(line)["id"] for line in out_p.read_text().splitlines() if line.strip()}
    todo = [(name, d, k, s, b, ch) for d, k, s, b, ch in _positions(pour_dir, deals) if f"{name}/{d}/{k}" not in done]
    print(f"{len(todo)} positions to probe ({len(done)} done)", flush=True)
    t0, n = time.time(), 0
    with Pool(workers) as pool, out_p.open("a") as f_rows, courses_p.open("a") as f_courses:
        it = pool.imap(_work, todo, chunksize=1)
        for row, c in it:
            f_courses.write(json.dumps(c, separators=(",", ":")) + "\n")
            f_rows.write(json.dumps(row) + "\n")
            n += 1
            if n % 20 == 0:
                f_rows.flush()
                f_courses.flush()
                print(f"{n}/{len(todo)} at {time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 > budget:
                pool.terminate()
                break
    print(f"probed {n} in {time.time() - t0:.0f}s", flush=True)


def silas(out: str) -> None:
    """Solve the pending courses with `silas family`, and join: Silas's root values must
    equal the reference's, exactly, at every position; its quotient counts go into the
    rows. The courses file is then removed: `run` rewrites any course on demand."""
    courses = Path(out + ".courses.jsonl")
    res = subprocess.run([str(SILAS), "family", "--courses", str(courses)], capture_output=True, text=True, check=True)
    reports = {r["id"]: r for r in map(json.loads, res.stdout.splitlines())}
    rows = [json.loads(line) for line in Path(out).read_text().splitlines() if line.strip()]
    bad = 0
    for row in rows:
        r = reports.get(row["id"])
        if r is None:
            continue
        cls = r["classes"][0]
        silas_exact = {str(t): str(Fraction(v)) for t, v in zip(r["menu"], cls["value"])}
        row["silas_agrees"] = silas_exact == row["exact"] and cls["best"] == row["best"]
        bad += not row["silas_agrees"]
        row["quotient"] = r["quotient"]
    Path(out).write_text("".join(json.dumps(row) + "\n" for row in rows))
    courses.unlink()
    print(f"{len(reports)} courses solved by Silas; {bad} disagree with the reference", flush=True)


# -- the report -----------------------------------------------------------------------------


def _stats(rows: list[dict], arm: str | None = None) -> dict[str, np.ndarray]:
    """Per position: Walt's tile (or an arm's) outside the exact maximizing set, its
    regret (the viewer side's success probability given up), and the uniform tile's
    chance of each."""
    out = {"outside": [], "regret": [], "chance_outside": [], "chance_regret": [], "open": [], "deal": []}
    for r in rows:
        v = {int(t): Fraction(x) for t, x in r["exact"].items()}
        top = max(v.values())
        tile = r["tile"] if arm is None else r["arms"][arm]
        out["outside"].append(float(tile not in r["best"]))
        out["regret"].append(float(top - v[tile]))
        out["chance_outside"].append(1 - len(r["best"]) / len(v))
        out["chance_regret"].append(float(sum(top - x for x in v.values()) / len(v)))
        out["open"].append(len(r["best"]) < len(v))
        out["deal"].append(r["deal"])
    return {k: np.array(x) for k, x in out.items()}


def _boot(x: np.ndarray, deal: np.ndarray, reps: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and its deal-resampling 95% percentile interval (each deal's positions together)."""
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    ids, inv = np.unique(deal, return_inverse=True)
    sums, counts = np.bincount(inv, weights=x), np.bincount(inv)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(reps, len(ids)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def report(paths: list[str]) -> str:
    rows = [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line.strip()]
    lines = []
    for walt in sorted({r["walt"] for r in rows}):
        mine = [r for r in rows if r["walt"] == walt]
        checks = (sum(r["twin_agrees"] for r in mine), sum(r.get("silas_agrees", False) for r in mine))
        lines.append(f"## {walt}: {len(mine)} positions from {len({r['deal'] for r in mine})} deals; "
                     f"twin = reference at {checks[0]}, Silas = reference at {checks[1]}")
        lines.append("| stratum | n | open | Walt outside the set | chance | regret (success prob.) | chance regret |")
        lines.append("|---|---:|---:|---|---|---|---|")
        strata = [("all", lambda r: True)]
        for own in (3, 2):
            strata.append((f"{own} tricks left", lambda r, own=own: r["own"] == own))
            strata.append((f"{own} left, sampled", lambda r, own=own: r["own"] == own and r["sampled"]))
            strata.append((f"{own} left, enumerated", lambda r, own=own: r["own"] == own and not r["sampled"]))
        for label, keep in strata:
            sub = [r for r in mine if keep(r)]
            if not sub:
                continue
            s = _stats(sub)
            f = lambda k, scale=1.0: "%.4f [%.4f, %.4f]" % tuple(scale * v for v in _boot(s[k], s["deal"]))
            lines.append(f"| {label} | {len(sub)} | {int(s['open'].sum())} | {f('outside')} | {f('chance_outside')} "
                         f"| {f('regret')} | {f('chance_regret')} |")
        three = [r for r in mine if r["own"] == 3]
        lines.append("")
        lines.append("| arm, 3 tricks left | outside the set | regret |")
        lines.append("|---|---|---|")
        for arm in (None, "nofloor", "n160"):
            s = _stats(three, arm)
            f = lambda k: "%.4f [%.4f, %.4f]" % _boot(s[k], s["deal"])
            lines.append(f"| {arm or 'as configured'} | {f('outside')} | {f('regret')} |")
        solved = [r for r in mine if "quotient" in r]
        if solved:
            lines.append("")
            lines.append("| controller (raw states), open positions | median | mean | max |")
            lines.append("|---|---:|---:|---:|")
            for own in (3, 2):
                sub = [r for r in solved if r["own"] == own and len(r["best"]) < len(r["menu"])]
                for reach in ("every", "path", "plan"):
                    x = np.array([r["quotient"][reach]["raw"]["states"] for r in sub])
                    if len(x):
                        lines.append(f"| {own} left, {reach} (n = {len(x)}) | {np.median(x):.0f} | {x.mean():.1f} "
                                     f"| {x.max()} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lem.endgame")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("pour")
    r.add_argument("walt", choices=("zeb", "champion"))
    r.add_argument("out")
    r.add_argument("--deals", required=True, help="A:B, the pour's deal indices")
    r.add_argument("--workers", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    r.add_argument("--budget", type=float, default=240.0)
    s = sub.add_parser("silas")
    s.add_argument("out")
    p = sub.add_parser("report")
    p.add_argument("out", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        lo, hi = map(int, a.deals.split(":"))
        run(a.pour, a.walt, a.out, range(lo, hi), a.workers, a.budget)
    elif a.cmd == "silas":
        silas(a.out)
    else:
        print(report(a.out))


if __name__ == "__main__":
    main()
