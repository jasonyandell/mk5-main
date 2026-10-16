# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Pour A: positions with splats on all nine playable declarations.

    python -u -m lem.pour.a chunk --seed 400001 --deals 5000 --out DIR [--pip-spec S] [--dtnt-spec S]
    python -u -m lem.pour.a card DIR...          # the coverage counts over finished chunks
    python -u -m lem.pour.a merge OUT DIR...     # one pour per half (OUT/pip, OUT/dtnt) from finished chunks

A chunk is `deals` seeded deals from `seed` (deal i is `deal(seed + i)`, as
every pour numbers them). Seat 0 bids 30 and declares by the book's menu
(`arena.hand_metrics.menu_choice`, #155): the heuristic pip choice below four
doubles, doubles trump or no trump from four by Chapter 9's conditions; a
hand that calls no suit declares the pours' pip bidder's trump. The book's
rule fires rarely, so a **coverage floor**: on a deal the book declares in a
pip, a seeded coin (`random.Random(f"pour-a:{seed}")`, one stream per chunk)
forces doubles trump or no trump, alternating, on one deal in `share`. The
forced deals are listed in the chunk's `chunk.json` and in each pour's
manifest (`forced_deals`, by deal seed), so training can weight or exclude
them; the book's own declarations are unlabelled.

The pip deals are planned over the shipped field net (`k0,n512,F=`), the
doubles and no-trump deals over the one-rung Walt (`k1,n512,m8`), which is
what the zeb package does for those games; each half is its own pour
directory (`DIR/pip`, `DIR/dtnt`) with the usual `pour.npz` and manifest, and
a deal's spec is the half it sits in. Deterministic budgets throughout (FAIR
CLOCKS); nothing reads a hidden hand (the planner's own information model).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from arena.hand_metrics import menu_choice
from arena.walt_play import _declared_trump, parse_walt_spec
import numpy as np

from scenario.atlas import _COLUMNS, N_PLAYS, Pour, _git_head, calibration, calibration_table, deal, load, pour, save
from scenario.compiled import CompiledWalt
from scenario.field import FieldNet
from scenario.rules import tiles_of

DOUBLES_TRUMP, NO_TRUMP = 7, 9
BIDDER, BID = 0, 30
SHARE = 8                                   # one pip deal in SHARE is forced to 7 or 9
NET_BLOB = Path(__file__).resolve().parents[2] / "zeb" / "net" / "margin_d250k_h128x64.fnet"
PIP_SPEC = "k0,n512,F={net}"
DTNT_SPEC = "k1,n512,m8"


@dataclass(frozen=True)
class Planned:
    seed: int           # the deal's seed, `deal(seed)`
    book: int           # what the book's menu declares for seat 0's hand
    decl: int           # what is played: the book's, or the forced 7 / 9
    forced: bool


def declare_book(hand: int) -> int:
    """The book's menu for a hand mask, with no minimum trump length, so below
    four doubles it is exactly the earlier pours' pip bidder (`_declared_trump`,
    `best_trump(hand, 0)`) and the pip deals keep their old declarations."""
    e = menu_choice(tuple(tiles_of(hand)), 0)
    return _declared_trump(hand) if e is None else e.decl


def plan(seed: int, n: int, share: int = SHARE) -> list[Planned]:
    """The declarations of a chunk: deterministic in (seed, n, share)."""
    rng = random.Random(f"pour-a:{seed}")
    out = []
    flip = 0
    for i in range(n):
        s = seed + i
        book = declare_book(deal(s)[BIDDER])
        forced = book <= 6 and rng.randrange(share) == 0
        if forced:
            decl = (DOUBLES_TRUMP, NO_TRUMP)[flip % 2]
            flip += 1
        else:
            decl = book
        out.append(Planned(s, book, decl, forced))
    return out


def threads() -> int:
    """The OS threads the core may use: what `scripts/cpu-slot` declared
    (`OMP_NUM_THREADS`, set from `--threads`), else every core (0)."""
    return int(os.environ.get("OMP_NUM_THREADS") or 0)


def _walt(spec: str) -> CompiledWalt:
    walt = parse_walt_spec(spec).walt
    if not isinstance(walt, CompiledWalt):
        raise SystemExit("the pour needs the compiled core")
    walt.threads = threads()          # the compiled core's pool is sized per caller, not by the env
    return walt


def _pour_half(planned: list[Planned], spec: str, out: Path, log) -> Pour | None:
    if not planned:
        return None
    deals = [deal(p.seed) for p in planned]
    decls = iter(p.decl for p in planned)
    t0 = time.time()

    def progress(wave: int, positions: int) -> None:
        log(f"    {out.name}: wave {wave:2d}, {positions:,} positions, {time.time() - t0:.0f}s")

    commit = _git_head()
    p = pour(deals, BIDDER, BID, lambda _hand: next(decls), _walt(spec), progress)
    save(p, out, spec=spec, seed=planned[0].seed, fiber=False, commit=commit,
         deal_seeds=[q.seed for q in planned], forced_deals=[q.seed for q in planned if q.forced],
         book_decls=[q.book for q in planned], pour="A", wall_seconds=round(time.time() - t0, 1))
    return p


def chunk(seed: int, n: int, out: str | Path, pip_spec: str = PIP_SPEC, dtnt_spec: str = DTNT_SPEC,
          share: int = SHARE, log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    net_npz = out / "net.npz"
    if "{net}" in pip_spec:
        FieldNet.from_bytes(NET_BLOB.read_bytes()).save(net_npz)
        pip_spec = pip_spec.format(net=net_npz)
    planned = plan(seed, n, share)
    pips = [p for p in planned if p.decl <= 6]
    dtnt = [p for p in planned if p.decl > 6]
    counts = {"deals": n, "pip": len(pips), "dtnt": len(dtnt),
              "book_dt": sum(p.book == DOUBLES_TRUMP for p in planned), "book_nt": sum(p.book == NO_TRUMP for p in planned),
              "forced_dt": sum(p.forced and p.decl == DOUBLES_TRUMP for p in planned),
              "forced_nt": sum(p.forced and p.decl == NO_TRUMP for p in planned),
              "by_decl": {d: sum(p.decl == d for p in planned) for d in range(10)}}
    log(f"chunk {seed}: {n} deals, {counts['pip']} pip, {counts['dtnt']} doubles/no-trump "
        f"(book {counts['book_dt']}+{counts['book_nt']}, forced {counts['forced_dt']}+{counts['forced_nt']})")
    t0 = time.time()
    a = _pour_half(pips, pip_spec, out / "pip", log)
    b = _pour_half(dtnt, dtnt_spec, out / "dtnt", log)
    record = {"pour": "A", "seed": seed, "deals": n, "share": share, "pip_spec": pip_spec, "dtnt_spec": dtnt_spec,
              "counts": counts, "positions": {"pip": 0 if a is None else len(a), "dtnt": 0 if b is None else len(b)},
              "planned": [asdict(p) for p in planned], "wall_seconds": round(time.time() - t0, 1), "commit": _git_head()}
    (out / "chunk.json").write_text(json.dumps(record, indent=1))
    log(f"chunk {seed} done in {record['wall_seconds']}s: {record['positions']}")
    for name, p in (("pip", a), ("dtnt", b)):
        if p is not None:
            log(f"  {name} calibration:\n" + calibration_table(calibration(p)))
    return record


def card(dirs: list[str | Path]) -> dict:
    """The coverage counts over finished chunks."""
    total = {"chunks": 0, "deals": 0, "pip": 0, "dtnt": 0, "book_dt": 0, "book_nt": 0, "forced_dt": 0, "forced_nt": 0,
             "positions_pip": 0, "positions_dtnt": 0, "by_decl": {str(d): 0 for d in range(10)}, "wall_seconds": 0.0}
    for d in dirs:
        f = Path(d) / "chunk.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        total["chunks"] += 1
        for k in ("deals", "pip", "dtnt", "book_dt", "book_nt", "forced_dt", "forced_nt"):
            total[k] += r["counts"][k]
        for k, v in r["counts"]["by_decl"].items():
            total["by_decl"][str(k)] += v
        total["positions_pip"] += r["positions"]["pip"]
        total["positions_dtnt"] += r["positions"]["dtnt"]
        total["wall_seconds"] += r["wall_seconds"]
    if total["deals"]:
        total["forced_share"] = (total["forced_dt"] + total["forced_nt"]) / total["deals"]
        total["book_dtnt_share"] = (total["book_dt"] + total["book_nt"]) / total["deals"]
    return total


def concat(pours: list[Pour]) -> Pour:
    """The pours' columns in order, each position's `deal` offset by the deals before it."""
    if len({(p.bidder, p.bid, p.utility) for p in pours}) != 1:
        raise ValueError("pours of one contract and utility merge")
    cols = {c: [] for c in _COLUMNS}
    first = 0
    for p in pours:
        for c in _COLUMNS:
            a = getattr(p, c)
            cols[c].append(a + first if c == "deal" else a)
        first += len(p.hands)
    return Pour(pours[0].bidder, pours[0].bid, pours[0].utility, **{c: np.concatenate(v) for c, v in cols.items()})


def merge(out: str | Path, dirs: list[str | Path], log=print) -> dict:
    """One pour directory per half from the finished chunks, with a manifest
    that keeps every chunk's seed, deal seeds, forced deals and book declarations."""
    out = Path(out)
    result = {}
    for half in ("pip", "dtnt"):
        pours, about = [], {"pour": "A", "half": half, "chunks": [], "deal_seeds": [], "forced_deals": [], "book_decls": [], "specs": set()}
        for d in sorted(Path(x) for x in dirs):
            if not (d / "chunk.json").exists() or not (d / half / "manifest.json").exists():
                continue
            m = json.loads((d / half / "manifest.json").read_text())
            pours.append(load(d / half))
            about["chunks"].append(d.name)
            about["deal_seeds"] += m["deal_seeds"]
            about["forced_deals"] += m["forced_deals"]
            about["book_decls"] += m["book_decls"]
            spec = m["spec"]
            if "F=" in spec:
                # every chunk wrote its own copy of the shipped net; the merged pour names one
                net_src = Path(spec.split("F=", 1)[1].split(",")[0])
                shared = out / "net.npz"
                if not shared.exists():
                    out.mkdir(parents=True, exist_ok=True)
                    shared.write_bytes(net_src.read_bytes())
                elif shared.read_bytes() != net_src.read_bytes():
                    raise AssertionError(f"{d.name} was poured over a different net")
                spec = spec.replace(str(net_src), str(shared))
            about["specs"].add(spec)
        if not pours:
            continue
        p = concat(pours)
        if len(about["deal_seeds"]) != len(p.hands):
            raise AssertionError("one deal seed per deal")
        about["specs"] = sorted(about["specs"])
        if len(about["specs"]) != 1:
            raise AssertionError(f"the {half} chunks were poured under several specs: {about['specs']}")
        about["spec"] = about["specs"][0]
        forced = len(about["forced_deals"])
        about["declare"] = (
            "declares by the book's menu (`arena.hand_metrics.menu_choice`, no minimum trump length: the pip "
            "heuristic below four doubles, doubles trump or no trump from four by Chapter 9's conditions), with "
            f"one pip deal in {SHARE} forced to doubles or no trump, alternating, and listed in `forced_deals` "
            f"({forced:,} of this split's {len(p.hands):,} deals)")
        save(p, out / half, **about, commit=_git_head())
        result[half] = {"deals": len(p.hands), "positions": len(p), "chunks": len(about["chunks"]), "forced": len(about["forced_deals"])}
        log(f"{out / half}: {len(p.hands):,} deals, {len(p):,} positions from {len(about['chunks'])} chunks")
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -u -m lem.pour.a")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chunk")
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--deals", type=int, required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--pip-spec", default=PIP_SPEC)
    c.add_argument("--dtnt-spec", default=DTNT_SPEC)
    c.add_argument("--share", type=int, default=SHARE)
    k = sub.add_parser("card")
    k.add_argument("dirs", nargs="+")
    m = sub.add_parser("merge")
    m.add_argument("out")
    m.add_argument("dirs", nargs="+")
    a = ap.parse_args(argv)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    if a.cmd == "chunk":
        chunk(a.seed, a.deals, a.out, a.pip_spec, a.dtnt_spec, a.share, log)
    elif a.cmd == "merge":
        print(json.dumps(merge(a.out, a.dirs, log), indent=1))
    else:
        print(json.dumps(card(a.dirs), indent=1))


if __name__ == "__main__":
    main()
