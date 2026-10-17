# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The book's Ch. 9 at the table (#189): doubles trump and no trump, priced
in marks per hand. PROVISIONAL (random field for 7-9).

The book offers two more declarations to a hand with four doubles or more
(W42-p075-b, W42-p079-a): doubles as trumps when a pip trump would leave a
high off (W42-p078-b), no trump when the hand holds neither the 6-6 nor a high
off and no trump prices above every pip suit (W42-p080-a). #155 built the
bidder that follows those conditions (`arena.bidders.MenuBidder`) and the
seat that can play what it declares (`arena.walt_play.MenuWalt`, deployed Zeb
with `DTNT=random`). This study prices the menu at the table.

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Runner.* The arena's ledger (`arena.ledger.run_ledger`), not
`scenario.book_batches`' games: the question is per hand, the deals must be
filtered to the chair's four doubles (about 4.3% of deals), and the ledger
pairs both arms on the same deal and shaker with the same auction dice. Every
hand is played from 0-0 in marks to 7, the straight auction (a hand nobody
bids is passed out and moves nothing). `book_batches.queue` runs the batches.

*Players.* Every seat plays deployed Zeb 0.3.0 (`scenario.small_n.player()`,
160 worlds, seed 0) as `MenuWalt`: the field net under a pip trump, its
random-field twin under doubles trump or no trump, at every seat, in both
arms, on both teams, the declarer included (the Mayor's symmetry ruling on
#155). The chair is seat 0 at every shaker; seats 1-3 bid by the arena's
`HeuristicBidder` (caution 0).
- **a**: the chair bids by `MenuBidder` (the book's full menu);
- **b**: the chair bids by `HeuristicBidder` (pip trumps only).
The ledger's chance arm (random bidder, lowest legal tile) is played alongside
and reported as the baseline.

*Deals.* `arena.ledger.deals` from each batch's seed, filtered to the deals
where seat 0 holds four doubles or more (`four_doubles`).

*Primary.* The paired mean of a - b in marks per hand for the chair's side, a
deal's value its mean over the four shakers. **The book predicts > 0.**
Interval: the house's 95% percentile bootstrap over deals (10,000 resamples,
`scenario.small_n.contrast`). The bar: above 0 = the menu pays at this table;
below 0 = it costs; else a null with its MDE.

*Secondaries.*
1. The same, restricted to the deals where the menu's choice differs from the
   heuristic's (`menu_kind` is DT or NT: a different declaration, or a
   different bid ceiling below one mark), with the count. On every other deal
   the two chairs bid and declare alike, so a - b is exactly 0 there (the
   players are deterministic and batch-independent, [[estimator-calibration-2026-10-16]]):
   the summary checks it (`unchanged_nonzero` must be 0).
2. Points per hand (`settle_points`), all deals and the changed deals.
3. The split by what the menu chose (DT, NT, unchanged), with counts.
Mechanism, not registered as tests: the deals where the table differed (a's
calls or contract differ from b's at some shaker), the chair's side's
contracts under DT and NT and the share made.

*Size.* From the timing pilot (`PILOT` = seed 4890, 96 filtered deals,
excluded): 170 s wall, 549 CPU-s (3.2 cores) under `cpu-slot --threads 4` at a
machine load near 20, so 1.8 s wall a deal; 15 of its 96 deals were changed
(the census over all 51,108 four-double hands says 18.3%: DT 6.9%, NT 11.4%).
Its per-deal SDs, used for sizing only: 0.20 marks on all deals, 0.49 on the
changed deals (taken as 0.6 for the sizing), 25 points on the changed deals.
`DEALS` = 80 a batch (about 140 s, half the 300 s cap) and `SEEDS` = 4901-5188:
**288 batches, 23,040 four-double deals**, about 2.8 hours at 4 batches at
once. Expected changed deals about 4,200: MDE (80% power, two-sided 5%,
2.8 SD / sqrt n) about 0.026 marks per hand on the changed deals, 0.004 on
the primary, 1.1 points per hand on the changed deals; about 0.04 marks per
hand within DT (about 1,600 deals) and 0.03 within NT (about 2,600).
Every batch through `scripts/cpu-slot --threads 4`, up to 4 at once, under
its 300 s cap.

*Batch-drop rule.* A batch that exits nonzero (the cap included) is dropped,
never rerun, and reported.

*Label.* Every number is "provisional (random field for 7-9)": under doubles
trump and no trump every seat plays the random field, not a net. The study is
re-run with Lem's first field-v3 net (#185, G4) once that net covers
declarations 7-9 (#177).

    python -u -m scenario.ch9 arena --arm menu --seed S --games DEALS --out DIR
    python -u -m scenario.ch9 queue --root DIR [--workers 4] [--pilot]
    python -u -m scenario.ch9 summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from scenario import book_batches

LABEL = "provisional (random field for 7-9)"
CHAIR = 0
ARM = "menu"                  # one arm per batch: the ledger holds a and b
PILOT = (4890,)               # the timing pilot, excluded
DEALS = 80                    # filtered deals per batch
SEEDS = tuple(range(4901, 5189))         # 288 batches, 23,040 deals
THREADS = 4                   # cores a batch declares to cpu-slot
WORKERS = 4                   # batches offered at once


def four_doubles(deal) -> bool:
    """The book's condition to consider Ch. 9: the chair holds four doubles or more."""
    from forge.oracle.tables import DOMINO_IS_DOUBLE

    return sum(DOMINO_IS_DOUBLE[t] for t in deal[CHAIR]) >= 4


def menu_kind(hand: Sequence[int]) -> str:
    """What the menu chose for `hand` against the heuristic's choice: "DT" or
    "NT" where the declaration or the bid it would go to differs, else "same"
    (then `MenuBidder` bids and declares exactly as `HeuristicBidder`)."""
    from arena.auction import ONE_MARK
    from arena.hand_metrics import best_trump, menu_choice
    from forge.oracle.declarations import DOUBLES_TRUMP, NOTRUMP

    pip, menu = best_trump(tuple(hand), 3), menu_choice(tuple(hand), 3)
    heuristic = None if pip is None else (pip.trump, min(ONE_MARK, pip.bid_ceiling))
    chosen = None if menu is None else (menu.decl, min(ONE_MARK, menu.bid_ceiling))
    if chosen == heuristic:
        return "same"
    return {DOUBLES_TRUMP: "DT", NOTRUMP: "NT"}[chosen[0]]


def seats():
    """(field, a, b): every seat MenuWalt, the field and b bidding heuristic, a the menu."""
    from arena.bidders import HeuristicBidder, MenuBidder
    from arena.ledger import Seat
    from arena.walt_play import MenuWalt
    from scenario.small_n import player

    play = MenuWalt(player())
    name = "zeb-0.3.0:DTNT=random"
    field = tuple(Seat(HeuristicBidder(), play, f"heuristic+{name}") for _ in range(4))
    return field, Seat(MenuBidder(), play, f"menu+{name}"), Seat(HeuristicBidder(), play, f"heuristic+{name}")


def arena(seed: int, deals: int, out: str) -> dict:
    """One batch: the ledger of a against b in the chair on `deals` filtered
    deals from `seed`, written with its cost as `ledger.json` and `meta.json`."""
    from arena import cost
    from arena.ledger import run_ledger

    field, a, b = seats()
    start = cost.snapshot()
    ledger = run_ledger(field=field, a=a, b=b, n=deals, seed=seed, seats=CHAIR, keep=four_doubles)
    spent = cost.since(start)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "ledger.json").write_text(json.dumps(ledger) + "\n")
    meta = {"arm": ARM, "seed": seed, "deals": deals, "filter": f"seat {CHAIR} holds four doubles or more",
            "label": LABEL, "wall_s": round(spent.pop("wall_s"), 1), **spent, "load_avg": list(os.getloadavg()),
            "players": ledger["players"], "field": ledger["field"]}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def _table_differed(rows: list[dict]) -> bool:
    by = {(r["shaker"], r["arm"]): (r["calls"], r["contract"]) for r in rows}
    return any(by[(s, "a")] != by[(s, "b")] for s in range(4))


def summarize(runs: Sequence[str]) -> dict:
    from arena.auction import contract_points
    from arena.ledger import deal_differences
    from scenario.small_n import contrast

    marks: dict[tuple[int, int], float] = {}
    points: dict[tuple[int, int], float] = {}
    chance: dict[tuple[int, int], float] = {}
    kind: dict[tuple[int, int], str] = {}
    differed: set[tuple[int, int]] = set()
    contracts = {k: {"a": 0, "made": 0} for k in ("DT", "NT")}
    batches = []
    for run in runs:
        meta = json.loads((Path(run) / "meta.json").read_text())
        ledger = json.loads((Path(run) / "ledger.json").read_text())
        seed = meta["seed"]
        batches.append({x: meta[x] for x in ("seed", "wall_s", "cpu_s", "load_avg") if x in meta})
        for d, v in deal_differences(ledger, "marks").items():
            marks[(seed, d)] = float(v)
        for d, v in deal_differences(ledger, "points").items():
            points[(seed, d)] = float(v)
        # The chance baseline: a against the chance arm, read as a - b with chance as b.
        as_chance = {**ledger, "rows": [dict(r, arm="b") if r["arm"] == "chance" else r
                                        for r in ledger["rows"] if r["arm"] != "b"]}
        for d, v in deal_differences(as_chance, "marks").items():
            chance[(seed, d)] = float(v)
        rows: dict[int, list[dict]] = {}
        for r in ledger["rows"]:
            rows.setdefault(r["deal"], []).append(r)
        for d, rs in rows.items():
            hand = [_tile_id(t) for t in rs[0]["hands"][CHAIR]]
            kind[(seed, d)] = menu_kind(hand)
            if _table_differed(rs):
                differed.add((seed, d))
            for r in rs:
                c = r["contract"]
                if r["arm"] == "a" and c is not None and c["declaration"] in contracts and c["declarer"] % 2 == CHAIR % 2:
                    contracts[c["declaration"]]["a"] += 1
                    contracts[c["declaration"]]["made"] += r["outcome"]["declaring"] >= contract_points(c["bid"])

    def read(values: dict, keys) -> dict:
        r = contrast({"x": {k: values[k] for k in keys}}, [(1.0, "x")]) if keys else {}
        if r:
            r["deals"] = r.pop("game_pairs")
            r["sd_per_deal"] = r.pop("sd_per_pair")
        return {"label": LABEL, **r}

    every = sorted(marks)
    changed = [k for k in every if kind[k] != "same"]
    out: dict = {"label": LABEL, "batches": batches, "deals": len(every), "registered": {}, "mechanism": {}}
    reg = out["registered"]
    reg["primary: a - b, marks per hand, four-double deals"] = read(marks, every)
    reg["changed deals (menu chose DT or NT): a - b, marks per hand"] = read(marks, changed)
    reg["points per hand: a - b, four-double deals"] = read(points, every)
    reg["points per hand: a - b, changed deals"] = read(points, changed)
    for k in ("DT", "NT", "same"):
        keys = [x for x in every if kind[x] == k]
        reg[f"split {k}: a - b, marks per hand"] = read(marks, keys)
    reg["chance baseline: a - chance, marks per hand"] = read(chance, every)
    out["mechanism"] = {
        "label": LABEL,
        "menu_kind_counts": {k: sum(kind[x] == k for x in every) for k in ("DT", "NT", "same")},
        "table_differed": len(differed),
        "unchanged_nonzero": sum(1 for x in every if kind[x] == "same" and marks[x] != 0),
        "chair_side_contracts_in_a": contracts,
    }
    return out


def _tile_id(name: str) -> int:
    from forge.oracle.tables import DOMINOES

    high, low = (int(x) for x in name.split("-"))
    return next(t for t, (h, lo) in enumerate(DOMINOES) if (h, lo) == (high, low))


def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch: (arm, base seed)."""
    return [(ARM, s) for s in (PILOT if pilot else SEEDS)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("arena")
    p.add_argument("--arm", choices=(ARM,), default=ARM)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--games", type=int, default=DEALS, help="filtered deals in the batch")
    p.add_argument("--out", required=True)
    p = sub.add_parser("queue")
    p.add_argument("--root", required=True)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--games", type=int, default=DEALS, help="filtered deals a batch")
    p = sub.add_parser("summarize")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "arena":
        print(json.dumps(arena(a.seed, a.games, a.out), indent=1), flush=True)
    elif a.cmd == "queue":
        Path(a.root).mkdir(parents=True, exist_ok=True)
        book_batches.queue("scenario.ch9", a.root, jobs(a.pilot), a.workers, a.games, threads=THREADS)
    else:
        r = summarize(a.runs)
        Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r["registered"], indent=1), flush=True)


if __name__ == "__main__":
    main()
