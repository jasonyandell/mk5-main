"""A pour as the splat visualizer's data.

The page draws and counts worlds; it does not plan. Every number it shows is
written here or counted from the worlds written here, read off the pour with
`scenario.atlas`, and checked against the pour's own
invariants on the way out (a position's `q` is reproduced from its splat,
the plays replay to the recorded ending).

A splat is stored as a histogram over the bidder team's final margin. A
hand's 42 points are all banked by someone, so a margin m means the bidding
side banked (42 + m) / 2 and is always even: the 85 integers -42..42 hold 43
reachable values. Bin k is "the bidder team banked k points", margin
2k - 42, and its value is the belief weight of the scenarios that end there.

The splat is also kept world by world, so the page can say what must be true
for any part of it (`atlas.holding_predictors`) and condition the position on
a holding. Per position: `seats`, the three seats hidden from the mover
(ascending); `hidden`, the tiles they hold between them (ascending); and
`holders`, one string of len(worlds) * len(hidden) digits, world-major, digit
j of world r naming the index into `seats` of the seat that holds hidden[j]
in that world. Per move: `outcomes`, one character per world, chr(48 + k)
for "the bidders banked k" (margin 2k - 42), in world order. `weights` is
written only when a position's worlds are not equally weighted, and the
document's `uniform` says that none were (a world's weight is then
`weight / worlds`).

    python -u -m scenario.viz.export scratch/viz/pour_viz_5 --out scenario/viz/data/pour_viz_5.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from scenario.atlas import N_PLAYS, Pour, Position, in_plan_units, load, makes
from scenario.planner import HAND_TOTAL
from scenario.record import empty_record
from scenario.rules import ALL_TILES, rules_for, tiles_of

BINS = HAND_TOTAL + 1  # bin k: the bidder team banked k points


def _num(x: float) -> float | int:
    """An exact JSON number: an int when the value is integral."""
    x = float(x)
    return int(x) if x.is_integer() else x


def splat_bins(pos: Position, a: int) -> list[float | int]:
    """The belief weight of action index `a`'s scenarios, per bin."""
    o = pos.outcomes[a].astype(np.int64)
    if np.any(o % 2):
        raise AssertionError("an odd margin: the hand's 42 points were not all banked")
    k = (o + HAND_TOTAL) // 2
    return [_num(v) for v in np.bincount(k, weights=pos.weights, minlength=BINS)]


def outcome_chars(pos: Position, a: int) -> str:
    """Action index `a`'s outcome in every world, one character each."""
    k = (pos.outcomes[a].astype(np.int64) + HAND_TOTAL) // 2
    return bytes((48 + k).astype(np.uint8)).decode("ascii")


def cell_doc(pos: Position) -> dict:
    """The worlds themselves: who holds each tile the mover cannot see."""
    seats = [s for s in range(4) if s != pos.viewer]
    hidden = tiles_of(ALL_TILES & ~pos.record.played & ~pos.own)
    held = np.stack([(pos.worlds[:, seats] >> np.uint32(t)) & 1 for t in hidden], axis=1)  # [W, H, 3]
    if not np.all(held.sum(axis=2) == 1):
        raise AssertionError(f"deal {pos.deal} play {len(pos.plays)}: a hidden tile is not in exactly one hidden hand")
    digits = held.argmax(axis=2).astype(np.uint8) + 48                                   # [W, H]
    doc = {"seats": seats, "hidden": hidden, "holders": bytes(digits.ravel()).decode("ascii")}
    if not np.all(pos.weights == pos.weights[0]):
        doc["weights"] = [_num(w) for w in pos.weights]
    return doc


def position_doc(pos: Position) -> dict:
    total = float(pos.weights.sum())
    moves = []
    for a, tile in enumerate(pos.actions):
        q = float(in_plan_units(pos, pos.outcomes[a]) @ pos.weights / total)
        if abs(q - float(pos.q[a])) > 1e-9:
            raise AssertionError(f"deal {pos.deal} play {len(pos.plays)}: q is not its splat's mean")
        moves.append({
            "tile": tile,
            "q": float(pos.q[a]),
            "mean": float(pos.outcomes[a].astype(np.float64) @ pos.weights / total),
            "pmake": float(pos.weights[makes(pos, tile)].sum() / total),
            "bins": splat_bins(pos, a),
            "outcomes": outcome_chars(pos, a),
        })
    # The real deal seen through the cell: for each tile the viewer could
    # not see, the share of belief on the seat that really holds it.
    belief = []
    played = pos.record.played
    for seat in range(4):
        if seat == pos.viewer:
            continue
        for t in tiles_of(pos.hands[seat] & ~played):
            there = (pos.worlds[:, seat] >> np.uint32(t)) & 1 == 1
            belief.append([seat, t, round(float(pos.weights[there].sum() / total), 4)])
    return {
        "play": len(pos.plays), "viewer": pos.viewer, "sign": pos.sign,
        "worlds": len(pos.weights), "weight": _num(total), "chosen": pos.chosen,
        "moves": moves, "belief": belief, **cell_doc(pos),
    }


def deal_doc(p: Pour, d: int, seed: int | None) -> dict:
    decl = int(p.decl[d])
    plays = [(int(s), int(t)) for s, t in p.plays[d]]
    rules = rules_for(decl)
    rec = empty_record(p.bidder)
    banked = [list(rec.banked)]
    tricks = []
    for i, (seat, tile) in enumerate(plays):
        if rec.seat != seat:
            raise AssertionError(f"deal {d}: play {i} is seat {seat}'s, the record says {rec.seat}")
        before = rec.banked
        rec = rec.advance(tile, rules)
        banked.append(list(rec.banked))
        if not rec.trick:
            side = rec.leader % 2
            tricks.append({"winner": rec.leader, "points": rec.banked[side] - before[side]})
    if rec.banked != (int(p.banked[d, 0]), int(p.banked[d, 1])):
        raise AssertionError(f"deal {d}: the plays do not replay to the recorded ending")
    bt = p.bidder % 2
    return {
        "seed": seed, "decl": decl,
        "hands": [tiles_of(int(h)) for h in p.hands[d]],
        "plays": [list(x) for x in plays],
        "banked": banked,                       # before play k, k = 0..28
        "tricks": tricks,
        "margin": rec.banked[bt] - rec.banked[1 - bt],
        "positions": [],
    }


def export(p: Pour, manifest: dict | None = None) -> dict:
    """The page's document for a whole pour."""
    manifest = manifest or {}
    seeds: Sequence[int | None] = manifest.get("deal_seeds") or [None] * len(p.hands)
    deals = [deal_doc(p, d, seeds[d]) for d in range(len(p.hands))]
    for pos in p.positions():
        deals[pos.deal]["positions"].append(position_doc(pos))
    return {
        "bidder": p.bidder, "bid": p.bid, "utility": p.utility,
        "threshold": 2 * p.bid - HAND_TOTAL,    # pmake is the mass at margin >= this
        "plays": N_PLAYS,
        "uniform": all("weights" not in x for dl in deals for x in dl["positions"]),
        "source": {k: manifest[k] for k in ("spec", "seed", "commit", "fiber", "wall_seconds")
                   if k in manifest},
        "deals": deals,
    }


def export_dir(src: str | Path) -> dict:
    src = Path(src)
    manifest = json.loads((src / "manifest.json").read_text())
    doc = export(load(src), manifest)
    doc["source"]["pour"] = src.name
    return doc


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -u -m scenario.viz.export")
    ap.add_argument("pour", help="a pour directory (manifest.json + pour.npz)")
    ap.add_argument("--out", required=True, help="the JSON file the page reads")
    args = ap.parse_args(argv)
    doc = export_dir(args.pour)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    n = sum(len(d["positions"]) for d in doc["deals"])
    print(f"{len(doc['deals'])} deals, {n} planned positions -> {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
