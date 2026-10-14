"""Lumps and the value of knowing one tile: where Zeb's move depends on who
holds a hidden tile, and how much knowing it would be worth.

A position's worlds (the pour's sample of the mover's cell, or the whole
fiber when it is small) carry every legal move's final margin under Zeb's
own plan: the root move is varied, the continuation is the plan's, fixed
([[metasplat-2026-10-13]]). No perfect-information value enters anywhere.

**Lumps.** For a tile T hidden from the mover (not played, not in his hand),
the worlds split three ways by which other seat holds T: the mover's left,
his partner, his right. A tile played or in the mover's hand is known: one
lump, nothing to learn, no row. Every hidden tile at every position is a row.

**Values,** in the mover's units, per move a and lump L: p_L(a) the share of
L's worlds in which the mover succeeds (the contract made for the bidders,
set for the setters), μ_L(a) the mean mover's margin in points (margin
signed toward the mover's side). The cell's values are the same over all
worlds. A best move is the argmax by p, then μ, then the lowest tile ("by
make"), or by μ, then p, then the lowest tile ("by margin").

**The value of knowing T** (VOI), with the continuation fixed:

    VOI(T) = Σ_L mass_L · max_a v_L(a) − max_a v(a)

for v = p (VOI_p, make probability) and v = μ (VOI_μ, points). By Jensen
(max of a mean ≤ mean of maxima) it is ≥ 0: an upper side. It is what a
mover told where T sits could gain at this move alone, with Zeb's
continuation unchanged; it is not money Zeb can collect, and on a sample it
is inflated by the maximum over moves within each lump.

**Fixed before counting** (2026-10-13, committed before the first count):

- *Population:* every planned position of the first `DEALS` deals of the
  pour chunk s100001 not decided by the banked score (as the metasplat).
- *Minimum mass:* a lump is non-trivial when it holds at least `MIN_MASS` of
  the cell's worlds. A (position, T) row is *split* when two or more lumps
  are non-trivial; otherwise *settled* (T's seat is nearly known) and it is
  not classified hedge or robust. VOI sums over every lump, whatever its
  mass.
- *Hedge or robust,* per split row and per value (make, margin): *robust*
  when one move is within the lump's noise bar of the lump's best in every
  non-trivial lump; *hedge* when no move is. The bar is `BAR_SE` paired
  standard errors of the difference over the lump's worlds, zero in an
  enumerated cell (fewer than `SAMPLED` worlds: the whole fiber). *Strict*
  robust (reported beside it): the same best move in every non-trivial lump.
- *The VOI noise bar:* each row's VOI is computed again with its lump labels
  shuffled over the cell's worlds (the same lump sizes, `SEED`): what an
  arbitrary partition of the same sizes earns from the maximum over moves
  alone. Reported beside every mean; the *excess* is VOI less this null.
  Means carry 95% intervals clustered by deal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from scenario.planner import HAND_TOTAL
from scenario.record import empty_record
from scenario.rules import ALL_TILES, rules_for, tiles_of

MIN_MASS = 0.10
BAR_SE = 2.0
SAMPLED = 512
SEED = 20261013
DEALS = 10_000
EPS = 1e-12
LUMPS = ("left", "partner", "right")   # the seat holding T, relative to the mover: +1, +2, +3
ROLES = ("bidder", "left setter", "partner", "right setter")   # relative to the bidder
COLUMNS = ("deal", "k", "pos", "tile", "seat", "decl", "n_worlds", "n_actions", "chosen",
           "mass", "split", "voi_p", "voi_mu", "null_p", "null_mu",
           "robust_p", "robust_mu", "strict_p", "strict_mu",
           "cell_best_p", "cell_best_mu", "best_p", "best_mu", "pl_best", "pl_z", "mul_best", "mul_z",
           "p_cell", "mu_cell", "p_z", "mu_z")


def _keys(sp: np.ndarray, smu: np.ndarray, tiles: np.ndarray, by: str) -> np.ndarray:
    """Integer sort keys over the last axis (moves): by make (p, then μ, then
    the lowest tile) or by margin (μ, then p, then the lowest tile). `sp`,
    `smu` are sums over the same worlds (at most `SAMPLED`)."""
    sp, smu = sp.astype(np.int64), smu.astype(np.int64) + HAND_TOTAL * SAMPLED
    if by == "make":
        k = sp * (2 * HAND_TOTAL * SAMPLED + 1) + smu
    else:
        k = smu * (SAMPLED + 1) + sp
    return k * 32 + (31 - tiles.astype(np.int64))


def lump_rows(outcomes: np.ndarray, worlds: np.ndarray, actions: Sequence[int], chosen: int, mover: int,
              hidden: int, sign: int, bid: int, exact: bool, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One position's rows, one per hidden tile. `outcomes` [A, W] the
    bidder team's final margins of each legal move in each world (uniform
    weights), `worlds` [W, 4] the remaining hands, `hidden` the mask of tiles
    hidden from the mover, `sign` +1 when the mover bids, `exact` when the
    worlds are the whole fiber."""
    o = np.asarray(outcomes, np.int64)
    A, W = o.shape
    tiles = np.asarray(actions, np.int64)
    succ = ((o >= 2 * bid - HAND_TOTAL) == (sign > 0)).astype(np.int64)
    marg = sign * o
    others = [(mover + i) % 4 for i in (1, 2, 3)]
    ht = np.array(tiles_of(hidden), np.int64)
    T = len(ht)
    if T == 0:
        return {}
    held = ((np.asarray(worlds, np.int64)[:, others][:, :, None] >> ht[None, None, :]) & 1)   # [W, 3, T]
    if not np.all(held.sum(axis=1) == 1):
        raise AssertionError("a hidden tile is not held by exactly one other seat in some world")
    label = held.argmax(axis=1)                                     # [W, T]

    def sums(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        flat = (lab.T[:, None, :] == np.arange(3)[None, :, None]).astype(np.int64).reshape(T * 3, W)
        return (flat, flat.sum(axis=1).reshape(T, 3), (flat @ succ.T).reshape(T, 3, A),
                (flat @ marg.T).reshape(T, 3, A))

    onehot, n_l, sp_l, smu_l = sums(label)
    sp, smu = succ.sum(axis=1), marg.sum(axis=1)

    def voi(sp_l: np.ndarray, smu_l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ((sp_l.max(axis=2).sum(axis=1) - sp.max()) / W, (smu_l.max(axis=2).sum(axis=1) - smu.max()) / W)

    voi_p, voi_mu = voi(sp_l, smu_l)
    _, _, nsp_l, nsmu_l = sums(rng.permuted(label, axis=0))
    null_p, null_mu = voi(nsp_l, nsmu_l)

    mass = n_l / W
    nontrivial = mass >= MIN_MASS - EPS
    split = nontrivial.sum(axis=1) >= 2
    kp, kmu = _keys(sp_l, smu_l, tiles, "make"), _keys(sp_l, smu_l, tiles, "margin")
    bp, bmu = kp.argmax(axis=2), kmu.argmax(axis=2)                # [T, 3] lump-best columns
    cp = int(_keys(sp, smu, tiles, "make").argmax())
    cmu = int(_keys(sp, smu, tiles, "margin").argmax())
    z = list(actions).index(chosen)

    def robust(values: np.ndarray, sums_l: np.ndarray, best: np.ndarray) -> np.ndarray:
        """Per tile: one move within the bar of the best in every non-trivial lump."""
        pair = (values[:, None, :] * values[None, :, :]).reshape(A * A, W)
        gram = (onehot @ pair.T).reshape(T, 3, A, A)                  # Σ over the lump of v_a v_b
        tt, ll = np.meshgrid(np.arange(T), np.arange(3), indexing="ij")
        n = np.maximum(n_l, 1).astype(np.float64)[:, :, None]
        gap = (sums_l[tt, ll, best][:, :, None] - sums_l) / n        # [T, 3, A] best's mean less a's
        diag = np.einsum("tlaa->tla", gram)
        d2 = diag[tt, ll, best][:, :, None] - 2 * gram[tt, ll, best, :] + diag   # Σ (v_b − v_a)²
        var = np.maximum(d2 - n * gap * gap, 0.0) / np.maximum(n - 1, 1)
        bar = 0.0 if exact else BAR_SE * np.sqrt(var / n)
        ok = (gap <= bar + EPS) | ~nontrivial[:, :, None]
        return ok.all(axis=1).any(axis=1)

    def strict(best: np.ndarray) -> np.ndarray:
        b = np.where(nontrivial, best, -1)
        first = np.take_along_axis(b, nontrivial.argmax(axis=1)[:, None], axis=1)
        return np.all((b == first) | (b < 0), axis=1)

    n = np.maximum(n_l, 1)
    tt, ll = np.meshgrid(np.arange(T), np.arange(3), indexing="ij")
    return dict(
        tile=ht.astype(np.int8), mass=mass.astype(np.float32), split=split,
        voi_p=voi_p, voi_mu=voi_mu, null_p=null_p, null_mu=null_mu,
        robust_p=robust(succ, sp_l, bp), robust_mu=robust(marg, smu_l, bmu),
        strict_p=strict(bp), strict_mu=strict(bmu),
        cell_best_p=np.full(T, tiles[cp], np.int8), cell_best_mu=np.full(T, tiles[cmu], np.int8),
        best_p=np.where(n_l > 0, tiles[bp], -1).astype(np.int8), best_mu=np.where(n_l > 0, tiles[bmu], -1).astype(np.int8),
        pl_best=(sp_l[tt, ll, bp] / n).astype(np.float32), pl_z=(sp_l[:, :, z] / n).astype(np.float32),
        mul_best=(smu_l[tt, ll, bmu] / n).astype(np.float32), mul_z=(smu_l[:, :, z] / n).astype(np.float32),
        p_cell=np.full(T, sp.max() / W), mu_cell=np.full(T, smu.max() / W),
        p_z=np.full(T, sp[z] / W), mu_z=np.full(T, smu[z] / W),
    )


# -- the pour ------------------------------------------------------------------------------


def open_pour(pour_dir: str | Path) -> dict[str, np.ndarray]:
    """A pour's columns: from `<column>.npy` files in `pour_dir` when present
    (memory-mapped; `unpack` writes them), else from its `pour.npz`."""
    pour_dir = Path(pour_dir)
    names = ("hands", "decl", "plays", "deal", "n_plays", "n_actions", "n_worlds", "actions", "worlds", "outcomes")
    if (pour_dir / "worlds.npy").exists():
        return {c: np.load(pour_dir / f"{c}.npy", mmap_mode="r") for c in names}
    with np.load(pour_dir / "pour.npz") as z:
        if not np.all(z["weights"] == 1):
            raise ValueError("a pour with unequal world weights: the lumps read uniform worlds")
        return {c: z[c] for c in names}


def unpack(pour_npz_dir: str | Path, out: str | Path) -> None:
    """Write a pour's columns as plain .npy files (to memory-map them), with
    its manifest. The weights are checked uniform and not written."""
    import shutil
    import zipfile

    src, out = Path(pour_npz_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src / "pour.npz") as z:
        with z.open("weights.npy") as f:
            if not np.all(np.lib.format.read_array(f) == 1):
                raise ValueError("a pour with unequal world weights: the lumps read uniform worlds")
        for name in z.namelist():
            if name != "weights.npy":
                z.extract(name, out)
    shutil.copy(src / "manifest.json", out / "manifest.json")


def deal_rows(p: dict[str, np.ndarray], deals: Sequence[int], bid: int, bidder: int,
              sampled: int = SAMPLED, seed: int = SEED) -> dict[str, np.ndarray]:
    """Every live planned position of `deals`, one row per hidden tile."""
    na, nw = p["n_actions"].astype(np.int64), p["n_worlds"].astype(np.int64)
    a_off = np.concatenate(([0], np.cumsum(na)))
    w_off = np.concatenate(([0], np.cumsum(nw)))
    o_off = np.concatenate(([0], np.cumsum(na * nw)))
    want = np.isin(p["deal"], np.asarray(deals))
    by_deal: dict[int, dict[int, int]] = {}
    for q in np.flatnonzero(want):
        by_deal.setdefault(int(p["deal"][q]), {})[int(p["n_plays"][q])] = int(q)
    out: dict[str, list[np.ndarray]] = {}
    for d in sorted(by_deal):
        rules = rules_for(int(p["decl"][d]))
        hands = [int(h) for h in p["hands"][d]]
        plays = np.asarray(p["plays"][d])
        rec = empty_record(bidder)
        for k in range(max(by_deal[d]) + 1):
            q = by_deal[d].get(k)
            live = rec.banked[bidder % 2] < bid and rec.banked[1 - bidder % 2] <= HAND_TOTAL - bid
            if q is not None and live:
                seat = rec.seat
                own = hands[seat] & ~rec.played
                hidden = ALL_TILES & ~rec.played & ~own
                A, W = int(na[q]), int(nw[q])
                worlds = np.asarray(p["worlds"][w_off[q]:w_off[q] + W])
                if not np.all(worlds[:, seat] == own):
                    raise AssertionError("a world does not hold the mover's own hand")
                actions = [int(t) for t in p["actions"][a_off[q]:a_off[q] + A]]
                rows = lump_rows(np.asarray(p["outcomes"][o_off[q]:o_off[q] + A * W]).reshape(A, W), worlds,
                                 actions, int(plays[k, 1]), seat, hidden, 1 if seat % 2 == bidder % 2 else -1,
                                 bid, W < sampled, np.random.default_rng([seed, d, k]))
                T = len(rows["tile"])
                rows.update(deal=np.full(T, d, np.int32), k=np.full(T, k, np.int8), pos=np.full(T, q, np.int32),
                            seat=np.full(T, seat, np.int8), decl=np.full(T, int(p["decl"][d]), np.int8),
                            n_worlds=np.full(T, W, np.int16), n_actions=np.full(T, A, np.int8),
                            chosen=np.full(T, int(plays[k, 1]), np.int8))
                for key, v in rows.items():
                    out.setdefault(key, []).append(v)
            rec = rec.advance(int(plays[k, 1]), rules)
    return {k: np.concatenate(out[k]) for k in COLUMNS}


def _work(args: tuple) -> dict[str, np.ndarray]:
    pour_dir, deals, bid, bidder, sampled = args
    return deal_rows(open_pour(pour_dir), deals, bid, bidder, sampled)


def run(pour_dir: str | Path, out: str | Path, deals: int = DEALS, workers: int = 16) -> dict[str, np.ndarray]:
    """The rows of the first `deals` deals, in parallel; saved as `out` (npz)
    with a json sidecar."""
    import re
    from multiprocessing import Pool

    manifest = json.loads((Path(pour_dir) / "manifest.json").read_text())
    n = re.search(r"(?:^|,)n(\d+)", manifest["spec"])
    sampled = int(n[1]) if n else SAMPLED
    parts = [list(range(s, min(deals, s + 250))) for s in range(0, deals, 250)]
    with Pool(workers) as pool:
        got = pool.map(_work, [(str(pour_dir), part, manifest["bid"], manifest["bidder"], sampled) for part in parts])
    res = {k: np.concatenate([g[k] for g in got]) for k in COLUMNS}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    out.with_suffix(".json").write_text(json.dumps(dict(
        deals=deals, rows=int(len(res["tile"])), positions=int(len(np.unique(res["pos"]))),
        min_mass=MIN_MASS, bar_se=BAR_SE, sampled=sampled, seed=SEED, lumps=list(LUMPS),
        pour={k: manifest.get(k) for k in ("spec", "seed", "commit", "deals", "utility", "bid", "bidder")},
    ), indent=1) + "\n")
    return res


# -- the summary ---------------------------------------------------------------------------

Z95 = 1.959964


def tile_name(t: int) -> str:
    high = max(h for h in range(7) if h * (h + 1) // 2 <= t)
    return f"{high}-{t - high * (high + 1) // 2}"


def clustered_mean(y: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """The mean and its 95% half-width, a cluster's rows taken together."""
    n = len(y)
    if n == 0:
        return float("nan"), float("nan")
    _, inv = np.unique(cluster, return_inverse=True)
    m = np.bincount(inv).astype(np.float64)
    s = np.bincount(inv, weights=y.astype(np.float64))
    mean = s.sum() / n
    return float(mean), float(Z95 * np.sqrt(((s - mean * m) ** 2).sum()) / n)


def tile_kind(tile: np.ndarray, decl: np.ndarray) -> np.ndarray:
    """A tile's part in its hand: trump or not, and count (5 or 10) or not."""
    out = np.empty(len(tile), dtype=object)
    trump = np.zeros(len(tile), bool)
    for d in np.unique(decl):
        r = rules_for(int(d))
        s = decl == d
        trump[s] = np.array([r.led_suit[t] == 7 for t in range(28)])[tile[s]]
    count = np.array(rules_for(0).count)[tile]
    for i in range(len(tile)):
        out[i] = ("trump" if trump[i] else "off") + (f", count {count[i]}" if count[i] else "")
    return out


def summarize(r: dict[str, np.ndarray], bidder: int = 0) -> dict:
    """Hedge and robust rates; VOI by tile, overall and by trick, mover's
    role and side; and the rows with the most VOI."""
    split = r["split"]
    cl = r["deal"].astype(np.int64)
    role = np.asarray(ROLES, dtype=object)[(r["seat"].astype(np.int64) - bidder) % 4]
    side = np.where(r["seat"] % 2 == bidder % 2, "bidders", "setters")
    trick = r["k"].astype(np.int64) // 4 + 1
    exact = r["n_worlds"] < SAMPLED
    res: dict = dict(rows=int(len(split)), positions=int(len(np.unique(r["pos"]))), split=int(split.sum()),
                     min_mass=MIN_MASS, bar_se=BAR_SE)

    def rates(sel: np.ndarray) -> dict:
        out = {"n": int(sel.sum())}
        for v in ("p", "mu"):
            out[f"robust_{v}"] = clustered_mean(r[f"robust_{v}"][sel], cl[sel])
            out[f"strict_{v}"] = clustered_mean(r[f"strict_{v}"][sel], cl[sel])
        return out

    res["hedge"] = {"all": rates(split)}
    for name, lab in (("trick", trick), ("role", role), ("cell", np.where(exact, "enumerated", "sampled"))):
        res["hedge"][name] = {str(v): rates(split & (lab == v)) for v in sorted(set(lab.tolist()))}
    # per position: robust for every split tile
    pos = r["pos"]
    up, inv = np.unique(pos[split], return_inverse=True)
    for v in ("p", "mu"):
        hedged = np.bincount(inv, weights=(~r[f"robust_{v}"][split]).astype(np.float64)) > 0
        res["hedge"][f"positions_any_hedge_{v}"] = dict(n=int(len(up)), share=float(hedged.mean()))

    def voi(sel: np.ndarray) -> dict:
        out = {"n": int(sel.sum())}
        for v in ("p", "mu"):
            out[f"voi_{v}"] = clustered_mean(r[f"voi_{v}"][sel], cl[sel])
            out[f"null_{v}"] = clustered_mean(r[f"null_{v}"][sel], cl[sel])
            out[f"excess_{v}"] = clustered_mean((r[f"voi_{v}"] - r[f"null_{v}"])[sel], cl[sel])
        return out

    res["voi"] = {"all": voi(np.ones(len(split), bool)), "split": voi(split)}
    res["tile"] = {}
    for t in range(28):
        s = r["tile"] == t
        if s.any():
            e = voi(s)
            e["by_trick"] = {str(k): voi(s & (trick == k)) for k in range(1, 7)}
            e["by_role"] = {k: voi(s & (role == k)) for k in ROLES}
            e["by_side"] = {k: voi(s & (side == k)) for k in ("bidders", "setters")}
            e["split_share"] = float(split[s].mean())
            res["tile"][tile_name(t)] = e
    kind = tile_kind(r["tile"].astype(np.int64), r["decl"].astype(np.int64))
    res["kind"] = {k: voi(kind == k) for k in sorted(set(kind.tolist()))}
    five5 = r["tile"] == 20
    res["kind_5_5"] = {k: voi(five5 & (kind == k)) for k in sorted(set(kind[five5].tolist()))}
    res["by_trick"] = {str(k): voi(trick == k) for k in range(1, 7)}
    res["by_role"] = {k: voi(role == k) for k in ROLES}
    return res


def top_rows(r: dict[str, np.ndarray], n: int = 5, per_deal: int = 1) -> list[int]:
    """The rows with the largest VOI_p, at most `per_deal` per deal."""
    order = np.argsort(-r["voi_p"], kind="stable")
    seen: dict[int, int] = {}
    picks = []
    for i in order:
        d = int(r["deal"][i])
        if seen.get(d, 0) < per_deal:
            seen[d] = seen.get(d, 0) + 1
            picks.append(int(i))
            if len(picks) == n:
                break
    return picks


def describe(r: dict[str, np.ndarray], i: int, bidder: int = 0) -> dict:
    """One row in the mover's words: who holds T in each lump (by the
    bidder's roles), the best move there by make, and Zeb's pick's cost."""
    seat = int(r["seat"][i])
    lumps = []
    for l in range(3):
        holder = (seat + l + 1) % 4
        lumps.append(dict(holder=ROLES[(holder - bidder) % 4], mass=round(float(r["mass"][i, l]), 3),
                          best=tile_name(int(r["best_p"][i, l])) if r["best_p"][i, l] >= 0 else None,
                          p_best=round(float(r["pl_best"][i, l]), 3), p_zeb=round(float(r["pl_z"][i, l]), 3),
                          mu_best=round(float(r["mul_best"][i, l]), 1), mu_zeb=round(float(r["mul_z"][i, l]), 1)))
    return dict(deal=int(r["deal"][i]), k=int(r["k"][i]), trick=int(r["k"][i]) // 4 + 1,
                mover=ROLES[(seat - bidder) % 4], decl=int(r["decl"][i]), tile=tile_name(int(r["tile"][i])),
                n_worlds=int(r["n_worlds"][i]), zeb=tile_name(int(r["chosen"][i])),
                p_zeb=round(float(r["p_z"][i]), 3), voi_p=round(float(r["voi_p"][i]), 3),
                null_p=round(float(r["null_p"][i]), 3), voi_mu=round(float(r["voi_mu"][i]), 2), lumps=lumps)


def tables(res: dict) -> str:
    """The summary as Markdown."""
    def ci(x: tuple[float, float], scale: float = 1.0, f: str = ".3f") -> str:
        return f"{scale * x[0]:{f}} ± {scale * x[1]:{f}}"

    out = ["### hedge or robust (split rows; robust share, %)\n",
           "| | n | robust, make | strict, make | robust, margin | strict, margin |", "|---|---|---|---|---|---|"]

    def hrow(label: str, e: dict) -> str:
        return (f"| {label} | {e['n']:,} | {ci(e['robust_p'], 100, '.1f')} | {ci(e['strict_p'], 100, '.1f')} | "
                f"{ci(e['robust_mu'], 100, '.1f')} | {ci(e['strict_mu'], 100, '.1f')} |")

    out.append(hrow("all", res["hedge"]["all"]))
    for axis in ("trick", "role", "cell"):
        for k, e in res["hedge"][axis].items():
            out.append(hrow(f"{axis} {k}", e))
    out.append("")
    head = ("| | n | VOI_p | null_p | excess_p | VOI_μ | null_μ | excess_μ |\n"
            "|---|---|---|---|---|---|---|---|")

    def vrow(label: str, e: dict) -> str:
        return (f"| {label} | {e['n']:,} | {ci(e['voi_p'])} | {ci(e['null_p'])} | {ci(e['excess_p'])} | "
                f"{ci(e['voi_mu'], f='.2f')} | {ci(e['null_mu'], f='.2f')} | {ci(e['excess_mu'], f='.2f')} |")

    out += ["### VOI by tile (ranked by excess_p)\n", head]
    for name, e in sorted(res["tile"].items(), key=lambda kv: -kv[1]["excess_p"][0]):
        out.append(vrow(name, e))
    out += ["", "### VOI by the tile's part\n", head] + [vrow(k, e) for k, e in res["kind"].items()]
    out += ["", "### the 5-5 by its part\n", head] + [vrow(k, e) for k, e in res["kind_5_5"].items()]
    out += ["", "### VOI by trick (every tile)\n", head] + [vrow(f"trick {k}", e) for k, e in res["by_trick"].items()]
    out += ["", "### VOI by mover (every tile)\n", head] + [vrow(k, e) for k, e in res["by_role"].items()]
    for axis in ("by_trick", "by_role", "by_side"):
        out += ["", f"### excess_p by tile, {axis[3:]} (×1000)\n"]
        keys = list(next(iter(res["tile"].values()))[axis])
        out += ["| tile | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for name, e in sorted(res["tile"].items(), key=lambda kv: -kv[1]["excess_p"][0]):
            out.append(f"| {name} | " + " | ".join(
                "" if not e[axis][k]["n"] else f"{1000 * e[axis][k]['excess_p'][0]:.1f}" for k in keys) + " |")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="python -u -m scenario.lumps_voi", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unpack", help="a pour's columns as .npy files, to memory-map")
    u.add_argument("pour")
    u.add_argument("--out", required=True)
    a = sub.add_parser("run", help="the rows: every live position, every hidden tile")
    a.add_argument("--pour", required=True, help="an unpacked pour directory (or one with pour.npz)")
    a.add_argument("--out", required=True)
    a.add_argument("--deals", type=int, default=DEALS)
    a.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("summarize", help="results.json, tables.md and top.json from the rows")
    s.add_argument("rows")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "unpack":
        unpack(args.pour, args.out)
    elif args.cmd == "run":
        res = run(args.pour, args.out, args.deals, args.workers)
        print(f"{len(res['tile']):,} rows over {len(np.unique(res['pos'])):,} positions -> {args.out}")
    else:
        with np.load(args.rows) as z:
            r = {k: z[k] for k in z.files}
        res = summarize(r)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(res, indent=1) + "\n")
        (out / "tables.md").write_text(tables(res))
        (out / "top.json").write_text(json.dumps([describe(r, i) for i in top_rows(r, 12)], indent=1) + "\n")
        print(f"{res['rows']:,} rows, {res['split']:,} split -> {out}")


if __name__ == "__main__":
    main()
