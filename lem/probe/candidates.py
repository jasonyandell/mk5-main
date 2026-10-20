# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Two candidate units read in both directions (#232; wiki/experiments/lem-lead-units-2026-10-20.md).

Upstream at how a unit is built (its exact input votes), downstream at what it does (its effect on
the move distribution and on its readers), whether its mutually exclusive partner is specific or
arbitrary (on held-out deals), its raw shape, and whether the same union recurs in other nets.
Four verdicts in order of precedence: packing, one thing, two related jobs, unresolved.

    python -u -m lem.probe candidates SAMPLE.npz LABELS_V2.npz OUT.json --unit NET:J:B [--unit ...] --recur NET [NET ...]
"""
from __future__ import annotations

import numpy as np

from lem.probe.audit import BASE_CAP, DEAD, TOP, condition, top_set
from lem.probe.causal import downstream
from lem.probe.net import forward, load_net
from lem.probe.squeeze import FAMILIES, GAIN, union_table

SEED = 20261020
EXCLUSIVE = 0.005
NEAR = 0.05
ENRICH = 0.3
READER_MASS = 0.8
TRICK_STATE = ("ten-on-trick", "safe-trick", "count-on-trick", "winner-cannot-be-beaten", "partner-winning", "bidder-winning")
DISTANT = ("score", "doubles")
V2_DISTANT_PREFIX = ("partner-void", "setter-void")
FAM = {c: f for f, cs in FAMILIES.items() for c in cs}


def family(c: str) -> str:
    if c in FAM:
        return FAM[c]
    if c.startswith("trump-is-"):
        return "declaration"
    if c.startswith(V2_DISTANT_PREFIX):
        return "voids"
    return "v2:" + c.split("-")[0]


def split_deals(deal: np.ndarray) -> np.ndarray:
    """True for the held-out half of deals."""
    rng = np.random.default_rng(SEED)
    u = np.unique(deal)
    held = set(rng.permutation(u)[: len(u) // 2].tolist())
    return np.isin(deal, list(held))


def matched_mean(v: np.ndarray, rows: np.ndarray, base: np.ndarray, strata: np.ndarray):
    """Mean and sd of v over `base` rows, reweighted to the play-count mix of `rows`."""
    w = np.zeros(len(v))
    for s, c in zip(*np.unique(strata[rows], return_counts=True)):
        m = base & (strata == s)
        if m.any():
            w[m] = c / m.sum()
    if w.sum() == 0:
        return None, None
    mu = (w[:, None] * v).sum(axis=0) / w.sum() if v.ndim > 1 else float((w * v).sum() / w.sum())
    var = (w[:, None] * (v - mu) ** 2).sum(axis=0) / w.sum() if v.ndim > 1 else float((w * (v - mu) ** 2).sum() / w.sum())
    return mu, np.sqrt(var)


def log_softmax_legal(out: np.ndarray, legal_slots: np.ndarray) -> np.ndarray:
    o = np.where(legal_slots, out.astype(np.float64), -np.inf)
    m = o.max(axis=1, keepdims=True)
    return o - m - np.log(np.exp(o - m).sum(axis=1, keepdims=True))


def read_unit(sample, L: np.ndarray, names: list[str], net, j: int, b: str, log=print) -> dict:
    x, strata = sample.x, sample.n_plays.astype(np.int64)
    col = {c: i for i, c in enumerate(names)}
    held = split_deals(sample.deal)
    acts = forward(net, x)
    cz = condition(acts, strata).z1[:, j]
    hrows = np.flatnonzero(held)
    k = max(1, int(round(TOP * len(hrows))))
    top = hrows[np.argsort(-cz[hrows], kind="stable")[:k]]
    lead, bb = L[:, col["lead"]].astype(bool), L[:, col[b]].astype(bool)
    A, B = top[lead[top]], top[bb[top]]
    firing = acts.a1[:, j] > 0
    off = held & ~firing
    rec = {"unit": j, "b": b, "n_top": int(k), "n_A": int(len(A)), "n_B": int(len(B))}
    # check 1: upstream votes, enrichment against non-firing rows at matched play counts
    w = net.w1[:, j].astype(np.float64)
    from lem.probe.audit import feature_names
    fn = feature_names(x.shape[1])
    thr = np.percentile(np.abs(w), 90)
    H = hrows                                         # held-out rows only, float32: the full matrix is ~1 GB in float64
    xh = x[H].astype(np.float32)
    sh, offh = strata[H], off[H]
    tops_in = {}
    shared = []
    enr = {}
    for side, R in (("A", A), ("B", B)):
        xm = x[R].astype(np.float64).mean(axis=0)
        bm, _ = matched_mean(xh, np.isin(H, R), offh, sh)
        enr[side] = xm - bm
        contrib = w * xm
        tops_in[side] = [(fn[i], round(float(contrib[i]), 1)) for i in np.argsort(-contrib)[:10]]
    for i in np.flatnonzero(np.abs(w) >= thr):
        if enr["A"][i] >= ENRICH and enr["B"][i] >= ENRICH:
            shared.append(fn[i])
    rec.update(top_inputs=tops_in, shared_inputs=shared)
    # check 2: per-half use (KL over legal tiles) and readers' targeted cancellation
    legal_slots = np.zeros((len(x), 28), bool)
    canon = sample.canon()
    for kk in range(sample.actions.shape[1]):
        t = sample.actions[:, kk].astype(np.int64)
        ok = t >= 0
        legal_slots[np.flatnonzero(ok), canon[np.flatnonzero(ok), t[ok]]] = True
    kl = {}
    for side, R in (("A", A), ("B", B)):
        p0 = log_softmax_legal(acts.out[R], legal_slots[R])
        p1 = log_softmax_legal(downstream(net, x[R], 1, j, 0), legal_slots[R])
        e0 = np.exp(p0)
        kl[side] = float(np.nanmean(np.where(legal_slots[R], e0 * (p0 - p1), 0).sum(axis=1)))
    rec["kl"] = kl
    z2 = acts.a1.astype(np.float64) @ net.w2.T.astype(np.float64) + net.b2
    w2 = net.w2[:, j].astype(np.float64)
    other = z2 - w2[None, :] * acts.a1[:, j:j + 1]
    readers = []
    for i in np.flatnonzero(w2 != 0):
        r = {"reader": int(i), "w2": float(w2[i])}
        for side, R in (("A", A), ("B", B)):
            mask = np.isin(np.arange(len(x)), R)
            mu, sd = matched_mean(other[:, i], mask, off, strata)
            d = float(other[R, i].mean() - mu) if mu is not None else 0.0
            sgn = 1 if w2[i] > 0 else -1
            r[f"cancel_{side}"] = bool(sd and sgn * d <= -sd)
            r[f"positive_{side}"] = float((z2[R, i] > 0).mean())
        readers.append(r)
    mass = np.abs(w2)
    order = np.argsort(-mass)
    keep, cum = [], 0.0
    for i in order:
        if mass[i] == 0:
            break
        keep.append(int(i)); cum += mass[i]
        if cum >= READER_MASS * mass.sum():
            break
    rd = {r["reader"]: r for r in readers}
    groups = {"A": [i for i in keep if rd[i]["cancel_A"] and not rd[i]["cancel_B"]],
              "B": [i for i in keep if rd[i]["cancel_B"] and not rd[i]["cancel_A"]]}
    both_pos = [r["reader"] for r in readers if r["w2"] > 0 and r["positive_A"] >= 0.5 and r["positive_B"] >= 0.5]
    demux = bool(groups["A"] and groups["B"] and len(groups["A"]) + len(groups["B"]) == len(keep) and not both_pos)
    rec.update(readers=readers, heavy_readers=keep, cancel_groups=groups, union_positive_readers=both_pos, demultiplexing=demux)
    # check 3: partner specificity on held-out top set
    Lb = L.astype(bool)
    nlead = lead.sum()
    pool = [c for c in names if c != "lead" and (Lb[:, col[c]] & lead).sum() < EXCLUSIVE * nlead]
    tl = Lb[top]
    up = {c: float((tl[:, col["lead"]] | tl[:, col[c]]).mean()) for c in pool}
    pb = up.get(b)
    near = [c for c in pool if c != b and pb is not None and up[c] >= pb - NEAR]
    fams = sorted({family(c) for c in near} | ({family(b)} if pb is not None else set()))
    base = Lb.mean(axis=0)
    dist = [c for c in names if family(c) in DISTANT + ("voids",) and 0.5 * base[col[b]] <= base[col[c]] <= 2 * base[col[b]]]
    dctl = max((float((tl[:, col["lead"]] | tl[:, col[c]]).mean()), c) for c in dist) if dist else (None, None)
    margin = (pb - max([up[c] for c in pool if c != b], default=0.0)) if pb is not None else None
    specific = pb is not None and (margin > NEAR or all(family(c) == "trick" or c in TRICK_STATE for c in near))
    arbitrary = pb is not None and (len(fams) >= 3 or (dctl[0] is not None and dctl[0] >= pb - NEAR))
    rec.update(pool={f: sorted(c for c in pool if family(c) == f) for f in sorted({family(c) for c in pool})},
               b_union=pb, margin=margin, near_ties=near, near_families=fams, distant_control=dctl,
               specific=bool(specific), arbitrary=bool(arbitrary))
    # check 4: raw shape
    a1 = acts.a1[:, j].astype(np.float64)
    mx = float(a1.max()) if a1.max() > 0 else 1.0
    rec["shape"] = {side: {"rarity": len(R) / max(1, len(top)), "weakness": float(a1[R].mean() / mx)} for side, R in (("A", A), ("B", B))}
    log(f"  unit {j} lead+{b}: shared {shared}, demux {demux}, specific {specific}, arbitrary {arbitrary}, kl {kl}")
    return rec


def broadened_unions(sample, L1: np.ndarray, names1: list[str], net) -> list[tuple[str, str]]:
    x = sample.x
    acts = forward(net, x)
    c = condition(acts, sample.n_plays.astype(np.int64))
    n = len(L1); k = max(1, int(round(TOP * n)))
    Lf = L1.astype(np.float64); base = Lf.mean(axis=0)
    base_u = base[:, None] + base[None, :] - (Lf.T @ Lf) / n
    ok, single_ok = base_u <= BASE_CAP, base <= BASE_CAP
    out = []
    for j in range(net.b1.shape[0]):
        if (acts.a1[:, j] > 0).mean() < DEAD:
            continue
        tl = L1[top_set(c.z1[:, j], k)]
        prec = tl.mean(axis=0); u = union_table(tl, ok)
        if u.max() - float(np.where(single_ok, prec, -1).max()) >= GAIN:
            a, bb = np.unravel_index(int(np.argmax(u)), u.shape)
            out.append((names1[a], names1[bb]))
    return out


def verdict(rec: dict, recurs: bool) -> str:
    if rec["arbitrary"] and rec["demultiplexing"] and not recurs:
        return "packing"
    if rec["shared_inputs"] and rec["specific"] and not rec["demultiplexing"]:
        return "one thing"
    if rec["specific"] and recurs and not rec["demultiplexing"]:
        return "two related jobs"
    return "unresolved"
