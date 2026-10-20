# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Coarse or mixed, by the essay's own discriminator (#229; wiki/experiments/lem-coarse-or-mixed-2026-10-20.md).

A broadened layer-1 pole joins two concepts a and b. The essay calls a unit mixed when it folds
unrelated things that do not interfere, and coarse when it merges things the net treats alike,
read "upstream at how it's built and downstream at what it does". So each pole's top set is split
into A (a, not b) and B (b, not a) and two cosines are read:

- `r`, routes in: the unit's mean input contribution x ⊙ w₁[:, j] on each side, centred on all rows;
- `d`, uses out: the mean change in the 28 tile scores when the unit is ablated, on each side.

Each comes with a ceiling: the same cosine with A/B shuffled inside A ∪ B, the shared-direction
bias every split carries. The positive controls are *planted folds*: unit k merged into unit j of a
trained net (input weights, bias and outgoing weights summed; k removed), a real unit whose routes
and uses the instrument has to find through its own weights. The donors are picked by their labels
only.

    python -u -m lem.probe coarse SAMPLE.npz LABELS_V2.npz AUDIT_V1_DIR DONOR_AUDIT_V2_DIR OUT.json --net NET [--folds]
"""
from __future__ import annotations

import itertools

import numpy as np

from lem.probe.audit import BASE_CAP, TOP, condition, top_set
from lem.probe.causal import downstream
from lem.probe.net import forward
from lem.probe.squeeze import GAIN, union_table

SEED = 20261020
MIN_SIDE = 30
SHUFFLES = 20
MAX_FOLDS = 40
SEATS = ("lead", "second", "third", "last")
MIXED_LIFT = 0.5


def cos(u: np.ndarray, v: np.ndarray) -> float | None:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else None


def fold(net, j: int, k: int):
    """A copy of `net` with layer-1 unit k merged into unit j and removed."""
    from dataclasses import replace
    w1, b1, w2 = net.w1.copy(), net.b1.copy(), net.w2.copy()
    w1[:, j] += w1[:, k]
    b1[j] += b1[k]
    w2[:, j] += w2[:, k]
    w1[:, k] = 0
    b1[k] = -(1 << 40)                      # never fires again
    w2[:, k] = 0
    return replace(net, w1=w1, b1=b1, w2=w2)


def read_split(net, x: np.ndarray, j: int, top: np.ndarray, side_a: np.ndarray, side_b: np.ndarray, xmean: np.ndarray,
               rng) -> dict | None:
    """`r` and `d` for layer-1 unit j over its top rows split into two sides, with their ceilings."""
    ia, ib = top[side_a], top[side_b]
    if len(ia) < MIN_SIDE or len(ib) < MIN_SIDE:
        return None
    w = net.w1[:, j].astype(np.float64)
    base = forward(net, x[top]).out.astype(np.float64)
    eff = base - downstream(net, x[top], 1, j, 0).astype(np.float64)        # [len(top), 28]
    pos = {int(r): i for i, r in enumerate(top)}
    E = lambda rows: eff[[pos[int(r)] for r in rows]].mean(axis=0)
    R = lambda rows: (x[rows].mean(axis=0) - xmean) * w
    r, d = cos(R(ia), R(ib)), cos(E(ia), E(ib))
    both = np.concatenate([ia, ib])
    rc, dc = [], []
    for _ in range(SHUFFLES):
        p = rng.permutation(both)
        sa, sb = p[:len(ia)], p[len(ia):]
        rc.append(cos(R(sa), R(sb)))
        dc.append(cos(E(sa), E(sb)))
    med = lambda v: float(np.median([t for t in v if t is not None])) if any(t is not None for t in v) else None
    return {"r": r, "d": d, "r_ceiling": med(rc), "d_ceiling": med(dc), "n_a": int(len(ia)), "n_b": int(len(ib))}


def broadened(sample, labels_v1: np.ndarray, names_v1: list[str], net, audit_v1: dict, log=print) -> list[dict]:
    """Every broadened live `+` layer-1 pole, read."""
    x = sample.x
    acts = condition(forward(net, x), sample.n_plays.astype(np.int64))
    n = len(labels_v1)
    k = max(1, int(round(TOP * n)))
    L = labels_v1.astype(np.float64)
    base = L.mean(axis=0)
    base_u = base[:, None] + base[None, :] - (L.T @ L) / n
    ok = base_u <= BASE_CAP
    single_ok = base <= BASE_CAP
    xmean = x.mean(axis=0)
    rng = np.random.default_rng(SEED)
    h1 = net.b1.shape[0]
    out = []
    for row in audit_v1["rows"]:
        p = row["pole"]
        if row["kind"] == "dead" or p >= 2 * h1 or p % 2:
            continue
        j = p // 2
        top = top_set(acts.pole(p), k)
        tl = labels_v1[top]
        prec = tl.mean(axis=0)
        u = union_table(tl, ok)
        if u.max() - float(np.where(single_ok, prec, -1).max()) < GAIN:
            continue
        a, b = np.unravel_index(int(np.argmax(u)), u.shape)
        la, lb = labels_v1[top, a].astype(bool), labels_v1[top, b].astype(bool)
        rec = read_split(net, x, j, top, la & ~lb, lb & ~la, xmean, rng)
        if rec is None:
            continue
        rec.update(pole=row["name"], union=[names_v1[a], names_v1[b]])
        out.append(rec)
        log(f"  {row['name']} {names_v1[a]} ∪ {names_v1[b]}: r {rec['r']} d {rec['d']}")
    return out


def folds(sample, labels_v2: np.ndarray, names_v2: list[str], net, donor_audit_v2: dict, log=print) -> list[dict]:
    """Planted folds of clean layer-1 `+` poles, picked by their labels only: mixed (lift < 0.5,
    not one variable) and coarse (two values of one variable: the mover's seat or the declaration)."""
    x = sample.x
    h1 = net.b1.shape[0]
    clean = [(r["pole"] // 2, r["label"]) for r in donor_audit_v2["rows"]
             if r["kind"] == "clean" and r["pole"] < 2 * h1 and r["pole"] % 2 == 0 and not r["label"].startswith("x:")]
    L = labels_v2.astype(np.float64)
    col = {c: i for i, c in enumerate(names_v2)}
    P = L.mean(axis=0)
    def variable(c):
        return "seat" if c in SEATS else "decl" if c.startswith("trump-is-") else None
    mixed, coarse = [], []
    for (j, a), (kk, b) in itertools.combinations(clean, 2):
        if j == kk or a == b:
            continue
        va, vb = variable(a), variable(b)
        if va is not None and va == vb:
            coarse.append((j, kk, a, b))
            continue
        ia, ib = col[a], col[b]
        lift = float((L[:, ia] * L[:, ib]).mean() / max(P[ia] * P[ib], 1e-12))
        if lift < MIXED_LIFT:
            mixed.append((j, kk, a, b))
    rng = np.random.default_rng(SEED)
    pick = lambda v: [v[i] for i in rng.permutation(len(v))[:MAX_FOLDS]]
    out = []
    xmean = x.mean(axis=0)
    n = len(x)
    k = max(1, int(round(TOP * n)))
    for kind, pairs in (("mixed", pick(mixed)), ("coarse", pick(coarse))):
        for j, kk, a, b in pairs:
            f = fold(net, j, kk)
            acts = condition(forward(f, x), sample.n_plays.astype(np.int64))
            top = top_set(acts.pole(2 * j), k)
            la, lb = labels_v2[top, col[a]].astype(bool), labels_v2[top, col[b]].astype(bool)
            rec = read_split(f, x, j, top, la & ~lb, lb & ~la, xmean, rng)
            if rec is None:
                continue
            rec.update(kind=kind, units=[int(j), int(kk)], labels=[a, b])
            out.append(rec)
            log(f"  fold {kind} {a} + {b}: r {rec['r']} d {rec['d']}")
    return out
