# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""How a broadened unit grew (#231; wiki/experiments/lem-unit-origin-2026-10-20.md).

The essay decides generalization against superposition by "why its things are together in the
first place". For each layer-1 `+` pole broadened at the end of a run (its best union {a, b} beats
its best single concept), this reads the run's checkpoints and asks three things:

- was the unit **born coarse**, with a and b together from the first checkpoint where either matters?
- if not, did b **enter through a bridge**: rows that hold a, or rows whose inputs resemble the a rows'?
- at the end, does any layer-2 unit **read the union** (active on both halves)?

Signals are conditioned on the play count at every checkpoint. The reference for "shared inputs" is
the same input cosine for the run's clean poles, between their entering rows and their previous top
rows, at the same checkpoint interval.

    python -u -m lem.probe origin SAMPLE.npz LABELS_V1.npz RUN_DIR OUT.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from lem.probe.audit import BASE_CAP, CLEAN, DEAD, TOP, condition, top_set
from lem.probe.coarse import cos
from lem.probe.net import forward, load_net
from lem.probe.squeeze import GAIN, union_table

BORN = 0.3
MIN_NEW = 20
BRIDGE = 0.1
REF_PCT = 5
UNION_ON, SPECIAL_OFF = 0.3, 0.1


def checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    return sorted((int(p.stem.split("_")[1]), p) for p in Path(run_dir).glob("step_*.npz"))


def run(sample, labels: np.ndarray, names: list[str], run_dir: str | Path, log=print) -> dict:
    x = sample.x
    strata = sample.n_plays.astype(np.int64)
    n = len(x)
    k = max(1, int(round(TOP * n)))
    L = labels.astype(bool)
    Lf = labels.astype(np.float64)
    base = Lf.mean(axis=0)
    base_u = base[:, None] + base[None, :] - (Lf.T @ Lf) / n
    ok, single_ok = base_u <= BASE_CAP, base <= BASE_CAP
    xmu = x.mean(axis=0)
    cks = checkpoints(run_dir)
    tops, alive = [], []
    for step, path in cks:
        net = load_net(path)
        acts = forward(net, x)
        c = condition(acts, strata)
        h1 = net.b1.shape[0]
        tops.append([top_set(c.z1[:, j], k) for j in range(h1)])
        alive.append((acts.a1 > 0).mean(axis=0))
        log(f"  step {step}: tops read")
    final_net = load_net(cks[-1][1])
    fa = forward(final_net, x)
    h1 = final_net.b1.shape[0]
    steps = [s for s, _ in cks]
    # epoch-60 clean and broadened layer-1 + poles
    clean, broad = [], []
    for j in range(h1):
        if alive[-1][j] < DEAD:
            continue
        tl = labels[tops[-1][j]]
        prec = tl.mean(axis=0)
        p1 = np.where(single_ok, prec, -1)
        u = union_table(tl, ok)
        if p1.max() >= CLEAN:
            clean.append(j)
        elif u.max() - p1.max() >= GAIN:
            a, b = np.unravel_index(int(np.argmax(u)), u.shape)
            broad.append((j, int(a), int(b)))
    # the reference: clean poles' entering rows against their previous top rows, per interval
    ref: dict[int, list[float]] = {}
    for j in clean:
        for t in range(1, len(steps)):
            new = np.setdiff1d(tops[t][j], tops[t - 1][j])
            if len(new) >= MIN_NEW:
                v = cos(x[new].mean(axis=0) - xmu, x[tops[t - 1][j]].mean(axis=0) - xmu)
                if v is not None:
                    ref.setdefault(t, []).append(v)
    first = steps.index(30) if 30 in steps else 1
    out = []
    for j, a, b in broad:
        sa = [float(L[tops[t][j], a].mean()) for t in range(len(steps))]
        sb = [float(L[tops[t][j], b].mean()) for t in range(len(steps))]
        if sb[first] > sa[first]:
            a, b, sa, sb = b, a, sb, sa
        jac = [None] + [float(len(np.intersect1d(tops[t][j], tops[t - 1][j])) / len(np.union1d(tops[t][j], tops[t - 1][j])))
                        for t in range(1, len(steps))]
        rec = {"unit": j, "a": names[a], "b": names[b], "s_a": sa, "s_b": sb, "jaccard": jac}
        t0 = next((t for t in range(first, len(steps)) if max(sa[t], sb[t]) >= BORN), None)
        if t0 is not None and sa[t0] >= BORN and sb[t0] >= BORN:
            rec["kind"] = "born coarse"
            rec["t0"] = steps[t0]
        else:
            tb = next((t for t in range(first, len(steps)) if sb[t] >= 0.5 * sb[-1]), None)
            if tb is None or tb == 0:
                rec["kind"] = "unresolved"
            else:
                prev, cur = tops[tb - 1][j], tops[tb][j]
                new = np.setdiff1d(cur, prev)
                new = new[L[new, b]]
                rec["t_b"] = steps[tb]
                if len(new) < MIN_NEW or steps[tb] >= steps[-2]:
                    rec["kind"] = "unresolved"
                else:
                    prev_a = prev[L[prev, a]]
                    bridge = float(L[new, a].mean())
                    common = cos(x[new].mean(axis=0) - xmu, x[prev_a].mean(axis=0) - xmu) if len(prev_a) else None
                    r = ref.get(tb, [])
                    floor = float(np.percentile(r, REF_PCT)) if len(r) >= 5 else None
                    rec.update(bridge=bridge, input_commonality=common, reference_p5=floor, n_new=int(len(new)))
                    if floor is None or common is None:
                        rec["kind"] = "unresolved"
                    else:
                        rec["kind"] = "non-local" if bridge < BRIDGE and common < floor else "sprout"
        # readers at the end, on side A (a, not b) and side B (b, not a)
        top = tops[-1][j]
        A, B = top[L[top, a] & ~L[top, b]], top[L[top, b] & ~L[top, a]]
        union_r = special = 0
        if len(A) >= MIN_NEW and len(B) >= MIN_NEW:
            for i in np.flatnonzero(final_net.w2[:, j] > 0):
                fa_, fb_ = float((fa.a2[A, i] > 0).mean()), float((fa.a2[B, i] > 0).mean())
                union_r += fa_ >= UNION_ON and fb_ >= UNION_ON
                special += (fa_ >= UNION_ON and fb_ < SPECIAL_OFF) or (fb_ >= UNION_ON and fa_ < SPECIAL_OFF)
            rec.update(union_readers=int(union_r), specialists=int(special))
        out.append(rec)
        log(f"  unit {j} {names[a]} -> {names[b]}: {rec['kind']} readers {rec.get('union_readers')}")
    return {"steps": steps, "clean": clean, "broadened": out,
            "reference": {str(steps[t]): float(np.percentile(v, REF_PCT)) for t, v in ref.items() if len(v) >= 5}}
