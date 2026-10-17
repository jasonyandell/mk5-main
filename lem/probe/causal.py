# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Causal tests of named units: ablate a neuron or clamp it on, and see where the play changes
(already-clean test 5, #198; wiki/experiments/lem-causal-2026-10-17.md).

A neuron is set after its ReLU (`a`), and everything downstream is recomputed with the
referee's integer arithmetic. *Ablate* sets it to 0. *Clamp* sets it to its 99th-percentile
value over the rows where it fires. The play change per row is whether the picked tile moves,
and the teacher's regret before and after (`fidelity.regret`).

Ablation only touches rows where the neuron fires, and the audit named the `+` pole from
exactly those rows. So the `+` comparison is made **among the rows where the neuron fires**:
does ablation change the move more where the pole's concept holds than where it does not? For
a `−` pole the rows are those where the neuron is off, and the intervention is the clamp.

*Steer* (`--steer`) adds k·σ instead, σ the unit's standard deviation over the rows where it
fires, for each dose k, scored on the rows where the unit was off: there the signal is new, and a
right name says where it belongs.

    python -u -m lem.probe causal SAMPLE.npz LABELS.npz AUDIT_DIR Q_DIR OUT.json --net NET [--steer]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lem.probe.audit import feature_names
from lem.probe.net import CHUNK, Activations, forward, picks

SEED = 20261017
MIN_ROWS = 200


def downstream(net, x: np.ndarray, layer: int, j: int, value: int | None, add: int = 0) -> np.ndarray:
    """The net's tile scores with neuron j of `layer` set to `value` after its ReLU
    (None: unchanged), then `add` added to it, [N, 28] int64, in chunks."""
    outs = []
    for at in range(0, len(x), CHUNK):
        xi = x[at:at + CHUNK].astype(np.int64)
        a1 = np.maximum((xi @ net.w1 + net.b1) >> net.s1, 0)
        if layer == 1 and value is not None:
            a1[:, j] = value
        if layer == 1:
            a1[:, j] += add
        a2 = np.maximum((a1 @ net.w2.T + net.b2) >> net.s2, 0)
        if layer == 2 and value is not None:
            a2[:, j] = value
        if layer == 2:
            a2[:, j] += add
        outs.append(a2 @ net.w3.T + net.b3)
    return np.concatenate(outs)


def concept_mask(label: str, labels: np.ndarray, names: list[str], x: np.ndarray) -> np.ndarray | None:
    if label.startswith("x:"):
        fn = feature_names(x.shape[1])
        return x[:, fn.index(label)].astype(bool) if label in fn else None
    return labels[:, names.index(label)].astype(bool) if label in names else None


def _rate(changed: np.ndarray, m: np.ndarray) -> float | None:
    return float(changed[m].mean()) if m.sum() >= MIN_ROWS else None


def _mean(v: np.ndarray, m: np.ndarray) -> float | None:
    mm = m & ~np.isnan(v)
    return float(v[mm].mean()) if mm.sum() >= MIN_ROWS else None


def run(net, sample, labels: np.ndarray, names: list[str], audit: dict, regret_of, log=print) -> dict:
    """`regret_of(out)` gives per-row regret for tile scores `out`. Returns one record per
    named pole: the rates of move change and the regret change, split by its concept."""
    x = sample.x
    acts: Activations = forward(net, x)
    canon, actions = sample.canon(), sample.actions
    base_pick = picks(acts.out, actions, canon)
    base_r = regret_of(acts.out)
    h1 = net.b1.shape[0]
    clean = [r for r in audit["rows"] if r["kind"] == "clean"]
    rng = np.random.default_rng(SEED)
    out = []
    cache: dict[tuple[int, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for r in clean:
        p = r["pole"]
        layer, j = (1, p // 2) if p < 2 * h1 else (2, (p - 2 * h1) // 2)
        a = acts.a1[:, j] if layer == 1 else acts.a2[:, j]
        on = a > 0
        if on.sum() < MIN_ROWS:
            continue
        kind = "ablate" if p % 2 == 0 else "clamp"
        key = (layer, j, kind)
        if key not in cache:
            value = 0 if kind == "ablate" else int(np.percentile(a[on], 99))
            o = downstream(net, x, layer, j, value)
            cache[key] = (picks(o, actions, canon) != base_pick, regret_of(o) - base_r)
        changed, dr = cache[key]
        rows = on if kind == "ablate" else ~on
        c = concept_mask(r["label"], labels, names, x)
        if c is None:
            continue
        others = [q["label"] for q in clean if q["label"] != r["label"]]
        null_label = str(rng.choice(others)) if others else None
        n = concept_mask(null_label, labels, names, x) if null_label else None
        rec = {"pole": r["name"], "label": r["label"], "kind": kind, "rows": int(rows.sum()),
               "changed_in": _rate(changed, rows & c), "changed_out": _rate(changed, rows & ~c),
               "regret_in": _mean(dr, rows & c), "regret_out": _mean(dr, rows & ~c),
               "regret_all": float(np.nanmean(dr)), "changed_all": float(changed.mean()),
               "null_label": null_label,
               "null_in": _rate(changed, rows & n) if n is not None else None,
               "null_out": _rate(changed, rows & ~n) if n is not None else None}
        out.append(rec)
        log(f"  {r['name']} {r['label']} ({kind}): changed {rec['changed_in']} in / {rec['changed_out']} out")
    return {"base_regret": float(np.nanmean(base_r)), "poles": out}


DOSES = (0.25, 0.5, 1.0, 2.0)


def steer(net, sample, labels: np.ndarray, names: list[str], audit: dict, regret_of, log=print) -> dict:
    """Per named pole and dose: move-change rate and regret rise among rows where the unit was off,
    split by the pole's concept."""
    x = sample.x
    acts: Activations = forward(net, x)
    canon, actions = sample.canon(), sample.actions
    base_pick = picks(acts.out, actions, canon)
    base_r = regret_of(acts.out)
    h1 = net.b1.shape[0]
    clean = [r for r in audit["rows"] if r["kind"] == "clean"]
    cache: dict[tuple[int, int, float], tuple[np.ndarray, np.ndarray]] = {}
    out = []
    for r in clean:
        p = r["pole"]
        layer, j = (1, p // 2) if p < 2 * h1 else (2, (p - 2 * h1) // 2)
        a = acts.a1[:, j] if layer == 1 else acts.a2[:, j]
        on = a > 0
        if on.sum() < MIN_ROWS or (~on).sum() < MIN_ROWS:
            continue
        c = concept_mask(r["label"], labels, names, x)
        if c is None:
            continue
        sigma = float(a[on].std())
        rec = {"pole": r["name"], "label": r["label"], "sign": "+" if p % 2 == 0 else "-", "sigma": sigma, "doses": []}
        for k in DOSES:
            key = (layer, j, k)
            if key not in cache:
                o = downstream(net, x, layer, j, None, add=int(round(k * sigma)))
                cache[key] = (picks(o, actions, canon) != base_pick, regret_of(o) - base_r)
            changed, dr = cache[key]
            off = ~on
            rec["doses"].append({"k": k, "changed_all": float(changed.mean()),
                                 "changed_in": _rate(changed, off & c), "changed_out": _rate(changed, off & ~c),
                                 "regret_in": _mean(dr, off & c), "regret_out": _mean(dr, off & ~c)})
        out.append(rec)
        d1 = rec["doses"][2]
        log(f"  {r['name']} {r['label']} k=1: regret {d1['regret_in']} in / {d1['regret_out']} out")
    return {"base_regret": float(np.nanmean(base_r)), "poles": out}
