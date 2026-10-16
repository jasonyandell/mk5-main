# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The neuron-count curve, first form: how many neurons does the play need?

Keep a set S of hidden neurons (the rest are zeroed after their ReLU, in
either layer) and measure *fidelity*: the share of rows on which the
reduced net picks the same tile as the whole net. Greedy forward selection
on a fit slice adds the neuron that raises fidelity most; the curve is
reported on a disjoint eval slice. The *knee* is the smallest k whose eval
fidelity reaches `KNEE` of the whole net's (which is 1 by definition).

This is the feature-count curve of stage 4 in its cheapest form, with
"feature" meaning "neuron"; the chance baseline is the fidelity of the
empty set (bias-only play) and of k random neurons.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lem.probe.net import Activations
from lem.probe.sample import Sample
from scenario.field import FieldNet

FIT = 20_000
EVAL = 20_000
KNEE = 0.95
SEED = 20261015


def _pick_slots(out: np.ndarray, legal: np.ndarray) -> np.ndarray:
    masked = np.where(legal, out, np.iinfo(np.int64).min)
    return masked.argmax(axis=1)


def _out_with(net: FieldNet, a1: np.ndarray, keep1: np.ndarray, keep2: np.ndarray) -> np.ndarray:
    a1m = a1 * keep1
    a2 = np.maximum((a1m @ net.w2.T + net.b2) >> net.s2, 0) * keep2
    return a2 @ net.w3.T + net.b3


def fidelity(net: FieldNet, a1: np.ndarray, legal: np.ndarray, ref: np.ndarray, keep: np.ndarray) -> float:
    h1 = net.b1.shape[0]
    out = _out_with(net, a1, keep[:h1], keep[h1:])
    return float((_pick_slots(out, legal) == ref).mean())


def run(sample: Sample, acts: Activations, net: FieldNet, fit: int = FIT, eval_n: int = EVAL,
        log=print) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(sample)
    order = rng.permutation(n)
    fit = min(fit, n // 2)
    fit_rows, eval_rows = order[:fit], order[fit:fit + eval_n]
    legal = sample.legal
    ref = _pick_slots(acts.out, legal)
    h = net.b1.shape[0] + net.b2.shape[0]
    keep = np.zeros(h, np.int64)
    chosen: list[int] = []
    curve = []
    base_fit = fidelity(net, acts.a1[fit_rows], legal[fit_rows], ref[fit_rows], keep)
    base_eval = fidelity(net, acts.a1[eval_rows], legal[eval_rows], ref[eval_rows], keep)
    curve.append({"k": 0, "neuron": None, "fit": base_fit, "eval": base_eval})
    log(f"  k=0 fit {base_fit:.3f} eval {base_eval:.3f}")
    a1f, lf, rf = acts.a1[fit_rows], legal[fit_rows], ref[fit_rows]
    a1e, le, re_ = acts.a1[eval_rows], legal[eval_rows], ref[eval_rows]
    knee = None
    for step in range(h):
        best, best_f = None, -1.0
        for j in range(h):
            if keep[j]:
                continue
            keep[j] = 1
            f = fidelity(net, a1f, lf, rf, keep)
            keep[j] = 0
            if f > best_f:
                best, best_f = j, f
        keep[best] = 1
        chosen.append(best)
        fe = fidelity(net, a1e, le, re_, keep)
        curve.append({"k": step + 1, "neuron": int(best), "fit": best_f, "eval": fe})
        if knee is None and fe >= KNEE:
            knee = step + 1
        if step % 8 == 0 or fe >= 0.999:
            log(f"  k={step + 1} add {best} fit {best_f:.3f} eval {fe:.3f}")
        if fe >= 0.999 and step >= 32:
            break
    # random-k baseline on the eval slice, one draw per k of the curve
    random_curve = []
    for k in (1, 2, 4, 8, 16, 32, 64, 96, 128, 192):
        if k > h:
            break
        fs = []
        for _ in range(5):
            rk = np.zeros(h, np.int64)
            rk[rng.choice(h, k, replace=False)] = 1
            fs.append(fidelity(net, a1e, le, re_, rk))
        random_curve.append({"k": k, "eval_mean": float(np.mean(fs)), "eval_min": float(np.min(fs)), "eval_max": float(np.max(fs))})
    return {"curve": curve, "random": random_curve, "knee": knee, "KNEE": KNEE, "fit_rows": int(fit),
            "eval_rows": int(len(eval_rows)), "order": chosen}


def write(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=1))
