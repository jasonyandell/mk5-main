# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The width tree (#195, already-clean test 2): match each narrow net's units to the units of
a wider net trained from scratch, and ask whether a narrow unit's counterparts share its
concept family (a coarse merge) or come from unrelated families (a mixture).

    python -u -m lem.probe tree SAMPLE.npz NARROW_NET NARROW_AUDIT WIDE_NET WIDE_AUDIT OUT.json
"""
from __future__ import annotations

import numpy as np

from lem.probe.squeeze import FAMILIES

SEED = 20261017
ROWS = 40_000
NULLS = 200
TOP3, TOP5 = 3, 5
FAM = {c: f for f, cs in FAMILIES.items() for c in cs}


def rows(deal: np.ndarray, n: int = ROWS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """n random rows, split by deal into fit and eval halves."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(deal), n, replace=False))
    deals = np.unique(deal[pick]); rng.shuffle(deals)
    fit = np.isin(deal[pick], deals[: len(deals) // 2])
    return pick[fit], pick[~fit]


def signals(acts, audit: dict, at: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """The live poles with Silas or raw labels, and their signals on `at` (columns standardized)."""
    live = [r for r in audit["rows"] if r["kind"] != "dead"]
    S = np.stack([acts.pole(r["pole"])[at].astype(np.float64) for r in live], axis=1)
    S = (S - S.mean(axis=0)) / (S.std(axis=0) + 1e-12)
    return live, S


def r2(y: np.ndarray, X: np.ndarray, yv: np.ndarray, Xv: np.ndarray) -> float:
    A = np.c_[X, np.ones(len(X))]
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = yv - np.c_[Xv, np.ones(len(Xv))] @ w
    v = yv - yv.mean()
    return float(1 - (e @ e) / max(v @ v, 1e-12))


def match(nacts, naudit: dict, wacts, waudit: dict, deal: np.ndarray, log=print) -> dict:
    fit, ev = rows(deal)
    nl, Nf = signals(nacts, naudit, fit)
    _, Ne = signals(nacts, naudit, ev)
    wl, Wf = signals(wacts, waudit, fit)
    _, We = signals(wacts, waudit, ev)
    C = Nf.T @ Wf / len(fit)                                # [narrow, wide] correlation on fit rows
    wfam = np.array([FAM.get(r["label"], "raw") for r in wl])
    silas_w = wfam != "raw"
    out = []
    for i, r in enumerate(nl):
        f = FAM.get(r["label"])
        if f is None:
            continue
        order = np.argsort(-C[i])
        top3 = [j for j in order if silas_w[j]][:TOP3]
        top5 = [j for j in order if silas_w[j]][:TOP5]
        own = [j for j in order if wfam[j] == f][:TOP5]
        other = [j for j in order if silas_w[j] and wfam[j] != f][:TOP5]
        rec = {"pole": r["pole"], "label": r["label"], "family": f, "kind": r["kind"],
               "top3": [(wl[j]["label"], round(float(C[i, j]), 3)) for j in top3],
               "top5_labels": [wl[j]["label"] for j in top5], "top3_idx": top3}
        if len(own) == TOP5 and len(other) == TOP5:
            rec["r2_own"] = r2(Nf[:, i], Wf[:, own], Ne[:, i], We[:, own])
            rec["r2_other"] = r2(Nf[:, i], Wf[:, other], Ne[:, i], We[:, other])
        out.append(rec)

    def within(fams: np.ndarray) -> float:
        hits = [fams[j] == p["family"] for p in out for j in p["top3_idx"]]
        return float(np.mean(hits)) if hits else float("nan")

    obs = within(wfam)
    rng = np.random.default_rng(SEED)
    silas_idx = np.flatnonzero(silas_w)
    null = []
    for _ in range(NULLS):
        g = wfam.copy(); g[silas_idx] = rng.permutation(wfam[silas_idx]); null.append(within(g))
    split = [p for p in out if "r2_own" in p]
    late = [p for p in out if p["label"] in ("last", "third")]
    res = {"narrow_live": len(nl), "narrow_silas": len(out), "wide_live": len(wl), "wide_silas": int(silas_w.sum()),
           "within_top3": obs, "null_mean": float(np.mean(null)), "null_hi": float(np.percentile(null, 97.5)),
           "split_n": len(split), "split_own_wins": float(np.mean([p["r2_own"] > p["r2_other"] for p in split])) if split else None,
           "split_r2_own_mean": float(np.mean([p["r2_own"] for p in split])) if split else None,
           "split_r2_other_mean": float(np.mean([p["r2_other"] for p in split])) if split else None,
           "late_n": len(late), "late_both_seats": float(np.mean([("last" in p["top5_labels"]) and ("third" in p["top5_labels"]) for p in late])) if late else None,
           "poles": [{k: v for k, v in p.items() if k != "top3_idx"} for p in out]}
    log(f"within-family top-3 {obs:.3f} vs null {res['null_mean']:.3f} (hi {res['null_hi']:.3f}); split own wins {res['split_own_wins']} "
        f"(n {len(split)}); late both seats {res['late_both_seats']} (n {len(late)})")
    return res
