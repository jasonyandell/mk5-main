# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Belief probes (the paper's G1): does the student encode the teacher's
exact posterior over who holds what?

The teacher's belief at a position is exact and computable. For every
live tile the viewer does not hold and every other seat, Silas gives
P(seat holds tile | the viewer's information) under uniform worlds that
respect the shown voids (`SilasBelief.uniform_resolving` on that one tile,
then `(query (out c) (holds c <tile>))`). The **counting prior** ignores
the voids: a seat's remaining hand size over the hidden tiles. The
difference is what the voids tell the teacher.

A probe is a ridge regression from a representation to the posterior's
deviation from the prior, fit on one set of deals and scored on another
(R² over the defined entries, and over the entries the voids move). The
representations: the net's raw input bits (what the inputs state linearly),
its first hidden layer, its second, and both. Chance is the same probe on
row-shuffled targets.

    python -u -m lem.probe belief SAMPLE.npz OUT_DIR [--net ...] [--fit 20000] [--eval 20000]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW

from lem.probe.net import Activations
from lem.probe.sample import Sample

SEATS = 3                      # relative: left, partner, right
N_TILES = 28
RIDGE = 1.0
SEED = 20261016


def targets(sample: Sample, rows: np.ndarray, log=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row: the exact posterior P[rel, slot] (NaN where the tile is
    not live-hidden), the counting prior, and the defined-entry mask. Tiles
    sit in the net's canonical slots."""
    from scenario_core import SilasBelief, SilasFrame, SilasQuery
    canon = sample.canon()
    queries = {t: SilasQuery(f"(query (out c) (holds c {DOMINO_HIGH[t]}-{DOMINO_LOW[t]}))") for t in range(N_TILES)}
    n = len(rows)
    post = np.full((n, SEATS, N_TILES), np.nan)
    prior = np.full((n, SEATS, N_TILES), np.nan)
    for i, r in enumerate(rows):
        f = SilasFrame(*sample.frame_args(r))
        viewer, own = int(sample.viewer[r]), int(sample.own[r])
        sizes = f.sizes
        hidden = [t for t in range(N_TILES) if not (f.played >> t) & 1 and not (own >> t) & 1]
        total = sum(sizes[(viewer + rel) % 4] for rel in (1, 2, 3))
        if total == 0:
            continue
        for t in hidden:
            slot = canon[r, t]
            belief = SilasBelief.uniform_resolving(f, 1 << t, 64)
            masses = queries[t].summarize(belief, 10_000_000).presence
            for rel in (1, 2, 3):
                seat = (viewer + rel) % 4
                post[i, rel - 1, slot] = float(masses.get((f"S{seat}",), 0.0))
                prior[i, rel - 1, slot] = sizes[seat] / total
        if log is not None and i % 5000 == 0:
            log(f"  beliefs: row {i:,} of {n:,}")
    mask = ~np.isnan(post)
    return post, prior, mask


def ridge_fit(F: np.ndarray, Y: np.ndarray, M: np.ndarray, lam: float = RIDGE) -> np.ndarray:
    """One ridge per output column over the rows where that column is defined.
    F [n, p] float, Y [n, k], M [n, k] bool -> W [p + 1, k] (bias last)."""
    n, p = F.shape
    X = np.concatenate([F, np.ones((n, 1))], axis=1)
    W = np.zeros((p + 1, Y.shape[1]))
    reg = lam * np.eye(p + 1)
    reg[-1, -1] = 0.0
    for k in range(Y.shape[1]):
        m = M[:, k]
        if m.sum() < p + 2:
            continue
        Xk, yk = X[m], Y[m, k]
        W[:, k] = np.linalg.solve(Xk.T @ Xk + reg, Xk.T @ yk)
    return W


def ridge_predict(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.concatenate([F, np.ones((len(F), 1))], axis=1) @ W


def r2(pred: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    e = y[m] - pred[m]
    v = y[m] - y[m].mean()
    return float(1.0 - (e @ e) / max(v @ v, 1e-12))


def select_rows(sample: Sample, fit: int = 20_000, eval_n: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Deal-disjoint fit and eval rows, deterministic in the sample and the sizes."""
    rng = np.random.default_rng(SEED)
    deals = np.unique(sample.deal)
    rng.shuffle(deals)
    half = len(deals) // 2
    fit_deals = set(deals[:half].tolist())
    is_fit = np.array([d in fit_deals for d in sample.deal.tolist()])
    fit_rows = rng.choice(np.flatnonzero(is_fit), fit, replace=False)
    eval_rows = rng.choice(np.flatnonzero(~is_fit), eval_n, replace=False)
    return fit_rows, eval_rows


def targets_part(sample: Sample, out_dir: str | Path, part: int, parts: int, fit: int = 20_000, eval_n: int = 20_000,
                 log=print) -> Path:
    """The exact targets for one slice of the fit+eval rows (one `cpu-slot` call), saved as a part."""
    fit_rows, eval_rows = select_rows(sample, fit, eval_n)
    rows = np.concatenate([fit_rows, eval_rows])
    bounds = np.linspace(0, len(rows), parts + 1).astype(int)
    mine = rows[bounds[part]:bounds[part + 1]]
    post, prior, mask = targets(sample, mine, log)
    out = Path(out_dir) / f"targets_{part:02d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, rows=mine, post=post, prior=prior, start=bounds[part], stop=bounds[part + 1], parts=parts)
    log(f"{out}: {len(mine):,} rows")
    return out


def run(sample: Sample, acts: Activations, fit: int = 20_000, eval_n: int = 20_000, log=print,
        parts_dir: str | Path | None = None) -> dict:
    fit_rows, eval_rows = select_rows(sample, fit, eval_n)
    rng = np.random.default_rng(SEED + 1)
    if parts_dir is None:
        log(f"targets for {fit:,} fit rows and {eval_n:,} eval rows (deal-disjoint)")
        Pf, Bf, Mf = targets(sample, fit_rows, log)
        Pe, Be, Me = targets(sample, eval_rows, log)
    else:
        files = sorted(Path(parts_dir).glob("targets_*.npz"))
        rows_all = np.concatenate([fit_rows, eval_rows])
        post = np.full((len(rows_all), SEATS, N_TILES), np.nan)
        prior = np.full((len(rows_all), SEATS, N_TILES), np.nan)
        got = np.zeros(len(rows_all), bool)
        for f in files:
            with np.load(f) as z:
                a, b = int(z["start"]), int(z["stop"])
                if not np.array_equal(z["rows"], rows_all[a:b]):
                    raise AssertionError(f"{f} does not hold rows {a}:{b} of this selection")
                post[a:b], prior[a:b], got[a:b] = z["post"], z["prior"], True
        if not got.all():
            raise AssertionError(f"targets missing for {int((~got).sum()):,} rows; parts present: {len(files)}")
        Pf, Bf, Mf = post[:fit], prior[:fit], ~np.isnan(post[:fit])
        Pe, Be, Me = post[fit:], prior[fit:], ~np.isnan(post[fit:])
    Yf, Ye = (Pf - Bf).reshape(fit, -1), (Pe - Be).reshape(eval_n, -1)
    Mf2, Me2 = Mf.reshape(fit, -1), Me.reshape(eval_n, -1)
    moved = Me2 & (np.abs(Ye) > 1e-9)                      # where the voids move the posterior
    reps = {
        "x": (sample.x[fit_rows].astype(np.float64), sample.x[eval_rows].astype(np.float64)),
        "a1": (acts.a1[fit_rows].astype(np.float64), acts.a1[eval_rows].astype(np.float64)),
        "a2": (acts.a2[fit_rows].astype(np.float64), acts.a2[eval_rows].astype(np.float64)),
    }
    reps["a1+a2"] = (np.concatenate([reps["a1"][0], reps["a2"][0]], axis=1), np.concatenate([reps["a1"][1], reps["a2"][1]], axis=1))
    out = {"fit_rows": int(fit), "eval_rows": int(eval_n), "defined_eval_entries": int(Me2.sum()),
           "moved_eval_entries": int(moved.sum()), "moved_share": float(moved.sum() / max(Me2.sum(), 1)),
           "prior_mae": float(np.abs(Ye[Me2]).mean()), "prior_mae_moved": float(np.abs(Ye[moved]).mean()), "probes": {}}
    for name, (Ff, Fe) in reps.items():
        sf, mu = Ff.std(axis=0) + 1e-9, Ff.mean(axis=0)
        Ff, Fe = (Ff - mu) / sf, (Fe - mu) / sf
        W = ridge_fit(Ff, Yf, Mf2)
        pred = ridge_predict(Fe, W)
        perm = rng.permutation(fit)
        Wn = ridge_fit(Ff, Yf[perm], Mf2[perm])
        pred_n = ridge_predict(Fe, Wn)
        out["probes"][name] = {
            "r2": r2(pred, Ye, Me2), "r2_moved": r2(pred, Ye, moved),
            "mae": float(np.abs(Ye[Me2] - pred[Me2]).mean()), "mae_moved": float(np.abs(Ye[moved] - pred[moved]).mean()),
            "r2_shuffled": r2(pred_n, Ye, Me2), "r2_moved_shuffled": r2(pred_n, Ye, moved), "width": int(Ff.shape[1]),
        }
        log(f"  {name:6s} width {Ff.shape[1]:3d}: R2 {out['probes'][name]['r2']:.3f} (moved {out['probes'][name]['r2_moved']:.3f}); "
            f"shuffled {out['probes'][name]['r2_shuffled']:.3f}")
    return out


def write(result: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "belief.json").write_text(json.dumps(result, indent=1))
    lines = [f"Fit {result['fit_rows']:,} rows, eval {result['eval_rows']:,} rows (deal-disjoint); {result['defined_eval_entries']:,} defined "
             f"eval entries, {result['moved_eval_entries']:,} moved by the voids ({result['moved_share']:.3f}).",
             f"Counting prior MAE {result['prior_mae']:.4f} (moved entries {result['prior_mae_moved']:.4f}).", "",
             "| representation | width | R² all | R² moved | MAE all | MAE moved | R² shuffled | R² moved shuffled |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, p in result["probes"].items():
        lines.append(f"| {name} | {p['width']} | {p['r2']:.3f} | {p['r2_moved']:.3f} | {p['mae']:.4f} | {p['mae_moved']:.4f} | {p['r2_shuffled']:.3f} | {p['r2_moved_shuffled']:.3f} |")
    (out_dir / "belief.md").write_text("\n".join(lines) + "\n")
