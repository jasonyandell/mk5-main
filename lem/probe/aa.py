# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Split-half A/A for the probe kit's intervals (#191).

The audit quotes a Wilson 95% interval over the rows of a pole's top set, and
the belief probe quotes an R² with no interval. Rows are positions, and a
deal's positions share the hand and the voids, so a row interval may be too
narrow. This module asks how often the intervals are wrong, with no new
claim about the net: the sample's deals are split into two disjoint halves,
the audit's procedure is run on each half (its own top 1%, the pole's label
held fixed to the audit's), and the two halves' estimates are compared.

For a calibrated 95% interval:
  * each half's interval misses the other half's point estimate about
    16.6% of the time (the other estimate carries its own noise:
    P(|Z| > 1.96/sqrt(2)));
  * two intervals fail to overlap about 0.6% of the time;
  * z = (p_A − p_B) / sqrt(SE_A² + SE_B²) is about N(0, 1), so |z| > 1.96
    about 5% of the time.

Two interval estimators are compared per pole: row-Wilson (the audit's), and a
deal-clustered bootstrap over the top set's deals. The belief probe's A→B and
B→A R² are compared the same way, each with a deal-clustered bootstrap over
its eval deals.

    python -u -m lem.probe aa SAMPLE.npz LABELS.npz AUDIT_DIR BELIEF_PARTS OUT_DIR [--net]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lem.probe.audit import DEAD, TOP, CLEAN, feature_names, top_set, wilson
from lem.probe.net import Activations
from lem.probe.sample import Sample

SEED = 20261017
BOOT = 1000
Z = 1.959964
Z_OTHER = Z / np.sqrt(2.0)


def halves(deal: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Row masks of two deal-disjoint halves."""
    rng = np.random.default_rng(seed)
    deals = np.unique(deal)
    rng.shuffle(deals)
    a = np.isin(deal, deals[: len(deals) // 2])
    return a, ~a


def deal_bootstrap(y: np.ndarray, deal: np.ndarray, rng, boot: int = BOOT) -> np.ndarray:
    """`boot` means of y under resampling of its deals with replacement."""
    order = np.argsort(deal, kind="stable")
    y, d = y[order].astype(np.float64), deal[order]
    bounds = np.flatnonzero(np.diff(d, prepend=d[0] - 1))
    sums = np.add.reduceat(y, bounds)
    counts = np.diff(np.append(bounds, len(y))).astype(np.float64)
    pick = rng.integers(0, len(sums), (boot, len(sums)))
    return sums[pick].sum(axis=1) / counts[pick].sum(axis=1)


def wilson_se(p: float, k: int) -> float:
    lo, hi = wilson(int(round(p * k)), k)
    return (hi - lo) / (2 * Z)


def compare(est: list[dict]) -> dict:
    """Calibration counts over paired estimates {pA, loA, hiA, seA, pB, loB, hiB, seB}."""
    n = len(est)
    pA = np.array([e["pA"] for e in est]); pB = np.array([e["pB"] for e in est])
    loA = np.array([e["loA"] for e in est]); hiA = np.array([e["hiA"] for e in est])
    loB = np.array([e["loB"] for e in est]); hiB = np.array([e["hiB"] for e in est])
    seA = np.array([e["seA"] for e in est]); seB = np.array([e["seB"] for e in est])
    eps = 1e-9                                                       # Wilson at p = 1 returns hi = 1 up to rounding
    miss = np.concatenate([(pB < loA - eps) | (pB > hiA + eps), (pA < loB - eps) | (pA > hiB + eps)])
    no_overlap = (hiA < loB - eps) | (hiB < loA - eps)
    se = np.sqrt(seA ** 2 + seB ** 2)
    ok = se > 0
    z = (pA[ok] - pB[ok]) / se[ok]
    return {"pairs": n, "degenerate": int((~ok).sum()),
            "miss_other_point": float(miss.mean()), "miss_other_point_target": float(2 * (1 - _phi(Z_OTHER))),
            "no_overlap": float(no_overlap.mean()),
            "z_sd": float(z.std()) if ok.any() else None, "z_abs_gt_1_96": float(np.mean(np.abs(z) > Z)) if ok.any() else None,
            "z_abs_gt_1_96_target": 0.05, "mean_abs_diff": float(np.abs(pA - pB).mean())}


def recount(result: dict) -> dict:
    """Recompute the calibration counts from the stored per-pole estimates (no rerun)."""
    au = result["audit"]
    rows_w = [{"pA": pp["A"]["p"], "loA": pp["A"]["w_lo"], "hiA": pp["A"]["w_hi"], "seA": pp["A"]["w_se"],
               "pB": pp["B"]["p"], "loB": pp["B"]["w_lo"], "hiB": pp["B"]["w_hi"], "seB": pp["B"]["w_se"]} for pp in au["poles"]]
    rows_b = [{"pA": pp["A"]["p"], "loA": pp["A"]["b_lo"], "hiA": pp["A"]["b_hi"], "seA": pp["A"]["b_se"],
               "pB": pp["B"]["p"], "loB": pp["B"]["b_lo"], "hiB": pp["B"]["b_hi"], "seB": pp["B"]["b_se"]} for pp in au["poles"]]
    au["wilson"], au["bootstrap"] = compare(rows_w), compare(rows_b)
    nondeg = [w for w, b in zip(rows_w, rows_b) if b["seA"] ** 2 + b["seB"] ** 2 > 0]
    au["wilson_nondegenerate"] = compare(nondeg)
    return result


def _phi(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def audit_aa(sample: Sample, acts: Activations, labels: np.ndarray, concept_names: list[str], audit: dict, log=print) -> dict:
    x = sample.x.astype(bool)
    allc = np.concatenate([labels, x], axis=1)
    names = list(concept_names) + feature_names(x.shape[1])
    col = {nm: i for i, nm in enumerate(names)}
    a, b = halves(sample.deal)
    deal = sample.deal.astype(np.int64)
    rng = np.random.default_rng(SEED + 1)
    rows_w, rows_b, per_pole = [], [], []
    clean = {"A": {"wilson": 0, "boot": 0, "n": 0}, "B": {"wilson": 0, "boot": 0, "n": 0}}
    for r in audit["rows"]:
        if r["kind"] == "dead":
            continue
        p, lab = r["pole"], r["label"]
        signal = acts.pole(p)
        c = allc[:, col[lab]]
        est = {}
        for hn, h in (("A", a), ("B", b)):
            idx = np.flatnonzero(h)
            k = max(1, int(round(TOP * len(idx))))
            top = idx[top_set(signal[idx], k)]
            y = c[top]
            pm = float(y.mean())
            lo, hi = wilson(int(round(pm * k)), k)
            bs = deal_bootstrap(y, deal[top], rng)
            blo, bhi = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
            est[hn] = {"p": pm, "k": k, "deals": int(len(np.unique(deal[top]))), "w_lo": lo, "w_hi": hi, "w_se": wilson_se(pm, k),
                       "b_lo": blo, "b_hi": bhi, "b_se": float(bs.std())}
            clean[hn]["n"] += 1
            clean[hn]["wilson"] += pm >= CLEAN
        A, B = est["A"], est["B"]
        rows_w.append({"pA": A["p"], "loA": A["w_lo"], "hiA": A["w_hi"], "seA": A["w_se"], "pB": B["p"], "loB": B["w_lo"], "hiB": B["w_hi"], "seB": B["w_se"]})
        rows_b.append({"pA": A["p"], "loA": A["b_lo"], "hiA": A["b_hi"], "seA": A["b_se"], "pB": B["p"], "loB": B["b_lo"], "hiB": B["b_hi"], "seB": B["b_se"]})
        per_pole.append({"pole": p, "name": r["name"], "label": lab, "kind": r["kind"], "A": A, "B": B})
        if log is not None and len(per_pole) % 50 == 0:
            log(f"  pole {p}: {len(per_pole)} live poles done")
    both_clean = sum((pp["A"]["p"] >= CLEAN) == (pp["B"]["p"] >= CLEAN) for pp in per_pole)
    return {"seed": SEED, "boot": BOOT, "live_poles": len(per_pole),
            "rows_A": int(a.sum()), "rows_B": int(b.sum()), "deals_A": int(len(np.unique(deal[a]))), "deals_B": int(len(np.unique(deal[b]))),
            "wilson": compare(rows_w), "bootstrap": compare(rows_b),
            "clean_share": {"A": clean["A"]["wilson"] / max(clean["A"]["n"], 1), "B": clean["B"]["wilson"] / max(clean["B"]["n"], 1),
                            "same_kind_share": both_clean / max(len(per_pole), 1)},
            "poles": per_pole}


def r2_bootstrap(pred: np.ndarray, y: np.ndarray, m: np.ndarray, deal: np.ndarray, rng, boot: int = BOOT) -> np.ndarray:
    """R² under resampling of the eval rows' deals: per-deal sums of e², y, y² and counts."""
    e2 = np.where(m, (y - pred) ** 2, 0.0).sum(axis=1)
    sy = np.where(m, y, 0.0).sum(axis=1)
    syy = np.where(m, y * y, 0.0).sum(axis=1)
    cnt = m.sum(axis=1).astype(np.float64)
    order = np.argsort(deal, kind="stable")
    d = deal[order]
    bounds = np.flatnonzero(np.diff(d, prepend=d[0] - 1))
    parts = [np.add.reduceat(v[order], bounds) for v in (e2, sy, syy, cnt)]
    pick = rng.integers(0, len(bounds), (boot, len(bounds)))
    E2, SY, SYY, N = (v[pick].sum(axis=1) for v in parts)
    return 1.0 - E2 / np.maximum(SYY - SY * SY / N, 1e-12)


def belief_aa(sample: Sample, acts: Activations, parts_dir: str | Path, fit: int = 20_000, eval_n: int = 20_000, log=print) -> dict:
    from lem.probe.belief import N_TILES, SEATS, r2, ridge_fit, ridge_predict, select_rows
    fit_rows, eval_rows = select_rows(sample, fit, eval_n)
    rows_all = np.concatenate([fit_rows, eval_rows])
    post = np.full((len(rows_all), SEATS, N_TILES), np.nan); prior = np.full_like(post, np.nan)
    for f in sorted(Path(parts_dir).glob("targets_*.npz")):
        with np.load(f) as z:
            s, t = int(z["start"]), int(z["stop"])
            if not np.array_equal(z["rows"], rows_all[s:t]):
                raise AssertionError(f"{f} does not hold rows {s}:{t} of this selection")
            post[s:t], prior[s:t] = z["post"], z["prior"]
    if np.isnan(post).all(axis=(1, 2)).any():
        raise AssertionError("targets missing for some rows")
    Y = (post - prior).reshape(len(rows_all), -1)
    M = ~np.isnan(post).reshape(len(rows_all), -1)
    sides = {"A": slice(0, fit), "B": slice(fit, fit + eval_n)}        # A = the registered fit half, B = its eval half
    deal = sample.deal.astype(np.int64)[rows_all]
    reps = {"x": sample.x[rows_all].astype(np.float64), "a1": acts.a1[rows_all].astype(np.float64), "a2": acts.a2[rows_all].astype(np.float64)}
    reps["a1+a2"] = np.concatenate([reps["a1"], reps["a2"]], axis=1)
    rng = np.random.default_rng(SEED + 2)
    out = {"probes": {}}
    for name, F in reps.items():
        est = {}
        for fit_side, eval_side in (("A", "B"), ("B", "A")):
            Ff, Fe = F[sides[fit_side]], F[sides[eval_side]]
            sf, mu = Ff.std(axis=0) + 1e-9, Ff.mean(axis=0)
            Ff, Fe = (Ff - mu) / sf, (Fe - mu) / sf
            W = ridge_fit(Ff, Y[sides[fit_side]], M[sides[fit_side]])
            pred = ridge_predict(Fe, W)
            Ye, Me = Y[sides[eval_side]], M[sides[eval_side]]
            moved = Me & (np.abs(Ye) > 1e-9)
            res = {}
            for tag, mask in (("all", Me), ("moved", moved)):
                bs = r2_bootstrap(pred, Ye, mask, deal[sides[eval_side]], rng)
                lo, hi = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
                res[tag] = {"r2": r2(pred, Ye, mask), "lo": lo, "hi": hi, "se": float(bs.std())}
            est[f"{fit_side}->{eval_side}"] = res
            log(f"  {name:6s} fit {fit_side} eval {eval_side}: R2 {res['all']['r2']:.3f} [{res['all']['lo']:.3f}, {res['all']['hi']:.3f}] "
                f"moved {res['moved']['r2']:.3f} [{res['moved']['lo']:.3f}, {res['moved']['hi']:.3f}]")
        pairs = {}
        for tag in ("all", "moved"):
            u, v = est["A->B"][tag], est["B->A"][tag]
            pairs[tag] = compare([{"pA": u["r2"], "loA": u["lo"], "hiA": u["hi"], "seA": u["se"], "pB": v["r2"], "loB": v["lo"], "hiB": v["hi"], "seB": v["se"]}])
            pairs[tag]["z"] = float((u["r2"] - v["r2"]) / max(np.sqrt(u["se"] ** 2 + v["se"] ** 2), 1e-12))
        out["probes"][name] = {"estimates": est, "pairs": pairs}
    return out


def write(result: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aa.json").write_text(json.dumps(result, indent=1))
    au = result["audit"]
    lines = [f"Split-half A/A over {au['live_poles']} live poles (deals halved, seed {au['seed']}; A {au['deals_A']:,} deals / {au['rows_A']:,} rows, "
             f"B {au['deals_B']:,} / {au['rows_B']:,}; each half's own top 1%; the pole's label fixed to the audit's). Calibrated targets: "
             f"interval misses the other half's point 0.166; |z| > 1.96 0.05; no overlap 0.006.", "",
             "| interval | pairs | misses the other half's point | no overlap | sd of z | |z| > 1.96 | degenerate (SE 0) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for nm in ("wilson", "bootstrap"):
        c = au[nm]
        zsd = f"{c['z_sd']:.2f}" if c["z_sd"] is not None else "—"
        zr = f"{c['z_abs_gt_1_96']:.3f}" if c["z_abs_gt_1_96"] is not None else "—"
        lines.append(f"| {'row Wilson' if nm == 'wilson' else 'deal bootstrap'} | {c['pairs']} | {c['miss_other_point']:.3f} | {c['no_overlap']:.3f} | {zsd} | {zr} | {c['degenerate']} |")
    cs = au["clean_share"]
    lines += ["", f"Clean share by half: A {cs['A']:.3f}, B {cs['B']:.3f}; poles with the same kind (clean or not) in both halves: {cs['same_kind_share']:.3f}."]
    if "belief" not in result:
        (out_dir / "aa.md").write_text("\n".join(lines) + "\n")
        return
    lines += ["",
              "Belief probe, fit on one half and scored on the other, both ways (deal bootstrap over the eval deals):", "",
              "| representation | A→B R² all [95%] | B→A R² all [95%] | z | A→B moved | B→A moved | z |", "|---|---|---|---:|---|---|---:|"]
    for name, p in result["belief"]["probes"].items():
        u, v = p["estimates"]["A->B"], p["estimates"]["B->A"]
        lines.append(f"| {name} | {u['all']['r2']:.3f} [{u['all']['lo']:.3f}, {u['all']['hi']:.3f}] | {v['all']['r2']:.3f} [{v['all']['lo']:.3f}, {v['all']['hi']:.3f}] | {p['pairs']['all']['z']:+.2f} "
                     f"| {u['moved']['r2']:.3f} [{u['moved']['lo']:.3f}, {u['moved']['hi']:.3f}] | {v['moved']['r2']:.3f} [{v['moved']['lo']:.3f}, {v['moved']['hi']:.3f}] | {p['pairs']['moved']['z']:+.2f} |")
    (out_dir / "aa.md").write_text("\n".join(lines) + "\n")
