# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Fidelity to the teacher by concept region (the paper's G7).

The teacher scores every legal move of a position (`q`, the belief-weighted
make probability of the plan that starts with it). The student's *regret*
at a position is the teacher's best `q` minus the `q` of the move the student
picks, in the mover's units (a defender's values are flipped), so zero is
agreement with the teacher's best and the mean regret is `scenario.field_train`'s
grade. This module asks *where* the regret sits: by the Silas concepts that
hold at the position (the audit's labels), by trick position, and by how
many legal moves the position offers.

    python -u -m lem.probe fidelity-collect SAMPLE.npz POUR_DIR K OUT.npz   # q for the sample's rows from pour K
    python -u -m lem.probe fidelity SAMPLE.npz LABELS.npz Q_DIR OUT_DIR [--net]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lem.probe.net import Activations, picks
from lem.probe.sample import MAX_ACTIONS, Sample

SEED = 20261016
BOOT = 2000


def collect(sample: Sample, pour_dir: str | Path, k: int, log=print) -> dict[str, np.ndarray]:
    """The teacher's q for every legal move of the sample's rows that came from pour k
    (in the row's `actions` order, −inf padded), with the mover's sign."""
    from scenario.atlas import load
    rows = np.flatnonzero(sample.pour == k)
    p = load(pour_dir)
    q = np.full((len(rows), MAX_ACTIONS), -np.inf)
    sign = np.zeros(len(rows), np.int8)
    for i, r in enumerate(rows):
        pos = p.position(int(sample.index[r]))
        if tuple(pos.actions) != tuple(int(a) for a in sample.actions[r] if a >= 0) or pos.chosen != int(sample.chosen[r]):
            raise AssertionError(f"row {r} is not position {sample.index[r]} of pour {k}")
        q[i, :len(pos.actions)] = pos.q
        sign[i] = pos.sign
        if log is not None and i % 20000 == 0:
            log(f"  pour {k}: row {i:,} of {len(rows):,}")
    return {"rows": rows, "q": q, "sign": sign}


def regret(sample: Sample, acts: Activations, q_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: the student's regret (NaN where no q was collected), the teacher's
    best-minus-second gap (how decisive the position is), and the number of legal moves."""
    n = len(sample)
    q = np.full((n, MAX_ACTIONS), np.nan)
    sign = np.zeros(n, np.int8)
    for f in sorted(Path(q_dir).glob("q_*.npz")):
        with np.load(f) as z:
            q[z["rows"]] = z["q"]
            sign[z["rows"]] = z["sign"]
    pick = picks(acts.out, sample.actions, sample.canon())           # real tile ids
    actions = sample.actions.astype(np.int64)
    qm = q * sign[:, None]                                            # the mover's way up
    qm[np.isneginf(q) | np.isposinf(q)] = -np.inf
    best = qm.max(axis=1)
    at = (actions == pick[:, None])
    q_pick = np.where(at.any(axis=1), (np.where(at, qm, -np.inf)).max(axis=1), np.nan)
    r = best - q_pick
    srt = np.sort(np.where(np.isfinite(qm), qm, -np.inf), axis=1)
    gap = srt[:, -1] - srt[:, -2]
    n_legal = (actions >= 0).sum(axis=1)
    r[np.isnan(q).all(axis=1)] = np.nan
    return r, gap, n_legal


def by_group(values: np.ndarray, groups: dict[str, np.ndarray], deals: np.ndarray, rng) -> dict:
    """Mean regret per group with a deal-clustered bootstrap interval."""
    out = {}
    ok = ~np.isnan(values)
    uniq = np.unique(deals[ok])
    for name, m in groups.items():
        mm = m & ok
        if mm.sum() == 0:
            continue
        v, d = values[mm], deals[mm]
        order = np.argsort(d, kind="stable")
        v, d = v[order], d[order]
        bounds = np.flatnonzero(np.diff(d, prepend=d[0] - 1))
        sums = np.add.reduceat(v, bounds)
        counts = np.diff(np.append(bounds, len(v)))
        means = []
        for _ in range(BOOT):
            pick = rng.integers(0, len(sums), len(sums))
            means.append(sums[pick].sum() / counts[pick].sum())
        lo, hi = np.percentile(means, [2.5, 97.5])
        out[name] = {"n": int(mm.sum()), "mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}
    return out


def run(sample: Sample, acts: Activations, labels: np.ndarray, names: list[str], q_dir: str | Path, log=print) -> dict:
    rng = np.random.default_rng(SEED)
    r, gap, n_legal = regret(sample, acts, q_dir)
    ok = ~np.isnan(r)
    log(f"regret on {ok.sum():,} rows: mean {np.nanmean(r):.4f}, agree {np.mean(r[ok] == 0):.3f}")
    deals = sample.deal
    groups = {"all": ok}
    for k, nm in enumerate(names):
        groups[f"concept:{nm}"] = labels[:, k]
        groups[f"not:{nm}"] = ~labels[:, k]
    for k in range(2, 8):
        groups[f"legal:{k}"] = n_legal == k
    for lo, hi, nm in ((0, 1e-9, "gap:0"), (1e-9, 0.02, "gap:(0,0.02]"), (0.02, 0.1, "gap:(0.02,0.1]"), (0.1, 1.01, "gap:>0.1")):
        groups[nm] = (gap > lo) & (gap <= hi) if lo > 0 else (gap <= hi)
    table = by_group(r, groups, deals, rng)
    overall = table["all"]["mean"]
    concepts = {nm: {"holds": table.get(f"concept:{nm}"), "absent": table.get(f"not:{nm}")} for nm in names}
    ranked = sorted((nm for nm in names if table.get(f"concept:{nm}")), key=lambda nm: -table[f"concept:{nm}"]["mean"])
    return {"rows": int(ok.sum()), "overall": table["all"], "agree": float(np.mean(r[ok] == 0)),
            "by_concept": concepts, "ranked_high": ranked[:10], "ranked_low": ranked[-10:][::-1],
            "by_legal": {k: v for k, v in table.items() if k.startswith("legal:")},
            "by_gap": {k: v for k, v in table.items() if k.startswith("gap:")},
            "regret_share_top_decile": float(np.sort(r[ok])[::-1][:max(1, ok.sum() // 10)].sum() / max(r[ok].sum(), 1e-12)),
            "overall_mean": overall}


def write(result: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fidelity.json").write_text(json.dumps(result, indent=1))
    o = result["overall"]
    lines = [f"Regret of the net's pick under the teacher's plan, make-probability units, {result['rows']:,} rows: "
             f"mean {o['mean']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}] (deal bootstrap); exact agreement {result['agree']:.3f}; "
             f"the top decile of positions carries {result['regret_share_top_decile']:.3f} of all regret.", "",
             "| concept | n (holds) | regret when it holds [95%] | regret when absent |", "|---|---:|---|---:|"]
    for nm, v in result["by_concept"].items():
        h, a = v["holds"], v["absent"]
        if h is None:
            continue
        lines.append(f"| {nm} | {h['n']:,} | {h['mean']:.4f} [{h['lo']:.4f}, {h['hi']:.4f}] | {a['mean']:.4f} |" if a else f"| {nm} | {h['n']:,} | {h['mean']:.4f} [{h['lo']:.4f}, {h['hi']:.4f}] | |")
    lines += ["", "| legal moves | n | regret [95%] |", "|---|---:|---|"]
    for k, v in result["by_legal"].items():
        lines.append(f"| {k.split(':')[1]} | {v['n']:,} | {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] |")
    lines += ["", "| teacher's best-minus-second gap | n | regret [95%] |", "|---|---:|---|"]
    for k, v in result["by_gap"].items():
        lines.append(f"| {k.split(':', 1)[1]} | {v['n']:,} | {v['mean']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}] |")
    (out_dir / "fidelity.md").write_text("\n".join(lines) + "\n")
