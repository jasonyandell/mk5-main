# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Formation (#194, already-clean test 1): read a series of in-run checkpoint audits.

For each run (one width, one seed) and each live pole of its final net:
- the **first clean step** is the earliest checkpoint from which the pole is clean at every later one;
- the **first label** is its label at that step; the **final label** is its label at the last checkpoint;
- **family kept** means the two labels lie in one of the squeeze's concept families
  (`lem.probe.squeeze.FAMILIES`; Silas concepts only, as registered; raw features are
  reported apart).

    python -u -m lem.probe formation-report AUDIT_ROOT RUNS_ROOT SAMPLE LABELS Q_DIR OUT.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from lem.probe.squeeze import FAMILIES

FAM = {c: f for f, cs in FAMILIES.items() for c in cs}
POSITION_TRICK = {"position", "trick"}
COUNT_MASTERY = {"count", "mastery"}
TRUMP_HELD = {"holds-boss-trump", "holds-sure-trump"}
MASTERY = {"holds-master", "masters-2plus", "can-take-it"}


def series(audit_run: Path) -> list[tuple[int, dict]]:
    """(step, audit.json) in step order."""
    out = []
    for d in sorted(audit_run.glob("step_*")):
        if (d / "audit.json").exists():
            out.append((int(d.name.split("_")[1]), json.loads((d / "audit.json").read_text())))
    return out


def poles(ser: list[tuple[int, dict]]) -> list[dict]:
    """Per pole of the final checkpoint: first clean step, first and final label, kinds over time."""
    steps = [s for s, _ in ser]
    rows = [{r["pole"]: r for r in a["rows"]} for _, a in ser]
    out = []
    for p, last in rows[-1].items():
        if last["kind"] == "dead":
            continue
        kinds = [rw[p]["kind"] for rw in rows]
        first = None
        if kinds[-1] == "clean":
            i = len(kinds) - 1
            while i > 0 and kinds[i - 1] == "clean":
                i -= 1
            first = i
        out.append({"pole": p, "kinds": kinds, "labels": [rw[p]["label"] for rw in rows],
                    "precision": [rw[p]["precision"] for rw in rows],
                    "first_clean_step": steps[first] if first is not None else None,
                    "first_label": rows[first][p]["label"] if first is not None else None,
                    "final_label": last["label"], "final_kind": last["kind"]})
    return out


def kept_share(ps: list[dict]) -> dict:
    """Of the poles clean at the end with Silas first and final labels: the share whose first
    label's family is the final label's, against independent draws from the final labels."""
    both = [p for p in ps if p["final_kind"] == "clean" and p["first_label"] in FAM and p["final_label"] in FAM]
    if not both:
        return {"n": 0}
    kept = sum(FAM[p["first_label"]] == FAM[p["final_label"]] for p in both)
    same_label = sum(p["first_label"] == p["final_label"] for p in both)
    f = Counter(FAM[p["final_label"]] for p in both)
    null = sum((c / len(both)) ** 2 for c in f.values())
    raw = [p for p in ps if p["final_kind"] == "clean" and not (p["first_label"] in FAM and p["final_label"] in FAM)]
    raw_kept = sum(p["first_label"].split(":")[:2] == p["final_label"].split(":")[:2] for p in raw if p["first_label"])
    return {"n": len(both), "kept": kept / len(both), "same_label": same_label / len(both), "null": null,
            "raw_n": len(raw), "raw_same_block": raw_kept / max(1, len(raw))}


def report(audit_root: str | Path, runs_root: str | Path, sample_path: str, labels_path: str, q_dir: str, log=print) -> dict:
    from lem.probe import fidelity
    from lem.probe.net import forward, load_net
    from lem.probe.sample import Sample
    sample = Sample(sample_path)
    with np.load(labels_path) as z:
        L, names = z["labels"], [str(n) for n in z["names"]]
    col = {n: i for i, n in enumerate(names)}
    out = {"runs": {}}
    for run_dir in sorted(Path(audit_root).iterdir()):
        ser = series(run_dir)
        if not ser:
            continue
        steps = [s for s, _ in ser]
        ps = poles(ser)
        traj = []
        for s, a in ser:
            net = load_net(str(Path(runs_root) / run_dir.name / f"step_{s:07d}.npz"))
            r, _, _ = fidelity.regret(sample, forward(net, sample.x), q_dir)
            traj.append({"step": s, "clean_share": a["shares"]["clean_share"], "live": a["shares"]["live"],
                         "dead": a["shares"]["dead"], "regret": float(np.nanmean(r))})
        # prediction 4: the "late in the trick" units (16×8): single-label clean vs broadened (last ∪ third)
        late = []
        if "h16x8" in run_dir.name:
            last_net = load_net(str(Path(runs_root) / run_dir.name / f"step_{steps[-1]:07d}.npz"))
            for p in ps:
                if p["final_label"] not in ("last", "third"):
                    continue
                gains = []
                for s, _ in ser:
                    acts = forward(load_net(str(Path(runs_root) / run_dir.name / f"step_{s:07d}.npz")), sample.x)
                    k = max(1, int(round(0.01 * len(sample))))
                    top = np.argsort(-acts.pole(p["pole"]), kind="stable")[:k]
                    a_, b_ = L[top, col["last"]], L[top, col["third"]]
                    single = max(a_.mean(), b_.mean())
                    gains.append({"step": s, "single": float(single), "union": float((a_ | b_).mean())})
                late.append({"pole": p["pole"], "final_label": p["final_label"], "gains": gains})
            del last_net
        kept = kept_share(ps)
        first_by_family = {}
        for p in ps:
            if p["first_clean_step"] is not None and p["final_label"] in FAM:
                first_by_family.setdefault(FAM[p["final_label"]], []).append(p["first_clean_step"])
        chains = [p["pole"] for p in ps if p["first_label"] in TRUMP_HELD and p["final_label"] in MASTERY]
        out["runs"][run_dir.name] = {"steps": steps, "trajectory": traj, "kept": kept, "first_by_family": first_by_family,
                                     "chains": chains, "late": late, "poles": ps}
        log(f"{run_dir.name}: clean {traj[0]['clean_share']:.3f} -> {traj[-1]['clean_share']:.3f}; kept {kept.get('kept', float('nan')):.3f} "
            f"vs null {kept.get('null', float('nan')):.3f} (n {kept['n']}); chains {len(chains)}")
    return out
