"""Distribution-free calibration for the float32 Metal reference lane.

The calibration and held-out ledgers must be disjoint paired runs of the same
seeds under ``engine=fused`` and ``engine=metal``. The report turns observed
absolute errors into finite-sample split-conformal upper bounds and verifies
their coverage on the untouched holdout. It also reports convergence and
root-argmax behavior, stratified by the CPU decision margin when available.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

DEFAULT_COVERAGE = 0.95
DEFAULT_GAP_TARGET = 0.05


def conformal_upper(errors, coverage: float = DEFAULT_COVERAGE) -> float:
    """Finite-sample upper bound for a future exchangeable absolute error."""
    x = np.sort(np.asarray(errors, dtype=np.float64).reshape(-1))
    if not len(x):
        raise ValueError("calibration needs at least one paired error")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between zero and one")
    rank = int(math.ceil((len(x) + 1) * coverage))
    return float(x[rank - 1]) if rank <= len(x) else float("inf")


def br_shortfall_upper(exact_gaps, sampled_upper_gaps,
                       coverage: float = DEFAULT_COVERAGE) -> float:
    """One-sided conformal bound on sampled-BR optimization shortfall.

    A trained candidate BR is a lower bound on the true best response. Its
    evaluation upper confidence limit still may miss the optimum. This bound
    calibrates that missing optimization mass against exact roots.
    """
    exact = np.asarray(exact_gaps, dtype=np.float64)
    sampled = np.asarray(sampled_upper_gaps, dtype=np.float64)
    if exact.shape != sampled.shape:
        raise ValueError("exact and sampled gap arrays must have equal shape")
    return conformal_upper(np.maximum(exact - sampled, 0.0), coverage)


def calibrated_gap_interval(candidate_gains, candidate_ses, shortfall_bar,
                            alpha: float = 0.05) -> tuple[float, float]:
    """Simultaneous candidate interval plus calibrated shortfall.

    ``alpha`` is the candidate-evaluation error budget only. The caller must
    combine it with the shortfall calibration's miscoverage by a union bound;
    for example, candidate alpha .025 plus 97.5% shortfall calibration gives
    at least 95% joint coverage. ``audit_sampled_gap`` enforces this split.
    """
    gain = np.asarray(candidate_gains, dtype=np.float64).reshape(-1)
    se = np.asarray(candidate_ses, dtype=np.float64).reshape(-1)
    if not len(gain) or gain.shape != se.shape:
        raise ValueError("candidate gains/SEs must be equal nonempty vectors")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    # Bonferroni two-sided normal interval for the fixed, independently
    # evaluated BR candidates. Calibration covers their optimization miss.
    z = NormalDist().inv_cdf(1.0 - alpha / (2.0 * len(gain)))
    lower = max(0.0, float(np.max(gain - z * se)))
    upper = max(0.0, float(np.max(gain + z * se))) + float(shortfall_bar)
    return lower, upper


def _load(path) -> dict[int, dict]:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("verdict") not in ("converged", "gap_capped"):
            continue
        rows[int(row["seed"])] = row
    return rows


def _pairs(cpu, metal):
    seeds = sorted(set(cpu) & set(metal))
    if not seeds:
        raise ValueError("paired ledgers have no usable seeds in common")
    return [(s, cpu[s], metal[s]) for s in seeds]


def _errors(pairs, field):
    return np.asarray([abs(float(c[field]) - float(m[field]))
                       for _, c, m in pairs], dtype=np.float64)


def _margin_bins(pairs):
    bins = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.20),
            (0.20, float("inf")))
    out = []
    for lo, hi in bins:
        rows = [(c, m) for _, c, m in pairs
                if "root_margin" in c and "root_argmax" in c
                and "root_argmax" in m
                and lo <= float(c["root_margin"]) < hi]
        out.append({
            "margin_lo": lo,
            "margin_hi": None if math.isinf(hi) else hi,
            "n": len(rows),
            "argmax_changes": sum(int(c["root_argmax"])
                                  != int(m["root_argmax"])
                                  for c, m in rows),
        })
    return out


def calibration_report(cal_pairs, hold_pairs,
                       coverage: float = DEFAULT_COVERAGE,
                       gap_target: float = DEFAULT_GAP_TARGET) -> dict:
    value_bar = conformal_upper(_errors(cal_pairs, "cfr_reference_value"),
                                coverage)
    gap_bar = conformal_upper(_errors(cal_pairs, "final_gap"), coverage)
    hv = _errors(hold_pairs, "cfr_reference_value")
    hg = _errors(hold_pairs, "final_gap")

    certain = wrong = uncertain = 0
    for _, cpu, metal in hold_pairs:
        truth = float(cpu["final_gap"]) <= gap_target
        gm = float(metal["final_gap"])
        if gm + gap_bar <= gap_target:
            pred = True
        elif gm - gap_bar > gap_target:
            pred = False
        else:
            uncertain += 1
            continue
        certain += 1
        wrong += pred != truth

    return {
        "contract": {
            "method": "split-conformal absolute-error upper bound",
            "coverage": coverage,
            "gap_target": gap_target,
            "calibration_n": len(cal_pairs),
            "heldout_n": len(hold_pairs),
        },
        "value": {
            "error_bar": value_bar,
            "heldout_covered": int(np.sum(hv <= value_bar)),
            "heldout_n": len(hv),
            "heldout_max_error": float(np.max(hv)),
        },
        "gap": {
            "error_bar": gap_bar,
            "heldout_covered": int(np.sum(hg <= gap_bar)),
            "heldout_n": len(hg),
            "heldout_max_error": float(np.max(hg)),
        },
        "convergence": {
            "certain_n": certain,
            "uncertain_n": uncertain,
            "wrong_when_certain": wrong,
        },
        "root_argmax_by_cpu_margin": _margin_bins(hold_pairs),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration-cpu", required=True)
    ap.add_argument("--calibration-metal", required=True)
    ap.add_argument("--heldout-cpu", required=True)
    ap.add_argument("--heldout-metal", required=True)
    ap.add_argument("--coverage", type=float, default=DEFAULT_COVERAGE)
    ap.add_argument("--gap-target", type=float, default=DEFAULT_GAP_TARGET)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    cal = _pairs(_load(a.calibration_cpu), _load(a.calibration_metal))
    hold = _pairs(_load(a.heldout_cpu), _load(a.heldout_metal))
    overlap = {s for s, _, _ in cal} & {s for s, _, _ in hold}
    if overlap:
        raise SystemExit(f"calibration/heldout seeds overlap: {sorted(overlap)}")
    report = calibration_report(cal, hold, a.coverage, a.gap_target)
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    finite = math.isfinite(report["value"]["error_bar"]) and \
        math.isfinite(report["gap"]["error_bar"])
    covered = (report["value"]["heldout_covered"] / len(hold) >= a.coverage
               and report["gap"]["heldout_covered"] / len(hold)
               >= a.coverage)
    return 0 if finite and covered else 1


if __name__ == "__main__":
    raise SystemExit(main())
