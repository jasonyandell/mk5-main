"""Lens v1 reproducibility / re-analysis script.

Reads the per-hand margins CSV produced by `round_robin.py` and recomputes:
  - per-pairing mean margin + 95% bootstrap CI
  - decisive-hand rate, A win rate
  - verdict statement

This script does NOT touch the model or play games. Use it to re-derive the
verdict from existing CSVs.

Usage:
    python w42/lens_v1/analyze.py [--results-dir w42/lens_v1/results]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = values.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_per_hand(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory containing per_hand_margins.csv",
    )
    parser.add_argument("--out-table", default=None,
                        help="Optional path to write the analysis table")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    per_hand_csv = results_dir / "per_hand_margins.csv"
    if not per_hand_csv.exists():
        print(f"Missing: {per_hand_csv}", file=sys.stderr)
        return 1

    rows = load_per_hand(per_hand_csv)

    # Group by (utility_a, utility_b, n_samples)
    groups: dict[tuple, list[float]] = {}
    a_wins_per_group: dict[tuple, int] = {}
    decisive_per_group: dict[tuple, int] = {}
    for r in rows:
        key = (r["utility_a"], r["utility_b"], int(r["n_samples"]))
        margin = int(r["margin"])
        groups.setdefault(key, []).append(margin)
        a_wins_per_group[key] = a_wins_per_group.get(key, 0) + (1 if margin > 0 else 0)
        decisive_per_group[key] = decisive_per_group.get(key, 0) + (1 if margin != 0 else 0)

    print(f"\n{'utility_a':>12}  {'utility_b':>12}  {'N':>3}  {'n_hands':>7}  "
          f"{'mean_margin':>11}  {'95% CI':>23}  {'a_wr':>5}  {'decisive':>8}  CI≠0", flush=True)
    print("-" * 110, flush=True)

    summaries: list[dict] = []
    for (ua, ub, n_samp), margins_list in sorted(groups.items()):
        margins = np.array(margins_list, dtype=np.float64)
        mean = float(margins.mean())
        lo, hi = bootstrap_ci_mean(margins)
        a_wr = a_wins_per_group[(ua, ub, n_samp)] / margins.size
        dec = decisive_per_group[(ua, ub, n_samp)] / margins.size
        excl = (lo > 0) or (hi < 0)
        marker = "**" if excl else "  "
        print(
            f"{ua:>12}  {ub:>12}  {n_samp:>3}  {margins.size:>7}  "
            f"{mean:>+11.3f}  [{lo:>+8.3f}, {hi:>+8.3f}]  {a_wr*100:>4.1f}%  "
            f"{dec*100:>7.1f}%  {marker}",
            flush=True,
        )
        summaries.append({
            "utility_a": ua, "utility_b": ub, "n_samples": n_samp,
            "n_hands": int(margins.size),
            "mean_margin": mean,
            "ci_lo_95": lo, "ci_hi_95": hi,
            "ci_excludes_zero": int(excl),
            "a_win_rate": a_wr,
            "decisive_rate": dec,
        })

    # Verdict
    decisive = [s for s in summaries if s["ci_excludes_zero"]]
    print()
    if decisive:
        # Pick highest-magnitude decisive pairing
        winner = max(decisive, key=lambda s: abs(s["mean_margin"]))
        if winner["mean_margin"] > 0:
            v = (f"VERDICT: Lens({winner['utility_a']}) beats "
                 f"Lens({winner['utility_b']}) by {winner['mean_margin']:+.2f} pts/hand "
                 f"(95% CI [{winner['ci_lo_95']:+.2f}, {winner['ci_hi_95']:+.2f}]).")
        else:
            v = (f"VERDICT: Lens({winner['utility_b']}) beats "
                 f"Lens({winner['utility_a']}) by {-winner['mean_margin']:+.2f} pts/hand "
                 f"(95% CI [{-winner['ci_hi_95']:+.2f}, {-winner['ci_lo_95']:+.2f}]).")
    else:
        v = (f"VERDICT: No utility distinguishably beats another at the tested "
             f"hand counts. All {len(summaries)} pairing 95% CIs span zero.")
    print(v, flush=True)

    if args.out_table:
        with open(args.out_table, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote {args.out_table}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
