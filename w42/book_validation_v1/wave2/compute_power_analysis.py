#!/usr/bin/env python3
"""Power analysis for the full 50-seed bid-aware atlas corpus.

Computes, for each claim × bid bucket:
1. ch10-special-bid-mark-multiplier: mark_ev change-rate at each bid vs bid=30, with 95% CI.
2. ch02-bid-only-enough: paired (same seed) bid=30 vs bid=32 EV/mark_ev/threshold_mass deltas, with paired CI.
3. ch12-setter-pounce-high-bid-off: bid>=35 vs bid=30 setter-side action value deltas.

Output: power_analysis.csv
"""

from __future__ import annotations
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "w42/book_validation_v1/wave2/bid_aware_atlas"
CSV_PATH = OUTPUT_DIR / "bid_aware_actions.csv"
OUT_PATH = OUTPUT_DIR / "power_analysis.csv"


def bootstrap_ci(data: list[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap percentile CI. Returns (mean, lower, upper)."""
    import random
    n = len(data)
    if n < 2:
        m = data[0] if data else float("nan")
        return m, m, m
    mean_obs = statistics.mean(data)
    boot_means = []
    for _ in range(n_boot):
        sample = [data[random.randrange(n)] for _ in range(n)]
        boot_means.append(statistics.mean(sample))
    boot_means.sort()
    lo_idx = int(alpha / 2 * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot)
    return mean_obs, boot_means[lo_idx], boot_means[hi_idx]


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(v) -> float | None:
    if v is None or v == "" or v == "None":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def main():
    print(f"Loading {CSV_PATH}...")
    rows = load_csv(CSV_PATH)
    print(f"Loaded {len(rows)} rows.")

    # Index by (seed, decl_id, bid_value, decision_idx, action_slot)
    # For actual actions only (is_actual_action=1)
    actual_rows = [r for r in rows if r.get("is_actual_action") == "1"]
    print(f"Actual action rows: {len(actual_rows)}")

    # Group by (seed, decl_id, bid_value, decision_idx)
    # Key: (seed, decl_id, decision_idx) -> bid -> mark_ev
    by_key: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in actual_rows:
        seed = int(r["seed"])
        decl_id = int(r["decl_id"])
        decision_idx = int(r["decision_idx"])
        bid = int(r["bid_value"])
        key = (seed, decl_id, decision_idx)
        by_key[key][bid] = {
            "mark_ev": safe_float(r.get("mark_ev")),
            "p_make": safe_float(r.get("p_make")),
            "threshold_mass": safe_float(r.get("threshold_mass")),
            "mean": safe_float(r.get("mean")),
            "seat_role": r.get("seat_role", ""),
            "team": r.get("team", ""),
        }

    bid_values = [30, 32, 35, 36, 39, 42, 84]
    results = []

    # ─────────────────────────────────────────────────────────
    # Claim 1: ch10-special-bid-mark-multiplier
    # mark_ev change-rate at each bid vs bid=30
    # ─────────────────────────────────────────────────────────
    print("\n=== ch10-special-bid-mark-multiplier ===")
    for bv in bid_values:
        if bv == 30:
            continue
        deltas = []
        for key, bid_dict in by_key.items():
            me_30 = bid_dict.get(30, {}).get("mark_ev")
            me_bv = bid_dict.get(bv, {}).get("mark_ev")
            if me_30 is not None and me_bv is not None:
                deltas.append(abs(me_bv - me_30))

        n = len(deltas)
        if n < 2:
            print(f"  bid={bv}: insufficient data (n={n})")
            continue

        # Change rate = fraction of decisions where mark_ev changed by > 0.001
        changed = [d for d in deltas if d > 0.001]
        change_rate = len(changed) / n
        # CI on change_rate via bootstrap
        indicators = [1.0 if d > 0.001 else 0.0 for d in deltas]
        mean_obs, ci_lo, ci_hi = bootstrap_ci(indicators)
        half_width = (ci_hi - ci_lo) / 2
        mean_delta = statistics.mean(deltas)
        std_delta = statistics.stdev(deltas) if n > 1 else 0.0
        sem_delta = std_delta / math.sqrt(n)

        print(f"  bid={bv}: n={n}, change_rate={change_rate:.4f} ({ci_lo:.4f}-{ci_hi:.4f}), mean_|delta|={mean_delta:.4f}")
        results.append({
            "claim_id": "ch10-special-bid-mark-multiplier",
            "bid_value": bv,
            "comparison": f"bid={bv}_vs_bid=30",
            "metric": "mark_ev_change_rate",
            "n": n,
            "value": round(change_rate, 4),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
            "half_width": round(half_width, 4),
            "mean_abs_delta": round(mean_delta, 4),
            "std_delta": round(std_delta, 4),
            "sem_delta": round(sem_delta, 4),
            "power_verdict": "sufficient" if n >= 100 and half_width < 0.05 else "borderline" if n >= 50 else "insufficient",
            "notes": "Fraction of paired decisions where |mark_ev(bid) - mark_ev(30)| > 0.001",
        })

    # ─────────────────────────────────────────────────────────
    # Claim 2: ch02-bid-only-enough
    # Paired (same seed, decl_id, decision_idx) bid=30 vs bid=32 EV / mark_ev / threshold_mass deltas
    # ─────────────────────────────────────────────────────────
    print("\n=== ch02-bid-only-enough ===")
    for metric_name in ["mark_ev", "p_make", "threshold_mass"]:
        deltas = []
        for key, bid_dict in by_key.items():
            v_30 = bid_dict.get(30, {}).get(metric_name)
            v_32 = bid_dict.get(32, {}).get(metric_name)
            if v_30 is not None and v_32 is not None:
                deltas.append(v_32 - v_30)

        n = len(deltas)
        if n < 2:
            print(f"  {metric_name}: insufficient (n={n})")
            continue

        mean_d, ci_lo, ci_hi = bootstrap_ci(deltas)
        half_width = (ci_hi - ci_lo) / 2
        std_d = statistics.stdev(deltas) if n > 1 else 0.0
        sem_d = std_d / math.sqrt(n)
        # Effect size: Cohen's d for paired comparison
        cohen_d = mean_d / std_d if std_d > 0 else float("nan")

        print(f"  {metric_name}: n={n}, delta={mean_d:.4f} ({ci_lo:.4f}-{ci_hi:.4f}), cohen_d={cohen_d:.4f}")
        results.append({
            "claim_id": "ch02-bid-only-enough",
            "bid_value": "32_vs_30",
            "comparison": f"bid=32_minus_bid=30_{metric_name}",
            "metric": f"paired_delta_{metric_name}",
            "n": n,
            "value": round(mean_d, 4),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
            "half_width": round(half_width, 4),
            "cohen_d": round(cohen_d, 4) if math.isfinite(cohen_d) else None,
            "std_delta": round(std_d, 4),
            "sem_delta": round(sem_d, 4),
            "power_verdict": (
                "sufficient" if n >= 100 and half_width < 0.05 and abs(mean_d) > 2 * half_width
                else "borderline" if n >= 100 and half_width < 0.1
                else "insufficient"
            ),
            "notes": "Paired delta (bid=32 - bid=30) per (seed,decl_id,decision_idx) actual-action",
        })

    # ─────────────────────────────────────────────────────────
    # Claim 3: ch12-setter-pounce-high-bid-off
    # Setter-side (team=defense) action value deltas at bid>=35 vs bid=30
    # ─────────────────────────────────────────────────────────
    print("\n=== ch12-setter-pounce-high-bid-off ===")
    for bv in [35, 36, 39, 42, 84]:
        deltas = []
        for key, bid_dict in by_key.items():
            # Only setter-side decisions
            entry_30 = bid_dict.get(30, {})
            entry_bv = bid_dict.get(bv, {})
            if entry_30.get("team") != "defense" and entry_bv.get("team") != "defense":
                continue
            # Use mean Q (scalar EV from the oracle) as the proxy
            v_30 = entry_30.get("mean")
            v_bv = entry_bv.get("mean")
            if v_30 is not None and v_bv is not None:
                if entry_30.get("team") == "defense" or entry_bv.get("team") == "defense":
                    deltas.append(v_bv - v_30)

        n = len(deltas)
        if n < 2:
            print(f"  bid={bv}: insufficient setter data (n={n})")
            continue

        mean_d, ci_lo, ci_hi = bootstrap_ci(deltas)
        half_width = (ci_hi - ci_lo) / 2
        std_d = statistics.stdev(deltas) if n > 1 else 0.0
        cohen_d = mean_d / std_d if std_d > 0 else float("nan")

        print(f"  bid={bv}: n={n}, delta_mean_q={mean_d:.4f} ({ci_lo:.4f}-{ci_hi:.4f}), cohen_d={cohen_d:.4f}")
        results.append({
            "claim_id": "ch12-setter-pounce-high-bid-off",
            "bid_value": bv,
            "comparison": f"bid={bv}_vs_bid=30_setter_mean_Q",
            "metric": "paired_delta_mean_Q_setter",
            "n": n,
            "value": round(mean_d, 4),
            "ci_lower_95": round(ci_lo, 4),
            "ci_upper_95": round(ci_hi, 4),
            "half_width": round(half_width, 4),
            "cohen_d": round(cohen_d, 4) if math.isfinite(cohen_d) else None,
            "std_delta": round(std_d, 4),
            "sem_delta": round(std_d / math.sqrt(n), 4),
            "power_verdict": (
                "sufficient" if n >= 50 and half_width < 1.0 and abs(mean_d) > half_width
                else "borderline" if n >= 30
                else "insufficient"
            ),
            "notes": "Setter-team (defense) mean Q delta (bid=X - bid=30), proxy for pounce value shift",
        })

    # Write output
    fieldnames = [
        "claim_id", "bid_value", "comparison", "metric", "n",
        "value", "ci_lower_95", "ci_upper_95", "half_width",
        "cohen_d", "mean_abs_delta", "std_delta", "sem_delta",
        "power_verdict", "notes",
    ]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {OUT_PATH}")

    # Summary
    print("\n=== POWER VERDICT SUMMARY ===")
    from collections import Counter
    verdicts = Counter(r["power_verdict"] for r in results)
    for v, c in sorted(verdicts.items()):
        print(f"  {v}: {c}")

    print("\n=== CLAIM SUMMARIES ===")
    by_claim = defaultdict(list)
    for r in results:
        by_claim[r["claim_id"]].append(r)
    for claim, rr in sorted(by_claim.items()):
        verdicts = [r["power_verdict"] for r in rr]
        all_sufficient = all(v == "sufficient" for v in verdicts)
        any_sufficient = any(v == "sufficient" for v in verdicts)
        print(f"\n  {claim}:")
        print(f"    rows={len(rr)}, all_sufficient={all_sufficient}, any_sufficient={any_sufficient}")
        for r in rr:
            print(f"      {r['comparison']}: n={r['n']}, value={r['value']}, CI=[{r['ci_lower_95']},{r['ci_upper_95']}], verdict={r['power_verdict']}")


if __name__ == "__main__":
    main()
