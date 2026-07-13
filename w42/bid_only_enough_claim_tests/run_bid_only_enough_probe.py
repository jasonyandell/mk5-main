#!/usr/bin/env python3
"""Tiny bid-margin counterfactual probe for the Chapter 2 bid-only-enough claim.

The existing W42 static report can describe hand/declaration risk, but it cannot
test whether bidding extra points is strategically different from bidding only
enough. This probe uses the existing ``forge.bidding`` simulator to hold the
deal and declaration fixed, vary only bid threshold / auction high bid, and
summarize the resulting make-probability and one-mark swing deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.bidding.estimator import BID_THRESHOLDS, evaluate_bids
from forge.bidding.inference import PolicyModel
from forge.bidding.simulator import simulate_games
from forge.oracle.declarations import DECL_ID_TO_NAME
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import DOMINOES, DOMINO_COUNT_POINTS
from w42.bidding_risk_budget_claim_validation.validate_bidding_risk_budget import (
    domino_name,
    hand_eval,
)


OUT_DIR = ROOT / "w42" / "bid_only_enough_claim_tests"
BEAD_ID = "t42-0b4l.5"
CURRENT_HIGH_BIDS = (29, 30, 31, 34, 35)
PIP_DECLS = tuple(range(7))


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def hand_label(hand: list[int]) -> str:
    return ",".join(domino_name(d) for d in hand)


def risk_bucket(unique_exposed_points: int) -> str:
    if unique_exposed_points <= 12:
        return "risk_le_12"
    if unique_exposed_points <= 20:
        return "risk_13_20"
    return "risk_gt_20"


def static_key(ev: dict[str, Any]) -> tuple[int, int, int]:
    """Rank declarations by the existing static detector's conservative shape."""
    return (
        -int(ev["unique_exposed_points"]),
        int(ev["trump_count"]),
        int(ev["held_count_points"]),
    )


def pick_declarations(hand: list[int], n: int) -> list[tuple[int, dict[str, Any]]]:
    scored: list[tuple[tuple[int, int, int], int, dict[str, Any]]] = []
    hand_tuple = tuple(sorted(hand))
    for decl_id in PIP_DECLS:
        ev = hand_eval(hand_tuple, decl_id)
        scored.append((static_key(ev), decl_id, ev))
    scored.sort(reverse=True)
    return [(decl_id, ev) for _key, decl_id, ev in scored[:n]]


def bid_results_by_threshold(points: list[int], decl_id: int) -> dict[int, dict[str, float]]:
    result = evaluate_bids(points, decl_id)
    rows: dict[int, dict[str, float]] = {}
    for bid in result.bid_results:
        if bid.bid in BID_THRESHOLDS:
            rows[bid.bid] = {
                "p_make": float(bid.p_make),
                "mark_swing": float(bid.mark_swing),
                "ci_low": float(bid.ci_low),
                "ci_high": float(bid.ci_high),
            }
    return rows


def make_generation_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "w42.bid_only_enough_generation_spec.v1",
        "bead": BEAD_ID,
        "purpose": "Generate auction-aware candidate bid rows that separate static hand/declaration strength from bid-level margin quality.",
        "current_probe_command": (
            "python w42/bid_only_enough_claim_tests/run_bid_only_enough_probe.py "
            f"--start-seed {args.start_seed} --hands {args.hands} --samples {args.samples} "
            f"--declarations-per-hand {args.declarations_per_hand} --device {args.device}"
        ),
        "minimum_viable_rows": [
            "seed",
            "hand",
            "decl_id",
            "decl_name",
            "current_high_bid",
            "minimum_winning_bid",
            "actual_bid",
            "unnecessary_bid_margin",
            "points_samples",
            "p_make",
            "mark_swing",
            "static_unique_exposed_points",
            "static_trump_count",
            "static_risk_bucket",
        ],
        "larger_run_recommendation": {
            "hands": ">=1000 arbitrary generated hands or auction-policy hands",
            "samples_per_decl": ">=200 for screening, >=500 for narrower claim promotion",
            "auction_context": "Add real or simulated auction histories when available: bidder seat, partner/opponent bids, current high bid, score, pass/bid alternatives.",
            "primary_test": "Within identical seed/hand/decl/current_high groups, compare minimum_winning_bid against higher actual_bid rows; report p_make and mark_swing deltas by margin and static risk bucket.",
        },
        "anti_leakage_boundary": "Static features use only P0 hand and candidate declaration. Points samples and make outcomes are offline labels, not live bidding features.",
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = PolicyModel(
        checkpoint_path=args.checkpoint,
        device=args.device,
        compile_model=False,
    )

    hand_decl_rows: list[dict[str, Any]] = []
    bid_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    seeds = list(range(args.start_seed, args.start_seed + args.hands))
    for seed in seeds:
        p0_hand = list(deal_from_seed(seed)[0])
        for decl_id, ev in pick_declarations(p0_hand, args.declarations_per_hand):
            sim_seed = args.sim_seed + seed * 1000 + decl_id
            points_tensor = simulate_games(
                model=model,
                bidder_hand=p0_hand,
                decl_id=decl_id,
                n_games=args.samples,
                seed=sim_seed,
                greedy=True,
            )
            points = [int(x) for x in points_tensor.cpu().tolist()]
            by_bid = bid_results_by_threshold(points, decl_id)
            static_bucket = risk_bucket(int(ev["unique_exposed_points"]))
            base = {
                "seed": seed,
                "hand": hand_label(p0_hand),
                "decl_id": decl_id,
                "decl_name": DECL_ID_TO_NAME.get(decl_id, f"decl_{decl_id}"),
                "n_samples": args.samples,
                "sim_seed": sim_seed,
                "static_unique_exposed_points": int(ev["unique_exposed_points"]),
                "static_naive_exposed_points": int(ev["naive_exposed_points"]),
                "static_duplicate_overcount_points": int(ev["duplicate_overcount_points"]),
                "static_bid_ceiling_proxy": int(ev["bid_ceiling_proxy"]),
                "static_trump_count": int(ev["trump_count"]),
                "static_held_count_points": int(ev["held_count_points"]),
                "static_risk_bucket": static_bucket,
            }
            hand_decl_rows.append(
                {
                    **base,
                    "mean_points": round(mean([float(p) for p in points]), 6),
                    "min_points": min(points),
                    "max_points": max(points),
                    "points_ge_30_rate": round(sum(1 for p in points if p >= 30) / len(points), 6),
                    "points_ge_35_rate": round(sum(1 for p in points if p >= 35) / len(points), 6),
                    "points_ge_42_rate": round(sum(1 for p in points if p >= 42) / len(points), 6),
                }
            )
            sample_rows.append({**base, "points": "|".join(str(p) for p in points)})

            for current_high_bid in CURRENT_HIGH_BIDS:
                minimum_winning_bid = max(30, current_high_bid + 1)
                min_result = by_bid[minimum_winning_bid]
                for actual_bid in range(minimum_winning_bid, 43):
                    result = by_bid[actual_bid]
                    row = {
                        **base,
                        "current_high_bid": current_high_bid,
                        "minimum_winning_bid": minimum_winning_bid,
                        "actual_bid": actual_bid,
                        "unnecessary_bid_margin": actual_bid - minimum_winning_bid,
                        "p_make": round(result["p_make"], 6),
                        "mark_swing": round(result["mark_swing"], 6),
                        "ci_low": round(result["ci_low"], 6),
                        "ci_high": round(result["ci_high"], 6),
                        "delta_p_make_vs_min": round(result["p_make"] - min_result["p_make"], 6),
                        "delta_mark_swing_vs_min": round(result["mark_swing"] - min_result["mark_swing"], 6),
                    }
                    bid_rows.append(row)
                    if actual_bid > minimum_winning_bid:
                        contrast_rows.append(row)

    if args.smoke:
        smoke_assertions(hand_decl_rows, bid_rows, contrast_rows)

    margin_summary = summarize_margins(contrast_rows, ["unnecessary_bid_margin"])
    bucket_margin_summary = summarize_margins(contrast_rows, ["static_risk_bucket", "unnecessary_bid_margin"])

    write_csv(out_dir / "hand_decl_rows.csv", hand_decl_rows)
    write_csv(out_dir / "points_samples.csv", sample_rows)
    write_csv(out_dir / "bid_counterfactual_rows.csv", bid_rows)
    write_csv(out_dir / "margin_contrasts.csv", contrast_rows)
    write_csv(out_dir / "margin_summary.csv", margin_summary)
    write_csv(out_dir / "static_bucket_margin_summary.csv", bucket_margin_summary)
    write_json(out_dir / "generation_spec.json", make_generation_spec(args))

    negative = sum(1 for row in contrast_rows if float(row["delta_mark_swing_vs_min"]) < 0)
    zero = sum(1 for row in contrast_rows if float(row["delta_mark_swing_vs_min"]) == 0)
    positive = sum(1 for row in contrast_rows if float(row["delta_mark_swing_vs_min"]) > 0)
    row_counts = Counter(row["static_risk_bucket"] for row in hand_decl_rows)

    summary = {
        "schema_version": "w42.bid_only_enough_claim_tests.v1",
        "bead": BEAD_ID,
        "git_sha": git_sha(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "config": {
            "start_seed": args.start_seed,
            "hands": args.hands,
            "declarations_per_hand": args.declarations_per_hand,
            "samples": args.samples,
            "sim_seed": args.sim_seed,
            "device": args.device,
            "checkpoint": str(args.checkpoint or "default forge.bidding checkpoint"),
            "current_high_bids": list(CURRENT_HIGH_BIDS),
        },
        "counts": {
            "hand_decl_rows": len(hand_decl_rows),
            "bid_counterfactual_rows": len(bid_rows),
            "margin_contrast_rows": len(contrast_rows),
            "static_risk_bucket_hand_decl_counts": dict(sorted(row_counts.items())),
        },
        "effect_direction": {
            "negative_delta_mark_swing_rows": negative,
            "zero_delta_mark_swing_rows": zero,
            "positive_delta_mark_swing_rows": positive,
            "mean_delta_mark_swing_vs_min": round(mean([float(row["delta_mark_swing_vs_min"]) for row in contrast_rows]), 6),
            "mean_delta_p_make_vs_min": round(mean([float(row["delta_p_make_vs_min"]) for row in contrast_rows]), 6),
        },
        "scientific_status": {
            "empirically_testable_now": "partial",
            "what_this_tests": "For arbitrary generated hands, same hand/declaration/current-high counterfactuals show the cost of bidding above the minimum winning bid under the existing policy-simulation make labels.",
            "what_this_does_not_test": "No real auction policy, opponent response to bid size, partner bid signal, score-conditioned table style, or opening pass/bid decision is modeled.",
            "closeable_recommendation": "Close t42-0b4l.5 as a partial empirical probe plus executable generation spec, not as a central claim promotion.",
        },
        "artifacts": {
            "hand_decl_rows": str(out_dir / "hand_decl_rows.csv"),
            "points_samples": str(out_dir / "points_samples.csv"),
            "bid_counterfactual_rows": str(out_dir / "bid_counterfactual_rows.csv"),
            "margin_contrasts": str(out_dir / "margin_contrasts.csv"),
            "margin_summary": str(out_dir / "margin_summary.csv"),
            "static_bucket_margin_summary": str(out_dir / "static_bucket_margin_summary.csv"),
            "generation_spec": str(out_dir / "generation_spec.json"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def summarize_margins(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    out: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items(), key=lambda item: item[0]):
        deltas_p = [float(row["delta_p_make_vs_min"]) for row in group]
        deltas_s = [float(row["delta_mark_swing_vs_min"]) for row in group]
        record = {key: value for key, value in zip(keys, key_values)}
        record.update(
            {
                "n": len(group),
                "mean_delta_p_make_vs_min": round(mean(deltas_p), 6),
                "mean_delta_mark_swing_vs_min": round(mean(deltas_s), 6),
                "negative_delta_rate": round(sum(1 for x in deltas_s if x < 0) / len(deltas_s), 6),
                "zero_delta_rate": round(sum(1 for x in deltas_s if x == 0) / len(deltas_s), 6),
                "positive_delta_rate": round(sum(1 for x in deltas_s if x > 0) / len(deltas_s), 6),
            }
        )
        out.append(record)
    return out


def smoke_assertions(
    hand_decl_rows: list[dict[str, Any]],
    bid_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
) -> None:
    if not hand_decl_rows:
        raise AssertionError("No hand/declaration rows generated")
    if not bid_rows:
        raise AssertionError("No bid counterfactual rows generated")
    if not contrast_rows:
        raise AssertionError("No margin contrast rows generated")

    by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in bid_rows:
        by_group[(row["seed"], row["decl_id"], row["current_high_bid"])].append(row)
    for key, group in by_group.items():
        static_values = {
            (
                row["static_unique_exposed_points"],
                row["static_trump_count"],
                row["static_risk_bucket"],
            )
            for row in group
        }
        if len(static_values) != 1:
            raise AssertionError(f"Static features changed within bid counterfactual group {key}")
        ordered = sorted(group, key=lambda row: int(row["actual_bid"]))
        p_make = [float(row["p_make"]) for row in ordered]
        mark_swing = [float(row["mark_swing"]) for row in ordered]
        if any(later > earlier for earlier, later in zip(p_make, p_make[1:])):
            raise AssertionError(f"P(make) increased with higher bid in group {key}")
        if any(later > earlier for earlier, later in zip(mark_swing, mark_swing[1:])):
            raise AssertionError(f"Mark swing increased with higher bid in group {key}")

    if any(float(row["delta_mark_swing_vs_min"]) > 0 for row in contrast_rows):
        raise AssertionError("An unnecessary-margin row improved mark swing versus the minimum bid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--hands", type=int, default=12)
    parser.add_argument("--declarations-per-hand", type=int, default=2)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--sim-seed", type=int, default=420500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run_probe(parse_args())
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    print(json.dumps(summary["effect_direction"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
