#!/usr/bin/env python3
"""Paired same-hand Chapter 9 doubles-trump vs no-trump regime tests.

This phase-4 runner moves beyond the legacy decl-7/decl-9 mining by fixing the
bidder hand and simulation seed, then evaluating the same opponent worlds under
doubles-as-trump (decl 7) and standard no-trump (decl 9). Labels derived from
the model rollouts are offline targets; the hand-shape features are bidder-local
and do not inspect opponent tiles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.bidding.estimator import BID_THRESHOLDS, evaluate_bids
from forge.bidding.inference import PolicyModel
from forge.bidding.simulator import simulate_games
from forge.oracle.declarations import DECL_ID_TO_NAME, DOUBLES_TRUMP, NOTRUMP
from forge.oracle.tables import DOMINO_COUNT_POINTS, DOMINO_IS_DOUBLE, DOMINOES


OUT_DIR = ROOT / "w42" / "phase4_doubles_notrump_regime_tests"
BEAD_ID = "t42-br7n.5"
KEY_BIDS = (30, 31, 32, 35, 36, 42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-hands", type=int, default=192)
    parser.add_argument("--bucket-candidates", type=int, default=96)
    parser.add_argument("--selection-seed", type=int, default=20260503)
    parser.add_argument("--sim-seed", type=int, default=940000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_int(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if isinstance(v, int | float) and math.isfinite(float(v))]


def avg(values: Iterable[float]) -> float:
    nums = finite(values)
    return float(sum(nums) / len(nums)) if nums else float("nan")


def bootstrap_ci(values: Iterable[float], *, samples: int, seed: int) -> tuple[float, float, float]:
    nums = finite(values)
    center = avg(nums)
    if not nums:
        return center, float("nan"), float("nan")
    if len(nums) == 1 or samples <= 0:
        return center, center, center
    rng = random.Random(seed)
    draws: list[float] = []
    n = len(nums)
    for _ in range(samples):
        draws.append(sum(nums[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return center, float(lo), float(hi)


def domino_name(domino_id: int) -> str:
    high, low = DOMINOES[domino_id]
    return f"{high}-{low}"


def hand_label(hand: Iterable[int]) -> str:
    return ",".join(domino_name(d) for d in sorted(hand))


def double_id(pip: int) -> int:
    return DOMINOES.index((pip, pip))


def support_double_ids(domino_id: int, hand: frozenset[int]) -> set[int]:
    high, low = DOMINOES[domino_id]
    return {double_id(pip) for pip in {high, low} if double_id(pip) in hand}


def hand_features(hand_tuple: tuple[int, ...]) -> dict[str, Any]:
    hand = frozenset(hand_tuple)
    double_count = sum(int(DOMINO_IS_DOUBLE[d]) for d in hand_tuple)
    high_double_count = sum(int(DOMINO_IS_DOUBLE[d] and DOMINOES[d][0] >= 4) for d in hand_tuple)
    top_double_count = sum(int(DOMINO_IS_DOUBLE[d] and DOMINOES[d][0] >= 5) for d in hand_tuple)
    low_double_count = sum(int(DOMINO_IS_DOUBLE[d] and DOMINOES[d][0] <= 3) for d in hand_tuple)
    held_count_points = sum(int(DOMINO_COUNT_POINTS[d]) for d in hand_tuple)
    held_double_count_points = sum(int(DOMINO_COUNT_POINTS[d]) for d in hand_tuple if DOMINO_IS_DOUBLE[d])
    held_non_double_count_points = held_count_points - held_double_count_points

    support_tiles = 0
    support_count_points = 0
    support_double_count_points = 0
    for d in hand_tuple:
        if DOMINO_IS_DOUBLE[d]:
            continue
        supports = support_double_ids(d, hand)
        if supports:
            support_tiles += 1
            support_count_points += int(DOMINO_COUNT_POINTS[d])
            support_double_count_points += sum(int(DOMINO_COUNT_POINTS[s]) for s in supports)

    has_65 = int(DOMINOES.index((6, 5)) in hand)
    four_plus = double_count >= 4
    five_plus = double_count >= 5
    high_control = four_plus and top_double_count >= 2
    missing_top = four_plus and top_double_count < 2
    low_exposure = four_plus and low_double_count >= 2 and top_double_count < 2
    nt_support = support_tiles >= 3 or support_count_points >= 10
    regime_switch_proxy = four_plus and nt_support and (missing_top or double_id(6) not in hand)

    labels = []
    if four_plus:
        labels.append("four_plus_doubles")
    if five_plus:
        labels.append("five_plus_doubles")
    if high_control:
        labels.append("high_double_control")
    if missing_top:
        labels.append("missing_top_double")
    if low_exposure:
        labels.append("low_double_exposure")
    if nt_support:
        labels.append("no_trump_support")
    if regime_switch_proxy:
        labels.append("regime_switch_proxy")
    if has_65:
        labels.append("dual_suit_65")

    return {
        "hand": hand_label(hand_tuple),
        "hand_ids": "|".join(str(d) for d in hand_tuple),
        "double_count": double_count,
        "high_double_count": high_double_count,
        "top_double_count": top_double_count,
        "low_double_count": low_double_count,
        "missing_6_6": int(double_id(6) not in hand),
        "held_count_points": held_count_points,
        "held_double_count_points": held_double_count_points,
        "held_non_double_count_points": held_non_double_count_points,
        "support_non_double_tiles": support_tiles,
        "support_non_double_count_points": support_count_points,
        "support_double_count_points": support_double_count_points,
        "has_6_5": has_65,
        "four_plus_doubles": int(four_plus),
        "five_plus_doubles": int(five_plus),
        "high_double_control": int(high_control),
        "missing_top_double": int(missing_top),
        "low_double_exposure": int(low_exposure),
        "no_trump_support": int(nt_support),
        "regime_switch_proxy": int(regime_switch_proxy),
        "bucket_labels": "|".join(labels) if labels else "baseline",
    }


def candidate_buckets(features: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if int(features["four_plus_doubles"]):
        labels.append("four_plus_doubles")
    if int(features["five_plus_doubles"]):
        labels.append("five_plus_doubles")
    if int(features["high_double_control"]):
        labels.append("high_double_control")
    if int(features["low_double_exposure"]):
        labels.append("low_double_exposure")
    if int(features["no_trump_support"]):
        labels.append("no_trump_support")
    if int(features["regime_switch_proxy"]):
        labels.append("regime_switch_proxy")
    if int(features["has_6_5"]):
        labels.append("dual_suit_65")
    if not labels:
        labels.append("baseline")
    return labels


def select_candidate_hands(args: argparse.Namespace) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    target_buckets = (
        "regime_switch_proxy",
        "four_plus_doubles",
        "five_plus_doubles",
        "high_double_control",
        "low_double_exposure",
        "no_trump_support",
        "dual_suit_65",
        "baseline",
    )
    heaps: dict[str, list[tuple[int, tuple[int, ...]]]] = {bucket: [] for bucket in target_buckets}
    counts: Counter[str] = Counter()

    def keep(bucket: str, hand_tuple: tuple[int, ...]) -> None:
        score = stable_int(args.selection_seed, bucket, ",".join(map(str, hand_tuple)))
        rows = heaps[bucket]
        rows.append((score, hand_tuple))
        rows.sort(key=lambda item: item[0])
        if len(rows) > args.bucket_candidates:
            rows.pop()

    for hand_tuple in itertools.combinations(range(len(DOMINOES)), 7):
        features = hand_features(hand_tuple)
        for bucket in candidate_buckets(features):
            if bucket in heaps:
                counts[bucket] += 1
                keep(bucket, hand_tuple)

    selected: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(selected) < args.max_hands:
        moved = False
        for bucket in target_buckets:
            if len(selected) >= args.max_hands:
                break
            rows = heaps[bucket]
            if not rows:
                continue
            _score, hand_tuple = rows.pop(0)
            if hand_tuple in seen:
                continue
            seen.add(hand_tuple)
            selected.append(hand_tuple)
            moved = True
        if not moved:
            break

    meta = {
        "target_buckets": list(target_buckets),
        "exact_hand_space": math.comb(len(DOMINOES), 7),
        "bucket_population_counts": dict(counts),
        "selected_hands": len(selected),
        "bucket_candidate_cap": args.bucket_candidates,
        "selection_seed": args.selection_seed,
    }
    return selected, meta


def bid_result_map(points: list[int], decl_id: int) -> dict[int, dict[str, float]]:
    result = evaluate_bids(points, decl_id)
    return {
        bid.bid: {
            "p_make": float(bid.p_make),
            "mark_swing": float(bid.mark_swing),
            "ci_low": float(bid.ci_low),
            "ci_high": float(bid.ci_high),
        }
        for bid in result.bid_results
        if bid.bid in BID_THRESHOLDS
    }


def regime_row(
    *,
    hand_index: int,
    hand_tuple: tuple[int, ...],
    features: dict[str, Any],
    decl_id: int,
    points: list[int],
    sim_seed: int,
    source_bucket: str,
) -> dict[str, Any]:
    by_bid = bid_result_map(points, decl_id)
    mark_swings = {bid: by_bid[bid]["mark_swing"] for bid in BID_THRESHOLDS}
    profitable = [bid for bid in BID_THRESHOLDS if mark_swings[bid] >= 0.0]
    best_bid = max(BID_THRESHOLDS, key=lambda bid: (mark_swings[bid], -bid))
    row = {
        "hand_index": hand_index,
        "source_bucket": source_bucket,
        **features,
        "decl_id": decl_id,
        "decl_name": DECL_ID_TO_NAME.get(decl_id, str(decl_id)),
        "sim_seed": sim_seed,
        "samples": len(points),
        "mean_points": round(mean(points), 6),
        "min_points": min(points),
        "max_points": max(points),
        "best_bid_by_mark_swing": best_bid,
        "best_mark_swing": round(mark_swings[best_bid], 6),
        "max_profitable_bid": max(profitable) if profitable else 0,
    }
    for bid in KEY_BIDS:
        row[f"p_make_{bid}"] = round(by_bid[bid]["p_make"], 6)
        row[f"mark_swing_{bid}"] = round(by_bid[bid]["mark_swing"], 6)
        row[f"ci_low_{bid}"] = round(by_bid[bid]["ci_low"], 6)
        row[f"ci_high_{bid}"] = round(by_bid[bid]["ci_high"], 6)
    return row


def pair_row(
    hand_index: int,
    features: dict[str, Any],
    dt_row: dict[str, Any],
    nt_row: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "hand_index": hand_index,
        **features,
        "dt_mean_points": dt_row["mean_points"],
        "nt_mean_points": nt_row["mean_points"],
        "nt_minus_dt_mean_points": round(float(nt_row["mean_points"]) - float(dt_row["mean_points"]), 6),
        "dt_best_bid_by_mark_swing": dt_row["best_bid_by_mark_swing"],
        "nt_best_bid_by_mark_swing": nt_row["best_bid_by_mark_swing"],
        "dt_best_mark_swing": dt_row["best_mark_swing"],
        "nt_best_mark_swing": nt_row["best_mark_swing"],
        "nt_minus_dt_best_mark_swing": round(float(nt_row["best_mark_swing"]) - float(dt_row["best_mark_swing"]), 6),
        "dt_max_profitable_bid": dt_row["max_profitable_bid"],
        "nt_max_profitable_bid": nt_row["max_profitable_bid"],
        "nt_minus_dt_max_profitable_bid": int(nt_row["max_profitable_bid"]) - int(dt_row["max_profitable_bid"]),
    }
    for bid in KEY_BIDS:
        delta = float(nt_row[f"mark_swing_{bid}"]) - float(dt_row[f"mark_swing_{bid}"])
        row[f"dt_p_make_{bid}"] = dt_row[f"p_make_{bid}"]
        row[f"nt_p_make_{bid}"] = nt_row[f"p_make_{bid}"]
        row[f"nt_minus_dt_p_make_{bid}"] = round(float(nt_row[f"p_make_{bid}"]) - float(dt_row[f"p_make_{bid}"]), 6)
        row[f"dt_mark_swing_{bid}"] = dt_row[f"mark_swing_{bid}"]
        row[f"nt_mark_swing_{bid}"] = nt_row[f"mark_swing_{bid}"]
        row[f"nt_minus_dt_mark_swing_{bid}"] = round(delta, 6)
        row[f"preferred_decl_at_{bid}"] = "no-trump" if delta > 0 else ("doubles-trump" if delta < 0 else "tie")
    return row


def point_sample_rows(
    *,
    hand_index: int,
    hand: str,
    dt_points: list[int],
    nt_points: list[int],
    sim_seed: int,
) -> list[dict[str, Any]]:
    return [
        {
            "hand_index": hand_index,
            "hand": hand,
            "sample_index": sample_index,
            "sim_seed": sim_seed,
            "dt_points": dt,
            "nt_points": nt,
            "nt_minus_dt_points": nt - dt,
        }
        for sample_index, (dt, nt) in enumerate(zip(dt_points, nt_points, strict=True))
    ]


def summarize_bucket(
    bucket: str,
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    delta30 = [float(row["nt_minus_dt_mark_swing_30"]) for row in rows]
    delta_best = [float(row["nt_minus_dt_best_mark_swing"]) for row in rows]
    mean_delta30, delta30_lo, delta30_hi = bootstrap_ci(
        delta30, samples=bootstrap_samples, seed=stable_int(bootstrap_seed, bucket, "30")
    )
    mean_best, best_lo, best_hi = bootstrap_ci(
        delta_best, samples=bootstrap_samples, seed=stable_int(bootstrap_seed, bucket, "best")
    )
    return {
        "bucket": bucket,
        "hand_n": len(rows),
        "mean_nt_minus_dt_points": round(avg(float(row["nt_minus_dt_mean_points"]) for row in rows), 6),
        "mean_nt_minus_dt_mark_swing_30": round(mean_delta30, 6),
        "mean_nt_minus_dt_mark_swing_30_ci95_low": round(delta30_lo, 6),
        "mean_nt_minus_dt_mark_swing_30_ci95_high": round(delta30_hi, 6),
        "mean_nt_minus_dt_best_mark_swing": round(mean_best, 6),
        "mean_nt_minus_dt_best_mark_swing_ci95_low": round(best_lo, 6),
        "mean_nt_minus_dt_best_mark_swing_ci95_high": round(best_hi, 6),
        "nt_preferred_at_30_rate": round(avg(1.0 if row["preferred_decl_at_30"] == "no-trump" else 0.0 for row in rows), 6),
        "dt_preferred_at_30_rate": round(avg(1.0 if row["preferred_decl_at_30"] == "doubles-trump" else 0.0 for row in rows), 6),
        "nt_higher_max_profitable_bid_rate": round(
            avg(1.0 if int(row["nt_minus_dt_max_profitable_bid"]) > 0 else 0.0 for row in rows), 6
        ),
        "dt_higher_max_profitable_bid_rate": round(
            avg(1.0 if int(row["nt_minus_dt_max_profitable_bid"]) < 0 else 0.0 for row in rows), 6
        ),
        "mean_dt_max_profitable_bid": round(avg(float(row["dt_max_profitable_bid"]) for row in rows), 6),
        "mean_nt_max_profitable_bid": round(avg(float(row["nt_max_profitable_bid"]) for row in rows), 6),
    }


def build_summaries(pair_rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bucket_specs: list[tuple[str, Any]] = [
        ("all_paired_hands", lambda row: True),
        ("four_plus_doubles", lambda row: int(row["four_plus_doubles"]) == 1),
        ("five_plus_doubles", lambda row: int(row["five_plus_doubles"]) == 1),
        ("four_plus_missing_top_double", lambda row: int(row["missing_top_double"]) == 1),
        ("high_double_control", lambda row: int(row["high_double_control"]) == 1),
        ("low_double_exposure", lambda row: int(row["low_double_exposure"]) == 1),
        ("no_trump_support", lambda row: int(row["no_trump_support"]) == 1),
        ("no_trump_support_without_fourplus", lambda row: int(row["no_trump_support"]) == 1 and int(row["four_plus_doubles"]) == 0),
        ("regime_switch_proxy", lambda row: int(row["regime_switch_proxy"]) == 1),
        ("dual_suit_65", lambda row: int(row["has_6_5"]) == 1),
    ]
    summaries: list[dict[str, Any]] = []
    for bucket, predicate in bucket_specs:
        rows = [row for row in pair_rows if predicate(row)]
        if rows:
            summaries.append(
                summarize_bucket(
                    bucket,
                    rows,
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_seed=args.selection_seed,
                )
            )

    claim_rows = []
    interpretations = {
        "four_plus_doubles": "Tests whether four-plus doubles are sufficient for doubles-trump. Positive NT deltas weaken the simple count rule.",
        "five_plus_doubles": "Tests the stronger five-plus doubles surface.",
        "four_plus_missing_top_double": "Tests the book caveat that missing high doubles can make doubles-trump fragile.",
        "high_double_control": "Tests whether high/top double coverage rescues doubles-trump.",
        "low_double_exposure": "Tests low-double/count-dump danger surfaces.",
        "no_trump_support": "Tests no-trump support-double surfaces where doubles remain native suit tops.",
        "regime_switch_proxy": "Tests same-hand no-trump-over-doubles-trump candidates.",
        "dual_suit_65": "Tests the 6-5 dual-top protection slice under doubles-trump.",
    }
    by_bucket = {row["bucket"]: row for row in summaries}
    for bucket, text in interpretations.items():
        if bucket not in by_bucket:
            continue
        row = dict(by_bucket[bucket])
        row["claim_probe"] = bucket
        row["interpretation"] = text
        if float(row["mean_nt_minus_dt_best_mark_swing"]) > 0:
            row["direction"] = "leans_no_trump"
        elif float(row["mean_nt_minus_dt_best_mark_swing"]) < 0:
            row["direction"] = "leans_doubles_trump"
        else:
            row["direction"] = "flat"
        claim_rows.append(row)
    return summaries, claim_rows


def build_examples(pair_rows: list[dict[str, Any]], limit: int = 8) -> dict[str, list[dict[str, Any]]]:
    def top_rows(predicate: Any, key: Any, reverse: bool = True) -> list[dict[str, Any]]:
        selected = [row for row in pair_rows if predicate(row)]
        selected.sort(key=key, reverse=reverse)
        return selected[:limit]

    return {
        "largest_no_trump_advantage": top_rows(lambda _row: True, lambda row: float(row["nt_minus_dt_best_mark_swing"])),
        "largest_doubles_trump_advantage": top_rows(lambda _row: True, lambda row: float(row["nt_minus_dt_best_mark_swing"]), reverse=False),
        "regime_switch_proxy_no_trump": top_rows(
            lambda row: int(row["regime_switch_proxy"]) == 1,
            lambda row: float(row["nt_minus_dt_best_mark_swing"]),
        ),
        "high_double_control_doubles_trump": top_rows(
            lambda row: int(row["high_double_control"]) == 1,
            lambda row: float(row["nt_minus_dt_best_mark_swing"]),
            reverse=False,
        ),
        "low_double_exposure_no_trump": top_rows(
            lambda row: int(row["low_double_exposure"]) == 1,
            lambda row: float(row["nt_minus_dt_best_mark_swing"]),
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.max_hands = min(args.max_hands, 24)
        args.samples = min(args.samples, 16)
        args.bucket_candidates = min(args.bucket_candidates, 24)
        args.bootstrap_samples = min(args.bootstrap_samples, 100)

    selected_hands, selection_meta = select_candidate_hands(args)
    model = PolicyModel(args.checkpoint, device=args.device, compile_model=False)

    regime_rows: list[dict[str, Any]] = []
    pair_rows_out: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    selected_hand_rows: list[dict[str, Any]] = []

    for hand_index, hand_tuple in enumerate(selected_hands):
        features = hand_features(hand_tuple)
        source_bucket = candidate_buckets(features)[0]
        selected_hand_rows.append({"hand_index": hand_index, "source_bucket": source_bucket, **features})
        sim_seed = args.sim_seed + hand_index * 1009
        dt_tensor = simulate_games(
            model=model,
            bidder_hand=list(hand_tuple),
            decl_id=DOUBLES_TRUMP,
            n_games=args.samples,
            seed=sim_seed,
            greedy=args.greedy,
        )
        nt_tensor = simulate_games(
            model=model,
            bidder_hand=list(hand_tuple),
            decl_id=NOTRUMP,
            n_games=args.samples,
            seed=sim_seed,
            greedy=args.greedy,
        )
        dt_points = [int(x) for x in dt_tensor.cpu().tolist()]
        nt_points = [int(x) for x in nt_tensor.cpu().tolist()]
        dt_row = regime_row(
            hand_index=hand_index,
            hand_tuple=hand_tuple,
            features=features,
            decl_id=DOUBLES_TRUMP,
            points=dt_points,
            sim_seed=sim_seed,
            source_bucket=source_bucket,
        )
        nt_row = regime_row(
            hand_index=hand_index,
            hand_tuple=hand_tuple,
            features=features,
            decl_id=NOTRUMP,
            points=nt_points,
            sim_seed=sim_seed,
            source_bucket=source_bucket,
        )
        regime_rows.extend([dt_row, nt_row])
        pair_rows_out.append(pair_row(hand_index, features, dt_row, nt_row))
        sample_rows.extend(point_sample_rows(hand_index=hand_index, hand=features["hand"], dt_points=dt_points, nt_points=nt_points, sim_seed=sim_seed))

    bucket_summary, claim_contrasts = build_summaries(pair_rows_out, args)
    examples = build_examples(pair_rows_out)

    write_csv(out_dir / "selected_hand_rows.csv", selected_hand_rows)
    write_csv(out_dir / "regime_contract_rows.csv", regime_rows)
    write_csv(out_dir / "paired_regime_rows.csv", pair_rows_out)
    write_csv(out_dir / "point_sample_pairs.csv", sample_rows)
    write_csv(out_dir / "bucket_summary.csv", bucket_summary)
    write_csv(out_dir / "claim_contrasts.csv", claim_contrasts)
    write_json(out_dir / "examples.json", examples)

    by_claim = {row["claim_probe"]: row for row in claim_contrasts}
    summary = {
        "schema_version": "w42.phase4_doubles_notrump_regime_tests.v0",
        "bead_id": BEAD_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_sha(),
        "command": "python " + " ".join(sys.argv),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "method": {
            "evidence_mode": "paired same-hand simulation under doubles-trump and standard no-trump",
            "same_hand_conditioning": "P0 hand fixed for both regimes.",
            "same_opponent_world_conditioning": "simulate_games receives the same seed for decl 7 and decl 9; deal_random_hands therefore emits paired opponent deals.",
            "policy": "forge.bidding PolicyModel greedy play by default",
            "offline_label_boundary": "Final points, p_make, and mark_swing are offline rollout labels; row features are bidder-hand shape features only.",
            "legacy_baseline": "w42/doubles_no_trump_legacy_mining was within-regime only and did not test same-hand declaration choice.",
        },
        "config": {
            "samples_per_hand_per_regime": args.samples,
            "max_hands": args.max_hands,
            "bucket_candidates": args.bucket_candidates,
            "selection_seed": args.selection_seed,
            "sim_seed": args.sim_seed,
            "device": args.device,
            "greedy": args.greedy,
            "smoke": bool(args.smoke),
        },
        "selection": selection_meta,
        "coverage": {
            "selected_hands": len(selected_hands),
            "regime_contract_rows": len(regime_rows),
            "paired_regime_rows": len(pair_rows_out),
            "point_sample_pairs": len(sample_rows),
            "claim_contrasts": len(claim_contrasts),
        },
        "headline_claims": {
            claim: {
                "hand_n": row["hand_n"],
                "direction": row["direction"],
                "mean_nt_minus_dt_best_mark_swing": row["mean_nt_minus_dt_best_mark_swing"],
                "mean_nt_minus_dt_mark_swing_30": row["mean_nt_minus_dt_mark_swing_30"],
                "nt_preferred_at_30_rate": row["nt_preferred_at_30_rate"],
                "dt_preferred_at_30_rate": row["dt_preferred_at_30_rate"],
            }
            for claim, row in by_claim.items()
        },
        "scientific_status": {
            "claim_ledger_impact": "No central claim ledger edit by Worker E; evidence is local to this artifact.",
            "caveats": [
                "This is play-policy simulation, not exhaustive minimax/oracle proof.",
                "P0 is always treated as bidder; auction pressure and partner/opponent bidding are outside this bead.",
                "The policy can prefer or mishandle a regime for model reasons, so strong deltas are evidence for generated-regime behavior, not final folk-wisdom truth by themselves.",
            ],
        },
        "artifacts": {
            "selected_hand_rows": str((out_dir / "selected_hand_rows.csv").relative_to(ROOT)),
            "regime_contract_rows": str((out_dir / "regime_contract_rows.csv").relative_to(ROOT)),
            "paired_regime_rows": str((out_dir / "paired_regime_rows.csv").relative_to(ROOT)),
            "point_sample_pairs": str((out_dir / "point_sample_pairs.csv").relative_to(ROOT)),
            "bucket_summary": str((out_dir / "bucket_summary.csv").relative_to(ROOT)),
            "claim_contrasts": str((out_dir / "claim_contrasts.csv").relative_to(ROOT)),
            "examples": str((out_dir / "examples.json").relative_to(ROOT)),
            "summary": str((out_dir / "summary.json").relative_to(ROOT)),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
