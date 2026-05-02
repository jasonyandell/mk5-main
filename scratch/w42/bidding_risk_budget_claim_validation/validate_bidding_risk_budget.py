#!/usr/bin/env python3
"""Exhaustive static checks for Winning 42 bidding risk-budget claims.

This report-only script enumerates every double-six seven-domino hand and every
pip-trump candidate. It validates detector arithmetic and population prevalence
for static bidding-risk claims; it intentionally does not estimate make/set rate
or oracle regret.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle.tables import DOMINO_COUNT_POINTS, DOMINOES, DOMINO_IS_DOUBLE, domino_contains_pip


OUT_DIR = Path("scratch/w42/bidding_risk_budget_claim_validation")
COUNT_TILE_IDS = tuple(i for i, points in enumerate(DOMINO_COUNT_POINTS) if points > 0)
PIP_TRUMPS = tuple(range(7))
TOTAL_HANDS = math.comb(len(DOMINOES), 7)


def domino_name(domino_id: int) -> str:
    high, low = DOMINOES[domino_id]
    return f"{high}-{low}"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def is_trump(domino_id: int, pip: int) -> bool:
    return domino_contains_pip(domino_id, pip)


def count_exposed_by_side(hand: frozenset[int], trump_pip: int, side_pip: int) -> set[int]:
    exposed: set[int] = set()
    for count_id in COUNT_TILE_IDS:
        if count_id in hand:
            continue
        if is_trump(count_id, trump_pip):
            continue
        if domino_contains_pip(count_id, side_pip):
            exposed.add(count_id)
    return exposed


def hand_eval(hand_tuple: tuple[int, ...], trump_pip: int) -> dict[str, Any]:
    hand = frozenset(hand_tuple)
    trump_tiles = [d for d in hand_tuple if is_trump(d, trump_pip)]
    off_tiles = [
        d
        for d in hand_tuple
        if not is_trump(d, trump_pip) and not DOMINO_IS_DOUBLE[d]
    ]

    unique_exposed: set[int] = set()
    naive_exposed_points = 0
    protected_side_points = 0
    unprotected_side_points = 0
    four_five_off = False
    max_single_off_points = 0
    side_rows = 0

    for off_id in off_tiles:
        high, low = DOMINOES[off_id]
        off_unique: set[int] = set()
        if set((high, low)) == {4, 5}:
            four_five_off = True
        for side_pip in (high, low):
            side_rows += 1
            side_exposed = count_exposed_by_side(hand, trump_pip, side_pip)
            side_points = sum(DOMINO_COUNT_POINTS[d] for d in side_exposed)
            naive_exposed_points += side_points
            off_unique.update(side_exposed)
            if side_pip * (side_pip + 3) // 2 in hand:
                protected_side_points += side_points
            else:
                unprotected_side_points += side_points
        unique_exposed.update(off_unique)
        max_single_off_points = max(max_single_off_points, sum(DOMINO_COUNT_POINTS[d] for d in off_unique))

    unique_exposed_points = sum(DOMINO_COUNT_POINTS[d] for d in unique_exposed)
    held_count_points = sum(DOMINO_COUNT_POINTS[d] for d in hand_tuple)
    held_count_tiles = sum(1 for d in hand_tuple if DOMINO_COUNT_POINTS[d] > 0)
    double_count = sum(1 for d in hand_tuple if DOMINO_IS_DOUBLE[d])

    return {
        "trump_count": len(trump_tiles),
        "off_count": len(off_tiles),
        "held_count_points": held_count_points,
        "held_count_tiles": held_count_tiles,
        "double_count": double_count,
        "unique_exposed_points": unique_exposed_points,
        "naive_exposed_points": naive_exposed_points,
        "duplicate_overcount_points": naive_exposed_points - unique_exposed_points,
        "protected_side_points": protected_side_points,
        "unprotected_side_points": unprotected_side_points,
        "side_rows": side_rows,
        "four_five_off": four_five_off,
        "max_single_off_points": max_single_off_points,
        "bid_ceiling_proxy": 42 - unique_exposed_points,
        "strong_trump_bad_risk_trap": len(trump_tiles) >= 4 and unique_exposed_points > 12,
        "minimum_shape_low_risk": len(trump_tiles) >= 3 and unique_exposed_points <= 12,
        "duplicate_exposure": naive_exposed_points > unique_exposed_points,
    }


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def write_counter_csv(path: Path, key_name: str, rows: list[tuple[Any, int]], denom: int) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[key_name, "n", "pct"], lineterminator="\n")
        writer.writeheader()
        for key, n in rows:
            writer.writerow({key_name: key, "n": n, "pct": round(pct(n, denom), 6)})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_total = 0
    candidate_metrics: Counter[str] = Counter()
    best_by_hand_metrics: Counter[str] = Counter()
    trump_count_dist: Counter[int] = Counter()
    risk_bucket_dist: Counter[str] = Counter()
    duplicate_overcount_dist: Counter[int] = Counter()
    bid_ceiling_dist: Counter[int] = Counter()
    ceiling_natural_bucket_dist: Counter[str] = Counter()
    strong_trump_risk_points: Counter[int] = Counter()
    best_candidate_risk_points: Counter[int] = Counter()
    best_candidate_trump_count: Counter[int] = Counter()
    held_count_points_by_trump_count: dict[int, list[int]] = defaultdict(list)
    side_point_sums = Counter()

    examples: dict[str, list[dict[str, Any]]] = {
        "duplicate_exposure": [],
        "strong_trump_bad_risk_trap": [],
        "minimum_shape_low_risk": [],
        "four_five_off": [],
    }

    def maybe_example(kind: str, hand_tuple: tuple[int, ...], trump_pip: int, ev: dict[str, Any]) -> None:
        if len(examples[kind]) >= 8:
            return
        row = {
            "hand": ",".join(domino_name(d) for d in hand_tuple),
            "trump_pip": trump_pip,
            "trump_count": ev["trump_count"],
            "unique_exposed_points": ev["unique_exposed_points"],
            "naive_exposed_points": ev["naive_exposed_points"],
            "duplicate_overcount_points": ev["duplicate_overcount_points"],
            "bid_ceiling_proxy": ev["bid_ceiling_proxy"],
        }
        examples[kind].append(row)

    for hand_tuple in combinations(range(len(DOMINOES)), 7):
        best = None
        best_key = None
        hand_has_shape_low_risk = False
        hand_has_strong_trump_bad_risk = False
        hand_has_duplicate = False
        hand_has_four_five = False

        for trump_pip in PIP_TRUMPS:
            ev = hand_eval(hand_tuple, trump_pip)
            candidate_total += 1
            trump_count_dist[ev["trump_count"]] += 1
            duplicate_overcount_dist[ev["duplicate_overcount_points"]] += 1
            bid_ceiling_dist[ev["bid_ceiling_proxy"]] += 1
            best_key = (
                -ev["unique_exposed_points"],
                ev["trump_count"],
                ev["held_count_points"],
            )
            if best is None or best_key > best[0]:
                best = (best_key, trump_pip, ev)

            if ev["unique_exposed_points"] <= 12:
                risk_bucket_dist["risk_le_12"] += 1
            elif ev["unique_exposed_points"] <= 20:
                risk_bucket_dist["risk_13_20"] += 1
            else:
                risk_bucket_dist["risk_gt_20"] += 1

            if ev["bid_ceiling_proxy"] in (30, 31, 35, 36):
                ceiling_natural_bucket_dist["natural_30_31_35_36"] += 1
            elif 32 <= ev["bid_ceiling_proxy"] <= 34:
                ceiling_natural_bucket_dist["odd_32_34"] += 1
            else:
                ceiling_natural_bucket_dist["other"] += 1

            for metric in ("minimum_shape_low_risk", "duplicate_exposure", "strong_trump_bad_risk_trap", "four_five_off"):
                if ev[metric]:
                    candidate_metrics[metric] += 1
                    maybe_example(metric, hand_tuple, trump_pip, ev)

            if ev["minimum_shape_low_risk"]:
                hand_has_shape_low_risk = True
            if ev["strong_trump_bad_risk_trap"]:
                hand_has_strong_trump_bad_risk = True
                strong_trump_risk_points[ev["unique_exposed_points"]] += 1
            if ev["duplicate_exposure"]:
                hand_has_duplicate = True
            if ev["four_five_off"]:
                hand_has_four_five = True
            side_point_sums["protected"] += ev["protected_side_points"]
            side_point_sums["unprotected"] += ev["unprotected_side_points"]
            side_point_sums["side_rows"] += ev["side_rows"]
            held_count_points_by_trump_count[ev["trump_count"]].append(ev["held_count_points"])

        assert best is not None
        best_ev = best[2]
        best_candidate_risk_points[best_ev["unique_exposed_points"]] += 1
        best_candidate_trump_count[best_ev["trump_count"]] += 1
        if best_ev["unique_exposed_points"] <= 12 and best_ev["trump_count"] >= 3:
            best_by_hand_metrics["best_minimum_shape_low_risk"] += 1
        if best_ev["trump_count"] >= 4 and best_ev["unique_exposed_points"] > 12:
            best_by_hand_metrics["best_strong_trump_bad_risk"] += 1
        if hand_has_shape_low_risk:
            best_by_hand_metrics["any_minimum_shape_low_risk"] += 1
        if hand_has_strong_trump_bad_risk:
            best_by_hand_metrics["any_strong_trump_bad_risk"] += 1
        if hand_has_duplicate:
            best_by_hand_metrics["any_duplicate_exposure"] += 1
        if hand_has_four_five:
            best_by_hand_metrics["any_four_five_off"] += 1

    summary = {
        "schema_version": "w42.bidding_risk_budget_claim_validation.v1",
        "bead": "t42-csw6.17",
        "git_sha": git_sha(),
        "random_seeds": "not applicable",
        "statistical_method": "exhaustive enumeration of all C(28,7) hands times seven pip-trump candidates; exact population proportions, no sampling CI",
        "hands": TOTAL_HANDS,
        "candidate_evaluations": candidate_total,
        "count_tiles": {domino_name(d): DOMINO_COUNT_POINTS[d] for d in COUNT_TILE_IDS},
        "candidate_metrics": {
            key: {"n": value, "pct": round(pct(value, candidate_total), 6)}
            for key, value in sorted(candidate_metrics.items())
        },
        "hand_metrics": {
            key: {"n": value, "pct": round(pct(value, TOTAL_HANDS), 6)}
            for key, value in sorted(best_by_hand_metrics.items())
        },
        "natural_bid_bucket_proxy": {
            key: {"n": value, "pct": round(pct(value, candidate_total), 6)}
            for key, value in sorted(ceiling_natural_bucket_dist.items())
        },
        "side_specific_exposure_points": {
            "protected": side_point_sums["protected"],
            "unprotected": side_point_sums["unprotected"],
            "protected_pct": round(pct(side_point_sums["protected"], side_point_sums["protected"] + side_point_sums["unprotected"]), 6),
            "side_rows": side_point_sums["side_rows"],
        },
        "examples": examples,
        "caveat": "Static risk-budget proxies use own hand plus candidate pip trump only; no auction history, partner bid signal, hidden-world make/set result, or forge E[Q] rollout is used.",
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    write_counter_csv(
        OUT_DIR / "candidate_trump_count_distribution.csv",
        "trump_count",
        sorted(trump_count_dist.items()),
        candidate_total,
    )
    write_counter_csv(
        OUT_DIR / "candidate_risk_bucket_distribution.csv",
        "risk_bucket",
        sorted(risk_bucket_dist.items()),
        candidate_total,
    )
    write_counter_csv(
        OUT_DIR / "duplicate_overcount_points_distribution.csv",
        "duplicate_overcount_points",
        sorted(duplicate_overcount_dist.items()),
        candidate_total,
    )
    write_counter_csv(
        OUT_DIR / "best_candidate_risk_points_distribution.csv",
        "unique_exposed_points",
        sorted(best_candidate_risk_points.items()),
        TOTAL_HANDS,
    )
    write_counter_csv(
        OUT_DIR / "best_candidate_trump_count_distribution.csv",
        "trump_count",
        sorted(best_candidate_trump_count.items()),
        TOTAL_HANDS,
    )
    write_counter_csv(
        OUT_DIR / "bid_ceiling_proxy_distribution.csv",
        "bid_ceiling_proxy",
        sorted(bid_ceiling_dist.items()),
        candidate_total,
    )
    write_counter_csv(
        OUT_DIR / "strong_trump_bad_risk_points_distribution.csv",
        "unique_exposed_points",
        sorted(strong_trump_risk_points.items()),
        max(candidate_metrics["strong_trump_bad_risk_trap"], 1),
    )

    with (OUT_DIR / "held_count_points_by_trump_count.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trump_count", "n", "mean_held_count_points"],
            lineterminator="\n",
        )
        writer.writeheader()
        for trump_count, values in sorted(held_count_points_by_trump_count.items()):
            writer.writerow(
                {
                    "trump_count": trump_count,
                    "n": len(values),
                    "mean_held_count_points": round(sum(values) / len(values), 6),
                }
            )

    with (OUT_DIR / "examples.json").open("w") as f:
        json.dump(examples, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
