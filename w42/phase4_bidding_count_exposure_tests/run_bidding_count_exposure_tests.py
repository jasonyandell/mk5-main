#!/usr/bin/env python3
"""Phase-4 evidence pack for W42 bidding count/exposure claims.

This script does not move central ledger status. It joins existing generated
auction/contract artifacts with exact hand-shape detectors so the seven
`needs_new_bidding_count_exposure_bead` claims have claim-specific evidence
tables and explicit caveats.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle.tables import DOMINO_COUNT_POINTS, DOMINO_IS_DOUBLE, DOMINOES, domino_contains_pip
from w42.bidding_risk_budget_claim_validation.validate_bidding_risk_budget import hand_eval


OUT_DIR = ROOT / "w42" / "phase4_bidding_count_exposure_tests"
BEAD_ID = "t42-br7n.7"
CLAIM_IDS = [
    "ch16-partner-two-plus-doubles-prior",
    "ch02-three-plus-trumps-good-start",
    "ch02-risk-budget-threshold",
    "ch02-strong-trump-bad-risk-trap",
    "ch02-four-five-off-danger",
    "ch12-natural-bid-bucket-anomaly",
    "ch02-double-side-protection",
]

SOURCE_FILES = {
    "completion_board": "w42/phase4_claim_completion_board/completion_board.csv",
    "ledger": "w42/statistics_claims_ledger/claims.csv",
    "auction_contract_rows": "w42/auction_bid_discipline_claim_tests/contract_rows.csv",
    "auction_bid_action_rows": "w42/auction_bid_discipline_claim_tests/bid_action_rows.csv",
    "auction_context_decision_rows": "w42/auction_bid_discipline_claim_tests/context_decision_rows.csv",
    "auction_natural_bucket_summary": "w42/auction_bid_discipline_claim_tests/natural_bucket_summary.csv",
    "auction_risk_budget_summary": "w42/auction_bid_discipline_claim_tests/risk_budget_summary.csv",
    "bid_only_hand_decl_rows": "w42/bid_only_enough_claim_tests/hand_decl_rows.csv",
    "static_risk_summary": "w42/bidding_risk_budget_claim_validation/summary.json",
    "static_double_count_priors": "w42/odds_ruleset_claim_validation/double_count_priors.csv",
}

COUNT_TILE_IDS = tuple(i for i, points in enumerate(DOMINO_COUNT_POINTS) if points > 0)
DOUBLE_ID_BY_PIP = {high: idx for idx, (high, low) in enumerate(DOMINOES) if high == low}


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def read_csv(rel_path: str) -> list[dict[str, str]]:
    path = ROOT / rel_path
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    return default if value in ("", None) else int(value)


def to_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value in ("", None) else float(value)


def pct(n: int | float, d: int | float) -> float:
    return round(100.0 * n / d, 6) if d else 0.0


def mean_or_blank(values: list[float]) -> float | str:
    return round(mean(values), 6) if values else ""


def parse_hand(hand_label: str) -> tuple[int, ...]:
    by_pair = {pair: idx for idx, pair in enumerate(DOMINOES)}
    ids: list[int] = []
    for token in hand_label.split(","):
        high_s, low_s = token.split("-")
        high = int(high_s)
        low = int(low_s)
        if high < low:
            high, low = low, high
        ids.append(by_pair[(high, low)])
    return tuple(sorted(ids))


def domino_name(domino_id: int) -> str:
    high, low = DOMINOES[domino_id]
    return f"{high}-{low}"


def count_exposed_by_side(hand: frozenset[int], trump_pip: int, side_pip: int) -> set[int]:
    exposed: set[int] = set()
    for count_id in COUNT_TILE_IDS:
        if count_id in hand:
            continue
        if domino_contains_pip(count_id, trump_pip):
            continue
        if domino_contains_pip(count_id, side_pip):
            exposed.add(count_id)
    return exposed


def side_exposure_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    hand_ids = parse_hand(str(row["hand"]))
    hand_set = frozenset(hand_ids)
    trump_pip = int(row["decl_id"])
    rows: list[dict[str, Any]] = []
    for off_id in hand_ids:
        if domino_contains_pip(off_id, trump_pip) or DOMINO_IS_DOUBLE[off_id]:
            continue
        high, low = DOMINOES[off_id]
        for side_pip in (high, low):
            exposed_ids = count_exposed_by_side(hand_set, trump_pip, side_pip)
            exposed_points = sum(DOMINO_COUNT_POINTS[d] for d in exposed_ids)
            rows.append(
                {
                    "seed": row["seed"],
                    "seat": row["seat"],
                    "decl_id": row["decl_id"],
                    "decl_name": row["decl_name"],
                    "hand": row["hand"],
                    "off_domino": domino_name(off_id),
                    "side_pip": side_pip,
                    "side_double_held": int(DOUBLE_ID_BY_PIP[side_pip] in hand_set),
                    "exposed_count_points": exposed_points,
                    "exposed_count_tiles": "|".join(domino_name(d) for d in sorted(exposed_ids)),
                    "p_make_30": row["p_make_30"],
                    "mark_swing_30": row["mark_swing_30"],
                    "static_risk_bucket": row["static_risk_bucket"],
                    "static_trump_count": row["static_trump_count"],
                    "static_unique_exposed_points": row["static_unique_exposed_points"],
                }
            )
    return rows


def enrich_contract_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        hand_ids = parse_hand(row["hand"])
        ev = hand_eval(hand_ids, int(row["decl_id"]))
        enriched: dict[str, Any] = dict(row)
        enriched.update(
            {
                "seed": int(row["seed"]),
                "seat": int(row["seat"]),
                "decl_id": int(row["decl_id"]),
                "n_samples": int(row["n_samples"]),
                "static_unique_exposed_points": int(row["static_unique_exposed_points"]),
                "static_trump_count": int(row["static_trump_count"]),
                "static_held_count_points": int(row["static_held_count_points"]),
                "static_off_count": int(row["static_off_count"]),
                "static_strong_trump_bad_risk_trap": int(row["static_strong_trump_bad_risk_trap"]),
                "static_four_five_off": int(row["static_four_five_off"]),
                "natural_bucket_candidate": int(row["natural_bucket_candidate"]),
                "p_make_30": float(row["p_make_30"]),
                "mark_swing_30": float(row["mark_swing_30"]),
                "mean_points": float(row["mean_points"]),
                "max_profitable_bid": int(row["max_profitable_bid"]),
                "best_mark_swing": float(row["best_mark_swing"]),
                "double_count": sum(1 for d in hand_ids if DOMINO_IS_DOUBLE[d]),
                "side_protected_points": int(ev["protected_side_points"]),
                "side_unprotected_points": int(ev["unprotected_side_points"]),
                "side_protected_point_share": round(
                    ev["protected_side_points"] / (ev["protected_side_points"] + ev["unprotected_side_points"]),
                    6,
                )
                if ev["protected_side_points"] + ev["unprotected_side_points"]
                else "",
            }
        )
        out.append(enriched)
    return out


def summarize_contract_group(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    total = len(rows)
    for key_values, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = {key: value for key, value in zip(keys, key_values)}
        n = len(group)
        record.update(
            {
                "n": n,
                "pct_rows": pct(n, total),
                "mean_p_make_30": mean_or_blank([float(r["p_make_30"]) for r in group]),
                "mean_mark_swing_30": mean_or_blank([float(r["mark_swing_30"]) for r in group]),
                "mean_mean_points": mean_or_blank([float(r["mean_points"]) for r in group]),
                "mean_max_profitable_bid": mean_or_blank([float(r["max_profitable_bid"]) for r in group]),
                "profitable_at_30_rate_pct": pct(sum(1 for r in group if float(r["mark_swing_30"]) >= 0), n),
                "natural_bucket_candidate_rate_pct": pct(sum(1 for r in group if int(r["natural_bucket_candidate"]) == 1), n),
                "mean_static_unique_exposed_points": mean_or_blank([float(r["static_unique_exposed_points"]) for r in group]),
            }
        )
        out.append(record)
    return out


def add_partner_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hand_by_seed_seat: dict[tuple[int, int], tuple[int, ...]] = {}
    for row in rows:
        key = (int(row["seed"]), int(row["seat"]))
        hand_by_seed_seat.setdefault(key, parse_hand(str(row["hand"])))

    out: list[dict[str, Any]] = []
    for row in rows:
        partner_seat = (int(row["seat"]) + 2) % 4
        partner_hand = hand_by_seed_seat[(int(row["seed"]), partner_seat)]
        partner_double_count = sum(1 for d in partner_hand if DOMINO_IS_DOUBLE[d])
        enriched = dict(row)
        enriched.update(
            {
                "partner_seat": partner_seat,
                "partner_double_count": partner_double_count,
                "partner_two_plus_doubles": int(partner_double_count >= 2),
                "bidder_off_bucket": "off_0_2" if int(row["static_off_count"]) <= 2 else "off_3_plus",
                "bidder_exposure_bucket": "risk_le_12"
                if int(row["static_unique_exposed_points"]) <= 12
                else "risk_gt_12",
            }
        )
        out.append(enriched)
    return out


def summarize_partner_prior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_contract_group(rows, ["bidder_off_bucket", "bidder_exposure_bucket", "partner_two_plus_doubles"])


def summarize_natural_bucket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = summarize_contract_group(rows, ["max_profitable_bid_bucket"])
    for row in grouped:
        bid_bucket = str(row["max_profitable_bid_bucket"])
        row["is_book_natural_bucket_family"] = int(bid_bucket in {"natural_30_31", "natural_35_36"})
    return grouped


def summarize_side_protection(side_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exposed = [row for row in side_rows if int(row["exposed_count_points"]) > 0]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in exposed:
        grouped[int(row["side_double_held"])].append(row)
    out: list[dict[str, Any]] = []
    total_points = sum(int(row["exposed_count_points"]) for row in exposed)
    total_rows = len(exposed)
    for protected, group in sorted(grouped.items()):
        points = sum(int(row["exposed_count_points"]) for row in group)
        out.append(
            {
                "side_double_held": protected,
                "n_side_exposure_rows": len(group),
                "pct_side_exposure_rows": pct(len(group), total_rows),
                "total_exposed_count_points": points,
                "pct_exposed_count_points": pct(points, total_points),
                "mean_exposed_count_points": mean_or_blank([float(row["exposed_count_points"]) for row in group]),
                "mean_p_make_30": mean_or_blank([float(row["p_make_30"]) for row in group]),
                "mean_mark_swing_30": mean_or_blank([float(row["mark_swing_30"]) for row in group]),
            }
        )
    return out


def static_partner_two_plus_prior() -> dict[str, Any]:
    rows = read_csv(SOURCE_FILES["static_double_count_priors"])
    total = 0
    two_plus = 0
    for row in rows:
        bucket = row["bucket"].split()[0]
        count = int(row["count"])
        total += count
        if int(bucket) >= 2:
            two_plus += count
    return {"n_two_plus_doubles": two_plus, "total_hands": total, "pct_two_plus_doubles": pct(two_plus, total)}


def claim_summary(
    *,
    contract_rows: list[dict[str, Any]],
    partner_rows: list[dict[str, Any]],
    side_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_trump_bucket = {
        "lt3": [r for r in contract_rows if int(r["static_trump_count"]) < 3],
        "ge3": [r for r in contract_rows if int(r["static_trump_count"]) >= 3],
        "ge4_high_risk": [
            r
            for r in contract_rows
            if int(r["static_trump_count"]) >= 4 and int(r["static_unique_exposed_points"]) > 12
        ],
    }
    risk_le_12 = [r for r in contract_rows if int(r["static_unique_exposed_points"]) <= 12]
    risk_gt_12 = [r for r in contract_rows if int(r["static_unique_exposed_points"]) > 12]
    trap_rows = [r for r in contract_rows if int(r["static_strong_trump_bad_risk_trap"]) == 1]
    four_five_rows = [r for r in contract_rows if int(r["static_four_five_off"]) == 1]
    natural_rows = [r for r in contract_rows if str(r["max_profitable_bid_bucket"]) in {"natural_30_31", "natural_35_36"}]
    partner_two_plus = [r for r in partner_rows if int(r["partner_two_plus_doubles"]) == 1]
    exposed_side_rows = [r for r in side_rows if int(r["exposed_count_points"]) > 0]
    protected_side_points = sum(int(r["exposed_count_points"]) for r in exposed_side_rows if int(r["side_double_held"]) == 1)
    all_side_points = sum(int(r["exposed_count_points"]) for r in exposed_side_rows)

    static_prior = static_partner_two_plus_prior()

    def avg(rows: list[dict[str, Any]], field: str) -> float | str:
        return mean_or_blank([float(r[field]) for r in rows])

    return [
        {
            "claim_id": "ch16-partner-two-plus-doubles-prior",
            "status_recommendation": "context_limited_static_prior_supported",
            "conclusion": "The exact unconditional two-plus-double prior is sizable, and the generated contract corpus preserves a similar but hand-conditioned partner-help rate. This still does not prove partner bidding intelligence.",
            "n_contract_rows": len(partner_rows),
            "headline_metric": "static_pct_two_plus_doubles",
            "headline_value": static_prior["pct_two_plus_doubles"],
            "secondary_metric": "generated_contract_partner_two_plus_pct",
            "secondary_value": pct(len(partner_two_plus), len(partner_rows)),
            "mean_p_make_30_when_partner_two_plus": avg(partner_two_plus, "p_make_30"),
            "mean_p_make_30_all": avg(partner_rows, "p_make_30"),
            "primary_table": "partner_two_plus_doubles_prior_summary.csv",
            "caveat": "Partner rows are hidden-hand priors joined to generated contract labels, not observed partner auctions or play choices.",
        },
        {
            "claim_id": "ch02-three-plus-trumps-good-start",
            "status_recommendation": "keep_context_limited_or_underpowered",
            "conclusion": "Three-plus trumps improve the generated contract averages, but the slice is still mixed with exposure and declaration-selection effects; it is not sufficient by itself.",
            "n_contract_rows": len(contract_rows),
            "headline_metric": "mean_p_make_30_ge3_minus_lt3",
            "headline_value": round(float(avg(by_trump_bucket["ge3"], "p_make_30")) - float(avg(by_trump_bucket["lt3"], "p_make_30")), 6),
            "secondary_metric": "mean_mark_swing_30_ge3",
            "secondary_value": avg(by_trump_bucket["ge3"], "mark_swing_30"),
            "primary_table": "trump_count_outcome_summary.csv",
            "caveat": "Generated make labels are fixed-contract simulations, not full opening-bid/pass policy rollouts.",
        },
        {
            "claim_id": "ch02-risk-budget-threshold",
            "status_recommendation": "context_limited_threshold_support",
            "conclusion": "Risk <=12 has materially better generated make/profit averages than risk >12. The threshold is directional evidence, not a standalone bid rule.",
            "n_contract_rows": len(contract_rows),
            "headline_metric": "mean_p_make_30_le12_minus_gt12",
            "headline_value": round(float(avg(risk_le_12, "p_make_30")) - float(avg(risk_gt_12, "p_make_30")), 6),
            "secondary_metric": "mean_mark_swing_30_le12_minus_gt12",
            "secondary_value": round(float(avg(risk_le_12, "mark_swing_30")) - float(avg(risk_gt_12, "mark_swing_30")), 6),
            "primary_table": "risk_threshold_outcome_summary.csv",
            "caveat": "Risk buckets are static own-hand exposure detectors; partner and opponent adaptation remains offline.",
        },
        {
            "claim_id": "ch02-strong-trump-bad-risk-trap",
            "status_recommendation": "static_trap_surface_supported_outcome_mixed",
            "conclusion": "Strong-trump high-exposure contracts exist in the generated corpus. They keep high make rates, but their max-profitable-bid average trails low-risk strong-trump contracts, so the trap is a static/exposure warning rather than a proved make-rate penalty.",
            "n_contract_rows": len(contract_rows),
            "headline_metric": "trap_rows",
            "headline_value": len(trap_rows),
            "secondary_metric": "mean_p_make_30_trap",
            "secondary_value": avg(trap_rows, "p_make_30"),
            "primary_table": "strong_trump_bad_risk_trap_summary.csv",
            "caveat": "The corpus is small for this interaction slice; stronger paired same-hand high-off contrasts would be better.",
        },
        {
            "claim_id": "ch02-four-five-off-danger",
            "status_recommendation": "context_limited_exposure_supported",
            "conclusion": "Four/five-off exposure rows are measurably higher-risk and lower-value in the contract corpus.",
            "n_contract_rows": len(contract_rows),
            "headline_metric": "four_five_off_rows",
            "headline_value": len(four_five_rows),
            "secondary_metric": "mean_p_make_30_four_five_off",
            "secondary_value": avg(four_five_rows, "p_make_30"),
            "primary_table": "four_five_off_exposure_summary.csv",
            "caveat": "This is a detector/outcome join, not a generated line-of-play punishment test for specific 4-5 tiles.",
        },
        {
            "claim_id": "ch12-natural-bid-bucket-anomaly",
            "status_recommendation": "partial_empirical_bucket_evidence",
            "conclusion": "Natural max-profitable buckets appear in the generated contract rows, but static ceiling buckets alone cannot produce the book's 30/31/35/36 auction story.",
            "n_contract_rows": len(contract_rows),
            "headline_metric": "generated_natural_30_31_or_35_36_pct",
            "headline_value": pct(len(natural_rows), len(contract_rows)),
            "secondary_metric": "natural_rows",
            "secondary_value": len(natural_rows),
            "primary_table": "natural_bid_bucket_anomaly_summary.csv",
            "caveat": "Natural buckets are max-profitable bid labels from simulated points; no real auction-increment model is present.",
        },
        {
            "claim_id": "ch02-double-side-protection",
            "status_recommendation": "static_side_detector_supported_context_limited",
            "conclusion": "The side-specific detector distinguishes protected from unprotected exposure points; protection applies only to the side whose double is held.",
            "n_contract_rows": len(contract_rows),
            "headline_metric": "protected_exposed_point_pct",
            "headline_value": pct(protected_side_points, all_side_points),
            "secondary_metric": "side_exposure_rows",
            "secondary_value": len(exposed_side_rows),
            "primary_table": "double_side_protection_summary.csv",
            "caveat": "The detector knows whether the side double is held, but not whether it remains live or can actually be cashed in sequence.",
        },
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    contract_rows = enrich_contract_rows(read_csv(SOURCE_FILES["auction_contract_rows"]))
    partner_rows = add_partner_features(contract_rows)
    side_rows = [side for row in contract_rows for side in side_exposure_rows(row)]

    trump_rows = []
    for row in contract_rows:
        enriched = dict(row)
        enriched["trump_bucket"] = "trumps_3_plus" if int(row["static_trump_count"]) >= 3 else "trumps_0_2"
        enriched["trump_exposure_bucket"] = (
            "trumps_4_plus_risk_gt_12"
            if int(row["static_trump_count"]) >= 4 and int(row["static_unique_exposed_points"]) > 12
            else "other"
        )
        trump_rows.append(enriched)

    risk_rows = []
    for row in contract_rows:
        enriched = dict(row)
        enriched["risk_threshold_bucket"] = "risk_le_12" if int(row["static_unique_exposed_points"]) <= 12 else "risk_gt_12"
        risk_rows.append(enriched)

    write_csv(OUT_DIR / "trump_count_outcome_summary.csv", summarize_contract_group(trump_rows, ["trump_bucket"]))
    write_csv(OUT_DIR / "trump_count_detail_summary.csv", summarize_contract_group(contract_rows, ["static_trump_count"]))
    write_csv(OUT_DIR / "risk_threshold_outcome_summary.csv", summarize_contract_group(risk_rows, ["risk_threshold_bucket"]))
    write_csv(
        OUT_DIR / "strong_trump_bad_risk_trap_summary.csv",
        summarize_contract_group(contract_rows, ["static_strong_trump_bad_risk_trap"]),
    )
    strong_interaction_rows = []
    for row in contract_rows:
        enriched = dict(row)
        enriched["strong_trump_bucket"] = "trumps_4_plus" if int(row["static_trump_count"]) >= 4 else "trumps_0_3"
        enriched["risk_threshold_bucket"] = "risk_le_12" if int(row["static_unique_exposed_points"]) <= 12 else "risk_gt_12"
        strong_interaction_rows.append(enriched)
    write_csv(
        OUT_DIR / "strong_trump_risk_interaction_summary.csv",
        summarize_contract_group(strong_interaction_rows, ["strong_trump_bucket", "risk_threshold_bucket"]),
    )
    write_csv(
        OUT_DIR / "four_five_off_exposure_summary.csv",
        summarize_contract_group(contract_rows, ["static_four_five_off"]),
    )
    write_csv(OUT_DIR / "natural_bid_bucket_anomaly_summary.csv", summarize_natural_bucket(contract_rows))
    write_csv(
        OUT_DIR / "natural_bid_bucket_by_risk_summary.csv",
        summarize_contract_group(contract_rows, ["static_risk_bucket", "max_profitable_bid_bucket"]),
    )
    write_csv(OUT_DIR / "partner_two_plus_doubles_prior_summary.csv", summarize_partner_prior(partner_rows))
    write_csv(
        OUT_DIR / "partner_two_plus_doubles_examples.csv",
        partner_rows[:80],
        [
            "seed",
            "seat",
            "partner_seat",
            "hand",
            "decl_id",
            "decl_name",
            "static_off_count",
            "static_unique_exposed_points",
            "partner_double_count",
            "partner_two_plus_doubles",
            "p_make_30",
            "mark_swing_30",
        ],
    )
    write_csv(OUT_DIR / "double_side_protection_summary.csv", summarize_side_protection(side_rows))
    write_csv(OUT_DIR / "double_side_exposure_rows.csv", side_rows)

    claims = claim_summary(contract_rows=contract_rows, partner_rows=partner_rows, side_rows=side_rows)
    write_csv(OUT_DIR / "claim_summary.csv", claims)

    blocker_rows = [
        {
            "claim_id": claim_id,
            "blocker_type": "full_policy_or_sequence_counterfactual_missing",
            "needed_generator_or_state_injection": "Auction-aware bid/pass policy rollouts and/or sequence-state injection with make/set or E[Q] regret labels.",
            "current_artifact_status": "Phase-4 table is claim-specific and reproducible, but remains generated-contract/static-detector evidence unless the claim summary says otherwise.",
        }
        for claim_id in CLAIM_IDS
    ]
    write_csv(OUT_DIR / "blocker_rows.csv", blocker_rows)

    summary = {
        "schema_version": "w42.phase4_bidding_count_exposure_tests.v1",
        "bead": BEAD_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "claim_ids": CLAIM_IDS,
        "counts": {
            "contract_rows": len(contract_rows),
            "partner_feature_rows": len(partner_rows),
            "side_exposure_rows": len(side_rows),
            "side_exposure_rows_with_count_points": sum(1 for row in side_rows if int(row["exposed_count_points"]) > 0),
            "claim_summary_rows": len(claims),
        },
        "source_files": SOURCE_FILES,
        "outputs": {
            "claim_summary": "w42/phase4_bidding_count_exposure_tests/claim_summary.csv",
            "trump_count": "w42/phase4_bidding_count_exposure_tests/trump_count_outcome_summary.csv",
            "risk_threshold": "w42/phase4_bidding_count_exposure_tests/risk_threshold_outcome_summary.csv",
            "strong_trump_trap": "w42/phase4_bidding_count_exposure_tests/strong_trump_bad_risk_trap_summary.csv",
            "strong_trump_risk_interaction": "w42/phase4_bidding_count_exposure_tests/strong_trump_risk_interaction_summary.csv",
            "four_five_off": "w42/phase4_bidding_count_exposure_tests/four_five_off_exposure_summary.csv",
            "natural_bucket": "w42/phase4_bidding_count_exposure_tests/natural_bid_bucket_anomaly_summary.csv",
            "partner_prior": "w42/phase4_bidding_count_exposure_tests/partner_two_plus_doubles_prior_summary.csv",
            "double_side_protection": "w42/phase4_bidding_count_exposure_tests/double_side_protection_summary.csv",
            "blockers": "w42/phase4_bidding_count_exposure_tests/blocker_rows.csv",
        },
        "claim_conclusions": {row["claim_id"]: row for row in claims},
        "scientific_status": {
            "what_this_tests": "Claim-specific joins over prior generated contract/action rows plus exact static hand-shape and side-exposure detectors.",
            "what_this_does_not_test": "No central ledger update, no wiki update, no observed human auction, and no new full auction/game state injection.",
            "recommendation": "Use these tables to close the phase-4 scope gap as conservative claim evidence; leave claims caveated where full policy or sequence counterfactuals are still missing.",
        },
    }
    write_json(OUT_DIR / "summary.json", summary)


if __name__ == "__main__":
    main()
