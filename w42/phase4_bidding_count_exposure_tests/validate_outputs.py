#!/usr/bin/env python3
"""Validate Worker G phase-4 bidding count/exposure outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "w42" / "phase4_bidding_count_exposure_tests"
CLAIM_IDS = {
    "ch16-partner-two-plus-doubles-prior",
    "ch02-three-plus-trumps-good-start",
    "ch02-risk-budget-threshold",
    "ch02-strong-trump-bad-risk-trap",
    "ch02-four-five-off-danger",
    "ch12-natural-bid-bucket-anomaly",
    "ch02-double-side-protection",
}

REQUIRED_CSVS = {
    "claim_summary.csv": 7,
    "trump_count_outcome_summary.csv": 2,
    "trump_count_detail_summary.csv": 1,
    "risk_threshold_outcome_summary.csv": 2,
    "strong_trump_bad_risk_trap_summary.csv": 2,
    "strong_trump_risk_interaction_summary.csv": 3,
    "four_five_off_exposure_summary.csv": 2,
    "natural_bid_bucket_anomaly_summary.csv": 1,
    "natural_bid_bucket_by_risk_summary.csv": 1,
    "partner_two_plus_doubles_prior_summary.csv": 1,
    "partner_two_plus_doubles_examples.csv": 1,
    "double_side_protection_summary.csv": 2,
    "double_side_exposure_rows.csv": 1,
    "blocker_rows.csv": 7,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    errors: list[str] = []
    summary_path = OUT_DIR / "summary.json"
    if not summary_path.exists():
        errors.append("missing summary.json")
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        claim_ids = set(summary.get("claim_ids", []))
        if claim_ids != CLAIM_IDS:
            errors.append(f"summary claim_ids mismatch: {sorted(claim_ids ^ CLAIM_IDS)}")
        counts = summary.get("counts", {})
        if counts.get("contract_rows", 0) < 300:
            errors.append("contract_rows unexpectedly low")
        if counts.get("claim_summary_rows") != 7:
            errors.append("claim_summary_rows must equal 7")

    for name, min_rows in REQUIRED_CSVS.items():
        path = OUT_DIR / name
        if not path.exists():
            errors.append(f"missing {name}")
            continue
        rows = read_csv(path)
        if len(rows) < min_rows:
            errors.append(f"{name} has {len(rows)} rows, expected at least {min_rows}")

    claim_rows = read_csv(OUT_DIR / "claim_summary.csv") if (OUT_DIR / "claim_summary.csv").exists() else []
    found_claims = {row.get("claim_id", "") for row in claim_rows}
    if found_claims != CLAIM_IDS:
        errors.append(f"claim_summary claim_ids mismatch: {sorted(found_claims ^ CLAIM_IDS)}")
    for row in claim_rows:
        for key in ("status_recommendation", "conclusion", "primary_table", "caveat"):
            if not row.get(key):
                errors.append(f"claim_summary missing {key} for {row.get('claim_id')}")

    blocker_rows = read_csv(OUT_DIR / "blocker_rows.csv") if (OUT_DIR / "blocker_rows.csv").exists() else []
    blocker_claims = {row.get("claim_id", "") for row in blocker_rows}
    if blocker_claims != CLAIM_IDS:
        errors.append(f"blocker_rows claim_ids mismatch: {sorted(blocker_claims ^ CLAIM_IDS)}")

    if errors:
        raise SystemExit("validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    print("validated phase4_bidding_count_exposure_tests outputs")


if __name__ == "__main__":
    main()
