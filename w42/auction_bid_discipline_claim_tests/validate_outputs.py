#!/usr/bin/env python3
"""Validate W42 auction bid discipline artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("w42/auction_bid_discipline_claim_tests"))
    parser.add_argument("--min-contract-rows", type=int, default=384)
    parser.add_argument("--min-bid-action-rows", type=int, default=16000)
    args = parser.parse_args()

    base = args.artifact_dir
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    contract_rows = read_csv(base / "contract_rows.csv")
    bid_action_rows = read_csv(base / "bid_action_rows.csv")
    context_rows = read_csv(base / "context_decision_rows.csv")
    margin_summary = read_csv(base / "bid_margin_summary.csv")

    if summary["counts"]["contract_rows"] < args.min_contract_rows:
        raise AssertionError("Too few contract rows")
    if summary["counts"]["bid_action_rows"] < args.min_bid_action_rows:
        raise AssertionError("Too few bid action rows")
    if len(contract_rows) != summary["counts"]["contract_rows"]:
        raise AssertionError("contract_rows.csv count mismatch")
    if len(bid_action_rows) != summary["counts"]["bid_action_rows"]:
        raise AssertionError("bid_action_rows.csv count mismatch")
    if len(context_rows) != summary["counts"]["context_decision_rows"]:
        raise AssertionError("context_decision_rows.csv count mismatch")
    if summary["headline"]["positive_margin_delta_vs_min_positive_rows"] != 0:
        raise AssertionError("Positive bid margin improved over minimum bid")
    if not margin_summary:
        raise AssertionError("Missing bid margin summary rows")
    if not any(row["high_bid_owner_role"] == "partner" for row in context_rows):
        raise AssertionError("Missing partner-high auction contexts")
    if not any(row["high_bid_owner_role"] in {"left_opponent", "right_opponent"} for row in context_rows):
        raise AssertionError("Missing opponent-high auction contexts")
    if not any(int(row["max_profitable_bid"]) > 0 for row in contract_rows):
        raise AssertionError("No profitable contract labels")

    print(
        json.dumps(
            {
                "artifact_dir": str(base),
                "contract_rows": len(contract_rows),
                "bid_action_rows": len(bid_action_rows),
                "context_decision_rows": len(context_rows),
                "wandb_run_url": summary.get("wandb", {}).get("run_url"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
