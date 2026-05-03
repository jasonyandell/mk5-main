#!/usr/bin/env python3
"""Validate generated W42 phase-4 completion-board artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REQUIRED_COLUMNS = {
    "claim_id",
    "family",
    "ledger_status",
    "completion_bucket",
    "evidence_strength",
    "phase4_next_bead",
    "phase4_recommendation",
    "claim",
    "headline_evidence",
    "primary_artifacts_existing",
    "known_blocker_or_caveat",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-claims", type=int, default=64)
    args = parser.parse_args()

    board_path = args.artifact_dir / "completion_board.csv"
    family_path = args.artifact_dir / "family_summary.csv"
    summary_path = args.artifact_dir / "summary.json"

    for path in [board_path, family_path, summary_path]:
        if not path.exists():
            fail(f"missing {path}")

    board_rows = read_csv(board_path)
    family_rows = read_csv(family_path)
    summary = json.loads(summary_path.read_text())

    if len(board_rows) != args.expected_claims:
        fail(f"expected {args.expected_claims} board rows, found {len(board_rows)}")

    if len({row["claim_id"] for row in board_rows}) != len(board_rows):
        fail("duplicate claim_id in completion board")

    with board_path.open(newline="") as fh:
        fieldnames = csv.DictReader(fh).fieldnames or []
    missing_columns = REQUIRED_COLUMNS - set(fieldnames)
    if missing_columns:
        fail(f"missing board columns: {sorted(missing_columns)}")

    for row in board_rows:
        for column in REQUIRED_COLUMNS:
            if not row.get(column):
                fail(f"{row['claim_id']} has empty required column {column}")

    if summary.get("claim_count") != len(board_rows):
        fail("summary claim_count does not match board row count")

    board_completion = Counter(row["completion_bucket"] for row in board_rows)
    if dict(sorted(board_completion.items())) != summary.get("completion_bucket_counts"):
        fail("summary completion_bucket_counts do not match board")

    board_next = Counter(row["phase4_next_bead"] for row in board_rows)
    if dict(sorted(board_next.items())) != summary.get("next_bead_counts"):
        fail("summary next_bead_counts do not match board")

    family_total = sum(int(row["claim_count"]) for row in family_rows)
    if family_total != len(board_rows):
        fail("family_summary claim_count does not sum to board row count")

    if board_next.get("t42-br7n.6"):
        fail("board should not route claims to itself")

    if not any(row["phase4_next_bead"].startswith("t42-br7n.") for row in board_rows):
        fail("board does not route any claims to active phase4 children")

    if not any(row["phase4_next_bead"] == "needs_new_bidding_count_exposure_bead" for row in board_rows):
        fail("board failed to flag the bidding/count-exposure scope gap")

    print(
        json.dumps(
            {
                "ok": True,
                "claim_rows": len(board_rows),
                "families": len(family_rows),
                "completion_bucket_counts": dict(sorted(board_completion.items())),
                "next_bead_counts": dict(sorted(board_next.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
