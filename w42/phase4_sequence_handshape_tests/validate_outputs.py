#!/usr/bin/env python3
"""Validate phase-4 W42 sequence hand-shape artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--min-actions", type=int, default=75_000)
    parser.add_argument("--min-decisions", type=int, default=28_000)
    parser.add_argument("--min-label-metrics", type=int, default=35)
    parser.add_argument("--min-blockers", type=int, default=6)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    required = [
        "summary.json",
        "labeled_handshape_action_rows.csv",
        "decision_shape_rows.csv",
        "label_metrics.csv",
        "paired_contrasts.csv",
        "paired_contrasts_by_slice.csv",
        "blocker_table.csv",
        "examples.json",
    ]
    missing = [name for name in required if not (args.artifact_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    summary = json.loads((args.artifact_dir / "summary.json").read_text(encoding="utf-8"))
    coverage = summary.get("coverage", {})
    if coverage.get("action_rows", 0) < args.min_actions:
        raise SystemExit(f"Too few action rows: {coverage.get('action_rows')}")
    if coverage.get("decision_rows", 0) < args.min_decisions:
        raise SystemExit(f"Too few decision rows: {coverage.get('decision_rows')}")
    if coverage.get("label_metric_rows", 0) < args.min_label_metrics:
        raise SystemExit(f"Too few label metrics: {coverage.get('label_metric_rows')}")
    if coverage.get("blocker_rows", 0) < args.min_blockers:
        raise SystemExit(f"Too few blocker rows: {coverage.get('blocker_rows')}")

    labeled = read_csv_rows(args.artifact_dir / "labeled_handshape_action_rows.csv")
    if len(labeled) != coverage.get("action_rows"):
        raise SystemExit("Labeled row count does not match summary coverage")
    if not any("ch03_commanding_called_double" in row.get("labels", "") for row in labeled):
        raise SystemExit("Missing commanding called-double labels")
    if not any("ch04_partner_count_before_closure_not_guaranteed" in row.get("labels", "") for row in labeled):
        raise SystemExit("Missing partner before-closure donation labels")
    if not any("ch05_count_before_certainty_to_partner_pounce" in row.get("labels", "") for row in labeled):
        raise SystemExit("Missing setter count-before-certainty labels")

    blockers = read_csv_rows(args.artifact_dir / "blocker_table.csv")
    blocker_text = " ".join(row.get("missing_fields", "") for row in blockers)
    if "hidden" not in blocker_text and "remaining" not in blocker_text:
        raise SystemExit("Blocker table does not document hidden/remaining-state gaps")

    print(
        "validated",
        coverage.get("action_rows"),
        "actions",
        coverage.get("decision_rows"),
        "decisions",
        coverage.get("label_metric_rows"),
        "label metrics",
    )


if __name__ == "__main__":
    main()
