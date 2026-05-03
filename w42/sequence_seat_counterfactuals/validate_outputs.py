#!/usr/bin/env python3
"""Validate W42 sequence/seat counterfactual artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("w42/sequence_seat_counterfactuals"))
    parser.add_argument("--min-actions", type=int, default=75000)
    parser.add_argument("--min-paired-contrasts", type=int, default=10)
    parser.add_argument("--min-slice-rows", type=int, default=80)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing artifact: {path}")
    if path.stat().st_size <= 0:
        raise AssertionError(f"Empty artifact: {path}")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise AssertionError(f"Missing numeric field {key}: {row}")
    return float(value)


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir
    required = [
        "summary.json",
        "manifest.json",
        "label_metrics.csv",
        "paired_contrasts.csv",
        "paired_contrasts_by_slice.csv",
        "labeled_sequence_action_rows.csv",
        "examples.json",
    ]
    for name in required:
        require(artifact_dir / name)

    summary = read_json(artifact_dir / "summary.json")
    coverage = summary["coverage"]
    if summary.get("bead") != "t42-qtwb.3":
        raise AssertionError(f"Unexpected bead id: {summary.get('bead')}")
    if coverage["action_rows"] < args.min_actions:
        raise AssertionError(f"Expected >= {args.min_actions} action rows, got {coverage['action_rows']}")
    if coverage["paired_contrast_rows"] < args.min_paired_contrasts:
        raise AssertionError(
            f"Expected >= {args.min_paired_contrasts} paired contrasts, got {coverage['paired_contrast_rows']}"
        )
    if coverage["paired_contrast_slice_rows"] < args.min_slice_rows:
        raise AssertionError(
            f"Expected >= {args.min_slice_rows} slice rows, got {coverage['paired_contrast_slice_rows']}"
        )

    contrasts = read_csv(artifact_dir / "paired_contrasts.csv")
    contrast_ids = {row["contrast_id"] for row in contrasts}
    expected = {
        "bidder_lead_called_suit_vs_off_suit",
        "follower_beat_opponent_vs_decline",
        "partner_count_when_bidder_side_controls_vs_other",
        "setter_pounce_count_vs_other",
        "closure_take_trick_vs_decline",
    }
    missing = expected - contrast_ids
    if missing:
        raise AssertionError(f"Missing expected contrasts: {sorted(missing)}")

    for row in contrasts:
        as_float(row, "mean_delta")
        as_float(row, "mean_delta_ci95_low")
        as_float(row, "mean_delta_ci95_high")
        if int(row["paired_decision_n"]) <= 0:
            raise AssertionError(f"Non-positive paired count: {row}")

    examples = read_json(artifact_dir / "examples.json")
    for contrast_id in expected:
        if contrast_id not in examples:
            raise AssertionError(f"Missing examples for {contrast_id}")
        if not examples[contrast_id]["positive_examples"]:
            raise AssertionError(f"Missing positive examples for {contrast_id}")
        if not examples[contrast_id]["negative_examples"]:
            raise AssertionError(f"Missing negative examples for {contrast_id}")

    print(
        json.dumps(
            {
                "ok": True,
                "artifact_dir": str(artifact_dir),
                "action_rows": coverage["action_rows"],
                "paired_contrast_rows": coverage["paired_contrast_rows"],
                "paired_contrast_slice_rows": coverage["paired_contrast_slice_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
