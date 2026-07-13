#!/usr/bin/env python3
"""Validate the seat/position claim-test artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED = [
    "summary.json",
    "manifest.json",
    "label_metrics.csv",
    "role_position_metrics.csv",
    "paired_contrasts.csv",
    "paired_contrasts_by_slice.csv",
    "labeled_action_rows.csv",
    "examples.json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("w42/seat_position_claim_tests"))
    parser.add_argument("--min-actions", type=int, default=1)
    parser.add_argument("--min-paired-contrasts", type=int, default=1)
    args = parser.parse_args()

    base = args.artifact_dir
    missing = [name for name in REQUIRED if not (base / name).exists()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")

    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    labels = read_csv(base / "label_metrics.csv")
    role_positions = read_csv(base / "role_position_metrics.csv")
    paired = read_csv(base / "paired_contrasts.csv")
    labeled_actions = read_csv(base / "labeled_action_rows.csv")
    examples = json.loads((base / "examples.json").read_text(encoding="utf-8"))

    coverage = summary["coverage"]
    assert summary["bead"] == "t42-0b4l.6"
    assert manifest["bead"] == "t42-0b4l.6"
    assert coverage["action_rows"] >= args.min_actions
    assert len(labeled_actions) == coverage["action_rows"]
    assert coverage["paired_contrast_rows"] >= args.min_paired_contrasts
    assert len(labels) == coverage["label_metric_rows"]
    assert len(role_positions) == coverage["role_position_metric_rows"]
    assert len(paired) == coverage["paired_contrast_rows"]
    assert examples

    label_names = {row["label"] for row in labels}
    for required_label in [
        "seat_first_lead",
        "seat_second_follow",
        "seat_third_follow",
        "seat_fourth_closure",
        "role_bidder",
        "role_partner",
        "role_setter",
        "role_defender",
        "defensive_pounce_count",
        "partner_support_count_current_control",
        "closure_take_trick",
    ]:
        assert required_label in label_names, f"missing label metric {required_label}"

    for row in labeled_actions[:100]:
        assert "world_hands" not in row
        assert "q_per_world" not in row

    print(
        json.dumps(
            {
                "validated": str(base),
                "action_rows": coverage["action_rows"],
                "paired_contrast_rows": coverage["paired_contrast_rows"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
