#!/usr/bin/env python3
"""Validate W42 phase-4 dynamic 84 mined-seed outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "summary.json",
    "manifest.json",
    "selected_seed_rows.csv",
    "dynamic_84_action_labels.csv",
    "dynamic_84_label_metrics.csv",
    "dynamic_84_paired_contrasts.csv",
    "score_gate_static_table.csv",
    "blocker_table.csv",
    "examples.json",
    "branch_atlas/summary.json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = args.artifact_dir
    missing = [name for name in REQUIRED_FILES if not (artifact_dir / name).exists()]
    require(not missing, f"Missing required outputs: {missing}")

    summary = json.loads((artifact_dir / "summary.json").read_text(encoding="utf-8"))
    actions = read_csv(artifact_dir / "dynamic_84_action_labels.csv")
    metrics = read_csv(artifact_dir / "dynamic_84_label_metrics.csv")
    contrasts = read_csv(artifact_dir / "dynamic_84_paired_contrasts.csv")
    selected = read_csv(artifact_dir / "selected_seed_rows.csv")
    blockers = read_csv(artifact_dir / "blocker_table.csv")

    require(summary["owner_bead"] == "t42-br7n.2", "wrong owner bead")
    require(summary["coverage"]["seed_games"] >= args.min_seed_games, "too few generated seed games")
    require(summary["coverage"]["action_rows"] >= args.min_action_rows, "too few action rows")
    require(summary["coverage"]["paired_contrasts"] >= args.min_paired_contrasts, "too few paired contrast families")
    require(len(actions) == summary["coverage"]["dynamic_labeled_actions"], "action label count mismatch")
    require(len(selected) == summary["coverage"]["seed_games"], "selected seed count mismatch")
    require(len(blockers) >= 4, "expected blocker table rows")

    labels = {row["dynamic_84_label"] for row in metrics}
    expected_labels = {
        "is_84_contract_decision",
        "bidder_trump_pull_candidate",
        "bidder_nontrump_final_off_candidate",
        "defender_live_double_weapon_proxy",
        "defender_live_same_suit_pair_proxy",
        "dead_asset_release_candidate_proxy",
    }
    require(not (expected_labels - labels), f"missing expected labels: {sorted(expected_labels - labels)}")

    contrast_ids = {row["contrast_id"] for row in contrasts}
    require("offense_trump_pull_vs_final_off_proxy" in contrast_ids, "missing offense trump/final-off contrast")
    require(
        "defense_preserve_expendable_vs_spend_live_asset_proxy" in contrast_ids
        or "dead_asset_release_vs_spend_live_asset_proxy" in contrast_ids,
        "missing defense preserve/spend contrast",
    )

    return {
        "seed_games": summary["coverage"]["seed_games"],
        "action_rows": summary["coverage"]["action_rows"],
        "paired_contrasts": summary["coverage"]["paired_contrasts"],
        "labels": sorted(labels),
        "contrast_ids": sorted(contrast_ids),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--min-seed-games", type=int, default=8)
    parser.add_argument("--min-action-rows", type=int, default=700)
    parser.add_argument("--min-paired-contrasts", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    result = validate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
