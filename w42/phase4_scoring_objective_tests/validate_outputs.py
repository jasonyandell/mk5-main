#!/usr/bin/env python3
"""Validate W42 phase-4 scoring objective artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("w42/phase4_scoring_objective_tests"))
    parser.add_argument("--min-generated-hands", type=int, default=4000)
    parser.add_argument("--min-match-rows", type=int, default=600)
    parser.add_argument("--min-timed-trials", type=int, default=100)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing artifact: {path}")
    if path.stat().st_size <= 0:
        raise AssertionError(f"Empty artifact: {path}")


def main() -> int:
    args = parse_args()
    required = [
        "summary.json",
        "claim_summary.csv",
        "generated_hand_traces.csv",
        "early_terminal_summary.csv",
        "objective_compression_summary.csv",
        "policy_population_signal.csv",
        "match_mode_rows.csv",
        "match_mode_summary.csv",
        "timed_advancement_rows.csv",
        "timed_advancement_summary.csv",
        "deterministic_terminal_transform.csv",
        "deterministic_thresholds.csv",
        "blockers.csv",
        "examples.json",
    ]
    for name in required:
        require(args.artifact_dir / name)

    summary = read_json(args.artifact_dir / "summary.json")
    if summary.get("bead") != "t42-br7n.3":
        raise AssertionError(f"Unexpected bead id: {summary.get('bead')}")
    coverage = summary["coverage"]
    if coverage["generated_hands"] < args.min_generated_hands:
        raise AssertionError(f"generated_hands below threshold: {coverage['generated_hands']}")
    if coverage["match_rows"] < args.min_match_rows:
        raise AssertionError(f"match_rows below threshold: {coverage['match_rows']}")
    if coverage["timed_trials"] < args.min_timed_trials:
        raise AssertionError(f"timed_trials below threshold: {coverage['timed_trials']}")
    if coverage["claims_reported"] != 9:
        raise AssertionError(f"Expected 9 claim rows, got {coverage['claims_reported']}")
    if coverage["deterministic_terminal_rows"] < 600:
        raise AssertionError("Deterministic terminal table is too small")

    claims = read_csv(args.artifact_dir / "claim_summary.csv")
    claim_ids = {row["claim_id"] for row in claims}
    expected = {
        "ch10-score-mode-objective",
        "ch10-early-terminal-under-marks",
        "ch10-nonbidder-partial-points-erased",
        "ch10-set-severity-compression",
        "ch10-special-bid-mark-multiplier",
        "ch10-low-bid-score-distortion",
        "ch10-point-system-skill-signal",
        "ch10-tournament-speed-tradeoff",
        "ch10-timed-marks-advancement-objective",
    }
    missing = expected - claim_ids
    if missing:
        raise AssertionError(f"Missing claims: {sorted(missing)}")

    metrics = summary["headline_metrics"]
    if metrics["early_terminal_rate"] <= 0:
        raise AssertionError("No generated early terminal states found")
    if metrics["partial_erasure_hands"] <= 0:
        raise AssertionError("No partial defender point erasure hands found")
    if metrics["distinct_ordinary_set_severity_values"] <= 1:
        raise AssertionError("Set severity compression was not exercised")

    examples = read_json(args.artifact_dir / "examples.json")
    for claim_id in expected:
        if claim_id not in examples:
            raise AssertionError(f"Missing examples entry for {claim_id}")
        if not examples[claim_id]:
            raise AssertionError(f"Empty examples entry for {claim_id}")

    blockers = read_csv(args.artifact_dir / "blockers.csv")
    blocker_claims = {row["claim_id"] for row in blockers}
    for claim_id in {
        "ch10-point-system-skill-signal",
        "ch10-tournament-speed-tradeoff",
        "ch10-timed-marks-advancement-objective",
    }:
        if claim_id not in blocker_claims:
            raise AssertionError(f"Missing blocker for {claim_id}")

    print(
        json.dumps(
            {
                "ok": True,
                "artifact_dir": str(args.artifact_dir),
                "generated_hands": coverage["generated_hands"],
                "match_rows": coverage["match_rows"],
                "timed_trials": coverage["timed_trials"],
                "early_terminal_rate": metrics["early_terminal_rate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
