#!/usr/bin/env python3
"""Validate W42 phase-4 laydown/rule-accounting artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "summary.json",
    "manifest.json",
    "rule_results.csv",
    "rule_fixtures.json",
    "laydown_results.csv",
    "laydown_fixtures.json",
    "laydown_counterexamples.json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing artifact: {path}")
    if path.stat().st_size <= 0:
        raise AssertionError(f"Empty artifact: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("w42/phase4_laydown_rule_accounting"))
    parser.add_argument("--min-rule-assertions", type=int, default=20)
    parser.add_argument("--min-laydown-fixtures", type=int, default=6)
    args = parser.parse_args()

    base = args.artifact_dir
    for name in REQUIRED_FILES:
        require(base / name)

    summary = read_json(base / "summary.json")
    manifest = read_json(base / "manifest.json")
    rule_rows = read_csv(base / "rule_results.csv")
    laydown_rows = read_csv(base / "laydown_results.csv")
    rule_fixtures = read_json(base / "rule_fixtures.json")
    laydown_fixtures = read_json(base / "laydown_fixtures.json")
    counterexamples = read_json(base / "laydown_counterexamples.json")

    if summary.get("bead") != "t42-br7n.4":
        raise AssertionError(f"Unexpected bead id: {summary.get('bead')}")
    if manifest.get("bead") != summary.get("bead"):
        raise AssertionError("Manifest bead does not match summary")
    if len(rule_rows) < args.min_rule_assertions:
        raise AssertionError(f"Too few rule assertions: {len(rule_rows)}")
    if len(laydown_rows) < args.min_laydown_fixtures:
        raise AssertionError(f"Too few laydown fixtures: {len(laydown_rows)}")
    if summary["coverage"]["rule_assertions"] != len(rule_rows):
        raise AssertionError("Rule assertion count mismatch")
    if summary["coverage"]["laydown_fixtures"] != len(laydown_rows):
        raise AssertionError("Laydown fixture count mismatch")
    if summary["coverage"]["rule_fixtures"] != len(rule_fixtures):
        raise AssertionError("Rule fixture count mismatch")
    if summary["coverage"]["laydown_counterexamples"] != len(counterexamples):
        raise AssertionError("Laydown counterexample count mismatch")

    failed_rule_rows = [row for row in rule_rows if row["status"] != "pass"]
    failed_laydown_rows = [row for row in laydown_rows if row["status"] != "pass"]
    if failed_rule_rows or failed_laydown_rows:
        raise AssertionError(
            f"Found failed rows: rule={len(failed_rule_rows)} laydown={len(failed_laydown_rows)}"
        )
    if summary["coverage"]["failures"] != 0:
        raise AssertionError(f"Summary reports failures: {summary['coverage']['failures']}")

    expected_families = {
        "count_identity",
        "suit_membership",
        "trump_exclusivity",
        "follow_suit_mask",
        "trick_winner",
        "lead_control",
        "count_capture",
        "contract_scoring",
        "laydown_exact_proof",
    }
    missing = expected_families - set(summary["families"])
    if missing:
        raise AssertionError(f"Missing rule families: {sorted(missing)}")

    laydown_ids = {row["fixture_id"] for row in laydown_rows}
    required_laydown = {
        "single_boss_trump_laydown",
        "two_boss_trumps_laydown",
        "false_off_can_be_taken",
        "false_secondary_trump_exclusion",
        "walker_after_suit_exhaustion",
        "book_like_final_deuce_ace_counterexample",
    }
    missing_laydown = required_laydown - laydown_ids
    if missing_laydown:
        raise AssertionError(f"Missing laydown fixtures: {sorted(missing_laydown)}")

    false_fixtures = {row["fixture_id"] for row in laydown_rows if row["observed_proven"] == "False"}
    counterexample_ids = {row["fixture_id"] for row in counterexamples}
    if not false_fixtures <= counterexample_ids:
        raise AssertionError("Every rejected laydown should emit a counterexample")

    print(
        json.dumps(
            {
                "ok": True,
                "artifact_dir": str(base),
                "rule_assertions": len(rule_rows),
                "laydown_fixtures": len(laydown_rows),
                "counterexamples": len(counterexamples),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
