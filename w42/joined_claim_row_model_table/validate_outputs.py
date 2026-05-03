#!/usr/bin/env python3
"""Validate joined W42 claim-row model table artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("w42/joined_claim_row_model_table"))
    parser.add_argument("--min-actions", type=int, default=75000)
    parser.add_argument("--min-eval-decisions", type=int, default=5000)
    parser.add_argument("--max-public-regret", type=float, default=10.0)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require(path: Path) -> None:
    if not path.exists():
        raise AssertionError(f"Missing artifact: {path}")
    if path.stat().st_size <= 0:
        raise AssertionError(f"Empty artifact: {path}")


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir
    for name in [
        "summary.json",
        "manifest.json",
        "feature_manifest.json",
        "metrics.json",
        "family_inventory.csv",
        "model_metrics.csv",
        "ablation_results.csv",
        "prediction_sample.jsonl",
        "joined_claim_action_rows.csv",
    ]:
        require(artifact_dir / name)

    summary = read_json(artifact_dir / "summary.json")
    manifest = read_json(artifact_dir / "manifest.json")
    if summary.get("bead") != "t42-qtwb.4":
        raise AssertionError(f"Unexpected bead: {summary.get('bead')}")
    coverage = summary["coverage"]
    split = summary["split"]
    if coverage["action_rows"] < args.min_actions:
        raise AssertionError(f"Expected >= {args.min_actions} rows, got {coverage['action_rows']}")
    if split["eval_decisions"] < args.min_eval_decisions:
        raise AssertionError(f"Expected >= {args.min_eval_decisions} eval decisions, got {split['eval_decisions']}")
    if summary["headline"]["public_mean_regret"] >= args.max_public_regret:
        raise AssertionError("public baseline regret too high")
    if "hidden_owner" not in manifest["offline_label_columns_excluded_from_features"]:
        raise AssertionError("hidden-owner exclusion missing")

    inventory = read_csv(artifact_dir / "family_inventory.csv")
    families = {row["family"]: row for row in inventory}
    for family in [
        "sequence_seat",
        "bidding_risk",
        "eighty_four_public",
        "doubles_no_trump",
        "hidden_public_proxy",
    ]:
        if families.get(family, {}).get("status") != "available":
            raise AssertionError(f"Missing available family: {family}")
        if int(float(families[family]["tag_action_n"])) <= 0:
            raise AssertionError(f"No tag rows for family: {family}")
    if families.get("hidden_truth_owner", {}).get("status") != "eval_only":
        raise AssertionError("hidden truth should be eval-only")

    metrics = read_csv(artifact_dir / "model_metrics.csv")
    variants = {row["variant"]: row for row in metrics}
    expected_variants = {
        "actual_policy",
        "public_features_only",
        "public_plus_all_claim_families",
        "drop_bidding_risk",
        "drop_doubles_no_trump",
        "drop_eighty_four_public",
        "drop_hidden_public_proxy",
        "drop_sequence_seat",
    }
    missing = expected_variants - set(variants)
    if missing:
        raise AssertionError(f"Missing variants: {sorted(missing)}")
    if float(variants["public_plus_all_claim_families"]["mean_regret"]) >= args.max_public_regret:
        raise AssertionError("claim-family model regret too high")

    sample_rows = sum(1 for _ in (artifact_dir / "prediction_sample.jsonl").open(encoding="utf-8"))
    if sample_rows <= 0:
        raise AssertionError("prediction sample is empty")

    print(
        json.dumps(
            {
                "ok": True,
                "artifact_dir": str(artifact_dir),
                "action_rows": coverage["action_rows"],
                "eval_decisions": split["eval_decisions"],
                "families": sorted(expected_variants),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
