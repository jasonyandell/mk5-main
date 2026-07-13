#!/usr/bin/env python3
"""Validate claim-tag model probe artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("w42/claim_tag_model_probe"))
    parser.add_argument("--min-eval-decisions", type=int, default=5000)
    parser.add_argument("--max-baseline-regret", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        "summary.json",
        "metrics.json",
        "feature_manifest.json",
        "ablation_results.csv",
        "prediction_sample.jsonl",
    ]
    for name in required:
        path = args.artifact_dir / name
        if not path.exists():
            raise SystemExit(f"missing artifact: {path}")

    summary = json.loads((args.artifact_dir / "summary.json").read_text())
    metrics = json.loads((args.artifact_dir / "metrics.json").read_text())
    manifest = json.loads((args.artifact_dir / "feature_manifest.json").read_text())

    split = metrics["split"]
    assert split["eval_decisions"] >= args.min_eval_decisions, split["eval_decisions"]
    assert split["train_actions"] > split["eval_actions"]
    assert split["train_seeds"] and split["eval_seeds"]
    assert max(split["train_seeds"]) < min(split["eval_seeds"])

    headline = summary["headline"]
    public_regret = headline.get(
        "public_baseline_eval_selected_mean_regret",
        headline.get("public_mean_regret"),
    )
    tag_regret = headline.get(
        "tag_model_eval_selected_mean_regret",
        headline.get("tag_mean_regret"),
    )
    assert public_regret is not None and public_regret < args.max_baseline_regret
    assert tag_regret is not None and tag_regret < args.max_baseline_regret
    assert headline.get("best_trained_variant_by_regret") or headline.get("tag_delta_vs_public_mean_regret") is not None

    assert manifest["tag_count"] >= 20
    assert "is_best_mean" in manifest["offline_label_columns_excluded_from_features"]
    assert "mean" in manifest["offline_label_columns_excluded_from_features"]
    assert "hidden_threat_distribution" in manifest["not_available_families"]

    with (args.artifact_dir / "ablation_results.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    variants = {row["variant"]: row for row in rows}
    for name in [
        "public_baseline",
        "public_plus_claim_tags",
        "drop_pounce_donation",
        "drop_seat_position_closure",
        "drop_bid_risk",
        "drop_eighty_four_endgame",
        "drop_doubles_no_trump",
        "drop_hidden_threat_distribution",
    ]:
        assert name in variants, name
    assert variants["drop_bid_risk"]["status"] == "not_available"
    assert variants["drop_eighty_four_endgame"]["status"] == "ood_eval_only"
    assert variants["drop_hidden_threat_distribution"]["status"] == "eval_only"

    sample_count = sum(1 for _ in (args.artifact_dir / "prediction_sample.jsonl").open())
    assert sample_count > 0

    print(
        "validated",
        split["eval_decisions"],
        "eval decisions;",
        manifest["tag_count"],
        "tags;",
        sample_count,
        "sample rows",
    )


if __name__ == "__main__":
    main()
