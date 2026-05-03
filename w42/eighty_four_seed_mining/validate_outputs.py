#!/usr/bin/env python3
"""Validate W42 84 seed-mining artifacts."""

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
    parser.add_argument("--artifact-dir", type=Path, default=Path("w42/eighty_four_seed_mining"))
    parser.add_argument("--min-candidate-rows", type=int, default=200000)
    parser.add_argument("--min-recommended-rows", type=int, default=256)
    args = parser.parse_args()

    base = args.artifact_dir
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    candidate_rows = read_csv(base / "candidate_84_seed_rows.csv")
    recommended_rows = read_csv(base / "recommended_84_seed_rows.csv")
    surface_rows = read_csv(base / "surface_summary.csv")

    if len(candidate_rows) != summary["counts"]["candidate_rows"]:
        raise AssertionError("candidate_84_seed_rows.csv count mismatch")
    if len(candidate_rows) < args.min_candidate_rows:
        raise AssertionError("Too few candidate rows")
    if len(recommended_rows) < args.min_recommended_rows:
        raise AssertionError("Too few recommended rows")
    required = {
        "protected_one_off",
        "straight_one_off",
        "two_off_same_suit",
        "defender_live_double_weapon",
        "defender_live_same_suit_pair",
        "pair_protector_pressure",
    }
    seen = {row["surface"] for row in surface_rows}
    missing = required - seen
    if missing:
        raise AssertionError(f"Missing required surfaces: {sorted(missing)}")
    if summary["counts"]["scanned_seeds"] < 50000:
        raise AssertionError("Seed scan is smaller than the archived run")
    if not summary.get("wandb", {}).get("run_url"):
        raise AssertionError("Missing W&B run URL")

    print(
        json.dumps(
            {
                "artifact_dir": str(base),
                "candidate_rows": len(candidate_rows),
                "recommended_rows": len(recommended_rows),
                "surfaces": sorted(seen),
                "wandb_run_url": summary["wandb"]["run_url"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
