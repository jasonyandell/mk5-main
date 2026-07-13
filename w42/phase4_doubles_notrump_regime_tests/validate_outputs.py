#!/usr/bin/env python3
"""Validate phase-4 doubles/no-trump paired-regime artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FILES = [
    "selected_hand_rows.csv",
    "regime_contract_rows.csv",
    "paired_regime_rows.csv",
    "point_sample_pairs.csv",
    "bucket_summary.csv",
    "claim_contrasts.csv",
    "examples.json",
    "summary.json",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("w42/phase4_doubles_notrump_regime_tests"))
    parser.add_argument("--min-hands", type=int, default=100)
    parser.add_argument("--min-claim-contrasts", type=int, default=6)
    args = parser.parse_args()

    missing = [name for name in REQUIRED_FILES if not (args.artifact_dir / name).exists()]
    if missing:
        raise SystemExit(f"missing required files: {missing}")

    summary = json.loads((args.artifact_dir / "summary.json").read_text(encoding="utf-8"))
    paired = read_csv(args.artifact_dir / "paired_regime_rows.csv")
    regimes = read_csv(args.artifact_dir / "regime_contract_rows.csv")
    claims = read_csv(args.artifact_dir / "claim_contrasts.csv")
    samples = read_csv(args.artifact_dir / "point_sample_pairs.csv")

    if len(paired) < args.min_hands:
        raise SystemExit(f"paired hand rows below floor: {len(paired)} < {args.min_hands}")
    if len(regimes) != len(paired) * 2:
        raise SystemExit(f"expected two regime rows per hand, got {len(regimes)} for {len(paired)} hands")
    if len(claims) < args.min_claim_contrasts:
        raise SystemExit(f"claim contrasts below floor: {len(claims)} < {args.min_claim_contrasts}")
    if len(samples) < len(paired):
        raise SystemExit("point sample pairs unexpectedly sparse")

    required_pair_columns = {
        "hand_index",
        "hand",
        "dt_mean_points",
        "nt_mean_points",
        "nt_minus_dt_best_mark_swing",
        "preferred_decl_at_30",
        "four_plus_doubles",
        "no_trump_support",
        "regime_switch_proxy",
    }
    missing_cols = required_pair_columns - set(paired[0])
    if missing_cols:
        raise SystemExit(f"paired_regime_rows missing columns: {sorted(missing_cols)}")

    if summary["coverage"]["paired_regime_rows"] != len(paired):
        raise SystemExit("summary paired_regime_rows does not match CSV")
    if summary["coverage"]["regime_contract_rows"] != len(regimes):
        raise SystemExit("summary regime_contract_rows does not match CSV")
    if "same_opponent_world_conditioning" not in summary["method"]:
        raise SystemExit("summary does not document same-opponent conditioning")

    claim_names = {row["claim_probe"] for row in claims}
    for required in {"four_plus_doubles", "no_trump_support", "regime_switch_proxy"}:
        if required not in claim_names:
            raise SystemExit(f"missing required claim contrast: {required}")

    print(
        json.dumps(
            {
                "ok": True,
                "paired_hands": len(paired),
                "regime_rows": len(regimes),
                "claim_contrasts": len(claims),
                "point_sample_pairs": len(samples),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
