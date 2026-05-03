#!/usr/bin/env python3
"""Direct claim-tag probe over W42 legal-action rows.

This is intentionally a small row-model probe, not a Gus replacement. It asks
whether public/action-local detector tags from the claim-test artifacts help a
cheap model rank legal actions inside the same decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "w42/seat_position_claim_tests/labeled_action_rows.csv"
DEFAULT_OUTPUT = ROOT / "w42/claim_tag_model_probe"
BEAD_ID = "t42-0b4l.9"

NUMERIC_FIELDS = (
    "actor",
    "trick_idx",
    "trick_position",
    "candidate_count_points",
)
CATEGORICAL_FIELDS = (
    "decl_name",
    "seat_role",
    "role_family",
    "team",
    "position_family",
    "phase",
    "current_winner_team_before",
    "candidate_domino",
    "candidate_beats_current",
    "candidate_would_win_trick_now",
)

TAG_FAMILIES = {
    "pounce_donation": (
        "ch04_",
        "ch05_",
        "partner_",
        "defensive_",
    ),
    "seat_position_closure": (
        "seat_",
        "position_",
        "role_",
        "phase_",
        "closure_",
    ),
}

UNAVAILABLE_FAMILIES = {
    "bid_risk": "available as t42-0b4l.5 bid-margin counterfactual rows, but not aligned to the tactical legal-action training table",
    "eighty_four_endgame": "available as a small generated 84 fixture table, but too small and out-of-distribution for this train split",
    "doubles_no_trump": "available in t42-0b4l.8 legacy mining, but not normalized into this row model input",
    "hidden_threat_distribution": "offline labels only; hidden truth and q_per_world are not legal live model features",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-seed-max", type=int, default=79)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--c", type=float, default=0.8)
    parser.add_argument("--example-limit", type=int, default=20)
    return parser.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        return float(text)
    except Exception:
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    labels = []
    for part in str(value).replace(",", "|").split("|"):
        part = part.strip()
        if part:
            labels.append(part)
    return tuple(sorted(set(labels)))


def label_family(label: str) -> str | None:
    for family, prefixes in TAG_FAMILIES.items():
        if label.startswith(prefixes):
            return family
    return None


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["_seed"] = int(float(row["seed"]))
        row["_labels"] = split_labels(row.get("derived_labels"))
        row["_is_best_mean"] = as_bool(row.get("is_best_mean"))
        row["_mean_regret"] = as_float(row.get("mean_regret"))
        row["_mean"] = as_float(row.get("mean"))
    return rows


def groups_by_decision(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["key"])].append(row)
    return groups


def feature_dict(row: dict[str, Any], *, include_tags: bool, drop_family: str | None = None) -> dict[str, float]:
    feats: dict[str, float] = {}
    for field in NUMERIC_FIELDS:
        value = as_float(row.get(field))
        if math.isfinite(value):
            feats[f"num:{field}"] = value
    for field in CATEGORICAL_FIELDS:
        feats[f"cat:{field}={row.get(field, '')}"] = 1.0
    if include_tags:
        for label in row["_labels"]:
            family = label_family(label)
            if family is None:
                continue
            if drop_family and family == drop_family:
                continue
            feats[f"tag:{label}"] = 1.0
            feats[f"tag_family:{family}"] = 1.0
    return feats


def train_model(rows: list[dict[str, Any]], *, include_tags: bool, drop_family: str | None, args: argparse.Namespace):
    x_dicts = [feature_dict(row, include_tags=include_tags, drop_family=drop_family) for row in rows]
    y = np.asarray([1 if row["_is_best_mean"] else 0 for row in rows], dtype=np.int64)
    model = make_pipeline(
        DictVectorizer(sparse=True),
        MaxAbsScaler(),
        LogisticRegression(
            C=args.c,
            class_weight="balanced",
            max_iter=args.max_iter,
            solver="liblinear",
            random_state=20260503,
        ),
    )
    model.fit(x_dicts, y)
    return model


def evaluate_model(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    include_tags: bool,
    drop_family: str | None,
    variant: str,
    example_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x_dicts = [feature_dict(row, include_tags=include_tags, drop_family=drop_family) for row in rows]
    probs = model.predict_proba(x_dicts)[:, 1]
    for row, prob in zip(rows, probs, strict=True):
        row[f"_score_{variant}"] = float(prob)

    chosen: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for key, bucket in groups_by_decision(rows).items():
        best = max(bucket, key=lambda row: row[f"_score_{variant}"])
        chosen.append(best)
        if len(examples) < example_limit and not best["_is_best_mean"]:
            oracle = min(bucket, key=lambda row: row["_mean_regret"])
            examples.append(
                {
                    "key": key,
                    "variant": variant,
                    "chosen_domino": best.get("candidate_domino"),
                    "chosen_regret": round(best["_mean_regret"], 6),
                    "chosen_labels": "|".join(best["_labels"]),
                    "oracle_domino": oracle.get("candidate_domino"),
                    "oracle_labels": "|".join(oracle["_labels"]),
                    "decl_name": best.get("decl_name"),
                    "seat_role": best.get("seat_role"),
                    "trick_position": best.get("trick_position"),
                }
            )

    regrets = [row["_mean_regret"] for row in chosen]
    metrics = {
        "variant": variant,
        "decision_n": len(chosen),
        "action_n": len(rows),
        "match_best_mean_rate": round(sum(1 for row in chosen if row["_is_best_mean"]) / len(chosen), 6),
        "mean_regret": round(float(np.mean(regrets)), 6),
        "median_regret": round(float(np.median(regrets)), 6),
        "near_tie_rate_regret_lt_0_5": round(sum(1 for value in regrets if value < 0.5) / len(regrets), 6),
        "tail_regret_rate_ge_5": round(sum(1 for value in regrets if value >= 5.0) / len(regrets), 6),
    }
    return metrics, examples


def actual_policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = []
    for bucket in groups_by_decision(rows).values():
        actual = [row for row in bucket if as_bool(row.get("is_actual_action"))]
        if actual:
            chosen.append(actual[0])
    regrets = [row["_mean_regret"] for row in chosen]
    return {
        "variant": "actual_policy",
        "decision_n": len(chosen),
        "action_n": len(rows),
        "match_best_mean_rate": round(sum(1 for row in chosen if row["_is_best_mean"]) / len(chosen), 6),
        "mean_regret": round(float(np.mean(regrets)), 6),
        "median_regret": round(float(np.median(regrets)), 6),
        "near_tie_rate_regret_lt_0_5": round(sum(1 for value in regrets if value < 0.5) / len(regrets), 6),
        "tail_regret_rate_ge_5": round(sum(1 for value in regrets if value >= 5.0) / len(regrets), 6),
    }


def tag_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(label for row in rows for label in row["_labels"])
    family_counts = Counter()
    for label, count in counts.items():
        family = label_family(label)
        if family:
            family_counts[family] += count
    out = [{"family": family, "tag_action_n": count, "status": "available"} for family, count in sorted(family_counts.items())]
    out.extend(
        {
            "family": family,
            "tag_action_n": 0,
            "status": f"not_in_training_table: {reason}",
        }
        for family, reason in sorted(UNAVAILABLE_FAMILIES.items())
    )
    return out


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    rows = load_rows(args.input)
    train_rows = [row for row in rows if row["_seed"] <= args.train_seed_max]
    eval_rows = [row.copy() for row in rows if row["_seed"] > args.train_seed_max]
    if not train_rows or not eval_rows:
        raise SystemExit("empty train/eval split")

    variants: list[dict[str, Any]] = [
        {"variant": "public_features_only", "include_tags": False, "drop_family": None},
        {"variant": "public_plus_claim_tags", "include_tags": True, "drop_family": None},
    ]
    variants.extend(
        {
            "variant": f"drop_{family}",
            "include_tags": True,
            "drop_family": family,
        }
        for family in sorted(TAG_FAMILIES)
    )

    metrics = [actual_policy_metrics(eval_rows)]
    examples: list[dict[str, Any]] = []
    for spec in variants:
        model = train_model(
            train_rows,
            include_tags=spec["include_tags"],
            drop_family=spec["drop_family"],
            args=args,
        )
        variant_metrics, variant_examples = evaluate_model(
            model,
            [row.copy() for row in eval_rows],
            include_tags=spec["include_tags"],
            drop_family=spec["drop_family"],
            variant=spec["variant"],
            example_limit=args.example_limit,
        )
        metrics.append(variant_metrics)
        examples.extend(variant_examples)

    baseline = next(row for row in metrics if row["variant"] == "public_features_only")
    full = next(row for row in metrics if row["variant"] == "public_plus_claim_tags")
    for row in metrics:
        if row["variant"] in {"actual_policy", "public_features_only"}:
            row["delta_vs_public_mean_regret"] = 0.0 if row["variant"] == "public_features_only" else None
            row["delta_vs_full_tags_mean_regret"] = None
            continue
        row["delta_vs_public_mean_regret"] = round(row["mean_regret"] - baseline["mean_regret"], 6)
        row["delta_vs_full_tags_mean_regret"] = round(row["mean_regret"] - full["mean_regret"], 6)

    inventory = tag_inventory(rows)
    summary = {
        "schema_version": "w42.claim_tag_model_probe.v1",
        "bead": BEAD_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "git_sha": git_sha(),
        "git_status_before_artifacts": git_status_short(),
        "input": str(args.input.relative_to(ROOT)),
        "split": {
            "train_seed_max": args.train_seed_max,
            "train_action_rows": len(train_rows),
            "train_decisions": len(groups_by_decision(train_rows)),
            "eval_action_rows": len(eval_rows),
            "eval_decisions": len(groups_by_decision(eval_rows)),
        },
        "headline": {
            "public_mean_regret": baseline["mean_regret"],
            "tag_mean_regret": full["mean_regret"],
            "tag_delta_vs_public_mean_regret": round(full["mean_regret"] - baseline["mean_regret"], 6),
            "public_match_rate": baseline["match_best_mean_rate"],
            "tag_match_rate": full["match_best_mean_rate"],
        },
        "scientific_status": {
            "claim_ledger_impact": "no central claim status movement",
            "interpretation": "Small direct row-model probe over public/action-local W42 detector tags; useful as model-feature evidence, not claim proof.",
            "leakage_boundary": "Inputs are public row context, candidate action facts, and derived public/action-local detector tags. Oracle mean/regret labels are targets/metrics only.",
            "blocked_families": UNAVAILABLE_FAMILIES,
        },
        "artifacts": {
            "metrics": "w42/claim_tag_model_probe/model_metrics.csv",
            "tag_inventory": "w42/claim_tag_model_probe/tag_inventory.csv",
            "examples": "w42/claim_tag_model_probe/error_examples.json",
            "manifest": "w42/claim_tag_model_probe/manifest.json",
        },
    }
    manifest = {
        "schema_version": "w42.claim_tag_model_probe.manifest.v1",
        "summary": summary,
        "config": {
            "numeric_fields": NUMERIC_FIELDS,
            "categorical_fields": CATEGORICAL_FIELDS,
            "tag_families": TAG_FAMILIES,
            "logistic_regression": {"C": args.c, "max_iter": args.max_iter, "class_weight": "balanced"},
        },
    }

    write_csv(args.output_dir / "model_metrics.csv", metrics)
    write_csv(args.output_dir / "tag_inventory.csv", inventory)
    write_json(args.output_dir / "error_examples.json", examples[: args.example_limit])
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(summary["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
