#!/usr/bin/env python3
"""Seat/position claim tests over existing W42 legal-action rows.

The runner consumes rows that already contain public action context and offline
E[Q] labels. Hidden truth, sampled worlds, and oracle values remain labels for
reporting only; derived labels here use public/action-local columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "w42/tactical_claim_replication/all_action_rows.jsonl"
DEFAULT_OUTPUT = ROOT / "w42/seat_position_claim_tests"
BEAD_ID = "t42-0b4l.6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260503)
    parser.add_argument("--min-label-n", type=int, default=20)
    parser.add_argument("--min-paired-n", type=int, default=20)
    parser.add_argument("--min-slice-n", type=int, default=50)
    parser.add_argument("--example-limit", type=int, default=8)
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


def read_jsonl(path: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "null"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if isinstance(v, int | float) and math.isfinite(float(v))]


def mean(values: Iterable[float]) -> float:
    nums = finite(values)
    if not nums:
        return float("nan")
    return float(sum(nums) / len(nums))


def bootstrap_ci(values: Iterable[float], *, samples: int, seed: int) -> tuple[float, float, float]:
    nums = finite(values)
    center = mean(nums)
    if not nums:
        return center, float("nan"), float("nan")
    if len(nums) == 1 or samples <= 0:
        return center, center, center
    rng = random.Random(seed)
    draws: list[float] = []
    n = len(nums)
    for _ in range(samples):
        draws.append(sum(nums[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return center, float(lo), float(hi)


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def split_existing_labels(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    labels: set[str] = set()
    for part in str(value).replace(",", "|").split("|"):
        part = part.strip()
        if part:
            labels.add(part)
    return labels


def role_family(row: dict[str, Any]) -> str:
    role = str(row.get("seat_role", ""))
    if role == "bidder":
        return "bidder"
    if role == "bidder_partner":
        return "partner"
    if "setter" in role:
        return "setter"
    if str(row.get("team", "")) == "defense":
        return "defender"
    return "unknown"


def phase_label(trick_idx: int) -> str:
    if trick_idx <= 1:
        return "phase_early_trick_0_1"
    if trick_idx <= 4:
        return "phase_middle_trick_2_4"
    return "phase_late_trick_5_6"


def set_threshold_from_bid(row: dict[str, Any]) -> int:
    bid_value = row.get("bid_value")
    if bid_value in {None, ""}:
        return 13
    bid = int(float(bid_value))
    if bid >= 84:
        return 0
    return 43 - bid


def projected_defense_score(row: dict[str, Any]) -> int:
    if str(row.get("team", "")) != "defense":
        return as_int(row.get("defense_score_before"))
    if not as_bool(row.get("candidate_would_win_trick_now")):
        return as_int(row.get("defense_score_before"))
    return (
        as_int(row.get("defense_score_before"))
        + as_int(row.get("current_trick_count_before"))
        + as_int(row.get("candidate_count_points"))
        + 1
    )


def derived_labels(row: dict[str, Any]) -> tuple[str, ...]:
    labels = set(split_existing_labels(row.get("labels")))
    trick_position = as_int(row.get("trick_position"))
    trick_idx = as_int(row.get("trick_idx"))
    role = str(row.get("seat_role", ""))
    team = str(row.get("team", ""))
    count_points = as_int(row.get("candidate_count_points"))
    current_winner_team = str(row.get("current_winner_team_before", ""))
    beats_current = as_bool(row.get("candidate_beats_current"))
    would_win = as_bool(row.get("candidate_would_win_trick_now"))

    seat_labels = {
        0: "seat_first_lead",
        1: "seat_second_follow",
        2: "seat_third_follow",
        3: "seat_fourth_closure",
    }
    labels.add(seat_labels.get(trick_position, "seat_unknown"))
    labels.add(phase_label(trick_idx))

    if trick_position == 0:
        labels.add("position_lead")
    else:
        labels.add("position_follow")
    if trick_position == 3:
        labels.add("position_last_to_act")

    family = role_family(row)
    labels.add(f"role_{family}")
    if team == "defense":
        labels.add("role_defender")
    if role in {"left_setter", "right_setter"}:
        labels.add(f"role_{role}")

    if role == "bidder_partner":
        labels.add("partner_support_context")
        if count_points > 0 and current_winner_team == "offense":
            labels.add("partner_support_count_current_control")
        if count_points > 0 and current_winner_team == "defense" and not beats_current:
            labels.add("partner_unsupported_count_into_defense")
        if count_points > 0 and would_win:
            labels.add("partner_count_takes_control")

    if team == "defense" and current_winner_team == "offense":
        labels.add("defensive_pounce_window")
        if count_points > 0 and beats_current:
            labels.add("defensive_pounce_count")
            if trick_position in {1, 2}:
                labels.add("defensive_pounce_before_certainty")
            if trick_position == 3:
                labels.add("defensive_pounce_closure")
            if projected_defense_score(row) >= set_threshold_from_bid(row):
                labels.add("defensive_pounce_sets_now")
        if count_points > 0 and not would_win:
            labels.add("defensive_reckless_count_to_bidder")

    if trick_position == 3:
        labels.add("closure_decision")
        if would_win:
            labels.add("closure_take_trick")
            if count_points > 0:
                labels.add("closure_take_count")
        else:
            labels.add("closure_decline_or_slough")
            if count_points > 0:
                labels.add("closure_slough_count")

    return tuple(sorted(labels))


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        labels = derived_labels(row)
        out = {
            **row,
            "_labels": labels,
            "_role_family": role_family(row),
            "_position_family": {
                0: "first_lead",
                1: "second_follow",
                2: "third_follow",
                3: "fourth_closure",
            }.get(as_int(row.get("trick_position")), "unknown"),
            "_phase": phase_label(as_int(row.get("trick_idx"))),
        }
        enriched.append(out)
    return enriched


def decision_groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("key"))].append(row)
    return groups


def summarize_bucket(rows: list[dict[str, Any]], *, label: str | None = None) -> dict[str, Any]:
    return {
        "label": label or "",
        "action_n": len(rows),
        "decision_n": len({str(row.get("key")) for row in rows}),
        "mean": mean(as_float(row.get("mean")) for row in rows),
        "mean_regret": mean(as_float(row.get("mean_regret")) for row in rows),
        "mean_threshold_mass": mean(as_float(row.get("threshold_mass")) for row in rows),
        "mean_lower_tail_mass": mean(as_float(row.get("lower_tail_mass")) for row in rows),
        "actual_action_rate": mean(1.0 if as_bool(row.get("is_actual_action")) else 0.0 for row in rows),
        "best_mean_rate": mean(1.0 if as_bool(row.get("is_best_mean")) else 0.0 for row in rows),
        "best_threshold_rate": mean(1.0 if as_bool(row.get("is_best_threshold")) else 0.0 for row in rows),
        "safest_tail_rate": mean(1.0 if as_bool(row.get("is_safest_tail")) else 0.0 for row in rows),
    }


def label_metrics(rows: list[dict[str, Any]], *, min_label_n: int) -> list[dict[str, Any]]:
    counts = Counter(label for row in rows for label in row["_labels"])
    out: list[dict[str, Any]] = []
    for label, count in sorted(counts.items()):
        if count < min_label_n:
            continue
        selected = [row for row in rows if label in row["_labels"]]
        metric = summarize_bucket(selected, label=label)
        metric["claim_class"] = label_claim_class(label)
        out.append(metric)
    return out


def label_claim_class(label: str) -> str:
    if label.startswith(("seat_", "position_", "role_", "phase_")):
        return "slice_only_context"
    if label.startswith("partner_"):
        return "partner_support"
    if label.startswith("defensive_"):
        return "defensive_pounce"
    if label.startswith("closure_"):
        return "closure_decision"
    if label.startswith("ch04_") or label.startswith("ch05_"):
        return "upstream_tactical_label"
    return "other"


def role_position_metrics(rows: list[dict[str, Any]], *, min_slice_n: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["_role_family"], row["_position_family"], row["_phase"])].append(row)
    out: list[dict[str, Any]] = []
    for (role, position, phase), bucket in sorted(buckets.items()):
        if len(bucket) < min_slice_n:
            continue
        metric = summarize_bucket(bucket)
        metric.update({"role_family": role, "position_family": position, "phase": phase})
        out.append(metric)
    return out


def contrast_specs() -> list[dict[str, Any]]:
    return [
        {
            "contrast_id": "defensive_pounce_count_vs_other_same_decision",
            "preferred_label": "defensive_pounce_count",
            "alternative": lambda row: "defensive_pounce_count" not in row["_labels"],
        },
        {
            "contrast_id": "defensive_pounce_before_certainty_vs_other_same_decision",
            "preferred_label": "defensive_pounce_before_certainty",
            "alternative": lambda row: "defensive_pounce_before_certainty" not in row["_labels"],
        },
        {
            "contrast_id": "defensive_pounce_closure_vs_other_closure_same_decision",
            "preferred_label": "defensive_pounce_closure",
            "alternative": lambda row: "defensive_pounce_closure" not in row["_labels"],
        },
        {
            "contrast_id": "partner_support_count_vs_other_same_decision",
            "preferred_label": "partner_support_count_current_control",
            "alternative": lambda row: "partner_support_count_current_control" not in row["_labels"],
        },
        {
            "contrast_id": "partner_unsupported_count_vs_other_same_decision",
            "preferred_label": "partner_unsupported_count_into_defense",
            "alternative": lambda row: "partner_unsupported_count_into_defense" not in row["_labels"],
        },
        {
            "contrast_id": "closure_take_trick_vs_decline_same_decision",
            "preferred_label": "closure_take_trick",
            "alternative": lambda row: "closure_decline_or_slough" in row["_labels"],
        },
        {
            "contrast_id": "closure_take_count_vs_other_same_decision",
            "preferred_label": "closure_take_count",
            "alternative": lambda row: "closure_take_count" not in row["_labels"],
        },
        {
            "contrast_id": "closure_slough_count_vs_other_same_decision",
            "preferred_label": "closure_slough_count",
            "alternative": lambda row: "closure_slough_count" not in row["_labels"],
        },
    ]


def best_by_mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: as_float(row.get("mean"), default=-1e12))


def paired_for_label(
    rows: list[dict[str, Any]],
    *,
    preferred_label: str,
    alternative: Callable[[dict[str, Any]], bool],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for bucket in decision_groups(rows).values():
        preferred = [row for row in bucket if preferred_label in row["_labels"]]
        alternatives = [row for row in bucket if alternative(row)]
        if preferred and alternatives:
            pairs.append((best_by_mean(preferred), best_by_mean(alternatives)))
    return pairs


def summarize_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    contrast_id: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    mean_deltas = [as_float(pref.get("mean")) - as_float(alt.get("mean")) for pref, alt in pairs]
    threshold_deltas = [
        as_float(pref.get("threshold_mass")) - as_float(alt.get("threshold_mass")) for pref, alt in pairs
    ]
    lower_tail_deltas = [
        as_float(pref.get("lower_tail_mass")) - as_float(alt.get("lower_tail_mass")) for pref, alt in pairs
    ]
    regret_deltas = [
        as_float(pref.get("mean_regret")) - as_float(alt.get("mean_regret")) for pref, alt in pairs
    ]
    center, lo, hi = bootstrap_ci(
        mean_deltas,
        samples=bootstrap_samples,
        seed=stable_seed(bootstrap_seed, contrast_id),
    )
    return {
        "contrast_id": contrast_id,
        "paired_decision_n": len(pairs),
        "mean_delta": center,
        "mean_delta_ci95_low": lo,
        "mean_delta_ci95_high": hi,
        "regret_delta": mean(regret_deltas),
        "threshold_mass_delta": mean(threshold_deltas),
        "lower_tail_mass_delta": mean(lower_tail_deltas),
    }


def paired_contrasts(
    rows: list[dict[str, Any]],
    *,
    min_paired_n: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]]:
    out: list[dict[str, Any]] = []
    pair_map: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for spec in contrast_specs():
        pairs = paired_for_label(rows, preferred_label=spec["preferred_label"], alternative=spec["alternative"])
        pair_map[spec["contrast_id"]] = pairs
        if len(pairs) < min_paired_n:
            continue
        out.append(
            summarize_pairs(
                pairs,
                contrast_id=spec["contrast_id"],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )
    return out, pair_map


def paired_contrast_slices(
    pair_map: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    *,
    min_slice_n: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for contrast_id, pairs in pair_map.items():
        buckets: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for pref, alt in pairs:
            slices = {
                "role_family": pref["_role_family"],
                "position_family": pref["_position_family"],
                "phase": pref["_phase"],
                "seat_role": str(pref.get("seat_role", "")),
                "decl": str(pref.get("decl_name", "")),
                "trick_idx": str(pref.get("trick_idx", "")),
                "count_points": str(pref.get("candidate_count_points", "")),
            }
            for name, value in slices.items():
                buckets[(name, value)].append((pref, alt))
        for (slice_name, slice_value), bucket in sorted(buckets.items()):
            if len(bucket) < min_slice_n:
                continue
            row = summarize_pairs(
                bucket,
                contrast_id=f"{contrast_id}__{slice_name}={slice_value}",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            row.update(
                {
                    "base_contrast_id": contrast_id,
                    "slice_name": slice_name,
                    "slice_value": slice_value,
                }
            )
            out.append(row)
    return out


def compact_action_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": row.get("key"),
        "source_file": row.get("source_file"),
        "seed": row.get("seed"),
        "game_idx": row.get("game_idx"),
        "decision_idx": row.get("decision_idx"),
        "decl_name": row.get("decl_name"),
        "actor": row.get("actor"),
        "seat_role": row.get("seat_role"),
        "role_family": row.get("_role_family"),
        "team": row.get("team"),
        "trick_idx": row.get("trick_idx"),
        "trick_position": row.get("trick_position"),
        "position_family": row.get("_position_family"),
        "phase": row.get("_phase"),
        "candidate_domino": row.get("candidate_domino"),
        "candidate_count_points": row.get("candidate_count_points"),
        "current_winner_team_before": row.get("current_winner_team_before"),
        "candidate_beats_current": row.get("candidate_beats_current"),
        "candidate_would_win_trick_now": row.get("candidate_would_win_trick_now"),
        "mean": row.get("mean"),
        "mean_regret": row.get("mean_regret"),
        "threshold_mass": row.get("threshold_mass"),
        "lower_tail_mass": row.get("lower_tail_mass"),
        "is_actual_action": row.get("is_actual_action"),
        "is_best_mean": row.get("is_best_mean"),
        "derived_labels": "|".join(row["_labels"]),
    }


def build_examples(
    pair_map: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    *,
    example_limit: int,
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {}
    for contrast_id, pairs in pair_map.items():
        ranked = sorted(
            pairs,
            key=lambda pair: abs(as_float(pair[0].get("mean")) - as_float(pair[1].get("mean"))),
            reverse=True,
        )[:example_limit]
        examples[contrast_id] = [
            {
                "key": pref.get("key"),
                "mean_delta": as_float(pref.get("mean")) - as_float(alt.get("mean")),
                "preferred": compact_action_row(pref),
                "alternative": compact_action_row(alt),
            }
            for pref, alt in ranked
        ]
    return examples


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_jsonl(args.input, max_rows=args.max_rows)
    rows = enrich_rows(raw_rows)
    label_metric_rows = label_metrics(rows, min_label_n=args.min_label_n)
    role_position_rows = role_position_metrics(rows, min_slice_n=args.min_slice_n)
    contrast_rows, pair_map = paired_contrasts(
        rows,
        min_paired_n=args.min_paired_n,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    contrast_slice_rows = paired_contrast_slices(
        pair_map,
        min_slice_n=args.min_slice_n,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    label_counts = Counter(label for row in rows for label in row["_labels"])
    same_context_labels = [
        "seat_first_lead",
        "seat_second_follow",
        "seat_third_follow",
        "seat_fourth_closure",
        "position_lead",
        "position_follow",
        "position_last_to_act",
        "role_bidder",
        "role_partner",
        "role_setter",
        "role_defender",
    ]
    summary = {
        "schema_version": "w42.seat_position_claim_tests.v1",
        "bead": BEAD_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha(),
        "git_status_before_artifacts": git_status_short(),
        "input": str(args.input),
        "max_rows": args.max_rows,
        "coverage": {
            "action_rows": len(rows),
            "decision_rows": len({str(row.get("key")) for row in rows}),
            "label_metric_rows": len(label_metric_rows),
            "role_position_metric_rows": len(role_position_rows),
            "paired_contrast_rows": len(contrast_rows),
            "paired_contrast_slice_rows": len(contrast_slice_rows),
            "top_label_counts": dict(label_counts.most_common(30)),
        },
        "unpaired_slice_only_labels": same_context_labels,
        "pairing_note": (
            "Seat, phase, role, and trick-position labels are decision-context slices: "
            "all legal actions in the same decision share them, so same-decision paired "
            "contrasts are not meaningful for those labels. Action-local support, pounce, "
            "and closure labels are paired against same-decision alternatives when present."
        ),
        "leakage_boundary": (
            "Derived labels use public/action-local columns from the row artifact. Oracle mean, "
            "threshold mass, lower-tail mass, q_per_world, world_hands, and hidden outcomes are "
            "offline report labels only, never model features."
        ),
        "scientific_status": (
            "Powered row-level claim test over existing W42 tactical legal-action rows. "
            "Structural seat/role/phase findings are slice evidence; action-local partner, "
            "defensive pounce, and closure claims have paired same-decision contrasts."
        ),
    }

    label_fields = [
        "label",
        "claim_class",
        "action_n",
        "decision_n",
        "mean",
        "mean_regret",
        "mean_threshold_mass",
        "mean_lower_tail_mass",
        "actual_action_rate",
        "best_mean_rate",
        "best_threshold_rate",
        "safest_tail_rate",
    ]
    role_fields = [
        "role_family",
        "position_family",
        "phase",
        "action_n",
        "decision_n",
        "mean",
        "mean_regret",
        "mean_threshold_mass",
        "mean_lower_tail_mass",
        "actual_action_rate",
        "best_mean_rate",
        "best_threshold_rate",
        "safest_tail_rate",
    ]
    contrast_fields = [
        "contrast_id",
        "paired_decision_n",
        "mean_delta",
        "mean_delta_ci95_low",
        "mean_delta_ci95_high",
        "regret_delta",
        "threshold_mass_delta",
        "lower_tail_mass_delta",
    ]
    contrast_slice_fields = ["base_contrast_id", *contrast_fields, "slice_name", "slice_value"]
    action_fields = list(compact_action_row(rows[0]).keys()) if rows else []

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "label_metrics.csv", label_metric_rows, label_fields)
    write_csv(output_dir / "role_position_metrics.csv", role_position_rows, role_fields)
    write_csv(output_dir / "paired_contrasts.csv", contrast_rows, contrast_fields)
    write_csv(output_dir / "paired_contrasts_by_slice.csv", contrast_slice_rows, contrast_slice_fields)
    write_csv(output_dir / "labeled_action_rows.csv", [compact_action_row(row) for row in rows], action_fields)
    write_json(output_dir / "examples.json", build_examples(pair_map, example_limit=args.example_limit))
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "w42.seat_position_claim_tests.manifest.v1",
            "bead": BEAD_ID,
            "created_at_utc": summary["created_at_utc"],
            "command": " ".join(sys.argv),
            "input": str(args.input),
            "outputs": {
                "summary": str(output_dir / "summary.json"),
                "label_metrics": str(output_dir / "label_metrics.csv"),
                "role_position_metrics": str(output_dir / "role_position_metrics.csv"),
                "paired_contrasts": str(output_dir / "paired_contrasts.csv"),
                "paired_contrasts_by_slice": str(output_dir / "paired_contrasts_by_slice.csv"),
                "labeled_action_rows": str(output_dir / "labeled_action_rows.csv"),
                "examples": str(output_dir / "examples.json"),
                "manifest": str(output_dir / "manifest.json"),
            },
            "leakage_boundary": summary["leakage_boundary"],
        },
    )
    print(json.dumps({"coverage": summary["coverage"], "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
