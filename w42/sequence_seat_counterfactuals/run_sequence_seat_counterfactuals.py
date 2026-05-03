#!/usr/bin/env python3
"""Generated sequence/seat counterfactuals for W42 book-claim tests.

The input legal-action rows already contain offline E[Q]-style branch labels for
each candidate action in a naturally occurring public sequence state. This
runner composes those rows into book-shaped contrasts: lead plans, follow-seat
obligations, last-seat closure, partner support timing, and setter pounce
windows.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from w42.wandb_utils import add_wandb_args, init_wandb

DEFAULT_INPUT = ROOT / "w42/tactical_claim_replication/all_action_rows.jsonl"
DEFAULT_OUTPUT = ROOT / "w42/sequence_seat_counterfactuals"
BEAD_ID = "t42-qtwb.3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260503)
    parser.add_argument("--min-label-n", type=int, default=20)
    parser.add_argument("--min-paired-n", type=int, default=20)
    parser.add_argument("--min-slice-n", type=int, default=20)
    parser.add_argument("--example-limit", type=int, default=8)
    add_wandb_args(parser, default_group="w42-sequence-seat-counterfactuals", default_enabled=True)
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
    if value in {None, ""}:
        return default
    return int(float(value))


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if isinstance(v, int | float) and math.isfinite(float(v))]


def mean(values: Iterable[float]) -> float:
    nums = finite(values)
    if not nums:
        return float("nan")
    return float(sum(nums) / len(nums))


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


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


def phase_label(trick_idx: int) -> str:
    if trick_idx <= 1:
        return "early"
    if trick_idx <= 4:
        return "middle"
    return "late"


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


def team_opponent(team: str) -> str:
    if team == "offense":
        return "defense"
    if team == "defense":
        return "offense"
    return ""


def split_existing_labels(value: Any) -> set[str]:
    if value is None:
        return set()
    labels: set[str] = set()
    if isinstance(value, list):
        parts = value
    else:
        parts = str(value).replace(",", "|").split("|")
    for part in parts:
        text = str(part).strip()
        if text:
            labels.add(text)
    return labels


def decision_groups(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("key"))].append(row)
    return groups


def decision_features(bucket: list[dict[str, Any]]) -> dict[str, Any]:
    first = bucket[0]
    team = str(first.get("team", ""))
    current_winner_team = str(first.get("current_winner_team_before", ""))
    return {
        "has_called_suit": any(as_bool(row.get("candidate_is_called_suit")) for row in bucket),
        "has_off_suit": any(not as_bool(row.get("candidate_is_called_suit")) for row in bucket),
        "has_count": any(as_int(row.get("candidate_count_points")) > 0 for row in bucket),
        "has_noncount": any(as_int(row.get("candidate_count_points")) == 0 for row in bucket),
        "can_beat_current": any(as_bool(row.get("candidate_beats_current")) for row in bucket),
        "can_take_trick": any(as_bool(row.get("candidate_would_win_trick_now")) for row in bucket),
        "current_winner_is_us": current_winner_team == team,
        "current_winner_is_them": current_winner_team == team_opponent(team),
        "legal_action_n": len(bucket),
    }


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


def row_labels(row: dict[str, Any], features: dict[str, Any]) -> tuple[str, ...]:
    labels = set(split_existing_labels(row.get("labels")))
    role = str(row.get("seat_role", ""))
    team = str(row.get("team", ""))
    current_winner_team = str(row.get("current_winner_team_before", ""))
    trick_position = as_int(row.get("trick_position"))
    trick_idx = as_int(row.get("trick_idx"))
    count_points = as_int(row.get("candidate_count_points"))
    called_suit = as_bool(row.get("candidate_is_called_suit"))
    is_double = as_bool(row.get("candidate_is_double"))
    beats_current = as_bool(row.get("candidate_beats_current"))
    would_win = as_bool(row.get("candidate_would_win_trick_now"))

    labels.add(f"phase_{phase_label(trick_idx)}")
    labels.add(f"role_{role_family(row)}")
    labels.add(f"seat_pos_{trick_position}")
    if trick_position == 0:
        labels.add("sequence_lead")
    elif trick_position == 3:
        labels.add("sequence_last_to_act")
    else:
        labels.add("sequence_follow")

    if role == "bidder" and trick_position == 0:
        labels.add("bidder_lead_plan")
        if called_suit:
            labels.add("bidder_lead_called_suit")
            if is_double:
                labels.add("bidder_lead_called_double")
            else:
                labels.add("bidder_lead_called_non_double")
        else:
            labels.add("bidder_lead_off_suit")
        if count_points > 0:
            labels.add("bidder_lead_count")
            if not called_suit:
                labels.add("bidder_lead_off_count")
        else:
            labels.add("bidder_lead_noncount")

    if trick_position > 0 and features["current_winner_is_them"]:
        labels.add("follower_opponent_currently_winning")
        if beats_current:
            labels.add("follower_can_beat_opponent")
        if would_win:
            labels.add("follower_take_control_from_opponent")
        elif count_points > 0:
            labels.add("follower_count_into_opponent_control")

    if trick_position > 0 and features["current_winner_is_us"]:
        labels.add("follower_partner_or_self_currently_winning")
        if count_points > 0 and not beats_current:
            labels.add("follower_safe_count_to_current_control")

    if role == "bidder_partner":
        labels.add("partner_support_sequence_context")
        if current_winner_team == "offense" and count_points > 0:
            labels.add("partner_count_when_bidder_side_controls")
            labels.add(f"partner_count_timing_{phase_label(trick_idx)}")
        if current_winner_team == "defense" and count_points > 0 and not beats_current:
            labels.add("partner_count_into_defense_control")

    if team == "defense" and current_winner_team == "offense":
        labels.add("setter_pounce_sequence_window")
        if count_points > 0 and beats_current:
            labels.add("setter_pounce_count")
            if trick_position in {1, 2}:
                labels.add("setter_pounce_before_last_seat")
            if trick_position == 3:
                labels.add("setter_pounce_last_seat")
            if projected_defense_score(row) >= set_threshold_from_bid(row):
                labels.add("setter_pounce_sets_now")
        if count_points > 0 and not would_win:
            labels.add("setter_reckless_count_to_bidder")

    if trick_position == 3:
        labels.add("closure_sequence_decision")
        if would_win:
            labels.add("closure_take_trick")
            if count_points > 0:
                labels.add("closure_take_count")
        else:
            labels.add("closure_decline_trick")
            if count_points > 0:
                labels.add("closure_slough_count")

    return tuple(sorted(labels))


def enrich_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for bucket in decision_groups(rows).values():
        features = decision_features(bucket)
        for row in bucket:
            out = dict(row)
            out["_labels"] = row_labels(row, features)
            out["_phase"] = phase_label(as_int(row.get("trick_idx")))
            out["_role_family"] = role_family(row)
            out["_position_family"] = {
                0: "lead",
                1: "second",
                2: "third",
                3: "last",
            }.get(as_int(row.get("trick_position")), "unknown")
            out["_decision_has_called_and_off"] = features["has_called_suit"] and features["has_off_suit"]
            out["_decision_has_count_and_noncount"] = features["has_count"] and features["has_noncount"]
            out["_decision_legal_action_n"] = features["legal_action_n"]
            enriched.append(out)
    return enriched


def best_by_mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: as_float(row.get("mean"), default=-1e12))


def worst_by_mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=lambda row: as_float(row.get("mean"), default=1e12))


def has_label(label: str) -> Callable[[dict[str, Any]], bool]:
    return lambda row: label in row["_labels"]


def lacks_label(label: str) -> Callable[[dict[str, Any]], bool]:
    return lambda row: label not in row["_labels"]


def contrast_specs() -> list[dict[str, Any]]:
    return [
        {
            "contrast_id": "bidder_lead_called_suit_vs_off_suit",
            "family": "lead_plan",
            "book_question": "When bidder is on lead, does pulling/using the called suit beat an off-suit lead in the same public state?",
            "preferred": has_label("bidder_lead_called_suit"),
            "alternative": has_label("bidder_lead_off_suit"),
        },
        {
            "contrast_id": "bidder_lead_called_double_vs_called_non_double",
            "family": "lead_plan",
            "book_question": "When both are legal called-suit leads, is the double lead better than a lower called-suit lead?",
            "preferred": has_label("bidder_lead_called_double"),
            "alternative": has_label("bidder_lead_called_non_double"),
        },
        {
            "contrast_id": "bidder_low_called_suit_vs_called_double",
            "family": "lead_plan_exception",
            "book_question": "Does the low-trump-first exception show up against leading the double?",
            "preferred": has_label("bidder_lead_called_non_double"),
            "alternative": has_label("bidder_lead_called_double"),
        },
        {
            "contrast_id": "bidder_lead_count_vs_noncount",
            "family": "count_inventory",
            "book_question": "Is leading count from the bidder side better than leading non-count in the same lead state?",
            "preferred": has_label("bidder_lead_count"),
            "alternative": has_label("bidder_lead_noncount"),
        },
        {
            "contrast_id": "follower_beat_opponent_vs_decline",
            "family": "follow_obligation",
            "book_question": "When an opponent is winning the trick and a follower can beat them, is taking control better than declining?",
            "preferred": has_label("follower_take_control_from_opponent"),
            "alternative": lambda row: "follower_opponent_currently_winning" in row["_labels"]
            and "follower_take_control_from_opponent" not in row["_labels"],
        },
        {
            "contrast_id": "follower_count_into_opponent_vs_noncount",
            "family": "follow_obligation",
            "book_question": "Is dumping count while the opponent controls the trick punished relative to non-count alternatives?",
            "preferred": has_label("follower_count_into_opponent_control"),
            "alternative": lambda row: "follower_opponent_currently_winning" in row["_labels"]
            and as_int(row.get("candidate_count_points")) == 0,
        },
        {
            "contrast_id": "partner_count_when_bidder_side_controls_vs_other",
            "family": "partner_support_timing",
            "book_question": "When bidder's side controls the trick, does partner count donation beat other same-state actions?",
            "preferred": has_label("partner_count_when_bidder_side_controls"),
            "alternative": lacks_label("partner_count_when_bidder_side_controls"),
        },
        {
            "contrast_id": "partner_count_into_defense_vs_other",
            "family": "partner_support_timing",
            "book_question": "When defense controls the trick, is partner count donation into that control punished?",
            "preferred": has_label("partner_count_into_defense_control"),
            "alternative": lacks_label("partner_count_into_defense_control"),
        },
        {
            "contrast_id": "setter_pounce_count_vs_other",
            "family": "setter_pounce",
            "book_question": "When the bidder side is winning, does a setter pounce with count beat other same-state actions?",
            "preferred": has_label("setter_pounce_count"),
            "alternative": lacks_label("setter_pounce_count"),
        },
        {
            "contrast_id": "setter_pounce_sets_now_vs_other",
            "family": "setter_pounce",
            "book_question": "When pouncing count sets the bid immediately, does it dominate other same-state actions?",
            "preferred": has_label("setter_pounce_sets_now"),
            "alternative": lacks_label("setter_pounce_sets_now"),
        },
        {
            "contrast_id": "setter_reckless_count_to_bidder_vs_other",
            "family": "setter_pounce_negative_control",
            "book_question": "Is setter count that fails to win the trick punished as the book warns?",
            "preferred": has_label("setter_reckless_count_to_bidder"),
            "alternative": lacks_label("setter_reckless_count_to_bidder"),
        },
        {
            "contrast_id": "closure_take_trick_vs_decline",
            "family": "last_seat_closure",
            "book_question": "From last seat, does taking the trick beat declining when both are available?",
            "preferred": has_label("closure_take_trick"),
            "alternative": has_label("closure_decline_trick"),
        },
        {
            "contrast_id": "closure_take_count_vs_other",
            "family": "last_seat_closure",
            "book_question": "From last seat, does taking count beat other same-state closure options?",
            "preferred": has_label("closure_take_count"),
            "alternative": lacks_label("closure_take_count"),
        },
        {
            "contrast_id": "closure_slough_count_vs_other",
            "family": "last_seat_closure_negative_control",
            "book_question": "From last seat, is sloughing count while declining the trick punished?",
            "preferred": has_label("closure_slough_count"),
            "alternative": lacks_label("closure_slough_count"),
        },
    ]


def paired_rows(
    rows: list[dict[str, Any]],
    *,
    preferred: Callable[[dict[str, Any]], bool],
    alternative: Callable[[dict[str, Any]], bool],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for bucket in decision_groups(rows).values():
        preferred_rows = [row for row in bucket if preferred(row)]
        alternative_rows = [row for row in bucket if alternative(row)]
        if preferred_rows and alternative_rows:
            pairs.append((best_by_mean(preferred_rows), best_by_mean(alternative_rows)))
    return pairs


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


def summarize_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    spec: dict[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    mean_deltas = [as_float(pref.get("mean")) - as_float(alt.get("mean")) for pref, alt in pairs]
    regret_deltas = [
        as_float(pref.get("mean_regret")) - as_float(alt.get("mean_regret")) for pref, alt in pairs
    ]
    threshold_deltas = [
        as_float(pref.get("threshold_mass")) - as_float(alt.get("threshold_mass")) for pref, alt in pairs
    ]
    lower_tail_deltas = [
        as_float(pref.get("lower_tail_mass")) - as_float(alt.get("lower_tail_mass")) for pref, alt in pairs
    ]
    center, lo, hi = bootstrap_ci(
        mean_deltas,
        samples=bootstrap_samples,
        seed=stable_seed(bootstrap_seed, str(spec["contrast_id"])),
    )
    return {
        "contrast_id": spec["contrast_id"],
        "family": spec["family"],
        "book_question": spec["book_question"],
        "paired_decision_n": len(pairs),
        "mean_delta": center,
        "mean_delta_ci95_low": lo,
        "mean_delta_ci95_high": hi,
        "regret_delta": mean(regret_deltas),
        "threshold_mass_delta": mean(threshold_deltas),
        "lower_tail_mass_delta": mean(lower_tail_deltas),
        "preferred_actual_action_rate": mean(1.0 if as_bool(pref.get("is_actual_action")) else 0.0 for pref, _ in pairs),
        "alternative_actual_action_rate": mean(1.0 if as_bool(alt.get("is_actual_action")) else 0.0 for _, alt in pairs),
    }


def label_metrics(rows: list[dict[str, Any]], *, min_label_n: int) -> list[dict[str, Any]]:
    counts = Counter(label for row in rows for label in row["_labels"])
    out: list[dict[str, Any]] = []
    for label, count in sorted(counts.items()):
        if count < min_label_n:
            continue
        selected = [row for row in rows if label in row["_labels"]]
        out.append(summarize_bucket(selected, label=label))
    return out


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
        pairs = paired_rows(rows, preferred=spec["preferred"], alternative=spec["alternative"])
        pair_map[str(spec["contrast_id"])] = pairs
        if len(pairs) < min_paired_n:
            continue
        out.append(
            summarize_pairs(
                pairs,
                spec=spec,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )
    return out, pair_map


def pair_slices(pref: dict[str, Any]) -> dict[str, str]:
    return {
        "family_role": pref["_role_family"],
        "position": pref["_position_family"],
        "phase": pref["_phase"],
        "seat_role": str(pref.get("seat_role", "")),
        "decl": str(pref.get("decl_name", "")),
        "trick_idx": str(pref.get("trick_idx", "")),
        "trick_position": str(pref.get("trick_position", "")),
        "count_points": str(pref.get("candidate_count_points", "")),
        "current_winner_team": str(pref.get("current_winner_team_before", "")),
    }


def paired_contrast_slices(
    pair_map: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    *,
    min_slice_n: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    specs = {str(spec["contrast_id"]): spec for spec in contrast_specs()}
    out: list[dict[str, Any]] = []
    for contrast_id, pairs in pair_map.items():
        spec = specs[contrast_id]
        buckets: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for pref, alt in pairs:
            for name, value in pair_slices(pref).items():
                buckets[(name, value)].append((pref, alt))
        for (slice_name, slice_value), bucket in sorted(buckets.items()):
            if len(bucket) < min_slice_n:
                continue
            row = summarize_pairs(
                bucket,
                spec={**spec, "contrast_id": f"{contrast_id}__{slice_name}={slice_value}"},
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


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": row.get("key"),
        "source_file": row.get("source_file"),
        "seed": row.get("seed"),
        "game_idx": row.get("game_idx"),
        "decision_idx": row.get("decision_idx"),
        "decl_name": row.get("decl_name"),
        "bid_value": row.get("bid_value"),
        "actor": row.get("actor"),
        "seat_role": row.get("seat_role"),
        "team": row.get("team"),
        "trick_idx": row.get("trick_idx"),
        "trick_position": row.get("trick_position"),
        "candidate_domino": row.get("candidate_domino"),
        "candidate_count_points": row.get("candidate_count_points"),
        "candidate_is_called_suit": row.get("candidate_is_called_suit"),
        "candidate_is_double": row.get("candidate_is_double"),
        "current_trick_count_before": row.get("current_trick_count_before"),
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


def human_example(contrast_id: str, pref: dict[str, Any], alt: dict[str, Any]) -> str:
    delta = as_float(pref.get("mean")) - as_float(alt.get("mean"))
    seat = pref.get("seat_role")
    decl = pref.get("decl_name")
    trick_idx = pref.get("trick_idx")
    position = pref.get("trick_position")
    current = pref.get("current_winner_team_before") or "none"
    return (
        f"{contrast_id}: seed {pref.get('seed')} game {pref.get('game_idx')} decision {pref.get('decision_idx')}, "
        f"{decl}, {seat}, trick {trick_idx} seat-position {position}, current winner {current}. "
        f"{pref.get('candidate_domino')} over {alt.get('candidate_domino')} changes mean by {delta:.3f}."
    )


def build_examples(
    pair_map: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    *,
    example_limit: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    examples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for contrast_id, pairs in pair_map.items():
        ranked_positive = sorted(
            pairs,
            key=lambda pair: as_float(pair[0].get("mean")) - as_float(pair[1].get("mean")),
            reverse=True,
        )[:example_limit]
        ranked_negative = sorted(
            pairs,
            key=lambda pair: as_float(pair[0].get("mean")) - as_float(pair[1].get("mean")),
        )[:example_limit]
        examples[contrast_id] = {
            "positive_examples": [
                {
                    "human_readable": human_example(contrast_id, pref, alt),
                    "mean_delta": as_float(pref.get("mean")) - as_float(alt.get("mean")),
                    "preferred": compact_row(pref),
                    "alternative": compact_row(alt),
                }
                for pref, alt in ranked_positive
            ],
            "negative_examples": [
                {
                    "human_readable": human_example(contrast_id, pref, alt),
                    "mean_delta": as_float(pref.get("mean")) - as_float(alt.get("mean")),
                    "preferred": compact_row(pref),
                    "alternative": compact_row(alt),
                }
                for pref, alt in ranked_negative
            ],
        }
    return examples


def write_compact_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "key",
        "source_file",
        "seed",
        "game_idx",
        "decision_idx",
        "decl_name",
        "bid_value",
        "actor",
        "seat_role",
        "role_family",
        "team",
        "trick_idx",
        "trick_position",
        "position_family",
        "phase",
        "candidate_domino",
        "candidate_count_points",
        "candidate_is_called_suit",
        "candidate_is_double",
        "current_winner_team_before",
        "candidate_beats_current",
        "candidate_would_win_trick_now",
        "mean",
        "mean_regret",
        "threshold_mass",
        "lower_tail_mass",
        "is_actual_action",
        "is_best_mean",
        "derived_labels",
    ]
    compact = []
    for row in rows:
        out = compact_row(row)
        out["role_family"] = row["_role_family"]
        out["position_family"] = row["_position_family"]
        out["phase"] = row["_phase"]
        compact.append(out)
    write_csv(path, compact, fieldnames)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "bead_id": BEAD_ID,
        "experiment": "w42-sequence-seat-counterfactuals",
        "input": str(args.input),
        "max_rows": args.max_rows,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "min_label_n": args.min_label_n,
        "min_paired_n": args.min_paired_n,
        "min_slice_n": args.min_slice_n,
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=args.output_dir,
        tags=["w42", "claim-analysis", "sequence", "seat", BEAD_ID],
    )

    raw_rows = read_jsonl(args.input, max_rows=args.max_rows)
    rows = enrich_rows(raw_rows)
    label_metric_rows = label_metrics(rows, min_label_n=args.min_label_n)
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
    examples = build_examples(pair_map, example_limit=args.example_limit)

    label_counts = Counter(label for row in rows for label in row["_labels"])
    summary = {
        "schema_version": "w42.sequence_seat_counterfactuals.v1",
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
            "paired_contrast_rows": len(contrast_rows),
            "paired_contrast_slice_rows": len(contrast_slice_rows),
            "top_label_counts": dict(label_counts.most_common(35)),
        },
        "scientific_status": (
            "Direct branch-value counterfactuals at naturally occurring sequence states. "
            "This tests legal candidate moves within a public sequence context; it does not yet "
            "simulate fully divergent multi-trick human plan trees."
        ),
        "leakage_boundary": (
            "Derived detector labels use public/action-local columns from all_action_rows. "
            "Oracle mean, threshold mass, lower-tail mass, and future branch outcomes are offline labels only."
        ),
        "wandb": wb.status(),
    }

    write_csv(
        args.output_dir / "label_metrics.csv",
        label_metric_rows,
        [
            "label",
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
        ],
    )
    write_csv(
        args.output_dir / "paired_contrasts.csv",
        contrast_rows,
        [
            "contrast_id",
            "family",
            "book_question",
            "paired_decision_n",
            "mean_delta",
            "mean_delta_ci95_low",
            "mean_delta_ci95_high",
            "regret_delta",
            "threshold_mass_delta",
            "lower_tail_mass_delta",
            "preferred_actual_action_rate",
            "alternative_actual_action_rate",
        ],
    )
    write_csv(
        args.output_dir / "paired_contrasts_by_slice.csv",
        contrast_slice_rows,
        [
            "base_contrast_id",
            "contrast_id",
            "family",
            "book_question",
            "slice_name",
            "slice_value",
            "paired_decision_n",
            "mean_delta",
            "mean_delta_ci95_low",
            "mean_delta_ci95_high",
            "regret_delta",
            "threshold_mass_delta",
            "lower_tail_mass_delta",
            "preferred_actual_action_rate",
            "alternative_actual_action_rate",
        ],
    )
    write_compact_rows(args.output_dir / "labeled_sequence_action_rows.csv", rows)
    write_json(args.output_dir / "examples.json", examples)
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "bead": BEAD_ID,
            "input": str(args.input),
            "artifacts": {
                "summary": str(args.output_dir / "summary.json"),
                "label_metrics": str(args.output_dir / "label_metrics.csv"),
                "paired_contrasts": str(args.output_dir / "paired_contrasts.csv"),
                "paired_contrasts_by_slice": str(args.output_dir / "paired_contrasts_by_slice.csv"),
                "labeled_sequence_action_rows": str(args.output_dir / "labeled_sequence_action_rows.csv"),
                "examples": str(args.output_dir / "examples.json"),
            },
            "boundary": summary["scientific_status"],
        },
    )

    if getattr(wb, "run", None) is not None:
        for i, row in enumerate(contrast_rows):
            wb.log_series_point(
                axis="contrast/index",
                value=i,
                metrics={
                    "contrast/paired_decision_n": row["paired_decision_n"],
                    "contrast/mean_delta": row["mean_delta"],
                    "contrast/threshold_mass_delta": row["threshold_mass_delta"],
                    "contrast/lower_tail_mass_delta": row["lower_tail_mass_delta"],
                },
            )
        wb.update_summary(
            {
                "action_rows": len(rows),
                "decision_rows": summary["coverage"]["decision_rows"],
                "paired_contrast_rows": len(contrast_rows),
                "paired_contrast_slice_rows": len(contrast_slice_rows),
            }
        )
        wb.log_artifact_files(
            name="w42-sequence-seat-counterfactuals",
            artifact_type="w42-claim-analysis",
            paths=[
                args.output_dir / "summary.json",
                args.output_dir / "paired_contrasts.csv",
                args.output_dir / "paired_contrasts_by_slice.csv",
                args.output_dir / "examples.json",
            ],
        )
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
