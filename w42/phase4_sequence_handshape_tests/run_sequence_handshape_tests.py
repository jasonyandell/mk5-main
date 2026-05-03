#!/usr/bin/env python3
"""Phase-4 public/action-local W42 hand-shape proxy tests.

This runner deliberately stays inside the row surface exposed by the phase-3
full legal-action artifact. It can test same-decision action contrasts and
lead-hand legal-candidate shape proxies; it cannot assert exact hidden
remaining hands, played-history exhaustion, or later-seat guarantees except at
closure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "w42/tactical_claim_replication/all_action_rows.jsonl"
DEFAULT_OUTPUT = ROOT / "w42/phase4_sequence_handshape_tests"
BEAD_ID = "t42-br7n.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260503)
    parser.add_argument("--min-paired-n", type=int, default=12)
    parser.add_argument("--min-label-n", type=int, default=12)
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


def as_int(value: Any, default: int = 0) -> int:
    if value in {None, ""}:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean(values: Iterable[float]) -> float:
    nums = finite(values)
    if not nums:
        return float("nan")
    return float(sum(nums) / len(nums))


def bootstrap_ci(values: Iterable[float], samples: int, seed: int) -> tuple[float, float, float]:
    nums = finite(values)
    center = mean(nums)
    if not nums:
        return center, float("nan"), float("nan")
    if len(nums) == 1 or samples <= 0:
        return center, center, center
    rng = random.Random(seed)
    draws = []
    n = len(nums)
    for _ in range(samples):
        draws.append(sum(nums[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return center, float(lo), float(hi)


def role_family(row: dict[str, Any]) -> str:
    role = str(row.get("seat_role", ""))
    if role == "bidder":
        return "bidder"
    if role == "bidder_partner":
        return "partner"
    if "setter" in role or str(row.get("team", "")) == "defense":
        return "setter"
    return "unknown"


def team_opponent(team: str) -> str:
    if team == "offense":
        return "defense"
    if team == "defense":
        return "offense"
    return ""


def domino_pips(text: Any) -> tuple[int, int]:
    parts = str(text).split("-")
    if len(parts) != 2:
        return (0, 0)
    return (as_int(parts[0]), as_int(parts[1]))


def pip_sum(text: Any) -> int:
    a, b = domino_pips(text)
    return a + b


def phase_label(trick_idx: int) -> str:
    if trick_idx <= 1:
        return "early"
    if trick_idx <= 4:
        return "middle"
    return "late"


def group_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("key"))].append(row)
    return groups


def decision_shape(bucket: list[dict[str, Any]]) -> dict[str, Any]:
    first = bucket[0]
    called = [r for r in bucket if as_bool(r.get("candidate_is_called_suit"))]
    off = [r for r in bucket if not as_bool(r.get("candidate_is_called_suit"))]
    count = [r for r in bucket if as_int(r.get("candidate_count_points")) > 0]
    noncount = [r for r in bucket if as_int(r.get("candidate_count_points")) == 0]
    doubles = [r for r in bucket if as_bool(r.get("candidate_is_double"))]
    beaters = [r for r in bucket if as_bool(r.get("candidate_beats_current"))]
    winners = [r for r in bucket if as_bool(r.get("candidate_would_win_trick_now"))]
    return {
        "key": str(first.get("key")),
        "source_file": first.get("source_file", ""),
        "seed": first.get("seed", ""),
        "game_idx": first.get("game_idx", ""),
        "decision_idx": first.get("decision_idx", ""),
        "actor": first.get("actor", ""),
        "role_family": role_family(first),
        "seat_role": first.get("seat_role", ""),
        "team": first.get("team", ""),
        "trick_idx": as_int(first.get("trick_idx")),
        "trick_position": as_int(first.get("trick_position")),
        "phase": phase_label(as_int(first.get("trick_idx"))),
        "bid_value": as_int(first.get("bid_value")),
        "decl_name": first.get("decl_name", ""),
        "current_winner_team_before": first.get("current_winner_team_before", ""),
        "current_trick_count_before": as_int(first.get("current_trick_count_before")),
        "legal_action_n": len(bucket),
        "called_suit_legal_n": len(called),
        "off_suit_legal_n": len(off),
        "count_legal_n": len(count),
        "noncount_legal_n": len(noncount),
        "double_legal_n": len(doubles),
        "beater_legal_n": len(beaters),
        "would_win_legal_n": len(winners),
        "max_count_points": max((as_int(r.get("candidate_count_points")) for r in bucket), default=0),
        "max_pip_sum": max((pip_sum(r.get("candidate_domino")) for r in bucket), default=0),
        "has_called_and_off": bool(called and off),
        "has_count_and_noncount": bool(count and noncount),
        "lead_full_hand_proxy": as_int(first.get("trick_position")) == 0,
    }


def row_labels(row: dict[str, Any], shape: dict[str, Any]) -> tuple[str, ...]:
    labels: set[str] = set()
    role = role_family(row)
    team = str(row.get("team", ""))
    current_winner_team = str(row.get("current_winner_team_before", ""))
    trick_pos = as_int(row.get("trick_position"))
    trick_idx = as_int(row.get("trick_idx"))
    bid = as_int(row.get("bid_value"))
    called = as_bool(row.get("candidate_is_called_suit"))
    double = as_bool(row.get("candidate_is_double"))
    count_points = as_int(row.get("candidate_count_points"))
    count = count_points > 0
    beats_current = as_bool(row.get("candidate_beats_current"))
    would_win_now = as_bool(row.get("candidate_would_win_trick_now"))
    pip_total = pip_sum(row.get("candidate_domino"))
    trick_count = as_int(row.get("current_trick_count_before"))

    labels.add(f"role_{role}")
    labels.add(f"phase_{phase_label(trick_idx)}")
    labels.add(f"seat_pos_{trick_pos}")
    if trick_pos == 0:
        labels.add("sequence_lead")
    elif trick_pos == 3:
        labels.add("sequence_closure")
    else:
        labels.add("sequence_follow")

    if role == "bidder" and trick_pos == 0:
        labels.add("ch03_bidder_lead_plan")
        if called:
            labels.add("ch03_bidder_lead_trump_or_called")
            if double:
                labels.add("ch03_commanding_called_double")
            else:
                labels.add("ch03_called_non_double")
            if shape["has_called_and_off"] and shape["called_suit_legal_n"] >= 2:
                labels.add("ch03_visible_reentry_preserved_proxy")
            if shape["has_called_and_off"] and shape["called_suit_legal_n"] == 1:
                labels.add("ch03_last_visible_trump_spent_before_off_proxy")
        else:
            labels.add("ch03_bidder_lead_off_suit")
            if count:
                labels.add("ch03_bidder_lead_off_count")
            if trick_idx <= 1 and shape["called_suit_legal_n"] >= 4 and shape["off_suit_legal_n"] == 1:
                labels.add("ch03_early_one_off_four_trump_exception_proxy")
        if called and not double and shape["off_suit_legal_n"] > 0 and shape["called_suit_legal_n"] > 1:
            labels.add("ch03_extra_nonboss_trump_before_off_proxy")
        if count:
            labels.add("ch03_bidder_count_liability_lead")

    if role == "partner":
        labels.add("ch04_partner_support_regime")
        offense_current = current_winner_team == "offense"
        if count and offense_current and trick_pos == 3:
            labels.add("ch04_partner_count_donation_closure_guaranteed")
        if not count and offense_current and trick_pos == 3:
            labels.add("ch04_partner_noncount_closure_alternative")
        if count and offense_current and trick_pos < 3:
            labels.add("ch04_partner_count_before_closure_not_guaranteed")
        if not count and offense_current and trick_pos < 3:
            labels.add("ch04_partner_noncount_before_closure_alternative")
        if count and offense_current and called and trick_pos < 3:
            labels.add("ch04_low_trump_trap_count_dump_proxy")
        if not count and offense_current and called and trick_pos < 3:
            labels.add("ch04_low_trump_trap_noncount_alternative")
        if trick_pos == 0:
            if count:
                labels.add("ch04_partner_lead_count_liability")
            else:
                labels.add("ch04_partner_lead_noncount_low_liability")
            if double:
                labels.add("ch04_partner_double_or_virtual_boss_proxy")
            if called:
                labels.add("ch04_partner_disruptive_trump_lead_proxy")

    if role == "setter":
        labels.add("ch05_setter_pressure_regime")
        defense_current = current_winner_team == "defense"
        offense_current = current_winner_team == "offense"
        if count and defense_current and trick_pos < 3:
            labels.add("ch05_count_before_certainty_to_partner_pounce")
        if not count and defense_current and trick_pos < 3:
            labels.add("ch05_noncount_before_certainty_alternative")
        if count and defense_current and trick_pos == 3:
            labels.add("ch05_count_on_closure_defensive_win")
        if not count and defense_current and trick_pos == 3:
            labels.add("ch05_noncount_on_closure_defensive_win")
        if count and offense_current:
            labels.add("ch05_reckless_count_to_bidder_control")
        if not count and offense_current:
            labels.add("ch05_noncount_to_bidder_control_alternative")
        if trick_pos == 0:
            if double or count:
                labels.add("ch05_count_calling_or_count_lead")
            else:
                labels.add("ch05_non_count_calling_lead")
            if bid >= 35 and not called and pip_total >= 4:
                labels.add("ch05_high_bid_off_pressure_lead")
            elif not double and not count:
                labels.add("ch05_generic_non_count_calling_lead")
            if bid >= 35 and count:
                labels.add("ch05_high_bid_count_attack")
        if called and beats_current and offense_current and (trick_count + count_points) > 0:
            labels.add("ch05_trump_in_on_count_or_bidder_control")
        if not called and offense_current and shape["beater_legal_n"] > 0 and trick_count == 0:
            labels.add("ch05_hold_trump_no_count_proxy")
        if would_win_now and team == "defense" and (trick_count + count_points) > 0:
            labels.add("ch05_pounce_take_count_now")

    return tuple(sorted(labels))


def metric_rows(labeled_rows: list[dict[str, Any]], min_n: int, samples: int, seed: int) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled_rows:
        for label in str(row["labels"]).split("|"):
            by_label[label].append(row)
    out = []
    for label, rows in sorted(by_label.items()):
        if len(rows) < min_n:
            continue
        regrets = [as_float(r["mean_regret"]) for r in rows]
        means = [as_float(r["mean"]) for r in rows]
        threshold_gaps = [as_float(r["threshold_gap"]) for r in rows]
        lower_tail = [as_float(r["lower_tail_mass"]) for r in rows]
        regret_mean, regret_lo, regret_hi = bootstrap_ci(regrets, samples, seed + len(out))
        out.append(
            {
                "label": label,
                "action_n": len(rows),
                "decision_n": len({r["key"] for r in rows}),
                "actual_action_rate": mean([1.0 if as_bool(r["is_actual_action"]) else 0.0 for r in rows]),
                "best_mean_rate": mean([1.0 if as_bool(r["is_best_mean"]) else 0.0 for r in rows]),
                "best_threshold_rate": mean([1.0 if as_bool(r["is_best_threshold"]) else 0.0 for r in rows]),
                "safest_tail_rate": mean([1.0 if as_bool(r["is_safest_tail"]) else 0.0 for r in rows]),
                "mean_branch_value": mean(means),
                "mean_regret": regret_mean,
                "mean_regret_ci95_low": regret_lo,
                "mean_regret_ci95_high": regret_hi,
                "mean_threshold_gap": mean(threshold_gaps),
                "mean_lower_tail_mass": mean(lower_tail),
            }
        )
    return out


def paired_contrasts(
    groups: dict[str, list[dict[str, Any]]],
    specs: list[dict[str, str]],
    min_n: int,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        diffs = []
        threshold_diffs = []
        tail_diffs = []
        examples = 0
        for bucket in groups.values():
            pos = [r for r in bucket if spec["positive"] in str(r["labels"]).split("|")]
            neg = [r for r in bucket if spec["negative"] in str(r["labels"]).split("|")]
            if not pos or not neg:
                continue
            best_pos = max(pos, key=lambda r: as_float(r["mean"]))
            best_neg = max(neg, key=lambda r: as_float(r["mean"]))
            diffs.append(as_float(best_pos["mean"]) - as_float(best_neg["mean"]))
            threshold_diffs.append(as_float(best_pos["threshold_mass"]) - as_float(best_neg["threshold_mass"]))
            tail_diffs.append(as_float(best_neg["lower_tail_mass"]) - as_float(best_pos["lower_tail_mass"]))
            examples += 1
        if len(diffs) < min_n:
            continue
        diff_mean, diff_lo, diff_hi = bootstrap_ci(diffs, samples, seed + len(rows))
        rows.append(
            {
                "contrast": spec["name"],
                "family": spec["family"],
                "positive_label": spec["positive"],
                "negative_label": spec["negative"],
                "paired_decision_n": examples,
                "mean_value_delta": diff_mean,
                "mean_value_delta_ci95_low": diff_lo,
                "mean_value_delta_ci95_high": diff_hi,
                "mean_threshold_mass_delta": mean(threshold_diffs),
                "mean_tail_safety_delta": mean(tail_diffs),
                "interpretation": spec["interpretation"],
            }
        )
    return rows


def slice_contrasts(groups: dict[str, list[dict[str, Any]]], paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requested = {row["contrast"]: row for row in paired}
    slices = ["phase_early", "phase_middle", "phase_late", "seat_pos_0", "seat_pos_1", "seat_pos_2", "seat_pos_3"]
    out = []
    for contrast, spec in requested.items():
        for slice_label in slices:
            diffs = []
            for bucket in groups.values():
                if not any(slice_label in str(r["labels"]).split("|") for r in bucket):
                    continue
                pos = [r for r in bucket if spec["positive_label"] in str(r["labels"]).split("|")]
                neg = [r for r in bucket if spec["negative_label"] in str(r["labels"]).split("|")]
                if not pos or not neg:
                    continue
                diffs.append(max(as_float(r["mean"]) for r in pos) - max(as_float(r["mean"]) for r in neg))
            if len(diffs) >= 12:
                out.append(
                    {
                        "contrast": contrast,
                        "slice": slice_label,
                        "paired_decision_n": len(diffs),
                        "mean_value_delta": mean(diffs),
                    }
                )
    return out


def examples_for_labels(labeled_rows: list[dict[str, Any]], labels: list[str], limit: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        rows = [r for r in labeled_rows if label in str(r["labels"]).split("|")]
        rows.sort(key=lambda r: abs(as_float(r["mean_regret"])), reverse=True)
        out[label] = [
            {
                "key": r["key"],
                "seed": r["seed"],
                "game_idx": r["game_idx"],
                "decision_idx": r["decision_idx"],
                "seat_role": r["seat_role"],
                "trick_idx": r["trick_idx"],
                "trick_position": r["trick_position"],
                "bid_value": r["bid_value"],
                "decl_name": r["decl_name"],
                "candidate_domino": r["candidate_domino"],
                "candidate_count_points": r["candidate_count_points"],
                "candidate_is_called_suit": r["candidate_is_called_suit"],
                "candidate_is_double": r["candidate_is_double"],
                "mean": r["mean"],
                "mean_regret": r["mean_regret"],
                "threshold_mass": r["threshold_mass"],
                "labels": r["labels"],
            }
            for r in rows[:limit]
        ]
    return out


def blocker_rows() -> list[dict[str, str]]:
    return [
        {
            "claim_family": "exact reentry preservation",
            "missing_fields": "remaining actor hand after candidate; unresolved off inventory; future lead regain path",
            "partial_artifact": "ch03_visible_reentry_preserved_proxy and ch03_last_visible_trump_spent_before_off_proxy",
            "why_it_matters": "legal lead candidates expose a hand-shape proxy, but follow decisions do not expose the full remaining hand.",
        },
        {
            "claim_family": "partner guaranteed donation before closure",
            "missing_fields": "later-seat forced responses / exact cannot-overtrump guarantee",
            "partial_artifact": "closure guarantee is exact at trick_position=3; earlier seats remain not-guaranteed public proxies.",
            "why_it_matters": "current winner is not enough when one or two seats still act.",
        },
        {
            "claim_family": "low-trump trap",
            "missing_fields": "led domino identity and remaining higher trump ownership",
            "partial_artifact": "ch04_low_trump_trap_count_dump_proxy",
            "why_it_matters": "the row table says candidate is trump, but not whether the already-led trump was low and overtrumpable.",
        },
        {
            "claim_family": "count-protection throwaway",
            "missing_fields": "full actor hand, same-suit protectors, future double-ahead exposure path",
            "partial_artifact": "none beyond role/count/free-discard proxies",
            "why_it_matters": "protector preservation is a hidden hand-shape claim, not an action-local row claim.",
        },
        {
            "claim_family": "trump-rich setter recognition",
            "missing_fields": "hidden distribution of outstanding trump and public first-trump follow history",
            "partial_artifact": "ch05_trump_in_on_count_or_bidder_control and ch05_hold_trump_no_count_proxy",
            "why_it_matters": "trump-rich means ownership concentration, which the current JSONL omits.",
        },
        {
            "claim_family": "void creation leading to later pounce",
            "missing_fields": "actor remaining suits before discard and multi-trick counterfactual continuation",
            "partial_artifact": "phase/trick pounce rows only",
            "why_it_matters": "the mechanism is delayed; current rows test immediate action branches, not causal multi-trick plan divergence.",
        },
        {
            "claim_family": "effective double / highest remaining tile",
            "missing_fields": "played-history suit exhaustion and unseen higher-tile accounting",
            "partial_artifact": "double lead proxy only",
            "why_it_matters": "a non-double can act as a double only after all higher same-suit tiles are dead or known.",
        },
        {
            "claim_family": "high-bid off pounce pressure",
            "missing_fields": "bid values above 30 in the current tactical row corpus",
            "partial_artifact": "none from all_action_rows.jsonl; bid distribution is entirely 30",
            "why_it_matters": "the book's 35/36 overbid pressure claim needs generated high-bid contracts or seed mining outside this row table.",
        },
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = read_jsonl(args.input, args.max_rows)
    groups = group_rows(raw_rows)
    shapes = {key: decision_shape(bucket) for key, bucket in groups.items()}
    shape_rows = list(shapes.values())

    labeled_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        shape = shapes[str(row.get("key"))]
        labels = row_labels(row, shape)
        labeled_rows.append(
            {
                "key": row.get("key", ""),
                "source_file": row.get("source_file", ""),
                "seed": row.get("seed", ""),
                "game_idx": row.get("game_idx", ""),
                "decision_idx": row.get("decision_idx", ""),
                "actor": row.get("actor", ""),
                "seat_role": row.get("seat_role", ""),
                "role_family": role_family(row),
                "team": row.get("team", ""),
                "trick_idx": row.get("trick_idx", ""),
                "trick_position": row.get("trick_position", ""),
                "phase": phase_label(as_int(row.get("trick_idx"))),
                "bid_value": row.get("bid_value", ""),
                "decl_name": row.get("decl_name", ""),
                "current_winner_team_before": row.get("current_winner_team_before", ""),
                "current_trick_count_before": row.get("current_trick_count_before", ""),
                "candidate_domino": row.get("candidate_domino", ""),
                "candidate_count_points": row.get("candidate_count_points", ""),
                "candidate_is_called_suit": row.get("candidate_is_called_suit", ""),
                "candidate_is_double": row.get("candidate_is_double", ""),
                "candidate_beats_current": row.get("candidate_beats_current", ""),
                "candidate_would_win_trick_now": row.get("candidate_would_win_trick_now", ""),
                "is_actual_action": row.get("is_actual_action", ""),
                "is_best_mean": row.get("is_best_mean", ""),
                "is_best_threshold": row.get("is_best_threshold", ""),
                "is_safest_tail": row.get("is_safest_tail", ""),
                "mean": row.get("mean", ""),
                "mean_regret": row.get("mean_regret", ""),
                "threshold_mass": row.get("threshold_mass", ""),
                "threshold_gap": row.get("threshold_gap", ""),
                "lower_tail_mass": row.get("lower_tail_mass", ""),
                "legal_action_n": shape["legal_action_n"],
                "called_suit_legal_n": shape["called_suit_legal_n"],
                "off_suit_legal_n": shape["off_suit_legal_n"],
                "count_legal_n": shape["count_legal_n"],
                "double_legal_n": shape["double_legal_n"],
                "beater_legal_n": shape["beater_legal_n"],
                "labels": "|".join(labels),
            }
        )

    labeled_groups = group_rows(labeled_rows)
    metrics = metric_rows(labeled_rows, args.min_label_n, args.bootstrap_samples, args.bootstrap_seed)
    specs = [
        {
            "name": "ch03_commanding_called_double_vs_off",
            "family": "bidder lead sequencing",
            "positive": "ch03_commanding_called_double",
            "negative": "ch03_bidder_lead_off_suit",
            "interpretation": "Commanding called doubles versus off-suit lead choices in the same bidder lead decision.",
        },
        {
            "name": "ch03_called_non_double_vs_off",
            "family": "bidder lead sequencing",
            "positive": "ch03_called_non_double",
            "negative": "ch03_bidder_lead_off_suit",
            "interpretation": "Blanket non-double trump/called leads versus off-suit alternatives.",
        },
        {
            "name": "ch03_reentry_preserved_vs_last_visible_trump",
            "family": "bidder reentry",
            "positive": "ch03_visible_reentry_preserved_proxy",
            "negative": "ch03_last_visible_trump_spent_before_off_proxy",
            "interpretation": "Lead-hand proxy for preserving at least one visible called-suit candidate while offs remain.",
        },
        {
            "name": "ch03_early_exception_off_vs_extra_nonboss_trump",
            "family": "bidder one-off exception",
            "positive": "ch03_early_one_off_four_trump_exception_proxy",
            "negative": "ch03_extra_nonboss_trump_before_off_proxy",
            "interpretation": "One-off/four-called proxy tests whether early off can beat another non-boss called-suit lead.",
        },
        {
            "name": "ch04_closure_count_donation_vs_noncount",
            "family": "partner donation timing",
            "positive": "ch04_partner_count_donation_closure_guaranteed",
            "negative": "ch04_partner_noncount_closure_alternative",
            "interpretation": "Exact closure count donation versus noncount in the same partner decision when bidder side already controls.",
        },
        {
            "name": "ch04_before_closure_count_vs_noncount",
            "family": "partner donation timing",
            "positive": "ch04_partner_count_before_closure_not_guaranteed",
            "negative": "ch04_partner_noncount_before_closure_alternative",
            "interpretation": "Partner count donation before closure versus same-decision noncount alternatives; current control is public but later seats may still overtake.",
        },
        {
            "name": "ch04_low_trump_trap_count_vs_noncount",
            "family": "partner low-trump trap",
            "positive": "ch04_low_trump_trap_count_dump_proxy",
            "negative": "ch04_low_trump_trap_noncount_alternative",
            "interpretation": "Count dump on called-suit before closure versus noncount alternatives under the low-trump-trap proxy.",
        },
        {
            "name": "ch04_partner_lead_count_liability_vs_noncount",
            "family": "partner count-liability leads",
            "positive": "ch04_partner_lead_count_liability",
            "negative": "ch04_partner_lead_noncount_low_liability",
            "interpretation": "Partner support lead choices that expose count versus noncount lead alternatives.",
        },
        {
            "name": "ch05_count_before_certainty_vs_reckless_to_bidder",
            "family": "setter pounce",
            "positive": "ch05_count_before_certainty_to_partner_pounce",
            "negative": "ch05_noncount_before_certainty_alternative",
            "interpretation": "Defensive count pressure before partner certainty versus noncount alternatives in the same defense-controlled decision.",
        },
        {
            "name": "ch05_reckless_count_to_bidder_vs_noncount",
            "family": "setter count liability",
            "positive": "ch05_reckless_count_to_bidder_control",
            "negative": "ch05_noncount_to_bidder_control_alternative",
            "interpretation": "Count into bidder-side control versus noncount alternatives; negative values support the reckless-count warning.",
        },
        {
            "name": "ch05_pounce_take_count_vs_hold_trump_no_count",
            "family": "setter pounce",
            "positive": "ch05_pounce_take_count_now",
            "negative": "ch05_hold_trump_no_count_proxy",
            "interpretation": "Immediate defensive count-taking pounce versus holding off when count is absent or proxy-safe.",
        },
        {
            "name": "ch05_count_calling_lead_vs_non_count_calling",
            "family": "setter count-calling leads",
            "positive": "ch05_count_calling_or_count_lead",
            "negative": "ch05_non_count_calling_lead",
            "interpretation": "Setter lead attack with doubles/count versus non-count-calling lead.",
        },
        {
            "name": "ch05_high_bid_pressure_vs_generic_non_count",
            "family": "high-bid off pressure",
            "positive": "ch05_high_bid_off_pressure_lead",
            "negative": "ch05_generic_non_count_calling_lead",
            "interpretation": "Bid >=35 off-pressure leads versus generic non-count-calling setter leads.",
        },
    ]
    paired = paired_contrasts(labeled_groups, specs, args.min_paired_n, args.bootstrap_samples, args.bootstrap_seed)
    slices = slice_contrasts(labeled_groups, paired)

    label_counts = Counter()
    for row in labeled_rows:
        label_counts.update(str(row["labels"]).split("|"))
    top_labels = [label for label, _ in label_counts.most_common(28)]
    examples = examples_for_labels(labeled_rows, top_labels, args.example_limit)

    blockers = blocker_rows()
    summary = {
        "bead": BEAD_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_sha(),
        "git_status_before_artifacts": git_status_short(),
        "schema_version": "w42.phase4_sequence_handshape_tests.v1",
        "input": str(args.input),
        "max_rows": args.max_rows,
        "scientific_status": (
            "Public/action-local and lead-legal-candidate hand-shape proxy tests over existing "
            "phase-3 full legal-action rows. Offline mean/threshold/tail labels are report-only."
        ),
        "leakage_boundary": (
            "Labels use public/action-local row columns and same-decision legal-candidate sets. "
            "Branch mean, threshold mass, lower-tail mass, and regret are offline evaluation labels, not model features."
        ),
        "coverage": {
            "action_rows": len(labeled_rows),
            "decision_rows": len(groups),
            "shape_rows": len(shape_rows),
            "label_metric_rows": len(metrics),
            "paired_contrast_rows": len(paired),
            "paired_contrast_slice_rows": len(slices),
            "blocker_rows": len(blockers),
            "top_label_counts": dict(label_counts.most_common(40)),
        },
        "headline_contrasts": paired[:12],
        "caveat": (
            "Exact hand-shape claims that depend on private remaining hands, played-history exhaustion, "
            "or multi-trick causal continuations are listed in blocker_table.csv."
        ),
    }

    write_csv(
        args.output_dir / "labeled_handshape_action_rows.csv",
        labeled_rows,
        [
            "key",
            "source_file",
            "seed",
            "game_idx",
            "decision_idx",
            "actor",
            "seat_role",
            "role_family",
            "team",
            "trick_idx",
            "trick_position",
            "phase",
            "bid_value",
            "decl_name",
            "current_winner_team_before",
            "current_trick_count_before",
            "candidate_domino",
            "candidate_count_points",
            "candidate_is_called_suit",
            "candidate_is_double",
            "candidate_beats_current",
            "candidate_would_win_trick_now",
            "is_actual_action",
            "is_best_mean",
            "is_best_threshold",
            "is_safest_tail",
            "mean",
            "mean_regret",
            "threshold_mass",
            "threshold_gap",
            "lower_tail_mass",
            "legal_action_n",
            "called_suit_legal_n",
            "off_suit_legal_n",
            "count_legal_n",
            "double_legal_n",
            "beater_legal_n",
            "labels",
        ],
    )
    write_csv(
        args.output_dir / "decision_shape_rows.csv",
        shape_rows,
        [
            "key",
            "source_file",
            "seed",
            "game_idx",
            "decision_idx",
            "actor",
            "role_family",
            "seat_role",
            "team",
            "trick_idx",
            "trick_position",
            "phase",
            "bid_value",
            "decl_name",
            "current_winner_team_before",
            "current_trick_count_before",
            "legal_action_n",
            "called_suit_legal_n",
            "off_suit_legal_n",
            "count_legal_n",
            "noncount_legal_n",
            "double_legal_n",
            "beater_legal_n",
            "would_win_legal_n",
            "max_count_points",
            "max_pip_sum",
            "has_called_and_off",
            "has_count_and_noncount",
            "lead_full_hand_proxy",
        ],
    )
    write_csv(args.output_dir / "label_metrics.csv", metrics, list(metrics[0].keys()) if metrics else ["label"])
    write_csv(args.output_dir / "paired_contrasts.csv", paired, list(paired[0].keys()) if paired else ["contrast"])
    write_csv(
        args.output_dir / "paired_contrasts_by_slice.csv",
        slices,
        list(slices[0].keys()) if slices else ["contrast", "slice", "paired_decision_n", "mean_value_delta"],
    )
    write_csv(args.output_dir / "blocker_table.csv", blockers, list(blockers[0].keys()))
    write_json(args.output_dir / "examples.json", examples)
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
