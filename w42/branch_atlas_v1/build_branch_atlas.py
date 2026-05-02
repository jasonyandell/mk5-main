#!/usr/bin/env python3
"""Build a branch-aware w42 atlas from saved joint-world E[Q] artifacts.

Input is a ``torch.save`` payload produced by ``python -m forge.eq.generate
--save-joint-worlds``. The atlas keeps scalar E[Q], distribution shape, and
hidden-domino attribution in one report surface. Hidden ownership is always an
offline label/eval target, never an online feature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle import schema, tables
from forge.oracle.tables import can_follow, led_suit_for_lead_domino, trick_rank
from w42.wandb_utils import add_wandb_args, init_wandb


OUT_DIR = Path(__file__).resolve().parent
OFFENSE_PLAYERS = {0, 2}
PARTNER = {0: 2, 1: 3, 2: 0, 3: 1}
SEAT_ROLE = {
    0: "bidder",
    1: "left_setter",
    2: "bidder_partner",
    3: "right_setter",
}
DECL_NAMES = {
    0: "blanks",
    1: "ones",
    2: "twos",
    3: "threes",
    4: "fours",
    5: "fives",
    6: "sixes",
    7: "doubles",
    8: "doubles-suit",
    9: "no-trump",
}
EQ_MIN = -42
EQ_MAX = 42
LOW_TAIL_Q = -18.0
OFFENSE_MAKE_Q = 18.0
DEFENSE_SET_Q = -17.0
TOTAL_COUNT_POINTS = sum(tables.DOMINO_COUNT_POINTS)


ACTION_COLUMNS = [
    "game_idx",
    "seed",
    "decl_id",
    "decl_name",
    "bid_value",
    "bid_value_source",
    "decision_idx",
    "trick_idx",
    "trick_position",
    "actor",
    "seat_role",
    "team",
    "partner_player",
    "offense_score_before",
    "defense_score_before",
    "played_count_points_before",
    "unplayed_count_points_public_before",
    "current_trick_count_before",
    "led_suit_before",
    "led_suit_name_before",
    "current_winner_before",
    "current_winner_team_before",
    "partner_currently_winning_before",
    "opponent_currently_winning_before",
    "actual_action_slot",
    "actual_action_domino_id",
    "actual_action_domino",
    "candidate_slot",
    "candidate_domino_id",
    "candidate_domino",
    "candidate_count_points",
    "candidate_is_double",
    "candidate_is_called_suit",
    "candidate_can_follow_led",
    "candidate_beats_current",
    "candidate_would_win_trick_now",
    "candidate_trick_rank",
    "actor_hand_remaining_count",
    "actor_hand_remaining_count_points",
    "actor_hand_remaining_called_suit_count",
    "actor_hand_remaining_double_count",
    "legal_action_count",
    "threshold_q",
    "mean",
    "std",
    "threshold_mass",
    "lower_tail_mass_le_neg18",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "iqr",
    "cvar_low_10",
    "branch_entropy",
    "branch_peak_count",
    "branch_peak_bins",
    "shelf_gap",
    "samples",
    "mean_rank_desc",
    "threshold_rank_desc",
    "lower_tail_rank_asc",
    "mean_gap_to_best",
    "threshold_gap_to_best",
    "lower_tail_gap_to_safest",
    "is_actual_action",
    "is_top_mean_action",
    "is_top_threshold_action",
    "is_safest_lower_tail_action",
    "high_variance_close_mean",
    "scalar_ev_lying_by_omission",
    "distribution_shape_tags",
    "strategy_context_tags",
    "matched_position_detectors",
    "direct_label_readiness",
    "missing_direct_label_fields",
    "hidden_threat_available",
    "hidden_threat_rows",
    "top_hidden_impact_holder",
    "top_hidden_impact_domino",
    "top_hidden_impact_score",
    "top_hidden_downside_holder",
    "top_hidden_downside_domino",
    "top_hidden_downside_score",
    "top_hidden_upside_holder",
    "top_hidden_upside_domino",
    "top_hidden_upside_score",
]

THREAT_COLUMNS = [
    "game_idx",
    "seed",
    "decl_id",
    "decl_name",
    "decision_idx",
    "actor",
    "seat_role",
    "team",
    "action_slot",
    "action_domino_id",
    "action_domino",
    "hidden_domino_id",
    "hidden_domino",
    "relative_holder",
    "absolute_holder",
    "absolute_holder_role",
    "world_count",
    "conditioned_world_count",
    "conditioned_mass",
    "baseline_mean_q",
    "conditioned_mean_q",
    "mean_q_delta",
    "baseline_tail_low_mass",
    "conditioned_tail_low_mass",
    "tail_low_mass_delta",
    "baseline_shelf_high_mass",
    "conditioned_shelf_high_mass",
    "shelf_high_mass_delta",
    "baseline_std",
    "conditioned_std",
    "std_delta",
    "impact_score",
    "downside_score",
    "upside_score",
]

DECISION_COLUMNS = [
    "game_idx",
    "seed",
    "decl_id",
    "decl_name",
    "bid_value",
    "decision_idx",
    "trick_idx",
    "trick_position",
    "actor",
    "seat_role",
    "team",
    "offense_score_before",
    "defense_score_before",
    "actual_action_domino",
    "legal_action_count",
    "top_mean_domino",
    "top_mean",
    "top_threshold_domino",
    "top_threshold_mass",
    "safest_tail_domino",
    "safest_lower_tail_mass",
    "actual_mean_rank_desc",
    "actual_threshold_rank_desc",
    "actual_lower_tail_rank_asc",
    "decision_scalar_ev_lying_by_omission",
    "decision_multi_peak_action_count",
    "decision_max_std",
    "decision_max_shelf_gap",
    "decision_max_hidden_impact",
    "decision_max_hidden_downside",
    "decision_max_hidden_upside",
    "strategy_context_tags",
    "matched_position_detectors",
    "distribution_shape_tags",
]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        return float("nan")
    return out


def round_or_none(value: Any, ndigits: int = 4) -> float | None:
    value = finite(value)
    if math.isnan(value):
        return None
    return round(value, ndigits)


def team_for_player(player: int) -> str:
    return "offense" if player in OFFENSE_PLAYERS else "defense"


def team_id_for_player(player: int) -> int:
    return 0 if player in OFFENSE_PLAYERS else 1


def domino_label(domino_id: int | None) -> str:
    if domino_id is None or int(domino_id) < 0:
        return ""
    high, low = schema.domino_pips(int(domino_id))
    return f"{high}-{low}"


def led_suit_name(led_suit: int | None) -> str:
    if led_suit is None:
        return ""
    if int(led_suit) == 7:
        return "called"
    return str(int(led_suit))


def threshold_q_for_player(player: int) -> float:
    return OFFENSE_MAKE_Q if player in OFFENSE_PLAYERS else DEFENSE_SET_Q


def values_to_pdf(values: torch.Tensor) -> list[float]:
    bins = (values.detach().cpu().float().round().clamp(EQ_MIN, EQ_MAX).long() - EQ_MIN).tolist()
    counts = [0 for _ in range(EQ_MAX - EQ_MIN + 1)]
    for idx in bins:
        counts[int(idx)] += 1
    total = max(len(bins), 1)
    return [count / total for count in counts]


def pdf_mass(pdf: list[float], lo: int | None = None, hi: int | None = None) -> float:
    total = 0.0
    for idx, prob in enumerate(pdf):
        value = EQ_MIN + idx
        if lo is not None and value < lo:
            continue
        if hi is not None and value > hi:
            continue
        total += float(prob)
    return total


def quantile(pdf: list[float], probability: float) -> int:
    cumulative = 0.0
    for idx, prob in enumerate(pdf):
        cumulative += float(prob)
        if cumulative >= probability:
            return EQ_MIN + idx
    return EQ_MAX


def cvar_low(pdf: list[float], alpha: float = 0.10) -> float | None:
    remaining = alpha
    mass = 0.0
    weighted = 0.0
    for idx, prob in enumerate(pdf):
        if remaining <= 1e-12:
            break
        take = min(float(prob), remaining)
        weighted += take * (EQ_MIN + idx)
        mass += take
        remaining -= take
    if mass <= 0.0:
        return None
    return weighted / mass


def entropy_bits(pdf: list[float]) -> float:
    return -sum(float(p) * math.log(float(p), 2) for p in pdf if p > 0.0)


def local_peak_bins(pdf: list[float], min_prob: float = 0.02) -> list[int]:
    peaks: list[int] = []
    for idx, prob in enumerate(pdf):
        left = pdf[idx - 1] if idx > 0 else -1.0
        right = pdf[idx + 1] if idx < len(pdf) - 1 else -1.0
        if prob >= min_prob and prob >= left and prob >= right:
            peaks.append(EQ_MIN + idx)
    return peaks


def shelf_gap(pdf: list[float]) -> int | None:
    peaks = local_peak_bins(pdf)
    if len(peaks) < 2:
        return None
    return max(peaks) - min(peaks)


def q_stats(values: torch.Tensor, threshold_q: float) -> dict[str, Any]:
    values = values.detach().cpu().float()
    pdf = values_to_pdf(values)
    q10 = quantile(pdf, 0.10)
    q25 = quantile(pdf, 0.25)
    q50 = quantile(pdf, 0.50)
    q75 = quantile(pdf, 0.75)
    q90 = quantile(pdf, 0.90)
    peaks = local_peak_bins(pdf)
    low_cvar = cvar_low(pdf, 0.10)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() else float("nan"),
        "threshold_mass": float((values >= threshold_q).float().mean().item()) if values.numel() else float("nan"),
        "lower_tail_mass_le_neg18": float((values <= LOW_TAIL_Q).float().mean().item()) if values.numel() else float("nan"),
        "q10": q10,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "iqr": q75 - q25,
        "cvar_low_10": low_cvar,
        "branch_entropy": entropy_bits(pdf),
        "branch_peak_count": len(peaks),
        "branch_peak_bins": "|".join(str(peak) for peak in peaks),
        "shelf_gap": shelf_gap(pdf) or "",
        "samples": int(values.numel()),
    }


def current_winner(trick_plays: list[tuple[int, int]], decl_id: int) -> tuple[int | None, int | None, int | None]:
    if not trick_plays:
        return None, None, None
    led_suit = led_suit_for_lead_domino(trick_plays[0][1], decl_id)
    winner = trick_plays[0][0]
    best_rank = trick_rank(trick_plays[0][1], led_suit, decl_id)
    for player, domino_id in trick_plays[1:]:
        rank = trick_rank(domino_id, led_suit, decl_id)
        if rank > best_rank:
            winner = player
            best_rank = rank
    return winner, led_suit, best_rank


def candidate_public_facts(candidate_domino_id: int, context: dict[str, Any], decl_id: int) -> dict[str, Any]:
    led_suit = context["led_suit_before"]
    candidate_rank: int | str = ""
    follows: bool | str = ""
    beats: bool | str = ""
    would_win = True
    if led_suit != "":
        led_suit_int = int(led_suit)
        candidate_rank = trick_rank(candidate_domino_id, led_suit_int, decl_id)
        follows = can_follow(candidate_domino_id, led_suit_int, decl_id)
        beats = candidate_rank > int(context["current_best_rank_before"])
        would_win = bool(beats)
    return {
        "candidate_count_points": int(tables.DOMINO_COUNT_POINTS[candidate_domino_id]),
        "candidate_is_double": bool(tables.DOMINO_IS_DOUBLE[candidate_domino_id]),
        "candidate_is_called_suit": bool(tables.is_in_called_suit(candidate_domino_id, decl_id)),
        "candidate_can_follow_led": follows,
        "candidate_beats_current": beats,
        "candidate_would_win_trick_now": would_win,
        "candidate_trick_rank": candidate_rank,
    }


def move_context(
    *,
    decision_idx: int,
    actor: int,
    decl_id: int,
    score: list[int],
    played_count_points: int,
    trick_plays: list[tuple[int, int]],
) -> dict[str, Any]:
    winner, led_suit, best_rank = current_winner(trick_plays, decl_id)
    partner = PARTNER[actor]
    current_trick_count = sum(tables.DOMINO_COUNT_POINTS[d] for _p, d in trick_plays)
    return {
        "decision_idx": decision_idx,
        "trick_idx": decision_idx // 4,
        "trick_position": len(trick_plays),
        "actor": actor,
        "seat_role": SEAT_ROLE[actor],
        "team": team_for_player(actor),
        "partner_player": partner,
        "offense_score_before": int(score[0]),
        "defense_score_before": int(score[1]),
        "played_count_points_before": int(played_count_points),
        "unplayed_count_points_public_before": int(TOTAL_COUNT_POINTS - played_count_points),
        "current_trick_count_before": int(current_trick_count),
        "led_suit_before": "" if led_suit is None else int(led_suit),
        "led_suit_name_before": led_suit_name(led_suit),
        "current_winner_before": "" if winner is None else int(winner),
        "current_winner_team_before": "" if winner is None else team_for_player(winner),
        "current_best_rank_before": "" if best_rank is None else int(best_rank),
        "partner_currently_winning_before": bool(winner == partner),
        "opponent_currently_winning_before": bool(
            winner is not None and team_id_for_player(winner) != team_id_for_player(actor)
        ),
    }


def actor_remaining_hand(hands: list[list[int]], played_slots: dict[int, set[int]], player: int) -> list[int]:
    remaining: list[int] = []
    for slot, domino_id in enumerate(hands[player]):
        if slot not in played_slots[player]:
            remaining.append(int(domino_id))
    return remaining


def describe_conditioned_q(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "tail_low_mass": float("nan"),
            "shelf_high_mass": float("nan"),
            "std": float("nan"),
        }
    values = values.detach().cpu().float()
    return {
        "n": int(values.numel()),
        "mean": float(values.mean().item()),
        "tail_low_mass": float((values <= LOW_TAIL_Q).float().mean().item()),
        "shelf_high_mass": float((values >= OFFENSE_MAKE_Q).float().mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def hidden_threat_rows_for_action(
    *,
    game_idx: int,
    seed: int | None,
    decl_id: int,
    decision_idx: int,
    actor: int,
    action_slot: int,
    action_domino_id: int,
    world_hands: torch.Tensor,
    q_values: torch.Tensor,
    top_k: int,
) -> list[dict[str, Any]]:
    baseline = describe_conditioned_q(q_values)
    if baseline["n"] == 0:
        return []

    candidates = sorted({int(d) for d in world_hands.flatten().tolist() if int(d) >= 0})
    rows: list[dict[str, Any]] = []
    for domino_id in candidates:
        for rel_holder in range(3):
            abs_holder = (actor + rel_holder + 1) % 4
            mask = (world_hands[:, rel_holder, :] == domino_id).any(dim=1)
            conditioned = describe_conditioned_q(q_values[mask])
            if conditioned["n"] == 0:
                continue

            mean_delta = conditioned["mean"] - baseline["mean"]
            tail_delta = conditioned["tail_low_mass"] - baseline["tail_low_mass"]
            shelf_delta = conditioned["shelf_high_mass"] - baseline["shelf_high_mass"]
            std_delta = conditioned["std"] - baseline["std"]
            impact_score = abs(mean_delta) + 10.0 * (abs(tail_delta) + abs(shelf_delta))
            downside_score = max(0.0, -mean_delta) + 10.0 * max(0.0, tail_delta) + 5.0 * max(0.0, -shelf_delta)
            upside_score = max(0.0, mean_delta) + 10.0 * max(0.0, shelf_delta) + 5.0 * max(0.0, -tail_delta)
            rows.append(
                {
                    "game_idx": game_idx,
                    "seed": "" if seed is None else int(seed),
                    "decl_id": decl_id,
                    "decl_name": DECL_NAMES.get(decl_id, f"unknown-{decl_id}"),
                    "decision_idx": decision_idx,
                    "actor": actor,
                    "seat_role": SEAT_ROLE[actor],
                    "team": team_for_player(actor),
                    "action_slot": action_slot,
                    "action_domino_id": action_domino_id,
                    "action_domino": domino_label(action_domino_id),
                    "hidden_domino_id": domino_id,
                    "hidden_domino": domino_label(domino_id),
                    "relative_holder": rel_holder,
                    "absolute_holder": abs_holder,
                    "absolute_holder_role": SEAT_ROLE[abs_holder],
                    "world_count": baseline["n"],
                    "conditioned_world_count": conditioned["n"],
                    "conditioned_mass": conditioned["n"] / baseline["n"],
                    "baseline_mean_q": baseline["mean"],
                    "conditioned_mean_q": conditioned["mean"],
                    "mean_q_delta": mean_delta,
                    "baseline_tail_low_mass": baseline["tail_low_mass"],
                    "conditioned_tail_low_mass": conditioned["tail_low_mass"],
                    "tail_low_mass_delta": tail_delta,
                    "baseline_shelf_high_mass": baseline["shelf_high_mass"],
                    "conditioned_shelf_high_mass": conditioned["shelf_high_mass"],
                    "shelf_high_mass_delta": shelf_delta,
                    "baseline_std": baseline["std"],
                    "conditioned_std": conditioned["std"],
                    "std_delta": std_delta,
                    "impact_score": impact_score,
                    "downside_score": downside_score,
                    "upside_score": upside_score,
                }
            )

    rows.sort(key=lambda row: float(row["impact_score"]), reverse=True)
    return rows[:top_k]


def strategy_tags(row: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    tags: list[str] = []
    detectors: list[str] = []
    readiness: list[str] = []
    missing: set[str] = {"bid_margin"}

    actor = int(row["actor"])
    trick_idx = int(row["trick_idx"])
    trick_position = int(row["trick_position"])
    team = str(row["team"])
    role = str(row["seat_role"])

    if trick_idx == 0:
        tags.append("first_trick")
        if actor == 0 and trick_position == 0:
            tags.append("bidder_opening_lead")
            detectors.append("bidder_first_lead_plan")
            readiness.append("public_context_ready")
        elif actor == 1 and trick_position == 1:
            tags.append("first_setter_response")
            detectors.append("first_setter_pounce_window")
            readiness.append("needs_real_bid_margin")
        elif actor == 2 and trick_position == 2:
            tags.append("partner_third_seat_support")
            detectors.append("partner_third_seat_safe_donation")
            readiness.append("public_context_ready")
        elif actor == 3 and trick_position == 3:
            tags.append("defender_last_to_act_closure")
            detectors.append("last_to_act_closure_policy")
            readiness.append("public_context_ready")

    if trick_position == 3:
        tags.append("last_to_act")
        detectors.append("last_to_act_closure_policy")
    if trick_idx >= 4:
        tags.append("late_hand")
        detectors.append("late_trick_threshold_closure")
        readiness.append("distribution_labels_ready")
    if team == "defense" and trick_position == 0:
        tags.append("setter_lead_pressure")
        detectors.append("defender_damage_lead_class")
        readiness.append("needs_real_bid_margin")
    if team == "defense" and int(row["candidate_count_points"]) > 0:
        tags.append("setter_count_pressure")
        detectors.append("setter_count_before_certainty")
        readiness.append("needs_real_bid_margin")
    if role == "bidder_partner" and int(row["candidate_count_points"]) > 0:
        tags.append("partner_count_donation")
        detectors.append("partner_forcedness_and_safety")
        readiness.append("needs_belief_model")
    if bool(row["candidate_is_called_suit"]):
        tags.append("called_suit_pressure")
    if int(row["decl_id"]) == 7:
        tags.append("doubles_trump_regime")
        detectors.append("doubles_regime_plan")
    if int(row["decl_id"]) == 9:
        tags.append("no_trump_regime")
        detectors.append("no_trump_lead_control_and_support")
    if int(row["decision_idx"]) >= 4:
        detectors.append("first_trick_public_belief_update")
        readiness.append("public_context_ready")
    if row["hidden_threat_available"]:
        readiness.append("hidden_threat_offline_label_ready")

    if not readiness:
        readiness.append("context_only")
    return sorted(set(tags)), sorted(set(detectors)), sorted(set(readiness)), sorted(missing)


def distribution_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if bool(row["scalar_ev_lying_by_omission"]):
        tags.append("scalar_ev_lying_by_omission")
    if bool(row["high_variance_close_mean"]):
        tags.append("high_variance_close_mean")
    if int(row["branch_peak_count"]) >= 2:
        tags.append("multi_peak_pdf")
    if row["shelf_gap"] != "" and int(row["shelf_gap"]) >= 20:
        tags.append("wide_shelf_gap")
    if float(row["std"]) >= 15.0:
        tags.append("high_std")
    if float(row["lower_tail_mass_le_neg18"]) >= 0.25:
        tags.append("large_lower_tail")
    if float(row.get("top_hidden_impact_score") or 0.0) >= 5.0:
        tags.append("hidden_threat_large_impact")
    return tags


def annotate_decision_ranks(rows: list[dict[str, Any]]) -> None:
    by_decision: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_decision[(int(row["game_idx"]), int(row["decision_idx"]))].append(row)

    for decision_rows in by_decision.values():
        sorted_mean = sorted(decision_rows, key=lambda r: float(r["mean"]), reverse=True)
        sorted_threshold = sorted(decision_rows, key=lambda r: float(r["threshold_mass"]), reverse=True)
        sorted_tail = sorted(decision_rows, key=lambda r: float(r["lower_tail_mass_le_neg18"]))
        top_mean = sorted_mean[0]
        top_threshold = sorted_threshold[0]
        safest_tail = sorted_tail[0]
        std_values = sorted(float(row["std"]) for row in decision_rows)
        high_std_cutoff = std_values[max(0, int(0.75 * (len(std_values) - 1)))]
        has_choice = len(decision_rows) > 1

        mean_rank = {id(row): idx + 1 for idx, row in enumerate(sorted_mean)}
        threshold_rank = {id(row): idx + 1 for idx, row in enumerate(sorted_threshold)}
        tail_rank = {id(row): idx + 1 for idx, row in enumerate(sorted_tail)}

        for row in decision_rows:
            row["mean_rank_desc"] = mean_rank[id(row)]
            row["threshold_rank_desc"] = threshold_rank[id(row)]
            row["lower_tail_rank_asc"] = tail_rank[id(row)]
            row["mean_gap_to_best"] = float(top_mean["mean"]) - float(row["mean"])
            row["threshold_gap_to_best"] = float(top_threshold["threshold_mass"]) - float(row["threshold_mass"])
            row["lower_tail_gap_to_safest"] = float(row["lower_tail_mass_le_neg18"]) - float(
                safest_tail["lower_tail_mass_le_neg18"]
            )
            row["is_actual_action"] = int(row["candidate_slot"]) == int(row["actual_action_slot"])
            row["is_top_mean_action"] = row is top_mean
            row["is_top_threshold_action"] = row is top_threshold
            row["is_safest_lower_tail_action"] = row is safest_tail
            row["high_variance_close_mean"] = has_choice and (
                float(row["mean_gap_to_best"]) <= 1.0 and float(row["std"]) >= high_std_cutoff
            )
            row["scalar_ev_lying_by_omission"] = (
                has_choice
                and row is top_mean
                and (
                    top_threshold is not top_mean
                    or safest_tail is not top_mean
                    or float(top_mean["std"]) >= 15.0
                    or float(top_mean["lower_tail_mass_le_neg18"]) - float(safest_tail["lower_tail_mass_le_neg18"])
                    >= 0.05
                    or float(top_threshold["threshold_mass"]) - float(top_mean["threshold_mass"]) >= 0.03
                )
            )


def build_decision_rows(action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_decision: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        by_decision[(int(row["game_idx"]), int(row["decision_idx"]))].append(row)

    decision_rows: list[dict[str, Any]] = []
    for (_game_idx, _decision_idx), rows in sorted(by_decision.items()):
        top_mean = min((row for row in rows if row["mean_rank_desc"] == 1), key=lambda r: r["candidate_domino"])
        top_threshold = min((row for row in rows if row["threshold_rank_desc"] == 1), key=lambda r: r["candidate_domino"])
        safest_tail = min((row for row in rows if row["lower_tail_rank_asc"] == 1), key=lambda r: r["candidate_domino"])
        actual = next((row for row in rows if row["is_actual_action"]), rows[0])
        max_shelf_gap = max(int(row["shelf_gap"] or 0) for row in rows)
        decision_rows.append(
            {
                "game_idx": actual["game_idx"],
                "seed": actual["seed"],
                "decl_id": actual["decl_id"],
                "decl_name": actual["decl_name"],
                "bid_value": actual["bid_value"],
                "decision_idx": actual["decision_idx"],
                "trick_idx": actual["trick_idx"],
                "trick_position": actual["trick_position"],
                "actor": actual["actor"],
                "seat_role": actual["seat_role"],
                "team": actual["team"],
                "offense_score_before": actual["offense_score_before"],
                "defense_score_before": actual["defense_score_before"],
                "actual_action_domino": actual["actual_action_domino"],
                "legal_action_count": len(rows),
                "top_mean_domino": top_mean["candidate_domino"],
                "top_mean": top_mean["mean"],
                "top_threshold_domino": top_threshold["candidate_domino"],
                "top_threshold_mass": top_threshold["threshold_mass"],
                "safest_tail_domino": safest_tail["candidate_domino"],
                "safest_lower_tail_mass": safest_tail["lower_tail_mass_le_neg18"],
                "actual_mean_rank_desc": actual["mean_rank_desc"],
                "actual_threshold_rank_desc": actual["threshold_rank_desc"],
                "actual_lower_tail_rank_asc": actual["lower_tail_rank_asc"],
                "decision_scalar_ev_lying_by_omission": any(row["scalar_ev_lying_by_omission"] for row in rows),
                "decision_multi_peak_action_count": sum(int(row["branch_peak_count"]) >= 2 for row in rows),
                "decision_max_std": max(float(row["std"]) for row in rows),
                "decision_max_shelf_gap": max_shelf_gap,
                "decision_max_hidden_impact": max(float(row["top_hidden_impact_score"] or 0.0) for row in rows),
                "decision_max_hidden_downside": max(float(row["top_hidden_downside_score"] or 0.0) for row in rows),
                "decision_max_hidden_upside": max(float(row["top_hidden_upside_score"] or 0.0) for row in rows),
                "strategy_context_tags": "|".join(sorted({tag for row in rows for tag in str(row["strategy_context_tags"]).split("|") if tag})),
                "matched_position_detectors": "|".join(sorted({tag for row in rows for tag in str(row["matched_position_detectors"]).split("|") if tag})),
                "distribution_shape_tags": "|".join(sorted({tag for row in rows for tag in str(row["distribution_shape_tags"]).split("|") if tag})),
            }
        )
    return decision_rows


def compact_action(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "domino": row["candidate_domino"],
        "mean": round_or_none(row["mean"]),
        "std": round_or_none(row["std"]),
        "threshold_mass": round_or_none(row["threshold_mass"]),
        "lower_tail_mass": round_or_none(row["lower_tail_mass_le_neg18"]),
        "q10": row["q10"],
        "q50": row["q50"],
        "q90": row["q90"],
        "branch_peaks": row["branch_peak_bins"],
        "shelf_gap": row["shelf_gap"] if row["shelf_gap"] != "" else None,
        "top_hidden_impact": {
            "holder": row["top_hidden_impact_holder"],
            "domino": row["top_hidden_impact_domino"],
            "score": round_or_none(row["top_hidden_impact_score"]),
        },
        "tags": row["distribution_shape_tags"],
    }


def build_examples(decision_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    actions_by_decision: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        actions_by_decision[(int(row["game_idx"]), int(row["decision_idx"]))].append(row)

    ranked = sorted(
        decision_rows,
        key=lambda row: (
            bool(row["decision_scalar_ev_lying_by_omission"]),
            float(row["decision_max_hidden_impact"]),
            int(row["decision_multi_peak_action_count"]),
            float(row["decision_max_std"]),
        ),
        reverse=True,
    )
    examples: list[dict[str, Any]] = []
    for decision in ranked[:limit]:
        key = (int(decision["game_idx"]), int(decision["decision_idx"]))
        rows = sorted(actions_by_decision[key], key=lambda row: float(row["mean"]), reverse=True)
        examples.append(
            {
                "game_idx": decision["game_idx"],
                "seed": decision["seed"],
                "decl_name": decision["decl_name"],
                "decision_idx": decision["decision_idx"],
                "seat_role": decision["seat_role"],
                "team": decision["team"],
                "actual_action": decision["actual_action_domino"],
                "why_interesting": "scalar EV branch risk plus hidden ownership impact",
                "decision_flags": {
                    "scalar_ev_lying_by_omission": decision["decision_scalar_ev_lying_by_omission"],
                    "multi_peak_action_count": decision["decision_multi_peak_action_count"],
                    "max_hidden_impact": round_or_none(decision["decision_max_hidden_impact"]),
                },
                "actions_by_mean": [compact_action(row) for row in rows],
            }
        )
    return examples


def update_top_hidden_columns(action_row: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    action_row["hidden_threat_available"] = bool(rows)
    action_row["hidden_threat_rows"] = len(rows)
    if not rows:
        for kind in ("impact", "downside", "upside"):
            action_row[f"top_hidden_{kind}_holder"] = ""
            action_row[f"top_hidden_{kind}_domino"] = ""
            action_row[f"top_hidden_{kind}_score"] = ""
        return

    top_impact = max(rows, key=lambda row: float(row["impact_score"]))
    top_downside = max(rows, key=lambda row: float(row["downside_score"]))
    top_upside = max(rows, key=lambda row: float(row["upside_score"]))
    for kind, row, score_key in (
        ("impact", top_impact, "impact_score"),
        ("downside", top_downside, "downside_score"),
        ("upside", top_upside, "upside_score"),
    ):
        action_row[f"top_hidden_{kind}_holder"] = row["absolute_holder"]
        action_row[f"top_hidden_{kind}_domino"] = row["hidden_domino"]
        action_row[f"top_hidden_{kind}_score"] = row[score_key]


def process_payload(
    payload: dict[str, Any],
    *,
    input_path: Path,
    output_dir: Path,
    top_k: int,
    log_every_decisions: int,
    wb: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    games = payload.get("results", [])
    seeds = payload.get("seeds", [])
    decl_ids = payload.get("decl_ids", [])
    action_rows: list[dict[str, Any]] = []
    threat_rows: list[dict[str, Any]] = []
    skipped_no_joint = 0
    skipped_no_legal = 0
    inspected_decisions = 0
    processed_decisions = 0
    sample_counts: Counter[int] = Counter()
    running_scalar_omission = 0
    running_multi_peak_actions = 0
    running_high_std_actions = 0
    running_top_impacts: list[float] = []

    if getattr(wb, "run", None) is not None:
        wb.run.define_metric("progress/decisions_processed")
        for pattern in ("progress/*", "branch/*", "threat/*", "coverage/*"):
            wb.run.define_metric(pattern, step_metric="progress/decisions_processed")

    t0 = time.perf_counter()
    for game_idx, game in enumerate(games):
        hands = [[int(d) for d in hand] for hand in getattr(game, "hands")]
        decl_id = int(getattr(game, "decl_id", decl_ids[game_idx] if game_idx < len(decl_ids) else -1))
        seed = seeds[game_idx] if game_idx < len(seeds) else None
        bid_value = getattr(game, "bid_value", None)
        bid_value_source = "schema_v2_fixed_cli_value" if bid_value is not None else "missing"
        score = [0, 0]
        played_count_points = 0
        trick_plays: list[tuple[int, int]] = []
        played_slots: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}

        for decision_idx, decision in enumerate(getattr(game, "decisions", [])):
            inspected_decisions += 1
            actor = int(getattr(decision, "player"))
            context = move_context(
                decision_idx=decision_idx,
                actor=actor,
                decl_id=decl_id,
                score=score,
                played_count_points=played_count_points,
                trick_plays=trick_plays,
            )
            actual_slot = int(getattr(decision, "action_taken"))
            actual_domino = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1

            world_hands = getattr(decision, "world_hands", None)
            q_per_world = getattr(decision, "q_per_world", None)
            legal_mask = getattr(decision, "legal_mask", None)
            if world_hands is None or q_per_world is None:
                skipped_no_joint += 1
            else:
                world_hands = world_hands.detach().cpu().long()
                q_per_world = q_per_world.detach().cpu().float()

            legal = torch.as_tensor(legal_mask, dtype=torch.bool).detach().cpu()
            legal_slots = [slot for slot, is_legal in enumerate(legal.tolist()) if is_legal]
            if not legal_slots:
                skipped_no_legal += 1
            elif world_hands is not None and q_per_world is not None:
                processed_decisions += 1
                sample_counts[int(world_hands.shape[0])] += 1
                remaining = actor_remaining_hand(hands, played_slots, actor)
                legal_action_count = len(legal_slots)
                hand_called_count = sum(1 for d in remaining if tables.is_in_called_suit(d, decl_id))
                hand_double_count = sum(1 for d in remaining if tables.DOMINO_IS_DOUBLE[d])
                threshold_q = threshold_q_for_player(actor)

                decision_start = len(action_rows)
                for slot in legal_slots:
                    candidate_domino = hands[actor][slot]
                    row = {
                        "game_idx": game_idx,
                        "seed": "" if seed is None else int(seed),
                        "decl_id": decl_id,
                        "decl_name": DECL_NAMES.get(decl_id, f"unknown-{decl_id}"),
                        "bid_value": "" if bid_value is None else int(bid_value),
                        "bid_value_source": bid_value_source,
                        **{k: v for k, v in context.items() if k != "current_best_rank_before"},
                        "actual_action_slot": actual_slot,
                        "actual_action_domino_id": actual_domino,
                        "actual_action_domino": domino_label(actual_domino),
                        "candidate_slot": slot,
                        "candidate_domino_id": candidate_domino,
                        "candidate_domino": domino_label(candidate_domino),
                        "actor_hand_remaining_count": len(remaining),
                        "actor_hand_remaining_count_points": sum(tables.DOMINO_COUNT_POINTS[d] for d in remaining),
                        "actor_hand_remaining_called_suit_count": hand_called_count,
                        "actor_hand_remaining_double_count": hand_double_count,
                        "legal_action_count": legal_action_count,
                        "threshold_q": threshold_q,
                    }
                    row.update(candidate_public_facts(candidate_domino, context, decl_id))
                    row.update(q_stats(q_per_world[:, slot], threshold_q))
                    h_rows = hidden_threat_rows_for_action(
                        game_idx=game_idx,
                        seed=seed,
                        decl_id=decl_id,
                        decision_idx=decision_idx,
                        actor=actor,
                        action_slot=slot,
                        action_domino_id=candidate_domino,
                        world_hands=world_hands,
                        q_values=q_per_world[:, slot],
                        top_k=top_k,
                    )
                    threat_rows.extend(h_rows)
                    update_top_hidden_columns(row, h_rows)
                    action_rows.append(row)

                annotate_decision_ranks(action_rows[decision_start:])
                for row in action_rows[decision_start:]:
                    tags, detectors, readiness, missing = strategy_tags(row)
                    row["strategy_context_tags"] = "|".join(tags)
                    row["matched_position_detectors"] = "|".join(detectors)
                    row["direct_label_readiness"] = "|".join(readiness)
                    row["missing_direct_label_fields"] = "|".join(missing)
                    row["distribution_shape_tags"] = "|".join(distribution_tags(row))
                    running_scalar_omission += int(bool(row["scalar_ev_lying_by_omission"]))
                    running_multi_peak_actions += int(int(row["branch_peak_count"]) >= 2)
                    running_high_std_actions += int(float(row["std"]) >= 15.0)
                    if row["top_hidden_impact_score"] != "":
                        running_top_impacts.append(float(row["top_hidden_impact_score"]))

                if processed_decisions % max(log_every_decisions, 1) == 0:
                    n_actions = max(len(action_rows), 1)
                    wb.log(
                        {
                            "progress/decisions_processed": processed_decisions,
                            "progress/action_rows": len(action_rows),
                            "progress/threat_rows": len(threat_rows),
                            "progress/wall_seconds": time.perf_counter() - t0,
                            "branch/scalar_omission_action_rate": running_scalar_omission / n_actions,
                            "branch/multi_peak_action_rate": running_multi_peak_actions / n_actions,
                            "branch/high_std_action_rate": running_high_std_actions / n_actions,
                            "threat/mean_top_impact_score": (
                                sum(running_top_impacts) / len(running_top_impacts)
                                if running_top_impacts
                                else 0.0
                            ),
                            "threat/max_top_impact_score": max(running_top_impacts) if running_top_impacts else 0.0,
                            "coverage/joint_world_decision_rate": processed_decisions / max(inspected_decisions, 1),
                        },
                        step=processed_decisions,
                    )

            if actual_domino >= 0:
                played_slots[actor].add(actual_slot)
                played_count_points += int(tables.DOMINO_COUNT_POINTS[actual_domino])
                trick_plays.append((actor, actual_domino))
                if len(trick_plays) == 4:
                    winner, _led_suit, _best_rank = current_winner(trick_plays, decl_id)
                    if winner is not None:
                        trick_points = 1 + sum(tables.DOMINO_COUNT_POINTS[d] for _p, d in trick_plays)
                        score[team_id_for_player(winner)] += int(trick_points)
                    trick_plays = []

    annotate_decision_ranks(action_rows)
    for row in action_rows:
        tags, detectors, readiness, missing = strategy_tags(row)
        row["strategy_context_tags"] = "|".join(tags)
        row["matched_position_detectors"] = "|".join(detectors)
        row["direct_label_readiness"] = "|".join(readiness)
        row["missing_direct_label_fields"] = "|".join(missing)
        row["distribution_shape_tags"] = "|".join(distribution_tags(row))

    decision_rows = build_decision_rows(action_rows)
    summary = summarize(
        input_path=input_path,
        payload=payload,
        action_rows=action_rows,
        decision_rows=decision_rows,
        threat_rows=threat_rows,
        inspected_decisions=inspected_decisions,
        processed_decisions=processed_decisions,
        skipped_no_joint=skipped_no_joint,
        skipped_no_legal=skipped_no_legal,
        sample_counts=sample_counts,
        output_dir=output_dir,
        started_seconds=time.perf_counter() - t0,
        wb=wb,
    )
    return action_rows, decision_rows, threat_rows, summary


def summarize(
    *,
    input_path: Path,
    payload: dict[str, Any],
    action_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    threat_rows: list[dict[str, Any]],
    inspected_decisions: int,
    processed_decisions: int,
    skipped_no_joint: int,
    skipped_no_legal: int,
    sample_counts: Counter[int],
    output_dir: Path,
    started_seconds: float,
    wb: Any,
) -> dict[str, Any]:
    action_tag_counts = Counter(
        tag for row in action_rows for tag in str(row["distribution_shape_tags"]).split("|") if tag
    )
    detector_counts = Counter(
        tag for row in action_rows for tag in str(row["matched_position_detectors"]).split("|") if tag
    )
    role_counts = Counter(str(row["seat_role"]) for row in decision_rows)
    top_impacts = [float(row["top_hidden_impact_score"]) for row in action_rows if row["top_hidden_impact_score"] != ""]
    scalar_omission_decisions = sum(bool(row["decision_scalar_ev_lying_by_omission"]) for row in decision_rows)
    multi_peak_decisions = sum(int(row["decision_multi_peak_action_count"]) > 0 for row in decision_rows)
    actual_top_mean = sum(int(row["actual_mean_rank_desc"]) == 1 for row in decision_rows)
    actual_top_threshold = sum(int(row["actual_threshold_rank_desc"]) == 1 for row in decision_rows)
    actual_safest_tail = sum(int(row["actual_lower_tail_rank_asc"]) == 1 for row in decision_rows)
    return {
        "schema_version": "w42.branch_atlas_v1.summary.v0",
        "bead_id": "t42-gc7m",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repo_commit": git_sha(),
        "git_status_before_artifacts": git_status_short(),
        "command": " ".join(sys.argv),
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "input_games": len(payload.get("results", [])),
            "n_samples": payload.get("n_samples"),
            "schema": payload.get("schema"),
            "start_seed": payload.get("start_seed"),
            "seeds": payload.get("seeds"),
            "decl_ids": payload.get("decl_ids"),
            "checkpoint": payload.get("checkpoint"),
            "bid_value_status": "schema_v2_fixed_cli_value" if payload.get("schema") == "v2" else "not present",
        },
        "coverage": {
            "inspected_decisions": inspected_decisions,
            "processed_joint_world_decisions": processed_decisions,
            "skipped_no_joint_worlds": skipped_no_joint,
            "skipped_no_legal_actions": skipped_no_legal,
            "joint_world_decision_rate": processed_decisions / max(inspected_decisions, 1),
            "sample_counts_by_decision": dict(sorted(sample_counts.items())),
        },
        "rows": {
            "decision_rows": len(decision_rows),
            "action_rows": len(action_rows),
            "hidden_threat_rows": len(threat_rows),
        },
        "branch_findings": {
            "decision_scalar_ev_lying_by_omission_count": scalar_omission_decisions,
            "decision_scalar_ev_lying_by_omission_rate": scalar_omission_decisions / max(len(decision_rows), 1),
            "decision_multi_peak_pdf_count": multi_peak_decisions,
            "decision_multi_peak_pdf_rate": multi_peak_decisions / max(len(decision_rows), 1),
            "action_distribution_tag_counts": dict(sorted(action_tag_counts.items())),
            "actual_action_top_mean_count": actual_top_mean,
            "actual_action_top_threshold_count": actual_top_threshold,
            "actual_action_safest_lower_tail_count": actual_safest_tail,
            "max_std": max((float(row["decision_max_std"]) for row in decision_rows), default=0.0),
            "max_hidden_impact": max(top_impacts, default=0.0),
            "mean_top_hidden_impact": sum(top_impacts) / len(top_impacts) if top_impacts else 0.0,
        },
        "strategy_surface": {
            "role_counts": dict(sorted(role_counts.items())),
            "detector_counts": dict(sorted(detector_counts.items())),
            "direct_label_readiness_values": sorted(
                {
                    tag
                    for row in action_rows
                    for tag in str(row["direct_label_readiness"]).split("|")
                    if tag
                }
            ),
            "missing_fields_all_rows": sorted(
                {
                    tag
                    for row in action_rows
                    for tag in str(row["missing_direct_label_fields"]).split("|")
                    if tag
                }
            ),
        },
        "scientific_status": {
            "training_run": False,
            "wandb": wb.status(),
            "hf": "not applicable",
            "claim_ledger_impact": "no claim-ledger movement",
            "interpretation": (
                "Powered branch-atlas pilot. It measures branch shape and offline hidden-owner "
                "impact on a generated joint-world E[Q] slice; it does not validate book claims "
                "or train a policy by itself."
            ),
            "leakage_boundary": (
                "Hidden domino ownership and per-world Q outcomes are offline diagnostic labels. "
                "They may supervise belief quality or reports, but they are not live policy inputs."
            ),
        },
        "wall_seconds": started_seconds,
        "artifacts": {
            "decision_actions_csv": str(output_dir / "decision_actions.csv"),
            "decision_states_csv": str(output_dir / "decision_states.csv"),
            "hidden_threat_rows_csv": str(output_dir / "hidden_threat_rows.csv"),
            "examples_json": str(output_dir / "examples.json"),
            "manifest_json": str(output_dir / "manifest.json"),
        },
    }


def build_manifest(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "w42.branch_atlas_v1.manifest.v0",
        "bead_id": "t42-gc7m",
        "created_at_utc": summary["created_at_utc"],
        "input": summary["source"],
        "outputs": summary["artifacts"],
        "config": {
            "top_k_hidden_threats_per_action": args.top_k,
            "log_every_decisions": args.log_every_decisions,
            "examples": args.examples,
            "wandb_project": args.wandb_project,
            "wandb_group": args.wandb_group,
            "wandb_name": args.wandb_name,
            "wandb_mode": args.wandb_mode,
        },
        "feature_groups": {
            "online_safe_context": [
                "seat_role",
                "trick_position",
                "public score",
                "played count",
                "current trick winner",
                "candidate action facts",
            ],
            "actor_private_context": [
                "actor remaining hand facts",
                "candidate legal action",
            ],
            "offline_distribution_labels": [
                "q_per_world-derived PDF shape",
                "threshold mass",
                "lower-tail mass",
                "quantiles",
                "CVaR",
                "scalar-EV omission flags",
            ],
            "offline_hidden_threat_labels": [
                "conditioned hidden domino holder impact",
                "top downside/upside hidden drivers",
                "holder/domino impact rows",
            ],
        },
        "leakage_boundary": summary["scientific_status"]["leakage_boundary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Saved .pt artifact from forge.eq.generate --save-joint-worlds")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--log-every-decisions", type=int, default=4)
    add_wandb_args(
        parser,
        default_project="w42",
        default_group="w42-powered-branch-atlas-v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    config = {
        "bead_id": "t42-gc7m",
        "git_sha": sha,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input) if args.input.exists() else "missing",
        "top_k": args.top_k,
        "log_every_decisions": args.log_every_decisions,
        "experiment": "w42-powered-branch-atlas-v1",
        "model_family": "not applicable; report/label artifact",
        "claim_ledger_status_before": "no claim-ledger movement",
        "hf_repo_id": "not applicable",
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=args.output_dir,
        tags=[
            "w42",
            "branch-atlas",
            "eq-pdf",
            "joint-worlds",
            "hidden-threat-attribution",
            "t42-gc7m",
        ],
    )
    try:
        payload = torch.load(args.input, weights_only=False, map_location="cpu")
        action_rows, decision_rows, threat_rows, summary = process_payload(
            payload,
            input_path=args.input,
            output_dir=args.output_dir,
            top_k=args.top_k,
            log_every_decisions=args.log_every_decisions,
            wb=wb,
        )
        examples = build_examples(decision_rows, action_rows, args.examples)
        manifest = build_manifest(args, summary)

        write_csv(args.output_dir / "decision_actions.csv", action_rows, ACTION_COLUMNS)
        write_csv(args.output_dir / "decision_states.csv", decision_rows, DECISION_COLUMNS)
        write_csv(args.output_dir / "hidden_threat_rows.csv", threat_rows, THREAT_COLUMNS)
        write_jsonl(args.output_dir / "hidden_threat_rows.jsonl", threat_rows)
        write_json(args.output_dir / "examples.json", examples)
        write_json(args.output_dir / "manifest.json", manifest)
        write_json(args.output_dir / "summary.json", summary)

        wb.log(
            {
                "progress/decisions_processed": summary["coverage"]["processed_joint_world_decisions"],
                "progress/action_rows": summary["rows"]["action_rows"],
                "progress/threat_rows": summary["rows"]["hidden_threat_rows"],
                "branch/scalar_omission_decision_rate": summary["branch_findings"][
                    "decision_scalar_ev_lying_by_omission_rate"
                ],
                "branch/multi_peak_decision_rate": summary["branch_findings"]["decision_multi_peak_pdf_rate"],
                "threat/max_hidden_impact": summary["branch_findings"]["max_hidden_impact"],
                "threat/mean_top_hidden_impact": summary["branch_findings"]["mean_top_hidden_impact"],
                "coverage/joint_world_decision_rate": summary["coverage"]["joint_world_decision_rate"],
            },
            step=summary["coverage"]["processed_joint_world_decisions"],
        )
        wb.update_summary(
            {
                "status": "completed",
                "processed_joint_world_decisions": summary["coverage"]["processed_joint_world_decisions"],
                "action_rows": summary["rows"]["action_rows"],
                "hidden_threat_rows": summary["rows"]["hidden_threat_rows"],
                "scalar_omission_decision_rate": summary["branch_findings"][
                    "decision_scalar_ev_lying_by_omission_rate"
                ],
                "max_hidden_impact": summary["branch_findings"]["max_hidden_impact"],
                "claim_ledger_impact": "no claim-ledger movement",
                "hf_links": "not applicable",
            }
        )
        wb.log_artifact_files(
            name=f"w42-branch-atlas-v1-{sha[:8]}",
            artifact_type="w42-branch-atlas",
            paths=[
                args.output_dir / "summary.json",
                args.output_dir / "manifest.json",
                args.output_dir / "examples.json",
                args.output_dir / "decision_states.csv",
                args.output_dir / "decision_actions.csv",
                args.output_dir / "hidden_threat_rows.csv",
            ],
        )
        wb.finish()
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception:
        wb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
