#!/usr/bin/env python3
"""Build the w42 phase-2 decision table from E[Q] visualizer JSONL records.

The table is intentionally a bridge artifact. It joins the legal-action PDFs
from the browser visualizer to public trick context, seat/role vocabulary, and
explicit readiness columns for direct tactical labels and hidden-threat
attribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle import schema, tables
from forge.oracle.tables import can_follow, led_suit_for_lead_domino, trick_rank

from w42.distribution_aware_ev_report.build_distribution_aware_ev_report import (
    EQ_MAX,
    EQ_MIN,
    cvar_low,
    entropy_bits,
    local_peak_bins,
    pdf_mass,
    quantile,
    shelf_gap,
)


DEFAULT_INPUT = ROOT / "forge" / "analysis" / "results" / "data" / "eq_pdf_v3_sample.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

OFFENSE_PLAYERS = {0, 2}
SEAT_ROLE = {
    0: "bidder",
    1: "left_setter",
    2: "bidder_partner",
    3: "right_setter",
}
PARTNER = {0: 2, 2: 0, 1: 3, 3: 1}
TOTAL_COUNT_POINTS = sum(tables.DOMINO_COUNT_POINTS)


ACTION_COLUMNS = [
    "game_id",
    "game_idx",
    "trump_id",
    "trump_name",
    "move_idx",
    "trick_idx",
    "trick_position",
    "active_player",
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
    "actual_action_global_idx",
    "actual_action_domino_id",
    "actual_action_domino",
    "candidate_global_idx",
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
    "legal_count_action_count",
    "legal_called_suit_action_count",
    "legal_double_action_count",
    "mean",
    "std",
    "visualizer_threshold_mass",
    "make_mass_ge_18_offense_only",
    "defense_threshold_mass_gt_neg18_existing",
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
    "converged",
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
    "strategy_context_tags",
    "matched_position_detectors",
    "distribution_shape_tags",
    "direct_label_readiness",
    "missing_direct_label_fields",
    "hidden_threat_available",
    "hidden_threat_source",
    "top_hidden_threat_holder",
    "top_hidden_threat_domino",
    "top_hidden_threat_impact",
]


DECISION_COLUMNS = [
    "game_id",
    "game_idx",
    "trump_id",
    "trump_name",
    "move_idx",
    "trick_idx",
    "trick_position",
    "active_player",
    "seat_role",
    "team",
    "offense_score_before",
    "defense_score_before",
    "current_trick_count_before",
    "current_winner_before",
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
    "actual_mean_gap_to_best",
    "actual_threshold_gap_to_best",
    "actual_lower_tail_gap_to_safest",
    "decision_scalar_ev_lying_by_omission",
    "decision_high_variance_close_mean_count",
    "decision_multi_peak_action_count",
    "decision_max_shelf_gap",
    "decision_max_std",
    "decision_min_cvar_low_10",
    "strategy_context_tags",
    "matched_position_detectors",
    "direct_label_readiness",
    "missing_direct_label_fields",
    "hidden_threat_available",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def team_for_player(player: int) -> str:
    return "offense" if player in OFFENSE_PLAYERS else "defense"


def team_id_for_player(player: int) -> int:
    return 0 if player in OFFENSE_PLAYERS else 1


def domino_label(domino_id: int) -> str:
    high, low = schema.domino_pips(domino_id)
    return f"{high}-{low}"


def led_suit_name(led_suit: int | None) -> str:
    if led_suit is None:
        return ""
    if led_suit == 7:
        return "called"
    return str(led_suit)


def global_lookup(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for player in record["players"]:
        player_id = int(player["id"])
        for domino in player["dominoes"]:
            slot = int(domino["slot"])
            global_idx = player_id * 7 + slot
            domino_id = int(domino["id"])
            lookup[global_idx] = {
                "player": player_id,
                "slot": slot,
                "domino_id": domino_id,
                "domino": domino["pips"],
            }
    return lookup


def actual_play_by_move(record: dict[str, Any]) -> dict[int, int]:
    by_move: dict[int, int] = {}
    for global_idx, row in enumerate(record.get("domino_played", [])):
        for move_idx, played in enumerate(row):
            if played:
                by_move[move_idx] = global_idx
    return by_move


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


def build_move_contexts(record: dict[str, Any], lookup: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    decl_id = int(record["trump_id"])
    actual_by_move = actual_play_by_move(record)
    active_players = [int(p) for p in record.get("active_player", [])]
    contexts: dict[int, dict[str, Any]] = {}
    trick_plays: list[tuple[int, int]] = []
    played_count_points = 0

    for move_idx, active_player in enumerate(active_players):
        score_before = record.get("score_history", [])[move_idx - 1] if move_idx > 0 else [0, 0]
        winner, led_suit, best_rank = current_winner(trick_plays, decl_id)
        current_trick_count = sum(tables.DOMINO_COUNT_POINTS[d] for _p, d in trick_plays)
        partner = PARTNER[active_player]
        contexts[move_idx] = {
            "move_idx": move_idx,
            "trick_idx": move_idx // 4,
            "trick_position": len(trick_plays),
            "active_player": active_player,
            "seat_role": SEAT_ROLE[active_player],
            "team": team_for_player(active_player),
            "partner_player": partner,
            "offense_score_before": int(score_before[0]),
            "defense_score_before": int(score_before[1]),
            "played_count_points_before": played_count_points,
            "unplayed_count_points_public_before": TOTAL_COUNT_POINTS - played_count_points,
            "current_trick_count_before": current_trick_count,
            "led_suit_before": "" if led_suit is None else led_suit,
            "led_suit_name_before": led_suit_name(led_suit),
            "current_winner_before": "" if winner is None else winner,
            "current_winner_team_before": "" if winner is None else team_for_player(winner),
            "current_best_rank_before": "" if best_rank is None else best_rank,
            "partner_currently_winning_before": bool(winner == partner),
            "opponent_currently_winning_before": bool(
                winner is not None and team_id_for_player(winner) != team_id_for_player(active_player)
            ),
            "actual_action_global_idx": actual_by_move.get(move_idx, ""),
        }

        actual_global_idx = actual_by_move.get(move_idx)
        if actual_global_idx is not None:
            domino_id = int(lookup[actual_global_idx]["domino_id"])
            played_count_points += tables.DOMINO_COUNT_POINTS[domino_id]
            trick_plays.append((active_player, domino_id))
            if len(trick_plays) == 4:
                trick_plays = []

    return contexts


def actor_remaining_hand(
    record: dict[str, Any], lookup: dict[int, dict[str, Any]], move_idx: int, player: int
) -> list[int]:
    played = set()
    for global_idx, row in enumerate(record.get("domino_played", [])):
        if lookup.get(global_idx, {}).get("player") != player:
            continue
        for played_move, did_play in enumerate(row):
            if did_play and played_move < move_idx:
                played.add(global_idx)
    remaining = []
    for global_idx, meta in lookup.items():
        if int(meta["player"]) == player and global_idx not in played:
            remaining.append(int(meta["domino_id"]))
    return remaining


def legal_action_summaries(
    legal_rows: list[dict[str, Any]], lookup: dict[int, dict[str, Any]], decl_id: int
) -> dict[str, int]:
    domino_ids = [int(lookup[int(row["d"])]["domino_id"]) for row in legal_rows]
    return {
        "legal_action_count": len(domino_ids),
        "legal_count_action_count": sum(1 for d in domino_ids if tables.DOMINO_COUNT_POINTS[d] > 0),
        "legal_called_suit_action_count": sum(1 for d in domino_ids if tables.is_in_called_suit(d, decl_id)),
        "legal_double_action_count": sum(1 for d in domino_ids if tables.DOMINO_IS_DOUBLE[d]),
    }


def candidate_public_facts(
    candidate_domino_id: int, context: dict[str, Any], decl_id: int
) -> dict[str, Any]:
    led_suit = context["led_suit_before"]
    candidate_rank = ""
    can_follow_led: bool | str = ""
    beats_current: bool | str = ""
    would_win: bool = True
    if led_suit != "":
        led_suit_int = int(led_suit)
        candidate_rank = trick_rank(candidate_domino_id, led_suit_int, decl_id)
        can_follow_led = can_follow(candidate_domino_id, led_suit_int, decl_id)
        current_best = int(context["current_best_rank_before"])
        beats_current = candidate_rank > current_best
        would_win = bool(beats_current)
    return {
        "candidate_count_points": tables.DOMINO_COUNT_POINTS[candidate_domino_id],
        "candidate_is_double": bool(tables.DOMINO_IS_DOUBLE[candidate_domino_id]),
        "candidate_is_called_suit": bool(tables.is_in_called_suit(candidate_domino_id, decl_id)),
        "candidate_can_follow_led": can_follow_led,
        "candidate_beats_current": beats_current,
        "candidate_would_win_trick_now": would_win,
        "candidate_trick_rank": candidate_rank,
    }


def pdf_features(entry: dict[str, Any], player: int) -> dict[str, Any]:
    pdf = [float(p) for p in entry["pdf"]]
    q10 = quantile(pdf, 0.10)
    q25 = quantile(pdf, 0.25)
    q50 = quantile(pdf, 0.50)
    q75 = quantile(pdf, 0.75)
    q90 = quantile(pdf, 0.90)
    peaks = local_peak_bins(pdf)
    return {
        "mean": float(entry["mean"]),
        "std": float(entry.get("std") or 0.0),
        "visualizer_threshold_mass": float(entry.get("win") or 0.0),
        "make_mass_ge_18_offense_only": pdf_mass(pdf, lo=18) if player in OFFENSE_PLAYERS else "",
        "defense_threshold_mass_gt_neg18_existing": pdf_mass(pdf, lo=-17)
        if player not in OFFENSE_PLAYERS
        else "",
        "lower_tail_mass_le_neg18": pdf_mass(pdf, hi=-18),
        "q10": q10,
        "q25": q25,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "iqr": q75 - q25,
        "cvar_low_10": cvar_low(pdf, 0.10),
        "branch_entropy": entropy_bits(pdf),
        "branch_peak_count": len(peaks),
        "branch_peak_bins": "|".join(str(p) for p in peaks),
        "shelf_gap": shelf_gap(pdf) or "",
        "samples": int(entry.get("samples") or 0),
        "converged": entry.get("converged"),
    }


def strategy_tags(row: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    tags: list[str] = []
    detectors: list[str] = []
    readiness: list[str] = []
    missing: set[str] = {"bid_amount", "bid_margin", "hidden_world_ownership"}

    player = int(row["active_player"])
    trick_idx = int(row["trick_idx"])
    trick_position = int(row["trick_position"])
    team = row["team"]
    role = row["seat_role"]

    if trick_idx == 0:
        tags.append("first_trick")
        if player == 0 and trick_position == 0:
            tags.append("bidder_opening_lead")
            detectors.append("bidder_first_lead_plan")
            readiness.append("public_context_ready")
        elif player == 1 and trick_position == 1:
            tags.append("first_setter_response")
            detectors.append("first_setter_pounce_window")
            readiness.append("needs_bid_and_set_margin")
        elif player == 2 and trick_position == 2:
            tags.append("partner_third_seat_support")
            detectors.append("partner_third_seat_safe_donation")
            readiness.append("public_context_ready")
        elif player == 3 and trick_position == 3:
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
        readiness.append("needs_bid_and_set_margin")
    if team == "defense" and int(row["candidate_count_points"]) > 0:
        tags.append("setter_count_pressure")
        detectors.append("setter_count_before_certainty")
        readiness.append("needs_bid_and_set_margin")
    if role == "bidder_partner" and int(row["candidate_count_points"]) > 0:
        tags.append("partner_count_donation")
        detectors.append("partner_forcedness_and_safety")
        readiness.append("needs_overtrump_belief")
    if row["candidate_is_called_suit"] == "True" or row["candidate_is_called_suit"] is True:
        tags.append("called_suit_pressure")
    if int(row["trump_id"]) == 7:
        tags.append("doubles_trump_regime")
        detectors.append("doubles_regime_plan")
    if int(row["trump_id"]) == 9:
        tags.append("no_trump_regime")
        detectors.append("no_trump_lead_control_and_support")
    if int(row["move_idx"]) >= 4:
        detectors.append("first_trick_public_belief_update")
        readiness.append("public_context_ready")

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
    return tags


def build_action_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        decl_id = int(record["trump_id"])
        lookup = global_lookup(record)
        contexts = build_move_contexts(record, lookup)
        pdf_by_move: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for entry in record.get("pdf_data", []):
            pdf_by_move[int(entry["m"])].append(entry)

        for move_idx, entries in sorted(pdf_by_move.items()):
            context = contexts[move_idx]
            active_player = int(context["active_player"])
            actual_global_idx = context["actual_action_global_idx"]
            actual_meta = lookup[int(actual_global_idx)] if actual_global_idx != "" else None
            hand_remaining = actor_remaining_hand(record, lookup, move_idx, active_player)
            hand_called_count = sum(1 for d in hand_remaining if tables.is_in_called_suit(d, decl_id))
            hand_double_count = sum(1 for d in hand_remaining if tables.DOMINO_IS_DOUBLE[d])
            legal_counts = legal_action_summaries(entries, lookup, decl_id)

            for entry in entries:
                global_idx = int(entry["d"])
                meta = lookup[global_idx]
                domino_id = int(meta["domino_id"])
                base: dict[str, Any] = {
                    "game_id": record["game_id"],
                    "game_idx": record.get("game_idx"),
                    "trump_id": record.get("trump_id"),
                    "trump_name": record.get("trump_name"),
                    **{k: v for k, v in context.items() if k != "current_best_rank_before"},
                    "actual_action_domino_id": actual_meta["domino_id"] if actual_meta else "",
                    "actual_action_domino": actual_meta["domino"] if actual_meta else "",
                    "candidate_global_idx": global_idx,
                    "candidate_domino_id": domino_id,
                    "candidate_domino": meta["domino"],
                    "actor_hand_remaining_count": len(hand_remaining),
                    "actor_hand_remaining_count_points": sum(
                        tables.DOMINO_COUNT_POINTS[d] for d in hand_remaining
                    ),
                    "actor_hand_remaining_called_suit_count": hand_called_count,
                    "actor_hand_remaining_double_count": hand_double_count,
                    "hidden_threat_available": False,
                    "hidden_threat_source": "missing_joint_world_ownership_artifact",
                    "top_hidden_threat_holder": "",
                    "top_hidden_threat_domino": "",
                    "top_hidden_threat_impact": "",
                }
                base.update(legal_counts)
                base.update(candidate_public_facts(domino_id, context, decl_id))
                base.update(pdf_features(entry, active_player))
                rows.append(base)

    annotate_decision_ranks(rows)
    for row in rows:
        tags, detectors, readiness, missing = strategy_tags(row)
        row["strategy_context_tags"] = "|".join(tags)
        row["matched_position_detectors"] = "|".join(detectors)
        row["direct_label_readiness"] = "|".join(readiness)
        row["missing_direct_label_fields"] = "|".join(missing)
        row["distribution_shape_tags"] = "|".join(distribution_tags(row))
    return rows


def annotate_decision_ranks(rows: list[dict[str, Any]]) -> None:
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_decision[(row["game_id"], int(row["move_idx"]))].append(row)

    for decision_rows in by_decision.values():
        sorted_mean = sorted(decision_rows, key=lambda r: float(r["mean"]), reverse=True)
        sorted_threshold = sorted(
            decision_rows, key=lambda r: float(r["visualizer_threshold_mass"]), reverse=True
        )
        sorted_tail = sorted(decision_rows, key=lambda r: float(r["lower_tail_mass_le_neg18"]))
        top_mean = sorted_mean[0]
        top_threshold = sorted_threshold[0]
        safest_tail = sorted_tail[0]
        std_values = sorted(float(r["std"]) for r in decision_rows)
        high_std_cutoff = std_values[max(0, int(0.75 * (len(std_values) - 1)))]

        mean_rank = {id(row): idx + 1 for idx, row in enumerate(sorted_mean)}
        threshold_rank = {id(row): idx + 1 for idx, row in enumerate(sorted_threshold)}
        tail_rank = {id(row): idx + 1 for idx, row in enumerate(sorted_tail)}
        has_choice = len(decision_rows) > 1

        for row in decision_rows:
            row["mean_rank_desc"] = mean_rank[id(row)]
            row["threshold_rank_desc"] = threshold_rank[id(row)]
            row["lower_tail_rank_asc"] = tail_rank[id(row)]
            row["mean_gap_to_best"] = float(top_mean["mean"]) - float(row["mean"])
            row["threshold_gap_to_best"] = float(top_threshold["visualizer_threshold_mass"]) - float(
                row["visualizer_threshold_mass"]
            )
            row["lower_tail_gap_to_safest"] = float(row["lower_tail_mass_le_neg18"]) - float(
                safest_tail["lower_tail_mass_le_neg18"]
            )
            row["is_actual_action"] = row["candidate_global_idx"] == row["actual_action_global_idx"]
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
                    or float(top_mean["lower_tail_mass_le_neg18"])
                    - float(safest_tail["lower_tail_mass_le_neg18"])
                    >= 0.05
                    or float(top_threshold["visualizer_threshold_mass"])
                    - float(top_mean["visualizer_threshold_mass"])
                    >= 0.03
                )
            )


def build_decision_rows(action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        by_decision[(row["game_id"], int(row["move_idx"]))].append(row)

    decision_rows: list[dict[str, Any]] = []
    for (_game_id, _move_idx), rows in sorted(by_decision.items()):
        top_mean = min(rows, key=lambda r: int(r["mean_rank_desc"]))
        top_threshold = min(rows, key=lambda r: int(r["threshold_rank_desc"]))
        safest_tail = min(rows, key=lambda r: int(r["lower_tail_rank_asc"]))
        actual = next((r for r in rows if r["is_actual_action"]), rows[0])
        multi_peak_count = sum(1 for r in rows if int(r["branch_peak_count"]) >= 2)
        shelf_gaps = [int(r["shelf_gap"]) for r in rows if r["shelf_gap"] != ""]
        tags = sorted({tag for r in rows for tag in str(r["strategy_context_tags"]).split("|") if tag})
        detectors = sorted(
            {tag for r in rows for tag in str(r["matched_position_detectors"]).split("|") if tag}
        )
        readiness = sorted(
            {tag for r in rows for tag in str(r["direct_label_readiness"]).split("|") if tag}
        )
        missing = sorted(
            {tag for r in rows for tag in str(r["missing_direct_label_fields"]).split("|") if tag}
        )
        decision_rows.append(
            {
                "game_id": actual["game_id"],
                "game_idx": actual["game_idx"],
                "trump_id": actual["trump_id"],
                "trump_name": actual["trump_name"],
                "move_idx": actual["move_idx"],
                "trick_idx": actual["trick_idx"],
                "trick_position": actual["trick_position"],
                "active_player": actual["active_player"],
                "seat_role": actual["seat_role"],
                "team": actual["team"],
                "offense_score_before": actual["offense_score_before"],
                "defense_score_before": actual["defense_score_before"],
                "current_trick_count_before": actual["current_trick_count_before"],
                "current_winner_before": actual["current_winner_before"],
                "actual_action_domino": actual["actual_action_domino"],
                "legal_action_count": len(rows),
                "top_mean_domino": top_mean["candidate_domino"],
                "top_mean": top_mean["mean"],
                "top_threshold_domino": top_threshold["candidate_domino"],
                "top_threshold_mass": top_threshold["visualizer_threshold_mass"],
                "safest_tail_domino": safest_tail["candidate_domino"],
                "safest_lower_tail_mass": safest_tail["lower_tail_mass_le_neg18"],
                "actual_mean_rank_desc": actual["mean_rank_desc"],
                "actual_threshold_rank_desc": actual["threshold_rank_desc"],
                "actual_lower_tail_rank_asc": actual["lower_tail_rank_asc"],
                "actual_mean_gap_to_best": actual["mean_gap_to_best"],
                "actual_threshold_gap_to_best": actual["threshold_gap_to_best"],
                "actual_lower_tail_gap_to_safest": actual["lower_tail_gap_to_safest"],
                "decision_scalar_ev_lying_by_omission": any(
                    bool(r["scalar_ev_lying_by_omission"]) for r in rows
                ),
                "decision_high_variance_close_mean_count": sum(
                    1 for r in rows if bool(r["high_variance_close_mean"])
                ),
                "decision_multi_peak_action_count": multi_peak_count,
                "decision_max_shelf_gap": max(shelf_gaps) if shelf_gaps else "",
                "decision_max_std": max(float(r["std"]) for r in rows),
                "decision_min_cvar_low_10": min(float(r["cvar_low_10"]) for r in rows),
                "strategy_context_tags": "|".join(tags),
                "matched_position_detectors": "|".join(detectors),
                "direct_label_readiness": "|".join(readiness),
                "missing_direct_label_fields": "|".join(missing),
                "hidden_threat_available": False,
            }
        )
    return decision_rows


def compact_examples(decision_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in action_rows:
        by_decision[(row["game_id"], int(row["move_idx"]))].append(row)

    interesting = sorted(
        decision_rows,
        key=lambda row: (
            bool(row["decision_scalar_ev_lying_by_omission"]),
            int(row["decision_multi_peak_action_count"]),
            float(row["decision_max_std"]),
            int(row["legal_action_count"]),
        ),
        reverse=True,
    )
    examples: list[dict[str, Any]] = []
    for row in interesting[:12]:
        actions = sorted(
            by_decision[(row["game_id"], int(row["move_idx"]))],
            key=lambda r: int(r["mean_rank_desc"]),
        )
        examples.append(
            {
                "game_id": row["game_id"],
                "move_idx": int(row["move_idx"]),
                "seat_role": row["seat_role"],
                "trick_position": int(row["trick_position"]),
                "tags": row["strategy_context_tags"].split("|")
                if row["strategy_context_tags"]
                else [],
                "actual_action": row["actual_action_domino"],
                "top_mean": row["top_mean_domino"],
                "top_threshold": row["top_threshold_domino"],
                "safest_tail": row["safest_tail_domino"],
                "scalar_ev_lying_by_omission": row["decision_scalar_ev_lying_by_omission"],
                "actions_by_mean": [
                    {
                        "domino": action["candidate_domino"],
                        "actual": bool(action["is_actual_action"]),
                        "mean": round(float(action["mean"]), 4),
                        "std": round(float(action["std"]), 4),
                        "threshold_mass": round(float(action["visualizer_threshold_mass"]), 4),
                        "lower_tail_mass": round(float(action["lower_tail_mass_le_neg18"]), 4),
                        "q10": int(action["q10"]),
                        "q50": int(action["q50"]),
                        "q90": int(action["q90"]),
                        "shelf_gap": action["shelf_gap"],
                        "distribution_tags": action["distribution_shape_tags"].split("|")
                        if action["distribution_shape_tags"]
                        else [],
                    }
                    for action in actions
                ],
            }
        )
    return examples


def schema_doc() -> dict[str, Any]:
    return {
        "schema_version": "w42.phase2_decision_table.v0",
        "grain": {
            "decision_actions.csv": "one legal action with an E[Q] PDF for one public decision state",
            "decision_states.csv": "one public decision state with top-mean, top-threshold, safest-tail, and actual-action summary",
        },
        "feature_groups": {
            "public_context": [
                "game_id",
                "trump_name",
                "move_idx",
                "trick_idx",
                "trick_position",
                "score before action",
                "played/unplayed public count",
                "current trick winner/count/led suit",
            ],
            "private_actor_context": [
                "actor remaining hand count",
                "actor remaining count points",
                "actor remaining called-suit count",
                "actor remaining doubles",
            ],
            "candidate_action": [
                "candidate domino",
                "count points",
                "double/called-suit flags",
                "follow/beat/current trick flags",
            ],
            "offline_distribution_labels": [
                "mean",
                "std",
                "threshold mass",
                "quantiles",
                "CVaR low 10%",
                "branch peaks",
                "shelf gap",
                "scalar-EV omission flag",
            ],
            "offline_missing_for_now": [
                "bid amount",
                "bid margin",
                "joint-world hidden ownership",
                "top hidden threat holder/domino/impact",
            ],
        },
        "leakage_boundary": {
            "online_safe": "public trick state plus the actor's own private hand and legal candidate action facts",
            "offline_label_only": "E[Q] PDFs, scalar E[Q], threshold/tail labels, future outcome rank, hidden-domino threat attribution when generated",
            "not_present_in_v0": "bid amount/margin and saved joint-world hidden ownership",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    records: list[dict[str, Any]], action_rows: list[dict[str, Any]], decision_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    role_counts = Counter(row["seat_role"] for row in decision_rows)
    detector_counts = Counter(
        detector
        for row in action_rows
        for detector in str(row["matched_position_detectors"]).split("|")
        if detector
    )
    distribution_tag_counts = Counter(
        tag
        for row in action_rows
        for tag in str(row["distribution_shape_tags"]).split("|")
        if tag
    )
    actual_top_mean = sum(1 for row in decision_rows if int(row["actual_mean_rank_desc"]) == 1)
    actual_top_threshold = sum(1 for row in decision_rows if int(row["actual_threshold_rank_desc"]) == 1)
    actual_safest_tail = sum(1 for row in decision_rows if int(row["actual_lower_tail_rank_asc"]) == 1)
    scalar_lying = sum(1 for row in decision_rows if row["decision_scalar_ev_lying_by_omission"])
    multi_peak_decisions = sum(1 for row in decision_rows if int(row["decision_multi_peak_action_count"]) > 0)
    samples = sorted({int(row["samples"]) for row in action_rows})

    return {
        "bead": "t42-dqow",
        "schema_version": "w42.phase2_decision_table.summary.v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": current_git_sha(),
        "source_records": len(records),
        "decision_state_rows": len(decision_rows),
        "legal_action_rows": len(action_rows),
        "samples_per_pdf_values": samples,
        "role_counts": dict(sorted(role_counts.items())),
        "distribution_findings": {
            "decision_scalar_ev_lying_by_omission_count": scalar_lying,
            "decision_multi_peak_pdf_count": multi_peak_decisions,
            "action_distribution_tag_counts": dict(sorted(distribution_tag_counts.items())),
            "actual_action_top_mean_count": actual_top_mean,
            "actual_action_top_threshold_count": actual_top_threshold,
            "actual_action_safest_lower_tail_count": actual_safest_tail,
            "mean_legal_action_count": round(
                sum(int(row["legal_action_count"]) for row in decision_rows) / len(decision_rows), 4
            ),
            "max_legal_action_count": max(int(row["legal_action_count"]) for row in decision_rows),
            "max_std": round(max(float(row["decision_max_std"]) for row in decision_rows), 4),
        },
        "strategy_surface": {
            "detector_counts": dict(sorted(detector_counts.items())),
            "direct_label_readiness_values": sorted(
                {
                    readiness
                    for row in action_rows
                    for readiness in str(row["direct_label_readiness"]).split("|")
                    if readiness
                }
            ),
            "missing_fields_all_rows": [
                "bid_amount",
                "bid_margin",
                "hidden_world_ownership",
            ],
        },
        "scientific_status": {
            "training_run": False,
            "wandb": "not applicable",
            "hf": "not applicable",
            "claim_ledger_impact": "no claim-ledger movement",
            "interpretation": (
                "This is a v0 bridge table from an inspectable 5-game E[Q] PDF sample. "
                "It is ready for schema review and small-slice analysis, not final claim conclusions."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    action_rows = build_action_rows(records)
    decision_rows = build_decision_rows(action_rows)
    examples = compact_examples(decision_rows, action_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "decision_actions.csv", action_rows, ACTION_COLUMNS)
    write_csv(args.output_dir / "decision_states.csv", decision_rows, DECISION_COLUMNS)
    (args.output_dir / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    (args.output_dir / "schema.json").write_text(json.dumps(schema_doc(), indent=2) + "\n")
    summary = build_summary(records, action_rows, decision_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "bead": "t42-dqow",
        "artifact_id": "w42-phase2-decision-table-v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "w42/phase2_decision_table/build_phase2_decision_table.py",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "outputs": [
            "decision_actions.csv",
            "decision_states.csv",
            "examples.json",
            "schema.json",
            "summary.json",
            "manifest.json",
        ],
        "commands": [
            f"python w42/phase2_decision_table/build_phase2_decision_table.py --input {args.input.as_posix()}"
        ],
        "wiki": ["wiki/experiments/w42-phase2-decision-table.md"],
        "wandb": "not applicable; no training run",
        "hf": "not applicable; local table artifact only",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        json.dumps(
            {
                "decision_states": len(decision_rows),
                "decision_actions": len(action_rows),
                "examples": len(examples),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
