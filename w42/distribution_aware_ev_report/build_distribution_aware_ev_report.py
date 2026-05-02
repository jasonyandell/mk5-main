#!/usr/bin/env python3
"""Build distribution-aware E[Q] report artifacts from visualizer JSONL data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EQ_MIN = -42
EQ_MAX = 42
OFFENSE_PLAYERS = {0, 2}
OFFENSE_MAKE_THRESHOLD = 18
DEFENSE_SET_THRESHOLD_EXCLUSIVE = -18
LOWER_TAIL_THRESHOLD = -18
NEAR_MEAN_EPSILON = 1.0
TAIL_GAP_EPSILON = 0.05
THRESHOLD_GAP_EPSILON = 0.03


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        value = EQ_MIN + idx
        weighted += take * value
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


def player_domino_lookup(record: dict[str, Any]) -> tuple[dict[int, int], dict[int, str]]:
    player_by_d: dict[int, int] = {}
    pips_by_d: dict[int, str] = {}
    for player in record["players"]:
        player_id = int(player["id"])
        for domino in player["dominoes"]:
            global_idx = player_id * 7 + int(domino["slot"])
            player_by_d[global_idx] = player_id
            pips_by_d[global_idx] = domino["pips"]
    return player_by_d, pips_by_d


def extract_action_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        player_by_d, pips_by_d = player_domino_lookup(record)
        active_player_by_move = {
            move_idx: int(player)
            for move_idx, player in enumerate(record.get("active_player", []))
        }
        for entry in record.get("pdf_data", []):
            pdf = [float(p) for p in entry["pdf"]]
            d_idx = int(entry["d"])
            player = player_by_d[d_idx]
            active_player = active_player_by_move.get(int(entry["m"]), player)
            make_mass = pdf_mass(pdf, lo=OFFENSE_MAKE_THRESHOLD)
            set_mass = pdf_mass(pdf, lo=DEFENSE_SET_THRESHOLD_EXCLUSIVE + 1)
            lower_tail_mass = pdf_mass(pdf, hi=LOWER_TAIL_THRESHOLD)
            q10 = quantile(pdf, 0.10)
            q25 = quantile(pdf, 0.25)
            q50 = quantile(pdf, 0.50)
            q75 = quantile(pdf, 0.75)
            q90 = quantile(pdf, 0.90)
            peaks = local_peak_bins(pdf)
            row = {
                "game_id": record["game_id"],
                "game_idx": record.get("game_idx"),
                "trump_id": record.get("trump_id"),
                "trump_name": record.get("trump_name"),
                "move_idx": int(entry["m"]),
                "player": player,
                "active_player": active_player,
                "team": "offense" if player in OFFENSE_PLAYERS else "defense",
                "global_domino_idx": d_idx,
                "domino": pips_by_d[d_idx],
                "mean": float(entry["mean"]),
                "std": float(entry.get("std") or 0.0),
                "threshold_mass": float(entry.get("win") or 0.0),
                "make_mass": make_mass if player in OFFENSE_PLAYERS else "",
                "set_mass": set_mass if player not in OFFENSE_PLAYERS else "",
                "lower_tail_mass": lower_tail_mass,
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
            rows.append(row)
    return rows


def annotate_decision_flags(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_decision: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_decision[(row["game_id"], row["move_idx"])].append(row)

    examples: list[dict[str, Any]] = []
    for key, decision_rows in by_decision.items():
        if len(decision_rows) < 2:
            for row in decision_rows:
                row["high_variance_close_mean"] = False
                row["scalar_ev_lying_by_omission"] = False
            continue

        top_mean = max(decision_rows, key=lambda r: r["mean"])
        top_threshold = max(decision_rows, key=lambda r: r["threshold_mass"])
        safest_tail = min(decision_rows, key=lambda r: r["lower_tail_mass"])
        std_values = sorted(float(r["std"]) for r in decision_rows)
        high_std_cutoff = std_values[max(0, int(0.75 * (len(std_values) - 1)))]
        mean_max = float(top_mean["mean"])

        for row in decision_rows:
            close_to_best = mean_max - float(row["mean"]) <= NEAR_MEAN_EPSILON
            row["high_variance_close_mean"] = close_to_best and float(row["std"]) >= high_std_cutoff
            row["scalar_ev_lying_by_omission"] = (
                row is top_mean
                and (
                    top_threshold is not top_mean
                    or safest_tail is not top_mean
                    or float(top_mean["std"]) >= 15.0
                    or float(top_mean["lower_tail_mass"]) - float(safest_tail["lower_tail_mass"])
                    >= TAIL_GAP_EPSILON
                    or float(top_threshold["threshold_mass"]) - float(top_mean["threshold_mass"])
                    >= THRESHOLD_GAP_EPSILON
                )
            )

        if any(r["scalar_ev_lying_by_omission"] for r in decision_rows):
            sorted_rows = sorted(decision_rows, key=lambda r: r["mean"], reverse=True)
            examples.append(
                {
                    "game_id": key[0],
                    "move_idx": key[1],
                    "active_player": sorted_rows[0]["active_player"],
                    "trump_name": sorted_rows[0]["trump_name"],
                    "legal_action_count": len(decision_rows),
                    "why": "top scalar mean disagrees with threshold mass, lower-tail safety, or high-variance branch evidence",
                    "top_mean_action": compact_action(top_mean),
                    "top_threshold_action": compact_action(top_threshold),
                    "lowest_tail_action": compact_action(safest_tail),
                    "actions_by_mean": [compact_action(r) for r in sorted_rows],
                }
            )

    examples.sort(
        key=lambda ex: (
            abs(ex["top_threshold_action"]["threshold_mass"] - ex["top_mean_action"]["threshold_mass"])
            + abs(ex["top_mean_action"]["lower_tail_mass"] - ex["lowest_tail_action"]["lower_tail_mass"]),
            ex["legal_action_count"],
        ),
        reverse=True,
    )
    return rows, examples


def compact_action(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "domino": row["domino"],
        "player": row["player"],
        "team": row["team"],
        "mean": round(float(row["mean"]), 4),
        "std": round(float(row["std"]), 4),
        "threshold_mass": round(float(row["threshold_mass"]), 4),
        "make_mass": round(float(row["make_mass"]), 4) if row["make_mass"] != "" else None,
        "set_mass": round(float(row["set_mass"]), 4) if row["set_mass"] != "" else None,
        "lower_tail_mass": round(float(row["lower_tail_mass"]), 4),
        "q10": row["q10"],
        "q50": row["q50"],
        "q90": row["q90"],
        "cvar_low_10": round(float(row["cvar_low_10"]), 4)
        if row["cvar_low_10"] is not None
        else None,
        "branch_entropy": round(float(row["branch_entropy"]), 4),
        "branch_peak_count": row["branch_peak_count"],
        "shelf_gap": row["shelf_gap"] if row["shelf_gap"] != "" else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def feature_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "E[Q] PDF visualizer records; one row per legal action with an 85-bin PDF over E[Q] values -42..42.",
        "features": [
            {
                "name": "mean",
                "type": "float",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Scalar E[Q]; useful baseline but intentionally not sufficient.",
            },
            {
                "name": "std",
                "type": "float",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Distribution spread from sampled hidden worlds.",
            },
            {
                "name": "make_mass",
                "type": "float|null",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Offense-side mass at E[Q] >= 18; null for defense rows.",
            },
            {
                "name": "set_mass",
                "type": "float|null",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Defense-side threshold mass using the visualizer's E[Q] > -18 rule; null for offense rows.",
            },
            {
                "name": "lower_tail_mass",
                "type": "float",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Mass at E[Q] <= -18; a compact tail-risk indicator.",
            },
            {
                "name": "q10/q25/q50/q75/q90",
                "type": "integer",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Empirical quantiles from the PDF bins.",
            },
            {
                "name": "cvar_low_10",
                "type": "float",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Approximate expectation over the lowest 10% of mass.",
            },
            {
                "name": "branch_entropy",
                "type": "float",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Entropy of the E[Q] PDF; crude branchiness indicator.",
            },
            {
                "name": "branch_peak_count",
                "type": "integer",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Number of local PDF peaks with at least 2% mass.",
            },
            {
                "name": "shelf_gap",
                "type": "integer|null",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Distance between lowest and highest local peak bins when at least two shelves are visible.",
            },
            {
                "name": "high_variance_close_mean",
                "type": "boolean",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Action is within one E[Q] point of the best mean and in the decision's upper spread quartile.",
            },
            {
                "name": "scalar_ev_lying_by_omission",
                "type": "boolean",
                "cheap_for_training_labels": True,
                "cheap_for_live_inference": False,
                "notes": "Top-mean action hides threshold, lower-tail, or high-variance evidence worth reporting.",
            },
        ],
        "live_inference_note": (
            "These features require sampled-world E[Q] PDFs. They are cheap once the "
            "offline visualizer tensor exists, but too expensive for a live selector unless "
            "a smaller model learns to predict them from legal public state."
        ),
    }


def build_summary(rows: list[dict[str, Any]], examples: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = {(row["game_id"], row["move_idx"]) for row in rows}
    multi_action_decisions = defaultdict(int)
    for row in rows:
        multi_action_decisions[(row["game_id"], row["move_idx"])] += 1
    lying_rows = [row for row in rows if row["scalar_ev_lying_by_omission"]]
    high_var_rows = [row for row in rows if row["high_variance_close_mean"]]
    return {
        "bead": "t42-5m82.4",
        "report": "w42 phase 2: distribution-aware EV report",
        "data_slice": {
            "games": len(records),
            "legal_action_pdf_rows": len(rows),
            "decisions_with_pdf": len(decisions),
            "decisions_with_multiple_legal_pdf_actions": sum(
                1 for count in multi_action_decisions.values() if count > 1
            ),
            "samples_per_pdf_values": sorted({row["samples"] for row in rows}),
        },
        "thresholds": {
            "pdf_bins": "85 integer bins from -42 to +42",
            "offense_make_mass": "E[Q] >= 18",
            "defense_set_mass": "E[Q] > -18, matching eq_pdf_discs/exporter threshold",
            "lower_tail_mass": "E[Q] <= -18",
            "cvar_low_10": "lowest 10% of PDF mass",
        },
        "aggregate_findings": {
            "scalar_ev_lying_by_omission_count": len(lying_rows),
            "high_variance_close_mean_count": len(high_var_rows),
            "mean_std": round(sum(float(r["std"]) for r in rows) / len(rows), 4) if rows else None,
            "max_std": round(max(float(r["std"]) for r in rows), 4) if rows else None,
            "mean_lower_tail_mass": round(
                sum(float(r["lower_tail_mass"]) for r in rows) / len(rows), 4
            )
            if rows
            else None,
            "max_lower_tail_mass": round(max(float(r["lower_tail_mass"]) for r in rows), 4)
            if rows
            else None,
        },
        "walkthrough_example": examples[0] if examples else None,
        "scientific_status": {
            "production_selector": False,
            "training_run": False,
            "wandb": "not applicable",
            "claim_ledger_impact": "no claim-ledger change",
            "note": "Feature/report design from a small N=1000 visualizer sample, not a final empirical selector result.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="eq_pdf_v3_sample.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for report artifacts.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)
    rows = extract_action_rows(records)
    rows, examples = annotate_decision_flags(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "action_distribution_features.csv", rows)
    (args.output_dir / "example_decisions.json").write_text(json.dumps(examples[:12], indent=2) + "\n")
    (args.output_dir / "candidate_feature_schema.json").write_text(
        json.dumps(feature_schema(), indent=2) + "\n"
    )
    summary = build_summary(rows, examples, records)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    manifest = {
        "bead": "t42-5m82.4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "w42/distribution_aware_ev_report/build_distribution_aware_ev_report.py",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "outputs": [
            "action_distribution_features.csv",
            "example_decisions.json",
            "candidate_feature_schema.json",
            "summary.json",
            "manifest.json",
        ],
        "commands": [
            f"python {Path(__file__).as_posix()} --input {args.input.as_posix()}",
        ],
        "wandb": "not applicable; no training run",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({"rows": len(rows), "examples": len(examples), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
