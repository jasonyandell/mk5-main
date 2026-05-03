#!/usr/bin/env python3
"""Joined W42 claim-row model table and family ablation probe.

This runner builds a public-safe legal-action row table from the current W42
claim artifacts. Full sequence/seat labels cover every row. Auction and 84
features join where generated seed/seat/declaration artifacts exist. Hidden
truth remains offline/eval-only; model features use only public/action-local
facts and derived public proxies.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from w42.wandb_utils import add_wandb_args, init_wandb

BEAD_ID = "t42-qtwb.4"
DEFAULT_INPUT = ROOT / "w42/sequence_seat_counterfactuals/labeled_sequence_action_rows.csv"
DEFAULT_OUTPUT = ROOT / "w42/joined_claim_row_model_table"
DEFAULT_AUCTION_CONTRACTS = ROOT / "w42/auction_bid_discipline_claim_tests/contract_rows.csv"
DEFAULT_84_CANDIDATES = ROOT / "w42/eighty_four_seed_mining/candidate_84_seed_rows.csv"

NUMERIC_FIELDS = (
    "actor",
    "trick_idx",
    "trick_position",
    "candidate_count_points",
    "bid_value",
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
    "candidate_is_called_suit",
    "candidate_is_double",
    "candidate_beats_current",
    "candidate_would_win_trick_now",
)

FEATURE_FAMILY_PREFIXES = {
    "sequence_seat": (
        "seq_",
        "seat_pos_",
        "sequence_",
        "role_",
        "phase_",
        "closure_",
        "follower_",
        "partner_",
        "setter_",
        "bidder_lead_",
        "ch04_",
        "ch05_",
    ),
    "bidding_risk": ("bidrisk_",),
    "eighty_four_public": ("e84_public_",),
    "doubles_no_trump": ("dnt_",),
    "hidden_public_proxy": ("hidden_proxy_",),
}

EVAL_ONLY_FAMILIES = {
    "eighty_four_defender_assets": "84 defender asset tags use full-deal opponent assets from seed mining and are report-only, not model features.",
    "hidden_truth_owner": "Hidden-owner impact/downside labels from legacy mining are offline report labels, not legal live features.",
    "auction_partner_opponent_values": "Auction partner/opponent pass values are offline generated labels, not live action-selection features.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--auction-contracts", type=Path, default=DEFAULT_AUCTION_CONTRACTS)
    parser.add_argument("--eighty-four-candidates", type=Path, default=DEFAULT_84_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-seed-mod", type=int, default=4)
    parser.add_argument("--seed-mod-base", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--c", type=float, default=0.8)
    parser.add_argument("--example-limit", type=int, default=30)
    add_wandb_args(parser, default_group="w42-joined-claim-row-model-table", default_enabled=True)
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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path, *, max_rows: int = 0) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return default
        return float(text)
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    value_float = as_float(value)
    if not math.isfinite(value_float):
        return default
    return int(value_float)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_labels(value: Any) -> tuple[str, ...]:
    labels = []
    for part in str(value or "").replace(",", "|").split("|"):
        text = part.strip()
        if text:
            labels.append(text)
    return tuple(sorted(set(labels)))


def bucket_float(value: Any, *, cuts: tuple[float, float], labels: tuple[str, str, str]) -> str:
    number = as_float(value)
    if not math.isfinite(number):
        return "unknown"
    if number <= cuts[0]:
        return labels[0]
    if number <= cuts[1]:
        return labels[1]
    return labels[2]


def domino_pips(domino: Any) -> tuple[int, int] | None:
    parts = str(domino or "").split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def add_label(labels: set[str], label: str) -> None:
    labels.add(label.replace(" ", "_").replace("/", "_").lower())


def auction_index(path: Path) -> dict[tuple[int, int, str], set[str]]:
    index: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    if not path.exists():
        return index
    for row in read_csv(path):
        seed = as_int(row.get("seed"), -1)
        seat = as_int(row.get("seat"), -1)
        decl = str(row.get("decl_name", ""))
        labels = index[(seed, seat, decl)]
        add_label(labels, "bidrisk_join_available")
        add_label(labels, f"bidrisk_static_{row.get('static_risk_bucket', 'unknown')}")
        add_label(labels, f"bidrisk_max_profitable_{row.get('max_profitable_bid_bucket', 'unknown')}")
        add_label(labels, f"bidrisk_wilson_safe_{row.get('max_wilson_safe_bid_bucket', 'unknown')}")
        add_label(labels, f"bidrisk_pmake30_{bucket_float(row.get('p_make_30'), cuts=(0.35, 0.6), labels=('low', 'mid', 'high'))}")
        add_label(labels, f"bidrisk_trump_count_{as_int(row.get('static_trump_count'))}")
        add_label(labels, f"bidrisk_off_count_{as_int(row.get('static_off_count'))}")
        if as_bool(row.get("natural_bucket_candidate")):
            add_label(labels, "bidrisk_natural_bucket_candidate")
        if as_bool(row.get("static_strong_trump_bad_risk_trap")):
            add_label(labels, "bidrisk_strong_trump_bad_risk_trap")
        if as_bool(row.get("static_minimum_shape_low_risk")):
            add_label(labels, "bidrisk_minimum_shape_low_risk")
        if as_bool(row.get("static_four_five_off")):
            add_label(labels, "bidrisk_four_five_off")
    return index


def eighty_four_index(path: Path) -> tuple[dict[tuple[int, int, str], set[str]], dict[tuple[int, int, str], set[str]]]:
    public_index: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    eval_only_index: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    if not path.exists():
        return public_index, eval_only_index
    public_surfaces = {
        "laydown_all_trumps",
        "protected_one_off",
        "straight_one_off",
        "three_trump_three_double_one_off",
        "two_off_same_suit",
    }
    eval_surfaces = {
        "defender_live_double_weapon",
        "defender_live_same_suit_pair",
        "pair_protector_pressure",
        "dead_asset_release_control",
    }
    for row in read_csv(path):
        key = (as_int(row.get("seed"), -1), as_int(row.get("bidder_seat"), -1), str(row.get("decl_name", "")))
        surfaces = split_labels(row.get("surfaces"))
        if surfaces:
            add_label(public_index[key], "e84_public_candidate")
        for surface in surfaces:
            if surface in public_surfaces:
                add_label(public_index[key], f"e84_public_{surface}")
            if surface in eval_surfaces:
                add_label(eval_only_index[key], f"eval_e84_{surface}")
        add_label(public_index[key], f"e84_public_trump_count_{as_int(row.get('trump_count'))}")
        add_label(public_index[key], f"e84_public_off_count_{as_int(row.get('off_count'))}")
        add_label(public_index[key], f"e84_public_double_count_{as_int(row.get('total_double_count'))}")
    return public_index, eval_only_index


def doubles_no_trump_labels(row: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    decl = str(row.get("decl_name", ""))
    pips = domino_pips(row.get("candidate_domino"))
    is_double = as_bool(row.get("candidate_is_double"))
    count_points = as_int(row.get("candidate_count_points"))
    trick_idx = as_int(row.get("trick_idx"))
    called = as_bool(row.get("candidate_is_called_suit"))

    if decl == "no-trump":
        add_label(labels, "dnt_regime_no_trump")
        if is_double:
            add_label(labels, "dnt_nt_candidate_double")
            if trick_idx <= 1:
                add_label(labels, "dnt_nt_early_support_double")
            if trick_idx >= 4:
                add_label(labels, "dnt_nt_late_double_weapon")
        if count_points > 0:
            add_label(labels, "dnt_nt_count_candidate")
        if as_bool(row.get("candidate_would_win_trick_now")) and count_points > 0:
            add_label(labels, "dnt_nt_count_capture")
    if decl in {"doubles", "doubles-suit"}:
        add_label(labels, f"dnt_regime_{decl}")
        if is_double:
            add_label(labels, "dnt_dt_candidate_double")
            if pips and pips[0] <= 2:
                add_label(labels, "dnt_dt_low_double")
            if pips and pips[0] >= 5:
                add_label(labels, "dnt_dt_high_double")
        if pips and set(pips) == {5, 6}:
            add_label(labels, "dnt_dt_dual_suit_65")
        if called and count_points > 0:
            add_label(labels, "dnt_dt_trump_count_candidate")
    return labels


def hidden_public_proxy_labels(row: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    trick_idx = as_int(row.get("trick_idx"))
    trick_position = as_int(row.get("trick_position"))
    count_points = as_int(row.get("candidate_count_points"))
    current_count = as_int(row.get("current_trick_count_before"))
    winner_team = str(row.get("current_winner_team_before", ""))
    team = str(row.get("team", ""))

    if trick_idx <= 1:
        add_label(labels, "hidden_proxy_early_high_uncertainty")
    if trick_idx >= 4:
        add_label(labels, "hidden_proxy_late_information_rich")
    if count_points > 0 or current_count > 0:
        add_label(labels, "hidden_proxy_count_pressure")
    if winner_team and winner_team != team:
        add_label(labels, "hidden_proxy_opponent_controls_trick")
    if winner_team and winner_team == team:
        add_label(labels, "hidden_proxy_our_side_controls_trick")
    if trick_position == 3:
        add_label(labels, "hidden_proxy_closure_information")
    if as_bool(row.get("candidate_beats_current")):
        add_label(labels, "hidden_proxy_control_shift_available")
    return labels


def family_for_label(label: str) -> str | None:
    for family, prefixes in FEATURE_FAMILY_PREFIXES.items():
        if label.startswith(prefixes):
            return family
    return None


def load_joined_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_csv(args.input, max_rows=args.max_rows)
    auction = auction_index(args.auction_contracts)
    e84_public, e84_eval = eighty_four_index(args.eighty_four_candidates)

    joined: list[dict[str, Any]] = []
    coverage = Counter()
    eval_only_counts = Counter()
    for row in rows:
        seed = as_int(row.get("seed"), -1)
        actor = as_int(row.get("actor"), -1)
        decl = str(row.get("decl_name", ""))
        join_key = (seed, actor, decl)
        labels = set(split_labels(row.get("derived_labels")))
        labels.update(doubles_no_trump_labels(row))
        labels.update(hidden_public_proxy_labels(row))
        if join_key in auction:
            labels.update(auction[join_key])
            coverage["auction_join_rows"] += 1
        if join_key in e84_public:
            labels.update(e84_public[join_key])
            coverage["eighty_four_public_join_rows"] += 1
        eval_labels = set(e84_eval.get(join_key, set()))
        for label in eval_labels:
            eval_only_counts[label] += 1
        out = dict(row)
        out["_seed"] = seed
        out["_labels"] = tuple(sorted(labels))
        out["_eval_only_labels"] = tuple(sorted(eval_labels))
        out["_is_best_mean"] = as_bool(row.get("is_best_mean"))
        out["_is_actual_action"] = as_bool(row.get("is_actual_action"))
        out["_mean_regret"] = as_float(row.get("mean_regret"))
        out["_mean"] = as_float(row.get("mean"))
        joined.append(out)

    label_counts = Counter(label for row in joined for label in row["_labels"])
    family_counts = Counter()
    for label, count in label_counts.items():
        family = family_for_label(label)
        if family:
            family_counts[family] += count
    coverage.update(
        {
            "action_rows": len(joined),
            "decision_rows": len({str(row.get("key")) for row in joined}),
            "feature_label_count": len(label_counts),
        }
    )
    return joined, {
        "coverage": dict(coverage),
        "label_counts": label_counts,
        "family_counts": family_counts,
        "eval_only_counts": eval_only_counts,
    }


def groups_by_decision(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("key"))].append(row)
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
            family = family_for_label(label)
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
    scored = [dict(row, **{f"_score_{variant}": float(prob)}) for row, prob in zip(rows, probs, strict=True)]

    chosen: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for key, bucket in groups_by_decision(scored).items():
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
                    "trick_idx": best.get("trick_idx"),
                    "trick_position": best.get("trick_position"),
                }
            )
    regrets = [row["_mean_regret"] for row in chosen]
    return (
        {
            "variant": variant,
            "status": "available",
            "decision_n": len(chosen),
            "action_n": len(rows),
            "match_best_mean_rate": round(sum(1 for row in chosen if row["_is_best_mean"]) / len(chosen), 6),
            "mean_regret": round(float(np.mean(regrets)), 6),
            "median_regret": round(float(np.median(regrets)), 6),
            "near_tie_rate_regret_lt_0_5": round(sum(1 for value in regrets if value < 0.5) / len(regrets), 6),
            "tail_regret_rate_ge_5": round(sum(1 for value in regrets if value >= 5.0) / len(regrets), 6),
        },
        examples,
    )


def actual_policy_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = []
    for bucket in groups_by_decision(rows).values():
        actual = [row for row in bucket if row["_is_actual_action"]]
        if actual:
            chosen.append(actual[0])
    regrets = [row["_mean_regret"] for row in chosen]
    return {
        "variant": "actual_policy",
        "status": "offline_reference",
        "decision_n": len(chosen),
        "action_n": len(rows),
        "match_best_mean_rate": round(sum(1 for row in chosen if row["_is_best_mean"]) / len(chosen), 6),
        "mean_regret": round(float(np.mean(regrets)), 6),
        "median_regret": round(float(np.median(regrets)), 6),
        "near_tie_rate_regret_lt_0_5": round(sum(1 for value in regrets if value < 0.5) / len(regrets), 6),
        "tail_regret_rate_ge_5": round(sum(1 for value in regrets if value >= 5.0) / len(regrets), 6),
    }


def split_rows(rows: list[dict[str, Any]], *, eval_seed_mod: int, seed_mod_base: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [row for row in rows if row["_seed"] % seed_mod_base != eval_seed_mod]
    eval_rows = [row for row in rows if row["_seed"] % seed_mod_base == eval_seed_mod]
    return train_rows, eval_rows


def inventory_rows(join_meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = join_meta["label_counts"]
    family_counts: Counter[str] = join_meta["family_counts"]
    for family in sorted(FEATURE_FAMILY_PREFIXES):
        rows.append(
            {
                "family": family,
                "feature_action_labels": sum(1 for label in label_counts if family_for_label(label) == family),
                "tag_action_n": family_counts.get(family, 0),
                "status": "available",
                "feature_prefixes": "|".join(FEATURE_FAMILY_PREFIXES[family]),
                "boundary": "public/action-local model feature",
            }
        )
    eval_only_counts: Counter[str] = join_meta["eval_only_counts"]
    rows.extend(
        {
            "family": family,
            "feature_action_labels": 0,
            "tag_action_n": sum(eval_only_counts.values()) if family == "eighty_four_defender_assets" else 0,
            "status": "eval_only",
            "feature_prefixes": "",
            "boundary": boundary,
        }
        for family, boundary in sorted(EVAL_ONLY_FAMILIES.items())
    )
    return rows


def compact_join_row(row: dict[str, Any]) -> dict[str, Any]:
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
        "role_family": row.get("role_family"),
        "team": row.get("team"),
        "trick_idx": row.get("trick_idx"),
        "trick_position": row.get("trick_position"),
        "candidate_domino": row.get("candidate_domino"),
        "candidate_count_points": row.get("candidate_count_points"),
        "candidate_is_called_suit": row.get("candidate_is_called_suit"),
        "candidate_is_double": row.get("candidate_is_double"),
        "current_winner_team_before": row.get("current_winner_team_before"),
        "mean": row.get("mean"),
        "mean_regret": row.get("mean_regret"),
        "threshold_mass": row.get("threshold_mass"),
        "lower_tail_mass": row.get("lower_tail_mass"),
        "is_actual_action": row.get("is_actual_action"),
        "is_best_mean": row.get("is_best_mean"),
        "feature_labels": "|".join(row["_labels"]),
        "eval_only_labels": "|".join(row["_eval_only_labels"]),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    config = {
        "bead_id": BEAD_ID,
        "experiment": "w42-joined-claim-row-model-table",
        "input": str(args.input),
        "auction_contracts": str(args.auction_contracts),
        "eighty_four_candidates": str(args.eighty_four_candidates),
        "eval_seed_mod": args.eval_seed_mod,
        "seed_mod_base": args.seed_mod_base,
        "max_rows": args.max_rows,
        "max_iter": args.max_iter,
        "c": args.c,
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=args.output_dir,
        tags=["w42", "claim-analysis", "model", "joined-table", BEAD_ID],
    )

    rows, join_meta = load_joined_rows(args)
    train_rows, eval_rows = split_rows(rows, eval_seed_mod=args.eval_seed_mod, seed_mod_base=args.seed_mod_base)
    if not train_rows or not eval_rows:
        raise SystemExit("empty train/eval split")

    variants: list[dict[str, Any]] = [
        {"variant": "public_features_only", "include_tags": False, "drop_family": None},
        {"variant": "public_plus_all_claim_families", "include_tags": True, "drop_family": None},
    ]
    variants.extend(
        {"variant": f"drop_{family}", "include_tags": True, "drop_family": family}
        for family in sorted(FEATURE_FAMILY_PREFIXES)
    )

    metrics = [actual_policy_metrics(eval_rows)]
    examples: list[dict[str, Any]] = []
    for i, spec in enumerate(variants):
        model = train_model(
            train_rows,
            include_tags=spec["include_tags"],
            drop_family=spec["drop_family"],
            args=args,
        )
        metric, variant_examples = evaluate_model(
            model,
            eval_rows,
            include_tags=spec["include_tags"],
            drop_family=spec["drop_family"],
            variant=spec["variant"],
            example_limit=args.example_limit,
        )
        metrics.append(metric)
        examples.extend(variant_examples)
        wb.log_series_point(
            axis="variant/index",
            value=i,
            metrics={
                "variant/mean_regret": metric["mean_regret"],
                "variant/match_best_mean_rate": metric["match_best_mean_rate"],
                "variant/tail_regret_rate_ge_5": metric["tail_regret_rate_ge_5"],
            },
        )

    public = next(row for row in metrics if row["variant"] == "public_features_only")
    full = next(row for row in metrics if row["variant"] == "public_plus_all_claim_families")
    for row in metrics:
        if row["variant"] == "actual_policy":
            row["delta_vs_public_mean_regret"] = ""
            row["delta_vs_full_mean_regret"] = ""
            continue
        row["delta_vs_public_mean_regret"] = round(row["mean_regret"] - public["mean_regret"], 6)
        row["delta_vs_full_mean_regret"] = round(row["mean_regret"] - full["mean_regret"], 6)

    inventory = inventory_rows(join_meta)
    summary = {
        "schema_version": "w42.joined_claim_row_model_table.v1",
        "bead": BEAD_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "git_sha": git_sha(),
        "git_status_before_artifacts": git_status_short(),
        "split": {
            "seed_mod_base": args.seed_mod_base,
            "eval_seed_mod": args.eval_seed_mod,
            "train_action_rows": len(train_rows),
            "train_decisions": len(groups_by_decision(train_rows)),
            "train_seeds": sorted({row["_seed"] for row in train_rows})[:8],
            "train_seeds_tail": sorted({row["_seed"] for row in train_rows})[-8:],
            "eval_action_rows": len(eval_rows),
            "eval_decisions": len(groups_by_decision(eval_rows)),
            "eval_seeds": sorted({row["_seed"] for row in eval_rows}),
        },
        "coverage": join_meta["coverage"],
        "headline": {
            "public_mean_regret": public["mean_regret"],
            "all_claim_families_mean_regret": full["mean_regret"],
            "all_claim_families_delta_vs_public": round(full["mean_regret"] - public["mean_regret"], 6),
            "public_match_rate": public["match_best_mean_rate"],
            "all_claim_families_match_rate": full["match_best_mean_rate"],
            "best_available_variant": min(
                [row for row in metrics if row["variant"] != "actual_policy"],
                key=lambda row: row["mean_regret"],
            )["variant"],
        },
        "scientific_status": {
            "claim_ledger_impact": "no central claim status movement",
            "interpretation": "Joined public/action-local claim-family tags can be ablated on one legal-action split. This is model-feature evidence and routing evidence, not proof of broad book claims.",
            "leakage_boundary": "Features exclude oracle mean/regret, threshold mass, lower-tail mass, future outcomes, hidden owners, q_per_world, and full-deal opponent asset labels. Eval-only labels are written for auditing but never used as model features.",
        },
        "wandb": wb.status(),
    }
    manifest = {
        "schema_version": "w42.joined_claim_row_model_table.manifest.v1",
        "summary": summary,
        "feature_families": FEATURE_FAMILY_PREFIXES,
        "eval_only_families": EVAL_ONLY_FAMILIES,
        "numeric_fields": NUMERIC_FIELDS,
        "categorical_fields": CATEGORICAL_FIELDS,
        "offline_label_columns_excluded_from_features": [
            "mean",
            "mean_regret",
            "threshold_mass",
            "lower_tail_mass",
            "is_best_mean",
            "hidden_owner",
            "world_hands",
            "q_per_world",
        ],
    }

    compact_rows = [compact_join_row(row) for row in rows]
    write_csv(args.output_dir / "joined_claim_action_rows.csv", compact_rows)
    write_csv(args.output_dir / "family_inventory.csv", inventory)
    write_csv(args.output_dir / "model_metrics.csv", metrics)
    write_csv(args.output_dir / "ablation_results.csv", metrics)
    write_jsonl(args.output_dir / "prediction_sample.jsonl", examples[: args.example_limit])
    write_json(args.output_dir / "error_examples.json", examples[: args.example_limit])
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "feature_manifest.json", manifest)
    write_json(args.output_dir / "metrics.json", {"split": summary["split"], "metrics": metrics})

    wb.update_summary(
        {
            "action_rows": len(rows),
            "eval_decisions": summary["split"]["eval_decisions"],
            "public_mean_regret": public["mean_regret"],
            "all_claim_families_mean_regret": full["mean_regret"],
            "all_claim_families_delta_vs_public": summary["headline"]["all_claim_families_delta_vs_public"],
        }
    )
    wb.log_artifact_files(
        name="w42-joined-claim-row-model-table",
        artifact_type="w42-claim-analysis",
        paths=[
            args.output_dir / "summary.json",
            args.output_dir / "model_metrics.csv",
            args.output_dir / "family_inventory.csv",
            args.output_dir / "prediction_sample.jsonl",
        ],
    )
    wb.finish()
    print(json.dumps(summary["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
