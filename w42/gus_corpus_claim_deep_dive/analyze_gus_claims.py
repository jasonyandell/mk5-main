#!/usr/bin/env python3
"""Targeted Winning 42 claim probes over existing Gus joint-world corpora.

This analyzer consumes saved Gus/forge GameRecordGPU payloads that retain
``q_per_world`` and ``world_hands``. It intentionally stays report-only: hidden
sampled worlds are labels/diagnostics, never live strategy features.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle import tables
from w42.branch_atlas_v1 import build_branch_atlas as atlas
from w42.wandb_utils import add_wandb_args, init_wandb


OUT_DIR = Path(__file__).resolve().parent
PIP_DECLS = set(range(7))


@dataclass(frozen=True)
class ActionRow:
    key: str
    source_file: str
    game_idx: int
    seed: int | None
    decl_id: int
    decl_name: str
    bid_value: int | None
    decision_idx: int
    trick_idx: int
    trick_position: int
    actor: int
    seat_role: str
    team: str
    offense_score_before: int
    defense_score_before: int
    current_winner_before: int | None
    current_winner_team_before: str
    current_trick_count_before: int
    candidate_slot: int
    candidate_domino_id: int
    candidate_domino: str
    candidate_count_points: int
    candidate_is_called_suit: bool
    candidate_is_double: bool
    candidate_beats_current: bool
    candidate_would_win_trick_now: bool
    mean: float
    std: float
    threshold_mass: float
    lower_tail_mass: float
    q10: int
    cvar_low_10: float | None
    mean_regret: float
    threshold_gap: float
    lower_tail_gap: float
    is_actual_action: bool
    is_best_mean: bool
    is_best_threshold: bool
    is_safest_tail: bool
    labels: tuple[str, ...]


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(p.resolve() for p in paths))


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> tuple[float, float, float]:
    center = mean(values)
    if not values:
        return center, float("nan"), float("nan")
    if len(values) == 1 or samples <= 0:
        return center, center, center
    generator = torch.Generator().manual_seed(seed)
    source = torch.tensor(values, dtype=torch.float64)
    estimates: list[float] = []
    n = int(source.numel())
    for _ in range(samples):
        idx = torch.randint(0, n, (n,), generator=generator)
        estimates.append(float(source[idx].mean().item()))
    estimates.sort()
    return center, percentile(estimates, 0.025), percentile(estimates, 0.975)


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(base_seed + int(digest[:8], 16) % 10_000)


def as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def q_summary(values: torch.Tensor, threshold_q: float) -> dict[str, Any]:
    values = values.detach().cpu().float()
    stats = atlas.q_stats(values, threshold_q)
    return {
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "threshold_mass": float(stats["threshold_mass"]),
        "lower_tail_mass": float(stats["lower_tail_mass_le_neg18"]),
        "q10": int(stats["q10"]),
        "cvar_low_10": None if stats["cvar_low_10"] is None else float(stats["cvar_low_10"]),
    }


def projected_defense_score(row: dict[str, Any]) -> int:
    if row["team"] != "defense":
        return int(row["defense_score_before"])
    if not row["candidate_would_win_trick_now"]:
        return int(row["defense_score_before"])
    return (
        int(row["defense_score_before"])
        + int(row["current_trick_count_before"])
        + int(row["candidate_count_points"])
        + 1
    )


def set_threshold_from_bid(bid_value: int | None) -> int:
    contract = atlas.contract_points_from_bid_value(bid_value)
    return 43 - contract


def action_labels(row: dict[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    if int(row["decl_id"]) not in PIP_DECLS:
        return ()
    count_points = int(row["candidate_count_points"])
    if count_points <= 0:
        return ()

    team = str(row["team"])
    role = str(row["seat_role"])
    trick_position = int(row["trick_position"])
    current_winner_team = str(row["current_winner_team_before"])
    beats_current = bool(row["candidate_beats_current"])
    would_win = bool(row["candidate_would_win_trick_now"])

    if team == "defense" and current_winner_team == "offense":
        if beats_current and trick_position in {1, 2}:
            labels.append("ch05_setter_pounce_count_before_certainty")
        if beats_current and trick_position in {1, 2, 3}:
            labels.append("ch05_setter_pounce_count")
            if projected_defense_score(row) >= set_threshold_from_bid(row["bid_value"]):
                labels.append("ch05_setter_pounce_count_sets_now")
        if not would_win:
            labels.append("ch05_reckless_count_to_bidder")

    if role == "bidder_partner":
        if current_winner_team == "offense":
            labels.append("ch04_partner_safe_count_donation_current_control")
        elif current_winner_team == "defense" and not beats_current:
            labels.append("ch04_partner_unsafe_count_to_defense")

    return tuple(sorted(set(labels)))


def process_game(
    *,
    source_file: str,
    game_idx: int,
    game: Any,
    seed: int | None,
    decl_id: int,
) -> list[ActionRow]:
    hands = [[int(d) for d in hand] for hand in getattr(game, "hands")]
    bid_value = getattr(game, "bid_value", None)
    bid_value = None if bid_value is None else int(bid_value)
    score = [0, 0]
    played_count_points = 0
    trick_plays: list[tuple[int, int]] = []
    played_slots: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
    rows: list[ActionRow] = []

    for decision_idx, decision in enumerate(getattr(game, "decisions", [])):
        actor = int(getattr(decision, "player"))
        legal_mask = torch.as_tensor(getattr(decision, "legal_mask"), dtype=torch.bool).detach().cpu()
        legal_slots = [slot for slot, is_legal in enumerate(legal_mask.tolist()) if is_legal]
        q_per_world = getattr(decision, "q_per_world", None)
        if q_per_world is None or not legal_slots:
            actual_slot = int(getattr(decision, "action_taken"))
            actual_domino = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1
            if actual_domino >= 0:
                played_slots[actor].add(actual_slot)
                played_count_points += int(tables.DOMINO_COUNT_POINTS[actual_domino])
                trick_plays.append((actor, actual_domino))
                if len(trick_plays) == 4:
                    winner, _led_suit, _best_rank = atlas.current_winner(trick_plays, decl_id)
                    if winner is not None:
                        trick_points = 1 + sum(tables.DOMINO_COUNT_POINTS[d] for _p, d in trick_plays)
                        score[atlas.team_id_for_player(winner)] += int(trick_points)
                    trick_plays = []
            continue

        q_per_world = q_per_world.detach().cpu().float()
        context = atlas.move_context(
            decision_idx=decision_idx,
            actor=actor,
            decl_id=decl_id,
            score=score,
            played_count_points=played_count_points,
            trick_plays=trick_plays,
        )
        threshold_q = atlas.threshold_q_for_player(actor, bid_value)
        per_slot: dict[int, dict[str, Any]] = {}
        for slot in legal_slots:
            candidate_domino = hands[actor][slot]
            facts = atlas.candidate_public_facts(candidate_domino, context, decl_id)
            stats = q_summary(q_per_world[:, slot], threshold_q)
            per_slot[slot] = {
                **context,
                **facts,
                **stats,
                "candidate_slot": slot,
                "candidate_domino_id": candidate_domino,
                "candidate_domino": atlas.domino_label(candidate_domino),
                "decl_id": decl_id,
                "decl_name": atlas.DECL_NAMES.get(decl_id, f"unknown-{decl_id}"),
                "bid_value": bid_value,
            }

        best_mean = max(float(row["mean"]) for row in per_slot.values())
        best_threshold = max(float(row["threshold_mass"]) for row in per_slot.values())
        safest_tail = min(float(row["lower_tail_mass"]) for row in per_slot.values())
        actual_slot = int(getattr(decision, "action_taken"))
        key = f"{source_file}:{game_idx}:{decision_idx}"
        for slot, base in per_slot.items():
            base_labels = action_labels(base)
            rows.append(
                ActionRow(
                    key=key,
                    source_file=source_file,
                    game_idx=game_idx,
                    seed=seed,
                    decl_id=decl_id,
                    decl_name=str(base["decl_name"]),
                    bid_value=bid_value,
                    decision_idx=decision_idx,
                    trick_idx=int(base["trick_idx"]),
                    trick_position=int(base["trick_position"]),
                    actor=actor,
                    seat_role=str(base["seat_role"]),
                    team=str(base["team"]),
                    offense_score_before=int(base["offense_score_before"]),
                    defense_score_before=int(base["defense_score_before"]),
                    current_winner_before=as_optional_int(base["current_winner_before"]),
                    current_winner_team_before=str(base["current_winner_team_before"]),
                    current_trick_count_before=int(base["current_trick_count_before"]),
                    candidate_slot=slot,
                    candidate_domino_id=int(base["candidate_domino_id"]),
                    candidate_domino=str(base["candidate_domino"]),
                    candidate_count_points=int(base["candidate_count_points"]),
                    candidate_is_called_suit=bool(base["candidate_is_called_suit"]),
                    candidate_is_double=bool(base["candidate_is_double"]),
                    candidate_beats_current=bool(base["candidate_beats_current"]),
                    candidate_would_win_trick_now=bool(base["candidate_would_win_trick_now"]),
                    mean=float(base["mean"]),
                    std=float(base["std"]),
                    threshold_mass=float(base["threshold_mass"]),
                    lower_tail_mass=float(base["lower_tail_mass"]),
                    q10=int(base["q10"]),
                    cvar_low_10=base["cvar_low_10"],
                    mean_regret=float(best_mean - float(base["mean"])),
                    threshold_gap=float(best_threshold - float(base["threshold_mass"])),
                    lower_tail_gap=float(float(base["lower_tail_mass"]) - safest_tail),
                    is_actual_action=slot == actual_slot,
                    is_best_mean=abs(float(base["mean"]) - best_mean) <= 1e-9,
                    is_best_threshold=abs(float(base["threshold_mass"]) - best_threshold) <= 1e-9,
                    is_safest_tail=abs(float(base["lower_tail_mass"]) - safest_tail) <= 1e-9,
                    labels=base_labels,
                )
            )

        actual_domino = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1
        if actual_domino >= 0:
            played_slots[actor].add(actual_slot)
            played_count_points += int(tables.DOMINO_COUNT_POINTS[actual_domino])
            trick_plays.append((actor, actual_domino))
            if len(trick_plays) == 4:
                winner, _led_suit, _best_rank = atlas.current_winner(trick_plays, decl_id)
                if winner is not None:
                    trick_points = 1 + sum(tables.DOMINO_COUNT_POINTS[d] for _p, d in trick_plays)
                    score[atlas.team_id_for_player(winner)] += int(trick_points)
                trick_plays = []

    return rows


def load_payload_rows(path: Path) -> tuple[list[ActionRow], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    games = payload.get("results", [])
    seeds = payload.get("seeds", [])
    decl_ids = payload.get("decl_ids", [])
    rows: list[ActionRow] = []
    for game_idx, game in enumerate(games):
        seed = int(seeds[game_idx]) if game_idx < len(seeds) else None
        decl_id = int(getattr(game, "decl_id", decl_ids[game_idx] if game_idx < len(decl_ids) else -1))
        rows.extend(
            process_game(
                source_file=path.name,
                game_idx=game_idx,
                game=game,
                seed=seed,
                decl_id=decl_id,
            )
        )
    meta = {
        "path": str(path),
        "games": len(games),
        "action_rows": len(rows),
        "decision_rows": len({row.key for row in rows}),
        "sample_counts": dict(Counter(int(getattr(decision, "q_per_world").shape[0]) for game in games for decision in getattr(game, "decisions", []) if getattr(decision, "q_per_world", None) is not None)),
        "seeds": [int(seed) for seed in seeds[:5]],
        "seeds_tail": [int(seed) for seed in seeds[-5:]],
    }
    return rows, meta


def row_to_dict(row: ActionRow) -> dict[str, Any]:
    out = row.__dict__.copy()
    out["labels"] = "|".join(row.labels)
    return out


def rows_with_label(rows: list[ActionRow], label: str) -> list[ActionRow]:
    return [row for row in rows if label in row.labels]


def summarize_label(
    *,
    rows: list[ActionRow],
    label: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected = rows_with_label(rows, label)
    decisions = {row.key for row in selected}
    mean_regret, regret_lo, regret_hi = bootstrap_mean_ci(
        [row.mean_regret for row in selected],
        samples=bootstrap_samples,
        seed=stable_seed(bootstrap_seed, label),
    )
    mean_threshold_gap, threshold_lo, threshold_hi = bootstrap_mean_ci(
        [row.threshold_gap for row in selected],
        samples=bootstrap_samples,
        seed=stable_seed(bootstrap_seed, label + ":threshold"),
    )
    return {
        "claim_label": label,
        "action_n": len(selected),
        "decision_n": len(decisions),
        "mean_regret": mean_regret,
        "mean_regret_ci95_low": regret_lo,
        "mean_regret_ci95_high": regret_hi,
        "mean_threshold_gap": mean_threshold_gap,
        "mean_threshold_gap_ci95_low": threshold_lo,
        "mean_threshold_gap_ci95_high": threshold_hi,
        "mean_threshold_mass": mean([row.threshold_mass for row in selected]),
        "mean_lower_tail_mass": mean([row.lower_tail_mass for row in selected]),
        "actual_action_rate": mean([1.0 if row.is_actual_action else 0.0 for row in selected]),
        "best_mean_rate": mean([1.0 if row.is_best_mean else 0.0 for row in selected]),
        "best_threshold_rate": mean([1.0 if row.is_best_threshold else 0.0 for row in selected]),
        "safest_tail_rate": mean([1.0 if row.is_safest_tail else 0.0 for row in selected]),
    }


def grouped_by_decision(rows: list[ActionRow]) -> dict[str, list[ActionRow]]:
    grouped: dict[str, list[ActionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.key].append(row)
    return grouped


def best_by_mean(rows: list[ActionRow]) -> ActionRow:
    return max(rows, key=lambda row: row.mean)


def paired_contrast(
    *,
    rows: list[ActionRow],
    contrast_id: str,
    preferred: Callable[[ActionRow], bool],
    alternative: Callable[[ActionRow], bool],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    deltas_mean: list[float] = []
    deltas_regret: list[float] = []
    deltas_threshold: list[float] = []
    examples: list[dict[str, Any]] = []

    for decision_rows in grouped_by_decision(rows).values():
        pref_rows = [row for row in decision_rows if preferred(row)]
        alt_rows = [row for row in decision_rows if alternative(row)]
        if not pref_rows or not alt_rows:
            continue
        pref = best_by_mean(pref_rows)
        alt = best_by_mean(alt_rows)
        mean_delta = pref.mean - alt.mean
        regret_delta = pref.mean_regret - alt.mean_regret
        threshold_delta = pref.threshold_mass - alt.threshold_mass
        deltas_mean.append(mean_delta)
        deltas_regret.append(regret_delta)
        deltas_threshold.append(threshold_delta)
        if len(examples) < 16 or abs(mean_delta) > max(abs(float(e["mean_delta"])) for e in examples):
            examples.append(
                {
                    "contrast_id": contrast_id,
                    "key": pref.key,
                    "seed": pref.seed,
                    "decl_id": pref.decl_id,
                    "decl_name": pref.decl_name,
                    "decision_idx": pref.decision_idx,
                    "trick_idx": pref.trick_idx,
                    "trick_position": pref.trick_position,
                    "actor": pref.actor,
                    "seat_role": pref.seat_role,
                    "score_before": f"{pref.offense_score_before}-{pref.defense_score_before}",
                    "preferred_domino": pref.candidate_domino,
                    "preferred_labels": "|".join(pref.labels),
                    "preferred_mean": pref.mean,
                    "preferred_threshold_mass": pref.threshold_mass,
                    "preferred_lower_tail_mass": pref.lower_tail_mass,
                    "alternative_domino": alt.candidate_domino,
                    "alternative_labels": "|".join(alt.labels),
                    "alternative_mean": alt.mean,
                    "alternative_threshold_mass": alt.threshold_mass,
                    "alternative_lower_tail_mass": alt.lower_tail_mass,
                    "mean_delta": mean_delta,
                    "regret_delta": regret_delta,
                    "threshold_mass_delta": threshold_delta,
                }
            )
            examples = sorted(examples, key=lambda e: abs(float(e["mean_delta"])), reverse=True)[:16]

    mean_delta, mean_lo, mean_hi = bootstrap_mean_ci(
        deltas_mean,
        samples=bootstrap_samples,
        seed=stable_seed(bootstrap_seed, contrast_id),
    )
    regret_delta, regret_lo, regret_hi = bootstrap_mean_ci(
        deltas_regret,
        samples=bootstrap_samples,
        seed=stable_seed(bootstrap_seed, contrast_id + ":regret"),
    )
    threshold_delta, threshold_lo, threshold_hi = bootstrap_mean_ci(
        deltas_threshold,
        samples=bootstrap_samples,
        seed=stable_seed(bootstrap_seed, contrast_id + ":threshold"),
    )
    return {
        "contrast_id": contrast_id,
        "paired_decision_n": len(deltas_mean),
        "mean_delta": mean_delta,
        "mean_delta_ci95_low": mean_lo,
        "mean_delta_ci95_high": mean_hi,
        "regret_delta": regret_delta,
        "regret_delta_ci95_low": regret_lo,
        "regret_delta_ci95_high": regret_hi,
        "threshold_mass_delta": threshold_delta,
        "threshold_mass_delta_ci95_low": threshold_lo,
        "threshold_mass_delta_ci95_high": threshold_hi,
        "examples": examples,
    }


def slice_by_decl(rows: list[ActionRow], labels: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in labels:
        for decl_id in sorted({row.decl_id for row in rows}):
            selected = [row for row in rows if row.decl_id == decl_id and label in row.labels]
            if not selected:
                continue
            out.append(
                {
                    "claim_label": label,
                    "decl_id": decl_id,
                    "decl_name": selected[0].decl_name,
                    "action_n": len(selected),
                    "decision_n": len({row.key for row in selected}),
                    "mean_regret": mean([row.mean_regret for row in selected]),
                    "mean_threshold_gap": mean([row.threshold_gap for row in selected]),
                    "mean_threshold_mass": mean([row.threshold_mass for row in selected]),
                    "best_mean_rate": mean([1.0 if row.is_best_mean else 0.0 for row in selected]),
                }
            )
    return out


def build_examples(rows: list[ActionRow], labels: list[str]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for label in labels:
        selected = sorted(rows_with_label(rows, label), key=lambda row: row.mean_regret, reverse=True)[:8]
        for row in selected:
            examples.append(
                {
                    "claim_label": label,
                    "key": row.key,
                    "seed": row.seed,
                    "decl_id": row.decl_id,
                    "decl_name": row.decl_name,
                    "decision_idx": row.decision_idx,
                    "trick_idx": row.trick_idx,
                    "trick_position": row.trick_position,
                    "actor": row.actor,
                    "seat_role": row.seat_role,
                    "team": row.team,
                    "score_before": f"{row.offense_score_before}-{row.defense_score_before}",
                    "current_winner_team_before": row.current_winner_team_before,
                    "candidate": row.candidate_domino,
                    "candidate_count_points": row.candidate_count_points,
                    "candidate_beats_current": row.candidate_beats_current,
                    "mean": row.mean,
                    "mean_regret": row.mean_regret,
                    "threshold_mass": row.threshold_mass,
                    "threshold_gap": row.threshold_gap,
                    "lower_tail_mass": row.lower_tail_mass,
                    "labels": "|".join(row.labels),
                }
            )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["gus/data/corpus_v2_train_*_d0-9.pt"],
        help="Input .pt files or glob patterns.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-files", type=int, default=0, help="Limit files for smoke runs; 0 means all.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260502)
    parser.add_argument("--log-every-files", type=int, default=1)
    add_wandb_args(
        parser,
        default_group="w42-gus-corpus-claim-deep-dive",
        default_enabled=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = expand_inputs(args.inputs)
    if args.max_files:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit("No input files matched.")

    config = {
        "experiment": "w42-gus-corpus-claim-deep-dive",
        "inputs": [str(path) for path in paths],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "claim_labels": [
            "ch05_setter_pounce_count_before_certainty",
            "ch05_setter_pounce_count",
            "ch05_setter_pounce_count_sets_now",
            "ch05_reckless_count_to_bidder",
            "ch04_partner_safe_count_donation_current_control",
            "ch04_partner_unsafe_count_to_defense",
        ],
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=output_dir,
        tags=["w42", "gus-corpus", "claim-deep-dive", "winning42"],
    )
    if getattr(wb, "run", None) is not None:
        wb.run.define_metric("progress/files_processed")
        for pattern in ("progress/*", "coverage/*", "claims/*"):
            wb.run.define_metric(pattern, step_metric="progress/files_processed")

    started = time.perf_counter()
    rows: list[ActionRow] = []
    file_meta: list[dict[str, Any]] = []
    for idx, path in enumerate(paths, start=1):
        file_rows, meta = load_payload_rows(path)
        rows.extend(file_rows)
        file_meta.append(meta)
        if idx % max(int(args.log_every_files), 1) == 0:
            label_counts = Counter(label for row in rows for label in row.labels)
            wb.log(
                {
                    "progress/files_processed": idx,
                    "progress/action_rows": len(rows),
                    "coverage/decision_rows": len({row.key for row in rows}),
                    "claims/setter_pounce_count": label_counts["ch05_setter_pounce_count"],
                    "claims/setter_pounce_sets_now": label_counts["ch05_setter_pounce_count_sets_now"],
                    "claims/reckless_count_to_bidder": label_counts["ch05_reckless_count_to_bidder"],
                    "claims/partner_safe_donation": label_counts[
                        "ch04_partner_safe_count_donation_current_control"
                    ],
                    "claims/partner_unsafe_donation": label_counts["ch04_partner_unsafe_count_to_defense"],
                    "progress/wall_seconds": time.perf_counter() - started,
                },
                step=idx,
            )

    labels = config["claim_labels"]
    claim_metrics = [
        summarize_label(
            rows=rows,
            label=label,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for label in labels
    ]

    contrasts = [
        paired_contrast(
            rows=rows,
            contrast_id="ch05_pounce_count_vs_other_same_decision",
            preferred=lambda row: "ch05_setter_pounce_count" in row.labels,
            alternative=lambda row: "ch05_setter_pounce_count" not in row.labels,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        paired_contrast(
            rows=rows,
            contrast_id="ch05_pounce_sets_now_vs_other_same_decision",
            preferred=lambda row: "ch05_setter_pounce_count_sets_now" in row.labels,
            alternative=lambda row: "ch05_setter_pounce_count_sets_now" not in row.labels,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        paired_contrast(
            rows=rows,
            contrast_id="ch05_reckless_count_vs_nonreckless_same_decision",
            preferred=lambda row: "ch05_reckless_count_to_bidder" in row.labels,
            alternative=lambda row: "ch05_reckless_count_to_bidder" not in row.labels,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        paired_contrast(
            rows=rows,
            contrast_id="ch04_safe_partner_count_vs_other_same_decision",
            preferred=lambda row: "ch04_partner_safe_count_donation_current_control" in row.labels,
            alternative=lambda row: "ch04_partner_safe_count_donation_current_control" not in row.labels,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        paired_contrast(
            rows=rows,
            contrast_id="ch04_unsafe_partner_count_vs_nonunsafe_same_decision",
            preferred=lambda row: "ch04_partner_unsafe_count_to_defense" in row.labels,
            alternative=lambda row: "ch04_partner_unsafe_count_to_defense" not in row.labels,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
    ]

    label_counts = Counter(label for row in rows for label in row.labels)
    summary = {
        "schema_version": "w42.gus_corpus_claim_deep_dive.v0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_sha(),
        "git_status_before_artifacts": git_status_short(),
        "input_files": [str(path) for path in paths],
        "file_meta": file_meta,
        "coverage": {
            "input_files": len(paths),
            "action_rows": len(rows),
            "decision_rows": len({row.key for row in rows}),
            "claim_action_rows": sum(1 for row in rows if row.labels),
            "claim_decision_rows": len({row.key for row in rows if row.labels}),
            "label_counts": dict(sorted(label_counts.items())),
        },
        "claim_metrics": claim_metrics,
        "paired_contrasts": [{k: v for k, v in contrast.items() if k != "examples"} for contrast in contrasts],
        "scientific_status": {
            "interpretation": (
                "Direct corpus report over existing Gus v2 joint-world records. "
                "Tests role-gated tactical labels with oracle E[Q] distributions; "
                "does not train a policy or use hidden labels as live features."
            ),
            "wandb": wb.status(),
            "hf": "not applicable",
            "claim_ledger_impact": "local evidence update; conservative wiki status required",
        },
        "wall_seconds": time.perf_counter() - started,
    }

    claim_metric_rows = claim_metrics
    contrast_rows = [{k: v for k, v in contrast.items() if k != "examples"} for contrast in contrasts]
    decl_rows = slice_by_decl(rows, labels)
    examples = build_examples(rows, labels)
    contrast_examples = [example for contrast in contrasts for example in contrast["examples"]]
    claim_rows = [row_to_dict(row) for row in rows if row.labels]

    write_json(output_dir / "summary.json", summary)
    write_csv(
        output_dir / "claim_metrics.csv",
        claim_metric_rows,
        [
            "claim_label",
            "action_n",
            "decision_n",
            "mean_regret",
            "mean_regret_ci95_low",
            "mean_regret_ci95_high",
            "mean_threshold_gap",
            "mean_threshold_gap_ci95_low",
            "mean_threshold_gap_ci95_high",
            "mean_threshold_mass",
            "mean_lower_tail_mass",
            "actual_action_rate",
            "best_mean_rate",
            "best_threshold_rate",
            "safest_tail_rate",
        ],
    )
    write_csv(
        output_dir / "paired_contrasts.csv",
        contrast_rows,
        [
            "contrast_id",
            "paired_decision_n",
            "mean_delta",
            "mean_delta_ci95_low",
            "mean_delta_ci95_high",
            "regret_delta",
            "regret_delta_ci95_low",
            "regret_delta_ci95_high",
            "threshold_mass_delta",
            "threshold_mass_delta_ci95_low",
            "threshold_mass_delta_ci95_high",
        ],
    )
    write_csv(
        output_dir / "slice_metrics_by_decl.csv",
        decl_rows,
        [
            "claim_label",
            "decl_id",
            "decl_name",
            "action_n",
            "decision_n",
            "mean_regret",
            "mean_threshold_gap",
            "mean_threshold_mass",
            "best_mean_rate",
        ],
    )
    write_json(output_dir / "examples.json", examples)
    write_json(output_dir / "paired_examples.json", contrast_examples)
    write_jsonl(output_dir / "claim_action_rows.jsonl", claim_rows)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "w42.gus_corpus_claim_deep_dive.manifest.v0",
            "created_at_utc": summary["created_at_utc"],
            "command": " ".join(sys.argv),
            "outputs": {
                "summary": str(output_dir / "summary.json"),
                "claim_metrics": str(output_dir / "claim_metrics.csv"),
                "paired_contrasts": str(output_dir / "paired_contrasts.csv"),
                "slice_metrics_by_decl": str(output_dir / "slice_metrics_by_decl.csv"),
                "examples": str(output_dir / "examples.json"),
                "paired_examples": str(output_dir / "paired_examples.json"),
                "claim_action_rows": str(output_dir / "claim_action_rows.jsonl"),
            },
            "leakage_boundary": (
                "Public role/trick/action labels are report features. E[Q], q_per_world, "
                "and sampled-world outcomes are offline labels and diagnostics only."
            ),
            "wandb": wb.status(),
        },
    )

    wb.update_summary(
        {
            "coverage/action_rows": summary["coverage"]["action_rows"],
            "coverage/decision_rows": summary["coverage"]["decision_rows"],
            "coverage/claim_action_rows": summary["coverage"]["claim_action_rows"],
            "status": "completed",
        }
    )
    wb.log_artifact_files(
        name=f"w42-gus-corpus-claim-deep-dive-{git_sha()[:7]}",
        artifact_type="w42-report",
        paths=[
            output_dir / "summary.json",
            output_dir / "claim_metrics.csv",
            output_dir / "paired_contrasts.csv",
            output_dir / "slice_metrics_by_decl.csv",
            output_dir / "examples.json",
            output_dir / "paired_examples.json",
            output_dir / "manifest.json",
        ],
    )
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
