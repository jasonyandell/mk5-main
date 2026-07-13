#!/usr/bin/env python3
"""Run W42 phase-4 dynamic 84 tests on mined natural seeds.

This runner promotes the phase-3 seed-mining corpus from static hand/deal rows
into reached legal-action tables. It deliberately stays inside the existing
branch-atlas path: full deals are rotated so the mined bidder is player 0, the
84 contract is generated with schema-v2 E[Q] PDFs, then reached legal actions
are labeled with Chapter 7/8 preservation surfaces.

The current engine path does not expose arbitrary late-state injection. Outputs
therefore distinguish action-value contrasts observed in generated policy
traces from blocker rows that still require injection or a stronger replay API.
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

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.eq.generate.pipeline import generate_eq_games_gpu
from forge.eq.oracle import Stage1Oracle
from forge.oracle import tables
from w42.branch_atlas_v1.build_branch_atlas import domino_label, git_sha, sha256_file


OUT_DIR = ROOT / "w42" / "phase4_84_dynamic_seed_tests"
SEED_DIR = ROOT / "w42" / "eighty_four_seed_mining"
EQ_PATH = OUT_DIR / "mined_84_seed_games.pt"
ATLAS_DIR = OUT_DIR / "branch_atlas"
BEAD_ID = "t42-br7n.2"
DECL_NAMES = {
    0: "blanks",
    1: "ones",
    2: "twos",
    3: "threes",
    4: "fours",
    5: "fives",
    6: "sixes",
}
PRIMARY_SURFACES = [
    "protected_one_off",
    "straight_one_off",
    "two_off_same_suit",
    "three_trump_three_double_one_off",
    "laydown_all_trumps",
]
DEFENSE_SURFACES = [
    "defender_live_double_weapon",
    "defender_live_same_suit_pair",
    "pair_protector_pressure",
    "dead_asset_release_control",
]


def domino_id(label: str) -> int:
    high_s, low_s = label.strip().split("-")
    high = int(high_s)
    low = int(low_s)
    if low > high:
        high, low = low, high
    return tables.DOMINOES.index((high, low))


def hand_ids(label_csv: str) -> list[int]:
    return [domino_id(part) for part in label_csv.split(",") if part]


def parse_full_deal(raw: str) -> list[list[int]]:
    return [hand_ids(hand) for hand in json.loads(raw)]


def rotate_deal_to_bidder_zero(deal: list[list[int]], bidder_seat: int) -> list[list[int]]:
    return [deal[(bidder_seat + offset) % 4] for offset in range(4)]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def intish(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no", ""}:
        return 0
    return int(float(text))


def mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def detect_checkpoint(raw: str | None) -> str:
    if raw:
        return raw
    model_dir = ROOT / "forge" / "models"
    candidates = [
        model_dir / "domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
        model_dir / "domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt",
        ROOT / "checkpoints" / "stage1" / "best.ckpt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("No E[Q] checkpoint found; pass --checkpoint")


def resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def row_surfaces(row: dict[str, str]) -> set[str]:
    return {part for part in row.get("surfaces", "").split("|") if part}


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["seed"], row["bidder_seat"], row["decl_id"], row.get("final_off", ""))


def load_seed_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    recommended = read_csv(args.recommended_rows)
    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(row: dict[str, str]) -> None:
        key = row_key(row)
        if key in seen:
            return
        seen.add(key)
        selected.append(row)

    for surface in ["protected_one_off", "straight_one_off"]:
        for row in recommended:
            if surface in row_surfaces(row):
                add(row)
                break

    # The recommendation table is dominated by one-off rows; pull rarer two-off
    # and laydown cases directly from the candidate corpus.
    candidate_path = args.candidate_rows
    needed_from_candidates = {
        "two_off_same_suit": args.rare_surface_games,
        "laydown_all_trumps": 1,
    }
    if candidate_path.exists():
        for target, wanted in needed_from_candidates.items():
            found = 0
            target_seeds: set[str] = set()
            with candidate_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if target in row_surfaces(row) and row["seed"] not in target_seeds:
                        add(row)
                        target_seeds.add(row["seed"])
                        found += 1
                        if found >= wanted:
                            break

    for row in recommended:
        if len(selected) >= args.max_games:
            break
        add(row)

    return selected[: args.max_games]


def selected_metadata(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        bidder_seat = int(row["bidder_seat"])
        surfaces = sorted(row_surfaces(row))
        out.append(
            {
                "game_idx": idx,
                "source_seed": int(row["seed"]),
                "source_bidder_seat": bidder_seat,
                "source_partner_seat": int(row["partner_seat"]),
                "rotation": {
                    "p0_bidder_from_source_seat": bidder_seat,
                    "p1_left_setter_from_source_seat": (bidder_seat + 1) % 4,
                    "p2_partner_from_source_seat": (bidder_seat + 2) % 4,
                    "p3_right_setter_from_source_seat": (bidder_seat + 3) % 4,
                },
                "decl_id": int(row["decl_id"]),
                "decl_name": row["decl_name"],
                "bid_value": 84,
                "final_off": row.get("final_off", ""),
                "surfaces": surfaces,
                "primary_surfaces": [surface for surface in PRIMARY_SURFACES if surface in surfaces],
                "defense_surfaces": [surface for surface in DEFENSE_SURFACES if surface in surfaces],
                "recommendation_score": int(row["recommendation_score"]),
                "trump_count": int(row["trump_count"]),
                "off_count": int(row["off_count"]),
                "nontrump_double_count": int(row["nontrump_double_count"]),
                "total_double_count": int(row["total_double_count"]),
                "opponent_matching_double_count": int(row["opponent_matching_double_count"]),
                "opponent_same_suit_pair_count": int(row["opponent_same_suit_pair_count"]),
                "opponent_pair_protector_count": int(row["opponent_pair_protector_count"]),
                "opponent_dead_double_assets": int(row["opponent_dead_double_assets"]),
                "rotated_hands": [
                    [domino_label(domino) for domino in hand]
                    for hand in rotate_deal_to_bidder_zero(parse_full_deal(row["full_deal"]), bidder_seat)
                ],
            }
        )
    return out


def run_generation(args: argparse.Namespace, rows: list[dict[str, str]], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = detect_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    print(f"Loading model from {checkpoint} on {device}...", flush=True)
    oracle = Stage1Oracle(checkpoint, device=device, compile=False)

    hands = [
        rotate_deal_to_bidder_zero(parse_full_deal(row["full_deal"]), int(row["bidder_seat"]))
        for row in rows
    ]
    decl_ids = [int(row["decl_id"]) for row in rows]
    bid_values = [84 for _ in rows]

    t0 = time.perf_counter()
    results = generate_eq_games_gpu(
        model=oracle.model,
        hands=hands,
        decl_ids=decl_ids,
        n_samples=args.samples,
        device=device,
        greedy=True,
        save_joint_worlds=True,
        schema_v2=True,
        bid_values=bid_values,
    )
    elapsed = time.perf_counter() - t0
    payload = {
        "results": results,
        "seeds": [int(row["seed"]) for row in rows],
        "decl_ids": decl_ids,
        "bid_values": bid_values,
        "schema": "v2",
        "n_samples": args.samples,
        "checkpoint": checkpoint,
        "bead_id": BEAD_ID,
        "experiment": "w42-phase4-84-dynamic-mined-seed-tests",
        "seed_metadata": metadata,
        "rotation_note": "Each full deal is rotated so mined bidder becomes player 0 for branch_atlas_v1 role labels.",
    }
    torch.save(payload, EQ_PATH)
    return {
        "checkpoint": checkpoint,
        "device": device,
        "samples": args.samples,
        "elapsed_seconds": elapsed,
        "games": len(results),
    }


def run_branch_atlas(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "w42/branch_atlas_v1/build_branch_atlas.py",
        str(EQ_PATH),
        "--output-dir",
        str(ATLAS_DIR),
        "--bead-id",
        BEAD_ID,
        "--experiment",
        "w42-phase4-84-dynamic-mined-seed-tests",
        "--top-k",
        str(args.top_k),
        "--examples",
        "24",
        "--log-every-decisions",
        "10",
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-group",
        "w42-phase4-84-dynamic-seed-tests",
        "--wandb-name",
        args.wandb_name,
    ]
    if args.no_wandb:
        cmd.append("--no-wandb")
    subprocess.run(cmd, cwd=ROOT, check=True)


def side_pips(domino: int, decl_id: int) -> set[int]:
    if tables.is_in_called_suit(domino, decl_id):
        return set()
    return {int(tables.DOMINO_HIGH[domino]), int(tables.DOMINO_LOW[domino])}


def pip_rank(domino: int) -> int:
    return 14 if tables.DOMINO_IS_DOUBLE[domino] else int(tables.DOMINO_SUM[domino])


def beats_nontrump_in_same_suit(candidate: int, target: int, decl_id: int) -> bool:
    return bool(side_pips(candidate, decl_id) & side_pips(target, decl_id)) and pip_rank(candidate) > pip_rank(target)


def reconstruct_remaining_by_decision(payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    by_decision: dict[tuple[int, int], dict[str, Any]] = {}
    for game_idx, game in enumerate(payload.get("results", [])):
        hands = [[int(domino) for domino in hand] for hand in getattr(game, "hands")]
        played_slots: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}
        for decision_idx, decision in enumerate(getattr(game, "decisions", [])):
            actor = int(getattr(decision, "player"))
            remaining = [domino for slot, domino in enumerate(hands[actor]) if slot not in played_slots[actor]]
            bidder_remaining = [domino for slot, domino in enumerate(hands[0]) if slot not in played_slots[0]]
            by_decision[(game_idx, decision_idx)] = {
                "actor": actor,
                "remaining": remaining,
                "bidder_remaining": bidder_remaining,
                "hands": hands,
            }
            played_slots[actor].add(int(getattr(decision, "action_taken")))
    return by_decision


def live_pair_suits(remaining: list[int], bidder_offs: list[int], decl_id: int) -> set[int]:
    suits: set[int] = set()
    for idx, left in enumerate(remaining):
        for right in remaining[idx + 1 :]:
            shared = side_pips(left, decl_id) & side_pips(right, decl_id)
            if not shared:
                continue
            if any(
                beats_nontrump_in_same_suit(left, off, decl_id)
                and beats_nontrump_in_same_suit(right, off, decl_id)
                for off in bidder_offs
            ):
                suits.update(shared)
    return suits


def candidate_labels(row: dict[str, str], context: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    labels = ["is_84_contract_decision"]
    candidate = int(row["candidate_domino_id"])
    decl_id = int(row["decl_id"])
    team = row["team"]
    candidate_double = bool(intish(row["candidate_is_double"]))
    candidate_called = bool(intish(row["candidate_is_called_suit"]))
    remaining = context["remaining"]
    bidder_remaining = context["bidder_remaining"]
    bidder_offs = [domino for domino in bidder_remaining if not tables.is_in_called_suit(domino, decl_id)]
    final_off_pips = set()
    if metadata.get("final_off"):
        final_off_pips = side_pips(domino_id(metadata["final_off"]), decl_id)

    if team == "offense" and candidate_called:
        labels.append("bidder_trump_pull_candidate")
    if team == "offense" and not candidate_called:
        labels.append("bidder_final_off_candidate")
        if not candidate_double:
            labels.append("bidder_nontrump_final_off_candidate")
    for surface in metadata["primary_surfaces"]:
        if team == "offense" or surface == "laydown_all_trumps":
            labels.append(f"surface_{surface}")

    asset_priority = 0
    live_target_count = 0
    throwaway_rank = 6
    if team == "defense":
        if candidate_double:
            target_pips = side_pips(candidate, decl_id)
            live_target_count = sum(1 for off in bidder_offs if target_pips & side_pips(off, decl_id))
            if any(beats_nontrump_in_same_suit(candidate, off, decl_id) for off in bidder_offs):
                labels.append("defender_live_double_weapon_proxy")
                asset_priority = max(asset_priority, 5)
                throwaway_rank = min(throwaway_rank, 1)

        matching_pair_tiles = [
            other
            for other in remaining
            if other != candidate
            and side_pips(candidate, decl_id) & side_pips(other, decl_id)
            and any(
                beats_nontrump_in_same_suit(candidate, off, decl_id)
                and beats_nontrump_in_same_suit(other, off, decl_id)
                for off in bidder_offs
            )
        ]
        if matching_pair_tiles:
            labels.append("defender_live_same_suit_pair_proxy")
            asset_priority = max(asset_priority, 4)
            throwaway_rank = min(throwaway_rank, 2)

        pair_suits = live_pair_suits(remaining, bidder_offs, decl_id)
        if asset_priority == 0 and side_pips(candidate, decl_id) & pair_suits:
            labels.append("pair_protector_proxy")
            asset_priority = max(asset_priority, 3)
            throwaway_rank = min(throwaway_rank, 3)

        if (
            candidate_double
            and asset_priority == 0
            and not candidate_called
            and not (side_pips(candidate, decl_id) & final_off_pips)
        ):
            labels.append("dead_asset_release_candidate_proxy")
            throwaway_rank = min(throwaway_rank, 4)

        if int(row["candidate_count_points"]) == 0 and not candidate_double and asset_priority == 0:
            labels.append("partner_readable_low_or_blank_throwaway_proxy")
            throwaway_rank = min(throwaway_rank, 5)

    return {
        "dynamic_84_labels": "|".join(sorted(set(labels))),
        "asset_priority": asset_priority,
        "book_throwaway_rank": throwaway_rank,
        "live_double_target_count": live_target_count,
    }


def add_contrast(
    contrast_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    contrast_id: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    preferred_label: str,
    alternate_label: str,
) -> None:
    if not pairs:
        return
    deltas = [preferred["mean"] - alternate["mean"] for preferred, alternate in pairs]
    threshold_deltas = [preferred["threshold_mass"] - alternate["threshold_mass"] for preferred, alternate in pairs]
    tail_deltas = [preferred["lower_tail_mass"] - alternate["lower_tail_mass"] for preferred, alternate in pairs]
    contrast_rows.append(
        {
            "contrast_id": contrast_id,
            "paired_decision_n": len(pairs),
            "preferred_label": preferred_label,
            "alternate_label": alternate_label,
            "mean_delta": mean(deltas),
            "threshold_mass_delta": mean(threshold_deltas),
            "lower_tail_mass_delta": mean(tail_deltas),
        }
    )
    for preferred, alternate in pairs[:6]:
        examples.append(
            {
                "contrast_id": contrast_id,
                "decision_key": preferred["decision_key"],
                "surface": preferred["primary_surfaces"],
                "preferred": {
                    "domino": preferred["candidate_domino"],
                    "mean": preferred["mean"],
                    "labels": preferred["dynamic_84_labels"],
                    "throwaway_rank": preferred["book_throwaway_rank"],
                    "live_double_target_count": preferred["live_double_target_count"],
                },
                "alternate": {
                    "domino": alternate["candidate_domino"],
                    "mean": alternate["mean"],
                    "labels": alternate["dynamic_84_labels"],
                    "throwaway_rank": alternate["book_throwaway_rank"],
                    "live_double_target_count": alternate["live_double_target_count"],
                },
            }
        )


def summarize_label(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "action_n": len(rows),
        "decision_n": len({row["decision_key"] for row in rows}),
        "mean": mean([row["mean"] for row in rows]),
        "threshold_mass": mean([row["threshold_mass"] for row in rows]),
        "lower_tail_mass": mean([row["lower_tail_mass"] for row in rows]),
        "actual_action_rate": mean([1.0 if row["is_actual_action"] else 0.0 for row in rows]),
        "top_mean_rate": mean([1.0 if row["is_top_mean_action"] else 0.0 for row in rows]),
        "top_threshold_rate": mean([1.0 if row["is_top_threshold_action"] else 0.0 for row in rows]),
    }


def write_blocker_table() -> list[dict[str, Any]]:
    rows = [
        {
            "claim_surface": "preserve_spend_final_set_attribution",
            "status": "partial_action_value_only",
            "current_output": "Reached legal actions can compare proxy preserve/spend alternatives.",
            "blocker": "No arbitrary late-state injector, so unreached final-two-trick tableaux cannot be forced.",
            "next_plan": "Use selected_seed_rows.csv to inject or replay to trick 5/6 and score final set attribution.",
        },
        {
            "claim_surface": "throwaway_priority_ladder",
            "status": "partial_reached_discards_only",
            "current_output": "Free-discard reached decisions receive book throwaway ranks and paired contrasts.",
            "blocker": "Policy trace may not reach bottleneck states where every ladder rung is simultaneously legal.",
            "next_plan": "Construct or inject states with live double, live pair, protector, and signal tile choices.",
        },
        {
            "claim_surface": "score_42_vs_84_gate",
            "status": "static_gate_only",
            "current_output": "score_gate_static_table.csv states the clean match-score dominance cases.",
            "blocker": "Current play generator starts after bidding and does not evaluate terminal match win-rate for alternate bids.",
            "next_plan": "Add bid-stage counterfactuals for 42 vs 84 with score before hand in 208..249.",
        },
        {
            "claim_surface": "straight_off_nearly_two_thirds_set_rate",
            "status": "seed/action_proxy_not_policy_proof",
            "current_output": "Straight-off mined seeds are action-scored under the current greedy policy trace.",
            "blocker": "The book's 'good players set nearly two out of three' is a policy/population claim, not just ownership.",
            "next_plan": "Run large straight-off vs protected-off cohorts with explicit defender preservation policies.",
        },
    ]
    write_csv(OUT_DIR / "blocker_table.csv", rows)
    return rows


def write_score_gate_table(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for meta in metadata:
        for score_before in [208, 209, 220, 240]:
            rows.append(
                {
                    "game_idx": meta["game_idx"],
                    "source_seed": meta["source_seed"],
                    "primary_surfaces": "|".join(meta["primary_surfaces"]),
                    "score_before_hand": score_before,
                    "score_after_42_make": score_before + 42,
                    "score_after_84_make": score_before + 84,
                    "bid_42_already_wins_match": score_before + 42 >= 250,
                    "extra_set_exposure_points_if_bid_84": 42,
                    "engine_status": "static gate; no bid-stage terminal win-rate counterfactual in this runner",
                }
            )
    write_csv(OUT_DIR / "score_gate_static_table.csv", rows)
    return rows


def analyze_outputs(generation_info: dict[str, Any], metadata: list[dict[str, Any]]) -> dict[str, Any]:
    payload = torch.load(EQ_PATH, weights_only=False, map_location="cpu")
    by_decision_ctx = reconstruct_remaining_by_decision(payload)
    raw_actions = read_csv(ATLAS_DIR / "decision_actions.csv")
    meta_by_game = {int(meta["game_idx"]): meta for meta in metadata}

    labeled_rows: list[dict[str, Any]] = []
    by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_actions:
        game_idx = int(row["game_idx"])
        meta = meta_by_game[game_idx]
        label_info = candidate_labels(row, by_decision_ctx[(game_idx, int(row["decision_idx"]))], meta)
        decision_key = f"{game_idx}:{row['decision_idx']}"
        out = {
            "game_idx": game_idx,
            "source_seed": int(row["seed"]),
            "source_bidder_seat": int(meta["source_bidder_seat"]),
            "decl_id": int(row["decl_id"]),
            "decl_name": row["decl_name"],
            "bid_value": int(row["bid_value"]),
            "primary_surfaces": "|".join(meta["primary_surfaces"]),
            "defense_surfaces": "|".join(meta["defense_surfaces"]),
            "final_off": meta["final_off"],
            "decision_key": decision_key,
            "decision_idx": int(row["decision_idx"]),
            "trick_idx": int(row["trick_idx"]),
            "trick_position": int(row["trick_position"]),
            "actor": int(row["actor"]),
            "seat_role": row["seat_role"],
            "team": row["team"],
            "candidate_domino": row["candidate_domino"],
            "actual_action_domino": row["actual_action_domino"],
            "candidate_can_follow_led": intish(row["candidate_can_follow_led"]),
            "candidate_count_points": int(row["candidate_count_points"]),
            **label_info,
            "mean": numeric(row["mean"]),
            "threshold_mass": numeric(row["threshold_mass"]),
            "lower_tail_mass": numeric(row["lower_tail_mass_le_neg18"]),
            "top_hidden_impact_score": numeric(row["top_hidden_impact_score"]),
            "top_hidden_downside_score": numeric(row["top_hidden_downside_score"]),
            "is_actual_action": bool(intish(row["is_actual_action"])),
            "is_top_mean_action": bool(intish(row["is_top_mean_action"])),
            "is_top_threshold_action": bool(intish(row["is_top_threshold_action"])),
        }
        labeled_rows.append(out)
        by_decision[decision_key].append(out)

    # Mark free-discard decisions after all rows are grouped.
    for rows in by_decision.values():
        is_free_discard = (
            rows[0]["team"] == "defense"
            and rows[0]["trick_position"] > 0
            and all(not row["candidate_can_follow_led"] for row in rows)
        )
        for row in rows:
            row["is_defense_free_discard_decision"] = is_free_discard

    label_names = sorted(
        {
            label
            for row in labeled_rows
            for label in row["dynamic_84_labels"].split("|")
            if label
        }
    )
    label_metrics = []
    for label in label_names:
        selected = [row for row in labeled_rows if label in row["dynamic_84_labels"].split("|")]
        metric = {"dynamic_84_label": label}
        metric.update(summarize_label(selected))
        label_metrics.append(metric)

    contrast_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    preserve_pairs = []
    dead_release_pairs = []
    lower_target_double_pairs = []
    for rows in by_decision.values():
        defense = [row for row in rows if row["team"] == "defense"]
        if not defense:
            continue
        critical = [row for row in defense if row["book_throwaway_rank"] <= 3]
        expendable = [row for row in defense if row["book_throwaway_rank"] >= 4]
        if critical and expendable:
            preserve_pairs.append((max(expendable, key=lambda row: row["mean"]), max(critical, key=lambda row: row["mean"])))
        dead = [row for row in defense if "dead_asset_release_candidate_proxy" in row["dynamic_84_labels"].split("|")]
        if dead and critical:
            dead_release_pairs.append((max(dead, key=lambda row: row["mean"]), max(critical, key=lambda row: row["mean"])))
        live_doubles = [
            row
            for row in defense
            if "defender_live_double_weapon_proxy" in row["dynamic_84_labels"].split("|")
            and row["live_double_target_count"] > 0
        ]
        if len(live_doubles) >= 2:
            high = max(live_doubles, key=lambda row: (row["live_double_target_count"], row["mean"]))
            low = min(live_doubles, key=lambda row: (row["live_double_target_count"], -row["mean"]))
            if high["live_double_target_count"] > low["live_double_target_count"]:
                lower_target_double_pairs.append((low, high))

    add_contrast(
        contrast_rows,
        examples,
        "defense_preserve_expendable_vs_spend_live_asset_proxy",
        preserve_pairs,
        "play expendable/dead asset, preserving live weapon or protector",
        "spend live weapon or protector",
    )
    add_contrast(
        contrast_rows,
        examples,
        "dead_asset_release_vs_spend_live_asset_proxy",
        dead_release_pairs,
        "release dead double",
        "spend live weapon or protector",
    )
    add_contrast(
        contrast_rows,
        examples,
        "throw_lower_target_double_vs_higher_target_double_proxy",
        lower_target_double_pairs,
        "throw lower-live-target double",
        "throw higher-live-target double",
    )

    trump_pairs = []
    for rows in by_decision.values():
        offense = [row for row in rows if row["team"] == "offense"]
        trumps = [row for row in offense if "bidder_trump_pull_candidate" in row["dynamic_84_labels"].split("|")]
        offs = [row for row in offense if "bidder_nontrump_final_off_candidate" in row["dynamic_84_labels"].split("|")]
        if trumps and offs:
            trump_pairs.append((max(trumps, key=lambda row: row["mean"]), max(offs, key=lambda row: row["mean"])))
    add_contrast(
        contrast_rows,
        examples,
        "offense_trump_pull_vs_final_off_proxy",
        trump_pairs,
        "best trump pull",
        "best final-off candidate",
    )

    write_csv(OUT_DIR / "selected_seed_rows.csv", metadata)
    write_csv(OUT_DIR / "dynamic_84_action_labels.csv", labeled_rows)
    write_csv(OUT_DIR / "dynamic_84_label_metrics.csv", label_metrics)
    write_csv(OUT_DIR / "dynamic_84_paired_contrasts.csv", contrast_rows)
    write_json(OUT_DIR / "examples.json", examples[:32])
    blockers = write_blocker_table()
    score_gate_rows = write_score_gate_table(metadata)

    atlas_summary = json.loads((ATLAS_DIR / "summary.json").read_text(encoding="utf-8"))
    surface_counts = Counter(surface for meta in metadata for surface in meta["surfaces"])
    summary = {
        "schema_version": "w42.phase4_84_dynamic_seed_tests.v0",
        "owner_bead": BEAD_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "commit_sha": git_sha(),
        "generation": generation_info,
        "coverage": {
            "seed_games": len(metadata),
            "decision_rows": atlas_summary["rows"]["decision_rows"],
            "action_rows": atlas_summary["rows"]["action_rows"],
            "hidden_threat_rows": atlas_summary["rows"]["hidden_threat_rows"],
            "dynamic_labeled_actions": len(labeled_rows),
            "dynamic_labels": len(label_names),
            "paired_contrasts": len(contrast_rows),
            "score_gate_rows": len(score_gate_rows),
            "blocker_rows": len(blockers),
        },
        "surface_counts_in_selected_games": dict(sorted(surface_counts.items())),
        "headline": {
            "defender_live_double_weapon_proxy_actions": count_label(labeled_rows, "defender_live_double_weapon_proxy"),
            "defender_live_same_suit_pair_proxy_actions": count_label(labeled_rows, "defender_live_same_suit_pair_proxy"),
            "pair_protector_proxy_actions": count_label(labeled_rows, "pair_protector_proxy"),
            "dead_asset_release_candidate_proxy_actions": count_label(labeled_rows, "dead_asset_release_candidate_proxy"),
            "defense_preserve_pairs": len(preserve_pairs),
            "dead_release_pairs": len(dead_release_pairs),
            "lower_target_double_choice_pairs": len(lower_target_double_pairs),
            "trump_vs_final_off_pairs": len(trump_pairs),
        },
        "scientific_status": {
            "claim_ledger_impact": "none by this worker; outputs are a scoped phase-4 artifact for later synthesis",
            "leakage_boundary": "full-deal defender assets are used only as offline eval labels; live features remain public state plus actor hand",
            "blockers": [
                "no arbitrary late-state injector in this runner",
                "reached decisions are conditioned on the current greedy E[Q] policy trace",
                "preserve/spend labels are Chapter 8 proxy labels, not final set-cause proof",
                "score 42-vs-84 gate is static here because bidding/terminal match counterfactuals are outside the play generator",
            ],
        },
        "artifacts": {
            "eq_games": str(EQ_PATH.relative_to(ROOT)),
            "branch_atlas": str(ATLAS_DIR.relative_to(ROOT)),
            "selected_seed_rows": "w42/phase4_84_dynamic_seed_tests/selected_seed_rows.csv",
            "dynamic_action_labels": "w42/phase4_84_dynamic_seed_tests/dynamic_84_action_labels.csv",
            "label_metrics": "w42/phase4_84_dynamic_seed_tests/dynamic_84_label_metrics.csv",
            "paired_contrasts": "w42/phase4_84_dynamic_seed_tests/dynamic_84_paired_contrasts.csv",
            "score_gate_static_table": "w42/phase4_84_dynamic_seed_tests/score_gate_static_table.csv",
            "blocker_table": "w42/phase4_84_dynamic_seed_tests/blocker_table.csv",
            "examples": "w42/phase4_84_dynamic_seed_tests/examples.json",
        },
    }
    write_json(OUT_DIR / "summary.json", summary)
    write_json(
        OUT_DIR / "manifest.json",
        {
            "schema_version": "w42.artifact_manifest.v1",
            "owner_bead": BEAD_ID,
            "created_at_utc": summary["created_at_utc"],
            "dataset_id": "w42-phase4-84-dynamic-mined-seed-tests",
            "input_sha256": sha256_file(EQ_PATH),
            "source_inputs": [
                "wiki/AGENTS.md",
                "wiki/experiments/winning42-ch07-taking-every-trick-84.md",
                "wiki/experiments/winning42-ch08-setting-84.md",
                "w42/eighty_four_seed_mining/recommended_84_seed_rows.csv",
                "w42/eighty_four_seed_mining/candidate_84_seed_rows.csv",
                "w42/eighty_four_weapon_preservation_probe/run_dynamic_branch_lab.py",
                "w42/branch_atlas_v1/build_branch_atlas.py",
            ],
            "outputs": summary["artifacts"],
            "leakage_boundary": summary["scientific_status"]["leakage_boundary"],
        },
    )
    return summary


def count_label(rows: list[dict[str, Any]], label: str) -> int:
    return sum(1 for row in rows if label in row["dynamic_84_labels"].split("|"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_seed_rows(args)
    if not rows:
        raise RuntimeError("No mined seed rows selected")
    metadata = selected_metadata(rows)
    generation_info = run_generation(args, rows, metadata)
    run_branch_atlas(args)
    summary = analyze_outputs(generation_info, metadata)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--recommended-rows", type=Path, default=SEED_DIR / "recommended_84_seed_rows.csv")
    parser.add_argument("--candidate-rows", type=Path, default=SEED_DIR / "candidate_84_seed_rows.csv")
    parser.add_argument("--max-games", type=int, default=18)
    parser.add_argument("--rare-surface-games", type=int, default=4)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-name", default="t42-br7n.2-phase4-84-dynamic-seed-tests")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir != OUT_DIR:
        raise ValueError("This worker is scoped to w42/phase4_84_dynamic_seed_tests only")
    summary = run(args)
    print(json.dumps(summary["coverage"], indent=2, sort_keys=True))
    print(json.dumps(summary["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
