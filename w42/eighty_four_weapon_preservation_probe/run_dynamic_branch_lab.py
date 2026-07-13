#!/usr/bin/env python3
"""Run targeted 84 branch-atlas fixtures for stopper/preservation claims.

This is the dynamic follow-up to ``build_probe.py``.  It uses explicit deals,
not seed search, so the first 84 run can exercise known laydown, straight-off,
protected-off, and defender-pair surfaces directly.
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


OUT = ROOT / "w42" / "eighty_four_weapon_preservation_probe"
DYNAMIC_OUT = OUT / "dynamic_branch_lab"
EQ_PATH = DYNAMIC_OUT / "targeted_84_games.pt"
ATLAS_DIR = DYNAMIC_OUT / "branch_atlas"
BEAD_ID = "t42-0b4l.7"


def domino_id(label: str) -> int:
    high_s, low_s = label.split("-")
    high = int(high_s)
    low = int(low_s)
    if low > high:
        high, low = low, high
    return tables.DOMINOES.index((high, low))


def ids(labels: list[str]) -> list[int]:
    return [domino_id(label) for label in labels]


def labels(dominoes: list[int]) -> list[str]:
    return [domino_label(domino) for domino in dominoes]


def fixture_deals() -> list[dict[str, Any]]:
    """Construct hand-picked 84 deals.

    Player 0 is the bidder because the E[Q] generator currently initializes
    bidder=P0.  Declarations are pip-trump ids, and every fixture uses
    ``bid_value=84`` so threshold mass is the take-all contract threshold.
    """

    fixtures = [
        {
            "fixture_id": "six_trump_laydown_control",
            "seed": 840700,
            "decl_id": 6,
            "claim_surface": "laydown_84_proof",
            "hypothesis": "Bidder owns all seven sixes; this should behave as a true laydown control.",
            "hands": [
                ids(["6-6", "6-5", "6-4", "6-3", "6-2", "6-1", "6-0"]),
                ids(["5-5", "5-4", "5-3", "5-2", "5-1", "5-0", "4-4"]),
                ids(["4-3", "4-2", "4-1", "4-0", "3-3", "3-2", "3-1"]),
                ids(["3-0", "2-2", "2-1", "2-0", "1-1", "1-0", "0-0"]),
            ],
        },
        {
            "fixture_id": "protected_one_off_43",
            "seed": 840701,
            "decl_id": 6,
            "claim_surface": "protected_one_off_84_shape",
            "hypothesis": "Bidder has four high trumps plus 4-4 protecting a 4-3 final off.",
            "hands": [
                ids(["6-6", "6-5", "6-4", "6-3", "5-5", "4-4", "4-3"]),
                ids(["6-2", "6-1", "6-0", "5-4", "5-3", "5-2", "5-1"]),
                ids(["5-0", "4-2", "4-1", "4-0", "3-3", "3-2", "3-1"]),
                ids(["3-0", "2-2", "2-1", "2-0", "1-1", "1-0", "0-0"]),
            ],
        },
        {
            "fixture_id": "straight_off_named_44_threat",
            "seed": 840702,
            "decl_id": 6,
            "claim_surface": "straight_off_84_risk",
            "hypothesis": "Bidder has a straight 4-3 off; left setter owns the named 4-4 stopper.",
            "hands": [
                ids(["6-6", "6-5", "6-4", "6-3", "5-5", "3-3", "4-3"]),
                ids(["4-4", "6-2", "6-1", "6-0", "5-4", "5-3", "5-2"]),
                ids(["5-1", "5-0", "4-2", "4-1", "4-0", "3-2", "3-1"]),
                ids(["3-0", "2-2", "2-1", "2-0", "1-1", "1-0", "0-0"]),
            ],
        },
        {
            "fixture_id": "two_off_same_suit_ordering",
            "seed": 840703,
            "decl_id": 6,
            "claim_surface": "two_off_same_suit_84",
            "hypothesis": "Bidder has two fours offs, making final-off ordering and same-suit pressure visible.",
            "hands": [
                ids(["6-6", "6-5", "6-4", "5-5", "4-4", "4-3", "4-2"]),
                ids(["6-3", "6-2", "6-1", "6-0", "5-4", "5-3", "5-2"]),
                ids(["5-1", "5-0", "4-1", "4-0", "3-3", "3-2", "3-1"]),
                ids(["3-0", "2-2", "2-1", "2-0", "1-1", "1-0", "0-0"]),
            ],
        },
        {
            "fixture_id": "defender_pair_protector_pressure",
            "seed": 840704,
            "decl_id": 6,
            "claim_surface": "live_same_suit_pair_and_protector",
            "hypothesis": "Left setter starts with a fours pair/protector against bidder's double-ahead 4-1 plan.",
            "hands": [
                ids(["6-6", "6-5", "6-4", "6-3", "5-5", "4-4", "4-1"]),
                ids(["4-3", "4-2", "4-0", "5-4", "5-3", "3-3", "3-2"]),
                ids(["6-2", "6-1", "6-0", "5-2", "5-1", "5-0", "3-1"]),
                ids(["3-0", "2-2", "2-1", "2-0", "1-1", "1-0", "0-0"]),
            ],
        },
        {
            "fixture_id": "dead_asset_release_control",
            "seed": 840705,
            "decl_id": 6,
            "claim_surface": "dead_asset_abandonment",
            "hypothesis": "Left setter owns doubles that look weapon-like but do not match the bidder's apparent final off.",
            "hands": [
                ids(["6-6", "6-5", "6-4", "6-3", "5-5", "4-4", "4-3"]),
                ids(["3-3", "2-2", "1-1", "5-4", "5-3", "5-2", "5-1"]),
                ids(["6-2", "6-1", "6-0", "5-0", "4-2", "4-1", "4-0"]),
                ids(["3-2", "3-1", "3-0", "2-1", "2-0", "1-0", "0-0"]),
            ],
        },
    ]
    for fixture in fixtures:
        flat = [domino for hand in fixture["hands"] for domino in hand]
        if sorted(flat) != list(range(28)):
            raise ValueError(f"{fixture['fixture_id']} is not a complete unique double-six deal")
        fixture["bid_value"] = 84
        fixture["hand_labels"] = [labels(hand) for hand in fixture["hands"]]
    return fixtures


def resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_generation(args: argparse.Namespace, fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    checkpoint = detect_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    print(f"Loading model from {checkpoint} on {device}...", flush=True)
    oracle = Stage1Oracle(checkpoint, device=device, compile=False)

    hands = [fixture["hands"] for fixture in fixtures]
    decl_ids = [int(fixture["decl_id"]) for fixture in fixtures]
    bid_values = [84 for _fixture in fixtures]

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
        "seeds": [int(fixture["seed"]) for fixture in fixtures],
        "decl_ids": decl_ids,
        "bid_values": bid_values,
        "schema": "v2",
        "n_samples": args.samples,
        "checkpoint": checkpoint,
        "fixture_metadata": [
            {key: value for key, value in fixture.items() if key != "hands"}
            for fixture in fixtures
        ],
        "bead_id": BEAD_ID,
        "experiment": "w42-eighty-four-dynamic-branch-lab",
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
        "w42-eighty-four-dynamic-branch-lab",
        "--top-k",
        str(args.top_k),
        "--examples",
        "16",
        "--log-every-decisions",
        "2",
        "--wandb-mode",
        args.wandb_mode,
        "--wandb-group",
        "w42-eighty-four-dynamic-branch-lab",
        "--wandb-name",
        args.wandb_name or "t42-0b4l.7-dynamic-84-branch-lab",
    ]
    if args.no_wandb:
        cmd.append("--no-wandb")
    print("Running branch atlas...", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


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
            actual_slot = int(getattr(decision, "action_taken"))
            played_slots[actor].add(actual_slot)
    return by_decision


def pip_rank(domino: int) -> int:
    return 14 if tables.DOMINO_IS_DOUBLE[domino] else tables.DOMINO_SUM[domino]


def side_pips(domino: int, decl_id: int) -> set[int]:
    if tables.is_in_called_suit(domino, decl_id):
        return set()
    return {int(tables.DOMINO_HIGH[domino]), int(tables.DOMINO_LOW[domino])}


def beats_nontrump_in_same_suit(candidate: int, target: int, decl_id: int) -> bool:
    shared = side_pips(candidate, decl_id) & side_pips(target, decl_id)
    if not shared:
        return False
    return pip_rank(candidate) > pip_rank(target)


def candidate_asset_labels(row: dict[str, str], context: dict[str, Any]) -> tuple[list[str], int]:
    labels_out: list[str] = ["is_84_contract_decision"]
    game_idx = int(row["game_idx"])
    fixture = FIXTURES_BY_IDX[game_idx]
    candidate = int(row["candidate_domino_id"])
    decl_id = int(row["decl_id"])
    actor = int(row["actor"])
    team = row["team"]
    remaining = context["remaining"]
    bidder_remaining = context["bidder_remaining"]
    candidate_called = bool(int_or_bool(row["candidate_is_called_suit"]))
    candidate_double = bool(int_or_bool(row["candidate_is_double"]))

    if fixture["fixture_id"] == "six_trump_laydown_control" and team == "offense":
        labels_out.append("laydown_84_proof_fixture")

    if team == "offense" and not candidate_called:
        labels_out.append("bidder_final_off_candidate")
        if not candidate_double:
            labels_out.append("bidder_nontrump_final_off_candidate")

    if team == "offense" and candidate_called:
        labels_out.append("bidder_trump_pull_candidate")

    asset_priority = 0
    if team == "defense":
        bidder_offs = [
            domino
            for domino in bidder_remaining
            if not tables.is_in_called_suit(domino, decl_id)
        ]
        if candidate_double and any(beats_nontrump_in_same_suit(candidate, off, decl_id) for off in bidder_offs):
            labels_out.append("defender_live_double_weapon_proxy")
            asset_priority = max(asset_priority, 5)

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
            labels_out.append("defender_live_same_suit_pair_proxy")
            asset_priority = max(asset_priority, 4)

        if asset_priority == 0:
            pair_suits: Counter[int] = Counter()
            for left_idx, left in enumerate(remaining):
                for right in remaining[left_idx + 1 :]:
                    shared = side_pips(left, decl_id) & side_pips(right, decl_id)
                    if not shared:
                        continue
                    if any(
                        beats_nontrump_in_same_suit(left, off, decl_id)
                        and beats_nontrump_in_same_suit(right, off, decl_id)
                        for off in bidder_offs
                    ):
                        for pip in shared:
                            pair_suits[pip] += 1
            if side_pips(candidate, decl_id) & set(pair_suits):
                labels_out.append("pair_protector_proxy")
                asset_priority = max(asset_priority, 3)

    if fixture["claim_surface"] == "dead_asset_abandonment" and team == "defense" and candidate_double and asset_priority == 0:
        labels_out.append("dead_asset_release_candidate_proxy")

    if actor != 0 and team == "defense" and int(row["trick_position"]) > 0 and candidate_asset_priority(row, asset_priority) > 0:
        labels_out.append("84_defender_asset_follow_pressure")

    return sorted(set(labels_out)), asset_priority


def int_or_bool(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no", ""}:
        return 0
    return int(float(text))


def candidate_asset_priority(row: dict[str, str], priority: int) -> int:
    if row["team"] != "defense":
        return 0
    return priority


def summarize_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "action_n": len(rows),
        "decision_n": len({row["decision_key"] for row in rows}),
        "mean": mean([row["mean"] for row in rows]),
        "mean_threshold_mass": mean([row["threshold_mass"] for row in rows]),
        "mean_lower_tail_mass": mean([row["lower_tail_mass"] for row in rows]),
        "actual_action_rate": mean([1.0 if row["is_actual_action"] else 0.0 for row in rows]),
        "top_mean_rate": mean([1.0 if row["is_top_mean_action"] else 0.0 for row in rows]),
        "top_threshold_rate": mean([1.0 if row["is_top_threshold_action"] else 0.0 for row in rows]),
    }


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def paired_delta(pref: dict[str, Any], alt: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_delta": pref["mean"] - alt["mean"],
        "threshold_mass_delta": pref["threshold_mass"] - alt["threshold_mass"],
        "lower_tail_mass_delta": pref["lower_tail_mass"] - alt["lower_tail_mass"],
    }


def analyze_dynamic_labels(generation_info: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(EQ_PATH, weights_only=False, map_location="cpu")
    remaining_by_decision = reconstruct_remaining_by_decision(payload)
    raw_rows = read_csv(ATLAS_DIR / "decision_actions.csv")
    labeled_rows: list[dict[str, Any]] = []

    by_decision: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key_tuple = (int(row["game_idx"]), int(row["decision_idx"]))
        labels_out, asset_priority = candidate_asset_labels(row, remaining_by_decision[key_tuple])
        decision_key = f"{row['game_idx']}:{row['decision_idx']}"
        out = {
            "fixture_id": FIXTURES_BY_IDX[int(row["game_idx"])]["fixture_id"],
            "claim_surface": FIXTURES_BY_IDX[int(row["game_idx"])]["claim_surface"],
            "decision_key": decision_key,
            "game_idx": int(row["game_idx"]),
            "seed": int(row["seed"]),
            "decl_id": int(row["decl_id"]),
            "bid_value": int(row["bid_value"]),
            "decision_idx": int(row["decision_idx"]),
            "trick_idx": int(row["trick_idx"]),
            "trick_position": int(row["trick_position"]),
            "actor": int(row["actor"]),
            "team": row["team"],
            "seat_role": row["seat_role"],
            "candidate_domino": row["candidate_domino"],
            "actual_action_domino": row["actual_action_domino"],
            "asset_priority": asset_priority,
            "dynamic_84_labels": "|".join(labels_out),
            "mean": numeric(row["mean"]),
            "threshold_mass": numeric(row["threshold_mass"]),
            "lower_tail_mass": numeric(row["lower_tail_mass_le_neg18"]),
            "top_hidden_impact_score": numeric(row["top_hidden_impact_score"]),
            "top_hidden_downside_score": numeric(row["top_hidden_downside_score"]),
            "is_actual_action": bool(int_or_bool(row["is_actual_action"])),
            "is_top_mean_action": bool(int_or_bool(row["is_top_mean_action"])),
            "is_top_threshold_action": bool(int_or_bool(row["is_top_threshold_action"])),
        }
        labeled_rows.append(out)
        by_decision[decision_key].append(out)

    label_metrics: list[dict[str, Any]] = []
    label_names = sorted(
        {
            label
            for row in labeled_rows
            for label in str(row["dynamic_84_labels"]).split("|")
            if label
        }
    )
    for label in label_names:
        selected = [row for row in labeled_rows if label in row["dynamic_84_labels"].split("|")]
        metric = {"dynamic_84_label": label}
        metric.update(summarize_values(selected))
        label_metrics.append(metric)

    contrast_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    preserve_pairs = []
    for decision_key, rows in by_decision.items():
        defense_rows = [row for row in rows if row["team"] == "defense"]
        if not defense_rows:
            continue
        max_priority = max(row["asset_priority"] for row in defense_rows)
        min_priority = min(row["asset_priority"] for row in defense_rows)
        if max_priority <= 0 or min_priority >= max_priority:
            continue
        preserve = max([row for row in defense_rows if row["asset_priority"] == min_priority], key=lambda row: row["mean"])
        spend = max([row for row in defense_rows if row["asset_priority"] == max_priority], key=lambda row: row["mean"])
        preserve_pairs.append((preserve, spend))

    if preserve_pairs:
        add_contrast(
            contrast_rows,
            examples,
            "defense_preserve_low_asset_vs_spend_live_asset_proxy",
            preserve_pairs,
        )

    trump_pairs = []
    for rows in by_decision.values():
        offense_rows = [row for row in rows if row["team"] == "offense"]
        trumps = [row for row in offense_rows if "bidder_trump_pull_candidate" in row["dynamic_84_labels"].split("|")]
        offs = [row for row in offense_rows if "bidder_nontrump_final_off_candidate" in row["dynamic_84_labels"].split("|")]
        if trumps and offs:
            trump_pairs.append((max(trumps, key=lambda row: row["mean"]), max(offs, key=lambda row: row["mean"])))
    if trump_pairs:
        add_contrast(contrast_rows, examples, "offense_trump_pull_vs_final_off_proxy", trump_pairs)

    write_csv(
        DYNAMIC_OUT / "dynamic_84_action_labels.csv",
        labeled_rows,
        [
            "fixture_id",
            "claim_surface",
            "decision_key",
            "game_idx",
            "seed",
            "decl_id",
            "bid_value",
            "decision_idx",
            "trick_idx",
            "trick_position",
            "actor",
            "team",
            "seat_role",
            "candidate_domino",
            "actual_action_domino",
            "asset_priority",
            "dynamic_84_labels",
            "mean",
            "threshold_mass",
            "lower_tail_mass",
            "top_hidden_impact_score",
            "top_hidden_downside_score",
            "is_actual_action",
            "is_top_mean_action",
            "is_top_threshold_action",
        ],
    )
    write_csv(DYNAMIC_OUT / "dynamic_84_label_metrics.csv", label_metrics)
    write_csv(DYNAMIC_OUT / "dynamic_84_paired_contrasts.csv", contrast_rows)
    write_json(DYNAMIC_OUT / "dynamic_84_examples.json", examples[:24])

    atlas_summary = json.loads((ATLAS_DIR / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": "w42.eighty_four_dynamic_branch_lab.v0",
        "owner_bead": BEAD_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "commit_sha": git_sha(),
        "generation": generation_info,
        "coverage": {
            "fixture_games": len(FIXTURES),
            "decision_rows": atlas_summary["rows"]["decision_rows"],
            "action_rows": atlas_summary["rows"]["action_rows"],
            "hidden_threat_rows": atlas_summary["rows"]["hidden_threat_rows"],
            "dynamic_labeled_actions": len(labeled_rows),
            "dynamic_labels": len(label_names),
        },
        "fixture_ids": [fixture["fixture_id"] for fixture in FIXTURES],
        "claim_surfaces": dict(Counter(fixture["claim_surface"] for fixture in FIXTURES)),
        "headline": {
            "laydown_control_actions": count_label(labeled_rows, "laydown_84_proof_fixture"),
            "defender_live_double_weapon_proxy_actions": count_label(
                labeled_rows, "defender_live_double_weapon_proxy"
            ),
            "defender_live_same_suit_pair_proxy_actions": count_label(
                labeled_rows, "defender_live_same_suit_pair_proxy"
            ),
            "pair_protector_proxy_actions": count_label(labeled_rows, "pair_protector_proxy"),
            "dead_asset_release_candidate_proxy_actions": count_label(
                labeled_rows, "dead_asset_release_candidate_proxy"
            ),
            "preserve_vs_spend_paired_decisions": len(preserve_pairs),
            "trump_vs_final_off_paired_decisions": len(trump_pairs),
        },
        "scientific_status": {
            "claim_ledger_impact": "no central claim-ledger status movement; targeted fixtures are dynamic examples, not powered corpus evidence",
            "leakage_boundary": "asset labels reconstructed from full generated deals are offline diagnostics; live model features must use public state/actor hand only",
            "blockers": [
                "generated fixtures are hand-picked and small",
                "policy trace determines reached decisions; unreached endgame states still need state injection or seed mining",
                "preserve/spend labels are proxy asset-priority labels, not full proof of final-trick set attribution",
            ],
        },
        "artifacts": {
            "eq_games": str(EQ_PATH.relative_to(ROOT)),
            "branch_atlas": str(ATLAS_DIR.relative_to(ROOT)),
            "dynamic_action_labels": "w42/eighty_four_weapon_preservation_probe/dynamic_branch_lab/dynamic_84_action_labels.csv",
            "label_metrics": "w42/eighty_four_weapon_preservation_probe/dynamic_branch_lab/dynamic_84_label_metrics.csv",
            "paired_contrasts": "w42/eighty_four_weapon_preservation_probe/dynamic_branch_lab/dynamic_84_paired_contrasts.csv",
            "examples": "w42/eighty_four_weapon_preservation_probe/dynamic_branch_lab/dynamic_84_examples.json",
        },
    }
    write_json(DYNAMIC_OUT / "summary.json", summary)
    write_json(
        DYNAMIC_OUT / "manifest.json",
        {
            "schema_version": "w42.artifact_manifest.v1",
            "dataset_id": "w42-eighty-four-dynamic-branch-lab",
            "created_at": summary["created_at"],
            "owner_bead": BEAD_ID,
            "source_inputs": [
                "wiki/experiments/winning42-ch07-taking-every-trick-84.md",
                "wiki/experiments/winning42-ch08-setting-84.md",
                "w42/eighty_four_weapon_preservation_probe/build_probe.py",
                "w42/branch_atlas_v1/build_branch_atlas.py",
            ],
            "outputs": summary["artifacts"],
            "input_sha256": sha256_file(EQ_PATH),
            "exact_command": ".venv/bin/python w42/eighty_four_weapon_preservation_probe/run_dynamic_branch_lab.py --samples 256 --wandb-mode online",
            "leakage_boundary": summary["scientific_status"]["leakage_boundary"],
        },
    )
    return summary


def count_label(rows: list[dict[str, Any]], label: str) -> int:
    return sum(label in row["dynamic_84_labels"].split("|") for row in rows)


def add_contrast(
    contrast_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    contrast_id: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    deltas = [paired_delta(pref, alt) for pref, alt in pairs]
    contrast_rows.append(
        {
            "contrast_id": contrast_id,
            "paired_decision_n": len(pairs),
            "mean_delta": mean([delta["mean_delta"] for delta in deltas]),
            "threshold_mass_delta": mean([delta["threshold_mass_delta"] for delta in deltas]),
            "lower_tail_mass_delta": mean([delta["lower_tail_mass_delta"] for delta in deltas]),
        }
    )
    for pref, alt in sorted(pairs, key=lambda pair: abs(pair[0]["mean"] - pair[1]["mean"]), reverse=True)[:12]:
        examples.append(
            {
                "contrast_id": contrast_id,
                "fixture_id": pref["fixture_id"],
                "decision_key": pref["decision_key"],
                "trick_idx": pref["trick_idx"],
                "trick_position": pref["trick_position"],
                "actor": pref["actor"],
                "preferred_domino": pref["candidate_domino"],
                "preferred_labels": pref["dynamic_84_labels"],
                "preferred_mean": pref["mean"],
                "preferred_threshold_mass": pref["threshold_mass"],
                "alternative_domino": alt["candidate_domino"],
                "alternative_labels": alt["dynamic_84_labels"],
                "alternative_mean": alt["mean"],
                "alternative_threshold_mass": alt["threshold_mass"],
                **paired_delta(pref, alt),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-atlas", action="store_true")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


FIXTURES = fixture_deals()
FIXTURES_BY_IDX = {idx: fixture for idx, fixture in enumerate(FIXTURES)}


def main() -> int:
    args = parse_args()
    DYNAMIC_OUT.mkdir(parents=True, exist_ok=True)
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(DYNAMIC_OUT / "fixture_deals.json", FIXTURES)
    generation_info = {
        "checkpoint": "reused existing artifact" if args.skip_generation and EQ_PATH.exists() else None,
        "device": "unknown",
        "samples": args.samples,
        "elapsed_seconds": None,
        "games": len(FIXTURES),
    }
    if not args.skip_generation:
        generation_info = run_generation(args, FIXTURES)
    if not args.skip_atlas:
        run_branch_atlas(args)
    summary = analyze_dynamic_labels(generation_info)
    print(json.dumps(summary["headline"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
