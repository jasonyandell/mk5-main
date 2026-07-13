#!/usr/bin/env python3
"""Analyze hidden-domino threat attribution from saved joint-world E[Q] artifacts.

The input is a ``torch.save`` payload produced by ``python -m forge.eq.generate
--save-joint-worlds``. Each decision record must contain:

- ``world_hands``: [M, 3, 7] opponent hands, relative to the acting player.
- ``q_per_world``: [M, 7] oracle Q values for each action slot.

The script never treats hidden ownership as an online feature. It produces
offline labels and evaluation rows that ask which hidden holders explain E[Q]
distribution shelves or tails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle import schema


DEFAULT_THRESHOLDS = (-18.0, 18.0)


def domino_label(domino_id: int) -> str:
    high, low = schema.domino_pips(int(domino_id))
    return f"{high}-{low}"


def finite_float(value: Any) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def tensor_to_list(tensor: Any) -> list[Any]:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().tolist()
    return list(tensor)


def ownership_mask(world_hands: Any, rel_holder: int, domino_id: int) -> torch.Tensor:
    hands = world_hands[:, rel_holder, :]
    return (hands == domino_id).any(dim=1)


def describe_q(q_values: torch.Tensor, low_threshold: float, high_threshold: float) -> dict[str, float]:
    if q_values.numel() == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "tail_low_mass": float("nan"),
            "shelf_high_mass": float("nan"),
            "std": float("nan"),
        }

    q_float = q_values.float()
    return {
        "n": int(q_float.numel()),
        "mean": float(q_float.mean().item()),
        "tail_low_mass": float((q_float <= low_threshold).float().mean().item()),
        "shelf_high_mass": float((q_float >= high_threshold).float().mean().item()),
        "std": float(q_float.std(unbiased=False).item()),
    }


def iter_threat_rows(
    payload: dict[str, Any],
    *,
    low_threshold: float,
    high_threshold: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    games = payload.get("results", [])
    seeds = payload.get("seeds", [])
    decl_ids = payload.get("decl_ids", [])

    inspected_decisions = 0
    decisions_with_joint_worlds = 0
    skipped_no_joint_worlds = 0
    skipped_no_legal_actions = 0
    sample_counts: Counter[int] = Counter()

    for game_idx, game in enumerate(games):
        hands = getattr(game, "hands", None)
        game_decl_id = int(getattr(game, "decl_id", decl_ids[game_idx] if game_idx < len(decl_ids) else -1))
        seed = seeds[game_idx] if game_idx < len(seeds) else None

        for decision_idx, decision in enumerate(getattr(game, "decisions", [])):
            inspected_decisions += 1
            world_hands = getattr(decision, "world_hands", None)
            q_per_world = getattr(decision, "q_per_world", None)
            legal_mask = getattr(decision, "legal_mask", None)
            actor = int(getattr(decision, "player"))

            if world_hands is None or q_per_world is None:
                skipped_no_joint_worlds += 1
                continue

            world_hands = world_hands.detach().cpu().long()
            q_per_world = q_per_world.detach().cpu().float()
            legal = torch.tensor(tensor_to_list(legal_mask), dtype=torch.bool)
            legal_slots = [slot for slot, is_legal in enumerate(legal.tolist()) if is_legal]
            if not legal_slots:
                skipped_no_legal_actions += 1
                continue

            decisions_with_joint_worlds += 1
            sample_counts[int(world_hands.shape[0])] += 1

            visible_actor_hand = hands[actor] if hands is not None else None
            candidate_dominoes = sorted(
                {
                    int(domino)
                    for domino in world_hands.flatten().tolist()
                    if int(domino) >= 0
                }
            )

            for action_slot in legal_slots:
                all_stats = describe_q(q_per_world[:, action_slot], low_threshold, high_threshold)
                if all_stats["n"] == 0:
                    continue

                action_domino = None
                if visible_actor_hand is not None and 0 <= action_slot < len(visible_actor_hand):
                    action_domino = int(visible_actor_hand[action_slot])

                action_rows: list[dict[str, Any]] = []
                for domino_id in candidate_dominoes:
                    for rel_holder in range(3):
                        abs_holder = (actor + rel_holder + 1) % 4
                        mask = ownership_mask(world_hands, rel_holder, domino_id)
                        holder_stats = describe_q(
                            q_per_world[mask, action_slot], low_threshold, high_threshold
                        )
                        if holder_stats["n"] == 0:
                            continue

                        mean_delta = holder_stats["mean"] - all_stats["mean"]
                        tail_delta = holder_stats["tail_low_mass"] - all_stats["tail_low_mass"]
                        shelf_delta = holder_stats["shelf_high_mass"] - all_stats["shelf_high_mass"]
                        impact = abs(mean_delta) + 10.0 * (abs(tail_delta) + abs(shelf_delta))

                        action_rows.append(
                            {
                                "game_idx": game_idx,
                                "seed": seed,
                                "decl_id": game_decl_id,
                                "decision_idx": decision_idx,
                                "actor": actor,
                                "action_slot": action_slot,
                                "action_domino_id": action_domino,
                                "action_domino": domino_label(action_domino)
                                if action_domino is not None
                                else None,
                                "hidden_domino_id": domino_id,
                                "hidden_domino": domino_label(domino_id),
                                "relative_holder": rel_holder,
                                "absolute_holder": abs_holder,
                                "world_count": all_stats["n"],
                                "conditioned_world_count": holder_stats["n"],
                                "conditioned_mass": holder_stats["n"] / all_stats["n"],
                                "baseline_mean_q": all_stats["mean"],
                                "conditioned_mean_q": holder_stats["mean"],
                                "mean_q_delta": mean_delta,
                                "baseline_tail_low_mass": all_stats["tail_low_mass"],
                                "conditioned_tail_low_mass": holder_stats["tail_low_mass"],
                                "tail_low_mass_delta": tail_delta,
                                "baseline_shelf_high_mass": all_stats["shelf_high_mass"],
                                "conditioned_shelf_high_mass": holder_stats["shelf_high_mass"],
                                "shelf_high_mass_delta": shelf_delta,
                                "impact_score": impact,
                            }
                        )

                action_rows.sort(key=lambda row: row["impact_score"], reverse=True)
                rows.extend(action_rows[:top_k])

    summary = {
        "artifact_kind": "hidden_domino_threat_attribution",
        "source_path": str(payload.get("_source_path", "unknown")),
        "input_games": len(games),
        "inspected_decisions": inspected_decisions,
        "decisions_with_joint_worlds": decisions_with_joint_worlds,
        "skipped_no_joint_worlds": skipped_no_joint_worlds,
        "skipped_no_legal_actions": skipped_no_legal_actions,
        "row_count": len(rows),
        "low_threshold_q": low_threshold,
        "high_threshold_q": high_threshold,
        "top_k_per_action": top_k,
        "sample_counts_by_decision": dict(sorted(sample_counts.items())),
        "empirical_status": "computed" if rows else "no_rows",
        "leakage_boundary": "hidden ownership is offline label/eval target only; not a live feature",
    }
    return rows, summary


def write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Saved .pt artifact from forge.eq.generate")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--low-threshold", type=float, default=DEFAULT_THRESHOLDS[0])
    parser.add_argument("--high-threshold", type=float, default=DEFAULT_THRESHOLDS[1])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.input, weights_only=False, map_location="cpu")
    payload["_source_path"] = str(args.input)

    rows, summary = iter_threat_rows(
        payload,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        top_k=args.top_k,
    )

    write_rows_jsonl(args.output_dir / "threat_rows.jsonl", rows)
    write_rows_csv(args.output_dir / "threat_rows.csv", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
