#!/usr/bin/env python3
"""Mine legacy Gus joint-world chunks for hidden-threat mitigation evidence.

The branch-atlas pilot proved the row schema on a small generated slice. This
script reuses the much larger legacy Gus corpus, but emits compact aggregates
instead of a full action table. Hidden ownership and q_per_world remain offline
diagnostic labels only.
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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle import tables
from w42.branch_atlas_v1 import build_branch_atlas as atlas
from w42.gus_corpus_claim_deep_dive import analyze_gus_claims as gus
from w42.wandb_utils import add_wandb_args, init_wandb


OUT_DIR = Path("w42/hidden_threat_legacy_mining")
LOW_TAIL_Q = -18.0

GROUP_FIELDS = [
    "group_type",
    "group",
    "action_n",
    "actual_action_rate",
    "top_mean_rate",
    "top_threshold_rate",
    "safest_tail_rate",
    "scalar_omission_rate",
    "hidden_large_impact_rate",
    "mean",
    "std",
    "threshold_mass",
    "lower_tail_mass",
    "mean_gap_to_best",
    "threshold_gap_to_best",
    "lower_tail_gap_to_safest",
    "top_hidden_impact_score",
    "max_hidden_impact_score",
    "top_hidden_downside_score",
    "max_hidden_downside_score",
    "top_hidden_upside_score",
    "max_hidden_upside_score",
]

DRIVER_FIELDS = [
    "driver_kind",
    "decl_id",
    "decl_name",
    "seat_role",
    "holder_role",
    "hidden_domino",
    "action_n",
    "mean_score",
    "max_score",
    "mean_q_delta",
    "tail_low_mass_delta",
    "shelf_high_mass_delta",
    "conditioned_mass",
]

CONTRAST_FIELDS = [
    "contrast_id",
    "paired_decision_n",
    "mean_delta",
    "mean_delta_ci95_low",
    "mean_delta_ci95_high",
    "threshold_mass_delta",
    "threshold_mass_delta_ci95_low",
    "threshold_mass_delta_ci95_high",
    "lower_tail_mass_delta",
    "lower_tail_mass_delta_ci95_low",
    "lower_tail_mass_delta_ci95_high",
    "hidden_downside_score_delta",
    "hidden_downside_score_delta_ci95_low",
    "hidden_downside_score_delta_ci95_high",
    "hidden_impact_score_delta",
    "hidden_impact_score_delta_ci95_low",
    "hidden_impact_score_delta_ci95_high",
]

CLAIM_EVIDENCE_FIELDS = [
    "claim_family",
    "claim_ids",
    "evidence_group_type",
    "evidence_group",
    "action_n",
    "hidden_large_impact_rate",
    "scalar_omission_rate",
    "mean_top_hidden_impact_score",
    "mean_top_hidden_downside_score",
    "status_impact",
    "interpretation",
]

CLAIM_EVIDENCE_SPECS = [
    {
        "claim_family": "setter defense",
        "claim_ids": [
            "ch05-pounce-count-before-certainty",
            "ch05-extra-count-to-set",
            "ch05-count-calling-lead",
            "ch05-count-protection",
            "ch05-trump-rich-count-intervention",
            "ch12-setter-pounce-high-bid-off",
            "ch05-reckless-count-to-bidder",
        ],
        "evidence_group_type": "claim_detector",
        "evidence_group": "setter_count_before_certainty",
        "interpretation": "branch-impact evidence now attached to setter count-pressure/pounce surfaces",
    },
    {
        "claim_family": "partner support",
        "claim_ids": [
            "ch04-safe-partner-count-donation",
            "ch04-low-trump-trap-against-count-dump",
            "ch04-lead-capture-for-support",
            "ch04-effective-double-highest-remaining",
            "ch04-lead-away-from-count-damage",
            "ch04-avoid-disruptive-partner-trump-lead",
            "ch04-unsafe-partner-count-donation",
        ],
        "evidence_group_type": "claim_detector",
        "evidence_group": "partner_forcedness_and_safety",
        "interpretation": "branch-impact evidence now attached to partner count-donation/forcedness surfaces",
    },
    {
        "claim_family": "bidder opening / bid risk",
        "claim_ids": [
            "ch16-four-trump-boss-first-policy",
            "ch02-three-plus-trumps-good-start",
            "ch12-natural-bid-bucket-anomaly",
            "ch02-bid-only-enough",
        ],
        "evidence_group_type": "claim_detector",
        "evidence_group": "bidder_first_lead_plan",
        "interpretation": "branch-impact evidence now attached to opening-lead and future bid-risk tests; bid margin still missing",
    },
    {
        "claim_family": "doubles and no-trump",
        "claim_ids": [
            "ch09-doubles-trump-doubles-leave-native-suits",
            "ch09-doubles-trump-follow-suit-removal",
            "ch09-no-trump-doubles-remain-native-tops",
            "ch09-dual-suit-top-protection",
            "ch09-no-trump-separate-doubles-suit-variant",
            "ch09-doubles-trump-candidate-4plus",
            "ch09-no-trump-support-proxy",
            "ch09-regime-switch-proxy-nt-over-dt",
            "ch09-five-doubles-opponent-void-prior",
        ],
        "evidence_group_type": "claim_detector",
        "evidence_group": "doubles_regime_plan",
        "interpretation": "branch-impact evidence now attached to doubles-trump action surfaces",
    },
    {
        "claim_family": "doubles and no-trump",
        "claim_ids": [
            "ch09-doubles-trump-doubles-leave-native-suits",
            "ch09-doubles-trump-follow-suit-removal",
            "ch09-no-trump-doubles-remain-native-tops",
            "ch09-dual-suit-top-protection",
            "ch09-no-trump-separate-doubles-suit-variant",
            "ch09-doubles-trump-candidate-4plus",
            "ch09-no-trump-support-proxy",
            "ch09-regime-switch-proxy-nt-over-dt",
            "ch09-five-doubles-opponent-void-prior",
        ],
        "evidence_group_type": "claim_detector",
        "evidence_group": "no_trump_lead_control_and_support",
        "interpretation": "branch-impact evidence now attached to no-trump control/support surfaces",
    },
    {
        "claim_family": "84 / stopper",
        "claim_ids": [
            "ch07-84-contract-regime",
            "ch07-protected-one-off-84-shape",
            "ch07-straight-off-two-to-one-double",
            "ch07-score-42-vs-84-gate",
            "ch08-one-to-four-last-trick-weapons",
            "ch08-double-ahead-needs-same-suit-pair",
            "ch08-abandon-dead-assets",
            "ch08-throwaway-priority-ladder",
            "ch07-protected-one-off-shape-frequency",
            "ch08-either-matching-double-not-two-to-one",
        ],
        "evidence_group_type": "claim_detector",
        "evidence_group": "late_trick_threshold_closure",
        "interpretation": "generic late-hand branch-impact evidence available; true 84 contract labels still require generated regimes",
    },
    {
        "claim_family": "distribution-aware EV crosscut",
        "claim_ids": ["crosscut-distribution-aware-ev", "crosscut-hidden-belief-impact"],
        "evidence_group_type": "distribution_shape",
        "evidence_group": "scalar_ev_lying_by_omission",
        "interpretation": "direct evidence that scalar EV hides branch/tail tradeoffs",
    },
    {
        "claim_family": "distribution-aware EV crosscut",
        "claim_ids": ["crosscut-distribution-aware-ev", "crosscut-hidden-belief-impact"],
        "evidence_group_type": "distribution_shape",
        "evidence_group": "hidden_threat_large_impact",
        "interpretation": "direct evidence that hidden-holder impact is widespread across legal actions",
    },
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


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(base_seed + int(digest[:8], 16) % 10_000)


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def bootstrap(values: list[float], *, samples: int, seed: int) -> tuple[float, float, float]:
    return gus.bootstrap_mean_ci(values, samples=samples, seed=seed)


def bool_float(value: bool) -> float:
    return 1.0 if value else 0.0


def score_delta_ci(
    values: list[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    return bootstrap(values, samples=samples, seed=seed)


@dataclass
class RunningStats:
    n: int = 0
    sums: Counter[str] = field(default_factory=Counter)
    maxes: dict[str, float] = field(default_factory=dict)

    def add(self, row: dict[str, Any]) -> None:
        self.n += 1
        numeric = {
            "actual_action_rate": bool_float(row["is_actual_action"]),
            "top_mean_rate": bool_float(row["is_top_mean_action"]),
            "top_threshold_rate": bool_float(row["is_top_threshold_action"]),
            "safest_tail_rate": bool_float(row["is_safest_lower_tail_action"]),
            "scalar_omission_rate": bool_float(row["scalar_ev_lying_by_omission"]),
            "hidden_large_impact_rate": bool_float(float(row["top_hidden_impact_score"]) >= 5.0),
            "mean": float(row["mean"]),
            "std": float(row["std"]),
            "threshold_mass": float(row["threshold_mass"]),
            "lower_tail_mass": float(row["lower_tail_mass"]),
            "mean_gap_to_best": float(row["mean_gap_to_best"]),
            "threshold_gap_to_best": float(row["threshold_gap_to_best"]),
            "lower_tail_gap_to_safest": float(row["lower_tail_gap_to_safest"]),
            "top_hidden_impact_score": float(row["top_hidden_impact_score"]),
            "top_hidden_downside_score": float(row["top_hidden_downside_score"]),
            "top_hidden_upside_score": float(row["top_hidden_upside_score"]),
        }
        for key, value in numeric.items():
            self.sums[key] += value
        for key, value in {
            "max_hidden_impact_score": float(row["top_hidden_impact_score"]),
            "max_hidden_downside_score": float(row["top_hidden_downside_score"]),
            "max_hidden_upside_score": float(row["top_hidden_upside_score"]),
        }.items():
            self.maxes[key] = max(value, self.maxes.get(key, float("-inf")))

    def row(self, group_type: str, group: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "group_type": group_type,
            "group": group,
            "action_n": self.n,
        }
        for key in GROUP_FIELDS:
            if key in {"group_type", "group", "action_n"}:
                continue
            if key.startswith("max_"):
                out[key] = self.maxes.get(key, float("nan"))
            else:
                out[key] = self.sums[key] / self.n if self.n else float("nan")
        return out


@dataclass
class DriverStats:
    n: int = 0
    score_sum: float = 0.0
    max_score: float = float("-inf")
    mean_delta_sum: float = 0.0
    tail_delta_sum: float = 0.0
    shelf_delta_sum: float = 0.0
    mass_sum: float = 0.0

    def add(self, driver: dict[str, Any], score_key: str) -> None:
        score = float(driver[score_key])
        self.n += 1
        self.score_sum += score
        self.max_score = max(self.max_score, score)
        self.mean_delta_sum += float(driver["mean_q_delta"])
        self.tail_delta_sum += float(driver["tail_low_mass_delta"])
        self.shelf_delta_sum += float(driver["shelf_high_mass_delta"])
        self.mass_sum += float(driver["conditioned_mass"])

    def row(self, key: tuple[str, int, str, str, str, str]) -> dict[str, Any]:
        driver_kind, decl_id, decl_name, seat_role, holder_role, hidden_domino = key
        return {
            "driver_kind": driver_kind,
            "decl_id": decl_id,
            "decl_name": decl_name,
            "seat_role": seat_role,
            "holder_role": holder_role,
            "hidden_domino": hidden_domino,
            "action_n": self.n,
            "mean_score": self.score_sum / self.n if self.n else float("nan"),
            "max_score": self.max_score,
            "mean_q_delta": self.mean_delta_sum / self.n if self.n else float("nan"),
            "tail_low_mass_delta": self.tail_delta_sum / self.n if self.n else float("nan"),
            "shelf_high_mass_delta": self.shelf_delta_sum / self.n if self.n else float("nan"),
            "conditioned_mass": self.mass_sum / self.n if self.n else float("nan"),
        }


class ContrastStore:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def add(self, contrast_id: str, preferred: dict[str, Any], alternative: dict[str, Any]) -> None:
        bucket = self.values[contrast_id]
        bucket["mean_delta"].append(float(preferred["mean"]) - float(alternative["mean"]))
        bucket["threshold_mass_delta"].append(
            float(preferred["threshold_mass"]) - float(alternative["threshold_mass"])
        )
        bucket["lower_tail_mass_delta"].append(
            float(preferred["lower_tail_mass"]) - float(alternative["lower_tail_mass"])
        )
        bucket["hidden_downside_score_delta"].append(
            float(preferred["top_hidden_downside_score"]) - float(alternative["top_hidden_downside_score"])
        )
        bucket["hidden_impact_score_delta"].append(
            float(preferred["top_hidden_impact_score"]) - float(alternative["top_hidden_impact_score"])
        )

    def rows(self, *, bootstrap_samples: int, bootstrap_seed: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for contrast_id, metrics in sorted(self.values.items()):
            row: dict[str, Any] = {
                "contrast_id": contrast_id,
                "paired_decision_n": len(metrics["mean_delta"]),
            }
            for metric in [
                "mean_delta",
                "threshold_mass_delta",
                "lower_tail_mass_delta",
                "hidden_downside_score_delta",
                "hidden_impact_score_delta",
            ]:
                center, lo, hi = score_delta_ci(
                    metrics[metric],
                    samples=bootstrap_samples,
                    seed=stable_seed(bootstrap_seed, f"{contrast_id}:{metric}"),
                )
                row[metric] = center
                row[f"{metric}_ci95_low"] = lo
                row[f"{metric}_ci95_high"] = hi
            out.append(row)
        return out


def topk_insert(items: list[dict[str, Any]], item: dict[str, Any], *, key: str, limit: int) -> None:
    items.append(item)
    items.sort(key=lambda row: float(row[key]), reverse=True)
    del items[limit:]


def topk_abs_insert(items: list[dict[str, Any]], item: dict[str, Any], *, key: str, limit: int) -> None:
    items.append(item)
    items.sort(key=lambda row: abs(float(row[key])), reverse=True)
    del items[limit:]


def q_stats_matrix(q_values: torch.Tensor, threshold_q: float) -> dict[str, torch.Tensor]:
    q_values = q_values.detach().cpu().float()
    return {
        "mean": q_values.mean(dim=0),
        "std": q_values.std(dim=0, unbiased=False),
        "threshold_mass": (q_values >= threshold_q).float().mean(dim=0),
        "lower_tail_mass": (q_values <= LOW_TAIL_Q).float().mean(dim=0),
        "q10": torch.quantile(q_values, 0.10, dim=0),
    }


def hidden_summary_for_actions(
    *,
    world_hands: torch.Tensor,
    q_values: torch.Tensor,
    actor: int,
    decl_id: int,
    threshold_q: float,
    action_domino_ids: list[int],
) -> list[dict[str, Any]]:
    q_values = q_values.detach().cpu().float()
    world_hands = world_hands.detach().cpu().long()
    m, action_count = q_values.shape
    baseline_mean = q_values.mean(dim=0)
    baseline_tail = (q_values <= LOW_TAIL_Q).float().mean(dim=0)
    baseline_shelf = (q_values >= threshold_q).float().mean(dim=0)
    baseline_std = q_values.std(dim=0, unbiased=False)
    tail_ind = (q_values <= LOW_TAIL_Q).float()
    shelf_ind = (q_values >= threshold_q).float()
    q2 = q_values * q_values

    best: list[dict[str, Any]] = [
        {
            "impact": {"score": float("-inf")},
            "downside": {"score": float("-inf")},
            "upside": {"score": float("-inf")},
        }
        for _ in range(action_count)
    ]

    for rel_holder in range(3):
        holder_hands = world_hands[:, rel_holder, :].clamp(min=0, max=27)
        mask = torch.zeros((m, 28), dtype=torch.float32)
        mask.scatter_(1, holder_hands, 1.0)
        counts = mask.sum(dim=0)
        valid = counts > 0
        denom = counts.clamp(min=1.0).unsqueeze(1)

        cond_mean = mask.transpose(0, 1).matmul(q_values) / denom
        cond_tail = mask.transpose(0, 1).matmul(tail_ind) / denom
        cond_shelf = mask.transpose(0, 1).matmul(shelf_ind) / denom
        cond_q2 = mask.transpose(0, 1).matmul(q2) / denom
        cond_var = (cond_q2 - cond_mean * cond_mean).clamp(min=0.0)
        cond_std = cond_var.sqrt()

        mean_delta = cond_mean - baseline_mean.unsqueeze(0)
        tail_delta = cond_tail - baseline_tail.unsqueeze(0)
        shelf_delta = cond_shelf - baseline_shelf.unsqueeze(0)
        std_delta = cond_std - baseline_std.unsqueeze(0)
        impact = mean_delta.abs() + 10.0 * (tail_delta.abs() + shelf_delta.abs())
        downside = (-mean_delta).clamp(min=0.0) + 10.0 * tail_delta.clamp(min=0.0) + 5.0 * (
            -shelf_delta
        ).clamp(min=0.0)
        upside = mean_delta.clamp(min=0.0) + 10.0 * shelf_delta.clamp(min=0.0) + 5.0 * (
            -tail_delta
        ).clamp(min=0.0)
        for action_idx in range(action_count):
            for kind, scores in [("impact", impact), ("downside", downside), ("upside", upside)]:
                score_values = scores[:, action_idx].clone()
                score_values[~valid] = float("-inf")
                domino_id = int(torch.argmax(score_values).item())
                score = float(score_values[domino_id].item())
                if score > float(best[action_idx][kind]["score"]):
                    abs_holder = (actor + rel_holder + 1) % 4
                    best[action_idx][kind] = {
                        "score": score,
                        "hidden_domino_id": domino_id,
                        "hidden_domino": atlas.domino_label(domino_id),
                        "relative_holder": rel_holder,
                        "absolute_holder": abs_holder,
                        "holder_role": atlas.SEAT_ROLE[abs_holder],
                        "conditioned_mass": float(counts[domino_id].item() / max(m, 1)),
                        "baseline_mean_q": float(baseline_mean[action_idx].item()),
                        "mean_q_delta": float(mean_delta[domino_id, action_idx].item()),
                        "tail_low_mass_delta": float(tail_delta[domino_id, action_idx].item()),
                        "shelf_high_mass_delta": float(shelf_delta[domino_id, action_idx].item()),
                        "std_delta": float(std_delta[domino_id, action_idx].item()),
                    }

    out: list[dict[str, Any]] = []
    for action_idx, action_domino_id in enumerate(action_domino_ids):
        item = best[action_idx]
        item["action_domino_id"] = action_domino_id
        item["action_domino"] = atlas.domino_label(action_domino_id)
        out.append(item)
    return out


def row_groups(row: dict[str, Any]) -> list[tuple[str, str]]:
    groups = [
        ("all", "all"),
        ("decl", str(row["decl_id"])),
        ("decl_name", str(row["decl_name"])),
        ("seat_role", str(row["seat_role"])),
        ("team", str(row["team"])),
        ("trick_position", str(row["trick_position"])),
    ]
    for tag in str(row["strategy_context_tags"]).split("|"):
        if tag:
            groups.append(("strategy_context", tag))
    for detector in str(row["matched_position_detectors"]).split("|"):
        if detector:
            groups.append(("claim_detector", detector))
    for tag in str(row["distribution_shape_tags"]).split("|"):
        if tag:
            groups.append(("distribution_shape", tag))
    return groups


def build_claim_evidence_rows(group_stats: dict[tuple[str, str], RunningStats]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in CLAIM_EVIDENCE_SPECS:
        key = (spec["evidence_group_type"], spec["evidence_group"])
        stats = group_stats.get(key)
        if stats is None:
            continue
        group_row = stats.row(*key)
        rows.append(
            {
                "claim_family": spec["claim_family"],
                "claim_ids": "|".join(spec["claim_ids"]),
                "evidence_group_type": spec["evidence_group_type"],
                "evidence_group": spec["evidence_group"],
                "action_n": group_row["action_n"],
                "hidden_large_impact_rate": group_row["hidden_large_impact_rate"],
                "scalar_omission_rate": group_row["scalar_omission_rate"],
                "mean_top_hidden_impact_score": group_row["top_hidden_impact_score"],
                "mean_top_hidden_downside_score": group_row["top_hidden_downside_score"],
                "status_impact": "branch-impact evidence marker only; no central claim status movement",
                "interpretation": spec["interpretation"],
            }
        )
    return rows


def process_decision(
    *,
    source_file: str,
    game_idx: int,
    seed: int | None,
    decl_id: int,
    decision_idx: int,
    decision: Any,
    hands: list[list[int]],
    score: list[int],
    played_count_points: int,
    trick_plays: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    actor = int(getattr(decision, "player"))
    legal_mask = torch.as_tensor(getattr(decision, "legal_mask"), dtype=torch.bool).detach().cpu()
    legal_slots = [slot for slot, is_legal in enumerate(legal_mask.tolist()) if is_legal]
    q_per_world = getattr(decision, "q_per_world", None)
    world_hands = getattr(decision, "world_hands", None)
    if q_per_world is None or world_hands is None or not legal_slots:
        return []

    bid_value = getattr(decision, "bid_value", None)
    threshold_q = atlas.threshold_q_for_player(actor, bid_value)
    q_values = q_per_world.detach().cpu().float()[:, legal_slots]
    context = atlas.move_context(
        decision_idx=decision_idx,
        actor=actor,
        decl_id=decl_id,
        score=score,
        played_count_points=played_count_points,
        trick_plays=trick_plays,
    )
    stats = q_stats_matrix(q_values, threshold_q)
    action_domino_ids = [int(hands[actor][slot]) for slot in legal_slots]
    hidden = hidden_summary_for_actions(
        world_hands=world_hands,
        q_values=q_values,
        actor=actor,
        decl_id=decl_id,
        threshold_q=threshold_q,
        action_domino_ids=action_domino_ids,
    )
    best_mean_idx = int(torch.argmax(stats["mean"]).item())
    best_threshold_idx = int(torch.argmax(stats["threshold_mass"]).item())
    safest_tail_idx = int(torch.argmin(stats["lower_tail_mass"]).item())
    best_mean = float(stats["mean"][best_mean_idx].item())
    best_threshold = float(stats["threshold_mass"][best_threshold_idx].item())
    safest_tail = float(stats["lower_tail_mass"][safest_tail_idx].item())
    actual_slot = int(getattr(decision, "action_taken"))
    high_std_cutoff = float(torch.quantile(stats["std"], 0.75).item()) if len(legal_slots) > 1 else float("inf")

    rows: list[dict[str, Any]] = []
    for idx, slot in enumerate(legal_slots):
        candidate_domino = action_domino_ids[idx]
        facts = atlas.candidate_public_facts(candidate_domino, context, decl_id)
        base: dict[str, Any] = {
            **context,
            **facts,
            "source_file": source_file,
            "game_idx": game_idx,
            "seed": "" if seed is None else int(seed),
            "decl_id": decl_id,
            "decl_name": atlas.DECL_NAMES.get(decl_id, f"unknown-{decl_id}"),
            "bid_value": "" if bid_value is None else int(bid_value),
            "decision_idx": decision_idx,
            "candidate_slot": slot,
            "candidate_domino_id": candidate_domino,
            "candidate_domino": atlas.domino_label(candidate_domino),
            "mean": float(stats["mean"][idx].item()),
            "std": float(stats["std"][idx].item()),
            "threshold_mass": float(stats["threshold_mass"][idx].item()),
            "lower_tail_mass": float(stats["lower_tail_mass"][idx].item()),
            "q10": float(stats["q10"][idx].item()),
            "samples": int(q_values.shape[0]),
            "mean_gap_to_best": best_mean - float(stats["mean"][idx].item()),
            "threshold_gap_to_best": best_threshold - float(stats["threshold_mass"][idx].item()),
            "lower_tail_gap_to_safest": float(stats["lower_tail_mass"][idx].item()) - safest_tail,
            "is_actual_action": slot == actual_slot,
            "is_top_mean_action": idx == best_mean_idx,
            "is_top_threshold_action": idx == best_threshold_idx,
            "is_safest_lower_tail_action": idx == safest_tail_idx,
            "top_hidden_impact_score": float(hidden[idx]["impact"]["score"]),
            "top_hidden_downside_score": float(hidden[idx]["downside"]["score"]),
            "top_hidden_upside_score": float(hidden[idx]["upside"]["score"]),
            "top_hidden_impact": hidden[idx]["impact"],
            "top_hidden_downside": hidden[idx]["downside"],
            "top_hidden_upside": hidden[idx]["upside"],
            "hidden_threat_available": True,
        }
        has_choice = len(legal_slots) > 1
        base["high_variance_close_mean"] = has_choice and base["mean_gap_to_best"] <= 1.0 and base["std"] >= high_std_cutoff
        base["scalar_ev_lying_by_omission"] = has_choice and idx == best_mean_idx and (
            best_threshold_idx != best_mean_idx
            or safest_tail_idx != best_mean_idx
            or base["std"] >= 15.0
            or base["lower_tail_gap_to_safest"] >= 0.05
            or base["threshold_gap_to_best"] >= 0.03
        )
        strategy_tags, detectors, readiness, missing = atlas.strategy_tags(
            {
                **base,
                "branch_peak_count": 0,
                "shelf_gap": "",
                "lower_tail_mass_le_neg18": base["lower_tail_mass"],
                "top_hidden_impact_score": base["top_hidden_impact_score"],
            }
        )
        distribution_tags = []
        if base["scalar_ev_lying_by_omission"]:
            distribution_tags.append("scalar_ev_lying_by_omission")
        if base["high_variance_close_mean"]:
            distribution_tags.append("high_variance_close_mean")
        if base["std"] >= 15.0:
            distribution_tags.append("high_std")
        if base["lower_tail_mass"] >= 0.25:
            distribution_tags.append("large_lower_tail")
        if base["top_hidden_impact_score"] >= 5.0:
            distribution_tags.append("hidden_threat_large_impact")
        base["strategy_context_tags"] = "|".join(strategy_tags)
        base["matched_position_detectors"] = "|".join(detectors)
        base["direct_label_readiness"] = "|".join(readiness)
        base["missing_direct_label_fields"] = "|".join(missing)
        base["distribution_shape_tags"] = "|".join(sorted(distribution_tags))
        rows.append(base)

    return rows


def advance_actual(
    *,
    decision: Any,
    hands: list[list[int]],
    decl_id: int,
    score: list[int],
    played_count_points: int,
    trick_plays: list[tuple[int, int]],
    played_slots: dict[int, set[int]],
) -> tuple[int, list[tuple[int, int]]]:
    actor = int(getattr(decision, "player"))
    actual_slot = int(getattr(decision, "action_taken"))
    actual_domino = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1
    if actual_domino < 0:
        return played_count_points, trick_plays
    played_slots[actor].add(actual_slot)
    played_count_points += int(tables.DOMINO_COUNT_POINTS[actual_domino])
    trick_plays.append((actor, int(actual_domino)))
    if len(trick_plays) == 4:
        winner, _led_suit, _best_rank = atlas.current_winner(trick_plays, decl_id)
        if winner is not None:
            trick_points = 1 + sum(tables.DOMINO_COUNT_POINTS[d] for _p, d in trick_plays)
            score[atlas.team_id_for_player(winner)] += int(trick_points)
        trick_plays = []
    return played_count_points, trick_plays


def add_decision_contrasts(
    rows: list[dict[str, Any]],
    contrasts: ContrastStore,
    examples: dict[str, list[dict[str, Any]]],
    *,
    example_limit: int,
) -> None:
    if len(rows) <= 1:
        return
    top_mean = max(rows, key=lambda row: float(row["mean"]))
    top_threshold = max(rows, key=lambda row: float(row["threshold_mass"]))
    safest_tail = min(rows, key=lambda row: float(row["lower_tail_mass"]))
    max_downside = max(rows, key=lambda row: float(row["top_hidden_downside_score"]))
    close_rows = [row for row in rows if float(row["mean_gap_to_best"]) <= 2.0]
    min_downside_close = min(close_rows, key=lambda row: float(row["top_hidden_downside_score"])) if close_rows else None
    max_downside_close = max(close_rows, key=lambda row: float(row["top_hidden_downside_score"])) if close_rows else None
    actual = next((row for row in rows if row["is_actual_action"]), None)

    if top_threshold is not top_mean and top_mean["top_hidden_impact_score"] >= 5.0:
        contrasts.add("top_threshold_vs_top_mean_hidden_threat_decisions", top_threshold, top_mean)
        add_example(examples, "top_threshold_vs_top_mean_hidden_threat_decisions", top_threshold, top_mean, example_limit)
    if safest_tail is not top_mean and max_downside["top_hidden_downside_score"] >= 5.0:
        contrasts.add("safest_tail_vs_top_mean_hidden_downside_decisions", safest_tail, top_mean)
        add_example(examples, "safest_tail_vs_top_mean_hidden_downside_decisions", safest_tail, top_mean, example_limit)
    if (
        min_downside_close is not None
        and max_downside_close is not None
        and min_downside_close is not max_downside_close
        and max_downside_close["top_hidden_downside_score"] - min_downside_close["top_hidden_downside_score"] >= 5.0
    ):
        contrasts.add(
            "low_hidden_downside_vs_high_hidden_downside_close_mean",
            min_downside_close,
            max_downside_close,
        )
        add_example(
            examples,
            "low_hidden_downside_vs_high_hidden_downside_close_mean",
            min_downside_close,
            max_downside_close,
            example_limit,
        )
    if actual is not None and actual is not safest_tail and max_downside["top_hidden_downside_score"] >= 5.0:
        contrasts.add("actual_vs_safest_tail_hidden_downside_decisions", actual, safest_tail)
        add_example(examples, "actual_vs_safest_tail_hidden_downside_decisions", actual, safest_tail, example_limit)


def add_example(
    examples: dict[str, list[dict[str, Any]]],
    contrast_id: str,
    preferred: dict[str, Any],
    alternative: dict[str, Any],
    limit: int,
) -> None:
    item = {
        "contrast_id": contrast_id,
        "source_file": preferred["source_file"],
        "seed": preferred["seed"],
        "decl_id": preferred["decl_id"],
        "decl_name": preferred["decl_name"],
        "decision_idx": preferred["decision_idx"],
        "trick_idx": preferred["trick_idx"],
        "trick_position": preferred["trick_position"],
        "seat_role": preferred["seat_role"],
        "team": preferred["team"],
        "preferred_domino": preferred["candidate_domino"],
        "preferred_mean": preferred["mean"],
        "preferred_threshold_mass": preferred["threshold_mass"],
        "preferred_lower_tail_mass": preferred["lower_tail_mass"],
        "preferred_hidden_downside": preferred["top_hidden_downside_score"],
        "preferred_hidden_impact": preferred["top_hidden_impact_score"],
        "preferred_top_downside_driver": preferred["top_hidden_downside"],
        "alternative_domino": alternative["candidate_domino"],
        "alternative_mean": alternative["mean"],
        "alternative_threshold_mass": alternative["threshold_mass"],
        "alternative_lower_tail_mass": alternative["lower_tail_mass"],
        "alternative_hidden_downside": alternative["top_hidden_downside_score"],
        "alternative_hidden_impact": alternative["top_hidden_impact_score"],
        "alternative_top_downside_driver": alternative["top_hidden_downside"],
        "mean_delta": preferred["mean"] - alternative["mean"],
        "threshold_mass_delta": preferred["threshold_mass"] - alternative["threshold_mass"],
        "lower_tail_mass_delta": preferred["lower_tail_mass"] - alternative["lower_tail_mass"],
        "hidden_downside_score_delta": preferred["top_hidden_downside_score"] - alternative["top_hidden_downside_score"],
        "hidden_impact_score_delta": preferred["top_hidden_impact_score"] - alternative["top_hidden_impact_score"],
    }
    topk_abs_insert(examples[contrast_id], item, key="hidden_downside_score_delta", limit=limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=["gus/data/corpus_train_chunk_*-*.pt"])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260503)
    parser.add_argument("--log-every-files", type=int, default=5)
    parser.add_argument("--example-limit", type=int, default=16)
    add_wandb_args(
        parser,
        default_group="w42-hidden-threat-legacy-mining",
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
        "bead_id": "t42-0b4l.4",
        "experiment": "w42-hidden-threat-legacy-mining",
        "inputs": [str(path) for path in paths],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "large_row_policy": "streaming compact aggregates only; no full action table is emitted",
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=output_dir,
        tags=["w42", "winning42", "hidden-threat", "gus-legacy", "t42-0b4l.4"],
    )
    if getattr(wb, "run", None) is not None:
        wb.run.define_metric("progress/files_processed")
        for pattern in ("progress/*", "coverage/*", "threat/*", "branch/*"):
            wb.run.define_metric(pattern, step_metric="progress/files_processed")

    started = time.perf_counter()
    group_stats: dict[tuple[str, str], RunningStats] = defaultdict(RunningStats)
    driver_stats: dict[tuple[str, int, str, str, str, str], DriverStats] = defaultdict(DriverStats)
    contrasts = ContrastStore()
    contrast_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    top_impact_examples: list[dict[str, Any]] = []
    file_meta: list[dict[str, Any]] = []
    coverage = Counter()
    decl_counts = Counter()
    sample_counts = Counter()
    scalar_omission_decisions = 0
    hidden_downside_decisions = 0
    actual_not_safest_decisions = 0

    for file_idx, path in enumerate(paths, start=1):
        file_started = time.perf_counter()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        games = payload.get("results", [])
        seeds = payload.get("seeds", [])
        decl_ids = payload.get("decl_ids", [])
        file_actions = 0
        file_decisions = 0
        file_threat_actions = 0

        for game_idx, game in enumerate(games):
            hands = [[int(d) for d in hand] for hand in getattr(game, "hands")]
            seed = int(seeds[game_idx]) if game_idx < len(seeds) else None
            decl_id = int(getattr(game, "decl_id", decl_ids[game_idx] if game_idx < len(decl_ids) else -1))
            decl_counts[decl_id] += 1
            score = [0, 0]
            played_count_points = 0
            trick_plays: list[tuple[int, int]] = []
            played_slots: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}

            for decision_idx, decision in enumerate(getattr(game, "decisions", [])):
                q_per_world = getattr(decision, "q_per_world", None)
                if q_per_world is not None:
                    sample_counts[int(q_per_world.shape[0])] += 1
                rows = process_decision(
                    source_file=path.name,
                    game_idx=game_idx,
                    seed=seed,
                    decl_id=decl_id,
                    decision_idx=decision_idx,
                    decision=decision,
                    hands=hands,
                    score=score,
                    played_count_points=played_count_points,
                    trick_plays=trick_plays,
                )
                if rows:
                    file_decisions += 1
                    coverage["decision_rows"] += 1
                    coverage["action_rows"] += len(rows)
                    file_actions += len(rows)
                    file_threat_actions += sum(1 for row in rows if row["top_hidden_impact_score"] >= 5.0)
                    if any(row["scalar_ev_lying_by_omission"] for row in rows):
                        scalar_omission_decisions += 1
                    if max(row["top_hidden_downside_score"] for row in rows) >= 5.0:
                        hidden_downside_decisions += 1
                    actual = next((row for row in rows if row["is_actual_action"]), None)
                    safest = min(rows, key=lambda row: row["lower_tail_mass"])
                    if actual is not None and actual is not safest:
                        actual_not_safest_decisions += 1
                    add_decision_contrasts(
                        rows,
                        contrasts,
                        contrast_examples,
                        example_limit=args.example_limit,
                    )

                    for row in rows:
                        for group in row_groups(row):
                            group_stats[group].add(row)
                        for kind, score_key in [
                            ("impact", "score"),
                            ("downside", "score"),
                            ("upside", "score"),
                        ]:
                            driver = row[f"top_hidden_{kind}"]
                            key = (
                                kind,
                                int(row["decl_id"]),
                                str(row["decl_name"]),
                                str(row["seat_role"]),
                                str(driver["holder_role"]),
                                str(driver["hidden_domino"]),
                            )
                            driver_stats[key].add(driver, score_key)
                        if row["top_hidden_impact_score"] >= 20.0:
                            topk_insert(
                                top_impact_examples,
                                {
                                    "source_file": row["source_file"],
                                    "seed": row["seed"],
                                    "decl_id": row["decl_id"],
                                    "decl_name": row["decl_name"],
                                    "decision_idx": row["decision_idx"],
                                    "trick_idx": row["trick_idx"],
                                    "trick_position": row["trick_position"],
                                    "seat_role": row["seat_role"],
                                    "team": row["team"],
                                    "candidate_domino": row["candidate_domino"],
                                    "mean": row["mean"],
                                    "threshold_mass": row["threshold_mass"],
                                    "lower_tail_mass": row["lower_tail_mass"],
                                    "top_hidden_impact_score": row["top_hidden_impact_score"],
                                    "top_hidden_impact": row["top_hidden_impact"],
                                    "top_hidden_downside_score": row["top_hidden_downside_score"],
                                    "top_hidden_downside": row["top_hidden_downside"],
                                    "strategy_context_tags": row["strategy_context_tags"],
                                    "matched_position_detectors": row["matched_position_detectors"],
                                    "distribution_shape_tags": row["distribution_shape_tags"],
                                },
                                key="top_hidden_impact_score",
                                limit=args.example_limit,
                            )

                played_count_points, trick_plays = advance_actual(
                    decision=decision,
                    hands=hands,
                    decl_id=decl_id,
                    score=score,
                    played_count_points=played_count_points,
                    trick_plays=trick_plays,
                    played_slots=played_slots,
                )

        meta = {
            "path": str(path),
            "games": len(games),
            "action_rows": file_actions,
            "decision_rows": file_decisions,
            "hidden_large_impact_action_rows": file_threat_actions,
            "load_and_process_seconds": time.perf_counter() - file_started,
        }
        file_meta.append(meta)
        if file_idx % max(args.log_every_files, 1) == 0 or file_idx == len(paths):
            wb.log(
                {
                    "progress/files_processed": file_idx,
                    "progress/wall_seconds": time.perf_counter() - started,
                    "coverage/decision_rows": coverage["decision_rows"],
                    "coverage/action_rows": coverage["action_rows"],
                    "branch/scalar_omission_decisions": scalar_omission_decisions,
                    "threat/hidden_downside_decisions": hidden_downside_decisions,
                    "threat/hidden_large_impact_action_rate": (
                        group_stats[("all", "all")].sums["hidden_large_impact_rate"]
                        / max(group_stats[("all", "all")].n, 1)
                    ),
                    "threat/mean_top_impact_score": (
                        group_stats[("all", "all")].sums["top_hidden_impact_score"]
                        / max(group_stats[("all", "all")].n, 1)
                    ),
                    "threat/max_top_impact_score": group_stats[("all", "all")].maxes.get(
                        "max_hidden_impact_score", 0.0
                    ),
                },
                step=file_idx,
            )

    group_rows = [stats.row(group_type, group) for (group_type, group), stats in sorted(group_stats.items())]
    driver_rows = [stats.row(key) for key, stats in driver_stats.items()]
    driver_rows.sort(key=lambda row: (row["driver_kind"], -float(row["mean_score"]), row["hidden_domino"]))
    contrast_rows = contrasts.rows(
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    claim_evidence_rows = build_claim_evidence_rows(group_stats)
    examples = {
        "top_hidden_impact_actions": top_impact_examples,
        "mitigation_contrasts": dict(contrast_examples),
    }
    summary = {
        "schema_version": "w42.hidden_threat_legacy_mining.v0",
        "bead_id": "t42-0b4l.4",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_sha(),
        "git_status_before_artifacts": git_status_short(),
        "inputs": [str(path) for path in paths],
        "file_meta": file_meta,
        "coverage": {
            "input_files": len(paths),
            "decision_rows": coverage["decision_rows"],
            "action_rows": coverage["action_rows"],
            "decl_counts": dict(sorted(decl_counts.items())),
            "sample_counts": dict(sorted(sample_counts.items())),
            "scalar_omission_decisions": scalar_omission_decisions,
            "hidden_downside_decisions": hidden_downside_decisions,
            "actual_not_safest_tail_decisions": actual_not_safest_decisions,
        },
        "headline": {
            "hidden_large_impact_action_rate": group_stats[("all", "all")].sums["hidden_large_impact_rate"]
            / max(group_stats[("all", "all")].n, 1),
            "mean_top_hidden_impact_score": group_stats[("all", "all")].sums["top_hidden_impact_score"]
            / max(group_stats[("all", "all")].n, 1),
            "max_top_hidden_impact_score": group_stats[("all", "all")].maxes.get("max_hidden_impact_score", 0.0),
            "mean_top_hidden_downside_score": group_stats[("all", "all")].sums["top_hidden_downside_score"]
            / max(group_stats[("all", "all")].n, 1),
            "max_top_hidden_downside_score": group_stats[("all", "all")].maxes.get(
                "max_hidden_downside_score", 0.0
            ),
        },
        "paired_mitigation_contrasts": contrast_rows,
        "claim_branch_impact_evidence": {
            "path": str(output_dir / "claim_branch_impact_evidence.csv"),
            "rows": len(claim_evidence_rows),
            "status_impact": "marker only; no central claim status movement",
        },
        "scientific_status": {
            "interpretation": (
                "Streaming legacy-corpus hidden-threat mining. It connects branch-atlas hidden-holder "
                "impact labels to distribution-aware mitigation contrasts and claim-detector surfaces. "
                "It is offline diagnostic evidence, not a live policy feature."
            ),
            "claim_ledger_impact": "No central claim status changes; use as branch-impact evidence for downstream claim families.",
            "leakage_boundary": "world_hands, q_per_world, E[Q], and hidden holders are offline labels only.",
            "wandb": wb.status(),
        },
        "wall_seconds": time.perf_counter() - started,
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "group_metrics.csv", group_rows, GROUP_FIELDS)
    write_csv(output_dir / "hidden_driver_rollup.csv", driver_rows, DRIVER_FIELDS)
    write_csv(output_dir / "mitigation_contrasts.csv", contrast_rows, CONTRAST_FIELDS)
    write_csv(output_dir / "claim_branch_impact_evidence.csv", claim_evidence_rows, CLAIM_EVIDENCE_FIELDS)
    write_json(output_dir / "examples.json", examples)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "w42.hidden_threat_legacy_mining.manifest.v0",
            "created_at_utc": summary["created_at_utc"],
            "command": " ".join(sys.argv),
            "outputs": {
                "summary": str(output_dir / "summary.json"),
                "group_metrics": str(output_dir / "group_metrics.csv"),
                "hidden_driver_rollup": str(output_dir / "hidden_driver_rollup.csv"),
                "mitigation_contrasts": str(output_dir / "mitigation_contrasts.csv"),
                "claim_branch_impact_evidence": str(output_dir / "claim_branch_impact_evidence.csv"),
                "examples": str(output_dir / "examples.json"),
            },
            "large_row_policy": "No full action-row dump is emitted; rerun from corpus paths for row-level detail.",
            "leakage_boundary": summary["scientific_status"]["leakage_boundary"],
            "wandb": wb.status(),
        },
    )
    wb.update_summary(
        {
            "coverage/action_rows": summary["coverage"]["action_rows"],
            "coverage/decision_rows": summary["coverage"]["decision_rows"],
            "threat/hidden_large_impact_action_rate": summary["headline"]["hidden_large_impact_action_rate"],
            "threat/max_top_hidden_impact_score": summary["headline"]["max_top_hidden_impact_score"],
            "status": "completed",
        }
    )
    wb.log_artifact_files(
        name=f"w42-hidden-threat-legacy-mining-{git_sha()[:7]}",
        artifact_type="w42-report",
        paths=[
            output_dir / "summary.json",
            output_dir / "group_metrics.csv",
            output_dir / "hidden_driver_rollup.csv",
            output_dir / "mitigation_contrasts.csv",
            output_dir / "claim_branch_impact_evidence.csv",
            output_dir / "examples.json",
            output_dir / "manifest.json",
        ],
    )
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
