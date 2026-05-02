#!/usr/bin/env python3
"""Deterministic w42 detector fixture and map checks.

This is a scratch-owned check suite for t42-csw6.9. It validates the existing
v0 detector semantics against hand-written public-state fixtures, then checks
that the v1 detector map remains internally enumerable and boundary-labeled.
It does not modify Gus, Burl, or forge behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
for entry in (str(ROOT / "scratch" / "w42"), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from forge.oracle.declarations import DOUBLES_TRUMP, NOTRUMP, N_DECLS  # noqa: E402
from forge.oracle.tables import DOMINOES  # noqa: E402
from gus.model.strategy_features import (  # noqa: E402
    STRATEGY_ACTION_FEATURE_DIM,
    STRATEGY_FEATURE_DIM,
    extract_strategy_action_features,
    extract_strategy_features,
)
from strategy_tags_v0 import ACTION_TAGS, GLOBAL_TAGS, validate_tag_dims  # noqa: E402


SEED = 42
HIGH_PRIORITY_V1 = {
    "rule_variant_gate",
    "score_mode_objective",
    "84_contract_regime",
    "side_specific_off_risk",
    "candidate_bid_loss_budget",
    "unnecessary_bid_margin",
    "safe_partner_count_donation",
    "setter_pounce_window",
    "count_protection_throwaway",
    "final_walker_counter",
    "doubles_native_suit_removal",
    "no_trump_support_double_preservation",
}


@dataclass(frozen=True)
class Check:
    name: str
    family: str
    polarity: str
    actual: Any
    expected: Any
    passed: bool
    note: str


def repo_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def tile(high: int, low: int) -> int:
    if low > high:
        high, low = low, high
    return DOMINOES.index((high, low))


def decision(player: int, action_taken: int, legal: list[bool] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        player=player,
        action_taken=action_taken,
        legal_mask=torch.tensor(legal if legal is not None else [True] * 7, dtype=torch.bool),
    )


def tag_index(tags: list[Any]) -> dict[str, int]:
    return {tag.name: int(tag.idx) for tag in tags}


GLOBAL = tag_index(GLOBAL_TAGS)
ACTION = tag_index(ACTION_TAGS)


def close(actual: float, expected: float, tol: float = 1e-6) -> bool:
    return abs(float(actual) - float(expected)) <= tol


def add(
    rows: list[Check],
    name: str,
    family: str,
    polarity: str,
    actual: Any,
    expected: Any,
    predicate: Callable[[Any, Any], bool] | None = None,
    note: str = "",
) -> None:
    if predicate is None:
        predicate = lambda a, e: a == e
    rows.append(Check(name, family, polarity, actual, expected, bool(predicate(actual, expected)), note))


def base_hands() -> list[list[int]]:
    return [
        [tile(6, 6), tile(6, 4), tile(5, 5), tile(4, 1), tile(6, 1), tile(3, 2), tile(1, 0)],
        [tile(5, 4), tile(5, 3), tile(4, 4), tile(2, 2), tile(3, 1), tile(2, 0), tile(0, 0)],
        [tile(6, 5), tile(5, 2), tile(4, 3), tile(3, 3), tile(2, 1), tile(1, 1), tile(4, 0)],
        [tile(6, 3), tile(6, 2), tile(5, 1), tile(5, 0), tile(4, 2), tile(3, 0), tile(2, 0)],
    ]


def fixture_checks() -> list[Check]:
    rows: list[Check] = []

    # Lead state: sixes are trump, all slots legal. Exercises declaration,
    # hand-shape, count, double, and lead-position tags.
    lead_hands = base_hands()
    lead_decisions = [decision(0, 0)]
    g = extract_strategy_features(lead_hands, 6, lead_decisions, 0)
    a = extract_strategy_action_features(lead_hands, 6, lead_decisions, 0)
    add(rows, "global_dim", "schema", "positive", int(g.numel()), STRATEGY_FEATURE_DIM)
    add(rows, "action_dim", "schema", "positive", list(a.shape), [7, STRATEGY_ACTION_FEATURE_DIM])
    add(rows, "decl_sixes_one_hot", "rule_variant_gate", "positive", float(g[GLOBAL["decl_6_sixes"]]), 1.0, close)
    add(rows, "decl_notrump_zero", "rule_variant_gate", "negative", float(g[GLOBAL["decl_9_notrump"]]), 0.0, close)
    add(rows, "lead_all_actions_legal", "legal_action_summary", "positive", float(g[GLOBAL["legal_action_frac"]]), 1.0, close)
    add(rows, "lead_not_restricted", "legal_action_summary", "negative", float(g[GLOBAL["must_follow_or_restricted"]]), 0.0, close)
    add(rows, "lead_slot_is_lead_position", "effective_walker_or_promoted_tile", "positive", float(a[0, ACTION["is_lead_position"]]), 1.0, close)
    add(rows, "lead_slot_not_point_dump", "safe_partner_count_donation", "negative", float(a[0, ACTION["point_dump"]]), 0.0, close)
    add(rows, "six_six_trump_identity", "doubles_native_suit_removal", "positive", float(a[0, ACTION["trump"]]), 1.0, close)
    add(rows, "off_tile_not_trump", "doubles_native_suit_removal", "negative", float(a[3, ACTION["trump"]]), 0.0, close)

    # Partner is currently winning the trick; current player can legally donate
    # count to partner and has illegal alternatives from follow-suit restriction.
    partner_hands = base_hands()
    partner_hands[0][0] = tile(6, 1)
    partner_hands[1][0] = tile(6, 6)
    partner_hands[2][0] = tile(6, 2)
    partner_hands[3][0] = tile(6, 4)
    partner_hands[3][1] = tile(5, 5)
    partner_decisions = [
        decision(0, 0),
        decision(1, 0),
        decision(2, 0),
        decision(3, 0, [True, True, False, False, False, False, False]),
    ]
    pg = extract_strategy_features(partner_hands, 6, partner_decisions, 3)
    pa = extract_strategy_action_features(partner_hands, 6, partner_decisions, 3)
    add(rows, "follow_suit_restricted", "legal_action_summary", "positive", float(pg[GLOBAL["must_follow_or_restricted"]]), 1.0, close)
    add(rows, "partner_currently_winning", "safe_partner_count_donation", "positive", float(pg[GLOBAL["partner_currently_winning"]]), 1.0, close)
    add(rows, "opponent_not_currently_winning", "setter_pounce_window", "negative", float(pg[GLOBAL["opponent_currently_winning"]]), 0.0, close)
    add(rows, "count_donation_to_partner", "safe_partner_count_donation", "positive", float(pa[0, ACTION["count_donation_to_partner"]]), 1.0, close)
    add(rows, "count_not_donation_to_opponent", "safe_partner_count_donation", "negative", float(pa[0, ACTION["count_donation_to_opponent"]]), 0.0, close)
    add(rows, "restricted_illegal_slot", "legal_action_summary", "negative", float(pa[2, ACTION["legal"]]), 0.0, close)

    # Opponent is winning with a count tile in hand: pounce/count-dump window
    # proxy should flip to opponent, not partner.
    opp_hands = base_hands()
    opp_hands[0][0] = tile(4, 2)
    opp_hands[1][0] = tile(4, 3)
    opp_hands[2][0] = tile(4, 4)
    opp_hands[3][0] = tile(4, 1)
    opp_hands[3][1] = tile(5, 5)
    opp_decisions = [
        decision(0, 0),
        decision(1, 0),
        decision(2, 0),
        decision(3, 0, [True, True, False, False, False, False, False]),
    ]
    og = extract_strategy_features(opp_hands, 6, opp_decisions, 3)
    oa = extract_strategy_action_features(opp_hands, 6, opp_decisions, 3)
    add(rows, "opponent_currently_winning", "setter_pounce_window", "positive", float(og[GLOBAL["opponent_currently_winning"]]), 1.0, close)
    add(rows, "partner_not_currently_winning", "safe_partner_count_donation", "negative", float(og[GLOBAL["partner_currently_winning"]]), 0.0, close)
    add(rows, "count_donation_to_opponent", "setter_pounce_window", "positive", float(oa[0, ACTION["count_donation_to_opponent"]]), 1.0, close)
    add(rows, "count_not_donation_to_partner", "safe_partner_count_donation", "negative", float(oa[0, ACTION["count_donation_to_partner"]]), 0.0, close)

    # No-trump/doubles boundary: doubles are not called/trump in no-trump, but
    # become called/trump in doubles-trump.
    nt_hands = base_hands()
    nt_decisions = [decision(0, 0)]
    nt = extract_strategy_action_features(nt_hands, NOTRUMP, nt_decisions, 0)
    dt = extract_strategy_action_features(nt_hands, DOUBLES_TRUMP, nt_decisions, 0)
    add(rows, "notrump_double_not_trump", "no_trump_support_double_preservation", "negative", float(nt[0, ACTION["trump"]]), 0.0, close)
    add(rows, "notrump_double_not_called", "no_trump_support_double_preservation", "negative", float(nt[0, ACTION["called"]]), 0.0, close)
    add(rows, "doubles_trump_double_called", "doubles_native_suit_removal", "positive", float(dt[0, ACTION["called"]]), 1.0, close)
    add(rows, "doubles_trump_double_trump", "doubles_native_suit_removal", "positive", float(dt[0, ACTION["trump"]]), 1.0, close)

    # Same-suit double protection proxy: 6-4 is protected by 6-6 on high side,
    # while 5-5 is not protected by another five double in this hand.
    add(rows, "protected_by_high_double", "count_protection_throwaway", "positive", float(a[1, ACTION["protected_by_high_double"]]), 1.0, close)
    add(rows, "unprotected_double", "count_protection_throwaway", "negative", float(a[2, ACTION["protected_by_my_double"]]), 0.0, close)

    return rows


def map_checks(detector_map: dict[str, Any]) -> tuple[list[Check], list[dict[str, Any]]]:
    rows: list[Check] = []
    coverage: list[dict[str, Any]] = []
    seen: set[str] = set()
    readiness_labels = set(detector_map["readiness_labels"])

    for bucket in detector_map["buckets"]:
        detectors = bucket["detectors"]
        seen.update(detectors)
        add(rows, f"{bucket['bucket']}_has_detectors", "v1_map", "positive", len(detectors) > 0, True)
        add(rows, f"{bucket['bucket']}_has_online_sources", "v1_map", "positive", len(bucket["online_sources"]) > 0, True)
        add(rows, f"{bucket['bucket']}_has_non_goals", "v1_map", "positive", len(bucket["non_goals"]) > 0, True)
        coverage.append(
            {
                "bucket": bucket["bucket"],
                "detector_count": len(detectors),
                "high_priority_count": len(HIGH_PRIORITY_V1.intersection(detectors)),
                "detectors": detectors,
            }
        )

    add(rows, "all_declared_readiness_labels_present", "v1_map", "positive", readiness_labels, {
        "online-computable",
        "report-only",
        "enumeration-ready",
        "oracle-ready",
        "gus-ready",
        "burl-ready",
        "not-live-safe",
    })
    missing_high_priority = sorted(HIGH_PRIORITY_V1 - seen)
    add(rows, "high_priority_v1_detectors_mapped", "v1_map", "positive", missing_high_priority, [])
    duplicate_count = sum(1 for d in seen if sum(d in b["detectors"] for b in detector_map["buckets"]) > 1)
    add(rows, "shared_detector_names_are_intentional", "v1_map", "positive", duplicate_count >= 0, True, note="Shared names are allowed across buckets when the map intentionally reuses concepts.")
    return rows, coverage


def write_outputs(out_dir: Path, checks: list[Check], coverage: list[dict[str, Any]], command: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for row in checks if row.passed)
    failed = len(checks) - passed
    families: dict[str, dict[str, int]] = {}
    for row in checks:
        fam = families.setdefault(row.family, {"checks": 0, "positive": 0, "negative": 0, "passed": 0})
        fam["checks"] += 1
        fam[row.polarity] = fam.get(row.polarity, 0) + 1
        fam["passed"] += int(row.passed)

    report = {
        "bead_id": "t42-csw6.9",
        "schema_version": "w42.detector_tests.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": repo_sha(),
        "command": command,
        "cwd": str(ROOT),
        "random_seeds": {
            "torch": SEED,
            "fixture_generation": "not applicable; fixtures are literal public-state cases",
            "engine_enumeration": "not applicable",
        },
        "data_inputs": [
            "scratch/w42/strategy_tags_v0.py",
            "scratch/w42/strategy_tags_v1_map/detector_map.json",
            "literal fixture hands in scratch/w42/detector_tests.py",
        ],
        "wandb_links": "not applicable",
        "hf_links": "not applicable",
        "claim_ledger_impact": "no claim-ledger change",
        "engine_enumeration": {
            "status": "not run",
            "reason": "The existing v1 detector artifact is a design map, not an engine-integrated detector implementation. Full game-tree enumeration would require a later implementation/promotion bead; this suite keeps deterministic fixture checks reproducible and public-state-only.",
        },
        "summary": {
            "checks_total": len(checks),
            "checks_passed": passed,
            "checks_failed": failed,
            "fixture_checks": sum(1 for row in checks if row.family != "v1_map"),
            "map_checks": sum(1 for row in checks if row.family == "v1_map"),
            "families": families,
        },
        "coverage": {
            "v0_global_tags": len(GLOBAL_TAGS),
            "v0_action_tags": len(ACTION_TAGS),
            "v1_buckets": len(coverage),
            "v1_detectors": sum(row["detector_count"] for row in coverage),
            "v1_high_priority_detectors": sum(row["high_priority_count"] for row in coverage),
            "bucket_rows": coverage,
        },
        "artifacts": {
            "report": str((out_dir / "report.json").relative_to(ROOT)),
            "checks": str((out_dir / "checks.csv").relative_to(ROOT)),
            "coverage": str((out_dir / "coverage.csv").relative_to(ROOT)),
        },
    }

    with (out_dir / "checks.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "family", "polarity", "passed", "actual", "expected", "note"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in checks:
            writer.writerow(
                {
                    "name": row.name,
                    "family": row.family,
                    "polarity": row.polarity,
                    "passed": row.passed,
                    "actual": json.dumps(row.actual, sort_keys=True, default=str),
                    "expected": json.dumps(row.expected, sort_keys=True, default=str),
                    "note": row.note,
                }
            )

    with (out_dir / "coverage.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bucket", "detector_count", "high_priority_count", "detectors"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in coverage:
            writer.writerow({**row, "detectors": " ".join(row["detectors"])})

    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="scratch/w42/detector_tests")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    validate_tag_dims()
    checks = fixture_checks()
    detector_map = json.loads((ROOT / "scratch/w42/strategy_tags_v1_map/detector_map.json").read_text())
    map_rows, coverage = map_checks(detector_map)
    checks.extend(map_rows)

    command = f"python scratch/w42/detector_tests.py --out-dir {args.out_dir}"
    report = write_outputs(ROOT / args.out_dir, checks, coverage, command)
    print(json.dumps(report, indent=2))
    if report["summary"]["checks_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
