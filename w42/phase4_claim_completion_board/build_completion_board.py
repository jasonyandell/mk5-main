#!/usr/bin/env python3
"""Generate the W42 phase-4 claim completion board.

This script reconciles the central 64-row Winning 42 claim ledger with the
phase-2/phase-3 routing and evidence artifacts. It intentionally writes only
inside this artifact directory; it does not update the central ledger, wiki, or
bead database.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "w42" / "phase4_claim_completion_board"

LEDGER_PATH = REPO_ROOT / "w42" / "statistics_claims_ledger" / "claims.csv"
MATRIX_PATH = REPO_ROOT / "w42" / "claim_analysis_matrix" / "claim_analysis_matrix.csv"
READY_PATH = REPO_ROOT / "w42" / "claim_analysis_matrix" / "ready_powered_tests.csv"
SYNTHESIS_PATH = REPO_ROOT / "w42" / "claim_analysis_synthesis" / "claim_family_synthesis.csv"


PHASE4_CHILDREN = {
    "t42-br7n.1": "bidder/partner/setter hand-shape and sequence counterfactuals",
    "t42-br7n.2": "84 dynamic mined-seed action tests",
    "t42-br7n.3": "scoring objective and tournament tests",
    "t42-br7n.4": "deterministic laydown and rule-accounting proof tests",
    "t42-br7n.5": "doubles/no-trump paired-regime tests",
    "t42-br7n.6": "completion board and ledger reconciliation",
}


PHASE_ARTIFACTS = {
    "odds_static": [
        "w42/odds_ruleset_claim_validation/validation_summary.json",
        "w42/statistics_claims_ledger/claims.csv",
    ],
    "phase2_matrix": [
        "w42/claim_analysis_matrix/claim_analysis_matrix.csv",
        "w42/claim_analysis_matrix/ready_powered_tests.csv",
    ],
    "phase2_synthesis": [
        "w42/claim_analysis_synthesis/claim_family_synthesis.csv",
        "w42/claim_analysis_synthesis/summary.json",
    ],
    "auction_phase3": [
        "w42/auction_bid_discipline_claim_tests/summary.json",
        "w42/auction_bid_discipline_claim_tests/bid_margin_summary.csv",
        "w42/auction_bid_discipline_claim_tests/natural_bucket_summary.csv",
        "w42/auction_bid_discipline_claim_tests/partner_signal_summary.csv",
        "w42/auction_bid_discipline_claim_tests/risk_budget_summary.csv",
    ],
    "bid_only_enough_phase2": [
        "w42/bid_only_enough_claim_tests/summary.json",
        "w42/bid_only_enough_claim_tests/margin_summary.csv",
    ],
    "seat_sequence_phase3": [
        "w42/sequence_seat_counterfactuals/summary.json",
        "w42/sequence_seat_counterfactuals/paired_contrasts.csv",
        "w42/sequence_seat_counterfactuals/label_metrics.csv",
        "w42/seat_position_claim_tests/summary.json",
        "w42/tactical_claim_replication/summary.json",
    ],
    "eighty_four_phase3": [
        "w42/eighty_four_seed_mining/summary.json",
        "w42/eighty_four_seed_mining/surface_summary.csv",
        "w42/eighty_four_claim_validation/summary.json",
        "w42/eighty_four_claim_validation/claim_ledger_delta.json",
    ],
    "doubles_no_trump": [
        "w42/doubles_no_trump_claim_validation/summary.json",
        "w42/doubles_no_trump_claim_validation/claim_ledger_delta.json",
        "w42/doubles_no_trump_legacy_mining/summary.json",
    ],
    "scoring": [
        "w42/scoring_objective_drift_claim_validation/summary.json",
        "w42/scoring_objective_drift_claim_validation/claim_ledger_delta.json",
    ],
    "joined_model": [
        "w42/joined_claim_row_model_table/summary.json",
        "w42/joined_claim_row_model_table/ablation_results.csv",
        "w42/joined_claim_row_model_table/family_inventory.csv",
    ],
}


CLAIM_OVERRIDES: dict[str, dict[str, str]] = {
    "ch02-bid-only-enough": {
        "completion_bucket": "review_for_ledger_update",
        "evidence_strength": "generated_direct",
        "next_bead": "ledger_reconciliation",
        "phase4_recommendation": "review_status_move_to_supported_or_context_limited",
        "headline": "Phase 3 found 0/14976 positive-margin rows where bidding above minimum improved over the minimum bid.",
        "caveat": "Auction contexts are synthetic and use bidder-hand Monte Carlo labels rather than a full table auction policy.",
    },
    "ch12-natural-bid-bucket-anomaly": {
        "completion_bucket": "phase3_context_limited",
        "evidence_strength": "generated_partial",
        "next_bead": "needs_new_bidding_count_exposure_bead",
        "phase4_recommendation": "add_or_assign_bid_bucket_followup",
        "headline": "Natural max-profitable bid buckets appeared in 68/384 contracts (17.708%).",
        "caveat": "The corpus measures max-profitable thresholds, not cultural/natural bid selection under live opponents.",
    },
    "ch02-three-plus-trumps-good-start": {
        "completion_bucket": "phase3_context_limited",
        "evidence_strength": "generated_partial",
        "next_bead": "needs_new_bidding_count_exposure_bead",
        "phase4_recommendation": "add_or_assign_bid_shape_followup",
        "headline": "Phase 3 joined static trump/count buckets to empirical make labels across 384 contract rows.",
        "caveat": "Still lacks full auction competition and score-conditioned pass/bid decisions.",
    },
    "ch02-risk-budget-threshold": {
        "completion_bucket": "phase3_context_limited",
        "evidence_strength": "generated_partial",
        "next_bead": "needs_new_bidding_count_exposure_bead",
        "phase4_recommendation": "add_or_assign_risk_budget_followup",
        "headline": "Static risk buckets showed lower make/profit rates as off/count exposure increased.",
        "caveat": "Risk buckets are coarse; exact bid margins and partner/opponent adaptation remain offline.",
    },
    "ch02-strong-trump-bad-risk-trap": {
        "completion_bucket": "needs_phase4_or_new_scope",
        "evidence_strength": "generated_partial",
        "next_bead": "needs_new_bidding_count_exposure_bead",
        "phase4_recommendation": "add_or_assign_count_exposure_followup",
        "headline": "Auction/risk artifacts can route this, but no dedicated phase-4 child currently owns Ch02 count-exposure refinements.",
        "caveat": "Requires paired hand-shape contrasts where strong trump is separated from off/count liability.",
    },
    "ch02-four-five-off-danger": {
        "completion_bucket": "needs_phase4_or_new_scope",
        "evidence_strength": "generated_partial",
        "next_bead": "needs_new_bidding_count_exposure_bead",
        "phase4_recommendation": "add_or_assign_count_exposure_followup",
        "headline": "Phase 3 has risk buckets, but this exact four/five off danger claim remains a shape-specific blocker.",
        "caveat": "Needs matched generated hands that isolate off-suit count exposure.",
    },
    "ch02-double-side-protection": {
        "completion_bucket": "needs_phase4_or_new_scope",
        "evidence_strength": "generated_partial",
        "next_bead": "needs_new_bidding_count_exposure_bead",
        "phase4_recommendation": "add_or_assign_count_exposure_followup",
        "headline": "No direct paired double-side protection contrast exists yet.",
        "caveat": "Needs generated shape pairs or exact enumeration joined to make/set labels.",
    },
    "ch16-partner-two-plus-doubles-prior": {
        "completion_bucket": "phase3_context_limited",
        "evidence_strength": "generated_partial",
        "next_bead": "needs_new_bidding_count_exposure_bead",
        "phase4_recommendation": "add_or_assign_partner_prior_followup",
        "headline": "Partner-high context overcall rate was 77/192 (40.104%) in phase 3.",
        "caveat": "Partner values are offline bidder-hand labels, not observed partner bidding intelligence.",
    },
    "ch16-four-trump-boss-first-policy": {
        "completion_bucket": "active_phase4",
        "evidence_strength": "generated_partial",
        "next_bead": "t42-br7n.1",
        "phase4_recommendation": "keep_open_under_phase4_1",
        "headline": "Phase 3 sequence tests show called double leads are strong, but blanket called-suit-first is negative.",
        "caveat": "Needs shape-specific boss-first/low-trump exception tests.",
    },
    "ch03-trump-pull-sequencing": {
        "completion_bucket": "active_phase4",
        "evidence_strength": "generated_partial",
        "next_bead": "t42-br7n.1",
        "phase4_recommendation": "keep_open_under_phase4_1",
        "headline": "Called double vs lower called-suit was +5.720 Q; pooled called-suit lead vs off-suit was -1.935 Q.",
        "caveat": "The broad book wording needs exception buckets, not a blanket promotion.",
    },
    "ch03-count-inventory": {
        "completion_bucket": "active_phase4",
        "evidence_strength": "generated_direct_negative_control",
        "next_bead": "t42-br7n.1",
        "phase4_recommendation": "keep_open_under_phase4_1",
        "headline": "Bidder lead count vs noncount was -4.877 Q in phase 3.",
        "caveat": "This supports inventory caution, but not a complete bidder-plan proof.",
    },
    "ch04-safe-partner-count-donation": {
        "completion_bucket": "active_phase4",
        "evidence_strength": "generated_direct",
        "next_bead": "t42-br7n.1",
        "phase4_recommendation": "keep_open_under_phase4_1",
        "headline": "Partner count when bidder side controls was +0.608 Q.",
        "caveat": "The effect is real but small and timing-gated.",
    },
    "ch04-unsafe-partner-count-donation": {
        "completion_bucket": "complete_direct_empirical",
        "evidence_strength": "generated_direct",
        "next_bead": "none",
        "phase4_recommendation": "close_as_supported_negative_control",
        "headline": "Partner count into defense was -8.351 Q in phase 3.",
        "caveat": "Directly operationalized as same-decision action contrast.",
    },
    "ch05-pounce-count-before-certainty": {
        "completion_bucket": "complete_direct_empirical",
        "evidence_strength": "generated_direct",
        "next_bead": "none",
        "phase4_recommendation": "close_as_supported",
        "headline": "Setter pounce count was strongly positive; pounce-to-set-now was +5.930 Q in phase 3.",
        "caveat": "Still row-local, not full multi-trick plan simulation.",
    },
    "ch05-reckless-count-to-bidder": {
        "completion_bucket": "complete_direct_empirical",
        "evidence_strength": "generated_direct",
        "next_bead": "none",
        "phase4_recommendation": "close_as_supported_negative_control",
        "headline": "Setter reckless count to bidder was -7.829 Q; Ch05 label replication was also strongly negative.",
        "caveat": "Directly operationalized as an action-local negative-control claim.",
    },
    "ch05-extra-count-to-set": {
        "completion_bucket": "active_phase4",
        "evidence_strength": "generated_partial",
        "next_bead": "t42-br7n.1",
        "phase4_recommendation": "keep_open_under_phase4_1",
        "headline": "Pounce-to-set-now slices are positive, but extra-count-to-set needs bid-margin/set-accounting gates.",
        "caveat": "Needs explicit set-now and remaining-count controls.",
    },
    "ch03-laydown-correctness": {
        "completion_bucket": "active_phase4",
        "evidence_strength": "blocked_or_fixture_needed",
        "next_bead": "t42-br7n.4",
        "phase4_recommendation": "keep_open_under_phase4_4",
        "headline": "Phase 3 seed mining found one natural all-trump laydown seed; proof fixtures remain open.",
        "caveat": "Needs deterministic remaining-trick proof/checker, not row-model evidence.",
    },
    "ch07-protected-one-off-shape-frequency": {
        "completion_bucket": "review_for_ledger_update",
        "evidence_strength": "generated_seed_inventory",
        "next_bead": "ledger_reconciliation",
        "phase4_recommendation": "review_status_move_to_context_limited_or_supported_frequency",
        "headline": "Phase 3 mined 22,176 protected-one-off candidate rows in 50,000 seeds.",
        "caveat": "This is seed frequency/surface inventory, not late-state action value.",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def rel(path: Path | str) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)
    return str(p)


def artifact_exists(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def existing_artifacts(names: list[str]) -> list[str]:
    return [name for name in names if artifact_exists(name)]


def artifact_group_for_family(family: str, claim_id: str) -> list[str]:
    groups = ["phase2_matrix", "phase2_synthesis"]
    if family in {"hand odds", "void frequencies", "double-count odds", "four-trump odds", "trump counts"}:
        groups.append("odds_static")
    if family in {"bidding risk", "count exposure", "partner-help odds", "four-trump tactics"}:
        groups.extend(["bid_only_enough_phase2", "auction_phase3", "joined_model"])
    if family in {"bidder sequencing", "partner support", "setter defense"}:
        groups.extend(["seat_sequence_phase3", "joined_model"])
    if family in {"84 bidder/defender", "stopper ownership"}:
        groups.extend(["eighty_four_phase3", "joined_model"])
    if family in {"doubles/no-trump rules", "doubles/no-trump odds"}:
        groups.extend(["doubles_no_trump", "joined_model"])
    if family == "scoring objective":
        groups.append("scoring")
    if claim_id == "ch03-laydown-correctness":
        groups.append("eighty_four_phase3")
    return groups


def default_next_bead(family: str, claim_id: str, ledger_status: str, matrix_class: str) -> str:
    if ledger_status == "contradicted":
        return "none"
    if ledger_status == "supported":
        return "none"
    if family in {"bidder sequencing", "partner support", "setter defense"}:
        return "t42-br7n.1"
    if family == "four-trump tactics":
        return "t42-br7n.1"
    if family in {"84 bidder/defender", "stopper ownership"}:
        return "t42-br7n.2"
    if family == "scoring objective":
        if ledger_status == "supported":
            return "none"
        return "t42-br7n.3"
    if claim_id == "ch03-laydown-correctness":
        return "t42-br7n.4"
    if family == "doubles/no-trump odds":
        return "t42-br7n.5"
    if family == "doubles/no-trump rules":
        return "none"
    if family in {"bidding risk", "count exposure", "partner-help odds"}:
        return "needs_new_bidding_count_exposure_bead"
    return "none" if ledger_status in {"supported", "contradicted"} else "coordinator_triage"


def classify_default(
    family: str,
    claim_id: str,
    ledger_status: str,
    evidence_mode: str,
    matrix_class: str,
    next_bead: str,
) -> tuple[str, str, str, str, str]:
    mode = evidence_mode.lower()
    staticish = any(token in mode for token in ["exact", "ruleset", "deterministic", "enumeration"])
    if ledger_status == "contradicted":
        return (
            "complete_static_contradicted" if staticish else "complete_contradicted",
            "direct_or_deterministic",
            "close_as_contradicted",
            "Ledger already marks this contradicted; keep as tested unless a new operational definition is proposed.",
            "Contradiction should be preserved unless a phase-4 worker explicitly redefines the claim.",
        )
    if ledger_status == "supported" and (staticish or family in {"doubles/no-trump rules", "scoring objective"}):
        return (
            "complete_static_supported",
            "deterministic_or_ruleset",
            "close_as_already_supported",
            "Ledger already has deterministic/ruleset support.",
            "Report-only support does not imply broad policy optimality.",
        )
    if ledger_status == "supported":
        return (
            "complete_or_reviewed_supported",
            "existing_ledger_support",
            "close_as_already_supported",
            "Ledger already marks this supported; board records provenance and leaves it alone.",
            "If the support was proxy-based, a coordinator may still ask for sharper phase-4 replication.",
        )
    if next_bead.startswith("t42-br7n."):
        return (
            "active_phase4",
            "needs_generated_or_dynamic_evidence",
            f"keep_open_under_phase4_{next_bead.split('.')[-1]}",
            f"Route to {next_bead}: {PHASE4_CHILDREN[next_bead]}.",
            "No central ledger promotion without the owning phase-4 artifact.",
        )
    if next_bead.startswith("needs_new_"):
        return (
            "needs_new_phase4_scope",
            "partial_or_missing",
            "coordinator_scope_decision",
            "Phase 2/3 evidence exists, but no currently open phase-4 child directly owns the remaining claim.",
            "This is the board's main scope-gap signal.",
        )
    return (
        "needs_triage",
        "unknown",
        "coordinator_triage",
        "No rule matched this claim; inspect source artifacts before closing.",
        "Generated board fallback, not a scientific judgment.",
    )


def load_synthesis_by_family() -> dict[str, dict[str, str]]:
    if not SYNTHESIS_PATH.exists():
        return {}
    return {row["family"]: row for row in read_csv(SYNTHESIS_PATH)}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_phase_artifacts() -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, paths in PHASE_ARTIFACTS.items():
        summaries[name] = {
            "existing": existing_artifacts(paths),
            "missing": [path for path in paths if not artifact_exists(path)],
        }

    for key, path in {
        "auction_phase3_summary": REPO_ROOT / "w42" / "auction_bid_discipline_claim_tests" / "summary.json",
        "eighty_four_seed_mining_summary": REPO_ROOT / "w42" / "eighty_four_seed_mining" / "summary.json",
        "sequence_seat_summary": REPO_ROOT / "w42" / "sequence_seat_counterfactuals" / "summary.json",
        "joined_model_summary": REPO_ROOT / "w42" / "joined_claim_row_model_table" / "summary.json",
        "bid_only_enough_summary": REPO_ROOT / "w42" / "bid_only_enough_claim_tests" / "summary.json",
    }.items():
        data = read_json(path)
        if not data:
            continue
        summaries[key] = {
            "bead": data.get("bead") or data.get("bead_id"),
            "counts": data.get("counts") or data.get("coverage"),
            "headline": data.get("headline") or data.get("effect_direction") or data.get("surface_counts"),
            "wandb": data.get("wandb"),
            "scientific_status": data.get("scientific_status"),
        }
    return summaries


def build_board(out_dir: Path) -> dict[str, Any]:
    ledger_rows = read_csv(LEDGER_PATH)
    matrix_by_id = {row["claim_id"]: row for row in read_csv(MATRIX_PATH)}
    ready_by_id = {row["claim_id"]: row for row in read_csv(READY_PATH)} if READY_PATH.exists() else {}
    synthesis_by_family = load_synthesis_by_family()

    board_rows: list[dict[str, Any]] = []
    for ledger in ledger_rows:
        claim_id = ledger["claim_id"]
        family = ledger["family"]
        matrix = matrix_by_id.get(claim_id, {})
        ready = ready_by_id.get(claim_id, {})

        matrix_class = matrix.get("testability_class", "")
        next_bead = default_next_bead(family, claim_id, ledger["status"], matrix_class)
        completion_bucket, evidence_strength, recommendation, headline, caveat = classify_default(
            family=family,
            claim_id=claim_id,
            ledger_status=ledger["status"],
            evidence_mode=ledger.get("evidence_mode", ""),
            matrix_class=matrix_class,
            next_bead=next_bead,
        )

        override = CLAIM_OVERRIDES.get(claim_id)
        if override:
            completion_bucket = override.get("completion_bucket", completion_bucket)
            evidence_strength = override.get("evidence_strength", evidence_strength)
            next_bead = override.get("next_bead", next_bead)
            recommendation = override.get("phase4_recommendation", recommendation)
            headline = override.get("headline", headline)
            caveat = override.get("caveat", caveat)

        artifact_groups = artifact_group_for_family(family, claim_id)
        evidence_artifacts: list[str] = []
        for group in artifact_groups:
            evidence_artifacts.extend(existing_artifacts(PHASE_ARTIFACTS[group]))

        synthesis = synthesis_by_family.get(family, {})
        if synthesis:
            headline = f"{headline} Phase-2 family synthesis: {synthesis.get('headline', '')}".strip()

        board_rows.append(
            {
                "claim_id": claim_id,
                "family": family,
                "ledger_status": ledger["status"],
                "completion_bucket": completion_bucket,
                "evidence_strength": evidence_strength,
                "phase4_next_bead": next_bead,
                "phase4_recommendation": recommendation,
                "ready_for_powered_test_phase2": ready.get("ready_for_powered_test", matrix.get("ready_for_powered_test", "")),
                "testability_class_phase2": matrix_class,
                "claim": ledger["claim"],
                "headline_evidence": headline,
                "primary_artifacts_existing": "; ".join(dict.fromkeys(evidence_artifacts)),
                "ledger_evidence_artifacts": ledger.get("evidence_artifacts", ""),
                "target_wiki_page_phase2": matrix.get("target_wiki_page", ready.get("target_wiki_page", "")),
                "known_blocker_or_caveat": caveat or ledger.get("caveats", ""),
                "ledger_next_check": ledger.get("next_check", ""),
                "phase2_completion_evidence": matrix.get("completion_evidence", ""),
                "phase2_likely_blockers": matrix.get("likely_blockers", ready.get("likely_blockers", "")),
            }
        )

    fields = [
        "claim_id",
        "family",
        "ledger_status",
        "completion_bucket",
        "evidence_strength",
        "phase4_next_bead",
        "phase4_recommendation",
        "ready_for_powered_test_phase2",
        "testability_class_phase2",
        "claim",
        "headline_evidence",
        "primary_artifacts_existing",
        "ledger_evidence_artifacts",
        "target_wiki_page_phase2",
        "known_blocker_or_caveat",
        "ledger_next_check",
        "phase2_completion_evidence",
        "phase2_likely_blockers",
    ]
    write_csv(out_dir / "completion_board.csv", board_rows, fields)

    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_next: dict[str, Counter[str]] = defaultdict(Counter)
    family_recommendations: dict[str, Counter[str]] = defaultdict(Counter)
    for row in board_rows:
        family_counts[row["family"]][row["completion_bucket"]] += 1
        family_next[row["family"]][row["phase4_next_bead"]] += 1
        family_recommendations[row["family"]][row["phase4_recommendation"]] += 1

    family_rows: list[dict[str, Any]] = []
    for family in sorted(family_counts):
        total = sum(family_counts[family].values())
        family_rows.append(
            {
                "family": family,
                "claim_count": total,
                "completion_buckets": json.dumps(dict(sorted(family_counts[family].items())), sort_keys=True),
                "next_beads": json.dumps(dict(sorted(family_next[family].items())), sort_keys=True),
                "recommendations": json.dumps(dict(sorted(family_recommendations[family].items())), sort_keys=True),
            }
        )
    write_csv(
        out_dir / "family_summary.csv",
        family_rows,
        ["family", "claim_count", "completion_buckets", "next_beads", "recommendations"],
    )

    completion_counts = Counter(row["completion_bucket"] for row in board_rows)
    next_bead_counts = Counter(row["phase4_next_bead"] for row in board_rows)
    recommendation_counts = Counter(row["phase4_recommendation"] for row in board_rows)

    summary = {
        "schema_version": "w42.phase4_claim_completion_board.v1",
        "bead": "t42-br7n.6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "ledger": rel(LEDGER_PATH),
            "matrix": rel(MATRIX_PATH),
            "ready_powered_tests": rel(READY_PATH),
            "synthesis": rel(SYNTHESIS_PATH),
        },
        "outputs": {
            "completion_board": rel(out_dir / "completion_board.csv"),
            "family_summary": rel(out_dir / "family_summary.csv"),
            "summary": rel(out_dir / "summary.json"),
        },
        "claim_count": len(board_rows),
        "ledger_status_counts": dict(sorted(Counter(row["ledger_status"] for row in board_rows).items())),
        "completion_bucket_counts": dict(sorted(completion_counts.items())),
        "next_bead_counts": dict(sorted(next_bead_counts.items())),
        "phase4_recommendation_counts": dict(sorted(recommendation_counts.items())),
        "phase4_children": PHASE4_CHILDREN,
        "artifact_inventory": summarize_phase_artifacts(),
        "headline": {
            "claims_total": len(board_rows),
            "claims_routed_to_open_phase4_children": sum(
                count for bead, count in next_bead_counts.items() if bead.startswith("t42-br7n.")
            ),
            "claims_with_no_open_phase4_scope": next_bead_counts.get("needs_new_bidding_count_exposure_bead", 0),
            "claims_already_complete_or_closeable": sum(
                count
                for bucket, count in completion_counts.items()
                if bucket.startswith("complete_") or bucket == "complete_or_reviewed_supported"
            ),
            "claims_for_ledger_reconciliation_review": next_bead_counts.get("ledger_reconciliation", 0),
        },
        "caveats": [
            "This board is generated from current local artifacts and rule tables; it does not mutate wiki, beads, or the central ledger.",
            "Phase-3 evidence can recommend reconciliation, but broad ledger promotion still needs coordinator review.",
            "Bidding/count-exposure rows are the largest scope gap: phase 3 produced useful evidence, but no current t42-br7n child directly owns all remaining Ch02 refinements.",
            "84 seed mining removes the natural-seed blocker but does not measure late-state action value.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_board(args.out_dir)
    print(json.dumps(summary["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
