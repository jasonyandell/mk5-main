#!/usr/bin/env python3
"""Build the w42 claim-analysis test-design matrix.

This script is intentionally conservative. It does not decide that a book claim
is true; it classifies what kind of evidence would be needed to test it next.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "w42" / "statistics_claims_ledger" / "claims.csv"
OUT_DIR = REPO_ROOT / "w42" / "claim_analysis_matrix"
MATRIX_PATH = OUT_DIR / "claim_analysis_matrix.csv"
FAMILY_ROLLUP_PATH = OUT_DIR / "family_rollup.csv"
READY_QUEUE_PATH = OUT_DIR / "ready_powered_tests.csv"
SUMMARY_PATH = OUT_DIR / "summary.json"
MANIFEST_PATH = OUT_DIR / "manifest.json"


FIELDNAMES = [
    "claim_id",
    "family",
    "current_status",
    "claim",
    "current_evidence_mode",
    "current_readiness",
    "testability_class",
    "proposed_operational_label",
    "operational_definition",
    "primary_data_source",
    "required_fields",
    "leakage_risks",
    "likely_blockers",
    "estimated_sample_power_need",
    "power_bucket",
    "ready_for_powered_test",
    "next_experiment_bead",
    "target_wiki_page",
    "completion_evidence",
    "notes",
]


FAMILY_WIKI = {
    "84 bidder/defender": "wiki/experiments/w42-84-claim-validation.md; wiki/experiments/w42-phase2-84-weapon-preservation-probe.md",
    "bidder sequencing": "wiki/experiments/w42-bidder-sequencing-claim-validation.md; wiki/experiments/w42-phase2-seat-position-strategy-map.md",
    "bidding risk": "wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
    "count exposure": "wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
    "double-count odds": "wiki/experiments/w42-odds-ruleset-claim-validation.md; wiki/experiments/w42-phase2-statistics-claims-ledger.md",
    "doubles/no-trump odds": "wiki/experiments/w42-doubles-no-trump-claim-validation.md",
    "doubles/no-trump rules": "wiki/experiments/w42-doubles-no-trump-claim-validation.md",
    "four-trump odds": "wiki/experiments/w42-odds-ruleset-claim-validation.md; wiki/experiments/winning42-ch16-statistical-odds.md",
    "four-trump tactics": "wiki/experiments/w42-bidding-risk-budget-claim-validation.md; wiki/experiments/winning42-ch16-statistical-odds.md",
    "hand odds": "wiki/experiments/w42-odds-ruleset-claim-validation.md; wiki/experiments/winning42-ch16-statistical-odds.md",
    "partner support": "wiki/experiments/w42-partner-support-claim-validation.md; wiki/experiments/w42-gus-corpus-tactical-claim-deep-dive.md",
    "partner-help odds": "wiki/experiments/w42-odds-ruleset-claim-validation.md; wiki/experiments/winning42-ch16-statistical-odds.md",
    "scoring objective": "wiki/experiments/w42-scoring-objective-drift-claim-validation.md",
    "setter defense": "wiki/experiments/w42-setter-defense-claim-validation.md; wiki/experiments/w42-gus-corpus-tactical-claim-deep-dive.md",
    "stopper ownership": "wiki/experiments/w42-84-claim-validation.md; wiki/experiments/w42-phase2-84-weapon-preservation-probe.md",
    "trump counts": "wiki/experiments/w42-odds-ruleset-claim-validation.md; wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
    "void frequencies": "wiki/experiments/w42-odds-ruleset-claim-validation.md; wiki/experiments/winning42-ch16-statistical-odds.md",
}


FAMILY_BEADS = {
    "84 bidder/defender": "t42-0b4l.7",
    "bidder sequencing": "t42-0b4l.6; t42-0b4l.9",
    "bidding risk": "t42-0b4l.5",
    "count exposure": "t42-0b4l.5",
    "double-count odds": "fixture; no powered bead",
    "doubles/no-trump odds": "t42-0b4l.8",
    "doubles/no-trump rules": "t42-0b4l.8",
    "four-trump odds": "t42-0b4l.5",
    "four-trump tactics": "t42-0b4l.5",
    "hand odds": "fixture; no powered bead",
    "partner support": "t42-0b4l.3; t42-0b4l.6",
    "partner-help odds": "t42-0b4l.5",
    "scoring objective": "t42-0b4l.10; future tournament-simulation bead",
    "setter defense": "t42-0b4l.3; t42-0b4l.6",
    "stopper ownership": "t42-0b4l.7",
    "trump counts": "t42-0b4l.5",
    "void frequencies": "fixture; no powered bead",
}


PRIMARY_DATA = {
    "84 bidder/defender": "w42/eighty_four_claim_validation/; w42/eighty_four_weapon_preservation_probe/; future 84-specific branch-atlas or rollout corpus",
    "bidder sequencing": "w42/bidder_sequencing_claim_validation/; w42/phase2_decision_table/; gus/data/corpus_v2_train_*_d0-9.pt",
    "bidding risk": "w42/bidding_risk_budget_claim_validation/; future auction/bid-margin corpus",
    "count exposure": "w42/bidding_risk_budget_claim_validation/; future auction-aware E[Q] or rollout corpus",
    "double-count odds": "w42/odds_ruleset_claim_validation/",
    "doubles/no-trump odds": "w42/doubles_no_trump_claim_validation/; w42/branch_atlas_scaled_v0/",
    "doubles/no-trump rules": "w42/doubles_no_trump_claim_validation/; ruleset code fixtures",
    "four-trump odds": "w42/odds_ruleset_claim_validation/; future paired opening-line rollout corpus",
    "four-trump tactics": "future paired boss-first versus low-trump rollout corpus",
    "hand odds": "w42/odds_ruleset_claim_validation/",
    "partner support": "w42/gus_corpus_claim_deep_dive/; gus/data/corpus_v2_train_*_d0-9.pt",
    "partner-help odds": "w42/odds_ruleset_claim_validation/; future conditional partner-help simulation",
    "scoring objective": "w42/scoring_objective_drift_claim_validation/; future tournament simulation",
    "setter defense": "w42/gus_corpus_claim_deep_dive/; gus/data/corpus_v2_train_*_d0-9.pt; w42/setter_pounce_direct_label_probe/",
    "stopper ownership": "w42/eighty_four_claim_validation/; future 84 branch-atlas or rollout corpus",
    "trump counts": "w42/odds_ruleset_claim_validation/; w42/bidding_risk_budget_claim_validation/",
    "void frequencies": "w42/odds_ruleset_claim_validation/",
}


REQUIRED_FIELDS = {
    "84 bidder/defender": "84 contract flag; score; bidder hand and offs; full public trick history; stopper candidates; live/dead asset state; final-trick action labels; q_per_world/world_hands for offline attribution",
    "bidder sequencing": "bidder role; lead/follow phase; trump/off inventory; reentry tiles; legal action alternatives; public trick history; E[Q] PDF or rollout labels",
    "bidding risk": "auction history; bid amount; bid margin; candidate declaration; score; hand exposure; make/set outcome or per-bid E[Q]",
    "count exposure": "candidate trump; side-specific off exposure; duplicate count identities; double-ahead protection; bid amount; make/set or regret labels",
    "double-count odds": "standard double-six tile set; seven-card hand enumeration",
    "doubles/no-trump odds": "hand doubles; declaration regime; candidate regime alternatives; legal actions; paired regime rollout or E[Q] labels",
    "doubles/no-trump rules": "ruleset declaration; suit-membership and follow-suit predicates; legal action fixtures",
    "four-trump odds": "trump declaration; missing trump identities; holder assignment model; opening action alternatives; bid margin for tactical tests",
    "four-trump tactics": "opening-lead alternatives; trump hierarchy; off protection; score; bid margin; paired E[Q] rollout labels",
    "hand odds": "standard double-six tile set; seven-card hand enumeration",
    "partner support": "role; current trick; current winner; legal actions; count value; partner/offense control; later-seat overtake risk; guaranteed-win label; q_per_world",
    "partner-help odds": "bidder hand/off structure; partner hand distribution; conditional helper-double ownership; make/set or regret labels",
    "scoring objective": "terminal hand points; bid/contract; mark transform; match/tournament schedule for skill, speed, or advancement claims",
    "setter defense": "defender role; current trick; offense/defense current winner; count value; legal beating action; set threshold; bid_value; later-seat risk",
    "stopper ownership": "84/off suit context; named stopper candidates; any-stopper set; hidden holder labels for offline checks; public live/dead evidence",
    "trump counts": "candidate trump; hand trump count; auction context; score; make/set or bid-quality labels for strategic claims",
    "void frequencies": "standard double-six tile set; seven-card hand enumeration; generated-deal manifests for audit",
}


LEAKAGE_RISKS = {
    "84 bidder/defender": "Hidden stopper ownership and future asset death are report labels only; live detectors may use only public play evidence and the actor hand.",
    "bidder sequencing": "Future trick outcomes and oracle Q are labels, not features; model probes must use public state plus actor hand only.",
    "bidding risk": "Do not feed make/set outcome, true hidden partner help, or per-bid oracle result as a live bidding feature.",
    "count exposure": "Duplicate/off risk can be public-hand arithmetic; hidden holder and future capture are report labels only.",
    "double-count odds": "No live-model leakage; fixture is exact public math.",
    "doubles/no-trump odds": "Regime choice tests must not use counterfactual outcomes as features.",
    "doubles/no-trump rules": "No leakage for ruleset fixtures; strategy tests still need public-only features.",
    "four-trump odds": "Holder assignment is an offline prior; action choice cannot use true missing-trump owners.",
    "four-trump tactics": "Paired rollout outcomes and hidden holders are labels only.",
    "hand odds": "No live-model leakage; fixture is exact public math.",
    "partner support": "Partner/opponent private hands, guaranteed future win, and q_per_world are labels only; live tags must be public-context or learned-belief derived.",
    "partner-help odds": "True partner hand content is an offline label unless inferred from legal public evidence.",
    "scoring objective": "Terminal and tournament outcomes are evaluation labels, not decision-time features.",
    "setter defense": "Hidden holdings, future trick result, and q_per_world are offline labels only; pounce window must be public/action-local.",
    "stopper ownership": "True hidden stopper owners are offline diagnostics; live play can only use public evidence/beliefs.",
    "trump counts": "Trump count in own hand is legal; partner/opponent trump ownership is report-only unless belief-derived.",
    "void frequencies": "No live-model leakage; fixture is exact public math.",
}


POWER_NEEDS = {
    "84 bidder/defender": ("new_dynamic_data", "Generate or mine 84-specific decisions; target >=1000 candidate 84 decision states and >=200 paired stopper/asset-preservation contrasts per label."),
    "bidder sequencing": ("powered_slice", "Use Gus/branch-atlas decisions first, then generate paired trump-first/off-first rollouts; target >=500 same-decision contrasts per sequencing label."),
    "bidding risk": ("new_auction_data", "Needs auction/bid-margin corpus; target multi-seed bid candidates with >=1000 make/set labels per bid-margin bucket."),
    "count exposure": ("new_auction_data", "Needs auction-aware outcome labels; target >=1000 candidate declarations split by exposure bucket and bid margin."),
    "double-count odds": ("fixture_only", "Exact enumeration; no statistical power needed beyond regression fixture checks."),
    "doubles/no-trump odds": ("new_regime_data", "Generate paired doubles-trump/no-trump regime decisions; target >=1000 candidate regime states and declaration-specific CIs."),
    "doubles/no-trump rules": ("fixture_then_regime_data", "Rules are deterministic fixtures; tactical claims need paired regime data as above."),
    "four-trump odds": ("fixture_plus_rollout", "Exact assignment is fixed; tactical boss-first claims need paired opening-line rollouts by bid margin and score."),
    "four-trump tactics": ("new_rollout_data", "Needs paired boss-first/low-trump E[Q] rollouts; target >=500 matched four-trump opening states."),
    "hand odds": ("fixture_only", "Exact enumeration; no statistical power needed beyond generated-corpus audit."),
    "partner support": ("powered_slice", "Reuse Gus v2 corpus; target >=500 same-decision pairs per donation/support label and explicit later-seat safety slices."),
    "partner-help odds": ("conditional_sim", "Needs conditional partner-help simulation; target >=1000 bidder hand shapes per off/double bucket."),
    "scoring objective": ("simulation_needed", "Deterministic transforms are complete; skill/speed/advancement claims need policy-population tournament simulation with match-level CIs."),
    "setter defense": ("powered_slice", "Reuse Gus v2 corpus; target >=500 same-decision pairs per pounce/count label plus bid-margin/high-bid slices."),
    "stopper ownership": ("new_dynamic_data", "Needs 84-specific final-off/stopper states; target >=500 paired preserve/break/abandon contrasts."),
    "trump counts": ("fixture_plus_auction", "Exact trump-count priors are fixed; bidding-quality claims need auction/bid-margin data."),
    "void frequencies": ("fixture_only", "Exact enumeration; optional generated-corpus chi-square audit."),
}


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def snake_label(claim_id: str) -> str:
    return claim_id.replace("-", "_")


def testability_class(row: dict[str, str]) -> str:
    family = row["family"]
    mode = row["evidence_mode"]
    claim_id = row["claim_id"]
    status = row["status"]

    if family in {"hand odds", "void frequencies", "double-count odds"}:
        return "deterministic_static_fixture"
    if family in {"doubles/no-trump rules"}:
        return "deterministic_ruleset_fixture"
    if mode == "deterministic scoring transform":
        if claim_id in {"ch10-point-system-skill-signal", "ch10-tournament-speed-tradeoff", "ch10-timed-marks-advancement-objective"}:
            return "policy_population_simulation_needed"
        return "deterministic_objective_fixture"
    if mode == "direct role-gated corpus contrast":
        if status == "supported":
            return "direct_action_contrast_ready_for_replication"
        return "direct_action_contrast_needs_stronger_labels"
    if family in {"bidding risk", "count exposure", "trump counts", "four-trump odds", "four-trump tactics", "partner-help odds"}:
        if "exact" in mode and status == "supported":
            return "static_substrate_fixture_plus_auction_followup"
        return "auction_or_bid_margin_data_needed"
    if family in {"84 bidder/defender", "stopper ownership"}:
        if status == "contradicted":
            return "static_substrate_contradiction_then_dynamic_84_followup"
        return "static_84_substrate_needs_dynamic_rollout"
    if family in {"doubles/no-trump odds"}:
        return "static_regime_substrate_needs_paired_regime_rollout"
    if family in {"partner support", "setter defense"}:
        return "direct_detector_or_paired_contrast_needed"
    if family == "bidder sequencing":
        return "sequence_counterfactual_or_model_probe_needed"
    return "manual_review_needed"


def ready_state(row: dict[str, str], klass: str) -> str:
    if klass in {
        "deterministic_static_fixture",
        "deterministic_ruleset_fixture",
        "deterministic_objective_fixture",
        "static_substrate_fixture_plus_auction_followup",
    }:
        return "fixture_only"
    if klass == "direct_action_contrast_ready_for_replication":
        return "yes_after_harness"
    if klass == "direct_action_contrast_needs_stronger_labels":
        return "needs_label_refinement"
    if "auction" in klass or "bid_margin" in klass:
        return "needs_generation"
    if "dynamic_84" in klass or row["family"] in {"84 bidder/defender", "stopper ownership"}:
        return "needs_84_generation"
    if "paired_regime" in klass:
        return "needs_regime_generation"
    if "direct_detector" in klass:
        return "needs_detector_implementation"
    if "sequence_counterfactual" in klass:
        return "needs_sequence_counterfactuals"
    if "model_probe" in klass:
        return "needs_detector_features"
    if "simulation" in klass:
        return "needs_simulation_design"
    return "needs_review"


def blocker(row: dict[str, str], klass: str) -> str:
    family = row["family"]
    status = row.get("status", row.get("current_status", ""))
    if status == "supported" and klass.startswith("deterministic"):
        return "No blocker for fixture use; blocker only applies if promoted into policy advice."
    if family in {"bidding risk", "count exposure", "trump counts", "four-trump tactics", "four-trump odds", "partner-help odds"}:
        return "Real bid margin and auction counterfactuals are missing from current fixed-bid artifacts."
    if family in {"partner support", "setter defense"}:
        return "Needs the reusable harness plus stronger forcedness, guaranteed-control, later-seat-risk, and bid-margin slices."
    if family in {"84 bidder/defender", "stopper ownership"}:
        return "Needs 84-specific dynamic states; static ownership/asset counts do not prove preservation or abandonment timing."
    if family in {"doubles/no-trump odds", "doubles/no-trump rules"}:
        return "Ruleset facts exist, but declaration choice and tactical play need paired regime rollouts."
    if family == "bidder sequencing":
        return "Current evidence is tag/model-bucket proxy; no trump-first/off-first sequence counterfactual exists."
    if family == "scoring objective":
        return "Objective transforms are deterministic; claims about skill/speed/advancement need policy-population tournament data."
    return "No known blocker beyond packaging as a stable fixture."


def completion_evidence(row: dict[str, str], klass: str) -> str:
    family = row["family"]
    if klass.startswith("deterministic") or "fixture" in klass:
        return "Regenerate exact fixture, validate row count/percentages, and keep as report-only support unless a separate policy test exists."
    if family in {"partner support", "setter defense"}:
        return "Paired same-decision contrasts with bootstrap CIs, declaration/seat/trick slices, example decisions, W&B progress series, and conservative ledger update."
    if family in {"bidding risk", "count exposure", "trump counts", "four-trump odds", "four-trump tactics", "partner-help odds"}:
        return "Auction-aware or paired bid-margin dataset with make/set or regret CIs by risk bucket and explicit separation of static hand strength from bid strategy."
    if family in {"84 bidder/defender", "stopper ownership"}:
        return "84-specific paired preserve/break/abandon tests with stopper survival labels, hidden-holder diagnostics, public-evidence examples, and CIs."
    if family in {"doubles/no-trump odds", "doubles/no-trump rules"}:
        return "Paired doubles-trump/no-trump declaration or play rollouts with regime-specific slices and legal-rule fixtures unchanged."
    if family == "bidder sequencing":
        return "Sequence counterfactuals or proof-checker labels plus ablated w42 tag probes that show whether tags improve learning and where they fail."
    if family == "scoring objective":
        return "Tournament or match simulation showing skill/speed/advancement deltas with policy-population and schedule assumptions."
    return "Manual evidence plan required before status movement."


def operational_definition(row: dict[str, str]) -> str:
    parts = []
    if row.get("measurement"):
        parts.append(row["measurement"].strip())
    if row.get("next_check"):
        parts.append("Next test: " + row["next_check"].strip())
    return " ".join(parts)


def build_row(row: dict[str, str]) -> dict[str, str]:
    family = row["family"]
    klass = testability_class(row)
    power_bucket, power_need = POWER_NEEDS.get(family, ("manual_review", "Manual power estimate needed."))
    return {
        "claim_id": row["claim_id"],
        "family": family,
        "current_status": row["status"],
        "claim": row["claim"],
        "current_evidence_mode": row["evidence_mode"],
        "current_readiness": row["readiness"],
        "testability_class": klass,
        "proposed_operational_label": snake_label(row["claim_id"]),
        "operational_definition": operational_definition(row),
        "primary_data_source": PRIMARY_DATA.get(family, "manual review needed"),
        "required_fields": REQUIRED_FIELDS.get(family, "manual review needed"),
        "leakage_risks": LEAKAGE_RISKS.get(family, "manual review needed"),
        "likely_blockers": blocker(row, klass),
        "estimated_sample_power_need": power_need,
        "power_bucket": power_bucket,
        "ready_for_powered_test": ready_state(row, klass),
        "next_experiment_bead": FAMILY_BEADS.get(family, "manual review needed"),
        "target_wiki_page": FAMILY_WIKI.get(family, "wiki/experiments/w42-claim-analysis-matrix.md"),
        "completion_evidence": completion_evidence(row, klass),
        "notes": "Current ledger result: " + row["result"],
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)


def main() -> None:
    with LEDGER_PATH.open(newline="") as f:
        ledger_rows = list(csv.DictReader(f))

    matrix_rows = [build_row(row) for row in ledger_rows]
    write_csv(MATRIX_PATH, matrix_rows, FIELDNAMES)

    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in matrix_rows:
        by_family[row["family"]].append(row)

    family_fieldnames = [
        "family",
        "claims",
        "current_status_counts",
        "testability_class_counts",
        "ready_state_counts",
        "next_experiment_bead",
        "target_wiki_page",
        "primary_data_source",
        "main_blocker",
    ]
    family_rows = []
    for family, rows in sorted(by_family.items()):
        family_rows.append(
            {
                "family": family,
                "claims": str(len(rows)),
                "current_status_counts": json.dumps(Counter(r["current_status"] for r in rows), sort_keys=True),
                "testability_class_counts": json.dumps(Counter(r["testability_class"] for r in rows), sort_keys=True),
                "ready_state_counts": json.dumps(Counter(r["ready_for_powered_test"] for r in rows), sort_keys=True),
                "next_experiment_bead": FAMILY_BEADS.get(family, "manual review needed"),
                "target_wiki_page": FAMILY_WIKI.get(family, "wiki/experiments/w42-claim-analysis-matrix.md"),
                "primary_data_source": PRIMARY_DATA.get(family, "manual review needed"),
                "main_blocker": blocker(rows[0], rows[0]["testability_class"]),
            }
        )
    write_csv(FAMILY_ROLLUP_PATH, family_rows, family_fieldnames)

    ready_rows = [
        row
        for row in matrix_rows
        if row["ready_for_powered_test"]
        in {
            "yes_after_harness",
            "needs_label_refinement",
            "needs_generation",
            "needs_84_generation",
            "needs_regime_generation",
            "needs_detector_implementation",
            "needs_sequence_counterfactuals",
        }
    ]
    ready_fieldnames = [
        "claim_id",
        "family",
        "current_status",
        "testability_class",
        "ready_for_powered_test",
        "next_experiment_bead",
        "estimated_sample_power_need",
        "likely_blockers",
        "target_wiki_page",
    ]
    write_csv(READY_QUEUE_PATH, ready_rows, ready_fieldnames)

    summary = {
        "schema_version": "w42.claim_analysis_matrix.v1",
        "bead": "t42-0b4l.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ledger_path": str(LEDGER_PATH.relative_to(REPO_ROOT)),
        "matrix_path": str(MATRIX_PATH.relative_to(REPO_ROOT)),
        "family_rollup_path": str(FAMILY_ROLLUP_PATH.relative_to(REPO_ROOT)),
        "ready_queue_path": str(READY_QUEUE_PATH.relative_to(REPO_ROOT)),
        "row_count": len(matrix_rows),
        "family_count": len(by_family),
        "status_counts": Counter(row["current_status"] for row in matrix_rows),
        "testability_class_counts": Counter(row["testability_class"] for row in matrix_rows),
        "ready_state_counts": Counter(row["ready_for_powered_test"] for row in matrix_rows),
        "power_bucket_counts": Counter(row["power_bucket"] for row in matrix_rows),
        "next_experiment_bead_counts": Counter(row["next_experiment_bead"] for row in matrix_rows),
        "claim_ledger_impact": "No status changes. This matrix classifies evidence needs and routes claims to future beads.",
        "scientific_status": {
            "training_run": False,
            "wandb": "not applicable; matrix/report generation only",
            "hf": "not applicable",
            "leakage_boundary": "Hidden-world truth and oracle outcomes appear only in required-fields/leakage notes as offline labels, never as live features.",
            "interpretation": "Planning artifact for powered claim tests; not evidence that additional claims are supported.",
        },
        "ready_powered_queue_size": len(ready_rows),
        "ready_powered_queue_claims": [row["claim_id"] for row in ready_rows],
    }
    with SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    manifest = {
        "schema_version": "w42.claim_analysis_matrix.manifest.v1",
        "bead": "t42-0b4l.1",
        "created_at_utc": summary["created_at_utc"],
        "repo_commit": run_git(["rev-parse", "HEAD"]),
        "git_status_short": run_git(["status", "--short"]),
        "inputs": [str(LEDGER_PATH.relative_to(REPO_ROOT))],
        "outputs": [
            str(MATRIX_PATH.relative_to(REPO_ROOT)),
            str(FAMILY_ROLLUP_PATH.relative_to(REPO_ROOT)),
            str(READY_QUEUE_PATH.relative_to(REPO_ROOT)),
            str(SUMMARY_PATH.relative_to(REPO_ROOT)),
            str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        ],
        "command": "python w42/claim_analysis_matrix/build_claim_analysis_matrix.py",
        "notes": [
            "Generated from the phase-2 statistics claims ledger.",
            "Rows classify test design and blockers; they do not move claim statuses.",
        ],
    }
    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(json.dumps({"rows": len(matrix_rows), "families": len(by_family), "ready_queue": len(ready_rows)}, indent=2))


if __name__ == "__main__":
    main()
