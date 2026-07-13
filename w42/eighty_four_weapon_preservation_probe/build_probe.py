#!/usr/bin/env python3
"""Build the phase-2 84 weapon preservation probe artifacts.

This pass is intentionally report-shaped. It does not run new oracle rollouts;
it turns the existing 84 static validation into a direct-label plan, fixture
surface, and branch-aware measurement contract.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "w42" / "eighty_four_weapon_preservation_probe"
SOURCE = ROOT / "w42" / "eighty_four_claim_validation"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = read_json(SOURCE / "summary.json")
    created_at = datetime.now(timezone.utc).isoformat()

    label_spec = [
        {
            "label": "is_84_contract_decision",
            "scope": "state",
            "public_state_safe": "yes",
            "definition": "Contract requires bidder team to win all seven tricks; any opponent trick sets the bid.",
            "positive_evidence": "bid value in 84 ladder and current hand unresolved",
            "metric": "regime precision/recall; downstream bucket denominator",
            "status": "source-backed; needs engine fixture",
        },
        {
            "label": "bidder_final_off_candidate",
            "scope": "bidder hand / report",
            "public_state_safe": "own-hand for bidder; hidden for defenders",
            "definition": "A non-trump, non-double tile that may be saved as the final non-trump winner in an 84 plan.",
            "positive_evidence": "one-off or two-off same-suit 84 candidate shape",
            "metric": "final-off branch count; make-all threshold mass by candidate",
            "status": "static shapes counted; dynamic branch labels missing",
        },
        {
            "label": "defender_live_double_weapon",
            "scope": "defender action",
            "public_state_safe": "own-hand plus public history",
            "definition": "A defender-held double that can still beat at least one plausible bidder final off branch.",
            "positive_evidence": "same-suit final-off branch remains live and the double is not forced dead by bidder protection",
            "metric": "voluntary discard regret; final weapon recall; threshold-mass delta if preserved",
            "status": "static inventory proxy exists; replay labels missing",
        },
        {
            "label": "defender_live_same_suit_pair",
            "scope": "defender hand / action",
            "public_state_safe": "own-hand plus public history",
            "definition": "Two same-suit defender tiles that can answer a bidder double-ahead plan on the final two tricks.",
            "positive_evidence": "lower tile can survive next-to-last pressure and higher tile can beat final off",
            "metric": "pair survival rate; voluntary-break regret; set attribution",
            "status": "static pair availability counted; final-two-trick proof missing",
        },
        {
            "label": "pair_protector",
            "scope": "action-local",
            "public_state_safe": "yes for holder",
            "definition": "A side-suit tile that can absorb forced-follow pressure so a live same-suit pair remains intact.",
            "positive_evidence": "tile shares a vulnerable side suit with pair pressure but is not itself the pair weapon",
            "metric": "protector discard regret; pair survival conditional on protector count",
            "status": "designed; not measured",
        },
        {
            "label": "preservation_opportunity",
            "scope": "legal action set",
            "public_state_safe": "yes",
            "definition": "A free discard or choice point where at least one legal play preserves a live weapon/pair/protector and another legal play spends or breaks it.",
            "positive_evidence": "legal actions have different preservation ranks under the 84 ladder",
            "metric": "threshold-mass delta; tail-risk delta; ladder violation regret",
            "status": "requires replay/action labels",
        },
        {
            "label": "forced_spend",
            "scope": "legal action set",
            "public_state_safe": "yes",
            "definition": "The defender spends or breaks a live weapon because follow-suit legality leaves no preserving alternative.",
            "positive_evidence": "all legal actions consume the same live asset class or no safe alternative exists",
            "metric": "excluded from voluntary-blunder counts; forcedness calibration",
            "status": "requires engine legal-action replay",
        },
        {
            "label": "voluntary_weapon_break",
            "scope": "action-local",
            "public_state_safe": "yes",
            "definition": "A legal play spends a live weapon/pair/protector when another legal play would preserve it.",
            "positive_evidence": "chosen action has worse preservation rank than another legal action",
            "metric": "mean regret; set-tail increase for bidder; claim-specific bucket regret",
            "status": "requires E[Q] or oracle counterfactual labels",
        },
        {
            "label": "dead_asset_abandonment",
            "scope": "action-local",
            "public_state_safe": "yes",
            "definition": "A formerly live weapon/pair/protector is no longer attached to any plausible setting branch and can be released.",
            "positive_evidence": "all watched final-off branches are public-dead or protected against the asset",
            "metric": "correct release rate; regret of over-preserving dead assets",
            "status": "trigger table exists; replay labels missing",
        },
        {
            "label": "hidden_weapon_attribution",
            "scope": "offline evaluation",
            "public_state_safe": "no live feature; offline label only",
            "definition": "Attribution of a make/set branch or PDF shelf to the hidden holder of a weapon, pair, or final-off threat.",
            "positive_evidence": "sampled worlds with hidden ownership and outcome branch labels",
            "metric": "impact-weighted belief calibration; top-k hidden threat recall",
            "status": "needs saved-world E[Q] artifact",
        },
    ]

    fixtures = [
        {
            "fixture_id": "protected_one_off_bidder_shape",
            "source": "w42/eighty_four_claim_validation/example_cases.csv",
            "hand_or_state": "['0-0', '1-0', '1-1', '2-0', '2-1', '2-2', '3-0']; trump blanks; off 2-1 protected by double ahead",
            "expected_labels": "is_84_contract_decision; bidder_final_off_candidate; protected-off branch; defender pair path needed",
            "why_it_matters": "Scalar hand strength is not enough; the preservation question is whether defenders can carry same-suit pair pressure to the last two tricks.",
            "current_status": "static shape only",
        },
        {
            "fixture_id": "straight_off_named_double_threat",
            "source": "w42/eighty_four_claim_validation/example_cases.csv",
            "hand_or_state": "['0-0', '1-0', '1-1', '2-0', '2-2', '3-0', '4-3']; trump blanks; straight off 4-3",
            "expected_labels": "bidder_final_off_candidate; defender_live_double_weapon; hidden_weapon_attribution",
            "why_it_matters": "Prior validation showed a named matching double is opponent-team owned with probability 66.667%; preservation quality depends on carrying it to the last trick.",
            "current_status": "ownership odds counted; carry-to-last-trick unmeasured",
        },
        {
            "fixture_id": "two_off_same_suit_ordering",
            "source": "w42/eighty_four_claim_validation/example_cases.csv",
            "hand_or_state": "['0-0', '1-0', '1-1', '2-0', '2-1', '2-2', '3-0']; trump ones; offs 2-0 and 3-0 share suit",
            "expected_labels": "bidder_final_off_candidate; final-off ordering branch; defender_live_double_weapon",
            "why_it_matters": "The relevant branch is not only whether a defender has a weapon, but which off remains for the final trick after same-suit pressure.",
            "current_status": "static bucket counted; order/value counterfactual missing",
        },
        {
            "fixture_id": "defender_pair_protector_choice",
            "source": "Winning42 Ch08 concept table",
            "hand_or_state": "defender owns a same-suit pair plus side-suit protector during 84 defense with a free discard",
            "expected_labels": "defender_live_same_suit_pair; pair_protector; preservation_opportunity; voluntary_weapon_break if protector is discarded",
            "why_it_matters": "The book ranks pair protectors above partner-readable low discards; this is a direct action-label test.",
            "current_status": "designed fixture; needs constructed state",
        },
        {
            "fixture_id": "dead_double_release",
            "source": "w42/eighty_four_claim_validation/abandonment_trigger_table.csv",
            "hand_or_state": "saved double whose watched final-off suit has become public-dead",
            "expected_labels": "dead_asset_abandonment; no voluntary_weapon_break if released",
            "why_it_matters": "A good policy must distinguish preservation from superstition; holding a dead weapon can become a cost.",
            "current_status": "trigger designed; replay proof missing",
        },
    ]

    measurement_axes = [
        {
            "axis": "scalar_value",
            "question": "What is the mean E[Q] of preserving versus spending the weapon?",
            "primary_metric": "paired mean E[Q] delta",
            "why_mean_is_not_enough": "A moderate mean can hide make-all and set branches.",
        },
        {
            "axis": "threshold_mass",
            "question": "How does the action change all-seven-tricks make/set mass?",
            "primary_metric": "make84_mass_delta or set84_mass_delta",
            "why_mean_is_not_enough": "84 is threshold-dominated: one trick flips the contract.",
        },
        {
            "axis": "tail_risk",
            "question": "Does spending a weapon create a disaster tail even when mean remains close?",
            "primary_metric": "lower-tail mass / CVaR-style set branch",
            "why_mean_is_not_enough": "The book advice is often about bracing for the bad branch.",
        },
        {
            "axis": "hidden_threat_attribution",
            "question": "Which hidden holder/weapon explains the branch?",
            "primary_metric": "impact-weighted belief calibration",
            "why_mean_is_not_enough": "Belief should focus on high-impact threats, not average ownership accuracy.",
        },
        {
            "axis": "forcedness",
            "question": "Was the weapon break voluntary or forced by follow-suit?",
            "primary_metric": "voluntary break regret excluding forced spend",
            "why_mean_is_not_enough": "A model should not be punished for unavoidable losses.",
        },
    ]

    artifacts = {
        "label_spec_csv": "w42/eighty_four_weapon_preservation_probe/label_spec.csv",
        "fixture_cases_csv": "w42/eighty_four_weapon_preservation_probe/fixture_cases.csv",
        "measurement_axes_csv": "w42/eighty_four_weapon_preservation_probe/measurement_axes.csv",
        "summary_json": "w42/eighty_four_weapon_preservation_probe/summary.json",
        "manifest_json": "w42/eighty_four_weapon_preservation_probe/manifest.json",
    }

    summary = {
        "schema_version": "w42.eighty_four_weapon_preservation_probe.v1",
        "owner_bead": "t42-5m82.6",
        "created_at": created_at,
        "commit_sha": git_sha(),
        "evidence_mode": "label and fixture design from existing static 84 validation; no new oracle rollout",
        "prior_static_validation": {
            "source_summary": "w42/eighty_four_claim_validation/summary.json",
            "headline_metrics": prior["headline_metrics"],
            "claim_statuses": prior["claim_statuses"],
        },
        "probe_status": "design-ready; dynamic regret and E[Q] PDF labels not yet generated",
        "claim_ledger_impact": "no central claim-ledger change; this bead defines direct labels and next measurements",
        "wandb": "not applicable: report/spec-only, no training run",
        "huggingface": "not applicable",
        "artifacts": artifacts,
    }

    manifest = {
        "schema_version": "w42.artifact_manifest.v1",
        "dataset_id": "w42-eighty-four-weapon-preservation-probe",
        "created_at": created_at,
        "owner_bead": "t42-5m82.6",
        "source_inputs": [
            "wiki/experiments/winning42-ch07-taking-every-trick-84.md",
            "wiki/experiments/winning42-ch08-setting-84.md",
            "wiki/experiments/w42-84-claim-validation.md",
            "w42/eighty_four_claim_validation/summary.json",
            "w42/eighty_four_claim_validation/example_cases.csv",
            "w42/eighty_four_claim_validation/abandonment_trigger_table.csv",
        ],
        "outputs": artifacts,
        "leakage_boundary": "hidden ownership labels are offline diagnostics only; live features must use public state or learned beliefs",
        "exact_command": "python w42/eighty_four_weapon_preservation_probe/build_probe.py",
    }

    write_csv(OUT / "label_spec.csv", label_spec)
    write_csv(OUT / "fixture_cases.csv", fixtures)
    write_csv(OUT / "measurement_axes.csv", measurement_axes)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
