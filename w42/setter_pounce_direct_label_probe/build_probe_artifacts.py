#!/usr/bin/env python3
"""Build first-pass setter-pounce direct-label probe artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def label_spec() -> dict[str, Any]:
    return {
        "schema_version": "w42.setter_pounce_direct_label_probe.spec.v1",
        "bead_id": "t42-5m82.5",
        "scope": "first-pass direct labels for setter pounce and count-to-set windows",
        "live_feature_boundary": {
            "public_state_allowed": [
                "current player seat and team relation to bidder team",
                "public auction winner, bid amount, declaration, and ruleset metadata",
                "current hand/trick score visible in the public state",
                "public trick history including played tiles, trick winners, captured count, and public void evidence",
                "current trick position and current public winning seat",
                "current player's hand",
                "legal action set and action-local tile facts",
            ],
            "hidden_state_forbidden_live": [
                "unplayed domino owner truth outside current player's hand",
                "partner's hidden winning tile availability",
                "future rollout outcomes",
                "forge E[Q], E[Q] PDF, sampled-world outcome branches, or completed-hand set attribution",
            ],
            "offline_label_allowed": [
                "forge E[Q] and E[Q] PDF deltas",
                "sampled-world make/set threshold mass",
                "hidden-domino threat attribution",
                "completed-hand or rollout set attribution",
            ],
        },
        "direct_labels": [
            {
                "name": "is_setter_decision",
                "level": "decision",
                "live_safe": True,
                "definition": "current_player_team != bidder_team",
                "required_fields": ["current_player_seat", "bidder_seat"],
                "positive_condition": "The acting player is on the defending team.",
                "negative_condition": "The acting player is on the bidder team.",
                "anti_leakage": "Do not infer bidder team from final score or rollout result.",
            },
            {
                "name": "contract_context",
                "level": "decision",
                "live_safe": True,
                "definition": "normalized bid, declaration, ruleset, and points already captured by bidder team and defender team",
                "required_fields": [
                    "bid_amount",
                    "bidder_seat",
                    "declaration",
                    "ruleset_id",
                    "bidder_team_points_so_far",
                    "defender_team_points_so_far",
                ],
                "positive_condition": "Contract fields are present and internally consistent.",
                "negative_condition": "Unknown bidder, bid amount, declaration, or captured points.",
                "anti_leakage": "Use only points captured before the candidate action, not terminal hand totals.",
            },
            {
                "name": "count_to_set_window",
                "level": "decision",
                "live_safe": True,
                "definition": "A public threshold window where candidate count can bring the bidder team to or below set threshold, or materially reduce the remaining allowance.",
                "required_fields": [
                    "bid_amount",
                    "bidder_team_points_so_far",
                    "defender_team_points_so_far",
                    "current_trick_count_points",
                    "candidate_count_points",
                    "contract_point_target",
                ],
                "derived_fields": [
                    "bidder_points_needed_to_make = max(0, contract_point_target - bidder_team_points_so_far)",
                    "defender_points_needed_to_set = max(0, contract_point_target - bidder_team_points_so_far - current_public_defender_trick_points)",
                    "candidate_crosses_set_threshold = candidate_count_points >= defender_points_needed_to_set",
                ],
                "positive_condition": "Candidate count can cross the public set threshold or move the hand into a one-count-away set state.",
                "negative_condition": "Candidate count cannot affect the public make/set threshold under current contract accounting.",
                "anti_leakage": "Do not use hidden count still held by other players as live availability; only use candidate action and public trick count.",
            },
            {
                "name": "bidder_off_window",
                "level": "decision",
                "live_safe": True,
                "definition": "The current trick is in a non-trump/non-declaration suit that is publicly vulnerable for bidder team and can be won by a defender or defender's partner.",
                "required_fields": [
                    "declaration",
                    "led_suit",
                    "current_public_winning_seat",
                    "current_trick_position",
                    "public_voids_by_seat_suit",
                    "legal_action_set",
                    "bidder_seat",
                ],
                "positive_condition": "A defender is acting while bidder team is exposed on an off-suit trick and public void/legal-action evidence makes a defender win or partner win plausible.",
                "negative_condition": "The trick is trump/declaration control, the acting player is forced into non-count follow with no count choice, or public state gives no bidder-off exposure.",
                "anti_leakage": "Do not mark positive only because a hidden defender tile would win; the live label must be explainable from public voids, current winner, and legal actions.",
            },
            {
                "name": "count_before_certainty",
                "level": "action",
                "live_safe": True,
                "definition": "A candidate count action by a setter before the final trick winner is publicly certain.",
                "required_fields": [
                    "current_trick_position",
                    "current_public_winning_seat",
                    "candidate_tile",
                    "candidate_count_points",
                    "remaining_players_to_act",
                    "public_winning_certainty",
                ],
                "positive_condition": "candidate_count_points > 0 and at least one later player can legally change the current winner under public state.",
                "negative_condition": "Candidate is not count, the trick winner is already public-certain, or the action is not by a setter.",
                "anti_leakage": "Public certainty may use legal-action constraints and public voids, but not hidden later-player hands.",
            },
            {
                "name": "setter_pounce_window",
                "level": "decision",
                "live_safe": True,
                "definition": "is_setter_decision and bidder_off_window and count_to_set_window with at least one legal count action available to the setter.",
                "required_fields": [
                    "is_setter_decision",
                    "bidder_off_window",
                    "count_to_set_window",
                    "legal_action_set",
                    "candidate_count_points",
                ],
                "positive_condition": "All three public gates hold and at least one legal action carries count.",
                "negative_condition": "Any gate is absent, or only non-count legal actions exist.",
                "anti_leakage": "This is a window label, not a claim that every count action is optimal.",
            },
            {
                "name": "setter_pounce_action",
                "level": "action",
                "live_safe": True,
                "definition": "Legal candidate count action inside setter_pounce_window.",
                "required_fields": ["setter_pounce_window", "candidate_count_points", "legal_action_set"],
                "positive_condition": "Action is legal and candidate_count_points > 0 inside a setter_pounce_window.",
                "negative_condition": "Window is absent or action has no count.",
                "anti_leakage": "Do not use oracle regret to decide whether an action receives this structural label.",
            },
            {
                "name": "threshold_mass_delta",
                "level": "action",
                "live_safe": False,
                "definition": "Offline report label measuring how the action changes sampled-world mass at or below the set threshold for bidder team.",
                "required_fields": ["e_q_pdf_or_sampled_world_outcomes", "bid_amount", "bidder_team_score_distribution_by_action"],
                "positive_condition": "Action increases defender set probability by at least configured threshold tau versus non-count alternatives in same decision.",
                "negative_condition": "No material set-threshold mass increase.",
                "anti_leakage": "Allowed only as training/eval target or report metric, never as live policy input.",
            },
            {
                "name": "disaster_tail_mitigation_delta",
                "level": "action",
                "live_safe": False,
                "definition": "Offline report label for lower-tail bidder-team outcome change, including CVaR or low-quantile shift.",
                "required_fields": ["e_q_pdf_or_sampled_world_outcomes", "action_outcome_distribution"],
                "positive_condition": "Action materially worsens bidder-team lower tail or improves defender-team lower tail relative to alternatives.",
                "negative_condition": "No material tail movement.",
                "anti_leakage": "Hidden branch labels remain offline diagnostics.",
            },
        ],
        "default_thresholds": {
            "threshold_mass_delta_tau": 0.10,
            "disaster_tail_quantile": 0.10,
            "material_eq_delta_points": 1.0,
            "one_count_away_points": 5,
        },
    }


def fixture_rows() -> list[dict[str, Any]]:
    return [
        {
            "fixture_id": "public_positive_count_before_certainty",
            "description": "Defender can throw 10 count while bidder team is off and partner may still win.",
            "public_state": {
                "current_player_seat": 1,
                "bidder_seat": 0,
                "bid_amount": 35,
                "contract_point_target": 35,
                "declaration": "trump_6",
                "led_suit": "fives",
                "current_trick_position": 2,
                "current_public_winning_seat": 2,
                "remaining_players_to_act": [3],
                "bidder_team_points_so_far": 24,
                "defender_team_points_so_far": 8,
                "current_public_defender_trick_points": 0,
                "public_winning_certainty": False,
            },
            "candidate_action": {"tile": "5-5", "legal": True, "count_points": 10},
            "expected_labels": {
                "is_setter_decision": True,
                "contract_context": True,
                "count_to_set_window": True,
                "bidder_off_window": True,
                "count_before_certainty": True,
                "setter_pounce_window": True,
                "setter_pounce_action": True,
            },
            "hidden_oracle_fields": "absent",
        },
        {
            "fixture_id": "negative_bidder_team_actor",
            "description": "Bidder-team partner has count, so this is not a setter pounce even if count matters.",
            "public_state": {
                "current_player_seat": 2,
                "bidder_seat": 0,
                "bid_amount": 35,
                "contract_point_target": 35,
                "declaration": "trump_6",
                "led_suit": "fives",
                "current_trick_position": 2,
                "current_public_winning_seat": 2,
                "remaining_players_to_act": [3],
                "bidder_team_points_so_far": 24,
                "defender_team_points_so_far": 8,
                "current_public_defender_trick_points": 0,
                "public_winning_certainty": False,
            },
            "candidate_action": {"tile": "5-5", "legal": True, "count_points": 10},
            "expected_labels": {
                "is_setter_decision": False,
                "contract_context": True,
                "count_to_set_window": True,
                "bidder_off_window": False,
                "count_before_certainty": False,
                "setter_pounce_window": False,
                "setter_pounce_action": False,
            },
            "hidden_oracle_fields": "absent",
        },
        {
            "fixture_id": "negative_no_threshold_pressure",
            "description": "Setter has a count action, but bidder already has enough public points to make.",
            "public_state": {
                "current_player_seat": 1,
                "bidder_seat": 0,
                "bid_amount": 30,
                "contract_point_target": 30,
                "declaration": "trump_4",
                "led_suit": "ones",
                "current_trick_position": 3,
                "current_public_winning_seat": 1,
                "remaining_players_to_act": [],
                "bidder_team_points_so_far": 33,
                "defender_team_points_so_far": 7,
                "current_public_defender_trick_points": 5,
                "public_winning_certainty": True,
            },
            "candidate_action": {"tile": "1-4", "legal": True, "count_points": 5},
            "expected_labels": {
                "is_setter_decision": True,
                "contract_context": True,
                "count_to_set_window": False,
                "bidder_off_window": False,
                "count_before_certainty": False,
                "setter_pounce_window": False,
                "setter_pounce_action": False,
            },
            "hidden_oracle_fields": "absent",
        },
        {
            "fixture_id": "offline_hidden_tail_positive",
            "description": "Same public pounce window with offline sampled-world tail labels attached for evaluation only.",
            "public_state": {
                "current_player_seat": 3,
                "bidder_seat": 0,
                "bid_amount": 36,
                "contract_point_target": 36,
                "declaration": "trump_2",
                "led_suit": "sixes",
                "current_trick_position": 1,
                "current_public_winning_seat": 0,
                "remaining_players_to_act": [1, 2],
                "bidder_team_points_so_far": 21,
                "defender_team_points_so_far": 9,
                "current_public_defender_trick_points": 0,
                "public_winning_certainty": False,
            },
            "candidate_action": {"tile": "6-4", "legal": True, "count_points": 10},
            "expected_labels": {
                "is_setter_decision": True,
                "contract_context": True,
                "count_to_set_window": True,
                "bidder_off_window": True,
                "count_before_certainty": True,
                "setter_pounce_window": True,
                "setter_pounce_action": True,
                "threshold_mass_delta": True,
                "disaster_tail_mitigation_delta": True,
            },
            "offline_only": {
                "set_probability_delta_vs_best_non_count": 0.18,
                "bidder_team_q10_delta_points": -6.5,
                "high_impact_hidden_tiles": ["6-6", "2-6"],
            },
        },
    ]


def required_fields_rows() -> list[dict[str, str]]:
    return [
        {"field": "current_player_seat", "source": "public state", "live_safe": "yes", "needed_for": "setter role"},
        {"field": "bidder_seat", "source": "public auction", "live_safe": "yes", "needed_for": "setter role and bidder team"},
        {"field": "bid_amount", "source": "public auction", "live_safe": "yes", "needed_for": "count-to-set threshold"},
        {"field": "declaration", "source": "public auction", "live_safe": "yes", "needed_for": "trump/off classification"},
        {"field": "ruleset_id", "source": "metadata", "live_safe": "yes", "needed_for": "contract/scoring semantics"},
        {"field": "bidder_team_points_so_far", "source": "public trick score before action", "live_safe": "yes", "needed_for": "points needed to make/set"},
        {"field": "defender_team_points_so_far", "source": "public trick score before action", "live_safe": "yes", "needed_for": "threshold accounting"},
        {"field": "current_trick_count_points", "source": "public current trick", "live_safe": "yes", "needed_for": "threshold accounting"},
        {"field": "led_suit", "source": "public current trick", "live_safe": "yes", "needed_for": "off-window detection"},
        {"field": "current_trick_position", "source": "public current trick", "live_safe": "yes", "needed_for": "count-before-certainty"},
        {"field": "current_public_winning_seat", "source": "public current trick", "live_safe": "yes", "needed_for": "winner relation"},
        {"field": "public_voids_by_seat_suit", "source": "public history", "live_safe": "yes", "needed_for": "public pounce plausibility"},
        {"field": "legal_action_set", "source": "engine legal mask", "live_safe": "yes", "needed_for": "action labels"},
        {"field": "candidate_count_points", "source": "candidate tile", "live_safe": "yes", "needed_for": "pounce action and count-to-set"},
        {"field": "e_q_pdf_or_sampled_world_outcomes", "source": "forge offline report", "live_safe": "no", "needed_for": "threshold/tail metrics"},
        {"field": "hidden_domino_owner_truth", "source": "sampled/completed worlds", "live_safe": "no", "needed_for": "offline threat attribution only"},
    ]


def anti_leakage_rows() -> list[dict[str, str]]:
    return [
        {
            "check_id": "public_inputs_only_for_live_labels",
            "label_scope": "live",
            "rule": "Live label computation may read only public state, current hand, legal actions, and public contract metadata.",
            "failure_mode": "Using hidden owner truth or future rollout outcomes to create live detector inputs.",
        },
        {
            "check_id": "pre_action_score_accounting",
            "label_scope": "live",
            "rule": "Count-to-set thresholds use points captured before candidate action plus public current trick count.",
            "failure_mode": "Using terminal hand totals or post-action trick winner truth.",
        },
        {
            "check_id": "structural_label_not_oracle_label",
            "label_scope": "live",
            "rule": "setter_pounce_action marks legal count in the window, not optimality.",
            "failure_mode": "Filtering positives by E[Q] or observed make/set result.",
        },
        {
            "check_id": "hidden_labels_offline_only",
            "label_scope": "offline",
            "rule": "threshold_mass_delta and disaster_tail_mitigation_delta are report/training targets only.",
            "failure_mode": "Adding sampled-world or hidden-tail values as live strategy features.",
        },
        {
            "check_id": "bidder_off_public_explainability",
            "label_scope": "live",
            "rule": "bidder_off_window must be explainable from public void/legal/trick state.",
            "failure_mode": "Marking a pounce because hidden partner owns a winner.",
        },
    ]


def tiny_report_rows() -> list[dict[str, str]]:
    return [
        {
            "claim_id": "ch05-pounce-count-before-certainty",
            "existing_evidence": "v0 proxy count into opponent-currently-winning window",
            "proxy_paired_n": "52",
            "proxy_paired_regret_delta": "+5.111 preferred-minus-alternative",
            "direct_label_ready": "no",
            "missing_direct_fields": "bidder_seat; bid_amount; pre-action team points; bidder_off_window; public winning certainty",
            "next_status": "direct-label spec ready; rerun after seat/contract state export",
        },
        {
            "claim_id": "ch05-extra-count-to-set",
            "existing_evidence": "v0 proxy ten-count versus five-count to opponent-currently-winning",
            "proxy_paired_n": "4",
            "proxy_paired_regret_delta": "+4.353 preferred-minus-alternative",
            "direct_label_ready": "no",
            "missing_direct_fields": "bid_amount; points_needed_to_make; points_needed_to_set; current trick count ownership",
            "next_status": "count-to-set formulas and fixtures ready",
        },
        {
            "claim_id": "ch12-setter-pounce-high-bid-off",
            "existing_evidence": "broad current-trick count-pressure proxy",
            "proxy_paired_n": "0",
            "proxy_paired_regret_delta": "not applicable",
            "direct_label_ready": "no",
            "missing_direct_fields": "high-bid contract context; bidder_off_window; setter role; threshold/tail distribution",
            "next_status": "requires phase-2 seat/position plus distribution outputs",
        },
    ]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).isoformat()
    sha = git_sha()

    spec = label_spec()
    write_json(OUT_DIR / "label_spec.json", spec)
    write_jsonl(OUT_DIR / "fixtures.jsonl", fixture_rows())
    write_csv(
        OUT_DIR / "required_fields.csv",
        required_fields_rows(),
        ["field", "source", "live_safe", "needed_for"],
    )
    write_csv(
        OUT_DIR / "anti_leakage_checks.csv",
        anti_leakage_rows(),
        ["check_id", "label_scope", "rule", "failure_mode"],
    )
    write_csv(
        OUT_DIR / "tiny_report_slice.csv",
        tiny_report_rows(),
        [
            "claim_id",
            "existing_evidence",
            "proxy_paired_n",
            "proxy_paired_regret_delta",
            "direct_label_ready",
            "missing_direct_fields",
            "next_status",
        ],
    )

    summary = {
        "schema_version": "w42.setter_pounce_direct_label_probe.summary.v1",
        "bead_id": "t42-5m82.5",
        "created_at": created_at,
        "repo_commit": sha,
        "status": "direct-label spec and fixtures complete; model probe deferred",
        "claim_ledger_impact": "no central ledger change",
        "hf_links": "not applicable",
        "wandb_links": "not applicable",
        "data_inputs": [
            "wiki/decisions/w42-next-model-decision.md",
            "wiki/experiments/w42-setter-defense-claim-validation.md",
            "w42/setter_defense_claim_validation/summary.json",
            "w42/strategy_tags_v1_map/detector_map.json",
            "wiki/experiments/w42-final-empirical-strategy-report.md",
            "/Users/jason/code/mk5-main/gus/data/corpus_eval_20.pt field inspection",
        ],
        "current_corpus_inspection": {
            "eval_corpus": "/Users/jason/code/mk5-main/gus/data/corpus_eval_20.pt",
            "rows": 560,
            "available_item_fields": [
                "tokens",
                "attention_mask",
                "belief_target",
                "belief_mask",
                "world_assignment",
                "q_per_world",
                "e_q",
                "action_taken",
                "legal_mask",
                "decision_idx",
                "player",
                "voids",
                "strategy_features",
                "strategy_action_features",
            ],
            "missing_for_direct_probe": [
                "bidder_seat",
                "bid_amount",
                "declaration as named contract context",
                "pre-action bidder_team_points_so_far",
                "pre-action defender_team_points_so_far",
                "current trick led suit and public winning certainty as named fields",
                "E[Q] PDF or sampled-world outcome distribution by action",
            ],
        },
        "first_pass_labels": [entry["name"] for entry in spec["direct_labels"]],
        "public_state_labels": [
            entry["name"] for entry in spec["direct_labels"] if entry["live_safe"]
        ],
        "offline_only_labels": [
            entry["name"] for entry in spec["direct_labels"] if not entry["live_safe"]
        ],
        "tiny_report_slice": {
            "path": "w42/setter_pounce_direct_label_probe/tiny_report_slice.csv",
            "source": "manual slice from w42/setter_defense_claim_validation/summary.json",
            "interpretation": "Existing artifacts support dependency/readiness reporting, not a direct label verdict.",
        },
        "next_run_command_plan": [
            "Export or generate a phase-2 decision table with bidder_seat, bid_amount, declaration, pre-action team points, led_suit, current public winner, public voids, legal actions, and candidate tiles.",
            "Attach optional offline distribution columns from E[Q] PDFs or sampled worlds: set_probability_by_action, q10_by_action, cvar10_by_action, and high-impact hidden-domino attribution.",
            "Run a future analyzer over that table to emit direct-label coverage, paired pounce-action regret, threshold-mass deltas, and disaster-tail deltas.",
            "Only after that direct table exists, train raw/v0/rich/direct/distribution-aware variants or log W&B series.",
        ],
        "validation": {
            "fixtures": "4 JSONL fixture records with expected labels",
            "anti_leakage_checks": "5 checks",
            "json_artifacts": ["label_spec.json", "summary.json", "manifest.json"],
            "csv_artifacts": ["required_fields.csv", "anti_leakage_checks.csv", "tiny_report_slice.csv"],
        },
    }
    manifest = {
        "schema_version": "w42.setter_pounce_direct_label_probe.manifest.v1",
        "bead_id": "t42-5m82.5",
        "created_at": created_at,
        "repo_commit": sha,
        "producer": "w42/setter_pounce_direct_label_probe/build_probe_artifacts.py",
        "artifacts": [
            {"path": "w42/setter_pounce_direct_label_probe/label_spec.json", "kind": "direct label spec"},
            {"path": "w42/setter_pounce_direct_label_probe/fixtures.jsonl", "kind": "fixture examples"},
            {"path": "w42/setter_pounce_direct_label_probe/required_fields.csv", "kind": "required data fields"},
            {"path": "w42/setter_pounce_direct_label_probe/anti_leakage_checks.csv", "kind": "anti-leakage checks"},
            {"path": "w42/setter_pounce_direct_label_probe/tiny_report_slice.csv", "kind": "readiness report slice"},
            {"path": "w42/setter_pounce_direct_label_probe/summary.json", "kind": "summary"},
            {"path": "wiki/experiments/w42-phase2-setter-pounce-direct-label-probe.md", "kind": "wiki report"},
        ],
        "dependencies_for_revisit": [
            "phase-2 seat/position decision table",
            "contract and pre-action score export",
            "distribution-aware E[Q] PDF or sampled-world outputs",
        ],
        "claim_ledger_impact": "no central ledger change",
    }
    write_json(OUT_DIR / "summary.json", summary)
    write_json(OUT_DIR / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
