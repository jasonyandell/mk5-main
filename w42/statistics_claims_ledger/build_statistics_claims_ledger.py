#!/usr/bin/env python3
"""Build the w42 phase-2 statistics claims ledger artifacts.

This is a synthesis generator, not a new game simulator. It reads the existing
w42 claim-validation outputs, adds a few exact arithmetic checks that are simple
enough to keep here, and writes a conservative CSV plus JSON summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

CLAIMS_CSV = OUT_DIR / "claims.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"
MANIFEST_JSON = OUT_DIR / "manifest.json"

OWNER_BEAD = "t42-5m82.1"
SCHEMA_VERSION = "w42.statistics_claims_ledger.v1"
MANIFEST_VERSION = "w42.statistics_claims_ledger.manifest.v1"

ALLOWED_STATUSES = {
    "supported",
    "contradicted",
    "context-limited",
    "underpowered",
    "not-yet-tested",
}

CSV_FIELDS = [
    "claim_id",
    "family",
    "status",
    "claim",
    "source",
    "evidence_mode",
    "readiness",
    "measurement",
    "result",
    "evidence_artifacts",
    "caveats",
    "next_check",
    "wandb",
    "hf",
]


def read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="") as f:
        return list(csv.DictReader(f))


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def pct(value: float) -> str:
    if 0 < abs(value) < 0.001:
        return f"{value:.6f}%"
    return f"{value:.3f}%"


def wandb_url(value: str) -> str:
    if not value or value == "not applicable":
        return "not applicable"
    if value.startswith("http"):
        return value
    if value.startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed.get("url", value)
    return value


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row[key] == value:
            return row
    raise KeyError(f"missing {key}={value}")


def sum_percent(rows: list[dict[str, str]], key: str, min_value: int) -> float:
    total = 0.0
    for row in rows:
        if int(row[key]) >= min_value:
            total += as_float(row, "pct" if "pct" in row else "percent")
    return total


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        claim_id: str,
        family: str,
        status: str,
        claim: str,
        source: str,
        evidence_mode: str,
        readiness: str,
        measurement: str,
        result: str,
        evidence_artifacts: list[str],
        caveats: str,
        next_check: str,
        wandb: str = "not applicable",
        hf: str = "not applicable",
    ) -> None:
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"{claim_id}: bad status {status}")
        rows.append(
            {
                "claim_id": claim_id,
                "family": family,
                "status": status,
                "claim": claim,
                "source": source,
                "evidence_mode": evidence_mode,
                "readiness": readiness,
                "measurement": measurement,
                "result": result,
                "evidence_artifacts": "; ".join(evidence_artifacts),
                "caveats": caveats,
                "next_check": next_check,
                "wandb": wandb,
                "hf": hf,
            }
        )

    odds_summary = read_json("w42/odds_ruleset_claim_validation/validation_summary.json")
    hand_shape = read_csv("w42/odds_ruleset_claim_validation/hand_shape_priors.csv")
    double_counts = read_csv("w42/odds_ruleset_claim_validation/double_count_priors.csv")
    four_trump = read_csv("w42/odds_ruleset_claim_validation/four_trump_assignments.csv")
    ruleset_checks = read_csv("w42/odds_ruleset_claim_validation/ruleset_checks.csv")

    bidding_summary = read_json("w42/bidding_risk_budget_claim_validation/summary.json")
    candidate_trump = read_csv(
        "w42/bidding_risk_budget_claim_validation/candidate_trump_count_distribution.csv"
    )
    best_risk = read_csv(
        "w42/bidding_risk_budget_claim_validation/best_candidate_risk_points_distribution.csv"
    )
    candidate_risk = read_csv(
        "w42/bidding_risk_budget_claim_validation/candidate_risk_bucket_distribution.csv"
    )
    bid_ceiling = read_csv(
        "w42/bidding_risk_budget_claim_validation/bid_ceiling_proxy_distribution.csv"
    )

    dnt_summary = read_json("w42/doubles_no_trump_claim_validation/summary.json")
    dnt_slices = read_csv("w42/doubles_no_trump_claim_validation/hand_regime_proxy_slices.csv")
    dnt_rules = read_csv("w42/doubles_no_trump_claim_validation/ruleset_regime_checks.csv")

    scoring_delta = read_json(
        "w42/scoring_objective_drift_claim_validation/claim_ledger_delta.json"
    )

    eighty_four_summary = read_json("w42/eighty_four_claim_validation/summary.json")
    eighty_four_claims = read_csv("w42/eighty_four_claim_validation/claim_summary.csv")
    eighty_four_shapes = read_csv("w42/eighty_four_claim_validation/bidder_shape_summary.csv")
    eighty_four_weapons = read_csv(
        "w42/eighty_four_claim_validation/defender_last_trick_weapon_distribution.csv"
    )

    partner_summary = read_json("w42/partner_support_claim_validation/summary.json")
    partner_claims = read_csv("w42/partner_support_claim_validation/claim_proxy_stats.csv")
    setter_summary = read_json("w42/setter_defense_claim_validation/summary.json")
    setter_claims = read_csv("w42/setter_defense_claim_validation/claim_proxy_stats.csv")
    bidder_summary = read_json("w42/bidder_sequencing_claim_validation/summary.json")
    bidder_claims = read_csv("w42/bidder_sequencing_claim_validation/claim_summary.csv")

    odds_artifacts = [
        "w42/odds_ruleset_claim_validation/validation_summary.json",
        "w42/odds_ruleset_claim_validation/hand_shape_priors.csv",
        "w42/odds_ruleset_claim_validation/double_count_priors.csv",
        "w42/odds_ruleset_claim_validation/four_trump_assignments.csv",
    ]

    total_hands = odds_summary["sample_sizes"]["seven_domino_hands"]
    modal = odds_summary["headline"]["modal_joint_bucket"]
    add(
        "ch16-seven-card-hand-total",
        "hand odds",
        "supported",
        "There are 1,184,040 possible seven-domino hands from a double-six set.",
        "scratch/winning42/winning42.with_figures.md:9554; wiki/experiments/winning42-ch16-statistical-odds.md",
        "exact enumeration",
        "enumeration; report-only",
        "C(28, 7) over the standard double-six tile set.",
        f"C(28, 7) = {total_hands:,}.",
        odds_artifacts,
        "Arithmetic support only; it says nothing about strategy quality.",
        "Use as denominator fixture for generated-deal audits.",
    )

    hand_shape_result = "; ".join(
        f"{row['bucket']} = {pct(float(row['percent']))}" for row in hand_shape
    )
    add(
        "ch16-suit-void-frequency",
        "void frequencies",
        "supported",
        "Seven-card hands have the stated rough suit-coverage and void frequencies.",
        "scratch/winning42/winning42.with_figures.md:9561-9575; wiki/experiments/winning42-ch16-statistical-odds.md",
        "exact enumeration",
        "enumeration; report-only",
        "Exhaustive count of represented pips and void suits in all seven-domino hands.",
        hand_shape_result,
        odds_artifacts,
        "The source's three-void bucket rounds up; exact value is better described as less than 1%.",
        "Add generated-corpus chi-square audit if deal manifests become available.",
    )

    double_result = "; ".join(
        f"{row['bucket']} = {pct(float(row['percent']))}" for row in double_counts
    )
    double_ge2 = sum(
        float(row["percent"])
        for row in double_counts
        if int(row["bucket"].split()[0]) >= 2
    )
    add(
        "ch16-double-count-frequency",
        "double-count odds",
        "supported",
        "The book's double-count odds match exact enumeration within rounding.",
        "scratch/winning42/winning42.with_figures.md:9576-9591; wiki/experiments/winning42-ch16-statistical-odds.md",
        "exact enumeration",
        "enumeration; report-only",
        "Exhaustive count of doubles in all seven-domino hands.",
        double_result,
        odds_artifacts,
        "Exact priors do not establish a bid or play recommendation.",
        "Package as stable odds fixture if w42 promotes reusable odds tables.",
    )

    add(
        "ch16-modal-two-doubles-one-void",
        "hand odds",
        "supported",
        "The modal joint hand-shape bucket is two doubles and one void suit.",
        "scratch/winning42/winning42.with_figures.md:9593-9600; wiki/experiments/winning42-ch16-statistical-odds.md",
        "exact enumeration",
        "enumeration; report-only",
        "Joint bucket count over double_count and void_count.",
        (
            f"double_count={modal['double_count']}, void_count={modal['void_count']}: "
            f"{modal['count']:,} hands ({pct(float(modal['percent']))})."
        ),
        ["w42/odds_ruleset_claim_validation/validation_summary.json"],
        "A modal hand-shape prior is not a partner-help guarantee.",
        "Check whether generated corpora reproduce the same modal bucket.",
    )

    ft_missing_second = row_by(
        four_trump,
        "bucket",
        "missing second-highest only; setter can retain it after following double",
    )
    add(
        "ch16-four-trump-missing-second",
        "four-trump odds",
        "supported",
        "With four trumps including the double but missing the second-highest trump, the setter double-up risk is 10 of 27 assignments.",
        "scratch/winning42/winning42.with_figures.md:9515-9531; wiki/experiments/winning42-ch16-statistical-odds.md",
        "exact assignment enumeration",
        "enumeration; report-only",
        "Assign the three missing trumps independently to partner, left opponent, or right opponent, with partner not counted as a setter threat.",
        f"{ft_missing_second['count']} / {ft_missing_second['total']} = {pct(float(ft_missing_second['percent']))}.",
        odds_artifacts,
        "Supports the assignment arithmetic, not the downstream instruction to ignore trump-loss risk in all contexts.",
        "Run paired boss-first versus low-trump forge rollouts by bid margin, count, and score.",
    )

    ft_missing_next_two = row_by(
        four_trump,
        "bucket",
        "missing next two highest; setter can retain a high trump after double lead",
    )
    add(
        "ch16-four-trump-missing-next-two",
        "four-trump odds",
        "supported",
        "With four trumps including the double but missing the next two highest trumps, the setter double-up risk is 14 of 27 assignments.",
        "scratch/winning42/winning42.with_figures.md:9533-9543; wiki/experiments/winning42-ch16-statistical-odds.md",
        "exact assignment enumeration",
        "enumeration; report-only",
        "Same 27-assignment model as the missing-second claim.",
        f"{ft_missing_next_two['count']} / {ft_missing_next_two['total']} = {pct(float(ft_missing_next_two['percent']))}.",
        odds_artifacts,
        "The near-even prior does not decide the value of cashing count in trump.",
        "Run paired rollouts split by trump-count domino, off protection, bid margin, and score.",
    )

    add(
        "ch16-four-trump-boss-first-policy",
        "four-trump tactics",
        "underpowered",
        "The exact four-trump odds imply the recommended first lead is strategically optimal.",
        "scratch/winning42/winning42.with_figures.md:9524-9543; wiki/experiments/winning42-ch16-statistical-odds.md",
        "unsupported strategy inference",
        "oracle; w42; burl",
        "No paired action-value rollout is present in the available artifacts.",
        "Odds substrate is exact, but the policy claim is untested.",
        odds_artifacts,
        "Do not promote tactical advice to supported from assignment counts alone.",
        "Run forge E[Q] counterfactuals for boss-first versus low-trump lines.",
    )

    add(
        "ch16-partner-two-plus-doubles-prior",
        "partner-help odds",
        "context-limited",
        "A random partner hand has a decent chance of containing at least two doubles.",
        "scratch/winning42/winning42.with_figures.md:9593-9600; wiki/experiments/winning42-ch16-statistical-odds.md",
        "exact unconditional prior",
        "enumeration; oracle",
        "Sum exact double-count buckets for two or more doubles.",
        f"P(random hand has >=2 doubles) = {pct(double_ge2)}.",
        ["w42/odds_ruleset_claim_validation/double_count_priors.csv"],
        "This is unconditional; actual partner help must condition on bidder hand, bidding, position, and future tricks.",
        "Compute conditional partner-double priors by bidder off count and correlate with make/set or regret.",
    )

    fixed_ge3 = sum_percent(candidate_trump, "trump_count", 3)
    fixed_ge4 = sum_percent(candidate_trump, "trump_count", 4)
    add(
        "ch02-fixed-suit-trump-count-prior",
        "trump counts",
        "supported",
        "Fixed pip-trump candidate declarations have a measurable exact trump-count distribution.",
        "scratch/winning42/winning42.with_figures.md:766-770; w42/bidding_risk_budget_claim_validation/candidate_trump_count_distribution.csv",
        "exact enumeration",
        "enumeration; report-only",
        "All C(28,7) hands times seven pip-trump candidates.",
        f"P(candidate has >=3 trumps) = {pct(fixed_ge3)}; P(candidate has >=4 trumps) = {pct(fixed_ge4)}.",
        [
            "w42/bidding_risk_budget_claim_validation/candidate_trump_count_distribution.csv",
            "w42/eighty_four_claim_validation/trump_count_distribution.csv",
        ],
        "Descriptive candidate prior only; it does not say which suit should be declared or whether to bid.",
        "Add max-suit and auction-aware versions if a bidding dataset is promoted.",
    )

    add(
        "ch02-three-plus-trumps-good-start",
        "bidding risk",
        "underpowered",
        "Three or more trumps are enough to make a hand a good bidding start.",
        "scratch/winning42/winning42.with_figures.md:766-770; wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
        "static detector evidence only",
        "oracle; w42",
        "Current artifacts count candidate trump shape and exposure, not auction or make/set value.",
        "Static priors exist, but no outcome counterfactual validates the rule.",
        [
            "w42/bidding_risk_budget_claim_validation/summary.json",
            "w42/bidding_risk_budget_claim_validation/best_candidate_trump_count_distribution.csv",
        ],
        "Shape is a necessary detector surface, not a tactical verdict.",
        "Run bidding rollouts or generated auction counterfactuals.",
    )

    best_risk_le10 = sum(
        float(row["pct"]) for row in best_risk if int(row["unique_exposed_points"]) <= 10
    )
    risk_le12 = row_by(candidate_risk, "risk_bucket", "risk_le_12")
    add(
        "ch02-risk-budget-threshold",
        "count exposure",
        "underpowered",
        "A 12-or-fewer at-risk point total is sufficient to justify bidding.",
        "scratch/winning42/winning42.with_figures.md:880-904; wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
        "exact static enumeration",
        "enumeration; oracle",
        "Compute unique exposed count points for each hand/declaration candidate.",
        (
            f"{risk_le12['pct']}% of candidates have <=12 exposed points; "
            f"{pct(best_risk_le10)} of hands have a best static candidate with <=10 exposed points."
        ),
        [
            "w42/bidding_risk_budget_claim_validation/candidate_risk_bucket_distribution.csv",
            "w42/bidding_risk_budget_claim_validation/best_candidate_risk_points_distribution.csv",
        ],
        "The threshold is counted statically; no make/set, auction, or partner-help outcome is measured.",
        "Run auction-aware make/set or forge E[Q] bidding counterfactuals.",
    )

    add(
        "ch02-duplicate-count-exposure",
        "count exposure",
        "supported",
        "Repeated side exposure can double-count the same count domino unless de-duplicated.",
        "scratch/winning42/winning42.with_figures.md:880-885; wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
        "exact static enumeration",
        "enumeration; report-only",
        "Compare naive side-summed exposure with unique exposed count points.",
        (
            f"{bidding_summary['candidate_metrics']['duplicate_exposure']['pct']}% of candidate evaluations "
            f"and {bidding_summary['hand_metrics']['any_duplicate_exposure']['pct']}% of hands have duplicate exposure under at least one candidate declaration."
        ),
        [
            "w42/bidding_risk_budget_claim_validation/summary.json",
            "w42/bidding_risk_budget_claim_validation/duplicate_overcount_points_distribution.csv",
        ],
        "Supports exposure arithmetic only, not an optimal bid amount.",
        "Keep as a regression fixture for risk-budget detectors.",
    )

    add(
        "ch02-strong-trump-bad-risk-trap",
        "count exposure",
        "context-limited",
        "Strong-looking trump shape can still exceed the beginner risk budget because of offs.",
        "scratch/winning42/winning42.with_figures.md:954-994; wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
        "exact static enumeration",
        "enumeration; oracle",
        "Flag candidate declarations with >=4 trumps and >12 unique exposed count points.",
        (
            f"{bidding_summary['candidate_metrics']['strong_trump_bad_risk_trap']['pct']}% of candidates and "
            f"{bidding_summary['hand_metrics']['any_strong_trump_bad_risk']['pct']}% of hands have such a candidate."
        ),
        [
            "w42/bidding_risk_budget_claim_validation/summary.json",
            "w42/bidding_risk_budget_claim_validation/strong_trump_bad_risk_points_distribution.csv",
        ],
        "Exact static support for the trap surface; outcome value remains untested.",
        "Split by actual declared suit, auction pressure, and E[Q] outcome.",
    )

    add(
        "ch02-four-five-off-danger",
        "count exposure",
        "context-limited",
        "Four and five offs create high count-exposure risk.",
        "scratch/winning42/winning42.with_figures.md:810-821; wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
        "exact static enumeration",
        "enumeration; oracle",
        "Detect candidate declarations with a 5-4 off or equivalent high count exposure.",
        (
            f"{bidding_summary['candidate_metrics']['four_five_off']['pct']}% of candidate evaluations and "
            f"{bidding_summary['hand_metrics']['any_four_five_off']['pct']}% of hands expose a 5-4-off proxy."
        ),
        ["w42/bidding_risk_budget_claim_validation/summary.json"],
        "Prevalence and point arithmetic are measured; tactical cost is not.",
        "Run paired bidding/play outcomes for high-off exposure buckets.",
    )

    ceiling_values = ", ".join(row["bid_ceiling_proxy"] for row in bid_ceiling)
    add(
        "ch12-natural-bid-bucket-anomaly",
        "bidding risk",
        "underpowered",
        "Natural bid buckets such as 30/31/35/36 can be validated from static exposed-count ceilings.",
        "scratch/winning42/winning42.with_figures.md:1232-1238; wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
        "static proxy limitation",
        "oracle; w42",
        "Compute bid ceiling as 42 minus unique exposed count points.",
        f"The static proxy emits only these ceiling values: {ceiling_values}.",
        [
            "w42/bidding_risk_budget_claim_validation/bid_ceiling_proxy_distribution.csv",
        ],
        "Auction increments, partner bids, and trick-risk modeling are missing.",
        "Use generated auction logs or paired bidding rollouts.",
    )

    add(
        "ch02-double-side-protection",
        "count exposure",
        "context-limited",
        "A double ahead of an off reduces risk, but only for the protected side of the off.",
        "scratch/winning42/winning42.with_figures.md:847-878; wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
        "exact static enumeration",
        "enumeration; oracle",
        "Count side-specific exposed points that are protected by a same-pip double.",
        f"{bidding_summary['side_specific_exposure_points']['protected_pct']}% of side-exposure points are protected by a same-pip double.",
        ["w42/bidding_risk_budget_claim_validation/summary.json"],
        "The current proxy does not model when the double is played, forced, or still live.",
        "Add action-sequence and follow-pressure labels.",
    )

    add(
        "ch02-bid-only-enough",
        "bidding risk",
        "not-yet-tested",
        "A bidder should bid only enough to win because excess bid amount only increases set risk.",
        "scratch/winning42/winning42.with_figures.md:1360-1389; wiki/experiments/winning42-ch02-bidding.md",
        "not yet measured",
        "oracle; w42; burl",
        "Requires auction counterfactuals comparing unnecessary bid margin with set rate and regret.",
        "No auction-counterfactual artifact exists in the current w42 outputs.",
        ["not applicable"],
        "Static hand-risk enumeration cannot answer this auction claim.",
        "Generate auction logs or paired bid-margin E[Q] rollouts.",
    )

    dnt_wandb = dnt_summary.get("wandb", {}).get("url", "not applicable")
    for rule in dnt_rules:
        add(
            rule["claim_id"],
            "doubles/no-trump rules",
            rule["status"],
            rule["check"],
            "scratch/winning42/winning42.with_figures.md:3838-4239; wiki/experiments/w42-doubles-no-trump-claim-validation.md",
            "deterministic ruleset predicate",
            "ruleset; report-only",
            "Ruleset fixture check from the doubles/no-trump validation script.",
            f"{rule['pass_count']} / {rule['sample_size']} predicate cases passed; {rule['details']}.",
            [
                "w42/doubles_no_trump_claim_validation/ruleset_regime_checks.csv",
                "w42/doubles_no_trump_claim_validation/summary.json",
            ],
            "Ruleset substrate only; it does not prove a declaration or play choice.",
            "Keep as a regression fixture and add paired declaration rollouts for strategy choices.",
            wandb=dnt_wandb,
        )

    for slice_id, claim, status_override in [
        (
            "doubles_trump_candidate_4plus",
            "Four-plus doubles are a measurable candidate gate for doubles-as-trump hands.",
            "context-limited",
        ),
        (
            "no_trump_support_proxy",
            "No-trump support can be approximated by static double/off support buckets.",
            "underpowered",
        ),
        (
            "regime_switch_proxy_nt_over_dt",
            "Some four-plus-double hands are plausible no-trump-over-doubles-trump switch candidates.",
            "underpowered",
        ),
    ]:
        item = row_by(dnt_slices, "slice", slice_id)
        add(
            f"ch09-{slice_id.replace('_', '-')}",
            "doubles/no-trump odds",
            status_override,
            claim,
            "scratch/winning42/winning42.with_figures.md:3838-4239; wiki/experiments/w42-doubles-no-trump-claim-validation.md",
            "exact static enumeration",
            "enumeration; oracle",
            "Exact hand-slice count from all seven-domino hands.",
            f"{item['n']} / {item['denominator']} = {item['pct']}%; {item['interpretation']}",
            [
                "w42/doubles_no_trump_claim_validation/hand_regime_proxy_slices.csv",
                "w42/doubles_no_trump_claim_validation/summary.json",
            ],
            "Static regime membership is not a declaration-value verdict.",
            "Run paired doubles-trump versus no-trump rollouts.",
            wandb=dnt_wandb,
        )

    missing_trumps_slots = comb(21, 2)
    both_opponents_have_one = 7 * 7
    at_least_one_opponent_void = missing_trumps_slots - both_opponents_have_one
    opponent_void_pct = 100.0 * at_least_one_opponent_void / missing_trumps_slots
    add(
        "ch09-five-doubles-opponent-void-prior",
        "doubles/no-trump odds",
        "supported",
        "With five doubles as trumps and only two missing doubles, it is roughly 80% likely that at least one opponent cannot follow doubles.",
        "scratch/winning42/winning42.with_figures.md:4111-4120",
        "exact conditional arithmetic",
        "enumeration; report-only",
        "Assign the two missing doubles uniformly over 21 unknown slots: partner 7, each opponent 7.",
        f"{at_least_one_opponent_void} / {missing_trumps_slots} = {pct(opponent_void_pct)}.",
        ["w42/statistics_claims_ledger/build_statistics_claims_ledger.py"],
        "This validates the rounded prior, not the amount of count opponents can discard or whether no-trump is better.",
        "Add paired declaration/play rollouts for the sample hand class.",
    )

    for entry in scoring_delta["entries"]:
        prov = entry["provenance"]
        source = entry["chapter_source"]["wiki_page"]
        if "source_slice" in entry["chapter_source"]:
            source = f"{source}; {entry['chapter_source']['source_slice']}"
        add(
            entry["claim_id"],
            "scoring objective",
            entry["status"],
            entry["claim"],
            source,
            "deterministic scoring transform",
            "; ".join(entry["readiness"]),
            entry["metric_test"],
            f"See scoring transform tables; caveat: {entry['caveats']}",
            [
                entry["evidence_artifact"],
                "w42/scoring_objective_drift_claim_validation/terminal_objective_transform.csv",
                "w42/scoring_objective_drift_claim_validation/objective_thresholds.csv",
            ],
            entry["caveats"],
            "Use policy-population or tournament simulation for claims about skill, speed, or advancement.",
            wandb=wandb_url(prov["wandb_links"]),
            hf=prov["hf_links"],
        )

    w84_wandb = eighty_four_summary.get("wandb", {}).get("url", "not applicable")
    eighty_four_claim_text = {
        "ch07-84-contract-regime": "84 is an all-seven-tricks contract with special scoring semantics.",
        "ch07-protected-one-off-84-shape": "Protected one-off 84 shapes are strong enough to justify the 84 bidding advice.",
        "ch07-straight-off-two-to-one-double": "A straight-off 84 is exposed to a two-to-one missing-double ownership prior.",
        "ch07-score-42-vs-84-gate": "Near 250 points, the decision between bidding 42 and 84 should be score-gated.",
        "ch08-one-to-four-last-trick-weapons": "Each opponent usually starts an 84 defense with one to four possible last-trick weapons.",
        "ch08-double-ahead-needs-same-suit-pair": "When the bidder has the double ahead of the off, defenders need a same-suit pair line.",
        "ch08-abandon-dead-assets": "A defender should abandon stopper assets once public evidence proves they are dead.",
        "ch08-throwaway-priority-ladder": "The 84 throwaway priority ladder preserves the right final-trick stopper.",
    }
    for claim_row in eighty_four_claims:
        source = "wiki/experiments/winning42-ch07-taking-every-trick-84.md"
        if claim_row["claim_id"].startswith("ch08"):
            source = "wiki/experiments/winning42-ch08-setting-84.md"
        add(
            claim_row["claim_id"],
            "84 bidder/defender",
            claim_row["status"],
            eighty_four_claim_text.get(
                claim_row["claim_id"], claim_row["claim_id"].replace("-", " ")
            ),
            source,
            "exact static enumeration / hypergeometric proxy",
            "enumeration; report-only",
            claim_row["evidence"],
            f"{claim_row['bucket']} claim; {claim_row['evidence']}.",
            [
                "w42/eighty_four_claim_validation/claim_summary.csv",
                "w42/eighty_four_claim_validation/summary.json",
            ],
            claim_row["caveat"],
            "Use 84-specific generated states and paired last-trick tests.",
            wandb=w84_wandb,
        )

    straight_named = next(
        row
        for row in eighty_four_weapons
        if row["asset_pool_size"] == "single named matching double"
    )
    either_double = next(
        row
        for row in eighty_four_weapons
        if row["asset_pool_size"] == "either of two matching doubles"
    )
    add(
        "ch08-either-matching-double-not-two-to-one",
        "stopper ownership",
        "contradicted",
        "The straight-off two-to-one ownership prior still holds when either of two matching doubles can set the bidder.",
        "scratch/winning42/winning42.with_figures.md:3318-3324; scratch/winning42/winning42.with_figures.md:3799-3811; wiki/experiments/w42-84-claim-validation.md",
        "exact hypergeometric ownership check",
        "enumeration; report-only",
        "Compare opponent-team ownership for one named matching double versus either of two matching doubles.",
        (
            f"Named double opponent-team ownership = {straight_named['single_defender_p_one_to_four']}%; "
            f"either of two matching doubles = {either_double['single_defender_p_one_to_four']}%."
        ),
        [
            "w42/eighty_four_claim_validation/defender_last_trick_weapon_distribution.csv",
        ],
        "Contradicts only the broader two-matching-double reading, not the source's named-double case.",
        "Model final-off cases explicitly and separate named stopper from any-stopper claims.",
        wandb=w84_wandb,
    )

    one_off_shape = row_by(eighty_four_shapes, "bucket", "one off candidate shape")
    protected_shape = row_by(eighty_four_shapes, "bucket", "protected one off")
    add(
        "ch07-protected-one-off-shape-frequency",
        "84 bidder/defender",
        "underpowered",
        "Typical 84 candidate shapes with one protected off are available often enough to support the bidding advice.",
        "scratch/winning42/winning42.with_figures.md:3122-3154; wiki/experiments/w42-84-claim-validation.md",
        "exact static enumeration",
        "enumeration; oracle",
        "Count hand/declaration pairs with one off and protected one-off structure.",
        (
            f"One-off candidate shape = {one_off_shape['pct_of_all']}% of hand/declaration pairs; "
            f"protected one off = {protected_shape['pct_of_all']}%."
        ),
        [
            "w42/eighty_four_claim_validation/bidder_shape_summary.csv",
        ],
        "Shape existence does not prove an 84 bid is value-positive.",
        "Run make/set rollouts for score-gated 84 candidate states.",
        wandb=w84_wandb,
    )

    for claim_row in partner_claims:
        result = (
            f"preferred_n={claim_row['preferred_action_n']}, alternative_n={claim_row['alternative_action_n']}, "
            f"paired_n={claim_row['paired_decision_n']}, paired_delta={claim_row['paired_preferred_minus_alternative_mean_regret']}"
        )
        add(
            claim_row["claim_id"],
            "partner support",
            claim_row["verdict"],
            claim_row["label"],
            "wiki/experiments/winning42-ch04-partner-support.md; w42/partner_support_claim_validation/claim_proxy_stats.csv",
            "proxy regret analysis",
            "w42; oracle",
            "Bootstrap CIs over available eval-corpus proxy buckets.",
            result,
            [
                "w42/partner_support_claim_validation/claim_proxy_stats.csv",
                "w42/partner_support_claim_validation/summary.json",
            ],
            "Proxy tags do not fully encode bidder-partner intent, hidden risk, or future forcedness.",
            "Implement direct partner-intent and forcedness detectors before moving statuses broadly.",
            hf=partner_summary["hf_links"],
        )

    for claim_row in setter_claims:
        result = (
            f"directness={claim_row['directness']}, preferred_n={claim_row['preferred_action_n']}, "
            f"alternative_n={claim_row['alternative_action_n']}, paired_n={claim_row['paired_decision_n']}"
        )
        add(
            claim_row["claim_id"],
            "setter defense",
            claim_row["verdict"],
            claim_row["label"],
            "wiki/experiments/winning42-ch05-setter-defense.md; wiki/experiments/winning42-ch12-advanced-bidding-playing.md; w42/setter_defense_claim_validation/claim_proxy_stats.csv",
            "proxy regret analysis",
            "w42; oracle",
            "Bootstrap CIs over available eval-corpus proxy buckets.",
            result,
            [
                "w42/setter_defense_claim_validation/claim_proxy_stats.csv",
                "w42/setter_defense_claim_validation/summary.json",
            ],
            "The current v0 corpus lacks direct setter-role, bidder-off, void-creation, trump-set, and bid-margin detectors.",
            "Implement direct setter-pounce labels and set-threshold accounting.",
            hf=setter_summary["hf_links"],
        )

    for claim_row in bidder_claims:
        add(
            claim_row["claim_id"],
            "bidder sequencing",
            claim_row["status"],
            claim_row["verdict"],
            "wiki/experiments/winning42-ch03-bidder-play.md; w42/bidder_sequencing_claim_validation/claim_summary.csv",
            "secondary model-bucket analysis",
            "w42; oracle",
            f"Proxy buckets: {claim_row['proxy_buckets']}",
            (
                f"weighted_rows={claim_row['n_weighted_bucket_rows']}, "
                f"delta_mean_regret={claim_row['weighted_delta_mean_regret']}, "
                f"delta_match_rate={claim_row['weighted_delta_match_rate']}"
            ),
            [
                "w42/bidder_sequencing_claim_validation/claim_summary.csv",
                "w42/bidder_sequencing_claim_validation/summary.json",
            ],
            "Proxy bucket gains validate tag usefulness at most; paired sequence counterfactuals are missing.",
            "Run trump-first/off-first paired rollouts and late-endgame proof checks.",
            hf=bidder_summary["hf_links"] if "hf_links" in bidder_summary else "not applicable",
        )

    claim_ids = [row["claim_id"] for row in rows]
    duplicates = [claim for claim, count in Counter(claim_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate claim ids: {duplicates}")

    return rows


def build_summary(rows: list[dict[str, str]], generated_at: str) -> dict[str, Any]:
    status_counts = Counter(row["status"] for row in rows)
    family_counts = Counter(row["family"] for row in rows)
    evidence_counts = Counter(row["evidence_mode"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "owner_bead": OWNER_BEAD,
        "generated_at_utc": generated_at,
        "git_sha": git_sha(),
        "claim_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "evidence_mode_counts": dict(sorted(evidence_counts.items())),
        "headline_findings": [
            "Exact hand-count, void-frequency, double-count, modal-hand, four-trump assignment, ruleset, and scoring-algebra substrates are supported on their stated slices.",
            "Bidding, partner-support, setter-defense, bidder-sequencing, 84, and no-trump tactical recommendations remain mostly context-limited, underpowered, or not-yet-tested.",
            "The contradicted rows are narrow: one 84 stopper-ownership overgeneralization and one partner-support proxy result.",
            "No tactical advice was moved to supported solely because its statistical substrate is true.",
        ],
        "source_inventory": {
            "book_preview": [
                "scratch/winning42/winning42.with_figures.md",
            ],
            "wiki_inputs": [
                "wiki/AGENTS.md",
                "wiki/entities/w42.md",
                "wiki/experiments/w42-final-empirical-strategy-report.md",
                "wiki/experiments/winning42-ch16-statistical-odds.md",
                "wiki/experiments/w42-claim-ledger.md",
                "wiki/experiments/w42-odds-ruleset-claim-validation.md",
                "wiki/experiments/w42-bidding-risk-budget-claim-validation.md",
                "wiki/experiments/w42-doubles-no-trump-claim-validation.md",
                "wiki/experiments/w42-scoring-objective-drift-claim-validation.md",
                "wiki/experiments/w42-84-claim-validation.md",
            ],
            "w42_artifact_inputs": [
                "w42/odds_ruleset_claim_validation/",
                "w42/bidding_risk_budget_claim_validation/",
                "w42/doubles_no_trump_claim_validation/",
                "w42/scoring_objective_drift_claim_validation/",
                "w42/eighty_four_claim_validation/",
                "w42/partner_support_claim_validation/",
                "w42/setter_defense_claim_validation/",
                "w42/bidder_sequencing_claim_validation/",
            ],
        },
        "artifacts": {
            "claims_csv": str(CLAIMS_CSV.relative_to(ROOT)),
            "summary_json": str(SUMMARY_JSON.relative_to(ROOT)),
            "manifest_json": str(MANIFEST_JSON.relative_to(ROOT)),
            "script": str((OUT_DIR / "build_statistics_claims_ledger.py").relative_to(ROOT)),
        },
        "wandb_hf_note": (
            "No new W&B or HuggingFace run/artifact was created for this synthesis. "
            "Rows retain source W&B links where an upstream validation artifact logged one; HF remains not applicable."
        ),
        "claim_ledger_impact": (
            "Phase-2 statistics ledger created as a local artifact and report page. "
            "No central ledger, wiki index, entity page, decision page, or bead file was updated."
        ),
        "caveats": [
            "Static enumeration supports arithmetic, prevalence, and ruleset substrate claims, not strategy optimality.",
            "Proxy model/regret reports are retained with their original conservative statuses.",
            "Hidden ownership is used only for report-time odds or evaluation labels, not live-agent features.",
            "Book preview OCR lines are cited sparingly; the CSV points to paths and line ranges rather than copying prose.",
        ],
    }


def build_manifest(rows: list[dict[str, str]], generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_VERSION,
        "owner_bead": OWNER_BEAD,
        "generated_at_utc": generated_at,
        "git_sha": git_sha(),
        "generator": str((OUT_DIR / "build_statistics_claims_ledger.py").relative_to(ROOT)),
        "command": "python w42/statistics_claims_ledger/build_statistics_claims_ledger.py",
        "check_command": "python w42/statistics_claims_ledger/build_statistics_claims_ledger.py --check",
        "status_vocabulary": sorted(ALLOWED_STATUSES),
        "claim_count": len(rows),
        "inputs": [
            "wiki/AGENTS.md",
            "wiki/entities/w42.md",
            "wiki/experiments/w42-final-empirical-strategy-report.md",
            "wiki/experiments/winning42-ch16-statistical-odds.md",
            "wiki/experiments/w42-claim-ledger.md",
            "scratch/winning42/winning42.with_figures.md",
            "w42/odds_ruleset_claim_validation/validation_summary.json",
            "w42/odds_ruleset_claim_validation/*.csv",
            "w42/bidding_risk_budget_claim_validation/summary.json",
            "w42/bidding_risk_budget_claim_validation/*.csv",
            "w42/doubles_no_trump_claim_validation/summary.json",
            "w42/doubles_no_trump_claim_validation/*.csv",
            "w42/scoring_objective_drift_claim_validation/claim_ledger_delta.json",
            "w42/scoring_objective_drift_claim_validation/*.csv",
            "w42/eighty_four_claim_validation/summary.json",
            "w42/eighty_four_claim_validation/*.csv",
            "w42/partner_support_claim_validation/summary.json",
            "w42/partner_support_claim_validation/claim_proxy_stats.csv",
            "w42/setter_defense_claim_validation/summary.json",
            "w42/setter_defense_claim_validation/claim_proxy_stats.csv",
            "w42/bidder_sequencing_claim_validation/summary.json",
            "w42/bidder_sequencing_claim_validation/claim_summary.csv",
        ],
        "outputs": [
            str(CLAIMS_CSV.relative_to(ROOT)),
            str(SUMMARY_JSON.relative_to(ROOT)),
            str(MANIFEST_JSON.relative_to(ROOT)),
        ],
        "validation": {
            "row_statuses_checked": True,
            "unique_claim_ids_checked": True,
            "referenced_local_artifacts_checked_by_check_command": True,
        },
        "no_new_external_artifacts": {
            "wandb": True,
            "huggingface": True,
        },
    }


def write_outputs() -> None:
    rows = build_rows()
    generated_at = datetime.now(timezone.utc).isoformat()
    with CLAIMS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_JSON.write_text(json.dumps(build_summary(rows, generated_at), indent=2) + "\n")
    MANIFEST_JSON.write_text(json.dumps(build_manifest(rows, generated_at), indent=2) + "\n")
    print(f"wrote {len(rows)} claims to {CLAIMS_CSV.relative_to(ROOT)}")


def check_outputs() -> None:
    for path in [CLAIMS_CSV, SUMMARY_JSON, MANIFEST_JSON]:
        if not path.exists():
            raise SystemExit(f"missing output: {path.relative_to(ROOT)}")

    with CLAIMS_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("claims.csv has no rows")
    if list(rows[0].keys()) != CSV_FIELDS:
        raise SystemExit("claims.csv header does not match expected fields")

    claim_ids = [row["claim_id"] for row in rows]
    duplicates = [claim for claim, count in Counter(claim_ids).items() if count > 1]
    if duplicates:
        raise SystemExit(f"duplicate claim ids: {duplicates}")

    bad_status = sorted({row["status"] for row in rows} - ALLOWED_STATUSES)
    if bad_status:
        raise SystemExit(f"bad statuses: {bad_status}")

    summary = json.loads(SUMMARY_JSON.read_text())
    manifest = json.loads(MANIFEST_JSON.read_text())
    if summary["claim_count"] != len(rows):
        raise SystemExit("summary claim_count mismatch")
    if manifest["claim_count"] != len(rows):
        raise SystemExit("manifest claim_count mismatch")
    if dict(sorted(Counter(row["status"] for row in rows).items())) != summary["status_counts"]:
        raise SystemExit("summary status_counts mismatch")

    for output in manifest["outputs"]:
        if not (ROOT / output).exists():
            raise SystemExit(f"manifest output missing: {output}")

    missing: list[str] = []
    for row in rows:
        for artifact in row["evidence_artifacts"].split("; "):
            if artifact in {"not applicable", ""} or artifact.startswith("http"):
                continue
            if not (ROOT / artifact).exists():
                missing.append(f"{row['claim_id']} -> {artifact}")
    if missing:
        raise SystemExit("missing evidence artifacts:\n" + "\n".join(missing))

    print(f"statistics claims ledger validation ok ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate existing outputs")
    args = parser.parse_args()
    if args.check:
        check_outputs()
    else:
        write_outputs()


if __name__ == "__main__":
    main()
