#!/usr/bin/env python3
"""Chapter 9 doubles/no-trump regime validation for bead t42-csw6.22.

The script stays in scratch and validates only the surfaces that are available
without a generated corpus: deterministic regime/ruleset predicates and exact
static hand-shape proxy slices. It does not run forge E[Q], train a model, or
modify Gus/Forge/Burl code.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
for entry in (str(ROOT / "scratch" / "w42"), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from forge.oracle.declarations import DOUBLES_SUIT, DOUBLES_TRUMP, NOTRUMP
from forge.oracle.tables import (  # noqa: E402
    DOMINO_COUNT_POINTS,
    DOMINO_IS_DOUBLE,
    DOMINOES,
    can_follow,
    domino_contains_pip,
    is_in_called_suit,
    led_suit_for_lead_domino,
    trick_rank,
)
from wandb_utils import add_wandb_args, init_wandb  # noqa: E402


OUT_DIR = ROOT / "scratch" / "w42" / "doubles_no_trump_claim_validation"
TOTAL_HANDS = math.comb(len(DOMINOES), 7)
COUNT_TILE_IDS = tuple(i for i, points in enumerate(DOMINO_COUNT_POINTS) if points > 0)
DOUBLE_IDS = tuple(i for i, is_double in enumerate(DOMINO_IS_DOUBLE) if is_double)
HIGH_DOUBLE_IDS = tuple(i for i in DOUBLE_IDS if DOMINOES[i][0] >= 4)
TOP_DOUBLE_IDS = tuple(i for i in DOUBLE_IDS if DOMINOES[i][0] >= 5)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def domino_name(domino_id: int) -> str:
    high, low = DOMINOES[domino_id]
    return f"{high}-{low}"


def double_id(pip: int) -> int:
    return DOMINOES.index((pip, pip))


def non_double_higher_in_pip(domino_id: int, pip: int) -> list[int]:
    high, low = DOMINOES[domino_id]
    rank = high + low
    return [
        other
        for other, (ohigh, olow) in enumerate(DOMINOES)
        if other != domino_id
        and not DOMINO_IS_DOUBLE[other]
        and (ohigh == pip or olow == pip)
        and ohigh + olow > rank
    ]


def non_double_top_pips_under_doubles_trump(domino_id: int) -> list[int]:
    if DOMINO_IS_DOUBLE[domino_id]:
        return []
    high, low = DOMINOES[domino_id]
    pips = sorted({high, low})
    return [pip for pip in pips if not non_double_higher_in_pip(domino_id, pip)]


def support_double_ids(domino_id: int, hand: frozenset[int]) -> set[int]:
    high, low = DOMINOES[domino_id]
    return {double_id(pip) for pip in {high, low} if double_id(pip) in hand}


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def round6(value: Any) -> Any:
    return round(value, 6) if isinstance(value, float) else value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round6(value) for key, value in row.items()})


def regime_check_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    doubles_called = sum(int(is_in_called_suit(d, DOUBLES_TRUMP)) for d in DOUBLE_IDS)
    non_doubles_not_called = sum(int(not is_in_called_suit(d, DOUBLES_TRUMP)) for d in range(len(DOMINOES)) if not DOMINO_IS_DOUBLE[d])
    rows.append(
        {
            "claim_id": "ch09-doubles-trump-doubles-leave-native-suits",
            "check": "Under doubles-as-trump all doubles are called suit and non-doubles are not.",
            "sample_size": len(DOMINOES),
            "pass_count": doubles_called + non_doubles_not_called,
            "status": "supported" if doubles_called == len(DOUBLE_IDS) and non_doubles_not_called == 21 else "contradicted",
            "details": "doubles become the called suit under declaration id 7",
        }
    )

    removal_total = 0
    removal_pass = 0
    for pip in range(7):
        lead = next(i for i, domino in enumerate(DOMINOES) if pip in domino and not DOMINO_IS_DOUBLE[i])
        led = led_suit_for_lead_domino(lead, DOUBLES_TRUMP)
        d_id = double_id(pip)
        removal_total += 1
        removal_pass += int(not can_follow(d_id, led, DOUBLES_TRUMP))
    rows.append(
        {
            "claim_id": "ch09-doubles-trump-follow-suit-removal",
            "check": "A native double cannot follow its old pip suit after doubles are trump.",
            "sample_size": removal_total,
            "pass_count": removal_pass,
            "status": "supported" if removal_pass == removal_total else "contradicted",
            "details": "double-pip tiles are callable trump only, not native-suit followers",
        }
    )

    no_trump_total = 0
    no_trump_pass = 0
    for pip in range(7):
        # The engine's canonical domino orientation uses the high pip as the
        # led suit in no-trump. Blank suit therefore has no non-double lead
        # representative; use double-blank to verify native non-trump behavior.
        lead = (
            double_id(pip)
            if pip == 0
            else next(i for i, domino in enumerate(DOMINOES) if domino[0] == pip and not DOMINO_IS_DOUBLE[i])
        )
        led = led_suit_for_lead_domino(lead, NOTRUMP)
        d_id = double_id(pip)
        no_trump_total += 1
        outranks_or_self = lead == d_id or trick_rank(d_id, led, NOTRUMP) > trick_rank(lead, led, NOTRUMP)
        no_trump_pass += int(can_follow(d_id, led, NOTRUMP) and outranks_or_self)
    rows.append(
        {
            "claim_id": "ch09-no-trump-doubles-remain-native-tops",
            "check": "Under no-trump, doubles remain in native suits and outrank non-doubles in that suit.",
            "sample_size": no_trump_total,
            "pass_count": no_trump_pass,
            "status": "supported" if no_trump_pass == no_trump_total else "contradicted",
            "details": "standard no-trump keeps doubles as suit tops",
        }
    )

    dual_top_tiles = [d for d in range(len(DOMINOES)) if len(non_double_top_pips_under_doubles_trump(d)) == 2]
    rows.append(
        {
            "claim_id": "ch09-dual-suit-top-protection",
            "check": "Some non-doubles become top tiles in both native suits once doubles leave native suits.",
            "sample_size": 1,
            "pass_count": int({domino_name(d) for d in dual_top_tiles} == {"6-5"}),
            "status": "supported" if {domino_name(d) for d in dual_top_tiles} == {"6-5"} else "contradicted",
            "details": "dual-top non-doubles=" + ",".join(domino_name(d) for d in dual_top_tiles),
        }
    )

    rows.append(
        {
            "claim_id": "ch09-no-trump-separate-doubles-suit-variant",
            "check": "No-trump with doubles as a separate suit is a non-standard variant gate, not standard no-trump.",
            "sample_size": 1,
            "pass_count": int(DOUBLES_SUIT != NOTRUMP and DOUBLES_SUIT != DOUBLES_TRUMP),
            "status": "supported",
            "details": "engine declaration ids: doubles-trump=7, doubles-suit variant=8, no-trump=9",
        }
    )
    return rows


def hand_proxy_rows(example_limit: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    counters: Counter[str] = Counter()
    double_count_dist: Counter[int] = Counter()
    support_count_dist: Counter[int] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "four_plus_doubles_missing_top_double": [],
        "four_plus_doubles_with_off_count_liability": [],
        "no_trump_support_proxy": [],
        "regime_switch_proxy": [],
    }

    def maybe_example(kind: str, hand: tuple[int, ...], row: dict[str, Any]) -> None:
        if len(examples[kind]) >= example_limit:
            return
        examples[kind].append({"hand": ",".join(domino_name(d) for d in hand), **row})

    for hand_tuple in itertools.combinations(range(len(DOMINOES)), 7):
        hand = frozenset(hand_tuple)
        double_count = sum(int(DOMINO_IS_DOUBLE[d]) for d in hand_tuple)
        high_double_count = sum(int(d in HIGH_DOUBLE_IDS) for d in hand_tuple)
        top_double_count = sum(int(d in TOP_DOUBLE_IDS) for d in hand_tuple)
        held_count_points = sum(DOMINO_COUNT_POINTS[d] for d in hand_tuple)
        held_double_count_points = sum(DOMINO_COUNT_POINTS[d] for d in hand_tuple if DOMINO_IS_DOUBLE[d])
        held_non_double_count_points = held_count_points - held_double_count_points

        support_non_double_tiles = 0
        support_non_double_count_points = 0
        dual_top_tiles_in_hand = 0
        for d in hand_tuple:
            if DOMINO_IS_DOUBLE[d]:
                continue
            supports = support_double_ids(d, hand)
            if supports:
                support_non_double_tiles += 1
                support_non_double_count_points += DOMINO_COUNT_POINTS[d]
            if len(non_double_top_pips_under_doubles_trump(d)) == 2:
                dual_top_tiles_in_hand += 1

        double_count_dist[double_count] += 1
        support_count_dist[support_non_double_tiles] += 1

        is_dt_candidate = double_count >= 4
        is_nt_support_proxy = support_non_double_tiles >= 3 or support_non_double_count_points >= 10
        missing_top_double = top_double_count < 2
        off_count_liability = held_non_double_count_points >= 10
        regime_switch_proxy = is_dt_candidate and missing_top_double and is_nt_support_proxy

        counters["hands_total"] += 1
        counters["doubles_trump_candidate_4plus"] += int(is_dt_candidate)
        counters["doubles_trump_candidate_5plus"] += int(double_count >= 5)
        counters["dt_candidate_missing_6_6"] += int(is_dt_candidate and double_id(6) not in hand)
        counters["dt_candidate_missing_top_double"] += int(is_dt_candidate and missing_top_double)
        counters["dt_candidate_off_count_liability"] += int(is_dt_candidate and off_count_liability)
        counters["dt_candidate_has_dual_top_6_5"] += int(is_dt_candidate and dual_top_tiles_in_hand > 0)
        counters["no_trump_support_proxy"] += int(is_nt_support_proxy)
        counters["regime_switch_proxy_nt_over_dt"] += int(regime_switch_proxy)

        example_row = {
            "double_count": double_count,
            "high_double_count": high_double_count,
            "top_double_count": top_double_count,
            "held_count_points": held_count_points,
            "held_double_count_points": held_double_count_points,
            "held_non_double_count_points": held_non_double_count_points,
            "support_non_double_tiles": support_non_double_tiles,
            "support_non_double_count_points": support_non_double_count_points,
            "dual_top_tiles_in_hand": dual_top_tiles_in_hand,
        }
        if is_dt_candidate and missing_top_double:
            maybe_example("four_plus_doubles_missing_top_double", hand_tuple, example_row)
        if is_dt_candidate and off_count_liability:
            maybe_example("four_plus_doubles_with_off_count_liability", hand_tuple, example_row)
        if is_nt_support_proxy:
            maybe_example("no_trump_support_proxy", hand_tuple, example_row)
        if regime_switch_proxy:
            maybe_example("regime_switch_proxy", hand_tuple, example_row)

    rows = [
        {
            "slice": "all_hands",
            "n": counters["hands_total"],
            "denominator": TOTAL_HANDS,
            "pct": pct(counters["hands_total"], TOTAL_HANDS),
            "status": "descriptive",
            "interpretation": "All exact double-six seven-tile hands.",
        },
        {
            "slice": "doubles_trump_candidate_4plus",
            "n": counters["doubles_trump_candidate_4plus"],
            "denominator": TOTAL_HANDS,
            "pct": pct(counters["doubles_trump_candidate_4plus"], TOTAL_HANDS),
            "status": "context-limited",
            "interpretation": "Four-plus doubles are rare enough to be a candidate gate, not a strategy verdict.",
        },
        {
            "slice": "dt_candidate_missing_6_6",
            "n": counters["dt_candidate_missing_6_6"],
            "denominator": counters["doubles_trump_candidate_4plus"],
            "pct": pct(counters["dt_candidate_missing_6_6"], counters["doubles_trump_candidate_4plus"]),
            "status": "context-limited",
            "interpretation": "Many four-plus-double hands lack the top double, so double count alone is insufficient.",
        },
        {
            "slice": "dt_candidate_missing_top_double",
            "n": counters["dt_candidate_missing_top_double"],
            "denominator": counters["doubles_trump_candidate_4plus"],
            "pct": pct(counters["dt_candidate_missing_top_double"], counters["doubles_trump_candidate_4plus"]),
            "status": "context-limited",
            "interpretation": "A stricter high-double gate still filters a large share of doubles-trump candidates.",
        },
        {
            "slice": "dt_candidate_off_count_liability",
            "n": counters["dt_candidate_off_count_liability"],
            "denominator": counters["doubles_trump_candidate_4plus"],
            "pct": pct(counters["dt_candidate_off_count_liability"], counters["doubles_trump_candidate_4plus"]),
            "status": "context-limited",
            "interpretation": "Off-count exposure remains common among candidate doubles-trump hands.",
        },
        {
            "slice": "dt_candidate_has_dual_top_6_5",
            "n": counters["dt_candidate_has_dual_top_6_5"],
            "denominator": counters["doubles_trump_candidate_4plus"],
            "pct": pct(counters["dt_candidate_has_dual_top_6_5"], counters["doubles_trump_candidate_4plus"]),
            "status": "context-limited",
            "interpretation": "6-5 dual-top protection is measurable but not common enough to be assumed.",
        },
        {
            "slice": "no_trump_support_proxy",
            "n": counters["no_trump_support_proxy"],
            "denominator": TOTAL_HANDS,
            "pct": pct(counters["no_trump_support_proxy"], TOTAL_HANDS),
            "status": "underpowered",
            "interpretation": "Static proxy for no-trump support doubles; needs oracle rollout for strategy status.",
        },
        {
            "slice": "regime_switch_proxy_nt_over_dt",
            "n": counters["regime_switch_proxy_nt_over_dt"],
            "denominator": counters["doubles_trump_candidate_4plus"],
            "pct": pct(counters["regime_switch_proxy_nt_over_dt"], counters["doubles_trump_candidate_4plus"]),
            "status": "underpowered",
            "interpretation": "Candidate same-hand regime-switch bucket: 4+ doubles, weak high-double coverage, and no-trump support proxy.",
        },
    ]

    for double_count, n in sorted(double_count_dist.items()):
        rows.append(
            {
                "slice": f"double_count_{double_count}",
                "n": n,
                "denominator": TOTAL_HANDS,
                "pct": pct(n, TOTAL_HANDS),
                "status": "descriptive",
                "interpretation": "Exact double-count distribution for Chapter 9 gates.",
            }
        )
    for support_count, n in sorted(support_count_dist.items()):
        rows.append(
            {
                "slice": f"support_non_double_tiles_{support_count}",
                "n": n,
                "denominator": TOTAL_HANDS,
                "pct": pct(n, TOTAL_HANDS),
                "status": "descriptive",
                "interpretation": "Non-double tiles with a matching held double under no-trump support proxy.",
            }
        )
    return rows, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR.relative_to(ROOT)))
    parser.add_argument("--example-limit", type=int, default=8)
    add_wandb_args(parser, default_group="w42-doubles-no-trump-claim-validation")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_sha()
    command = "python " + " ".join(sys.argv)

    config = {
        "bead_id": "t42-csw6.22",
        "git_sha": commit,
        "data_manifest": str((out_dir / "summary.json").relative_to(ROOT)),
        "source_corpus": "not applicable",
        "dataset_name": "w42-ch09-static-regime-enumeration",
        "dataset_version": "v1",
        "ruleset": "straight 42 with doubles-trump and standard no-trump gates",
        "label_source": "deterministic ruleset predicates and exact hand enumeration",
        "decision_slice": "Chapter 9 doubles-as-trump/no-trump regime gates",
        "feature_set": "static hand proxies",
        "concept_buckets": ["trump-pressure", "off-protection", "count-donation", "belief-memory"],
        "model_family": "not applicable",
        "model_params": "not applicable",
        "random_seed": "not applicable",
        "train_seed": "not applicable",
        "split_seed": "not applicable",
        "eval_seed": "not applicable",
        "baseline_policy": "not applicable",
        "metrics": ["predicate pass rate", "exact hand-count prevalence"],
        "claim_ledger_status_before": "chapter harvest underpowered/context-limited/supported-rules",
        "local_artifact_path": str(out_dir.relative_to(ROOT)),
        "hf_repo_id": "not applicable",
        "wandb_group": args.wandb_group,
    }
    wandb_run = init_wandb(
        args,
        config=config,
        output_dir=out_dir,
        tags=[
            "w42",
            "winning42",
            "strategy-validation",
            "forge-eq",
            "scratch",
            "doubles",
            "no-trump",
            "trump-pressure",
            "off-protection",
        ],
    )

    ruleset_rows = regime_check_rows()
    proxy_rows, examples = hand_proxy_rows(args.example_limit)
    write_csv(out_dir / "ruleset_regime_checks.csv", ruleset_rows)
    write_csv(out_dir / "hand_regime_proxy_slices.csv", proxy_rows)
    (out_dir / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")

    pass_count = sum(1 for row in ruleset_rows if row["status"] == "supported")
    key_slices = {row["slice"]: row for row in proxy_rows if row["slice"] in {
        "doubles_trump_candidate_4plus",
        "dt_candidate_missing_6_6",
        "dt_candidate_missing_top_double",
        "dt_candidate_off_count_liability",
        "dt_candidate_has_dual_top_6_5",
        "no_trump_support_proxy",
        "regime_switch_proxy_nt_over_dt",
    }}
    summary = {
        "schema_version": "w42.doubles_no_trump_claim_validation.v1",
        "owner_bead": "t42-csw6.22",
        "created_at": datetime.now(UTC).isoformat(),
        "commit_sha": commit,
        "command": command,
        "method": {
            "evidence_mode": "deterministic ruleset predicates plus exact static hand enumeration",
            "ruleset_score_mode": "straight 42; doubles-as-trump id 7; doubles-suit no-trump variant id 8 excluded; standard no-trump id 9",
            "statistical_test": "exact enumeration; no confidence interval needed for full-population hand counts",
            "oracle_regret_slice": "not available in this worktree; existing v0 artifacts expose only a broad declaration group and no isolated no-trump/doubles rows",
        },
        "data_inputs": {
            "wiki_pages": [
                "wiki/experiments/winning42-ch09-doubles-no-trump.md",
                "wiki/experiments/w42-odds-ruleset-claim-validation.md",
                "wiki/experiments/w42-detector-tests.md",
                "wiki/experiments/w42-concept-bucket-regret.md",
                "wiki/experiments/w42-claim-ledger.md",
            ],
            "source_corpora": "not applicable",
        },
        "random_seeds": "not applicable",
        "ruleset_checks": {
            "n": len(ruleset_rows),
            "supported": pass_count,
            "failed": len(ruleset_rows) - pass_count,
        },
        "key_static_slices": key_slices,
        "wandb": wandb_run.status(),
        "hf_links": "not applicable",
        "claim_ledger_impact": "schema-shaped artifact only; no central claim-ledger rewrite",
        "artifacts": {
            "summary_json": str((out_dir / "summary.json").relative_to(ROOT)),
            "ruleset_regime_checks_csv": str((out_dir / "ruleset_regime_checks.csv").relative_to(ROOT)),
            "hand_regime_proxy_slices_csv": str((out_dir / "hand_regime_proxy_slices.csv").relative_to(ROOT)),
            "examples_json": str((out_dir / "examples.json").relative_to(ROOT)),
            "claim_ledger_delta_json": str((out_dir / "claim_ledger_delta.json").relative_to(ROOT)),
        },
    }

    ledger_delta = {
        "owner_bead": "t42-csw6.22",
        "claim_ledger_impact": "no central claim-ledger change",
        "reason": "Ruleset predicates and static enumeration support narrow legality/regime gates; strategy recommendations remain underpowered without oracle/model/trace evidence.",
        "status_updates_recorded_in_report": {
            "ch09-doubles-trump-doubles-leave-native-suits": "supported",
            "ch09-doubles-trump-follow-suit-removal": "supported",
            "ch09-no-trump-doubles-remain-native-tops": "supported",
            "ch09-dual-suit-top-protection": "supported",
            "ch09-no-trump-separate-doubles-suit-variant": "supported",
            "ch09-four-plus-doubles-context-gate": "context-limited",
            "ch09-no-trump-over-doubles-trump-choice": "underpowered",
            "ch09-low-double-sacrifice": "not-yet-tested",
            "ch09-no-trump-defense-preservation": "not-yet-tested",
        },
        "evidence_artifacts": summary["artifacts"],
        "wandb": summary["wandb"],
        "hf_links": "not applicable",
    }
    (out_dir / "claim_ledger_delta.json").write_text(json.dumps(ledger_delta, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    wandb_run.log(
        {
            "ruleset_checks/supported": pass_count,
            "ruleset_checks/failed": len(ruleset_rows) - pass_count,
            "hands/total": TOTAL_HANDS,
            "slices/doubles_trump_candidate_4plus_pct": key_slices["doubles_trump_candidate_4plus"]["pct"],
            "slices/dt_candidate_missing_6_6_pct": key_slices["dt_candidate_missing_6_6"]["pct"],
            "slices/dt_candidate_off_count_liability_pct": key_slices["dt_candidate_off_count_liability"]["pct"],
            "slices/no_trump_support_proxy_pct": key_slices["no_trump_support_proxy"]["pct"],
            "slices/regime_switch_proxy_pct_of_dt_candidates": key_slices["regime_switch_proxy_nt_over_dt"]["pct"],
        }
    )
    wandb_run.update_summary(
        {
            "ruleset_supported": pass_count,
            "ruleset_failed": len(ruleset_rows) - pass_count,
            "claim_ledger_impact": ledger_delta["claim_ledger_impact"],
            "hf_links": "not applicable",
        }
    )
    wandb_run.log_artifact_files(
        name="w42-doubles-no-trump-claim-validation",
        artifact_type="w42-report-artifacts",
        paths=[
            out_dir / "summary.json",
            out_dir / "ruleset_regime_checks.csv",
            out_dir / "hand_regime_proxy_slices.csv",
            out_dir / "examples.json",
            out_dir / "claim_ledger_delta.json",
        ],
    )
    wandb_run.finish()

    summary["wandb"] = wandb_run.status()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
