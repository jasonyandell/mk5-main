#!/usr/bin/env python3
"""Report-only 84 claim validation for bead t42-csw6.21.

This lives under w42 and validates measurable static claims from Winning 42
Chapters 7/8 without changing Gus, Burl, or forge code. The checks are exact
double-six combinatorics and detector-readiness summaries; they are not oracle
rollouts and do not prove action optimality.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "w42" / "eighty_four_claim_validation"

_WANDB_UTILS_PATH = ROOT / "w42" / "wandb_utils.py"
_WANDB_SPEC = importlib.util.spec_from_file_location("w42_wandb_utils", _WANDB_UTILS_PATH)
if _WANDB_SPEC is None or _WANDB_SPEC.loader is None:
    raise RuntimeError(f"could not load W&B helper at {_WANDB_UTILS_PATH}")
_WANDB_MODULE = importlib.util.module_from_spec(_WANDB_SPEC)
_WANDB_SPEC.loader.exec_module(_WANDB_MODULE)
add_wandb_args = _WANDB_MODULE.add_wandb_args
init_wandb = _WANDB_MODULE.init_wandb


PIPS = tuple(range(7))
DOMINOES = tuple((high, low) for high in range(7) for low in range(high + 1))


def tile_name(tile: tuple[int, int]) -> str:
    return f"{tile[0]}-{tile[1]}"


def is_double(tile: tuple[int, int]) -> bool:
    return tile[0] == tile[1]


def contains(tile: tuple[int, int], pip: int) -> bool:
    return tile[0] == pip or tile[1] == pip


def called_suit_member(tile: tuple[int, int], trump: int) -> bool:
    return contains(tile, trump)


def same_suit_tiles(pip: int, trump: int) -> set[tuple[int, int]]:
    return {tile for tile in DOMINOES if contains(tile, pip) and not called_suit_member(tile, trump)}


def rank_in_suit(tile: tuple[int, int], pip: int) -> int:
    if is_double(tile) and contains(tile, pip):
        return 7
    if tile[0] == pip:
        return tile[1]
    if tile[1] == pip:
        return tile[0]
    raise ValueError(f"{tile} is not in suit {pip}")


def higher_same_suit_tiles(tile: tuple[int, int], pip: int, trump: int) -> set[tuple[int, int]]:
    rank = rank_in_suit(tile, pip)
    return {other for other in same_suit_tiles(pip, trump) if rank_in_suit(other, pip) > rank}


def n_choose(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def hypergeom_dist(successes: int, population: int = 21, draws: int = 7) -> dict[int, float]:
    denom = n_choose(population, draws)
    return {
        k: n_choose(successes, k) * n_choose(population - successes, draws - k) / denom
        for k in range(0, min(successes, draws) + 1)
    }


def prob_at_least(successes: int, threshold: int, population: int = 21, draws: int = 7) -> float:
    return sum(v for k, v in hypergeom_dist(successes, population, draws).items() if k >= threshold)


def mean_from_dist(dist: dict[int, float]) -> float:
    return sum(k * p for k, p in dist.items())


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_short_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()


def pct(value: float) -> float:
    return 100.0 * value


def round6(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def hand_summary(hand: Iterable[tuple[int, int]], trump: int) -> dict[str, Any]:
    tiles = set(hand)
    trumps = {tile for tile in tiles if called_suit_member(tile, trump)}
    non_trumps = tiles - trumps
    doubles = {tile for tile in tiles if is_double(tile)}
    non_trump_doubles = {tile for tile in non_trumps if is_double(tile)}
    plain_offs = non_trumps - non_trump_doubles
    off_double_suits = {
        pip
        for off in plain_offs
        for pip in off
        if (pip, pip) in tiles and not called_suit_member((pip, pip), trump)
    }
    one_off = next(iter(plain_offs)) if len(plain_offs) == 1 else None
    protecting_suits = sorted({pip for pip in one_off or () if (pip, pip) in tiles}) if one_off else []
    return {
        "trump_count": len(trumps),
        "non_trump_count": len(non_trumps),
        "plain_off_count": len(plain_offs),
        "non_trump_double_count": len(non_trump_doubles),
        "double_count": len(doubles),
        "off_double_suit_count": len(off_double_suits),
        "one_off": one_off,
        "protecting_suits": protecting_suits,
    }


def analyze_bidder_shapes() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shape_counter: Counter[str] = Counter()
    trump_counter: Counter[int] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    straight_weapon_counts: Counter[int] = Counter()
    protected_pair_counts: Counter[int] = Counter()

    total_hand_decls = 0
    one_off_total = 0
    protected_one_off_total = 0
    straight_one_off_total = 0
    three_trump_three_double_one_off = 0
    two_off_same_suit_with_protector = 0

    for hand in __import__("itertools").combinations(DOMINOES, 7):
        hand_set = set(hand)
        for trump in PIPS:
            total_hand_decls += 1
            summary = hand_summary(hand_set, trump)
            trump_counter[summary["trump_count"]] += 1

            if summary["plain_off_count"] == 1:
                one_off_total += 1
                off = summary["one_off"]
                assert off is not None
                weapon_tiles = set()
                for pip in off:
                    weapon_tiles |= higher_same_suit_tiles(off, pip, trump) - hand_set
                if summary["protecting_suits"]:
                    protected_one_off_total += 1
                    shape_counter["protected_one_off_84_shape"] += 1
                    examples.setdefault(
                        "protected_one_off_84_shape",
                        {
                            "trump": trump,
                            "hand": sorted(tile_name(t) for t in hand_set),
                            "off": tile_name(off),
                            "protecting_suits": summary["protecting_suits"],
                        },
                    )
                    for pip in summary["protecting_suits"]:
                        pair_pool = same_suit_tiles(pip, trump) - hand_set
                        protected_pair_counts[len(pair_pool)] += 1
                else:
                    straight_one_off_total += 1
                    shape_counter["straight_off_84_risk"] += 1
                    straight_weapon_counts[len(weapon_tiles)] += 1
                    examples.setdefault(
                        "straight_off_84_risk",
                        {
                            "trump": trump,
                            "hand": sorted(tile_name(t) for t in hand_set),
                            "off": tile_name(off),
                            "live_weapon_pool": sorted(tile_name(t) for t in weapon_tiles),
                        },
                    )

            if (
                summary["trump_count"] == 3
                and summary["plain_off_count"] == 1
                and summary["non_trump_double_count"] >= 3
            ):
                three_trump_three_double_one_off += 1
                shape_counter["three_trump_three_double_84"] += 1
                examples.setdefault(
                    "three_trump_three_double_84",
                    {"trump": trump, "hand": sorted(tile_name(t) for t in hand_set)},
                )

            if summary["plain_off_count"] == 2:
                non_trumps = hand_set - {tile for tile in hand_set if called_suit_member(tile, trump)}
                offs = [tile for tile in non_trumps if not is_double(tile)]
                shared = set(offs[0]) & set(offs[1])
                protected_shared = [pip for pip in shared if (pip, pip) in hand_set]
                if protected_shared:
                    two_off_same_suit_with_protector += 1
                    shape_counter["two_off_same_suit_84"] += 1
                    examples.setdefault(
                        "two_off_same_suit_84",
                        {
                            "trump": trump,
                            "hand": sorted(tile_name(t) for t in hand_set),
                            "offs": sorted(tile_name(t) for t in offs),
                            "shared_protecting_suits": protected_shared,
                        },
                    )

    shape_rows = [
        {
            "bucket": "all hand/declaration pairs",
            "n": total_hand_decls,
            "pct_of_all": 100.0,
            "claim_relevance": "denominator",
        },
        {
            "bucket": "one off candidate shape",
            "n": one_off_total,
            "pct_of_all": pct(one_off_total / total_hand_decls),
            "claim_relevance": "Chapter 7 canonical 84 search space",
        },
        {
            "bucket": "protected one off",
            "n": protected_one_off_total,
            "pct_of_all": pct(protected_one_off_total / total_hand_decls),
            "claim_relevance": "bidder has same-suit double ahead of final off",
        },
        {
            "bucket": "straight one off",
            "n": straight_one_off_total,
            "pct_of_all": pct(straight_one_off_total / total_hand_decls),
            "claim_relevance": "unprotected final off risk bucket",
        },
        {
            "bucket": "three trump / at least three doubles / one off",
            "n": three_trump_three_double_one_off,
            "pct_of_all": pct(three_trump_three_double_one_off / total_hand_decls),
            "claim_relevance": "Chapter 7 non-four-trump candidate bucket",
        },
        {
            "bucket": "two offs same suit with shared protecting double",
            "n": two_off_same_suit_with_protector,
            "pct_of_all": pct(two_off_same_suit_with_protector / total_hand_decls),
            "claim_relevance": "Chapter 7 two-off ordering bucket",
        },
    ]

    trump_rows = [
        {
            "trump_count": trump_count,
            "n": count,
            "pct_of_all": pct(count / total_hand_decls),
        }
        for trump_count, count in sorted(trump_counter.items())
    ]

    example_rows = [
        {"bucket": bucket, **value}
        for bucket, value in sorted(examples.items())
    ]

    return shape_rows, trump_rows, example_rows


def analyze_defender_assets() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    straight_weapon_rows = []
    protected_pair_rows = []
    final_off_cases = []
    protected_cases = []

    for off in DOMINOES:
        if is_double(off):
            continue
        for trump in PIPS:
            if called_suit_member(off, trump):
                continue
            bidder_base = {off}
            weapon_tiles = set()
            for pip in off:
                weapon_tiles |= higher_same_suit_tiles(off, pip, trump) - bidder_base
            final_off_cases.append((off, trump, len(weapon_tiles)))
            for pip in off:
                bidder_hand = {off, (pip, pip)}
                if called_suit_member((pip, pip), trump):
                    continue
                pair_pool = same_suit_tiles(pip, trump) - bidder_hand
                protected_cases.append((off, trump, pip, len(pair_pool)))

    weapon_pool_counts = Counter(size for _, _, size in final_off_cases)
    for size, cases in sorted(weapon_pool_counts.items()):
        dist = hypergeom_dist(size)
        p_1_to_4 = sum(p for k, p in dist.items() if 1 <= k <= 4)
        straight_weapon_rows.append(
            {
                "asset_pool_size": size,
                "final_off_cases": cases,
                "single_defender_mean_weapons": mean_from_dist(dist),
                "single_defender_p_zero": pct(dist.get(0, 0.0)),
                "single_defender_p_one_to_four": pct(p_1_to_4),
                "single_defender_p_more_than_four": pct(sum(p for k, p in dist.items() if k > 4)),
                "interpretation": "static last-trick weapon inventory before play pressure",
            }
        )

    pair_pool_counts = Counter(size for _, _, _, size in protected_cases)
    for size, cases in sorted(pair_pool_counts.items()):
        protected_pair_rows.append(
            {
                "same_suit_pair_pool_size": size,
                "protected_off_cases": cases,
                "single_defender_p_at_least_two": pct(prob_at_least(size, 2)),
                "single_defender_mean_pair_tiles": mean_from_dist(hypergeom_dist(size)),
                "interpretation": "static chance one defender starts with enough same-suit tiles for a pair line",
            }
        )

    # Specific odds claim: one named matching double is held by an opponent with
    # probability 14/21 = 2/3 when the bidder does not hold it.
    straight_weapon_rows.append(
        {
            "asset_pool_size": "single named matching double",
            "final_off_cases": "all straight-off cases",
            "single_defender_mean_weapons": "not applicable",
            "single_defender_p_zero": "not applicable",
            "single_defender_p_one_to_four": pct(14 / 21),
            "single_defender_p_more_than_four": "not applicable",
            "interpretation": "opponent-team ownership of one specific missing double is exactly two-to-one",
        }
    )
    straight_weapon_rows.append(
        {
            "asset_pool_size": "either of two matching doubles",
            "final_off_cases": "straight off with two unheld same-suit doubles",
            "single_defender_mean_weapons": "not applicable",
            "single_defender_p_zero": "not applicable",
            "single_defender_p_one_to_four": pct(1 - (7 / 21) * (6 / 20)),
            "single_defender_p_more_than_four": "not applicable",
            "interpretation": "opponent-team ownership of at least one of two missing doubles is 90%, not a two-to-one event",
        }
    )
    return straight_weapon_rows, protected_pair_rows


def claim_rows() -> list[dict[str, str]]:
    return [
        {
            "claim_id": "ch07-84-contract-regime",
            "bucket": "bidder",
            "status": "context-limited",
            "evidence": "rules/source-backed; no engine score replay in this bead",
            "caveat": "The report relies on existing chapter/ruleset pages for scoring semantics.",
        },
        {
            "claim_id": "ch07-protected-one-off-84-shape",
            "bucket": "bidder",
            "status": "underpowered",
            "evidence": "exact hand/declaration shape count; no make/set rollout",
            "caveat": "Shape existence and asset pools do not prove that bidding 84 is value-positive.",
        },
        {
            "claim_id": "ch07-straight-off-two-to-one-double",
            "bucket": "bidder",
            "status": "context-limited",
            "evidence": "exact opponent-team ownership odds for a named missing matching double",
            "caveat": "The 'nearly set two out of three' policy claim still needs rollouts.",
        },
        {
            "claim_id": "ch07-score-42-vs-84-gate",
            "bucket": "bidder",
            "status": "not-yet-tested",
            "evidence": "not measured",
            "caveat": "Needs terminal score-state oracle or match utility simulation.",
        },
        {
            "claim_id": "ch08-one-to-four-last-trick-weapons",
            "bucket": "defender",
            "status": "context-limited",
            "evidence": "exact static weapon-pool distribution by final-off case",
            "caveat": "Initial inventory ignores forced-follow pressure and bidder inference.",
        },
        {
            "claim_id": "ch08-double-ahead-needs-same-suit-pair",
            "bucket": "defender",
            "status": "underpowered",
            "evidence": "exact same-suit pair pool size/probability proxy",
            "caveat": "A pair existing initially is not a set path without final-two-trick simulation.",
        },
        {
            "claim_id": "ch08-abandon-dead-assets",
            "bucket": "defender",
            "status": "not-yet-tested",
            "evidence": "detector trigger table only",
            "caveat": "Needs public replay or oracle action counterfactuals to tell correct abandonment from lucky discard.",
        },
        {
            "claim_id": "ch08-throwaway-priority-ladder",
            "bucket": "defender",
            "status": "not-yet-tested",
            "evidence": "not measured",
            "caveat": "Needs legal discard/action-local states and regret labels.",
        },
    ]


def abandonment_rows() -> list[dict[str, str]]:
    return [
        {
            "trigger": "watched final-off suit exhausted",
            "asset_before": "live double or same-suit tile that only beats a watched final off",
            "abandon_when": "all higher same-suit target tiles are public-played, held by bidder, or no longer plausible final offs",
            "current_artifact_status": "designed, not replay-measured",
        },
        {
            "trigger": "bidder double-ahead protection revealed/proved",
            "asset_before": "single saved double in the protected suit",
            "abandon_when": "a lone double can no longer beat the bidder's final off because the bidder can force the suit with the double ahead",
            "current_artifact_status": "static proxy only; no final-two-trick proof",
        },
        {
            "trigger": "same-suit pair target killed",
            "asset_before": "two-tile same-suit pair plus optional protectors",
            "abandon_when": "the lower/upper target role is impossible because target tiles are public-dead or the pair was forced apart",
            "current_artifact_status": "detector map exists; no replay labels",
        },
        {
            "trigger": "protector spent or no longer needed",
            "asset_before": "side-suit protector guarding a live pair from forced follow",
            "abandon_when": "the pair is dead, the watched lead suit is exhausted, or legal follow pressure can no longer force the pair tile",
            "current_artifact_status": "not measured",
        },
        {
            "trigger": "partner kills target branch",
            "asset_before": "defender asset kept for a final-off branch partner can publicly eliminate",
            "abandon_when": "partner play/discard makes the bidder final-off branch impossible under public evidence",
            "current_artifact_status": "requires belief/replay; not measurable from current v0 artifacts",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round6(v) for k, v in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=42)
    add_wandb_args(
        parser,
        default_group="w42-csw6-84-claim-validation",
        default_enabled=True,
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    config = {
        "bead_id": "t42-csw6.21",
        "git_sha": sha,
        "data_manifest": "w42/eighty_four_claim_validation/summary.json",
        "source_corpus": "not applicable",
        "dataset_name": "w42-eighty-four-static-claim-validation",
        "dataset_version": "v1",
        "ruleset": "straight 42, pip-trump 84 static double-six combinatorics",
        "label_source": "Winning 42 chapter pages plus exact double-six enumeration",
        "decision_slice": "84 bidder and defender report-only static buckets",
        "feature_set": "strategy_tags_v1_map 84 detector families",
        "concept_buckets": ["eighty-four", "bidder", "defender", "last-trick-weapons"],
        "model_family": "not applicable",
        "model_params": "not applicable",
        "seed": args.seed,
        "random_seed": args.seed,
        "train_seed": "not applicable",
        "split_seed": "not applicable",
        "eval_seed": "not applicable",
        "baseline_policy": "not applicable",
        "metrics": ["shape counts", "hypergeometric asset distributions"],
        "claim_ledger_status_before": "chapter harvest statuses only; no central ledger mutation",
        "local_artifact_path": str(args.output_dir),
        "hf_repo_id": "not applicable",
        "wandb_group": args.wandb_group,
    }
    wandb_run = init_wandb(
        args,
        config=config,
        output_dir=args.output_dir,
        tags=["w42", "winning42", "strategy-validation", "eighty-four", "promoted", "t42-csw6.21"],
    )

    shape_rows, trump_rows, example_rows = analyze_bidder_shapes()
    weapon_rows, pair_rows = analyze_defender_assets()
    claims = claim_rows()
    abandon = abandonment_rows()

    write_csv(args.output_dir / "bidder_shape_summary.csv", shape_rows)
    write_csv(args.output_dir / "trump_count_distribution.csv", trump_rows)
    write_csv(args.output_dir / "example_cases.csv", example_rows)
    write_csv(args.output_dir / "defender_last_trick_weapon_distribution.csv", weapon_rows)
    write_csv(args.output_dir / "defender_same_suit_pair_distribution.csv", pair_rows)
    write_csv(args.output_dir / "claim_summary.csv", claims)
    write_csv(args.output_dir / "abandonment_trigger_table.csv", abandon)

    shape_index = {row["bucket"]: row for row in shape_rows}
    numeric_weapon_rows = [row for row in weapon_rows if isinstance(row["asset_pool_size"], int)]
    summary = {
        "schema_version": "w42.eighty_four_claim_validation.v1",
        "owner_bead": "t42-csw6.21",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": sha,
        "method": {
            "evidence_mode": "exact double-six enumeration plus report-only hypergeometric defender asset proxies",
            "ruleset": "straight 42 pip-trump declarations; 84 contract semantics sourced from chapter/ruleset pages",
            "statistical_test": "exact counts and exact hypergeometric probabilities; no random sampling",
            "random_seed": args.seed,
        },
        "source_pages": [
            "wiki/experiments/winning42-ch07-taking-every-trick-84.md",
            "wiki/experiments/winning42-ch08-setting-84.md",
            "wiki/experiments/winning42-ch12-advanced-bidding-playing.md",
            "wiki/experiments/winning42-ch13-optional-variations.md",
            "wiki/experiments/w42-strategy-tags-v1-map.md",
            "wiki/experiments/w42-detector-tests.md",
            "wiki/experiments/w42-concept-bucket-regret.md",
            "wiki/experiments/w42-claim-ledger.md",
        ],
        "headline_metrics": {
            "hand_declaration_pairs": shape_index["all hand/declaration pairs"]["n"],
            "one_off_candidate_pct": shape_index["one off candidate shape"]["pct_of_all"],
            "protected_one_off_pct": shape_index["protected one off"]["pct_of_all"],
            "straight_one_off_pct": shape_index["straight one off"]["pct_of_all"],
            "three_trump_three_double_one_off_pct": shape_index[
                "three trump / at least three doubles / one off"
            ]["pct_of_all"],
            "two_off_same_suit_with_protector_pct": shape_index[
                "two offs same suit with shared protecting double"
            ]["pct_of_all"],
            "straight_weapon_pool_sizes": sorted(
                {
                    int(row["asset_pool_size"])
                    for row in numeric_weapon_rows
                }
            ),
            "named_matching_double_opponent_team_probability_pct": pct(14 / 21),
            "either_matching_double_opponent_team_probability_pct": pct(1 - (7 / 21) * (6 / 20)),
        },
        "claim_statuses": {row["claim_id"]: row["status"] for row in claims},
        "artifacts": {
            "bidder_shape_summary_csv": "w42/eighty_four_claim_validation/bidder_shape_summary.csv",
            "trump_count_distribution_csv": "w42/eighty_four_claim_validation/trump_count_distribution.csv",
            "example_cases_csv": "w42/eighty_four_claim_validation/example_cases.csv",
            "defender_last_trick_weapon_distribution_csv": "w42/eighty_four_claim_validation/defender_last_trick_weapon_distribution.csv",
            "defender_same_suit_pair_distribution_csv": "w42/eighty_four_claim_validation/defender_same_suit_pair_distribution.csv",
            "claim_summary_csv": "w42/eighty_four_claim_validation/claim_summary.csv",
            "abandonment_trigger_table_csv": "w42/eighty_four_claim_validation/abandonment_trigger_table.csv",
            "claim_ledger_delta_json": "w42/eighty_four_claim_validation/claim_ledger_delta.json",
            "summary_json": "w42/eighty_four_claim_validation/summary.json",
        },
        "wandb": wandb_run.status(),
        "huggingface": "not applicable",
        "claim_ledger_impact": "no central claim-ledger change; report-local claim summary and delta artifact only",
    }

    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    ledger_delta = {
        "owner_bead": "t42-csw6.21",
        "claim_ledger_impact": "no central claim-ledger change",
        "reason": "This bead produced static enumeration/proxy evidence and a report-local status table; it did not update wiki/experiments/w42-claim-ledger.md.",
        "claim_statuses_reported_in_page": summary["claim_statuses"],
        "provenance": {
            "commands": [
                "python w42/eighty_four_claim_validation/validate_84_claims.py --wandb-mode online"
            ],
            "configs": "not applicable",
            "data_inputs": summary["source_pages"],
            "commit_sha": sha,
            "random_seeds": {"analysis_seed": args.seed},
            "wandb_links": summary["wandb"],
            "hf_links": "not applicable",
        },
    }
    with (args.output_dir / "claim_ledger_delta.json").open("w") as f:
        json.dump(ledger_delta, f, indent=2)
        f.write("\n")

    wandb_run.log(
        {
            "bidder/one_off_candidate_pct": shape_index["one off candidate shape"]["pct_of_all"],
            "bidder/protected_one_off_pct": shape_index["protected one off"]["pct_of_all"],
            "bidder/straight_one_off_pct": shape_index["straight one off"]["pct_of_all"],
            "defender/named_matching_double_opponent_team_pct": pct(14 / 21),
            "defender/either_matching_double_opponent_team_pct": pct(1 - (7 / 21) * (6 / 20)),
        }
    )
    wandb_run.update_summary(
        {
            "hand_declaration_pairs": summary["headline_metrics"]["hand_declaration_pairs"],
            "claim_ledger_impact": summary["claim_ledger_impact"],
            "hf_links": "not applicable",
        }
    )
    wandb_run.log_artifact_files(
        name=f"w42-84-claim-validation-{git_short_sha()}",
        artifact_type="report",
        paths=[
            args.output_dir / "summary.json",
            args.output_dir / "claim_summary.csv",
            args.output_dir / "bidder_shape_summary.csv",
            args.output_dir / "defender_last_trick_weapon_distribution.csv",
            args.output_dir / "defender_same_suit_pair_distribution.csv",
            args.output_dir / "abandonment_trigger_table.csv",
        ],
    )
    wandb_run.finish()
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "wandb": wandb_run.status()}, indent=2))


if __name__ == "__main__":
    main()
