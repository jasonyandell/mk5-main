#!/usr/bin/env python3
"""Report-only Chapter 5/12 setter-defense proxy validation for w42."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[3]
for entry in (str(ROOT / "scratch" / "w42"), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from gus.model.dataset_seq_world import JointWorldFullDataset
from strategy_tags_v0 import ACTION_TAGS, GLOBAL_TAGS, validate_tag_dims
from wandb_utils import add_wandb_args, init_wandb


Predicate = Callable[[dict[str, Any], int], bool]


@dataclass(frozen=True)
class SliceSpec:
    claim_id: str
    label: str
    preferred: Predicate
    alternative: Predicate
    source_claim: str
    proxy_note: str
    directness: str
    expected_direction: str = "preferred_lower_regret"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _idx(tags: list[Any]) -> dict[str, int]:
    return {tag.name: tag.idx for tag in tags}


G = _idx(GLOBAL_TAGS)
A = _idx(ACTION_TAGS)


def truthy(value: torch.Tensor, eps: float = 1e-6) -> bool:
    return bool(float(value.item()) > eps)


def count_points(row: dict[str, Any], action_idx: int) -> float:
    return float(row["actions"][action_idx, A["count_points_frac"]].item())


def is_count(row: dict[str, Any], action_idx: int) -> bool:
    return count_points(row, action_idx) > 1e-6


def is_non_count(row: dict[str, Any], action_idx: int) -> bool:
    return not is_count(row, action_idx)


def is_ten_count(row: dict[str, Any], action_idx: int) -> bool:
    return truthy(row["actions"][action_idx, A["is_ten_count"]])


def is_opponent_currently_winning(row: dict[str, Any], action_idx: int) -> bool:
    return truthy(row["actions"][action_idx, A["opponent_currently_winning"]])


def is_partner_currently_winning(row: dict[str, Any], action_idx: int) -> bool:
    return truthy(row["actions"][action_idx, A["partner_currently_winning"]])


def is_count_to_opponent(row: dict[str, Any], action_idx: int) -> bool:
    return is_count(row, action_idx) and truthy(row["actions"][action_idx, A["count_donation_to_opponent"]])


def is_non_count_to_opponent(row: dict[str, Any], action_idx: int) -> bool:
    return is_non_count(row, action_idx) and is_opponent_currently_winning(row, action_idx)


def is_ten_count_to_opponent(row: dict[str, Any], action_idx: int) -> bool:
    return is_count_to_opponent(row, action_idx) and is_ten_count(row, action_idx)


def is_five_count_to_opponent(row: dict[str, Any], action_idx: int) -> bool:
    return is_count_to_opponent(row, action_idx) and not is_ten_count(row, action_idx)


def is_opponent_count_pressure(row: dict[str, Any], action_idx: int) -> bool:
    return is_opponent_currently_winning(row, action_idx) and float(row["features"][G["current_trick_count_point_frac"]].item()) > 0


def is_opponent_no_count_pressure(row: dict[str, Any], action_idx: int) -> bool:
    return is_opponent_currently_winning(row, action_idx) and float(row["features"][G["current_trick_count_point_frac"]].item()) <= 1e-6


def is_high_liability_lead(row: dict[str, Any], action_idx: int) -> bool:
    action = row["actions"][action_idx]
    return truthy(action[A["is_lead_position"]]) and float(action[A["live_count_frac"]].item()) > 1e-6


def is_low_liability_lead(row: dict[str, Any], action_idx: int) -> bool:
    action = row["actions"][action_idx]
    return truthy(action[A["is_lead_position"]]) and float(action[A["live_count_frac"]].item()) <= 1e-6


def is_protected_count_action(row: dict[str, Any], action_idx: int) -> bool:
    action = row["actions"][action_idx]
    protected = truthy(action[A["protected_by_my_double"]]) or truthy(action[A["protected_by_high_double"]]) or truthy(action[A["protected_by_low_double"]])
    return is_count(row, action_idx) and protected


def is_unprotected_count_action(row: dict[str, Any], action_idx: int) -> bool:
    action = row["actions"][action_idx]
    protected = truthy(action[A["protected_by_my_double"]]) or truthy(action[A["protected_by_high_double"]]) or truthy(action[A["protected_by_low_double"]])
    return is_count(row, action_idx) and not protected


def is_trump_in_on_count(row: dict[str, Any], action_idx: int) -> bool:
    action = row["actions"][action_idx]
    return truthy(action[A["trump_in"]]) and is_count(row, action_idx)


def is_hold_count_when_opponent_winning(row: dict[str, Any], action_idx: int) -> bool:
    return is_count(row, action_idx) and not truthy(row["actions"][action_idx, A["trump_in"]]) and is_opponent_currently_winning(row, action_idx)


def bootstrap_ci(values: list[float], seed: int, n_boot: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    means.sort()
    return means[int(0.025 * (n_boot - 1))], means[int(0.975 * (n_boot - 1))]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: Any) -> str:
    if value is None:
        return "not applicable"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def load_rows(eval_path: Path, seed: int, limit: int) -> list[dict[str, Any]]:
    ds = JointWorldFullDataset([eval_path], seed=seed, include_strategy_features=True)
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        item = ds[i]
        legal_e_q = item["e_q"].masked_fill(~item["legal_mask"], float("-inf"))
        oracle_best = legal_e_q.max()
        rows.append(
            {
                "row_idx": i,
                "decision_idx": int(item["decision_idx"].item()),
                "player": int(item["player"].item()),
                "legal_mask": item["legal_mask"],
                "features": item["strategy_features"],
                "actions": item["strategy_action_features"],
                "e_q": item["e_q"],
                "oracle_action": int(legal_e_q.argmax().item()),
                "regrets": oracle_best - item["e_q"],
            }
        )
    return rows


def analyze_slice(rows: list[dict[str, Any]], spec: SliceSpec, seed: int, n_boot: int) -> dict[str, Any]:
    preferred_regrets: list[float] = []
    alternative_regrets: list[float] = []
    paired_diffs: list[float] = []
    preferred_actions = 0
    alternative_actions = 0
    decisions_with_preferred = 0
    decisions_with_alternative = 0
    paired_decisions = 0
    preferred_oracle_best = 0
    alternative_oracle_best = 0

    for row in rows:
        preferred_idxs: list[int] = []
        alternative_idxs: list[int] = []
        for action_idx in range(7):
            if not bool(row["legal_mask"][action_idx].item()):
                continue
            if spec.preferred(row, action_idx):
                preferred_idxs.append(action_idx)
                preferred_regrets.append(float(row["regrets"][action_idx].item()))
            if spec.alternative(row, action_idx):
                alternative_idxs.append(action_idx)
                alternative_regrets.append(float(row["regrets"][action_idx].item()))

        if preferred_idxs:
            decisions_with_preferred += 1
            preferred_actions += len(preferred_idxs)
            if row["oracle_action"] in preferred_idxs:
                preferred_oracle_best += 1
        if alternative_idxs:
            decisions_with_alternative += 1
            alternative_actions += len(alternative_idxs)
            if row["oracle_action"] in alternative_idxs:
                alternative_oracle_best += 1
        if preferred_idxs and alternative_idxs:
            paired_decisions += 1
            best_preferred = min(float(row["regrets"][idx].item()) for idx in preferred_idxs)
            best_alternative = min(float(row["regrets"][idx].item()) for idx in alternative_idxs)
            paired_diffs.append(best_preferred - best_alternative)

    paired_mean = mean(paired_diffs)
    paired_ci = bootstrap_ci(paired_diffs, seed + 23, n_boot)
    if spec.directness != "direct":
        verdict = "underpowered"
        status_reason = f"Only {spec.directness} evidence is available; the current v0 corpus lacks the direct setter-role/off-window labels needed for a claim move."
    elif paired_mean is None:
        verdict = "underpowered"
        status_reason = "No paired decisions expose both proxy action classes."
    elif paired_ci[1] is not None and paired_ci[1] < 0:
        verdict = "context-limited"
        status_reason = "Preferred proxy has lower paired oracle regret on this slice."
    elif paired_ci[0] is not None and paired_ci[0] > 0:
        verdict = "contradicted"
        status_reason = "Preferred proxy has higher paired oracle regret on this slice."
    else:
        verdict = "underpowered"
        status_reason = "Paired confidence interval crosses zero or sample is too small for a status move."

    return {
        "claim_id": spec.claim_id,
        "label": spec.label,
        "source_claim": spec.source_claim,
        "proxy_note": spec.proxy_note,
        "directness": spec.directness,
        "preferred_action_n": preferred_actions,
        "alternative_action_n": alternative_actions,
        "decisions_with_preferred": decisions_with_preferred,
        "decisions_with_alternative": decisions_with_alternative,
        "paired_decision_n": paired_decisions,
        "preferred_mean_regret": mean(preferred_regrets),
        "preferred_mean_regret_ci95": bootstrap_ci(preferred_regrets, seed + 11, n_boot),
        "alternative_mean_regret": mean(alternative_regrets),
        "alternative_mean_regret_ci95": bootstrap_ci(alternative_regrets, seed + 17, n_boot),
        "paired_preferred_minus_alternative_mean_regret": paired_mean,
        "paired_ci95": paired_ci,
        "preferred_oracle_best_decision_rate": preferred_oracle_best / decisions_with_preferred if decisions_with_preferred else None,
        "alternative_oracle_best_decision_rate": alternative_oracle_best / decisions_with_alternative if decisions_with_alternative else None,
        "verdict": verdict,
        "status_reason": status_reason,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "claim_id",
        "label",
        "directness",
        "preferred_action_n",
        "alternative_action_n",
        "paired_decision_n",
        "preferred_mean_regret",
        "preferred_ci95_low",
        "preferred_ci95_high",
        "alternative_mean_regret",
        "alternative_ci95_low",
        "alternative_ci95_high",
        "paired_preferred_minus_alternative_mean_regret",
        "paired_ci95_low",
        "paired_ci95_high",
        "preferred_oracle_best_decision_rate",
        "alternative_oracle_best_decision_rate",
        "verdict",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "claim_id": row["claim_id"],
                    "label": row["label"],
                    "directness": row["directness"],
                    "preferred_action_n": row["preferred_action_n"],
                    "alternative_action_n": row["alternative_action_n"],
                    "paired_decision_n": row["paired_decision_n"],
                    "preferred_mean_regret": fmt(row["preferred_mean_regret"]),
                    "preferred_ci95_low": fmt(row["preferred_mean_regret_ci95"][0]),
                    "preferred_ci95_high": fmt(row["preferred_mean_regret_ci95"][1]),
                    "alternative_mean_regret": fmt(row["alternative_mean_regret"]),
                    "alternative_ci95_low": fmt(row["alternative_mean_regret_ci95"][0]),
                    "alternative_ci95_high": fmt(row["alternative_mean_regret_ci95"][1]),
                    "paired_preferred_minus_alternative_mean_regret": fmt(row["paired_preferred_minus_alternative_mean_regret"]),
                    "paired_ci95_low": fmt(row["paired_ci95"][0]),
                    "paired_ci95_high": fmt(row["paired_ci95"][1]),
                    "preferred_oracle_best_decision_rate": fmt(row["preferred_oracle_best_decision_rate"]),
                    "alternative_oracle_best_decision_rate": fmt(row["alternative_oracle_best_decision_rate"]),
                    "verdict": row["verdict"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/w42/setter_defense_claim_validation"))
    parser.add_argument("--eval-seed", type=int, default=43)
    parser.add_argument("--bootstrap-seed", type=int, default=20260502)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0)
    add_wandb_args(parser, default_group="t42-csw6", default_enabled=True)
    args = parser.parse_args()

    validate_tag_dims()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "bead_id": "t42-csw6.20",
        "git_sha": git_sha(),
        "seed": args.bootstrap_seed,
        "eval_seed": args.eval_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "data_manifest": "scratch/w42/v0_strategy_tags_baseline/manifest.json",
        "source_corpus": str(args.eval),
        "dataset_name": "w42-gus-eval-public-state-v0-tags",
        "dataset_version": "local-corpus-eval-20",
        "ruleset": "existing Gus corpus semantics",
        "label_source": "forge E[Q] values stored in Gus corpus rows",
        "decision_slice": "all eval decisions; setter-defense proxies only",
        "feature_set": "w42 strategy tags v0 public/action-local proxies",
        "concept_buckets": ["setter-pounce", "void-creation", "count-on-off", "trump-set", "count-protection"],
        "model_family": "not applicable",
        "model_params": "not applicable",
        "random_seed": args.bootstrap_seed,
        "train_seed": "not applicable",
        "split_seed": "not applicable",
        "baseline_policy": "oracle best E[Q] per decision",
        "metrics": "candidate-action regret, paired best-proxy regret, oracle-best decision rate",
        "claim_ledger_status_before": "not-yet-tested/underpowered chapter claims",
        "local_artifact_path": str(output_dir),
        "hf_repo_id": "not applicable",
        "wandb_group": args.wandb_group,
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=output_dir,
        tags=["w42", "winning42", "strategy-validation", "forge-eq", "gus-format", "scratch", "setter-defense", "pounce-window", "t42-csw6.20"],
    )

    rows = load_rows(args.eval, args.eval_seed, args.limit)
    specs = [
        SliceSpec(
            claim_id="ch05-pounce-count-before-certainty",
            label="Count into opponent-currently-winning window versus non-count in same window",
            preferred=is_count_to_opponent,
            alternative=is_non_count_to_opponent,
            source_claim="Setter should often play count on the bidder's off before knowing whether partner wins because the pounce window may not recur.",
            proxy_note="Uses v0 opponent-currently-winning and count-donation tags. It does not know bidder identity, off-suit exposure, defender role, or partner's hidden winner.",
            directness="proxy-only",
        ),
        SliceSpec(
            claim_id="ch05-extra-count-to-set",
            label="Ten-count pounce proxy versus five-count pounce proxy",
            preferred=is_ten_count_to_opponent,
            alternative=is_five_count_to_opponent,
            source_claim="Extra count on an off trick can set a bidder who priced only a smaller count loss into the bid.",
            proxy_note="Uses count magnitude in opponent-currently-winning windows; lacks bid margin and set-threshold accounting.",
            directness="proxy-only",
        ),
        SliceSpec(
            claim_id="ch05-count-calling-lead",
            label="High-liability count-calling lead versus low-liability lead",
            preferred=is_high_liability_lead,
            alternative=is_low_liability_lead,
            source_claim="After setters win lead, count-calling leads can attack the bidder's count exposure.",
            proxy_note="Uses live-count-by-led-suit as a broad lead-pressure proxy; it does not isolate setter lead after winning a trick.",
            directness="broad-proxy",
        ),
        SliceSpec(
            claim_id="ch05-count-protection",
            label="Protected count action versus unprotected count action",
            preferred=is_protected_count_action,
            alternative=is_unprotected_count_action,
            source_claim="Throwaway choices should protect count dominoes by preserving same-suit protectors where possible.",
            proxy_note="v0 says whether an action's tile is protected by doubles; it does not say whether the candidate discard preserves or spends the protector itself.",
            directness="broad-proxy",
        ),
        SliceSpec(
            claim_id="ch05-trump-rich-count-intervention",
            label="Trumping in on count versus holding count while opponent currently wins",
            preferred=is_trump_in_on_count,
            alternative=is_hold_count_when_opponent_winning,
            source_claim="Trump-rich setters should decide whether to trump in on likely count based on position, rank strength, and trick content.",
            proxy_note="Uses trump-in and count tags only; missing trump-rich-setter ownership, position-specific trump-set signal, and bid-margin state.",
            directness="proxy-only",
        ),
        SliceSpec(
            claim_id="ch12-setter-pounce-high-bid-off",
            label="Opponent count pressure present versus absent",
            preferred=is_opponent_count_pressure,
            alternative=is_opponent_no_count_pressure,
            source_claim="Advanced setter pounces should play count immediately on bidder high-bid off windows.",
            proxy_note="Broad current-trick count-pressure proxy only; no high-bid/off-suit gate exists in v0 artifacts.",
            directness="broad-proxy",
        ),
    ]
    results = [analyze_slice(rows, spec, args.bootstrap_seed + i * 1000, args.bootstrap_samples) for i, spec in enumerate(specs)]

    missing_detectors = [
        {
            "bucket": "void creation",
            "status": "missing direct detector",
            "needed_detector": "creates_void / later pounce attribution",
            "source": "winning42-ch05-setter-defense deliberate void creation; winning42-ch03 extra trump creates setter pounce",
        },
        {
            "bucket": "trump-set detection",
            "status": "missing direct detector",
            "needed_detector": "trump_set_signal / trump_rich_defender / partner recognition after only-one-opponent-followed-trump",
            "source": "winning42-ch05-setter-defense trump-set recognition and trump-rich setter policy",
        },
        {
            "bucket": "bid-margin set accounting",
            "status": "missing direct detector",
            "needed_detector": "points_needed_to_set / bidder_loss_allowance / set-threshold accounting",
            "source": "winning42-ch05 setter risk-budget and extra-count-to-set claims",
        },
        {
            "bucket": "high-bid off pounce",
            "status": "missing direct detector",
            "needed_detector": "bidder_off_window gated by 35/36 bid bucket and defender void/winning-double state",
            "source": "winning42-ch12 setter pounce with count on bidder off",
        },
    ]

    summary = {
        "schema_version": "w42.setter_defense_claim_validation.v1",
        "bead_id": "t42-csw6.20",
        "created_at": datetime.now(UTC).isoformat(),
        "repo_commit": config["git_sha"],
        "command": " ".join(sys.argv),
        "data_inputs": [str(args.eval)],
        "sample": {
            "eval_rows": len(rows),
            "legal_candidate_actions": sum(int(row["legal_mask"].sum().item()) for row in rows),
        },
        "random_seeds": {
            "eval_dataset_seed": args.eval_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_samples": args.bootstrap_samples,
            "data_generation": "not applicable",
            "dataset_shuffle": "not applicable",
            "train": "not applicable",
        },
        "statistical_method": "Candidate-action and paired-decision oracle E[Q] regret slices over v0 public/action-local proxies. Means use nonparametric bootstrap 95% confidence intervals; paired contrasts use per-decision best preferred proxy regret minus best alternative proxy regret.",
        "tag_dimensions": {
            "strategy_features": len(GLOBAL_TAGS),
            "strategy_action_features_per_action": len(ACTION_TAGS),
            "action_slots": 7,
        },
        "results": results,
        "missing_detectors": missing_detectors,
        "wandb": wb.status(),
        "hf_links": "not applicable",
        "claim_ledger_impact": "no central ledger change",
    }
    summary_path = output_dir / "summary.json"
    stats_path = output_dir / "claim_proxy_stats.csv"
    missing_path = output_dir / "missing_detectors.csv"
    ledger_path = output_dir / "claim_ledger_delta.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(stats_path, results)
    with missing_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "status", "needed_detector", "source"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(missing_detectors)
    ledger_delta = {
        "owner_bead": "t42-csw6.20",
        "claim_ledger_impact": "no central ledger change",
        "reason": "The run validates only v0 proxy slices. Direct Ch05/Ch12 setter-defense claim movement requires setter-role, bidder-off, void-creation, trump-set, and bid-margin detectors that are not present in the source artifacts.",
        "claim_statuses_reported_in_page": {row["claim_id"]: row["verdict"] for row in results}
        | {f"missing-{row['bucket'].replace(' ', '-')}": "not-yet-tested" for row in missing_detectors},
    }
    ledger_path.write_text(json.dumps(ledger_delta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    wb.log({"sample/eval_rows": len(rows), "sample/legal_candidate_actions": summary["sample"]["legal_candidate_actions"]})
    for row in results:
        wb.log(
            {
                f"{row['claim_id']}/preferred_mean_regret": row["preferred_mean_regret"] or 0.0,
                f"{row['claim_id']}/alternative_mean_regret": row["alternative_mean_regret"] or 0.0,
                f"{row['claim_id']}/paired_decision_n": row["paired_decision_n"],
                f"{row['claim_id']}/preferred_oracle_best_decision_rate": row["preferred_oracle_best_decision_rate"] or 0.0,
                f"{row['claim_id']}/alternative_oracle_best_decision_rate": row["alternative_oracle_best_decision_rate"] or 0.0,
            }
        )
    wb.update_summary(
        {
            "status": "completed",
            "eval_rows": len(rows),
            "legal_candidate_actions": summary["sample"]["legal_candidate_actions"],
            "claim_ledger_impact": "no central ledger change",
            "hf_links": "not applicable",
            "underpowered_claims": sum(1 for row in results if row["verdict"] == "underpowered"),
            "missing_direct_detectors": len(missing_detectors),
        }
    )
    wb.log_artifact_files(
        name=f"w42-setter-defense-claim-validation-{config['git_sha'][:8]}",
        artifact_type="w42-claim-validation",
        paths=[summary_path, stats_path, missing_path, ledger_path],
    )
    wb.finish(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
