#!/usr/bin/env python3
"""Report-only Chapter 4 partner-support proxy validation for w42.

The script consumes a Gus joint-world corpus with public-state strategy tags and
oracle E[Q] labels. It does not train a model and does not modify Gus paths.
"""

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


Predicate = Callable[[torch.Tensor], bool]


@dataclass(frozen=True)
class SliceSpec:
    claim_id: str
    label: str
    preferred: Predicate
    alternative: Predicate
    unit: str
    proxy_note: str
    source_claim: str
    expected_direction: str = "preferred_lower_regret"


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def tag_index(name: str) -> int:
    matches = [tag.idx for tag in ACTION_TAGS if tag.name == name]
    if not matches:
        raise KeyError(name)
    return matches[0]


IDX = {name: tag_index(name) for name in [tag.name for tag in ACTION_TAGS]}


def truthy(value: torch.Tensor, eps: float = 1e-6) -> bool:
    return bool(float(value.item()) > eps)


def is_count(action: torch.Tensor) -> bool:
    return float(action[IDX["count_points_frac"]].item()) > 1e-6


def is_ten_count(action: torch.Tensor) -> bool:
    return truthy(action[IDX["is_ten_count"]])


def is_partner_donation(action: torch.Tensor) -> bool:
    return is_count(action) and truthy(action[IDX["count_donation_to_partner"]])


def is_opponent_donation(action: torch.Tensor) -> bool:
    return is_count(action) and truthy(action[IDX["count_donation_to_opponent"]])


def is_beats_current(action: torch.Tensor) -> bool:
    return truthy(action[IDX["beats_current"]])


def is_not_beats_current(action: torch.Tensor) -> bool:
    return not is_beats_current(action)


def is_virtual_boss(action: torch.Tensor) -> bool:
    return (
        not truthy(action[IDX["double"]])
        and not truthy(action[IDX["trump"]])
        and truthy(action[IDX["off_non_double"]])
        and float(action[IDX["live_higher_suit_frac"]].item()) <= 1e-6
    )


def is_nonboss_off(action: torch.Tensor) -> bool:
    return (
        not truthy(action[IDX["double"]])
        and not truthy(action[IDX["trump"]])
        and truthy(action[IDX["off_non_double"]])
        and float(action[IDX["live_higher_suit_frac"]].item()) > 1e-6
    )


def is_lead_trump(action: torch.Tensor) -> bool:
    return truthy(action[IDX["is_lead_position"]]) and truthy(action[IDX["trump"]])


def is_lead_off(action: torch.Tensor) -> bool:
    return truthy(action[IDX["is_lead_position"]]) and truthy(action[IDX["off_non_double"]])


def is_low_liability_off(action: torch.Tensor) -> bool:
    return is_lead_off(action) and float(action[IDX["live_count_frac"]].item()) <= 1e-6


def is_high_liability_off(action: torch.Tensor) -> bool:
    return is_lead_off(action) and float(action[IDX["live_count_frac"]].item()) > 1e-6


def is_trump_in_count_dump(action: torch.Tensor) -> bool:
    return is_count(action) and truthy(action[IDX["trump_in"]])


def is_safe_partner_count(action: torch.Tensor) -> bool:
    return is_partner_donation(action) and not truthy(action[IDX["trump_in"]])


def bootstrap_ci(values: list[float], seed: int, n_boot: int = 5000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return lo, hi


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value: float | None) -> str:
    if value is None:
        return "not applicable"
    return f"{value:.6f}"


def analyze_slice(
    rows: list[dict[str, Any]],
    spec: SliceSpec,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    preferred_regrets: list[float] = []
    alternative_regrets: list[float] = []
    paired_diffs: list[float] = []
    decisions_with_preferred = 0
    decisions_with_alternative = 0
    decisions_with_both = 0
    preferred_oracle_best = 0
    alternative_oracle_best = 0
    preferred_actions = 0
    alternative_actions = 0

    for row in rows:
        legal_mask = row["legal_mask"]
        actions = row["strategy_action_features"]
        regrets = row["regrets"]
        oracle_action = int(row["oracle_action"])
        preferred_idxs: list[int] = []
        alternative_idxs: list[int] = []

        for idx in range(7):
            if not bool(legal_mask[idx].item()):
                continue
            action = actions[idx]
            if spec.preferred(action):
                preferred_idxs.append(idx)
                preferred_regrets.append(float(regrets[idx].item()))
            if spec.alternative(action):
                alternative_idxs.append(idx)
                alternative_regrets.append(float(regrets[idx].item()))

        if preferred_idxs:
            decisions_with_preferred += 1
            preferred_actions += len(preferred_idxs)
            if oracle_action in preferred_idxs:
                preferred_oracle_best += 1
        if alternative_idxs:
            decisions_with_alternative += 1
            alternative_actions += len(alternative_idxs)
            if oracle_action in alternative_idxs:
                alternative_oracle_best += 1
        if preferred_idxs and alternative_idxs:
            decisions_with_both += 1
            best_preferred = min(float(regrets[idx].item()) for idx in preferred_idxs)
            best_alternative = min(float(regrets[idx].item()) for idx in alternative_idxs)
            paired_diffs.append(best_preferred - best_alternative)

    pref_ci = bootstrap_ci(preferred_regrets, seed + 11, n_boot)
    alt_ci = bootstrap_ci(alternative_regrets, seed + 17, n_boot)
    paired_ci = bootstrap_ci(paired_diffs, seed + 23, n_boot)
    paired_mean = mean(paired_diffs)

    if paired_mean is None:
        verdict = "underpowered"
        status_reason = "No paired decisions expose both the preferred and alternative proxy actions."
    elif paired_ci[1] is not None and paired_ci[1] < 0:
        verdict = "context-limited"
        status_reason = "Preferred proxy has lower paired oracle regret on this public-state slice, but partner intent/role is not fully isolated."
    elif paired_ci[0] is not None and paired_ci[0] > 0:
        verdict = "contradicted"
        status_reason = "Preferred proxy has higher paired oracle regret on this public-state slice."
    else:
        verdict = "underpowered"
        status_reason = "Paired confidence interval crosses zero or sample is too small for a status move."

    return {
        "claim_id": spec.claim_id,
        "label": spec.label,
        "unit": spec.unit,
        "source_claim": spec.source_claim,
        "proxy_note": spec.proxy_note,
        "preferred_action_n": preferred_actions,
        "alternative_action_n": alternative_actions,
        "decisions_with_preferred": decisions_with_preferred,
        "decisions_with_alternative": decisions_with_alternative,
        "paired_decision_n": decisions_with_both,
        "preferred_mean_regret": mean(preferred_regrets),
        "preferred_mean_regret_ci95": pref_ci,
        "alternative_mean_regret": mean(alternative_regrets),
        "alternative_mean_regret_ci95": alt_ci,
        "paired_preferred_minus_alternative_mean_regret": paired_mean,
        "paired_ci95": paired_ci,
        "preferred_oracle_best_decision_rate": (
            preferred_oracle_best / decisions_with_preferred if decisions_with_preferred else None
        ),
        "alternative_oracle_best_decision_rate": (
            alternative_oracle_best / decisions_with_alternative if decisions_with_alternative else None
        ),
        "verdict": verdict,
        "status_reason": status_reason,
    }


def load_rows(eval_path: Path, seed: int, limit: int) -> list[dict[str, Any]]:
    ds = JointWorldFullDataset([eval_path], seed=seed, include_strategy_features=True)
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row = ds[i]
        legal_e_q = row["e_q"].masked_fill(~row["legal_mask"], float("-inf"))
        oracle_best = legal_e_q.max()
        rows.append(
            {
                "row_idx": i,
                "decision_idx": int(row["decision_idx"].item()),
                "player": int(row["player"].item()),
                "legal_mask": row["legal_mask"],
                "strategy_action_features": row["strategy_action_features"],
                "strategy_features": row["strategy_features"],
                "e_q": row["e_q"],
                "oracle_action": int(legal_e_q.argmax().item()),
                "regrets": oracle_best - row["e_q"],
            }
        )
    return rows


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "claim_id",
        "label",
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
        for r in results:
            writer.writerow(
                {
                    "claim_id": r["claim_id"],
                    "label": r["label"],
                    "preferred_action_n": r["preferred_action_n"],
                    "alternative_action_n": r["alternative_action_n"],
                    "paired_decision_n": r["paired_decision_n"],
                    "preferred_mean_regret": fmt(r["preferred_mean_regret"]),
                    "preferred_ci95_low": fmt(r["preferred_mean_regret_ci95"][0]),
                    "preferred_ci95_high": fmt(r["preferred_mean_regret_ci95"][1]),
                    "alternative_mean_regret": fmt(r["alternative_mean_regret"]),
                    "alternative_ci95_low": fmt(r["alternative_mean_regret_ci95"][0]),
                    "alternative_ci95_high": fmt(r["alternative_mean_regret_ci95"][1]),
                    "paired_preferred_minus_alternative_mean_regret": fmt(
                        r["paired_preferred_minus_alternative_mean_regret"]
                    ),
                    "paired_ci95_low": fmt(r["paired_ci95"][0]),
                    "paired_ci95_high": fmt(r["paired_ci95"][1]),
                    "preferred_oracle_best_decision_rate": fmt(r["preferred_oracle_best_decision_rate"]),
                    "alternative_oracle_best_decision_rate": fmt(r["alternative_oracle_best_decision_rate"]),
                    "verdict": r["verdict"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/w42/partner_support_claim_validation"))
    parser.add_argument("--eval-seed", type=int, default=43)
    parser.add_argument("--bootstrap-seed", type=int, default=20260502)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    validate_tag_dims()
    rows = load_rows(args.eval, args.eval_seed, args.limit)
    specs = [
        SliceSpec(
            claim_id="ch04-safe-partner-count-donation",
            label="Safe count donation to partner versus opponent-winning count donation",
            preferred=is_partner_donation,
            alternative=is_opponent_donation,
            unit="legal count candidate action",
            source_claim="Count should be given to the bidder/partner only when the partner is guaranteed to win.",
            proxy_note="Uses v0 `count_donation_to_partner/opponent` action tags; cannot prove bidder-partner intent or hidden future overtrump risk.",
        ),
        SliceSpec(
            claim_id="ch04-low-trump-trap-against-count-dump",
            label="Partner donation without trump-in versus count dumped while trumping in",
            preferred=is_safe_partner_count,
            alternative=is_trump_in_count_dump,
            unit="legal count candidate action",
            source_claim="Low trump leads can be traps; count dumped into uncertain trump-in windows can swing the hand.",
            proxy_note="Uses public action tags for count donation and trump-in; it does not know whether the led trump was a deliberate partner trap.",
        ),
        SliceSpec(
            claim_id="ch04-lead-capture-for-support",
            label="Winning the current trick versus non-winning alternatives",
            preferred=is_beats_current,
            alternative=is_not_beats_current,
            unit="legal candidate action",
            source_claim="The partner should actively try to win tricks and take lead when supporting the bidder.",
            proxy_note="Measures trick-capture value generally; v0 tags do not isolate bidder's partner or future support lead value.",
        ),
        SliceSpec(
            claim_id="ch04-effective-double-highest-remaining",
            label="Virtual boss off-suit tile versus lower live off-suit tile",
            preferred=is_virtual_boss,
            alternative=is_nonboss_off,
            unit="legal off-suit candidate action",
            source_claim="A middle-ranking tile can become an effective double when all higher same-suit tiles have been played.",
            proxy_note="Uses public live-higher-suit tag as a virtual-boss proxy; does not distinguish all reasons a non-double off tile is attractive.",
        ),
        SliceSpec(
            claim_id="ch04-lead-away-from-count-damage",
            label="Low-liability off lead versus count-exposing off lead",
            preferred=is_low_liability_off,
            alternative=is_high_liability_off,
            unit="legal lead candidate action",
            source_claim="If partner wins lead without a double, the next lead should minimize expected live count exposure.",
            proxy_note="Uses live-count-by-led-suit proxy only; it is not a full expected-count-exposure model.",
        ),
        SliceSpec(
            claim_id="ch04-avoid-disruptive-partner-trump-lead",
            label="Off lead versus trump lead from lead position",
            preferred=is_lead_off,
            alternative=is_lead_trump,
            unit="legal lead candidate action",
            source_claim="The bidder's partner should usually avoid leading trump, except when every off lead risks major count.",
            proxy_note="Measures ordinary lead-position trump/off tension; v0 tags do not identify bidder's partner, bidder trump length, or exception regimes.",
        ),
    ]

    results = [analyze_slice(rows, spec, args.bootstrap_seed + i * 1000, args.bootstrap_samples) for i, spec in enumerate(specs)]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "w42.partner_support_claim_validation.v1",
        "bead_id": "t42-csw6.19",
        "created_at": datetime.now(UTC).isoformat(),
        "repo_commit": git_sha(),
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
        "statistical_method": "Candidate-action and paired-decision oracle E[Q] regret slices. Means use nonparametric bootstrap 95% confidence intervals; paired contrasts use per-decision best preferred proxy regret minus best alternative proxy regret.",
        "wandb_links": "not applicable",
        "hf_links": "not applicable",
        "claim_ledger_impact": "no claim-ledger change",
        "tag_dimensions": {
            "strategy_features": len(GLOBAL_TAGS),
            "strategy_action_features_per_action": len(ACTION_TAGS),
            "action_slots": 7,
        },
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / "claim_proxy_stats.csv", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
