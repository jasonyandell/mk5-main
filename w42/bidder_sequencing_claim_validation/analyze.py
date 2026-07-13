#!/usr/bin/env python3
"""Summarize available w42 bidder-sequencing proxy evidence.

This bead intentionally does not modify Gus/Forge/Burl core paths. It reads the
existing w42 baseline metrics and emits conservative claim-level tables.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "w42" / "bidder_sequencing_claim_validation"
METRICS = ROOT / "w42" / "v0_strategy_tags_baseline" / "metrics.json"
RUN = ROOT / "w42" / "v0_strategy_tags_baseline" / "run.json"
RAW_METRICS = ROOT / "w42" / "raw_public_state_baseline" / "metrics.json"
EQ_SUMMARY = ROOT / "w42" / "eq_n10_comparison_slice" / "summary.json"


CLAIMS = [
    {
        "claim_id": "ch03-trump-pull-sequencing",
        "claim": "Trump-pull / trick-pressure features should identify bidder sequencing states where action quality changes.",
        "proxy_buckets": ["global:current_trick_pressure", "action:trick_relation", "action:suit_pressure"],
        "status": "underpowered",
        "verdict": "Proxy evidence improves tiny-model oracle alignment, but no trump-first/off-first counterfactual was run.",
    },
    {
        "claim_id": "ch03-reentry-preservation",
        "claim": "The bidder should preserve a reentry trump until unresolved offs are safe.",
        "proxy_buckets": ["action:double_protection", "global:hand_shape", "action:hand_shape"],
        "status": "underpowered",
        "verdict": "Available buckets touch hand/protection structure but do not encode final-trump-spent-before-off-clear.",
    },
    {
        "claim_id": "ch03-off-timing",
        "claim": "Off timing, including double-ahead/off-first exceptions, is measurable through suit-pressure and protection buckets.",
        "proxy_buckets": ["action:suit_pressure", "action:double_protection", "global:void_summary"],
        "status": "underpowered",
        "verdict": "Proxy buckets improve alignment; no paired early-off versus trump-first rollout exists in the artifacts.",
    },
    {
        "claim_id": "ch03-count-inventory",
        "claim": "Count inventory and count pressure are useful bidder-sequencing report slices.",
        "proxy_buckets": ["action:count_pressure", "global:public_count", "global:unseen_count_by_pip"],
        "status": "context-limited",
        "verdict": "Count buckets show the strongest proxy gains, but this validates tag usefulness rather than the book's full sequencing rule.",
    },
    {
        "claim_id": "ch03-laydown-correctness",
        "claim": "A laydown is valid only when every legal continuation wins the rest.",
        "proxy_buckets": [],
        "status": "not-yet-tested",
        "verdict": "No deterministic laydown proof checker or late-endgame enumeration artifact exists in the available w42 outputs.",
    },
]


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def wilson(p: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def diff_ci(p0: float, p1: float, n: int) -> tuple[float, float]:
    """Conservative Newcombe-style interval from aggregate bucket counts.

    The raw/tagged models were evaluated on the same slice, but the artifact
    does not retain paired per-row outcomes. Treating them as independent avoids
    overstating significance from aggregate data.
    """

    lo0, hi0 = wilson(p0, n)
    lo1, hi1 = wilson(p1, n)
    return (lo1 - hi0, hi1 - lo0)


def round6(x: Any) -> Any:
    if isinstance(x, float):
        if math.isnan(x):
            return "not applicable"
        return round(x, 6)
    return x


def bucket_row(bucket: str, metrics: dict[str, Any]) -> dict[str, Any]:
    item = metrics["bucket_comparison_raw_final_vs_tagged_best"][bucket]
    n = int(item["n"])
    match_lo, match_hi = diff_ci(item["raw_match_rate"], item["tagged_match_rate"], n)
    near_lo, near_hi = diff_ci(item["raw_near_tie"], item["tagged_near_tie"], n)
    return {
        "bucket": bucket,
        "n": n,
        "raw_mean_regret": item["raw_mean_regret"],
        "tagged_mean_regret": item["tagged_mean_regret"],
        "delta_mean_regret": item["delta_mean_regret"],
        "raw_match_rate": item["raw_match_rate"],
        "tagged_match_rate": item["tagged_match_rate"],
        "delta_match_rate": item["delta_match_rate"],
        "delta_match_rate_ci95_low": match_lo,
        "delta_match_rate_ci95_high": match_hi,
        "raw_near_tie": item["raw_near_tie"],
        "tagged_near_tie": item["tagged_near_tie"],
        "delta_near_tie": item["delta_near_tie"],
        "delta_near_tie_ci95_low": near_lo,
        "delta_near_tie_ci95_high": near_hi,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = read_json(METRICS)
    run = read_json(RUN)
    raw_metrics = read_json(RAW_METRICS)
    eq_summary = read_json(EQ_SUMMARY)

    buckets: dict[str, dict[str, Any]] = {}
    for claim in CLAIMS:
        for bucket in claim["proxy_buckets"]:
            buckets.setdefault(bucket, bucket_row(bucket, metrics))

    claim_rows = []
    for claim in CLAIMS:
        proxy_rows = [buckets[b] for b in claim["proxy_buckets"]]
        if proxy_rows:
            weighted_n = sum(row["n"] for row in proxy_rows)
            weighted_delta_regret = sum(row["delta_mean_regret"] * row["n"] for row in proxy_rows) / weighted_n
            weighted_delta_match = sum(row["delta_match_rate"] * row["n"] for row in proxy_rows) / weighted_n
            min_ci = min(row["delta_match_rate_ci95_low"] for row in proxy_rows)
            max_ci = max(row["delta_match_rate_ci95_high"] for row in proxy_rows)
        else:
            weighted_n = 0
            weighted_delta_regret = math.nan
            weighted_delta_match = math.nan
            min_ci = math.nan
            max_ci = math.nan
        claim_rows.append(
            {
                "claim_id": claim["claim_id"],
                "status": claim["status"],
                "proxy_buckets": ";".join(claim["proxy_buckets"]) if claim["proxy_buckets"] else "not applicable",
                "n_weighted_bucket_rows": weighted_n or "not applicable",
                "weighted_delta_mean_regret": round6(weighted_delta_regret),
                "weighted_delta_match_rate": round6(weighted_delta_match),
                "proxy_delta_match_rate_ci95_span": (
                    f"[{round6(min_ci)}, {round6(max_ci)}]" if weighted_n else "not applicable"
                ),
                "verdict": claim["verdict"],
            }
        )

    bucket_fields = list(next(iter(buckets.values())).keys())
    with (OUT / "bucket_proxy_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bucket_fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(buckets.values(), key=lambda r: r["bucket"]):
            writer.writerow({k: round6(v) for k, v in row.items()})

    claim_fields = list(claim_rows[0].keys())
    with (OUT / "claim_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claim_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(claim_rows)

    summary = {
        "schema_version": "w42.bidder_sequencing_claim_validation.v1",
        "owner_bead": "t42-csw6.18",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": git_sha(),
        "method": {
            "evidence_mode": "secondary analysis of existing w42 oracle-regret/model artifacts",
            "statistical_test": "Wilson/Newcombe-style 95% intervals for aggregate match-rate and near-tie deltas; descriptive mean-regret deltas because per-row variance was not retained",
            "paired_limitation": "The source eval is paired by row, but only aggregate bucket metrics are available in this worktree.",
        },
        "data_inputs": {
            "v0_strategy_tags_metrics": str(METRICS.relative_to(ROOT)),
            "v0_strategy_tags_run": str(RUN.relative_to(ROOT)),
            "raw_public_state_metrics": str(RAW_METRICS.relative_to(ROOT)),
            "eq_n10_summary": str(EQ_SUMMARY.relative_to(ROOT)),
        },
        "run_config": run["config"],
        "baseline_context": {
            "tagged_best_overall": {
                "epoch": 7,
                "n": 560,
                "mean_regret": run["history"][6]["mean_regret"],
                "match_rate": run["history"][6]["match_rate"],
                "near_tie_rate_regret_lt_0_5": run["history"][6]["near_tie_rate_regret_lt_0_5"],
            },
            "raw_best": raw_metrics["raw_public_state_action_model_best"],
            "raw_final": raw_metrics["raw_public_state_action_model_final"],
            "e_q_n_10": raw_metrics["e_q_n_10"],
            "eq_n10_comparison_slice": eq_summary,
        },
        "claims": claim_rows,
        "wandb_links": "not applicable",
        "hf_links": "not applicable",
        "claim_ledger_impact": "no claim-ledger change",
        "artifacts": {
            "summary_json": "w42/bidder_sequencing_claim_validation/summary.json",
            "claim_summary_csv": "w42/bidder_sequencing_claim_validation/claim_summary.csv",
            "bucket_proxy_stats_csv": "w42/bidder_sequencing_claim_validation/bucket_proxy_stats.csv",
            "claim_ledger_delta_json": "w42/bidder_sequencing_claim_validation/claim_ledger_delta.json",
        },
    }

    with (OUT / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    ledger_delta = {
        "owner_bead": "t42-csw6.18",
        "claim_ledger_impact": "no claim-ledger change",
        "reason": "Available artifacts provide proxy bucket evidence only; no direct oracle counterfactuals or laydown proof enumeration were run.",
        "claim_statuses_reported_in_page": {row["claim_id"]: row["status"] for row in claim_rows},
    }
    with (OUT / "claim_ledger_delta.json").open("w") as f:
        json.dump(ledger_delta, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
