"""Build the w42 style/partnership concept-bucket pilot report.

This is a report-artifact pilot, not a new model run. It reuses prior w42
metrics and logs the synthesized bucket table to W&B so the bead has live
provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "w42", ROOT):
    text = str(entry)
    if text not in sys.path:
        sys.path.insert(0, text)

from wandb_utils import add_wandb_args, init_wandb  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("w42/style_partnership_concept_buckets")
DEFAULT_RICH_BUCKETS = Path("w42/rich_tag_many_signal_probe/bucket_metrics_best.csv")
DEFAULT_PARTNER_STATS = Path("w42/partner_support_claim_validation/claim_proxy_stats.csv")
DEFAULT_BIDDING_SUMMARY = Path("w42/bidding_risk_budget_claim_validation/summary.json")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="") as fh:
        return {row[key]: row for row in csv.DictReader(fh)}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rich = _read_csv_by_key(args.rich_bucket_metrics, "bucket")
    partner = _read_csv_by_key(args.partner_support_stats, "claim_id")
    bidding = _read_json(args.bidding_summary)

    bidding_risk = rich["rich:bidding_risk"]
    donation = rich["rich:count_donation"]
    safe_donation = partner["ch04-safe-partner-count-donation"]
    count_damage = partner["ch04-lead-away-from-count-damage"]
    virtual_boss = partner["ch04-effective-double-highest-remaining"]

    strong_trump_bad_risk_pct = (
        bidding["hand_metrics"]["any_strong_trump_bad_risk"]["pct"] / 100.0
    )
    best_low_risk_pct = (
        bidding["hand_metrics"]["best_minimum_shape_low_risk"]["pct"] / 100.0
    )

    rows = [
        {
            "bucket": "overbid_restraint",
            "measurement_status": "partially-measured proxy",
            "online_safe_inputs": "own hand, candidate trump, off-risk/count exposure, auction/score when available",
            "report_only_labels": "make/set rate, bid regret, unnecessary bid margin, partner hidden rescue ownership",
            "pilot_evidence": (
                "rich bidding-risk slice over 560 eval decisions; exact static hand enumeration for risk traps"
            ),
            "pilot_n": int(float(bidding_risk["n"])),
            "pilot_metric": "rich-vs-raw regret delta; static risk prevalence",
            "pilot_value": round(_float(bidding_risk, "delta_mean_regret"), 6),
            "tail_or_residual_value": round(_float(bidding_risk, "delta_tail_ge_5"), 6),
            "supporting_detail": (
                f"rich:bidding_risk regret {float(bidding_risk['raw_mean_regret']):.3f}"
                f" -> {float(bidding_risk['tagged_mean_regret']):.3f}; "
                f"best static low-risk candidate in {_format_percent(best_low_risk_pct)} of hands; "
                f"any strong-trump bad-risk trap in {_format_percent(strong_trump_bad_risk_pct)} of hands"
            ),
            "claim_ledger_recommendation": "no central change; treat as underpowered proxy evidence",
            "caveat": (
                "No auction counterfactual or calibrated make threshold is present; raw bid height is not style."
            ),
        },
        {
            "bucket": "partner_legibility",
            "measurement_status": "partially-measured proxy",
            "online_safe_inputs": "team relation to current trick winner, legal action set, public count pressure, lead transfers",
            "report_only_labels": "true partner need, hidden future overtrump safety, support precision/recall",
            "pilot_evidence": "partner-support proxy report plus rich count/donation slice",
            "pilot_n": int(float(safe_donation["preferred_action_n"])),
            "pilot_metric": "safe partner donation regret vs opponent-winning count donation",
            "pilot_value": round(float(safe_donation["preferred_mean_regret"]), 6),
            "tail_or_residual_value": round(_float(donation, "delta_tail_ge_5"), 6),
            "supporting_detail": (
                f"partner donation mean regret {float(safe_donation['preferred_mean_regret']):.3f}"
                f" vs opponent-winning donation {float(safe_donation['alternative_mean_regret']):.3f}; "
                f"rich:count_donation regret delta {float(donation['delta_mean_regret']):.3f}"
            ),
            "claim_ledger_recommendation": "no central change; directional but unpaired and underpowered",
            "caveat": (
                "Current tags show public support opportunities, not stable partner readability or intent."
            ),
        },
        {
            "bucket": "partner_fit_residual",
            "measurement_status": "design-only / no-run rationale",
            "online_safe_inputs": "player or policy id, partner id, public support windows, style cohorts, shuffled-pair assignment",
            "report_only_labels": "team EV residual beyond additive individual ratings; fixed-vs-random partner delta",
            "pilot_evidence": "not run; required repeated identities or synthetic style-policy pool are absent",
            "pilot_n": "not applicable",
            "pilot_metric": "team residual = team EV - individual policy ratings - seat/deal controls",
            "pilot_value": "not applicable",
            "tail_or_residual_value": "not applicable",
            "supporting_detail": (
                "The current Gus eval corpus has decisions, not repeated player/policy identities or partner-shuffle cohorts."
            ),
            "claim_ledger_recommendation": "no central change; stay underpowered until identity/cohort data exists",
            "caveat": "Do not infer partner fit from a single generated hand or from hidden partner facts.",
        },
        {
            "bucket": "social_pressure_robustness",
            "measurement_status": "design-only / no-run rationale",
            "online_safe_inputs": "score pressure, loss streak, time/tool budget, prompt condition, tournament/marks context",
            "report_only_labels": "tail-regret shift under pressure; illegal/private trace claim rate; recovery after blunder",
            "pilot_evidence": "not run; needs paired clean-vs-pressure prompts or timed/tournament state wrapper",
            "pilot_n": "not applicable",
            "pilot_metric": "paired tail-regret and trace-leakage delta under pressure wrapper",
            "pilot_value": "not applicable",
            "tail_or_residual_value": "not applicable",
            "supporting_detail": (
                "Existing rich-tag run has state buckets but no social prompt, time pressure, or repeated-loss condition."
            ),
            "claim_ledger_recommendation": "no central change; report-only design until paired wrapper exists",
            "caveat": "Pressure evaluation must remain paired with oracle regret so style flavor cannot mask bad play.",
        },
        {
            "bucket": "support_proxy_contradiction_check",
            "measurement_status": "measured warning proxy",
            "online_safe_inputs": "highest remaining same-suit off tile, lead context, live count presence",
            "report_only_labels": "lead recovery value, future set line, partner rescue need",
            "pilot_evidence": "partner-support proxy report",
            "pilot_n": int(float(virtual_boss["paired_decision_n"])),
            "pilot_metric": "paired virtual-boss preferred minus alternative regret",
            "pilot_value": round(float(virtual_boss["paired_preferred_minus_alternative_mean_regret"]), 6),
            "tail_or_residual_value": round(float(count_damage["paired_preferred_minus_alternative_mean_regret"]), 6),
            "supporting_detail": (
                "Virtual-boss proxy was lower regret unpaired but worse in paired decisions; "
                "count-damage proxy remains directionally favorable with CI crossing zero."
            ),
            "claim_ledger_recommendation": "no central change; use as detector-caveat evidence",
            "caveat": "Highest-remaining tile is not sufficient for partner-support correctness.",
        },
    ]

    summary = {
        "schema_version": "w42.style_partnership_concept_buckets.v1",
        "bead_id": args.bead_id,
        "git_sha": _git_sha(),
        "created_at": datetime.now(UTC).isoformat(),
        "evidence_mode": "report-artifact pilot from existing w42 artifacts",
        "source_artifacts": {
            "rich_bucket_metrics": str(args.rich_bucket_metrics),
            "partner_support_stats": str(args.partner_support_stats),
            "bidding_summary": str(args.bidding_summary),
        },
        "random_seeds": {
            "fresh_run": "not applicable",
            "inherited_rich_tag_run": {"train": 42, "split": 42, "eval": 43},
            "inherited_partner_support_bootstrap": 20260502,
        },
        "rows": rows,
        "headline": {
            "overbid_restraint": rows[0]["supporting_detail"],
            "partner_legibility": rows[1]["supporting_detail"],
            "partner_fit_residual": rows[2]["pilot_evidence"],
            "social_pressure_robustness": rows[3]["pilot_evidence"],
        },
        "claim_ledger_impact": "no central claim-ledger change",
        "hf_links": "not applicable",
    }
    return rows, summary


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "bucket_matrix.csv"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    with table_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "w42.style_partnership_concept_buckets.manifest.v1",
        "bead_id": summary["bead_id"],
        "git_sha": summary["git_sha"],
        "created_at": summary["created_at"],
        "artifacts": {
            "bucket_matrix": str(table_path),
            "summary": str(summary_path),
        },
        "source_artifacts": summary["source_artifacts"],
        "wandb": "filled by run.json",
        "hf_links": "not applicable",
        "claim_ledger_impact": "no central claim-ledger change",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"table": table_path, "summary": summary_path, "manifest": manifest_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bead-id", default="t42-csw6.25")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rich-bucket-metrics", type=Path, default=DEFAULT_RICH_BUCKETS)
    parser.add_argument("--partner-support-stats", type=Path, default=DEFAULT_PARTNER_STATS)
    parser.add_argument("--bidding-summary", type=Path, default=DEFAULT_BIDDING_SUMMARY)
    add_wandb_args(parser, default_project="w42", default_group="t42-csw6")
    args = parser.parse_args()

    rows, summary = build_rows(args)
    paths = write_outputs(rows, summary, args.output_dir)

    wandb_config = {
        "bead_id": args.bead_id,
        "git_sha": summary["git_sha"],
        "data_manifest": str(paths["manifest"]),
        "source_corpus": "not applicable; existing w42 report artifacts only",
        "dataset_name": "w42-style-partnership-concept-buckets",
        "dataset_version": "v1",
        "ruleset": "existing w42/Gus corpus semantics where inherited; design rows are ruleset-neutral",
        "label_source": "existing rich-tag, bidding-risk, and partner-support report artifacts",
        "decision_slice": "style/partnership report buckets",
        "feature_set": "report-only style/partnership bucket matrix",
        "concept_buckets": [row["bucket"] for row in rows],
        "model_family": "not applicable",
        "model_params": "not applicable",
        "random_seed": "not applicable",
        "train_seed": "not applicable",
        "split_seed": "not applicable",
        "eval_seed": "not applicable",
        "baseline_policy": "raw public-state and rich-tag source reports",
        "metrics": "bucket proxy regret deltas and no-run design coverage",
        "claim_ledger_status_before": "underpowered/context-limited chapter-local claims",
        "local_artifact_path": str(args.output_dir),
        "hf_repo_id": "not applicable",
        "wandb_group": args.wandb_group,
        "seed": 0,
    }
    wandb_run = init_wandb(
        args,
        config=wandb_config,
        output_dir=args.output_dir,
        tags=["w42", args.bead_id, "style-partnership", "concept-buckets", "promoted"],
    )
    wandb_run.log(
        {
            "buckets/total": len(rows),
            "buckets/partially_measured": sum(
                "partially-measured" in str(row["measurement_status"]) for row in rows
            ),
            "buckets/design_only": sum(
                "design-only" in str(row["measurement_status"]) for row in rows
            ),
            "pilot/overbid_restraint_delta_regret": rows[0]["pilot_value"],
            "pilot/partner_legibility_preferred_regret": rows[1]["pilot_value"],
            "pilot/support_proxy_paired_warning": rows[4]["pilot_value"],
        }
    )
    wandb_run.update_summary(
        {
            "claim_ledger_impact": "no central claim-ledger change",
            "hf_links": "not applicable",
            "bucket_matrix": str(paths["table"]),
        }
    )
    wandb_run.log_artifact_files(
        name=f"w42-style-partnership-concept-buckets-{summary['git_sha'][:8]}",
        artifact_type="w42-report",
        paths=list(paths.values()),
    )
    summary["wandb"] = wandb_run.status()
    (args.output_dir / "run.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    wandb_run.finish()
    print(json.dumps({"outputs": {k: str(v) for k, v in paths.items()}, "wandb": summary["wandb"]}, indent=2))


if __name__ == "__main__":
    main()
