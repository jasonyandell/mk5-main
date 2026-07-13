#!/usr/bin/env python3
"""Run the reusable w42 claim-analysis harness on existing row artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from w42.claim_analysis.harness import analyze_rows, load_rows, log_row_progress, write_result_artifacts
from w42.claim_analysis.registry import specs_for_source_kind
from w42.wandb_utils import add_wandb_args, init_wandb


DEFAULT_LABEL_FIELDS = {
    "gus_claim_rows": ("labels",),
    "branch_atlas_actions": (
        "distribution_shape_tags",
        "strategy_context_tags",
        "matched_position_detectors",
        "direct_label_readiness",
    ),
    "phase2_decision_actions": (
        "distribution_shape_tags",
        "strategy_context_tags",
        "matched_position_detectors",
        "direct_label_readiness",
    ),
}


def infer_source_kind(path: Path) -> str:
    text = str(path)
    if "gus_corpus_claim_deep_dive" in text:
        return "gus_claim_rows"
    if "branch_atlas" in text:
        return "branch_atlas_actions"
    if "phase2_decision_table" in text:
        return "phase2_decision_actions"
    return "custom_rows"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="CSV or JSONL row artifact. May be passed more than once.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-kind", choices=["auto", "gus_claim_rows", "branch_atlas_actions", "phase2_decision_actions", "custom_rows"], default="auto")
    parser.add_argument("--label-field", action="append", default=None, help="Field containing pipe/comma-separated labels. Defaults by source kind.")
    parser.add_argument("--min-label-n", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260502)
    parser.add_argument("--progress-chunk-size", type=int, default=100)
    parser.add_argument("--bead-id", default="t42-0b4l.2")
    add_wandb_args(parser, default_group="w42-claim-analysis-harness", default_enabled=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_paths = [Path(path) for path in args.input]
    source_kind = args.source_kind
    if source_kind == "auto":
        source_kind = infer_source_kind(input_paths[0])
    label_fields = tuple(args.label_field or DEFAULT_LABEL_FIELDS.get(source_kind, ("labels",)))
    claim_specs = specs_for_source_kind(source_kind)

    output_dir = Path(args.output_dir)
    config = {
        "bead_id": args.bead_id,
        "source_kind": source_kind,
        "inputs": [str(path) for path in input_paths],
        "label_fields": label_fields,
        "min_label_n": args.min_label_n,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=output_dir,
        tags=["w42", "claim-analysis", "harness", args.bead_id, source_kind],
    )
    try:
        rows = []
        for path in input_paths:
            rows.extend(load_rows(path, label_fields=label_fields))

        if getattr(wb, "run", None) is not None:
            wb.run.define_metric("progress/rows_processed")
            for pattern in ("coverage/*", "labels/*", "claims/*"):
                wb.run.define_metric(pattern, step_metric="progress/rows_processed")

        log_row_progress(wb, rows, chunk_size=args.progress_chunk_size)
        result = analyze_rows(
            rows,
            min_label_n=args.min_label_n,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        paths = write_result_artifacts(
            result=result,
            output_dir=output_dir,
            input_paths=input_paths,
            label_fields=label_fields,
            claim_specs=claim_specs,
            source_kind=source_kind,
            bead_id=args.bead_id,
            bootstrap_samples=args.bootstrap_samples,
            wandb_status=wb.status(),
        )
        wb.update_summary(
            {
                "status": "completed",
                "coverage/action_rows": result.action_rows,
                "coverage/decision_rows": result.decision_rows,
                "claims/label_metric_rows": len(result.label_metrics),
                "claims/paired_contrast_rows": len(result.paired_contrasts),
            }
        )
        wb.log_artifact_files(
            name=f"{args.bead_id}-claim-analysis-harness-output",
            artifact_type="claim-analysis-report",
            paths=list(paths.values()),
        )
        print(
            json.dumps(
                {
                    "action_rows": result.action_rows,
                    "decision_rows": result.decision_rows,
                    "label_metric_rows": len(result.label_metrics),
                    "paired_contrast_rows": len(result.paired_contrasts),
                    "output_dir": str(output_dir),
                    "wandb": wb.status(),
                },
                indent=2,
            )
        )
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
