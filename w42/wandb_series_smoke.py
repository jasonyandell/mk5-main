#!/usr/bin/env python
"""Tiny smoke for the w42 W&B series logging standard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wandb_utils import add_wandb_args, init_wandb

ROOT = Path(__file__).resolve().parents[1]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("w42/wandb_series_smoke"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--bootstrap-checkpoints", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    add_wandb_args(
        parser,
        default_project="w42",
        default_group="w42-csw6-wandb-series-standard",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    config: dict[str, Any] = {
        "bead_id": "t42-csw6.32",
        "git_sha": sha,
        "data_manifest": "not applicable; synthetic W&B logging smoke",
        "source_corpus": "not applicable",
        "dataset_name": "not applicable",
        "dataset_version": "not applicable",
        "ruleset": "not applicable",
        "label_source": "not applicable",
        "decision_slice": "not applicable",
        "feature_set": "wandb-series-smoke",
        "concept_buckets": ["instrumentation"],
        "model_family": "not applicable",
        "model_params": {"epochs": args.epochs},
        "random_seed": args.seed,
        "train_seed": "not applicable",
        "split_seed": "not applicable",
        "eval_seed": "not applicable",
        "baseline_policy": "not applicable",
        "metrics": ["train/loss", "eval/mean_regret", "bootstrap/ci_width"],
        "claim_ledger_status_before": "no claim-ledger change",
        "local_artifact_path": str(out_dir),
        "hf_repo_id": "not applicable",
        "wandb_group": args.wandb_group,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_checkpoints": args.bootstrap_checkpoints,
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=out_dir,
        tags=[
            "w42",
            "winning42",
            "strategy-validation",
            "promoted",
            "instrumentation",
            "wandb-series",
            "t42-csw6.32",
        ],
    )
    started_at = datetime.now(UTC).isoformat()
    history: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        global_step += 1
        train_loss = 1.0 / epoch
        eval_regret = 2.5 - 0.08 * epoch
        row = {
            "phase": "epoch",
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_mean_regret": eval_regret,
        }
        history.append(row)
        wb.log_series_point(
            axis="epoch",
            value=epoch,
            step=global_step,
            metrics={
                "train/loss": train_loss,
                "eval/mean_regret": eval_regret,
                "series/global_step": global_step,
            },
        )

    samples_per_checkpoint = max(args.bootstrap_samples // max(args.bootstrap_checkpoints, 1), 1)
    for checkpoint in range(1, args.bootstrap_checkpoints + 1):
        global_step += 1
        samples = min(checkpoint * samples_per_checkpoint, args.bootstrap_samples)
        ci_width = 1.0 / (checkpoint + 1)
        row = {
            "phase": "bootstrap",
            "checkpoint": checkpoint,
            "bootstrap_samples_seen": samples,
            "ci_width": ci_width,
        }
        history.append(row)
        wb.log_series_point(
            axis="bootstrap/samples_seen",
            value=samples,
            step=global_step,
            metrics={
                "bootstrap/checkpoint": checkpoint,
                "bootstrap/ci_width": ci_width,
                "series/global_step": global_step,
            },
        )

    summary = {
        "schema_version": "w42.wandb_series_smoke.v1",
        "bead_id": "t42-csw6.32",
        "created_at": datetime.now(UTC).isoformat(),
        "started_at": started_at,
        "command": " ".join(sys.argv),
        "git_sha": sha,
        "points_logged": len(history),
        "history": history,
        "wandb": wb.status(),
        "hf_links": "not applicable",
        "claim_ledger_impact": "no claim-ledger change",
    }
    (out_dir / "run.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wb.update_summary(
        {
            "status": "completed",
            "points_logged": len(history),
            "epoch_points": args.epochs,
            "bootstrap_points": args.bootstrap_checkpoints,
            "hf_links": "not applicable",
            "claim_ledger_impact": "no claim-ledger change",
        }
    )
    wb.log_artifact_files(
        name=f"w42-wandb-series-smoke-{sha[:8]}",
        artifact_type="w42-instrumentation-smoke",
        paths=[out_dir / "run.json"],
    )
    wb.finish()
    print(json.dumps({"run_json": str(out_dir / "run.json"), "wandb": summary["wandb"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
