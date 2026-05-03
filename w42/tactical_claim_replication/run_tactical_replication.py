#!/usr/bin/env python3
"""Powered tactical replication for setter pounce and count donation claims."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from w42.claim_analysis.harness import analyze_rows, load_rows, write_result_artifacts
from w42.claim_analysis.registry import specs_for_source_kind
from w42.gus_corpus_claim_deep_dive import analyze_gus_claims as gus
from w42.wandb_utils import add_wandb_args, init_wandb


OUT_DIR = Path("w42/tactical_claim_replication")
TACTICAL_LABELS = [
    "ch05_setter_pounce_count_before_certainty",
    "ch05_setter_pounce_count",
    "ch05_setter_pounce_count_sets_now",
    "ch05_reckless_count_to_bidder",
    "ch04_partner_safe_count_donation_current_control",
    "ch04_partner_unsafe_count_to_defense",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row_slices(row: gus.ActionRow) -> dict[str, str]:
    later = max(0, 3 - int(row.trick_position))
    return {
        "decl": row.decl_name,
        "seat_role": row.seat_role,
        "team": row.team,
        "trick_idx": str(row.trick_idx),
        "trick_position": str(row.trick_position),
        "count_points": str(row.candidate_count_points),
        "current_winner_team": row.current_winner_team_before,
        "candidate_beats_current": str(row.candidate_beats_current).lower(),
        "candidate_would_win_now": str(row.candidate_would_win_trick_now).lower(),
        "later_seats_remaining": str(later),
        "sets_now_label": str("ch05_setter_pounce_count_sets_now" in row.labels).lower(),
    }


def slice_label_metrics(
    rows: list[gus.ActionRow],
    *,
    labels: list[str],
    min_slice_n: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in labels:
        selected = [row for row in rows if label in row.labels]
        buckets: dict[tuple[str, str], list[gus.ActionRow]] = defaultdict(list)
        for row in selected:
            for slice_name, slice_value in row_slices(row).items():
                buckets[(slice_name, slice_value)].append(row)
        for (slice_name, slice_value), bucket in sorted(buckets.items()):
            if len(bucket) < min_slice_n:
                continue
            out.append(
                {
                    "claim_label": label,
                    "slice_name": slice_name,
                    "slice_value": slice_value,
                    "action_n": len(bucket),
                    "decision_n": len({row.key for row in bucket}),
                    "mean": gus.mean([row.mean for row in bucket]),
                    "mean_regret": gus.mean([row.mean_regret for row in bucket]),
                    "mean_threshold_mass": gus.mean([row.threshold_mass for row in bucket]),
                    "mean_lower_tail_mass": gus.mean([row.lower_tail_mass for row in bucket]),
                    "actual_action_rate": gus.mean([1.0 if row.is_actual_action else 0.0 for row in bucket]),
                    "best_mean_rate": gus.mean([1.0 if row.is_best_mean else 0.0 for row in bucket]),
                    "best_threshold_rate": gus.mean([1.0 if row.is_best_threshold else 0.0 for row in bucket]),
                    "safest_tail_rate": gus.mean([1.0 if row.is_safest_tail else 0.0 for row in bucket]),
                }
            )
    return out


def grouped_by_decision(rows: list[gus.ActionRow]) -> dict[str, list[gus.ActionRow]]:
    grouped: dict[str, list[gus.ActionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.key].append(row)
    return grouped


def best_by_mean(rows: list[gus.ActionRow]) -> gus.ActionRow:
    return max(rows, key=lambda row: row.mean)


def paired_rows(
    rows: list[gus.ActionRow],
    *,
    preferred: Callable[[gus.ActionRow], bool],
    alternative: Callable[[gus.ActionRow], bool],
) -> list[tuple[gus.ActionRow, gus.ActionRow]]:
    pairs: list[tuple[gus.ActionRow, gus.ActionRow]] = []
    for decision_rows in grouped_by_decision(rows).values():
        pref_rows = [row for row in decision_rows if preferred(row)]
        alt_rows = [row for row in decision_rows if alternative(row)]
        if pref_rows and alt_rows:
            pairs.append((best_by_mean(pref_rows), best_by_mean(alt_rows)))
    return pairs


def summarize_pair_bucket(
    *,
    contrast_id: str,
    pairs: list[tuple[gus.ActionRow, gus.ActionRow]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    mean_deltas = [pref.mean - alt.mean for pref, alt in pairs]
    regret_deltas = [pref.mean_regret - alt.mean_regret for pref, alt in pairs]
    threshold_deltas = [pref.threshold_mass - alt.threshold_mass for pref, alt in pairs]
    lower_tail_deltas = [pref.lower_tail_mass - alt.lower_tail_mass for pref, alt in pairs]
    mean_delta, mean_lo, mean_hi = gus.bootstrap_mean_ci(
        mean_deltas,
        samples=bootstrap_samples,
        seed=gus.stable_seed(bootstrap_seed, contrast_id),
    )
    return {
        "contrast_id": contrast_id,
        "paired_decision_n": len(pairs),
        "mean_delta": mean_delta,
        "mean_delta_ci95_low": mean_lo,
        "mean_delta_ci95_high": mean_hi,
        "regret_delta": gus.mean(regret_deltas),
        "threshold_mass_delta": gus.mean(threshold_deltas),
        "lower_tail_mass_delta": gus.mean(lower_tail_deltas),
    }


def contrast_specs() -> list[tuple[str, Callable[[gus.ActionRow], bool], Callable[[gus.ActionRow], bool]]]:
    return [
        (
            "ch05_pounce_count_vs_other_same_decision",
            lambda row: "ch05_setter_pounce_count" in row.labels,
            lambda row: "ch05_setter_pounce_count" not in row.labels,
        ),
        (
            "ch05_pounce_sets_now_vs_other_same_decision",
            lambda row: "ch05_setter_pounce_count_sets_now" in row.labels,
            lambda row: "ch05_setter_pounce_count_sets_now" not in row.labels,
        ),
        (
            "ch05_reckless_count_vs_nonreckless_same_decision",
            lambda row: "ch05_reckless_count_to_bidder" in row.labels,
            lambda row: "ch05_reckless_count_to_bidder" not in row.labels,
        ),
        (
            "ch04_safe_partner_count_vs_other_same_decision",
            lambda row: "ch04_partner_safe_count_donation_current_control" in row.labels,
            lambda row: "ch04_partner_safe_count_donation_current_control" not in row.labels,
        ),
        (
            "ch04_unsafe_partner_count_vs_nonunsafe_same_decision",
            lambda row: "ch04_partner_unsafe_count_to_defense" in row.labels,
            lambda row: "ch04_partner_unsafe_count_to_defense" not in row.labels,
        ),
    ]


def paired_contrast_slices(
    rows: list[gus.ActionRow],
    *,
    min_paired_slice_n: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for contrast_id, preferred, alternative in contrast_specs():
        pairs = paired_rows(rows, preferred=preferred, alternative=alternative)
        buckets: dict[tuple[str, str], list[tuple[gus.ActionRow, gus.ActionRow]]] = defaultdict(list)
        for pref, alt in pairs:
            for slice_name, slice_value in row_slices(pref).items():
                buckets[(slice_name, slice_value)].append((pref, alt))
        for (slice_name, slice_value), bucket in sorted(buckets.items()):
            if len(bucket) < min_paired_slice_n:
                continue
            row = summarize_pair_bucket(
                contrast_id=f"{contrast_id}__{slice_name}={slice_value}",
                pairs=bucket,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            row["base_contrast_id"] = contrast_id
            row["slice_name"] = slice_name
            row["slice_value"] = slice_value
            out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=["gus/data/corpus_v2_train_*_d0-9.pt"])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260502)
    parser.add_argument("--min-slice-n", type=int, default=20)
    parser.add_argument("--min-paired-slice-n", type=int, default=20)
    parser.add_argument("--log-every-files", type=int, default=1)
    add_wandb_args(parser, default_group="w42-tactical-claim-replication", default_enabled=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = gus.expand_inputs(args.inputs)
    if args.max_files:
        paths = paths[: args.max_files]
    if not paths:
        raise SystemExit("No input files matched.")

    config = {
        "bead_id": "t42-0b4l.3",
        "experiment": "w42-tactical-claim-replication",
        "inputs": [str(path) for path in paths],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "min_slice_n": args.min_slice_n,
        "min_paired_slice_n": args.min_paired_slice_n,
        "claim_labels": TACTICAL_LABELS,
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=output_dir,
        tags=["w42", "claim-analysis", "tactical", "gus-corpus", "t42-0b4l.3"],
    )
    if getattr(wb, "run", None) is not None:
        wb.run.define_metric("progress/files_processed")
        for pattern in ("progress/*", "coverage/*", "claims/*", "slices/*"):
            wb.run.define_metric(pattern, step_metric="progress/files_processed")

    started = time.perf_counter()
    rows: list[gus.ActionRow] = []
    file_meta: list[dict[str, Any]] = []
    for idx, path in enumerate(paths, start=1):
        file_rows, meta = gus.load_payload_rows(path)
        rows.extend(file_rows)
        file_meta.append(meta)
        if idx % max(int(args.log_every_files), 1) == 0:
            label_counts = Counter(label for row in rows for label in row.labels)
            wb.log(
                {
                    "progress/files_processed": idx,
                    "coverage/action_rows": len(rows),
                    "coverage/decision_rows": len({row.key for row in rows}),
                    "coverage/claim_action_rows": sum(1 for row in rows if row.labels),
                    "claims/setter_pounce_count": label_counts["ch05_setter_pounce_count"],
                    "claims/reckless_count_to_bidder": label_counts["ch05_reckless_count_to_bidder"],
                    "claims/partner_safe_donation": label_counts[
                        "ch04_partner_safe_count_donation_current_control"
                    ],
                    "claims/partner_unsafe_donation": label_counts["ch04_partner_unsafe_count_to_defense"],
                    "progress/wall_seconds": time.perf_counter() - started,
                },
                step=idx,
            )

    all_action_rows_path = output_dir / "all_action_rows.jsonl"
    gus.write_jsonl(all_action_rows_path, [gus.row_to_dict(row) for row in rows])
    normalized_rows = load_rows(all_action_rows_path, label_fields=("labels",))
    harness_result = analyze_rows(
        normalized_rows,
        min_label_n=5,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    harness_paths = write_result_artifacts(
        result=harness_result,
        output_dir=output_dir / "harness",
        input_paths=[all_action_rows_path],
        label_fields=("labels",),
        claim_specs=specs_for_source_kind("gus_claim_rows"),
        source_kind="gus_claim_rows",
        bead_id="t42-0b4l.3",
        bootstrap_samples=args.bootstrap_samples,
        wandb_status=wb.status(),
    )

    claim_metrics = [
        gus.summarize_label(
            rows=rows,
            label=label,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for label in TACTICAL_LABELS
    ]
    official_contrasts = [
        summarize_pair_bucket(
            contrast_id=contrast_id,
            pairs=paired_rows(rows, preferred=preferred, alternative=alternative),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for contrast_id, preferred, alternative in contrast_specs()
    ]
    slice_rows = slice_label_metrics(rows, labels=TACTICAL_LABELS, min_slice_n=args.min_slice_n)
    paired_slice_rows = paired_contrast_slices(
        rows,
        min_paired_slice_n=args.min_paired_slice_n,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    label_counts = Counter(label for row in rows for label in row.labels)
    summary = {
        "schema_version": "w42.tactical_claim_replication.v1",
        "bead": "t42-0b4l.3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": gus.git_sha(),
        "git_status_before_artifacts": gus.git_status_short(),
        "inputs": [str(path) for path in paths],
        "file_meta": file_meta,
        "coverage": {
            "input_files": len(paths),
            "action_rows": len(rows),
            "decision_rows": len({row.key for row in rows}),
            "claim_action_rows": sum(1 for row in rows if row.labels),
            "claim_decision_rows": len({row.key for row in rows if row.labels}),
            "label_counts": dict(sorted(label_counts.items())),
        },
        "claim_metrics": claim_metrics,
        "paired_contrasts": official_contrasts,
        "slice_rows": len(slice_rows),
        "paired_slice_rows": len(paired_slice_rows),
        "harness": {
            "action_rows": harness_result.action_rows,
            "decision_rows": harness_result.decision_rows,
            "label_metric_rows": len(harness_result.label_metrics),
            "paired_contrast_rows": len(harness_result.paired_contrasts),
            "artifact_dir": str(output_dir / "harness"),
        },
        "scientific_status": {
            "interpretation": "Powered tactical replication over existing Gus v2 joint-world records with full legal-action row export and reusable harness smoke.",
            "wandb": wb.status(),
            "hf": "not applicable",
            "claim_ledger_impact": "no automatic status movement; existing narrow supported/context-limited readings remain conservative",
            "leakage_boundary": "Public role/trick/action labels are report features. E[Q], q_per_world, world_hands, and sampled-world outcomes are offline labels and diagnostics only.",
        },
        "wall_seconds": time.perf_counter() - started,
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(
        output_dir / "claim_metrics.csv",
        claim_metrics,
        [
            "claim_label",
            "action_n",
            "decision_n",
            "mean_regret",
            "mean_regret_ci95_low",
            "mean_regret_ci95_high",
            "mean_threshold_gap",
            "mean_threshold_gap_ci95_low",
            "mean_threshold_gap_ci95_high",
            "mean_threshold_mass",
            "mean_lower_tail_mass",
            "actual_action_rate",
            "best_mean_rate",
            "best_threshold_rate",
            "safest_tail_rate",
        ],
    )
    write_csv(
        output_dir / "paired_contrasts.csv",
        official_contrasts,
        [
            "contrast_id",
            "paired_decision_n",
            "mean_delta",
            "mean_delta_ci95_low",
            "mean_delta_ci95_high",
            "regret_delta",
            "threshold_mass_delta",
            "lower_tail_mass_delta",
        ],
    )
    slice_fields = [
        "claim_label",
        "slice_name",
        "slice_value",
        "action_n",
        "decision_n",
        "mean",
        "mean_regret",
        "mean_threshold_mass",
        "mean_lower_tail_mass",
        "actual_action_rate",
        "best_mean_rate",
        "best_threshold_rate",
        "safest_tail_rate",
    ]
    write_csv(output_dir / "slice_metrics.csv", slice_rows, slice_fields)
    write_csv(
        output_dir / "paired_contrasts_by_slice.csv",
        paired_slice_rows,
        [
            "base_contrast_id",
            "contrast_id",
            "slice_name",
            "slice_value",
            "paired_decision_n",
            "mean_delta",
            "mean_delta_ci95_low",
            "mean_delta_ci95_high",
            "regret_delta",
            "threshold_mass_delta",
            "lower_tail_mass_delta",
        ],
    )
    write_json(output_dir / "examples.json", gus.build_examples(rows, TACTICAL_LABELS))
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "w42.tactical_claim_replication.manifest.v1",
            "created_at_utc": summary["created_at_utc"],
            "command": " ".join(sys.argv),
            "outputs": {
                "summary": str(output_dir / "summary.json"),
                "claim_metrics": str(output_dir / "claim_metrics.csv"),
                "paired_contrasts": str(output_dir / "paired_contrasts.csv"),
                "slice_metrics": str(output_dir / "slice_metrics.csv"),
                "paired_contrasts_by_slice": str(output_dir / "paired_contrasts_by_slice.csv"),
                "all_action_rows": str(all_action_rows_path),
                "harness_outputs": {name: str(path) for name, path in harness_paths.items()},
                "examples": str(output_dir / "examples.json"),
            },
            "wandb": wb.status(),
            "leakage_boundary": summary["scientific_status"]["leakage_boundary"],
        },
    )

    wb.update_summary(
        {
            "status": "completed",
            "coverage/action_rows": summary["coverage"]["action_rows"],
            "coverage/decision_rows": summary["coverage"]["decision_rows"],
            "coverage/claim_action_rows": summary["coverage"]["claim_action_rows"],
            "claims/label_metric_rows": len(claim_metrics),
            "claims/paired_contrast_rows": len(official_contrasts),
            "slices/slice_rows": len(slice_rows),
            "slices/paired_slice_rows": len(paired_slice_rows),
        }
    )
    wb.log_artifact_files(
        name=f"t42-0b4l.3-tactical-claim-replication-{gus.git_sha()[:7]}",
        artifact_type="w42-report",
        paths=[
            output_dir / "summary.json",
            output_dir / "claim_metrics.csv",
            output_dir / "paired_contrasts.csv",
            output_dir / "slice_metrics.csv",
            output_dir / "paired_contrasts_by_slice.csv",
            output_dir / "examples.json",
            output_dir / "manifest.json",
        ],
    )
    wb.finish()
    print(json.dumps({"coverage": summary["coverage"], "wandb": wb.status()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
