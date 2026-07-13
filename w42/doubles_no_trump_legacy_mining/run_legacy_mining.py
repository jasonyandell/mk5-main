#!/usr/bin/env python3
"""Mine legacy Gus decl-7/decl-9 rows for Chapter 9 tactical probes.

This is a direct follow-up to the W42 claim data inventory. It does not test
same-hand declaration choice; the legacy corpus is seed-modulo by declaration.
It does test action-local, within-regime proxy labels on doubles-trump and
no-trump rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle import tables
from w42.gus_corpus_claim_deep_dive import analyze_gus_claims as gus
from w42.wandb_utils import add_wandb_args, init_wandb


OUT_DIR = Path("w42/doubles_no_trump_legacy_mining")
DOUBLES_TRUMP = 7
NO_TRUMP = 9

LABELS = [
    "ch09_dt_opening_low_double_sacrifice_proxy",
    "ch09_dt_opening_high_double_control_proxy",
    "ch09_dt_dual_suit_top_65_play",
    "ch09_dt_trump_count_capture",
    "ch09_nt_early_support_double_spend_proxy",
    "ch09_nt_late_support_double_spend_proxy",
    "ch09_nt_count_walker_or_capture_proxy",
    "ch09_nt_defender_double_weapon_spend_proxy",
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pip_high(domino_id: int) -> int:
    return int(tables.DOMINO_HIGH[domino_id])


def is_65(domino_id: int) -> bool:
    return int(tables.DOMINO_HIGH[domino_id]) == 6 and int(tables.DOMINO_LOW[domino_id]) == 5


def chapter9_labels(row: gus.ActionRow) -> tuple[str, ...]:
    labels: list[str] = []
    high = pip_high(row.candidate_domino_id)

    if row.decl_id == DOUBLES_TRUMP:
        if row.trick_idx == 0 and row.trick_position == 0 and row.seat_role == "bidder" and row.candidate_is_double:
            if high <= 3:
                labels.append("ch09_dt_opening_low_double_sacrifice_proxy")
            if high >= 5:
                labels.append("ch09_dt_opening_high_double_control_proxy")
        if is_65(row.candidate_domino_id):
            labels.append("ch09_dt_dual_suit_top_65_play")
        if (
            row.candidate_is_called_suit
            and row.candidate_count_points > 0
            and row.trick_position > 0
            and row.candidate_beats_current
        ):
            labels.append("ch09_dt_trump_count_capture")

    if row.decl_id == NO_TRUMP:
        if row.candidate_is_double:
            if row.trick_idx <= 3:
                labels.append("ch09_nt_early_support_double_spend_proxy")
            if row.trick_idx >= 4:
                labels.append("ch09_nt_late_support_double_spend_proxy")
            if (
                row.team == "defense"
                and row.current_winner_team_before == "offense"
                and row.candidate_beats_current
            ):
                labels.append("ch09_nt_defender_double_weapon_spend_proxy")
        if (
            not row.candidate_is_double
            and row.candidate_would_win_trick_now
            and row.candidate_count_points > 0
        ):
            labels.append("ch09_nt_count_walker_or_capture_proxy")

    return tuple(sorted(set(labels)))


def grouped_by_decision(rows: list[gus.ActionRow]) -> dict[str, list[gus.ActionRow]]:
    grouped: dict[str, list[gus.ActionRow]] = defaultdict(list)
    for row in rows:
        grouped[row.key].append(row)
    return grouped


def best_by_mean(rows: list[gus.ActionRow]) -> gus.ActionRow:
    return max(rows, key=lambda row: row.mean)


def labels_for(row: gus.ActionRow, labels_by_row: dict[tuple[str, int], tuple[str, ...]]) -> tuple[str, ...]:
    return labels_by_row.get((row.key, row.candidate_slot), ())


def has_label(label: str, labels_by_row: dict[tuple[str, int], tuple[str, ...]]) -> Callable[[gus.ActionRow], bool]:
    return lambda row: label in labels_for(row, labels_by_row)


def lacks_label(label: str, labels_by_row: dict[tuple[str, int], tuple[str, ...]]) -> Callable[[gus.ActionRow], bool]:
    return lambda row: label not in labels_for(row, labels_by_row)


def label_metrics(
    rows: list[gus.ActionRow],
    labels_by_row: dict[tuple[str, int], tuple[str, ...]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in LABELS:
        selected = [row for row in rows if label in labels_for(row, labels_by_row)]
        mean_regret, regret_lo, regret_hi = gus.bootstrap_mean_ci(
            [row.mean_regret for row in selected],
            samples=bootstrap_samples,
            seed=gus.stable_seed(bootstrap_seed, label),
        )
        out.append(
            {
                "claim_label": label,
                "action_n": len(selected),
                "decision_n": len({row.key for row in selected}),
                "decl_ids": "|".join(str(d) for d in sorted({row.decl_id for row in selected})),
                "mean": gus.mean([row.mean for row in selected]),
                "mean_regret": mean_regret,
                "mean_regret_ci95_low": regret_lo,
                "mean_regret_ci95_high": regret_hi,
                "mean_threshold_mass": gus.mean([row.threshold_mass for row in selected]),
                "mean_lower_tail_mass": gus.mean([row.lower_tail_mass for row in selected]),
                "actual_action_rate": gus.mean([1.0 if row.is_actual_action else 0.0 for row in selected]),
                "best_mean_rate": gus.mean([1.0 if row.is_best_mean else 0.0 for row in selected]),
                "best_threshold_rate": gus.mean([1.0 if row.is_best_threshold else 0.0 for row in selected]),
                "safest_tail_rate": gus.mean([1.0 if row.is_safest_tail else 0.0 for row in selected]),
            }
        )
    return out


def paired_contrast(
    *,
    rows: list[gus.ActionRow],
    labels_by_row: dict[tuple[str, int], tuple[str, ...]],
    contrast_id: str,
    preferred: Callable[[gus.ActionRow], bool],
    alternative: Callable[[gus.ActionRow], bool],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs: list[tuple[gus.ActionRow, gus.ActionRow]] = []
    examples: list[dict[str, Any]] = []
    for decision_rows in grouped_by_decision(rows).values():
        pref_rows = [row for row in decision_rows if preferred(row)]
        alt_rows = [row for row in decision_rows if alternative(row)]
        if not pref_rows or not alt_rows:
            continue
        pref = best_by_mean(pref_rows)
        alt = best_by_mean(alt_rows)
        pairs.append((pref, alt))

    mean_deltas = [pref.mean - alt.mean for pref, alt in pairs]
    threshold_deltas = [pref.threshold_mass - alt.threshold_mass for pref, alt in pairs]
    lower_tail_deltas = [pref.lower_tail_mass - alt.lower_tail_mass for pref, alt in pairs]
    mean_delta, mean_lo, mean_hi = gus.bootstrap_mean_ci(
        mean_deltas,
        samples=bootstrap_samples,
        seed=gus.stable_seed(bootstrap_seed, contrast_id),
    )
    threshold_delta, threshold_lo, threshold_hi = gus.bootstrap_mean_ci(
        threshold_deltas,
        samples=bootstrap_samples,
        seed=gus.stable_seed(bootstrap_seed, contrast_id + ":threshold"),
    )
    for pref, alt in sorted(pairs, key=lambda pair: abs(pair[0].mean - pair[1].mean), reverse=True)[:12]:
        examples.append(
            {
                "contrast_id": contrast_id,
                "key": pref.key,
                "seed": pref.seed,
                "decl_id": pref.decl_id,
                "decl_name": pref.decl_name,
                "decision_idx": pref.decision_idx,
                "trick_idx": pref.trick_idx,
                "trick_position": pref.trick_position,
                "seat_role": pref.seat_role,
                "preferred_domino": pref.candidate_domino,
                "preferred_labels": "|".join(labels_for(pref, labels_by_row)),
                "preferred_mean": pref.mean,
                "preferred_threshold_mass": pref.threshold_mass,
                "alternative_domino": alt.candidate_domino,
                "alternative_labels": "|".join(labels_for(alt, labels_by_row)),
                "alternative_mean": alt.mean,
                "alternative_threshold_mass": alt.threshold_mass,
                "mean_delta": pref.mean - alt.mean,
                "threshold_mass_delta": pref.threshold_mass - alt.threshold_mass,
                "lower_tail_mass_delta": pref.lower_tail_mass - alt.lower_tail_mass,
            }
        )
    return (
        {
            "contrast_id": contrast_id,
            "paired_decision_n": len(pairs),
            "mean_delta": mean_delta,
            "mean_delta_ci95_low": mean_lo,
            "mean_delta_ci95_high": mean_hi,
            "threshold_mass_delta": threshold_delta,
            "threshold_mass_delta_ci95_low": threshold_lo,
            "threshold_mass_delta_ci95_high": threshold_hi,
            "lower_tail_mass_delta": gus.mean(lower_tail_deltas),
        },
        examples,
    )


def build_contrasts(
    rows: list[gus.ActionRow],
    labels_by_row: dict[tuple[str, int], tuple[str, ...]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [
        (
            "ch09_dt_low_double_lead_vs_nonlow_lead_same_decision",
            has_label("ch09_dt_opening_low_double_sacrifice_proxy", labels_by_row),
            lacks_label("ch09_dt_opening_low_double_sacrifice_proxy", labels_by_row),
        ),
        (
            "ch09_dt_low_double_lead_vs_high_double_lead_same_decision",
            has_label("ch09_dt_opening_low_double_sacrifice_proxy", labels_by_row),
            has_label("ch09_dt_opening_high_double_control_proxy", labels_by_row),
        ),
        (
            "ch09_dt_dual_suit_65_vs_other_same_decision",
            has_label("ch09_dt_dual_suit_top_65_play", labels_by_row),
            lacks_label("ch09_dt_dual_suit_top_65_play", labels_by_row),
        ),
        (
            "ch09_nt_early_double_spend_vs_non_double_same_decision",
            has_label("ch09_nt_early_support_double_spend_proxy", labels_by_row),
            lambda row: row.decl_id == NO_TRUMP and not row.candidate_is_double,
        ),
        (
            "ch09_nt_late_double_spend_vs_non_double_same_decision",
            has_label("ch09_nt_late_support_double_spend_proxy", labels_by_row),
            lambda row: row.decl_id == NO_TRUMP and not row.candidate_is_double,
        ),
        (
            "ch09_nt_defender_double_weapon_vs_other_same_decision",
            has_label("ch09_nt_defender_double_weapon_spend_proxy", labels_by_row),
            lacks_label("ch09_nt_defender_double_weapon_spend_proxy", labels_by_row),
        ),
    ]
    rows_out: list[dict[str, Any]] = []
    examples_out: list[dict[str, Any]] = []
    for contrast_id, preferred, alternative in specs:
        row, examples = paired_contrast(
            rows=rows,
            labels_by_row=labels_by_row,
            contrast_id=contrast_id,
            preferred=preferred,
            alternative=alternative,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        rows_out.append(row)
        examples_out.extend(examples)
    return rows_out, examples_out


def example_rows(
    rows: list[gus.ActionRow],
    labels_by_row: dict[tuple[str, int], tuple[str, ...]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in LABELS:
        selected = [
            row for row in rows if label in labels_for(row, labels_by_row)
        ]
        for row in sorted(selected, key=lambda row: row.mean_regret, reverse=True)[:6]:
            out.append(
                {
                    "claim_label": label,
                    "key": row.key,
                    "seed": row.seed,
                    "decl_id": row.decl_id,
                    "decl_name": row.decl_name,
                    "decision_idx": row.decision_idx,
                    "trick_idx": row.trick_idx,
                    "trick_position": row.trick_position,
                    "seat_role": row.seat_role,
                    "team": row.team,
                    "current_winner_team_before": row.current_winner_team_before,
                    "candidate": row.candidate_domino,
                    "candidate_count_points": row.candidate_count_points,
                    "mean": row.mean,
                    "mean_regret": row.mean_regret,
                    "threshold_mass": row.threshold_mass,
                    "lower_tail_mass": row.lower_tail_mass,
                    "is_actual_action": row.is_actual_action,
                    "labels": "|".join(labels_for(row, labels_by_row)),
                }
            )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=["gus/data/corpus_train_chunk_*-*.pt"])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260503)
    parser.add_argument("--log-every-files", type=int, default=5)
    add_wandb_args(
        parser,
        default_group="w42-doubles-no-trump-legacy-mining",
        default_enabled=True,
    )
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
        "bead_id": "t42-0b4l.8",
        "experiment": "w42-doubles-no-trump-legacy-mining",
        "inputs": [str(path) for path in paths],
        "labels": LABELS,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "interpretation_boundary": "within-regime action-local proxies only; not same-hand declaration choice",
    }
    wb = init_wandb(
        args,
        config=config,
        output_dir=output_dir,
        tags=["w42", "winning42", "doubles", "no-trump", "gus-legacy", "t42-0b4l.8"],
    )
    if getattr(wb, "run", None) is not None:
        wb.run.define_metric("progress/files_processed")
        for pattern in ("progress/*", "coverage/*", "labels/*"):
            wb.run.define_metric(pattern, step_metric="progress/files_processed")

    started = time.perf_counter()
    rows: list[gus.ActionRow] = []
    file_meta: list[dict[str, Any]] = []
    labels_by_row: dict[tuple[str, int], tuple[str, ...]] = {}
    for idx, path in enumerate(paths, start=1):
        file_rows, meta = gus.load_payload_rows(path)
        filtered = [row for row in file_rows if row.decl_id in {DOUBLES_TRUMP, NO_TRUMP}]
        rows.extend(filtered)
        file_meta.append({**meta, "filtered_action_rows": len(filtered)})
        for row in filtered:
            labels = chapter9_labels(row)
            if labels:
                labels_by_row[(row.key, row.candidate_slot)] = labels
        if idx % max(args.log_every_files, 1) == 0 or idx == len(paths):
            counts = Counter(label for labels in labels_by_row.values() for label in labels)
            wb.log(
                {
                    "progress/files_processed": idx,
                    "progress/wall_seconds": time.perf_counter() - started,
                    "coverage/action_rows": len(rows),
                    "coverage/decision_rows": len({row.key for row in rows}),
                    "coverage/labeled_action_rows": len(labels_by_row),
                    **{f"labels/{label}": counts[label] for label in LABELS},
                },
                step=idx,
            )

    metrics = label_metrics(
        rows,
        labels_by_row,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    contrasts, contrast_examples = build_contrasts(
        rows,
        labels_by_row,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    examples = example_rows(rows, labels_by_row)
    label_counts = Counter(label for labels in labels_by_row.values() for label in labels)
    decl_counts = Counter(row.decl_id for row in rows)
    summary = {
        "schema_version": "w42.doubles_no_trump_legacy_mining.v0",
        "bead_id": "t42-0b4l.8",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": gus.git_sha(),
        "inputs": [str(path) for path in paths],
        "file_meta": file_meta,
        "coverage": {
            "input_files": len(paths),
            "action_rows": len(rows),
            "decision_rows": len({row.key for row in rows}),
            "decl_counts": dict(sorted(decl_counts.items())),
            "labeled_action_rows": len(labels_by_row),
            "label_counts": dict(sorted(label_counts.items())),
        },
        "label_metrics": metrics,
        "paired_contrasts": contrasts,
        "scientific_status": {
            "interpretation": (
                "Direct legacy Gus corpus mining for Chapter 9 within-regime tactical proxies. "
                "This does not validate no-trump-over-doubles-trump declaration choice because "
                "legacy seeds are not same-hand paired across declaration regimes."
            ),
            "claim_ledger_impact": "No central claim status changes from proxy labels alone.",
            "leakage_boundary": "E[Q], q_per_world, and hidden sampled worlds are offline labels only.",
            "wandb": wb.status(),
        },
        "wall_seconds": time.perf_counter() - started,
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(
        output_dir / "label_metrics.csv",
        metrics,
        [
            "claim_label",
            "action_n",
            "decision_n",
            "decl_ids",
            "mean",
            "mean_regret",
            "mean_regret_ci95_low",
            "mean_regret_ci95_high",
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
        contrasts,
        [
            "contrast_id",
            "paired_decision_n",
            "mean_delta",
            "mean_delta_ci95_low",
            "mean_delta_ci95_high",
            "threshold_mass_delta",
            "threshold_mass_delta_ci95_low",
            "threshold_mass_delta_ci95_high",
            "lower_tail_mass_delta",
        ],
    )
    write_json(output_dir / "examples.json", examples)
    write_json(output_dir / "paired_examples.json", contrast_examples)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "w42.doubles_no_trump_legacy_mining.manifest.v0",
            "created_at_utc": summary["created_at_utc"],
            "command": " ".join(sys.argv),
            "outputs": {
                "summary": str(output_dir / "summary.json"),
                "label_metrics": str(output_dir / "label_metrics.csv"),
                "paired_contrasts": str(output_dir / "paired_contrasts.csv"),
                "examples": str(output_dir / "examples.json"),
                "paired_examples": str(output_dir / "paired_examples.json"),
            },
            "large_row_policy": "No full action-row dump is emitted; rerun from corpus paths for row-level detail.",
            "leakage_boundary": summary["scientific_status"]["leakage_boundary"],
            "wandb": wb.status(),
        },
    )
    wb.update_summary(
        {
            "coverage/action_rows": summary["coverage"]["action_rows"],
            "coverage/decision_rows": summary["coverage"]["decision_rows"],
            "coverage/labeled_action_rows": summary["coverage"]["labeled_action_rows"],
            "status": "completed",
        }
    )
    wb.log_artifact_files(
        name=f"w42-doubles-no-trump-legacy-mining-{gus.git_sha()[:7]}",
        artifact_type="w42-report",
        paths=[
            output_dir / "summary.json",
            output_dir / "label_metrics.csv",
            output_dir / "paired_contrasts.csv",
            output_dir / "examples.json",
            output_dir / "paired_examples.json",
            output_dir / "manifest.json",
        ],
    )
    wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
