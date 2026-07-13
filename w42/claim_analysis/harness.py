"""Small reusable harness for w42 claim-label reports.

The harness operates on public/action rows that already expose candidate-action
facts and offline evaluation labels. Hidden truth, oracle Q, and sampled worlds
are treated as report labels only; detectors must declare their live/label
boundary in the calling artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ClaimSpec:
    """Metadata for a claim detector or label family."""

    claim_id: str
    family: str
    label: str
    description: str
    online_fields: tuple[str, ...]
    offline_label_fields: tuple[str, ...]
    leakage_policy: str


@dataclass(frozen=True)
class LabelMetric:
    label: str
    action_n: int
    decision_n: int
    actual_action_rate: float
    best_mean_rate: float
    best_threshold_rate: float
    safest_tail_rate: float
    mean: float
    mean_regret: float
    mean_regret_ci95_low: float
    mean_regret_ci95_high: float
    mean_threshold_mass: float
    mean_lower_tail_mass: float


@dataclass(frozen=True)
class ContrastResult:
    contrast_id: str
    label: str
    paired_decision_n: int
    mean_delta: float
    mean_delta_ci95_low: float
    mean_delta_ci95_high: float
    regret_delta: float
    threshold_mass_delta: float
    lower_tail_mass_delta: float


@dataclass(frozen=True)
class RunResult:
    label_metrics: list[LabelMetric]
    paired_contrasts: list[ContrastResult]
    examples: list[dict[str, Any]]
    label_counts: dict[str, int]
    decision_rows: int
    action_rows: int


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def split_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    labels: list[str] = []
    for part in text.replace(",", "|").split("|"):
        part = part.strip()
        if part:
            labels.append(part)
    return labels


def normalize_labels(row: dict[str, Any], label_fields: tuple[str, ...]) -> tuple[str, ...]:
    labels: list[str] = []
    for field in label_fields:
        labels.extend(split_labels(row.get(field)))
    return tuple(sorted(set(labels)))


def decision_key(row: dict[str, Any]) -> str:
    if row.get("key"):
        return str(row["key"])
    game = row.get("game_id") or row.get("seed") or row.get("game_idx") or "game"
    decl = row.get("decl_id") or row.get("trump_id") or row.get("decl_name") or "decl"
    decision = row.get("decision_idx") or row.get("move_idx") or "decision"
    actor = row.get("actor") or row.get("active_player") or "actor"
    return f"{game}:{decl}:{decision}:{actor}"


def normalized_row(row: dict[str, Any], *, label_fields: tuple[str, ...]) -> dict[str, Any]:
    labels = normalize_labels(row, label_fields)
    mean = parse_float(row.get("mean"))
    threshold_mass = parse_float(
        row.get("threshold_mass", row.get("visualizer_threshold_mass", row.get("make_mass_ge_18_offense_only"))),
        default=float("nan"),
    )
    lower_tail_mass = parse_float(row.get("lower_tail_mass", row.get("lower_tail_mass_le_neg18")), default=float("nan"))
    mean_regret = parse_float(row.get("mean_regret", row.get("mean_gap_to_best")), default=float("nan"))
    threshold_gap = parse_float(row.get("threshold_gap", row.get("threshold_gap_to_best")), default=float("nan"))
    lower_tail_gap = parse_float(row.get("lower_tail_gap", row.get("lower_tail_gap_to_safest")), default=float("nan"))
    return {
        **row,
        "_key": decision_key(row),
        "_labels": labels,
        "_mean": mean,
        "_threshold_mass": threshold_mass,
        "_lower_tail_mass": lower_tail_mass,
        "_mean_regret": mean_regret,
        "_threshold_gap": threshold_gap,
        "_lower_tail_gap": lower_tail_gap,
        "_is_actual_action": parse_bool(row.get("is_actual_action")),
        "_is_best_mean": parse_bool(row.get("is_best_mean", row.get("is_top_mean_action"))),
        "_is_best_threshold": parse_bool(row.get("is_best_threshold", row.get("is_top_threshold_action"))),
        "_is_safest_tail": parse_bool(row.get("is_safest_tail", row.get("is_safest_lower_tail_action"))),
    }


def load_rows(path: Path, *, label_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            raw_rows = [json.loads(line) for line in f if line.strip()]
    else:
        with path.open(newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))
    return [normalized_row(row, label_fields=label_fields) for row in raw_rows]


def finite(values: Iterable[float]) -> list[float]:
    return [v for v in values if isinstance(v, int | float) and math.isfinite(float(v))]


def mean(values: Iterable[float]) -> float:
    nums = finite(values)
    if not nums:
        return float("nan")
    return float(sum(nums) / len(nums))


def bootstrap_mean_ci(values: Iterable[float], *, samples: int, seed: int) -> tuple[float, float, float]:
    nums = finite(values)
    if not nums:
        return (float("nan"), float("nan"), float("nan"))
    center = mean(nums)
    if len(nums) == 1 or samples <= 0:
        return (center, center, center)
    rng = random.Random(seed)
    draws: list[float] = []
    n = len(nums)
    for _ in range(samples):
        draws.append(sum(nums[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return (center, float(lo), float(hi))


def grouped_by_decision(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["_key"])].append(row)
    return grouped


def best_by_mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: parse_float(row.get("_mean"), default=-1e12))


def row_label_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row["_labels"])
    return counts


def summarize_label(
    rows: list[dict[str, Any]],
    *,
    label: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> LabelMetric:
    selected = [row for row in rows if label in row["_labels"]]
    regret, regret_lo, regret_hi = bootstrap_mean_ci(
        [row["_mean_regret"] for row in selected],
        samples=bootstrap_samples,
        seed=stable_seed(bootstrap_seed, label),
    )
    return LabelMetric(
        label=label,
        action_n=len(selected),
        decision_n=len({row["_key"] for row in selected}),
        actual_action_rate=mean(1.0 if row["_is_actual_action"] else 0.0 for row in selected),
        best_mean_rate=mean(1.0 if row["_is_best_mean"] else 0.0 for row in selected),
        best_threshold_rate=mean(1.0 if row["_is_best_threshold"] else 0.0 for row in selected),
        safest_tail_rate=mean(1.0 if row["_is_safest_tail"] else 0.0 for row in selected),
        mean=mean(row["_mean"] for row in selected),
        mean_regret=regret,
        mean_regret_ci95_low=regret_lo,
        mean_regret_ci95_high=regret_hi,
        mean_threshold_mass=mean(row["_threshold_mass"] for row in selected),
        mean_lower_tail_mass=mean(row["_lower_tail_mass"] for row in selected),
    )


def paired_label_contrast(
    rows: list[dict[str, Any]],
    *,
    label: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[ContrastResult | None, list[dict[str, Any]]]:
    mean_deltas: list[float] = []
    regret_deltas: list[float] = []
    threshold_deltas: list[float] = []
    lower_tail_deltas: list[float] = []
    examples: list[dict[str, Any]] = []

    for decision_rows in grouped_by_decision(rows).values():
        preferred_rows = [row for row in decision_rows if label in row["_labels"]]
        alternative_rows = [row for row in decision_rows if label not in row["_labels"]]
        if not preferred_rows or not alternative_rows:
            continue
        preferred = best_by_mean(preferred_rows)
        alternative = best_by_mean(alternative_rows)
        mean_delta = preferred["_mean"] - alternative["_mean"]
        regret_delta = preferred["_mean_regret"] - alternative["_mean_regret"]
        threshold_delta = preferred["_threshold_mass"] - alternative["_threshold_mass"]
        lower_tail_delta = preferred["_lower_tail_mass"] - alternative["_lower_tail_mass"]
        mean_deltas.append(mean_delta)
        regret_deltas.append(regret_delta)
        threshold_deltas.append(threshold_delta)
        lower_tail_deltas.append(lower_tail_delta)
        examples.append(
            {
                "label": label,
                "decision_key": preferred["_key"],
                "preferred_domino": preferred.get("candidate_domino"),
                "alternative_domino": alternative.get("candidate_domino"),
                "preferred_mean": preferred["_mean"],
                "alternative_mean": alternative["_mean"],
                "mean_delta": mean_delta,
                "threshold_mass_delta": threshold_delta,
                "lower_tail_mass_delta": lower_tail_delta,
                "preferred_labels": "|".join(preferred["_labels"]),
                "alternative_labels": "|".join(alternative["_labels"]),
            }
        )

    if not mean_deltas:
        return None, []

    mean_delta, mean_lo, mean_hi = bootstrap_mean_ci(
        mean_deltas,
        samples=bootstrap_samples,
        seed=stable_seed(bootstrap_seed, label + ":paired"),
    )
    result = ContrastResult(
        contrast_id=f"{label}_vs_non_{label}",
        label=label,
        paired_decision_n=len(mean_deltas),
        mean_delta=mean_delta,
        mean_delta_ci95_low=mean_lo,
        mean_delta_ci95_high=mean_hi,
        regret_delta=mean(regret_deltas),
        threshold_mass_delta=mean(threshold_deltas),
        lower_tail_mass_delta=mean(lower_tail_deltas),
    )
    examples = sorted(examples, key=lambda item: abs(float(item["mean_delta"])), reverse=True)[:16]
    return result, examples


def analyze_rows(
    rows: list[dict[str, Any]],
    *,
    min_label_n: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> RunResult:
    counts = row_label_counts(rows)
    labels = sorted(label for label, count in counts.items() if count >= min_label_n)
    label_metrics = [
        summarize_label(rows, label=label, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
        for label in labels
    ]
    paired_contrasts: list[ContrastResult] = []
    examples: list[dict[str, Any]] = []
    for label in labels:
        contrast, contrast_examples = paired_label_contrast(
            rows,
            label=label,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        if contrast is not None:
            paired_contrasts.append(contrast)
            examples.extend(contrast_examples)

    return RunResult(
        label_metrics=label_metrics,
        paired_contrasts=paired_contrasts,
        examples=sorted(examples, key=lambda item: abs(float(item["mean_delta"])), reverse=True)[:32],
        label_counts=dict(counts),
        decision_rows=len({row["_key"] for row in rows}),
        action_rows=len(rows),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_result_artifacts(
    *,
    result: RunResult,
    output_dir: Path,
    input_paths: list[Path],
    label_fields: tuple[str, ...],
    claim_specs: tuple[ClaimSpec, ...],
    source_kind: str,
    bead_id: str,
    bootstrap_samples: int,
    wandb_status: Any,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_metrics_path = output_dir / "label_metrics.csv"
    paired_contrasts_path = output_dir / "paired_contrasts.csv"
    label_counts_path = output_dir / "label_counts.csv"
    claim_specs_path = output_dir / "claim_specs.csv"
    examples_path = output_dir / "examples.json"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    write_csv(label_metrics_path, [asdict(row) for row in result.label_metrics], list(LabelMetric.__annotations__))
    write_csv(paired_contrasts_path, [asdict(row) for row in result.paired_contrasts], list(ContrastResult.__annotations__))
    write_csv(
        label_counts_path,
        [{"label": label, "action_n": count} for label, count in sorted(result.label_counts.items())],
        ["label", "action_n"],
    )
    write_csv(
        claim_specs_path,
        [
            {
                **asdict(spec),
                "online_fields": "|".join(spec.online_fields),
                "offline_label_fields": "|".join(spec.offline_label_fields),
            }
            for spec in claim_specs
        ],
        list(ClaimSpec.__annotations__),
    )
    write_json(examples_path, result.examples)

    summary = {
        "schema_version": "w42.claim_analysis.run_result.v1",
        "bead": bead_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_kind": source_kind,
        "input_paths": [str(path) for path in input_paths],
        "label_fields": list(label_fields),
        "registered_claim_specs": len(claim_specs),
        "action_rows": result.action_rows,
        "decision_rows": result.decision_rows,
        "label_metric_rows": len(result.label_metrics),
        "paired_contrast_rows": len(result.paired_contrasts),
        "top_label_counts": dict(Counter(result.label_counts).most_common(20)),
        "bootstrap_samples": bootstrap_samples,
        "wandb": wandb_status,
        "claim_ledger_impact": "no claim-ledger movement; harness smoke/report only",
        "leakage_boundary": "Rows may contain oracle/distribution/hidden-world labels, but this harness treats them as offline labels and does not expose them as live detector inputs.",
    }
    write_json(summary_path, summary)

    manifest = {
        "schema_version": "w42.claim_analysis.manifest.v1",
        "bead": bead_id,
        "created_at_utc": summary["created_at_utc"],
        "repo_commit": git_sha(),
        "git_status_short": git_status_short(),
        "inputs": [str(path) for path in input_paths],
        "outputs": [
            str(label_metrics_path),
            str(paired_contrasts_path),
            str(label_counts_path),
            str(claim_specs_path),
            str(examples_path),
            str(summary_path),
            str(manifest_path),
        ],
    }
    write_json(manifest_path, manifest)
    return {
        "label_metrics": label_metrics_path,
        "paired_contrasts": paired_contrasts_path,
        "label_counts": label_counts_path,
        "claim_specs": claim_specs_path,
        "examples": examples_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def log_row_progress(wb: Any, rows: list[dict[str, Any]], *, chunk_size: int) -> None:
    if chunk_size <= 0:
        return
    running: Counter[str] = Counter()
    for idx, row in enumerate(rows, 1):
        running.update(row["_labels"])
        if idx % chunk_size == 0 or idx == len(rows):
            wb.log_series_point(
                axis="progress/rows_processed",
                value=idx,
                metrics={
                    "coverage/action_rows": idx,
                    "coverage/unique_labels": len(running),
                    "coverage/unique_decisions_seen": len({r["_key"] for r in rows[:idx]}),
                    "labels/top_label_count": max(running.values()) if running else 0,
                },
                step=idx,
            )
