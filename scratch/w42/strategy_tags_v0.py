#!/usr/bin/env python3
"""w42 Strategy Detector v0 public-state tag surface.

This wrapper keeps the detector/report surface under w42 while reusing the
cheap public-state and action-local feature extractors already proven useful by
the Gus strategy-tags probe. It does not mutate Gus code or training defaults.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
for _entry in (str(_ROOT_FOR_IMPORTS / "scratch" / "w42"), str(_ROOT_FOR_IMPORTS)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)


DECLARED_SOURCE_PATTERNS = [
    "gus/data/corpus_train_100.pt",
    "gus/data/corpus_train_chunk_*-*.pt",
    "gus/data/corpus_v2_train_*_d0-9.pt",
    "data/eq-games/train",
    "data/eq-games/val",
    "data/eq-games/test",
    "gus/data/corpus_eval_20.pt",
    "gus/data/corpus_v2_eval.pt",
]


@dataclass(frozen=True)
class TagDef:
    idx: int
    name: str
    group: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_imports(root: Path) -> None:
    root_str = str(root)
    scratch_w42 = str(root / "scratch" / "w42")
    for entry in (scratch_w42, root_str):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def git_sha(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def round_float(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def tensor_to_json(value: torch.Tensor) -> Any:
    if value.ndim == 0:
        item = value.item()
        return bool(item) if value.dtype == torch.bool else round_float(item)
    return value.detach().cpu().tolist()


def build_global_tags() -> list[TagDef]:
    from forge.oracle.declarations import DECL_ID_TO_NAME, N_DECLS

    tags: list[TagDef] = []

    def add(group: str, names: list[str]) -> None:
        start = len(tags)
        tags.extend(TagDef(start + i, name, group) for i, name in enumerate(names))

    add("declaration", [f"decl_{i}_{DECL_ID_TO_NAME.get(i, str(i)).replace('-', '_')}" for i in range(N_DECLS)])
    add("phase", ["decision_fraction", "trick_index_fraction", "trick_position_fraction"])
    add(
        "hand_shape",
        [
            "current_hand_frac",
            "hand_trump_frac",
            "hand_called_frac",
            "hand_double_frac",
            "hand_count_tile_frac",
            "hand_count_point_frac",
            "hand_off_tile_frac",
            "hand_max_called_rank_frac",
            "hand_has_called_count",
            "hand_off_count_point_frac",
        ],
    )
    add(
        "legal_action_summary",
        [
            "legal_action_frac",
            "legal_trump_frac",
            "legal_double_frac",
            "legal_count_tile_frac",
            "legal_count_point_frac",
            "must_follow_or_restricted",
        ],
    )
    add(
        "public_count",
        [
            "played_tile_frac",
            "played_count_point_frac",
            "unknown_count_point_frac",
            "played_called_frac",
            "unseen_called_frac",
        ],
    )
    add("void_summary", ["void_mean_all", "void_left_mean", "void_partner_mean", "void_right_mean"])
    add("visible_pip_coverage", [f"visible_pip_{pip}_coverage" for pip in range(7)])
    add(
        "current_trick_pressure",
        [
            "current_trick_count_point_frac",
            "current_trick_trump_frac",
            "current_trick_count_tile_frac",
            "current_trick_has_ten_count",
            "current_trick_has_trump",
            "partner_currently_winning",
            "opponent_currently_winning",
            "is_lead_position",
            "is_last_to_play",
        ],
    )
    add("own_pip_coverage", [f"own_pip_{pip}_coverage" for pip in range(7)])
    add("unseen_count_by_pip", [f"unseen_count_pip_{pip}_frac" for pip in range(7)])
    return tags


def build_action_tags() -> list[TagDef]:
    names = [
        ("slot", "present"),
        ("slot", "legal"),
        ("identity", "called"),
        ("identity", "trump"),
        ("identity", "double"),
        ("identity", "count_points_frac"),
        ("identity", "called_rank_frac"),
        ("identity", "high_pip_frac"),
        ("identity", "low_pip_frac"),
        ("identity", "off_non_double"),
        ("trick_relation", "follows_led"),
        ("trick_relation", "beats_current"),
        ("count_pressure", "is_ten_count"),
        ("trick_relation", "is_lead_position"),
        ("trick_relation", "partner_currently_winning"),
        ("trick_relation", "opponent_currently_winning"),
        ("trick_relation", "trump_in"),
        ("count_pressure", "point_dump"),
        ("suit_pressure", "live_suit_frac"),
        ("suit_pressure", "live_higher_suit_frac"),
        ("suit_pressure", "live_count_frac"),
        ("suit_pressure", "live_higher_count_frac"),
        ("pip_pressure", "pip_count_risk"),
        ("pip_pressure", "min_pip_coverage"),
        ("pip_pressure", "max_pip_coverage"),
        ("double_protection", "protected_by_my_double"),
        ("double_protection", "protected_by_high_double"),
        ("double_protection", "protected_by_low_double"),
        ("hand_shape", "hand_high_pip_frac"),
        ("hand_shape", "hand_low_pip_frac"),
        ("donation_window", "count_donation_to_partner"),
        ("donation_window", "count_donation_to_opponent"),
    ]
    return [TagDef(i, name, group) for i, (group, name) in enumerate(names)]


GLOBAL_TAGS = build_global_tags()
ACTION_TAGS = build_action_tags()


def validate_tag_dims() -> None:
    from gus.model.strategy_features import STRATEGY_ACTION_FEATURE_DIM, STRATEGY_FEATURE_DIM

    if len(GLOBAL_TAGS) != STRATEGY_FEATURE_DIM:
        raise AssertionError(f"global tag metadata drifted: {len(GLOBAL_TAGS)} != {STRATEGY_FEATURE_DIM}")
    if len(ACTION_TAGS) != STRATEGY_ACTION_FEATURE_DIM:
        raise AssertionError(
            f"action tag metadata drifted: {len(ACTION_TAGS)} != {STRATEGY_ACTION_FEATURE_DIM}"
        )


def existing_sources(root: Path, extra_inputs: list[str]) -> list[Path]:
    patterns = extra_inputs or DECLARED_SOURCE_PATTERNS
    for pattern in patterns:
        full_pattern = str(root / pattern)
        matches = sorted(Path(p) for p in glob.glob(full_pattern))
        files = [p for p in matches if p.is_file()]
        if files:
            return [files[0]]
        candidate = root / pattern
        if candidate.is_file():
            return [candidate]
    return []


def collate_rows(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = sorted(rows[0])
    return {key: torch.stack([row[key] for row in rows], dim=0) for key in keys}


def load_real_batch(paths: list[Path], seed: int, limit: int) -> tuple[dict[str, torch.Tensor], int]:
    from gus.model.dataset_seq_world import JointWorldFullDataset

    ds = JointWorldFullDataset(paths, seed=seed, include_strategy_features=True)
    if len(ds) == 0:
        raise RuntimeError("existing corpus paths loaded, but no joint-world decisions were found")
    rows = [ds[i] for i in range(min(limit, len(ds)))]
    return collate_rows(rows), len(ds)


def load_fixture_batch(seed: int) -> tuple[dict[str, torch.Tensor], int]:
    from data_adapter_smoke import fixture_batch

    return fixture_batch(seed), 1


def shape_map(batch: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in sorted(batch.items())}


def group_slices(tags: list[TagDef]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for tag in tags:
        groups.setdefault(tag.group, []).append(tag.idx)
    return groups


def summarize_groups(values: torch.Tensor, tags: list[TagDef]) -> dict[str, dict[str, Any]]:
    groups = group_slices(tags)
    flat = values.reshape(-1, values.shape[-1]).float()
    out: dict[str, dict[str, Any]] = {}
    for group, idxs in groups.items():
        group_values = flat[:, idxs]
        out[group] = {
            "width": len(idxs),
            "mean": round(float(group_values.mean().item()), 6),
            "min": round(float(group_values.min().item()), 6),
            "max": round(float(group_values.max().item()), 6),
            "nonzero_fraction": round(float((group_values.abs() > 1e-9).float().mean().item()), 6),
        }
    return out


def named_global_row(row: torch.Tensor) -> dict[str, float]:
    return {tag.name: round(float(row[tag.idx].item()), 6) for tag in GLOBAL_TAGS}


def named_action_rows(rows: torch.Tensor) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for slot, row in enumerate(rows):
        values = {tag.name: round(float(row[tag.idx].item()), 6) for tag in ACTION_TAGS}
        nonzero = {name: value for name, value in values.items() if abs(value) > 1e-9}
        out.append({"slot": slot, "nonzero_tags": nonzero})
    return out


def write_schema(out_dir: Path) -> None:
    schema = {
        "schema_version": "w42.strategy_tags_v0.schema.v1",
        "global_dim": len(GLOBAL_TAGS),
        "action_dim": len(ACTION_TAGS),
        "global_tags": [tag.__dict__ for tag in GLOBAL_TAGS],
        "action_tags": [tag.__dict__ for tag in ACTION_TAGS],
        "source": "Gus cheap public-state/action-local extractors, exposed through w42-owned metadata.",
    }
    (out_dir / "tag_schema.json").write_text(json.dumps(schema, indent=2) + "\n")


def write_group_csv(path: Path, global_groups: dict[str, Any], action_groups: dict[str, Any]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["surface", "group", "width", "mean", "min", "max", "nonzero_fraction"],
            lineterminator="\n",
        )
        writer.writeheader()
        for surface, groups in (("global", global_groups), ("action", action_groups)):
            for group, row in groups.items():
                writer.writerow({"surface": surface, "group": group, **row})


def build_manifest(
    root: Path,
    commit: str,
    source_mode: str,
    sources: list[Path],
    seed: int,
    command: str,
) -> dict[str, Any]:
    source_paths = sources or [root / "scratch" / "w42" / "data_adapter_smoke" / "example_row.json"]
    return {
        "schema_version": "w42.dataset_manifest.v1",
        "dataset_id": "w42-strategy-tags-v0-20260502",
        "dataset_version": "v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": commit,
        "owner_bead": "t42-csw6.7",
        "status": "scratch",
        "source_corpora": [
            {
                "path": str(path.relative_to(root) if path.is_absolute() and path.is_relative_to(root) else path),
                "role": "train",
                "exists_at_manifest_time": source_mode == "real-corpus",
                "source_kind": "gus-joint-world" if source_mode == "real-corpus" else "other",
                "provenance": (
                    "Existing local Gus joint-world corpus loaded with include_strategy_features=True."
                    if source_mode == "real-corpus"
                    else "Existing w42 data-adapter smoke deterministic fixture fallback."
                ),
                "split_policy": "w42-seed-bucket-v1",
            }
            for path in source_paths
        ],
        "generation": {
            "command": command,
            "cwd": str(root),
            "environment": {
                "device": "cpu",
                "wandb": "not applicable",
                "huggingface": "not applicable",
            },
            "inputs": {
                "checkpoint": "not applicable",
                "seed_ranges": [seed],
                "declarations": "from corpus" if source_mode == "real-corpus" else "fixture declaration id 0",
                "sampling": (
                    "JointWorldFullDataset deterministic world sampling seed"
                    if source_mode == "real-corpus"
                    else "deterministic fixture"
                ),
            },
        },
        "splits": {
            "policy": "w42-seed-bucket-v1",
            "train": (
                "corpus rows inherit their source seed split"
                if source_mode == "real-corpus"
                else "seed 42 (42 % 1000 < 900) for fixture mode"
            ),
            "val": "not applicable",
            "test": "not applicable",
            "eval": "not applicable",
            "random_seeds": {
                "data_generation": "not applicable",
                "dataset_shuffle": seed,
                "train_loader": seed,
                "eval_sampling": "not applicable",
                "world_sampling": seed,
            },
        },
        "leakage_exclusions": [
            "Oracle labels are labels only; they are not part of public-state detector tags.",
            "No Burl traces or table-talk text consumed.",
            "No held-out eval corpus consumed unless explicitly supplied by --input.",
        ],
        "labels_available": ["e_q", "q_per_world", "action_taken", "legal_mask"],
        "tags_available": {
            "cheap_strategy_tags": [tag.name for tag in GLOBAL_TAGS],
            "action_local_strategy_tags": [tag.name for tag in ACTION_TAGS],
            "chapter_derived_tags": [],
            "analysis_buckets": list(group_slices(GLOBAL_TAGS).keys()) + list(group_slices(ACTION_TAGS).keys()),
        },
        "versioning": {
            "parent_dataset_id": None,
            "parent_dataset_version": None,
            "change_type": "initial",
            "compatibility": "compatible",
            "supersedes": [],
            "notes": "v0 detector surface only; no strategy claim tested.",
        },
        "claim_ledger_impact": "no claim-ledger change",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--input", action="append", default=[], help="Optional corpus path or glob, repo-relative.")
    parser.add_argument("--out-dir", default="scratch/w42/strategy_tags_v0")
    args = parser.parse_args()

    root = repo_root()
    ensure_repo_imports(root)
    validate_tag_dims()

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = git_sha(root)
    command = f"python scratch/w42/strategy_tags_v0.py --seed {args.seed} --limit {args.limit}"
    sources = existing_sources(root, args.input)
    if sources:
        source_mode = "real-corpus"
        batch, available_rows = load_real_batch(sources, args.seed, args.limit)
        blocker = None
    else:
        source_mode = "fixture"
        batch, available_rows = load_fixture_batch(args.seed)
        blocker = "Declared local Gus/Forge corpora are absent in this worktree; using the data-adapter smoke fixture fallback."

    global_values = batch["strategy_features"].float()
    action_values = batch["strategy_action_features"].float()
    global_groups = summarize_groups(global_values, GLOBAL_TAGS)
    action_groups = summarize_groups(action_values, ACTION_TAGS)

    e_q = batch["e_q"]
    legal = batch["legal_mask"]
    oracle_best_action = e_q.masked_fill(~legal, float("-inf")).argmax(dim=-1)
    example = {
        "source_mode": source_mode,
        "decision_idx": tensor_to_json(batch["decision_idx"][0]),
        "player": tensor_to_json(batch["player"][0]),
        "action_taken": tensor_to_json(batch["action_taken"][0]),
        "oracle_best_action": tensor_to_json(oracle_best_action[0]),
        "legal_mask": tensor_to_json(batch["legal_mask"][0]),
        "global_tags": named_global_row(global_values[0]),
        "action_tags_by_slot": named_action_rows(action_values[0]),
    }

    report = {
        "bead_id": "t42-csw6.7",
        "source_mode": source_mode,
        "blocker": blocker,
        "repo_commit": commit,
        "command": command,
        "cwd": str(root),
        "random_seeds": {
            "torch": args.seed,
            "dataset_shuffle": args.seed,
            "train_loader": args.seed,
            "world_sampling": args.seed,
        },
        "wandb_links": "not applicable",
        "hf_links": "not applicable",
        "claim_ledger_impact": "no claim-ledger change",
        "data_inputs": [str(p.relative_to(root)) for p in sources] if sources else DECLARED_SOURCE_PATTERNS,
        "available_rows": available_rows,
        "sampled_rows": int(global_values.shape[0]),
        "shapes": shape_map(batch),
        "tag_dimensions": {
            "strategy_features": len(GLOBAL_TAGS),
            "strategy_action_features_per_action": len(ACTION_TAGS),
            "action_slots": int(action_values.shape[1]),
        },
        "global_group_summary": global_groups,
        "action_group_summary": action_groups,
        "derived_labels": {
            "oracle_best_action": tensor_to_json(oracle_best_action),
            "action_taken": tensor_to_json(batch["action_taken"]),
        },
        "artifacts": {
            "manifest": str((out_dir / "manifest.json").relative_to(root)),
            "tag_schema": str((out_dir / "tag_schema.json").relative_to(root)),
            "example_row": str((out_dir / "example_row.json").relative_to(root)),
            "summary_csv": str((out_dir / "summary.csv").relative_to(root)),
        },
    }

    write_schema(out_dir)
    write_group_csv(out_dir / "summary.csv", global_groups, action_groups)
    (out_dir / "manifest.json").write_text(
        json.dumps(build_manifest(root, commit, source_mode, sources, args.seed, command), indent=2) + "\n"
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "example_row.json").write_text(json.dumps(example, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
