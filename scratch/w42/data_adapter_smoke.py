#!/usr/bin/env python3
"""w42 Forge/Gus data adapter smoke.

This is intentionally tiny: it either reads the first available Gus-style public
state / action / oracle-label corpus row, or emits a deterministic fixture with
the same promoted adapter shape when the declared local corpora are absent.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SEQ_LEN = 33
N_DOMINOES = 28
N_SEATS = 3
N_ACTIONS = 7
STRATEGY_FEATURE_DIM = 68
STRATEGY_ACTION_FEATURE_DIM = 32

DECLARED_SOURCE_PATTERNS = [
    "gus/data/corpus_train_100.pt",
    "gus/data/corpus_train_chunk_*-*.pt",
    "gus/data/corpus_v2_train_*_d0-9.pt",
    "gus/data/corpus_eval_20.pt",
    "gus/data/corpus_v2_eval.pt",
    "data/eq-games/train",
    "data/eq-games/val",
    "data/eq-games/test",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_sha(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def tensor_to_json(x: torch.Tensor) -> Any:
    if x.ndim == 0:
        value = x.item()
        return bool(value) if x.dtype == torch.bool else value
    return x.detach().cpu().tolist()


def shape_of(x: Any) -> list[int]:
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    raise TypeError(f"unsupported batch value: {type(x)!r}")


def summarize_example(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(batch):
        value = batch[key]
        row = value[0] if value.ndim > 0 else value
        out[key] = tensor_to_json(row)
    return out


def fixture_batch(seed: int) -> dict[str, torch.Tensor]:
    """Build one deterministic Gus-shaped training/eval batch.

    The values are synthetic and deliberately small, but every field is shaped
    like the existing Gus full-dataset adapter emits when strategy features are
    requested.
    """
    torch.manual_seed(seed)

    tokens = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones((1, SEQ_LEN), dtype=torch.bool)
    belief_target = torch.tensor([(i + seed) % N_SEATS for i in range(N_DOMINOES)], dtype=torch.long).unsqueeze(0)
    belief_mask = torch.tensor([i % 4 != 0 for i in range(N_DOMINOES)], dtype=torch.bool).unsqueeze(0)

    world_assignment = torch.zeros((1, N_DOMINOES, N_SEATS), dtype=torch.float32)
    for domino in range(N_DOMINOES):
        if bool(belief_mask[0, domino]):
            world_assignment[0, domino, (domino + seed) % N_SEATS] = 1.0

    legal_mask = torch.tensor([[True, True, True, False, True, False, False]], dtype=torch.bool)
    e_q = torch.tensor([[3.25, 6.50, 5.75, -99.0, 4.00, -99.0, -99.0]], dtype=torch.float32)
    action_taken = torch.tensor([1], dtype=torch.long)
    q_per_world = torch.tensor([[2.50, 7.00, 4.75, -8.0, 3.25, -8.0, -8.0]], dtype=torch.float32)

    strategy_features = torch.linspace(0.0, 1.0, STRATEGY_FEATURE_DIM, dtype=torch.float32).unsqueeze(0)
    strategy_action_features = torch.zeros((1, N_ACTIONS, STRATEGY_ACTION_FEATURE_DIM), dtype=torch.float32)
    for action in range(N_ACTIONS):
        strategy_action_features[0, action, 0] = 1.0
        strategy_action_features[0, action, 1] = float(legal_mask[0, action])
        strategy_action_features[0, action, 2] = action / float(N_ACTIONS - 1)

    return {
        "tokens": tokens,
        "attention_mask": attention_mask,
        "belief_target": belief_target,
        "belief_mask": belief_mask,
        "world_assignment": world_assignment,
        "q_per_world": q_per_world,
        "e_q": e_q,
        "action_taken": action_taken,
        "legal_mask": legal_mask,
        "decision_idx": torch.tensor([0], dtype=torch.long),
        "player": torch.tensor([0], dtype=torch.long),
        "voids": torch.zeros((1, 24), dtype=torch.float32),
        "strategy_features": strategy_features,
        "strategy_action_features": strategy_action_features,
    }


def collate_first_rows(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = sorted(set(rows[0]) & set(rows[-1]))
    batch: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [row[key] for row in rows]
        batch[key] = torch.stack(values, dim=0)
    return batch


def existing_sources(root: Path, extra_inputs: list[str]) -> list[Path]:
    patterns = extra_inputs or DECLARED_SOURCE_PATTERNS
    paths: list[Path] = []
    for pattern in patterns:
        full_pattern = str(root / pattern)
        matches = sorted(Path(p) for p in glob.glob(full_pattern))
        if matches:
            paths.extend(p for p in matches if p.is_file())
        else:
            candidate = root / pattern
            if candidate.is_file():
                paths.append(candidate)
    return paths


def load_real_batch(paths: list[Path], seed: int, batch_size: int) -> dict[str, torch.Tensor]:
    from gus.model.dataset_seq_world import JointWorldFullDataset

    ds = JointWorldFullDataset(paths, seed=seed, include_strategy_features=True)
    if len(ds) == 0:
        raise RuntimeError("existing corpus paths loaded, but no joint-world decisions were found")
    rows = [ds[i] for i in range(min(batch_size, len(ds)))]
    return collate_first_rows(rows)


def manifest(root: Path, commit: str, source_mode: str, sources: list[Path], seed: int) -> dict[str, Any]:
    return {
        "schema_version": "w42.dataset_manifest.v1",
        "dataset_id": "w42-data-adapter-smoke-20260502",
        "dataset_version": "v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_commit": commit,
        "owner_bead": "t42-csw6.4",
        "status": "scratch",
        "source_corpora": [
            {
                "path": str(path.relative_to(root) if path.is_absolute() and path.is_relative_to(root) else path),
                "role": "train" if source_mode == "real-corpus" else "train",
                "exists_at_manifest_time": source_mode == "real-corpus",
                "source_kind": "gus-joint-world" if source_mode == "real-corpus" else "other",
                "provenance": (
                    "Existing local Gus joint-world corpus loaded by JointWorldFullDataset."
                    if source_mode == "real-corpus"
                    else "Deterministic synthetic fixture because declared local Gus/Forge corpora were absent."
                ),
                "split_policy": "w42-seed-bucket-v1",
            }
            for path in (sources or [Path("scratch/w42/data_adapter_smoke/example_row.json")])
        ],
        "generation": {
            "command": "python scratch/w42/data_adapter_smoke.py --seed 42 --batch-size 1",
            "cwd": str(root),
            "environment": {
                "device": "cpu",
                "wandb": "not applicable",
                "huggingface": "not applicable",
            },
            "inputs": {
                "checkpoint": "not applicable",
                "seed_ranges": [seed],
                "declarations": "fixture declaration id 0" if source_mode == "fixture" else "from corpus",
                "sampling": "deterministic fixture" if source_mode == "fixture" else "JointWorldFullDataset deterministic seed",
            },
        },
        "splits": {
            "policy": "w42-seed-bucket-v1",
            "train": "seed 42 (42 % 1000 < 900)",
            "val": "not applicable",
            "test": "not applicable",
            "eval": "not applicable",
            "random_seeds": {
                "data_generation": "not applicable",
                "dataset_shuffle": seed,
                "train_loader": seed,
                "eval_sampling": "not applicable",
            },
        },
        "leakage_exclusions": [
            "No held-out eval corpora consumed in fixture mode.",
            "Oracle labels are emitted only as labels, not public-state features.",
        ],
        "labels_available": ["e_q", "q_per_world", "action_taken", "legal_mask"],
        "tags_available": {
            "cheap_strategy_tags": ["strategy_features", "strategy_action_features"],
            "chapter_derived_tags": [],
            "analysis_buckets": [],
        },
        "versioning": {
            "parent_dataset_id": None,
            "parent_dataset_version": None,
            "change_type": "initial",
            "compatibility": "compatible",
            "supersedes": [],
            "notes": "Fixture smoke only; replace with real corpus manifest when local corpora exist.",
        },
        "claim_ledger_impact": "no claim-ledger change",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input", action="append", default=[], help="Optional corpus path or glob, repo-relative.")
    parser.add_argument("--out-dir", default="scratch/w42/data_adapter_smoke")
    args = parser.parse_args()

    root = repo_root()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    commit = git_sha(root)
    sources = existing_sources(root, args.input)
    if sources:
        source_mode = "real-corpus"
        batch = load_real_batch(sources, args.seed, args.batch_size)
        blocker = None
    else:
        source_mode = "fixture"
        batch = fixture_batch(args.seed)
        blocker = "Declared local Gus/Forge corpora are absent in this worktree; real-data adapter smoke is blocked."

    shapes = {key: shape_of(value) for key, value in sorted(batch.items())}
    example = summarize_example(batch)
    e_q = batch["e_q"]
    legal = batch["legal_mask"]
    oracle_best_action = e_q.masked_fill(~legal, float("-inf")).argmax(dim=-1)

    report = {
        "bead_id": "t42-csw6.4",
        "source_mode": source_mode,
        "blocker": blocker,
        "repo_commit": commit,
        "command": "python scratch/w42/data_adapter_smoke.py --seed 42 --batch-size 1",
        "cwd": str(root),
        "random_seeds": {"torch": args.seed, "dataset_shuffle": args.seed, "train_loader": args.seed},
        "wandb_links": "not applicable",
        "hf_links": "not applicable",
        "claim_ledger_impact": "no claim-ledger change",
        "data_inputs": [str(p.relative_to(root)) for p in sources] if sources else DECLARED_SOURCE_PATTERNS,
        "batch_size": int(batch["action_taken"].shape[0]),
        "shapes": shapes,
        "derived_labels": {
            "oracle_best_action": tensor_to_json(oracle_best_action),
            "action_taken": tensor_to_json(batch["action_taken"]),
        },
        "example_row_path": str((out_dir / "example_row.json").relative_to(root)),
        "manifest_path": str((out_dir / "manifest.json").relative_to(root)),
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest(root, commit, source_mode, sources, args.seed), indent=2) + "\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "example_row.json").write_text(json.dumps(example, indent=2) + "\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
