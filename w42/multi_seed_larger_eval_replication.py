"""Multi-seed larger-eval replication for the w42 raw/v0/rich comparison.

This is intentionally a bead-local runner. It reuses the existing w42 research
models and Gus corpus adapter, but owns the replication config, W&B run names,
and aggregate uncertainty tables for t42-csw6.31.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_W42 = ROOT / "w42"
for entry in (str(SCRATCH_W42), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from gus.model.dataset_seq_world import JointWorldFullDataset
from raw_public_state_baseline import (
    EQNWrapper,
    RawPublicStateActionModel,
    _device,
    _subset,
    evaluate_eq_n,
)
from rich_tag_many_signal_probe import RawPlusRichTagsModel
from strategy_tags_v0 import validate_tag_dims
from v0_strategy_tags_baseline import RawPlusV0StrategyTagsModel
from wandb_utils import add_wandb_args, init_wandb


FEATURES = ("raw", "v0", "rich")


@dataclass
class ReplicationConfig:
    bead_id: str
    feature_set: str
    train_paths: list[str]
    eval_paths: list[str]
    train_limit: int
    eval_limit: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    d_model: int
    n_heads: int
    n_layers: int
    ff_dim: int
    dropout: float
    action_hidden: int
    eq_n: int
    seed: int
    split_seed: int
    eval_seed: int
    device: str
    output_dir: str
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_group: str | None
    wandb_name: str | None
    wandb_mode: str
    model_family: str
    dataset_name: str
    dataset_version: str
    ruleset: str
    label_source: str
    decision_slice: str
    baseline_policy: str
    metrics: list[str]
    claim_ledger_status_before: str
    local_artifact_path: str
    hf_repo_id: str


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_status_short() -> str:
    return subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def model_for_feature(feature: str, args: argparse.Namespace) -> nn.Module:
    kwargs = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "action_hidden": args.action_hidden,
    }
    if feature == "raw":
        return RawPublicStateActionModel(**kwargs)
    if feature == "v0":
        return RawPlusV0StrategyTagsModel(**kwargs)
    if feature == "rich":
        return RawPlusRichTagsModel(**kwargs)
    raise ValueError(f"unknown feature set: {feature}")


def oracle_best_action(batch: dict[str, Tensor]) -> Tensor:
    e_q = batch["e_q"].masked_fill(~batch["legal_mask"], float("-inf"))
    return e_q.argmax(dim=-1)


def policy_loss(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    logits = model(batch).masked_fill(~batch["legal_mask"], -1e9)
    return nn.functional.cross_entropy(logits, oracle_best_action(batch))


@torch.no_grad()
def evaluate_policy(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    regret_sum = 0.0
    total = 0
    hits = 0
    near = 0
    tail_ge_5 = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch).masked_fill(~batch["legal_mask"], -1e9)
        action = logits.argmax(dim=-1)
        e_q_legal = batch["e_q"].masked_fill(~batch["legal_mask"], float("-inf"))
        oracle_best = e_q_legal.max(dim=-1).values
        oracle_action = e_q_legal.argmax(dim=-1)
        idx = torch.arange(action.numel(), device=device)
        regret = oracle_best - batch["e_q"][idx, action]
        regret_sum += float(regret.sum().item())
        total += int(action.numel())
        hits += int((action == oracle_action).sum().item())
        near += int((regret < 0.5).sum().item())
        tail_ge_5 += int((regret >= 5.0).sum().item())
    denom = max(total, 1)
    return {
        "mean_regret": regret_sum / denom,
        "match_rate": hits / denom,
        "near_tie_rate_regret_lt_0_5": near / denom,
        "tail_regret_rate_ge_5": tail_ge_5 / denom,
        "n": float(total),
    }


def train_one(
    *,
    feature: str,
    seed: int,
    train_ds: torch.utils.data.Dataset,
    eval_ds: torch.utils.data.Dataset,
    eq_metrics: dict[str, float],
    args: argparse.Namespace,
    device: str,
    sha: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    out_dir = Path(args.output_dir) / f"{feature}_s{seed:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.wandb_name or f"t42-csw6.31-{feature}-larger-eval-s{seed:04d}-{sha[:8]}"
    config = ReplicationConfig(
        bead_id="t42-csw6.31",
        feature_set=feature,
        train_paths=args.train,
        eval_paths=args.eval,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        action_hidden=args.action_hidden,
        eq_n=args.eq_n,
        seed=seed,
        split_seed=args.split_seed,
        eval_seed=args.eval_seed,
        device=device,
        output_dir=str(out_dir),
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_name=run_name,
        wandb_mode=args.wandb_mode,
        model_family="tiny-public-state-action-scorer",
        dataset_name="w42-multi-seed-larger-eval-replication",
        dataset_version="v0",
        ruleset="existing Gus corpus semantics",
        label_source="forge/Gus marginal e_q oracle labels",
        decision_slice="all decisions in selected train/eval corpus rows after deterministic subset",
        baseline_policy=f"E[Q] N={args.eq_n}",
        metrics=[
            "mean_regret",
            "match_rate",
            "near_tie_rate_regret_lt_0_5",
            "tail_regret_rate_ge_5",
        ],
        claim_ledger_status_before="no claim-ledger change",
        local_artifact_path=str(out_dir),
        hf_repo_id="not applicable",
    )
    wb_args = SimpleNamespace(
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_name=run_name,
        wandb_mode=args.wandb_mode,
    )
    wb = init_wandb(
        wb_args,
        config={**asdict(config), "git_sha": sha, "data_manifest": str(out_dir / "manifest.json")},
        output_dir=out_dir,
        tags=[
            "w42",
            "winning42",
            "strategy-validation",
            "forge-eq",
            "gus-format",
            "promoted",
            "replication",
            "larger-eval",
            feature,
            "t42-csw6.31",
        ],
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    model = model_for_feature(feature, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    started_at = datetime.now(UTC).isoformat()
    t0 = time.perf_counter()
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        model.train()
        epoch_t0 = time.perf_counter()
        loss_sum = 0.0
        batches = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = policy_loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            batches += 1
        eval_metrics = evaluate_policy(model, eval_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(batches, 1),
            "epoch_seconds": time.perf_counter() - epoch_t0,
            **eval_metrics,
        }
        history.append(row)
        wb.log(
            {
                "epoch": epoch + 1,
                "train/loss": row["train_loss"],
                "train/epoch_seconds": row["epoch_seconds"],
                "eval/mean_regret": eval_metrics["mean_regret"],
                "eval/match_rate": eval_metrics["match_rate"],
                "eval/near_tie_rate_regret_lt_0_5": eval_metrics[
                    "near_tie_rate_regret_lt_0_5"
                ],
                "eval/tail_regret_rate_ge_5": eval_metrics["tail_regret_rate_ge_5"],
                "eval/n": eval_metrics["n"],
            },
            step=epoch + 1,
        )
        if best_metrics is None or eval_metrics["mean_regret"] < best_metrics["mean_regret"]:
            best_metrics = eval_metrics
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
        print(
            f"{feature} seed={seed} epoch {epoch+1:02d}/{args.epochs} "
            f"loss={row['train_loss']:.3f} regret={eval_metrics['mean_regret']:.3f} "
            f"match={eval_metrics['match_rate']:.2%}",
            flush=True,
        )

    assert best_metrics is not None
    assert best_state is not None
    final_metrics = evaluate_policy(model, eval_loader, device)
    metrics = {
        "feature_set": feature,
        "seed": seed,
        "best_epoch": best_epoch,
        "best": best_metrics,
        "final": final_metrics,
        f"e_q_n_{args.eq_n}": eq_metrics,
        "deltas": {
            f"best_minus_e_q_n_{args.eq_n}_mean_regret": best_metrics["mean_regret"]
            - eq_metrics["mean_regret"],
            f"final_minus_e_q_n_{args.eq_n}_mean_regret": final_metrics["mean_regret"]
            - eq_metrics["mean_regret"],
        },
    }
    wb.log(
        {
            "best/epoch": best_epoch,
            "best/mean_regret": best_metrics["mean_regret"],
            "best/match_rate": best_metrics["match_rate"],
            "best/near_tie_rate_regret_lt_0_5": best_metrics[
                "near_tie_rate_regret_lt_0_5"
            ],
            "best/tail_regret_rate_ge_5": best_metrics["tail_regret_rate_ge_5"],
            "final/mean_regret": final_metrics["mean_regret"],
            f"baseline/e_q_n_{args.eq_n}_mean_regret": eq_metrics["mean_regret"],
        },
        step=args.epochs,
    )
    wb.update_summary(
        {
            "status": "completed",
            "feature_set": feature,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_mean_regret": best_metrics["mean_regret"],
            "best_match_rate": best_metrics["match_rate"],
            "best_tail_regret_rate_ge_5": best_metrics["tail_regret_rate_ge_5"],
            "final_mean_regret": final_metrics["mean_regret"],
            f"e_q_n_{args.eq_n}_mean_regret": eq_metrics["mean_regret"],
        }
    )
    wandb_status = wb.status()
    finished_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": "w42.multi_seed_larger_eval_replication.v0",
        "bead_id": "t42-csw6.31",
        "feature_set": feature,
        "created_at": finished_at,
        "repo_commit": sha,
        "git_status_before_artifacts": git_status_short(),
        "source_corpora": [
            {"path": path, "role": "train", "exists_at_run_time": Path(path).exists()}
            for path in args.train
        ]
        + [
            {"path": path, "role": "eval", "exists_at_run_time": Path(path).exists()}
            for path in args.eval
        ],
        "splits": {
            "train_rows": len(train_ds),
            "eval_rows": len(eval_ds),
            "base_train_rows": args.base_train_rows,
            "base_eval_rows": args.base_eval_rows,
            "random_seeds": {
                "data_generation": "not applicable; existing Gus corpora",
                "train": seed,
                "dataset_shuffle": args.split_seed,
                "eval_subset": args.eval_seed,
                "dataloader_shuffle": seed,
                "oracle_world_sampling": "first-N deterministic",
            },
        },
        "leakage_exclusions": [
            "train and eval use disjoint explicit corpus paths",
            "oracle E[Q] values are labels and metrics only, not model inputs",
            "features are public-state/action-local Gus detector outputs",
            "Burl traces and private table-talk text are not consumed",
        ],
        "generation": {
            "command": " ".join(sys.argv),
            "cwd": str(ROOT),
            "environment": {
                "device": device,
                "wandb": wandb_status,
                "huggingface": "not applicable",
            },
        },
        "claim_ledger_impact": "no claim-ledger change",
    }
    run = {
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": time.perf_counter() - t0,
        "config": asdict(config),
        "metrics": metrics,
        "history": history,
        "wandb": wandb_status,
        "huggingface": "not applicable",
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "run.json", run)
    write_json(out_dir / "metrics.json", metrics)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(config),
            "metrics": metrics,
            "history": history,
        },
        out_dir / "model.pt",
    )
    wb.log_artifact_files(
        name=f"w42-multi-seed-larger-eval-{feature}-s{seed:04d}-{sha[:8]}",
        artifact_type="w42-replication",
        paths=[out_dir / "manifest.json", out_dir / "run.json", out_dir / "metrics.json"],
    )
    wb.finish()
    return {
        "feature_set": feature,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_mean_regret": best_metrics["mean_regret"],
        "best_match_rate": best_metrics["match_rate"],
        "best_near_tie_rate_regret_lt_0_5": best_metrics["near_tie_rate_regret_lt_0_5"],
        "best_tail_regret_rate_ge_5": best_metrics["tail_regret_rate_ge_5"],
        "final_mean_regret": final_metrics["mean_regret"],
        "final_match_rate": final_metrics["match_rate"],
        "eval_n": best_metrics["n"],
        f"e_q_n_{args.eq_n}_mean_regret": eq_metrics["mean_regret"],
        "wandb_id": wandb_status.get("id") if isinstance(wandb_status, dict) else "",
        "wandb_url": wandb_status.get("url") if isinstance(wandb_status, dict) else "",
        "output_dir": str(out_dir),
    }


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def ci95_half_width(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    # Conservative enough for n=2..5 without adding scipy as a dependency.
    t_by_df = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    df = len(values) - 1
    t = t_by_df.get(df, 1.96)
    return t * sample_sd(values) / math.sqrt(len(values))


def aggregate(rows: list[dict[str, Any]], eq_key: str) -> dict[str, Any]:
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_feature.setdefault(str(row["feature_set"]), []).append(row)

    feature_summary: list[dict[str, Any]] = []
    for feature in FEATURES:
        vals = by_feature.get(feature, [])
        if not vals:
            continue
        regrets = [float(row["best_mean_regret"]) for row in vals]
        matches = [float(row["best_match_rate"]) for row in vals]
        tails = [float(row["best_tail_regret_rate_ge_5"]) for row in vals]
        feature_summary.append(
            {
                "feature_set": feature,
                "n_seeds": len(vals),
                "mean_best_regret": mean(regrets),
                "sd_best_regret": sample_sd(regrets),
                "ci95_half_width_best_regret": ci95_half_width(regrets),
                "mean_best_match_rate": mean(matches),
                "mean_best_tail_regret_rate_ge_5": mean(tails),
                eq_key: vals[0].get(eq_key, ""),
            }
        )

    seeds = sorted({int(row["seed"]) for row in rows})
    paired_rows: list[dict[str, Any]] = []
    for left, right in (("v0", "raw"), ("rich", "raw"), ("rich", "v0")):
        deltas: list[float] = []
        for seed in seeds:
            l_row = next(
                (row for row in rows if row["feature_set"] == left and int(row["seed"]) == seed),
                None,
            )
            r_row = next(
                (row for row in rows if row["feature_set"] == right and int(row["seed"]) == seed),
                None,
            )
            if l_row is None or r_row is None:
                continue
            deltas.append(float(l_row["best_mean_regret"]) - float(r_row["best_mean_regret"]))
        paired_rows.append(
            {
                "comparison": f"{left}_minus_{right}",
                "n_pairs": len(deltas),
                "mean_delta_best_regret": mean(deltas) if deltas else "",
                "sd_delta_best_regret": sample_sd(deltas) if deltas else "",
                "ci95_half_width_delta_best_regret": ci95_half_width(deltas) if deltas else "",
                "direction": "negative is better for left feature set",
            }
        )
    return {
        "feature_summary": feature_summary,
        "paired_deltas": paired_rows,
        "status": "pilot" if len(seeds) < 5 else "multi-seed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        nargs="+",
        default=[
            "gus/data/corpus_train_chunk_0-99.pt",
            "gus/data/corpus_train_chunk_100-199.pt",
        ],
    )
    parser.add_argument(
        "--eval",
        nargs="+",
        default=["gus/data/corpus_train_chunk_9000-9099.pt"],
    )
    parser.add_argument("--train-limit", type=int, default=5600)
    parser.add_argument("--eval-limit", type=int, default=2800)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action-hidden", type=int, default=96)
    parser.add_argument("--eq-n", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=43)
    parser.add_argument("--features", nargs="+", choices=FEATURES, default=list(FEATURES))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="w42/multi_seed_larger_eval_replication")
    parser.add_argument("--skip-existing", action="store_true")
    add_wandb_args(
        parser,
        default_project="w42",
        default_group="w42-csw6-31-multi-seed-larger-eval",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_tag_dims()
    sha = git_sha()
    device = args.device or _device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_t0 = time.perf_counter()
    base_train = JointWorldFullDataset(args.train, seed=args.seeds[0], include_strategy_features=True)
    base_eval = JointWorldFullDataset(args.eval, seed=args.seeds[0], include_strategy_features=True)
    train_ds = _subset(base_train, args.train_limit, args.split_seed)
    eval_ds = _subset(base_eval, args.eval_limit, args.eval_seed)
    eq_eval_ds = _subset(EQNWrapper(base_eval, args.eq_n), args.eval_limit, args.eval_seed)
    args.base_train_rows = len(base_train)
    args.base_eval_rows = len(base_eval)
    print(
        f"loaded train={len(train_ds)} eval={len(eval_ds)} "
        f"(base train={len(base_train)} eval={len(base_eval)}) "
        f"in {time.perf_counter() - load_t0:.1f}s",
        flush=True,
    )
    eq_loader = DataLoader(eq_eval_ds, batch_size=args.batch_size, shuffle=False)
    eq_metrics = evaluate_eq_n(eq_loader, device)
    print(f"eq_n_{args.eq_n}_mean_regret={eq_metrics['mean_regret']:.3f}", flush=True)

    started_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        for feature in args.features:
            metrics_path = Path(args.output_dir) / f"{feature}_s{seed:04d}" / "metrics.json"
            if args.skip_existing and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                run_json = json.loads(
                    (Path(args.output_dir) / f"{feature}_s{seed:04d}" / "run.json").read_text(
                        encoding="utf-8"
                    )
                )
                best = metrics["best"]
                final = metrics["final"]
                wandb = run_json.get("wandb", {})
                rows.append(
                    {
                        "feature_set": feature,
                        "seed": seed,
                        "best_epoch": metrics["best_epoch"],
                        "best_mean_regret": best["mean_regret"],
                        "best_match_rate": best["match_rate"],
                        "best_near_tie_rate_regret_lt_0_5": best[
                            "near_tie_rate_regret_lt_0_5"
                        ],
                        "best_tail_regret_rate_ge_5": best["tail_regret_rate_ge_5"],
                        "final_mean_regret": final["mean_regret"],
                        "final_match_rate": final["match_rate"],
                        "eval_n": best["n"],
                        f"e_q_n_{args.eq_n}_mean_regret": eq_metrics["mean_regret"],
                        "wandb_id": wandb.get("id", "") if isinstance(wandb, dict) else "",
                        "wandb_url": wandb.get("url", "") if isinstance(wandb, dict) else "",
                        "output_dir": str(Path(args.output_dir) / f"{feature}_s{seed:04d}"),
                    }
                )
                continue
            rows.append(
                train_one(
                    feature=feature,
                    seed=seed,
                    train_ds=train_ds,
                    eval_ds=eval_ds,
                    eq_metrics=eq_metrics,
                    args=args,
                    device=device,
                    sha=sha,
                )
            )

    eq_key = f"e_q_n_{args.eq_n}_mean_regret"
    summary = aggregate(rows, eq_key)
    finished_at = datetime.now(UTC).isoformat()
    row_fields = [
        "feature_set",
        "seed",
        "best_epoch",
        "best_mean_regret",
        "best_match_rate",
        "best_near_tie_rate_regret_lt_0_5",
        "best_tail_regret_rate_ge_5",
        "final_mean_regret",
        "final_match_rate",
        "eval_n",
        eq_key,
        "wandb_id",
        "wandb_url",
        "output_dir",
    ]
    write_csv(out_dir / "per_seed_metrics.csv", rows, row_fields)
    write_csv(
        out_dir / "feature_summary.csv",
        summary["feature_summary"],
        [
            "feature_set",
            "n_seeds",
            "mean_best_regret",
            "sd_best_regret",
            "ci95_half_width_best_regret",
            "mean_best_match_rate",
            "mean_best_tail_regret_rate_ge_5",
            eq_key,
        ],
    )
    write_csv(
        out_dir / "paired_deltas.csv",
        summary["paired_deltas"],
        [
            "comparison",
            "n_pairs",
            "mean_delta_best_regret",
            "sd_delta_best_regret",
            "ci95_half_width_delta_best_regret",
            "direction",
        ],
    )
    write_json(
        out_dir / "summary.json",
        {
            "schema_version": "w42.multi_seed_larger_eval_replication.summary.v0",
            "bead_id": "t42-csw6.31",
            "status": summary["status"],
            "started_at": started_at,
            "finished_at": finished_at,
            "git_sha": sha,
            "git_status_before_artifacts": git_status_short(),
            "command": " ".join(sys.argv),
            "config": {
                "train_paths": args.train,
                "eval_paths": args.eval,
                "train_limit": args.train_limit,
                "eval_limit": args.eval_limit,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seeds": args.seeds,
                "features": args.features,
                "split_seed": args.split_seed,
                "eval_seed": args.eval_seed,
                "device": device,
                "wandb_group": args.wandb_group,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_mode": args.wandb_mode,
            },
            "base_rows": {
                "train": len(base_train),
                "eval": len(base_eval),
                "selected_train": len(train_ds),
                "selected_eval": len(eval_ds),
            },
            "eq_metrics": eq_metrics,
            "feature_summary": summary["feature_summary"],
            "paired_deltas": summary["paired_deltas"],
            "wandb_urls": [row["wandb_url"] for row in rows if row.get("wandb_url")],
            "huggingface": "not applicable",
            "claim_ledger_impact": "no claim-ledger change",
            "interpretation": (
                "Pilot/replication evidence only. Do not promote book claims or final model "
                "conclusions from this artifact without the planned full run or additional replication."
            ),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
