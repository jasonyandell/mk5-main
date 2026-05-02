"""W42 raw-plus-v0-strategy-tags baseline.

This script is w42-owned scratch code. It keeps the raw public-state baseline's
split, core encoder shape, and metrics, then adds the v0 global/action-local
strategy tags as public-state inputs. It also reloads the raw baseline checkpoint
for paired aggregate and tag-group slice comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_W42 = ROOT / "scratch" / "w42"
for entry in (str(SCRATCH_W42), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.student import TransformerEncoder, VoidsEncoder
from raw_public_state_baseline import (
    EQNWrapper,
    RawPublicStateActionModel,
    _device,
    _git_sha,
    _git_status_short,
    _subset,
    evaluate_eq_n,
    evaluate_oracle_best,
)
from strategy_tags_v0 import ACTION_TAGS, GLOBAL_TAGS, group_slices, validate_tag_dims
from wandb_utils import add_wandb_args, init_wandb


@dataclass
class RunConfig:
    bead_id: str
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
    raw_metrics_path: str
    raw_checkpoint: str
    output_dir: str
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_group: str | None
    wandb_name: str | None
    wandb_mode: str


class RawPlusV0StrategyTagsModel(nn.Module):
    """Tiny policy scorer over public state, candidate actions, and v0 tags."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        ff_dim: int = 128,
        dropout: float = 0.1,
        action_hidden: int = 96,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.voids_encoder = VoidsEncoder(d_model, hidden_dim=64)
        self.action_emb = nn.Embedding(7, d_model)
        self.global_tag_proj = nn.Sequential(
            nn.Linear(len(GLOBAL_TAGS), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.action_tag_proj = nn.Sequential(
            nn.Linear(len(ACTION_TAGS), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.score = nn.Sequential(
            nn.Linear(d_model * 2, action_hidden),
            nn.GELU(),
            nn.LayerNorm(action_hidden),
            nn.Linear(action_hidden, 1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        h = self.encoder(batch["tokens"], batch["attention_mask"])
        state = self.norm(
            h[:, 0, :]
            + self.voids_encoder(batch["voids"])
            + self.global_tag_proj(batch["strategy_features"].float())
        )
        action_ids = torch.arange(7, device=state.device)
        actions = self.action_emb(action_ids).unsqueeze(0).expand(state.shape[0], -1, -1)
        actions = actions + self.action_tag_proj(batch["strategy_action_features"].float())
        state7 = state.unsqueeze(1).expand(-1, 7, -1)
        logits = self.score(torch.cat([state7, actions], dim=-1)).squeeze(-1)
        return logits


def _oracle_best_action(batch: dict[str, Tensor]) -> Tensor:
    e_q = batch["e_q"].masked_fill(~batch["legal_mask"], float("-inf"))
    return e_q.argmax(dim=-1)


def _policy_loss(logits: Tensor, batch: dict[str, Tensor]) -> Tensor:
    masked = logits.masked_fill(~batch["legal_mask"], -1e9)
    return nn.functional.cross_entropy(masked, _oracle_best_action(batch))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _load_raw_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"raw metrics not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_model(path: Path, config: RunConfig, device: str) -> RawPublicStateActionModel:
    if not path.exists():
        raise FileNotFoundError(f"raw checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    model = RawPublicStateActionModel(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        action_hidden=config.action_hidden,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _metric_row(regret: Tensor, action: Tensor, oracle_action: Tensor) -> dict[str, float]:
    total = int(action.numel())
    return {
        "mean_regret": float(regret.sum().item()) / max(total, 1),
        "match_rate": float((action == oracle_action).float().mean().item()) if total else 0.0,
        "near_tie_rate_regret_lt_0_5": float((regret < 0.5).float().mean().item()) if total else 0.0,
        "n": float(total),
    }


def _empty_acc() -> dict[str, float]:
    return {"regret_sum": 0.0, "hits": 0.0, "near": 0.0, "n": 0.0}


def _add_acc(acc: dict[str, float], regret: Tensor, action: Tensor, oracle_action: Tensor, mask: Tensor) -> None:
    if int(mask.sum().item()) == 0:
        return
    r = regret[mask]
    a = action[mask]
    o = oracle_action[mask]
    acc["regret_sum"] += float(r.sum().item())
    acc["hits"] += float((a == o).sum().item())
    acc["near"] += float((r < 0.5).sum().item())
    acc["n"] += float(r.numel())


def _finish_acc(acc: dict[str, float]) -> dict[str, float]:
    n = max(acc["n"], 1.0)
    return {
        "mean_regret": acc["regret_sum"] / n,
        "match_rate": acc["hits"] / n,
        "near_tie_rate_regret_lt_0_5": acc["near"] / n,
        "n": acc["n"],
    }


def _active_group_masks(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    masks: dict[str, Tensor] = {}
    global_values = batch["strategy_features"].float()
    action_values = batch["strategy_action_features"].float()
    legal = batch["legal_mask"].unsqueeze(-1)

    for group, idxs in group_slices(GLOBAL_TAGS).items():
        masks[f"global:{group}"] = (global_values[:, idxs].abs() > 1e-9).any(dim=1)
    masked_actions = action_values.masked_fill(~legal, 0.0)
    for group, idxs in group_slices(ACTION_TAGS).items():
        masks[f"action:{group}"] = (masked_actions[:, :, idxs].abs() > 1e-9).any(dim=(1, 2))
    return masks


@torch.no_grad()
def evaluate_model_with_buckets(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    prediction_sample_limit: int = 0,
) -> tuple[dict[str, float], dict[str, dict[str, float]], list[dict[str, Any]]]:
    model.eval()
    all_acc = _empty_acc()
    bucket_accs: dict[str, dict[str, float]] = {}
    predictions: list[dict[str, Any]] = []
    offset = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch).masked_fill(~batch["legal_mask"], -1e9)
        action = logits.argmax(dim=-1)
        e_q_legal = batch["e_q"].masked_fill(~batch["legal_mask"], float("-inf"))
        oracle_best = e_q_legal.max(dim=-1).values
        oracle_action = e_q_legal.argmax(dim=-1)
        idx = torch.arange(action.numel(), device=device)
        regret = oracle_best - batch["e_q"][idx, action]

        all_mask = torch.ones(action.shape[0], dtype=torch.bool, device=device)
        _add_acc(all_acc, regret, action, oracle_action, all_mask)
        active_masks = _active_group_masks(batch)
        for bucket, mask in active_masks.items():
            bucket_acc = bucket_accs.setdefault(bucket, _empty_acc())
            _add_acc(bucket_acc, regret, action, oracle_action, mask)

        if prediction_sample_limit and len(predictions) < prediction_sample_limit:
            active_names = {
                name: mask.detach().cpu().tolist()
                for name, mask in active_masks.items()
                if bool(mask.any().item())
            }
            for i in range(action.numel()):
                if len(predictions) >= prediction_sample_limit:
                    break
                predictions.append(
                    {
                        "eval_row": offset + i,
                        "decision_idx": int(batch["decision_idx"][i].item()),
                        "player": int(batch["player"][i].item()),
                        "chosen_action": int(action[i].item()),
                        "oracle_best_action": int(oracle_action[i].item()),
                        "regret": float(regret[i].item()),
                        "active_tag_groups": [
                            name for name, values in active_names.items() if values[i]
                        ],
                        "legal_mask": [
                            bool(x) for x in batch["legal_mask"][i].detach().cpu().tolist()
                        ],
                        "e_q": [float(x) for x in batch["e_q"][i].detach().cpu().tolist()],
                        "logits": [float(x) for x in logits[i].detach().cpu().tolist()],
                    }
                )
        offset += int(action.numel())

    bucket_metrics = {
        bucket: _finish_acc(acc)
        for bucket, acc in sorted(bucket_accs.items())
        if acc["n"] > 0
    }
    return _finish_acc(all_acc), bucket_metrics, predictions


def _comparison_metrics(
    raw_final: dict[str, float],
    raw_best: dict[str, Any],
    tagged_final: dict[str, float],
    tagged_best: dict[str, Any],
    eq_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "tagged_best_minus_raw_best_mean_regret": tagged_best["mean_regret"] - raw_best["mean_regret"],
        "tagged_best_minus_raw_best_match_rate": tagged_best["match_rate"] - raw_best["match_rate"],
        "tagged_best_minus_raw_best_near_tie": tagged_best["near_tie_rate_regret_lt_0_5"]
        - raw_best["near_tie_rate_regret_lt_0_5"],
        "tagged_final_minus_raw_final_checkpoint_mean_regret": tagged_final["mean_regret"]
        - raw_final["mean_regret"],
        "tagged_best_minus_e_q_n_mean_regret": tagged_best["mean_regret"] - eq_metrics["mean_regret"],
    }


def _bucket_comparison(
    raw_buckets: dict[str, dict[str, float]],
    tagged_buckets: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for bucket in sorted(set(raw_buckets) & set(tagged_buckets)):
        raw = raw_buckets[bucket]
        tagged = tagged_buckets[bucket]
        out[bucket] = {
            "raw_mean_regret": raw["mean_regret"],
            "tagged_mean_regret": tagged["mean_regret"],
            "delta_mean_regret": tagged["mean_regret"] - raw["mean_regret"],
            "raw_match_rate": raw["match_rate"],
            "tagged_match_rate": tagged["match_rate"],
            "delta_match_rate": tagged["match_rate"] - raw["match_rate"],
            "raw_near_tie": raw["near_tie_rate_regret_lt_0_5"],
            "tagged_near_tie": tagged["near_tie_rate_regret_lt_0_5"],
            "delta_near_tie": tagged["near_tie_rate_regret_lt_0_5"]
            - raw["near_tie_rate_regret_lt_0_5"],
            "n": tagged["n"],
        }
    return out


def _write_bucket_csv(path: Path, bucket_comparison: dict[str, dict[str, float]]) -> None:
    fieldnames = [
        "bucket",
        "n",
        "raw_mean_regret",
        "tagged_mean_regret",
        "delta_mean_regret",
        "raw_match_rate",
        "tagged_match_rate",
        "delta_match_rate",
        "raw_near_tie",
        "tagged_near_tie",
        "delta_near_tie",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for bucket, row in bucket_comparison.items():
            writer.writerow({"bucket": bucket, **row})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", default=["gus/data/corpus_train_100.pt"])
    parser.add_argument("--eval", nargs="+", default=["gus/data/corpus_eval_20.pt"])
    parser.add_argument("--train-limit", type=int, default=2800)
    parser.add_argument("--eval-limit", type=int, default=560)
    parser.add_argument("--epochs", type=int, default=8)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=43)
    parser.add_argument("--device", default=None)
    parser.add_argument("--raw-metrics-path", default="scratch/w42/raw_public_state_baseline/metrics.json")
    parser.add_argument("--raw-checkpoint", default="scratch/w42/raw_public_state_baseline/model.pt")
    parser.add_argument("--output-dir", default="scratch/w42/v0_strategy_tags_baseline")
    parser.add_argument("--prediction-sample-limit", type=int, default=32)
    add_wandb_args(parser, default_project="w42", default_group="t42-csw6")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_tag_dims()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or _device()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config = RunConfig(
        bead_id="t42-csw6.11",
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
        seed=args.seed,
        split_seed=args.split_seed,
        eval_seed=args.eval_seed,
        device=device,
        raw_metrics_path=args.raw_metrics_path,
        raw_checkpoint=args.raw_checkpoint,
        output_dir=args.output_dir,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        wandb_mode=args.wandb_mode,
    )
    wb = init_wandb(
        args,
        config=asdict(config),
        output_dir=out_dir,
        tags=["w42", "v0-strategy-tags", "baseline", "t42-csw6.11"],
    )

    started_at = datetime.now(UTC).isoformat()
    run_t0 = time.perf_counter()
    print(f"device={device}", flush=True)
    print(f"train={args.train}", flush=True)
    print(f"eval={args.eval}", flush=True)

    raw_metrics_file = ROOT / args.raw_metrics_path
    raw_checkpoint_file = ROOT / args.raw_checkpoint
    raw_prior_metrics = _load_raw_metrics(raw_metrics_file)

    load_t0 = time.perf_counter()
    train_base = JointWorldFullDataset(args.train, seed=args.seed, include_strategy_features=True)
    eval_base = JointWorldFullDataset(args.eval, seed=args.seed, include_strategy_features=True)
    train_ds = _subset(train_base, args.train_limit, args.split_seed)
    eval_ds = _subset(eval_base, args.eval_limit, args.eval_seed)
    eq_eval_ds = _subset(EQNWrapper(eval_base, args.eq_n), args.eval_limit, args.eval_seed)
    print(
        f"loaded train={len(train_ds)} eval={len(eval_ds)} "
        f"(base train={len(train_base)} eval={len(eval_base)}) in {time.perf_counter()-load_t0:.1f}s",
        flush=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    eq_loader = DataLoader(eq_eval_ds, batch_size=args.batch_size, shuffle=False)

    model = RawPlusV0StrategyTagsModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        action_hidden=args.action_hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        train_t0 = time.perf_counter()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = _policy_loss(model(batch), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            batches += 1
        eval_metrics, _, _ = evaluate_model_with_buckets(model, eval_loader, device)
        row = {
            "epoch": float(epoch + 1),
            "train_loss": loss_sum / max(batches, 1),
            "epoch_seconds": time.perf_counter() - train_t0,
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
                "eval/n": eval_metrics["n"],
            },
            step=epoch + 1,
        )
        if best_metrics is None or eval_metrics["mean_regret"] < best_metrics["mean_regret"]:
            best_metrics = eval_metrics
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
        print(
            f"epoch {epoch+1:02d}/{args.epochs} "
            f"loss={row['train_loss']:.3f} eval_regret={eval_metrics['mean_regret']:.3f} "
            f"match={eval_metrics['match_rate']:.2%} near={eval_metrics['near_tie_rate_regret_lt_0_5']:.2%}",
            flush=True,
        )

    assert best_metrics is not None
    assert best_state is not None
    final_metrics, tagged_final_buckets, final_predictions = evaluate_model_with_buckets(
        model, eval_loader, device, prediction_sample_limit=args.prediction_sample_limit
    )

    final_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    tagged_best_metrics, tagged_best_buckets, best_predictions = evaluate_model_with_buckets(
        model, eval_loader, device, prediction_sample_limit=args.prediction_sample_limit
    )
    model.load_state_dict(final_state)

    raw_model = _load_raw_model(raw_checkpoint_file, config, device)
    raw_final_metrics, raw_final_buckets, raw_predictions = evaluate_model_with_buckets(
        raw_model, eval_loader, device, prediction_sample_limit=args.prediction_sample_limit
    )
    eq_metrics = evaluate_eq_n(eq_loader, device)
    oracle_metrics = evaluate_oracle_best(eval_loader, device)
    finished_at = datetime.now(UTC).isoformat()

    raw_best_prior = raw_prior_metrics["raw_public_state_action_model_best"]
    bucket_comparison = _bucket_comparison(raw_final_buckets, tagged_final_buckets)
    best_bucket_comparison = _bucket_comparison(raw_final_buckets, tagged_best_buckets)
    metrics = {
        "raw_public_state_action_model_final_checkpoint": raw_final_metrics,
        "raw_public_state_action_model_best_prior_run": raw_best_prior,
        "raw_public_state_action_model_final_prior_run": raw_prior_metrics[
            "raw_public_state_action_model_final"
        ],
        "raw_plus_v0_strategy_tags_model_final": final_metrics,
        "raw_plus_v0_strategy_tags_model_best": {"epoch": best_epoch, **tagged_best_metrics},
        f"e_q_n_{args.eq_n}": eq_metrics,
        "oracle_best_e_q_ceiling": oracle_metrics,
        "deltas": _comparison_metrics(
            raw_final_metrics,
            raw_best_prior,
            final_metrics,
            {"epoch": best_epoch, **tagged_best_metrics},
            eq_metrics,
        ),
        "bucket_comparison_raw_final_vs_tagged_final": bucket_comparison,
        "bucket_comparison_raw_final_vs_tagged_best": best_bucket_comparison,
    }
    deltas = metrics["deltas"]
    wb.log_metric_groups(
        {
            "final/tagged": final_metrics,
            "best/tagged": {"epoch": float(best_epoch), **tagged_best_metrics},
            "final/raw_checkpoint": raw_final_metrics,
            "prior/raw_best": raw_best_prior,
            f"final/e_q_n_{args.eq_n}": eq_metrics,
            "final/oracle": oracle_metrics,
            "delta": deltas,
        },
        step=args.epochs,
    )
    top_bucket_rows = sorted(
        best_bucket_comparison.items(),
        key=lambda item: item[1]["delta_mean_regret"],
    )[:10]
    wb.log(
        {
            f"bucket_best/{bucket}/delta_mean_regret": row["delta_mean_regret"]
            for bucket, row in top_bucket_rows
        },
        step=args.epochs,
    )
    wb.update_summary(
        {
            "best_epoch": best_epoch,
            "best_mean_regret": tagged_best_metrics["mean_regret"],
            "best_match_rate": tagged_best_metrics["match_rate"],
            "raw_best_mean_regret": raw_best_prior["mean_regret"],
            "tagged_best_minus_raw_best_mean_regret": deltas[
                "tagged_best_minus_raw_best_mean_regret"
            ],
            f"e_q_n_{args.eq_n}_mean_regret": eq_metrics["mean_regret"],
        }
    )
    wandb_status = wb.status()

    manifest = {
        "schema_version": "w42.v0_strategy_tags_baseline.v1",
        "bead_id": "t42-csw6.11",
        "created_at": finished_at,
        "repo_commit": _git_sha(),
        "git_status_before_artifacts": _git_status_short(),
        "source_corpora": [
            {
                "path": path,
                "role": "train",
                "exists_at_run_time": Path(path).exists(),
                "source_kind": "gus-joint-world",
                "split_policy": "w42-seed-bucket-v1 train corpus",
            }
            for path in args.train
        ]
        + [
            {
                "path": path,
                "role": "eval-only",
                "exists_at_run_time": Path(path).exists(),
                "source_kind": "gus-joint-world",
                "split_policy": "w42-seed-bucket-v1 eval-only seeds",
            }
            for path in args.eval
        ],
        "raw_baseline_artifacts": {
            "metrics_path": args.raw_metrics_path,
            "checkpoint_path": args.raw_checkpoint,
            "prior_run_metrics_loaded": True,
            "final_checkpoint_re_evaluated_on_same_eval_loader": True,
        },
        "generation": {
            "command": " ".join(sys.argv),
            "cwd": str(ROOT),
            "environment": {
                "device": device,
                "wandb": wandb_status,
                "huggingface": "not applicable",
            },
        },
        "splits": {
            "policy": "w42-seed-bucket-v1 via existing Gus train/eval corpora",
            "train_rows": len(train_ds),
            "eval_rows": len(eval_ds),
            "base_train_rows": len(train_base),
            "base_eval_rows": len(eval_base),
            "train_game_seeds": train_base.seeds,
            "eval_game_seeds": eval_base.seeds,
            "random_seeds": {
                "data_generation": "not applicable",
                "dataset_shuffle": args.split_seed,
                "train": args.seed,
                "eval": args.eval_seed,
                "oracle_world_sampling": "first-N deterministic",
            },
        },
        "leakage_exclusions": [
            "eval corpus uses 900000-900019 seeds and is eval-only",
            "oracle E[Q] values are labels/metrics only, not model inputs",
            "v0 strategy tags are public-state/action-local Gus detector outputs",
            "Burl traces and private table-talk text are not consumed",
        ],
        "labels_available": ["e_q", "legal_mask", "oracle_best_action derived from e_q"],
        "features_used": [
            "tokens",
            "attention_mask",
            "voids",
            "candidate_action_id",
            "strategy_features",
            "strategy_action_features",
        ],
        "tags_available": {
            "cheap_strategy_tags": [tag.name for tag in GLOBAL_TAGS],
            "action_local_strategy_tags": [tag.name for tag in ACTION_TAGS],
            "chapter_derived_tags": "not used",
            "analysis_buckets": sorted(bucket_comparison),
        },
        "claim_ledger_impact": "no claim-ledger change",
    }
    run = {
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": time.perf_counter() - run_t0,
        "config": asdict(config),
        "metrics": metrics,
        "history": history,
        "wandb": wandb_status,
        "huggingface": "not applicable",
    }

    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "run.json", run)
    _write_json(out_dir / "metrics.json", metrics)
    _write_jsonl(out_dir / "predictions_sample_tagged_best.jsonl", best_predictions)
    _write_jsonl(out_dir / "predictions_sample_tagged_final.jsonl", final_predictions)
    _write_jsonl(out_dir / "predictions_sample_raw_final.jsonl", raw_predictions)
    _write_bucket_csv(out_dir / "bucket_metrics.csv", bucket_comparison)
    _write_bucket_csv(out_dir / "bucket_metrics_best.csv", best_bucket_comparison)
    torch.save(
        {
            "model_state_dict": best_state,
            "final_model_state_dict": final_state,
            "config": asdict(config),
            "metrics": metrics,
            "history": history,
        },
        out_dir / "model.pt",
    )
    wb.log_artifact_files(
        name=f"w42-v0-strategy-tags-{_git_sha()[:8]}",
        artifact_type="w42-baseline",
        paths=[
            out_dir / "manifest.json",
            out_dir / "run.json",
            out_dir / "metrics.json",
            out_dir / "bucket_metrics.csv",
            out_dir / "bucket_metrics_best.csv",
            out_dir / "predictions_sample_tagged_best.jsonl",
            out_dir / "predictions_sample_tagged_final.jsonl",
            out_dir / "predictions_sample_raw_final.jsonl",
            out_dir / "model.pt",
        ],
    )
    wb.finish()

    print("\n=== Summary ===", flush=True)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"artifacts={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
