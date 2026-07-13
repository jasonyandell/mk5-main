"""W42 raw public-state/action-input baseline.

This is intentionally separate from Gus training. It reuses the Gus corpus
adapter/tokenization shape, trains a tiny action scorer from public tokens,
voids, and action ids, and writes local-only provenance artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.student import TransformerEncoder, VoidsEncoder
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
    output_dir: str
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_group: str | None
    wandb_name: str | None
    wandb_mode: str


class RawPublicStateActionModel(nn.Module):
    """Tiny policy scorer over public state plus candidate action id."""

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
        self.score = nn.Sequential(
            nn.Linear(d_model * 2, action_hidden),
            nn.GELU(),
            nn.LayerNorm(action_hidden),
            nn.Linear(action_hidden, 1),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        h = self.encoder(batch["tokens"], batch["attention_mask"])
        state = self.norm(h[:, 0, :] + self.voids_encoder(batch["voids"]))
        action_ids = torch.arange(7, device=state.device)
        actions = self.action_emb(action_ids).unsqueeze(0).expand(state.shape[0], -1, -1)
        state7 = state.unsqueeze(1).expand(-1, 7, -1)
        logits = self.score(torch.cat([state7, actions], dim=-1)).squeeze(-1)
        return logits


class EQNWrapper(torch.utils.data.Dataset):
    """Adds the first N per-world Q rows for the cheap E[Q] baseline."""

    def __init__(self, base: JointWorldFullDataset, n_worlds: int):
        self.base = base
        self.n_worlds = n_worlds

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item = self.base[idx]
        g_idx, d_idx = self.base.index[idx]
        dec = self.base.games[g_idx].decisions[d_idx]
        qpw = dec.q_per_world.float()
        item["q_worlds_n"] = qpw[: min(self.n_worlds, qpw.shape[0])]
        return item


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _git_status_short() -> str:
    return subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True
    )


def _subset(ds: torch.utils.data.Dataset, limit: int, seed: int) -> torch.utils.data.Dataset:
    if limit <= 0 or limit >= len(ds):
        return ds
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    return Subset(ds, idxs[:limit])


def _oracle_best_action(batch: dict[str, Tensor]) -> Tensor:
    e_q = batch["e_q"].masked_fill(~batch["legal_mask"], float("-inf"))
    return e_q.argmax(dim=-1)


def _policy_loss(logits: Tensor, batch: dict[str, Tensor]) -> Tensor:
    masked = logits.masked_fill(~batch["legal_mask"], -1e9)
    return nn.functional.cross_entropy(masked, _oracle_best_action(batch))


@torch.no_grad()
def evaluate_model(
    model: RawPublicStateActionModel,
    loader: DataLoader,
    device: str,
    prediction_sample_limit: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_regret = 0.0
    total = 0
    hits = 0
    near = 0
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
        total_regret += float(regret.sum().item())
        total += int(action.numel())
        hits += int((action == oracle_action).sum().item())
        near += int((regret < 0.5).sum().item())
        if prediction_sample_limit and len(predictions) < prediction_sample_limit:
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
                        "legal_mask": [bool(x) for x in batch["legal_mask"][i].detach().cpu().tolist()],
                        "e_q": [float(x) for x in batch["e_q"][i].detach().cpu().tolist()],
                        "logits": [float(x) for x in logits[i].detach().cpu().tolist()],
                    }
                )
        offset += int(action.numel())
    metrics = {
        "mean_regret": total_regret / max(total, 1),
        "match_rate": hits / max(total, 1),
        "near_tie_rate_regret_lt_0_5": near / max(total, 1),
        "n": float(total),
    }
    return metrics, predictions


@torch.no_grad()
def evaluate_oracle_best(loader: DataLoader, device: str) -> dict[str, float]:
    total = 0
    for batch in loader:
        total += int(batch["action_taken"].numel())
    return {
        "mean_regret": 0.0,
        "match_rate": 1.0,
        "near_tie_rate_regret_lt_0_5": 1.0,
        "n": float(total),
    }


@torch.no_grad()
def evaluate_eq_n(loader: DataLoader, device: str) -> dict[str, float]:
    total_regret = 0.0
    total = 0
    hits = 0
    near = 0
    for batch in loader:
        qn = batch["q_worlds_n"].to(device)
        batch = {k: v.to(device) for k, v in batch.items() if k != "q_worlds_n"}
        q_est = qn.mean(dim=1).masked_fill(~batch["legal_mask"], float("-inf"))
        action = q_est.argmax(dim=-1)
        e_q_legal = batch["e_q"].masked_fill(~batch["legal_mask"], float("-inf"))
        oracle_best = e_q_legal.max(dim=-1).values
        oracle_action = e_q_legal.argmax(dim=-1)
        idx = torch.arange(action.numel(), device=device)
        regret = oracle_best - batch["e_q"][idx, action]
        total_regret += float(regret.sum().item())
        total += int(action.numel())
        hits += int((action == oracle_action).sum().item())
        near += int((regret < 0.5).sum().item())
    return {
        "mean_regret": total_regret / max(total, 1),
        "match_rate": hits / max(total, 1),
        "near_tie_rate_regret_lt_0_5": near / max(total, 1),
        "n": float(total),
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


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
    parser.add_argument("--output-dir", default="w42/raw_public_state_baseline")
    parser.add_argument("--prediction-sample-limit", type=int, default=32)
    add_wandb_args(parser, default_project="w42", default_group="t42-csw6")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or _device()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = RunConfig(
        bead_id="t42-csw6.10",
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
        output_dir=str(out_dir),
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
        tags=["w42", "raw-public-state", "baseline", "t42-csw6.10"],
    )

    started_at = datetime.now(UTC).isoformat()
    run_t0 = time.perf_counter()
    print(f"device={device}", flush=True)
    print(f"train={args.train}", flush=True)
    print(f"eval={args.eval}", flush=True)

    load_t0 = time.perf_counter()
    train_base = JointWorldFullDataset(args.train, seed=args.seed, include_strategy_features=False)
    eval_base = JointWorldFullDataset(args.eval, seed=args.seed, include_strategy_features=False)
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

    model = RawPublicStateActionModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        action_hidden=args.action_hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_metrics: dict[str, float] | None = None
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
        eval_metrics, _ = evaluate_model(model, eval_loader, device)
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
            best_epoch = epoch + 1
        print(
            f"epoch {epoch+1:02d}/{args.epochs} "
            f"loss={row['train_loss']:.3f} eval_regret={eval_metrics['mean_regret']:.3f} "
            f"match={eval_metrics['match_rate']:.2%} near={eval_metrics['near_tie_rate_regret_lt_0_5']:.2%}",
            flush=True,
        )

    assert best_metrics is not None
    final_metrics, predictions = evaluate_model(
        model, eval_loader, device, prediction_sample_limit=args.prediction_sample_limit
    )
    eq_metrics = evaluate_eq_n(eq_loader, device)
    oracle_metrics = evaluate_oracle_best(eval_loader, device)
    finished_at = datetime.now(UTC).isoformat()

    metrics = {
        "raw_public_state_action_model_final": final_metrics,
        "raw_public_state_action_model_best": {"epoch": best_epoch, **best_metrics},
        f"e_q_n_{args.eq_n}": eq_metrics,
        "oracle_best_e_q_ceiling": oracle_metrics,
        "deltas": {
            f"model_final_minus_e_q_n_{args.eq_n}_mean_regret": final_metrics["mean_regret"] - eq_metrics["mean_regret"],
            "model_final_minus_oracle_ceiling_mean_regret": final_metrics["mean_regret"],
        },
    }
    wb.log_metric_groups(
        {
            "final/model": final_metrics,
            f"final/e_q_n_{args.eq_n}": eq_metrics,
            "final/oracle": oracle_metrics,
            "best/model": {"epoch": float(best_epoch), **best_metrics},
        },
        step=args.epochs,
    )
    wb.update_summary(
        {
            "best_epoch": best_epoch,
            "best_mean_regret": best_metrics["mean_regret"],
            "final_mean_regret": final_metrics["mean_regret"],
            f"e_q_n_{args.eq_n}_mean_regret": eq_metrics["mean_regret"],
            "final_minus_eq_n_mean_regret": metrics["deltas"][
                f"model_final_minus_e_q_n_{args.eq_n}_mean_regret"
            ],
        }
    )
    wandb_status = wb.status()
    manifest = {
        "schema_version": "w42.raw_public_state_baseline.v1",
        "bead_id": "t42-csw6.10",
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
            "strategy_features and strategy_action_features are not loaded",
        ],
        "labels_available": ["e_q", "legal_mask", "oracle_best_action derived from e_q"],
        "features_used": ["tokens", "attention_mask", "voids", "candidate_action_id"],
        "tags_available": {
            "cheap_strategy_tags": "not used",
            "chapter_derived_tags": "not used",
            "analysis_buckets": "not used",
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
    _write_jsonl(out_dir / "predictions_sample.jsonl", predictions)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": metrics,
            "history": history,
        },
        out_dir / "model.pt",
    )
    wb.log_artifact_files(
        name=f"w42-raw-public-state-{_git_sha()[:8]}",
        artifact_type="w42-baseline",
        paths=[
            out_dir / "manifest.json",
            out_dir / "run.json",
            out_dir / "metrics.json",
            out_dir / "predictions_sample.jsonl",
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
