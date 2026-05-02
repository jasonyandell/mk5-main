"""W42 strategy tag family ablations.

This is w42-owned scratch code. It reuses the rich-tag many-signal probe's
public-state feature construction and trains paired small models with selected
rich concept families zeroed. The comparison is a cheap family-drop probe, not a
claim-specific detector verdict.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_W42 = ROOT / "scratch" / "w42"
for entry in (str(SCRATCH_W42), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.student import TransformerEncoder, VoidsEncoder
from raw_public_state_baseline import (
    EQNWrapper,
    _device,
    _git_sha,
    _git_status_short,
    _subset,
    evaluate_eq_n,
    evaluate_oracle_best,
)
from rich_tag_many_signal_probe import (
    ACTION_TAGS,
    GLOBAL_TAGS,
    RICH_ACTION_SIGNAL_NAMES,
    RICH_GLOBAL_SIGNAL_NAMES,
    _bucket_comparison,
    _load_raw_metrics,
    _load_raw_model,
    _policy_loss,
    _rich_action_features,
    _rich_global_features,
    _write_json,
    evaluate_model_with_buckets,
)
from strategy_tags_v0 import validate_tag_dims
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
    v0_metrics_path: str
    rich_metrics_path: str
    output_dir: str
    ablation_mode: str
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_group: str | None
    wandb_name: str | None
    wandb_mode: str


FAMILY_DROPS: dict[str, dict[str, list[str]]] = {
    "bidding_risk": {
        "global": ["off_risk_proxy", "bidding_risk_proxy", "count_budget_proxy"],
        "action": ["candidate_bidding_risk_proxy"],
    },
    "count_donation": {
        "global": [
            "legal_donation_window_any",
            "legal_count_pressure_any",
            "legal_count_point_frac",
            "partner_count_donation_any",
            "opponent_count_donation_any",
            "count_pressure_x_opponent_winning",
            "donation_x_partner_winning",
            "lead_count_risk",
            "last_to_play_point_pressure",
            "donation_proxy",
            "count_budget_proxy",
            "legal_count_per_legal_action",
        ],
        "action": [
            "candidate_count_points_frac",
            "candidate_is_ten_count",
            "candidate_point_dump",
            "candidate_live_count_frac",
            "candidate_live_higher_count_frac",
            "candidate_pip_count_risk",
            "candidate_count_donation_to_partner",
            "candidate_count_donation_to_opponent",
            "candidate_count_pressure",
            "candidate_donation_window",
            "candidate_count_x_beats",
            "candidate_count_x_partner_winning",
            "candidate_count_x_opponent_winning",
            "candidate_ten_x_last_to_play",
            "candidate_donation_x_partner_winning",
            "candidate_dump_x_opponent_winning",
            "candidate_safe_donation_proxy",
            "candidate_unsafe_donation_proxy",
            "global_current_trick_count_context",
        ],
    },
    "trump_pressure": {
        "global": [
            "hand_trump_frac",
            "legal_trump_frac",
            "current_trick_has_trump",
            "legal_suit_pressure_mean",
            "legal_trump_action_any",
            "suit_pressure_x_must_follow",
            "trump_pressure_proxy",
        ],
        "action": [
            "candidate_trump",
            "candidate_live_suit_frac",
            "candidate_live_higher_suit_frac",
            "candidate_suit_pressure",
            "candidate_trump_x_live_higher",
            "candidate_trump_pressure_proxy",
        ],
    },
    "off_protection": {
        "global": [
            "hand_off_count_point_frac",
            "legal_off_non_double_any",
            "legal_double_protection_any",
            "double_protection_x_off_count",
            "off_risk_proxy",
        ],
        "action": [
            "candidate_off_non_double",
            "candidate_protected_by_my_double",
            "candidate_protected_by_high_double",
            "candidate_double_protection",
            "candidate_off_x_double_protected",
            "candidate_off_protection_proxy",
        ],
    },
    "pounce_window": {
        "global": [
            "opponent_currently_winning",
            "count_pressure_x_opponent_winning",
            "pounce_proxy",
        ],
        "action": [
            "candidate_count_donation_to_opponent",
            "candidate_count_x_opponent_winning",
            "candidate_dump_x_opponent_winning",
            "candidate_pounce_window_proxy",
            "candidate_unsafe_donation_proxy",
        ],
    },
    "eighty_four_preservation": {
        "global": ["late_hand_proxy", "is_last_to_play"],
        "action": ["candidate_double", "candidate_84_preservation_proxy", "global_late_hand_context"],
    },
    "walker_endgame": {
        "global": ["late_hand_proxy", "is_last_to_play"],
        "action": [
            "candidate_beats_current",
            "candidate_walker_proxy",
            "candidate_trump_x_live_higher",
            "global_late_hand_context",
        ],
    },
    "no_trump_doubles": {
        "global": ["current_trick_has_trump"],
        "action": ["candidate_double", "candidate_no_trump_double_proxy"],
    },
    "scoring_pressure": {
        "global": [
            "current_trick_count_point_frac",
            "legal_count_point_frac",
            "unknown_count_point_frac",
            "count_budget_proxy",
        ],
        "action": [
            "candidate_count_points_frac",
            "candidate_scoring_pressure_proxy",
            "global_current_trick_count_context",
        ],
    },
}


def _mask(names: list[str], dropped: list[str]) -> Tensor:
    values = torch.ones(len(names), dtype=torch.float32)
    missing = sorted(set(dropped) - set(names))
    if missing:
        raise KeyError(f"unknown rich signals: {missing}")
    for name in dropped:
        values[names.index(name)] = 0.0
    return values


class FamilyAblatedRichTagsModel(nn.Module):
    """Tiny policy scorer with configurable rich-family masks."""

    def __init__(
        self,
        *,
        drop_global: list[str],
        drop_action: list[str],
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
        self.rich_global_proj = nn.Sequential(
            nn.Linear(len(RICH_GLOBAL_SIGNAL_NAMES), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.action_tag_proj = nn.Sequential(
            nn.Linear(len(ACTION_TAGS), d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.rich_action_proj = nn.Sequential(
            nn.Linear(len(RICH_ACTION_SIGNAL_NAMES), d_model),
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
        self.register_buffer("rich_global_mask", _mask(RICH_GLOBAL_SIGNAL_NAMES, drop_global))
        self.register_buffer("rich_action_mask", _mask(RICH_ACTION_SIGNAL_NAMES, drop_action))

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        h = self.encoder(batch["tokens"], batch["attention_mask"])
        rich_global = _rich_global_features(batch) * self.rich_global_mask
        state = self.norm(
            h[:, 0, :]
            + self.voids_encoder(batch["voids"])
            + self.global_tag_proj(batch["strategy_features"].float())
            + self.rich_global_proj(rich_global)
        )
        action_ids = torch.arange(7, device=state.device)
        actions = self.action_emb(action_ids).unsqueeze(0).expand(state.shape[0], -1, -1)
        actions = actions + self.action_tag_proj(batch["strategy_action_features"].float())
        rich_action = _rich_action_features(batch) * self.rich_action_mask
        actions = actions + self.rich_action_proj(rich_action)
        state7 = state.unsqueeze(1).expand(-1, 7, -1)
        return self.score(torch.cat([state7, actions], dim=-1)).squeeze(-1)


def _variant_specs(mode: str) -> list[dict[str, Any]]:
    specs = [{"variant": "full_rich_retrain", "drop_family": "none", "drop_global": [], "drop_action": []}]
    if mode in {"all", "with-rich-disabled"}:
        specs.append(
            {
                "variant": "rich_disabled_retrain",
                "drop_family": "all_rich_signals",
                "drop_global": list(RICH_GLOBAL_SIGNAL_NAMES),
                "drop_action": list(RICH_ACTION_SIGNAL_NAMES),
            }
        )
    for family, drops in FAMILY_DROPS.items():
        specs.append(
            {
                "variant": f"drop_{family}",
                "drop_family": family,
                "drop_global": drops["global"],
                "drop_action": drops["action"],
            }
        )
    return specs


def _train_variant(
    spec: dict[str, Any],
    config: RunConfig,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: str,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    model = FamilyAblatedRichTagsModel(
        drop_global=spec["drop_global"],
        drop_action=spec["drop_action"],
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        action_hidden=config.action_hidden,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        epoch_t0 = time.perf_counter()
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
            "epoch_seconds": time.perf_counter() - epoch_t0,
            **eval_metrics,
        }
        history.append(row)
        if best_metrics is None or eval_metrics["mean_regret"] < best_metrics["mean_regret"]:
            best_metrics = eval_metrics
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch + 1
        print(
            f"{spec['variant']} epoch {epoch+1:02d}/{config.epochs} "
            f"loss={row['train_loss']:.3f} regret={eval_metrics['mean_regret']:.3f} "
            f"match={eval_metrics['match_rate']:.2%}",
            flush=True,
        )
    assert best_metrics is not None
    assert best_state is not None
    final_metrics, final_buckets, _ = evaluate_model_with_buckets(model, eval_loader, device)
    final_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    best_metrics, best_buckets, _ = evaluate_model_with_buckets(model, eval_loader, device)
    result = {
        **spec,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "best_buckets": best_buckets,
        "final_buckets": final_buckets,
        "history": history,
        "wall_seconds": time.perf_counter() - t0,
        "checkpoint": {
            "model_state_dict": best_state,
            "final_model_state_dict": final_state,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "final_metrics": final_metrics,
        },
    }
    return model, result


def _comparison_row(
    variant: dict[str, Any],
    full: dict[str, Any],
    rich_prior: dict[str, Any],
    v0_best: dict[str, Any],
    raw_final: dict[str, Any],
) -> dict[str, Any]:
    best = variant["best_metrics"]
    full_best = full["best_metrics"]
    prior_best = rich_prior["raw_plus_rich_tags_many_signal_model_best"]
    return {
        "variant": variant["variant"],
        "drop_family": variant["drop_family"],
        "dropped_global_count": len(variant["drop_global"]),
        "dropped_action_count": len(variant["drop_action"]),
        "best_epoch": variant["best_epoch"],
        "best_mean_regret": best["mean_regret"],
        "best_match_rate": best["match_rate"],
        "best_near_tie": best["near_tie_rate_regret_lt_0_5"],
        "best_tail_ge_5": best["tail_regret_rate_ge_5"],
        "delta_vs_full_retrain_mean_regret": best["mean_regret"] - full_best["mean_regret"],
        "delta_vs_full_retrain_match_rate": best["match_rate"] - full_best["match_rate"],
        "delta_vs_prior_rich_best_mean_regret": best["mean_regret"] - prior_best["mean_regret"],
        "delta_vs_v0_best_mean_regret": best["mean_regret"] - v0_best["mean_regret"],
        "delta_vs_raw_final_mean_regret": best["mean_regret"] - raw_final["mean_regret"],
        "interpretation": _interpret(best["mean_regret"] - full_best["mean_regret"], best["n"]),
    }


def _interpret(delta_regret: float, n: float) -> str:
    if n < 80:
        return "underpowered"
    if delta_regret >= 0.15:
        return "drop hurt: likely useful signal"
    if delta_regret <= -0.15:
        return "drop helped: likely noise or regularization artifact"
    return "near zero: weak/noisy marginal signal"


def _write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "variant",
        "drop_family",
        "dropped_global_count",
        "dropped_action_count",
        "best_epoch",
        "best_mean_regret",
        "best_match_rate",
        "best_near_tie",
        "best_tail_ge_5",
        "delta_vs_full_retrain_mean_regret",
        "delta_vs_full_retrain_match_rate",
        "delta_vs_prior_rich_best_mean_regret",
        "delta_vs_v0_best_mean_regret",
        "delta_vs_raw_final_mean_regret",
        "interpretation",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--v0-metrics-path", default="scratch/w42/v0_strategy_tags_baseline/metrics.json")
    parser.add_argument("--rich-metrics-path", default="scratch/w42/rich_tag_many_signal_probe/metrics.json")
    parser.add_argument("--output-dir", default="scratch/w42/strategy_tag_family_ablations")
    parser.add_argument(
        "--ablation-mode",
        choices=["families-only", "with-rich-disabled", "all"],
        default="with-rich-disabled",
    )
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
        bead_id="t42-csw6.15",
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
        v0_metrics_path=args.v0_metrics_path,
        rich_metrics_path=args.rich_metrics_path,
        output_dir=args.output_dir,
        ablation_mode=args.ablation_mode,
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
        tags=[
            "w42",
            "winning42",
            "strategy-validation",
            "forge-eq",
            "gus-format",
            "scratch",
            "ablation",
            "tag-family",
            "t42-csw6.15",
        ],
    )
    started_at = datetime.now(UTC).isoformat()
    run_t0 = time.perf_counter()
    print(f"device={device}", flush=True)

    raw_prior_metrics = _load_raw_metrics(ROOT / args.raw_metrics_path)
    v0_prior_metrics = _load_raw_metrics(ROOT / args.v0_metrics_path)
    rich_prior_metrics = _load_raw_metrics(ROOT / args.rich_metrics_path)

    train_base = JointWorldFullDataset(args.train, seed=args.seed, include_strategy_features=True)
    eval_base = JointWorldFullDataset(args.eval, seed=args.seed, include_strategy_features=True)
    train_ds = _subset(train_base, args.train_limit, args.split_seed)
    eval_ds = _subset(eval_base, args.eval_limit, args.eval_seed)
    eq_eval_ds = _subset(EQNWrapper(eval_base, args.eq_n), args.eval_limit, args.eval_seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    eq_loader = DataLoader(eq_eval_ds, batch_size=args.batch_size, shuffle=False)

    specs = _variant_specs(args.ablation_mode)
    results: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    for variant_idx, spec in enumerate(specs, start=1):
        print(f"\n=== {variant_idx}/{len(specs)} {spec['variant']} ===", flush=True)
        _, result = _train_variant(spec, config, train_loader, eval_loader, device)
        checkpoints[spec["variant"]] = result.pop("checkpoint")
        results.append(result)
        best = result["best_metrics"]
        wb.log_series_point(
            axis="variant/index",
            value=variant_idx,
            step=variant_idx,
            metrics={
                "variant/total": len(specs),
                "variant/best_mean_regret": best["mean_regret"],
                "variant/best_match_rate": best["match_rate"],
                "variant/best_tail_ge_5": best["tail_regret_rate_ge_5"],
                "variant/best_epoch": result["best_epoch"],
                "variant/dropped_global_count": len(spec["drop_global"]),
                "variant/dropped_action_count": len(spec["drop_action"]),
            },
        )
        wb.log(
            {
                f"variant/{spec['variant']}/best_mean_regret": best["mean_regret"],
                f"variant/{spec['variant']}/best_match_rate": best["match_rate"],
                f"variant/{spec['variant']}/best_tail_ge_5": best["tail_regret_rate_ge_5"],
                f"variant/{spec['variant']}/best_epoch": result["best_epoch"],
            },
            step=variant_idx,
        )

    full = next(r for r in results if r["variant"] == "full_rich_retrain")
    raw_model = _load_raw_model(ROOT / args.raw_checkpoint, config, device)
    raw_final_metrics, raw_final_buckets, _ = evaluate_model_with_buckets(raw_model, eval_loader, device)
    eq_metrics = evaluate_eq_n(eq_loader, device)
    oracle_metrics = evaluate_oracle_best(eval_loader, device)
    v0_best = v0_prior_metrics["raw_plus_v0_strategy_tags_model_best"]
    matrix_rows = [
        _comparison_row(result, full, rich_prior_metrics, v0_best, raw_final_metrics)
        for result in results
    ]
    family_rows = [row for row in matrix_rows if row["drop_family"] not in {"none", "all_rich_signals"}]
    hurt_rows = sorted(family_rows, key=lambda row: row["delta_vs_full_retrain_mean_regret"], reverse=True)
    helped_rows = sorted(family_rows, key=lambda row: row["delta_vs_full_retrain_mean_regret"])
    full_vs_raw_buckets = _bucket_comparison(raw_final_buckets, full["best_buckets"])
    finished_at = datetime.now(UTC).isoformat()
    wandb_status = wb.status()

    metrics = {
        "full_rich_retrain": full["best_metrics"],
        "prior_rich_best": rich_prior_metrics["raw_plus_rich_tags_many_signal_model_best"],
        "prior_v0_best": v0_best,
        "raw_final_checkpoint": raw_final_metrics,
        f"e_q_n_{args.eq_n}": eq_metrics,
        "oracle_best_e_q_ceiling": oracle_metrics,
        "ablation_matrix": matrix_rows,
        "top_drop_hurts": hurt_rows[:5],
        "top_drop_helps": helped_rows[:5],
        "bucket_comparison_raw_final_vs_full_rich_retrain_best": full_vs_raw_buckets,
    }
    run = {
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": time.perf_counter() - run_t0,
        "config": asdict(config),
        "variants": results,
        "metrics": metrics,
        "wandb": wandb_status,
        "huggingface": "not applicable",
    }
    manifest = {
        "schema_version": "w42.strategy_tag_family_ablations.v1",
        "bead_id": "t42-csw6.15",
        "created_at": finished_at,
        "repo_commit": _git_sha(),
        "git_status_before_artifacts": _git_status_short(),
        "source_corpora": [
            {"path": p, "role": "train", "exists_at_run_time": Path(p).exists()}
            for p in args.train
        ]
        + [
            {"path": p, "role": "eval-only", "exists_at_run_time": Path(p).exists()}
            for p in args.eval
        ],
        "source_artifacts": {
            "raw_metrics": args.raw_metrics_path,
            "raw_checkpoint": args.raw_checkpoint,
            "v0_metrics": args.v0_metrics_path,
            "rich_metrics": args.rich_metrics_path,
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
        "ablation_method": {
            "mode": "cheap paired family-drop retraining",
            "baseline": "raw + v0 tags + rich signals",
            "drop_scope": "selected rich global/action proxy columns are zeroed before projection; raw and v0 tags remain available",
            "family_drops": FAMILY_DROPS,
            "caveat": "families overlap and use public proxies, so this estimates marginal training signal/noise rather than causal detector truth",
        },
        "leakage_exclusions": [
            "eval corpus uses 900000-900019 seeds and is eval-only",
            "oracle E[Q] values are labels/metrics only, not model inputs",
            "family drops affect public-state/action-local rich proxy columns only",
            "Burl traces and private table-talk text are not consumed",
        ],
        "claim_ledger_impact": "no claim-ledger change",
        "wandb": wandb_status,
        "huggingface": "not applicable",
    }

    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "run.json", run)
    _write_json(out_dir / "metrics.json", metrics)
    _write_matrix_csv(out_dir / "ablation_matrix.csv", matrix_rows)
    torch.save({"config": asdict(config), "checkpoints": checkpoints, "metrics": metrics}, out_dir / "models.pt")
    wb.update_summary(
        {
            "status": "completed",
            "full_best_mean_regret": full["best_metrics"]["mean_regret"],
            "full_best_match_rate": full["best_metrics"]["match_rate"],
            "largest_hurt_family": hurt_rows[0]["drop_family"] if hurt_rows else "not applicable",
            "largest_hurt_delta_regret": hurt_rows[0]["delta_vs_full_retrain_mean_regret"] if hurt_rows else 0.0,
            "largest_help_family": helped_rows[0]["drop_family"] if helped_rows else "not applicable",
            "largest_help_delta_regret": helped_rows[0]["delta_vs_full_retrain_mean_regret"] if helped_rows else 0.0,
        }
    )
    wb.log_artifact_files(
        name=f"w42-strategy-tag-family-ablations-{_git_sha()[:8]}",
        artifact_type="w42-ablation",
        paths=[
            out_dir / "manifest.json",
            out_dir / "run.json",
            out_dir / "metrics.json",
            out_dir / "ablation_matrix.csv",
            out_dir / "models.pt",
        ],
    )
    wb.finish()

    print("\n=== Summary ===", flush=True)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"artifacts={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
