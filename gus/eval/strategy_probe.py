"""Tiny Gus policy probe with optional explicit strategy features.

This is an intentionally small experiment:

  baseline: tokens + voids -> policy
  strategy: tokens + voids + strategy_features -> policy

Both are evaluated by regret against the corpus marginal E[Q]. The script also
computes an E[Q] N=10 baseline directly from each decision's sampled worlds:
average the first/random 10 q_per_world rows, choose the best legal action, and
score that choice against the full marginal E[Q].
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from gus.hf_data import resolve
from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.strategy_features import STRATEGY_ACTION_FEATURE_DIM, STRATEGY_FEATURE_DIM
from gus.model.student import TransformerEncoder, VoidsEncoder


class TinyPolicyProbe(nn.Module):
    def __init__(
        self,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 192,
        dropout: float = 0.1,
        use_strategy: bool = False,
        strategy_hidden: int = 96,
        use_action_strategy: bool = False,
    ):
        super().__init__()
        self.use_strategy = use_strategy
        self.use_action_strategy = use_action_strategy
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.voids_encoder = VoidsEncoder(d_model, hidden_dim=64)
        if use_strategy:
            self.strategy_encoder = nn.Sequential(
                nn.Linear(STRATEGY_FEATURE_DIM, strategy_hidden),
                nn.GELU(),
                nn.LayerNorm(strategy_hidden),
                nn.Linear(strategy_hidden, d_model),
            )
        else:
            self.strategy_encoder = None
        if use_action_strategy:
            self.action_head = nn.Sequential(
                nn.Linear(d_model + STRATEGY_ACTION_FEATURE_DIM, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            )
        else:
            self.action_head = None
        self.norm = nn.LayerNorm(d_model)
        self.pi = nn.Linear(d_model, 7)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        h = self.encoder(batch["tokens"], batch["attention_mask"])
        z = h[:, 0, :] + self.voids_encoder(batch["voids"])
        if self.use_strategy:
            z = z + self.strategy_encoder(batch["strategy_features"])
        z = self.norm(z)
        logits = self.pi(z)
        if self.use_action_strategy:
            z7 = z.unsqueeze(1).expand(-1, 7, -1)
            action_in = torch.cat([z7, batch["strategy_action_features"]], dim=-1)
            logits = logits + self.action_head(action_in).squeeze(-1)
        return logits


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _oracle_best_action(batch: dict[str, Tensor]) -> Tensor:
    e_q = batch["e_q"].masked_fill(~batch["legal_mask"], float("-inf"))
    return e_q.argmax(dim=-1)


def _policy_loss(logits: Tensor, batch: dict[str, Tensor]) -> Tensor:
    target = _oracle_best_action(batch)
    masked = logits.masked_fill(~batch["legal_mask"], -1e9)
    return nn.functional.cross_entropy(masked, target)


@torch.no_grad()
def evaluate_model(model: TinyPolicyProbe, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total_regret = 0.0
    total = 0
    hits = 0
    near = 0
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
    return {
        "regret": total_regret / max(total, 1),
        "match": hits / max(total, 1),
        "near": near / max(total, 1),
        "n": float(total),
    }


@torch.no_grad()
def evaluate_eq_n(loader: DataLoader, n_worlds: int, device: str, random_worlds: bool) -> dict[str, float]:
    total_regret = 0.0
    total = 0
    hits = 0
    near = 0
    for batch in loader:
        # q_per_world is [B, 7] because the default dataset samples one world.
        # For the N-world baseline we need direct access to decision tensors, so
        # this function is implemented on a loader whose dataset emits q_worlds_n.
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
        "regret": total_regret / max(total, 1),
        "match": hits / max(total, 1),
        "near": near / max(total, 1),
        "n": float(total),
    }


class EQNWrapper(torch.utils.data.Dataset):
    """Wrap JointWorldFullDataset and add q_worlds_n for E[Q] N baselines."""

    def __init__(self, base: JointWorldFullDataset, n_worlds: int, seed: int = 123, random_worlds: bool = False):
        self.base = base
        self.n_worlds = n_worlds
        self.seed = seed
        self.random_worlds = random_worlds

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        item = self.base[idx]
        g_idx, d_idx = self.base.index[idx]
        dec = self.base.games[g_idx].decisions[d_idx]
        qpw = dec.q_per_world.float()
        n = min(self.n_worlds, qpw.shape[0])
        if self.random_worlds:
            gen = torch.Generator().manual_seed(self.seed + idx)
            rows = torch.randperm(qpw.shape[0], generator=gen)[:n]
            item["q_worlds_n"] = qpw[rows]
        else:
            item["q_worlds_n"] = qpw[:n]
        return item


def _subset(ds: torch.utils.data.Dataset, limit: int | None, seed: int) -> torch.utils.data.Dataset:
    if limit is None or limit <= 0 or limit >= len(ds):
        return ds
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    return Subset(ds, idxs[:limit])


def train_one(
    name: str,
    use_strategy: bool,
    use_action_strategy: bool,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    model = TinyPolicyProbe(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        use_strategy=use_strategy,
        use_action_strategy=use_action_strategy,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {"regret": float("inf"), "match": 0.0, "near": 0.0, "n": 0.0}
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = _policy_loss(model(batch), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            batches += 1
        ev = evaluate_model(model, eval_loader, device)
        if ev["regret"] < best["regret"]:
            best = ev
        print(
            f"{name:>10s} epoch {epoch+1:02d}/{args.epochs} "
            f"loss={loss_sum/max(batches,1):.3f} eval_regret={ev['regret']:.3f} "
            f"match={ev['match']:.2%} near={ev['near']:.2%}",
            flush=True,
        )
    print(f"{name:>10s} best regret={best['regret']:.3f} match={best['match']:.2%} near={best['near']:.2%}", flush=True)
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", default=["gus/data/corpus_train_100.pt"])
    parser.add_argument("--eval", nargs="+", default=["gus/data/corpus_eval_20.pt"])
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--eval-limit", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eq-n", type=int, default=10)
    parser.add_argument("--random-eq-worlds", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or _device()
    print(f"device={device}", flush=True)

    t0 = time.perf_counter()
    train_base = JointWorldFullDataset([resolve(p) for p in args.train], seed=args.seed, include_strategy_features=True)
    eval_base = JointWorldFullDataset([resolve(p) for p in args.eval], seed=args.seed, include_strategy_features=True)
    train_ds = _subset(train_base, args.train_limit, args.seed)
    eval_ds = _subset(eval_base, args.eval_limit, args.seed + 1)
    eq_eval_ds = _subset(EQNWrapper(eval_base, args.eq_n, seed=args.seed, random_worlds=args.random_eq_worlds), args.eval_limit, args.seed + 1)
    print(
        f"loaded train={len(train_ds)} eval={len(eval_ds)} "
        f"(base train={len(train_base)} eval={len(eval_base)}) in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
    eq_loader = DataLoader(eq_eval_ds, batch_size=args.batch_size, shuffle=False)

    eq = evaluate_eq_n(eq_loader, args.eq_n, device, args.random_eq_worlds)
    print(
        f"E[Q] N={args.eq_n} baseline: regret={eq['regret']:.3f} "
        f"match={eq['match']:.2%} near={eq['near']:.2%}",
        flush=True,
    )

    base = train_one("base", False, False, train_loader, eval_loader, device, args)
    strat = train_one("strategy", True, True, train_loader, eval_loader, device, args)

    print("\n=== Summary ===")
    print(f"E[Q] N={args.eq_n}: regret={eq['regret']:.3f} match={eq['match']:.2%} near={eq['near']:.2%}")
    print(f"base model: regret={base['regret']:.3f} match={base['match']:.2%} near={base['near']:.2%}")
    print(f"strategy model: regret={strat['regret']:.3f} match={strat['match']:.2%} near={strat['near']:.2%}")
    print(f"strategy - base regret delta: {strat['regret'] - base['regret']:+.3f}")
    print(f"strategy - E[Q] N={args.eq_n} regret delta: {strat['regret'] - eq['regret']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
