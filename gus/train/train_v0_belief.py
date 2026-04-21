"""Train the v0 belief-only student on a joint-world corpus.

Usage:
    python -u -m gus.train.train_v0_belief \
        --train gus/data/corpus_train_100.pt \
        --eval  gus/data/corpus_eval_20.pt \
        --epochs 20 --batch-size 128 --lr 1e-3 \
        --out gus/adapters/v0_belief.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gus.model.dataset import JointWorldDecisionDataset
from gus.model.student import (
    DEFAULT_HIDDEN_DIM,
    StudentV0,
    belief_accuracy,
    belief_loss,
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_epoch(
    model: StudentV0,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> tuple[float, float]:
    """Return (mean_loss, top1_accuracy). optimizer=None for eval."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_slots = 0

    for batch in loader:
        features = batch["features"].to(device)
        target = batch["belief_target"].to(device)
        mask = batch["belief_mask"].to(device)

        with torch.set_grad_enabled(is_train):
            out = model(features)
            loss = belief_loss(out["belief_logits"], target, mask)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        c, t = belief_accuracy(out["belief_logits"], target, mask)
        total_correct += c
        total_slots += t

    avg_loss = total_loss / max(total_batches, 1)
    acc = (total_correct / total_slots) if total_slots > 0 else 0.0
    return avg_loss, acc


def main() -> int:
    parser = argparse.ArgumentParser(description="Train v0 belief-only student")
    parser.add_argument("--train", required=True, help="Path to training corpus .pt")
    parser.add_argument("--eval", required=True, help="Path to eval corpus .pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--out", type=str, default="gus/adapters/v0_belief.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Device: {device}", flush=True)

    # Data
    t0 = time.perf_counter()
    train_ds = JointWorldDecisionDataset(args.train)
    eval_ds = JointWorldDecisionDataset(args.eval)
    print(f"Loaded datasets: train={len(train_ds)} decisions, eval={len(eval_ds)} "
          f"decisions (in {time.perf_counter() - t0:.1f}s)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # Model
    model = StudentV0(hidden_dim=args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Baseline: uniform-random belief = 1/3 = 33.3% top-1
    print("Baseline (uniform-random belief): 33.33%", flush=True)

    best_eval_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        train_loss, train_acc = _run_epoch(model, train_loader, optimizer, device)
        eval_loss, eval_acc = _run_epoch(model, eval_loader, None, device)
        dt = time.perf_counter() - t_epoch
        print(
            f"epoch {epoch+1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3%}  "
            f"eval_loss={eval_loss:.4f}  eval_acc={eval_acc:.3%}  "
            f"({dt:.1f}s)",
            flush=True,
        )
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "eval_acc": eval_acc,
                "epoch": epoch + 1,
            }, out_path)
            print(f"  -> saved best model (eval_acc={eval_acc:.3%}) to {out_path}", flush=True)

    print(f"\nFinal best eval_acc: {best_eval_acc:.3%}  (chance: 33.33%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
