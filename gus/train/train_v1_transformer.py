"""Train the v1 transformer belief-only student.

Usage:
    python -u -m gus.train.train_v1_transformer \
        --train gus/data/corpus_train_100.pt \
        --eval  gus/data/corpus_eval_20.pt \
        --epochs 30 --batch-size 128 --lr 3e-4 \
        --d-model 128 --n-heads 4 --n-layers 2 \
        --out gus/adapters/v1_transformer_belief.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gus.model.dataset_seq import JointWorldSequenceDataset
from gus.model.student import StudentTransformer, belief_accuracy, belief_loss


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_epoch(
    model: StudentTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_slots = 0

    for batch in loader:
        tokens = batch["tokens"].to(device)
        attn = batch["attention_mask"].to(device)
        target = batch["belief_target"].to(device)
        mask = batch["belief_mask"].to(device)

        with torch.set_grad_enabled(is_train):
            out = model(tokens, attn)
            loss = belief_loss(out["belief_logits"], target, mask)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="gus/adapters/v1_transformer_belief.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Device: {device}", flush=True)

    t0 = time.perf_counter()
    train_ds = JointWorldSequenceDataset(args.train)
    eval_ds = JointWorldSequenceDataset(args.eval)
    print(f"Loaded datasets: train={len(train_ds)} decisions, "
          f"eval={len(eval_ds)} decisions (in {time.perf_counter() - t0:.1f}s)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = StudentTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    print("Baseline (uniform): 33.33%", flush=True)

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
            print(f"  -> saved best model (eval_acc={eval_acc:.3%})", flush=True)

    print(f"\nFinal best eval_acc: {best_eval_acc:.3%}  (chance: 33.33%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
