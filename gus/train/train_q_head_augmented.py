"""Fine-tune Q_head with partial-depletion world_assignment augmentation.

Hypothesis: Q_head fails at rollout leaves because it's only trained on
initial-deal world assignments (full 21 opp dominos assigned). At a leaf
state after 1-3 opp plays, 1-3 of those dominos are absent. Fine-tuning
with random partial depletion teaches Q_head to be OOD-robust.

Architecture change: freeze everything except q_head. The world_encoder
is also left trainable — it processes world_assign, so it must adapt too.

Augmentation: with p=aug_prob, pick k ~ Uniform(1, max_depletion) and
zero k random assigned slots from world_assign. The Q target (q_per_world)
is unchanged — the assumption is that Q should be invariant to whether a
played domino is still marked in the assignment (played info is in tokens).

Usage:
    python -u -m gus.train.train_q_head_augmented \\
        --init gus/adapters/v3_consistency_10000g.pt \\
        --train gus/data/corpus_train_chunk_*.pt \\
        --eval gus/data/corpus_eval_20.pt \\
        --epochs 15 --batch-size 256 --lr 5e-5 \\
        --out gus/adapters/q_head_aug.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gus.model.dataset_seq_world import JointWorldFullDataset, JointWorldFullIterable
from gus.model.student import StudentTransformerFullVoids


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _augment_world_assign(
    world_assign: torch.Tensor,  # [B, 28, 3]
    aug_prob: float,
    max_depletion: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Randomly zero k assigned slots per item with probability aug_prob.

    For augmented items: picks k ~ Uniform(1, max_depletion), then picks
    k of the assigned domino rows (rows where world_assign.sum(-1) > 0)
    and zeros them. Unaugmented items pass through unchanged.
    """
    B = world_assign.shape[0]
    out = world_assign.clone()
    for i in range(B):
        if torch.rand(1, generator=rng).item() > aug_prob:
            continue
        # Find assigned domino indices (rows with any nonzero entry)
        assigned = (out[i].sum(dim=-1) > 0).nonzero(as_tuple=True)[0]  # [n_assigned]
        n = len(assigned)
        if n == 0:
            continue
        k = int(torch.randint(1, max_depletion + 1, (1,), generator=rng).item())
        k = min(k, n)
        # Sample k indices without replacement
        perm = torch.randperm(n, generator=rng)[:k]
        zero_rows = assigned[perm]
        out[i, zero_rows, :] = 0.0
    return out


def _run_epoch(
    model: StudentTransformerFullVoids,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    aug_prob: float,
    max_depletion: int,
    rng: torch.Generator,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    # Always keep frozen parts in eval mode
    model.encoder.eval()
    model.belief.eval()
    model.v_head.eval()
    model.pi_me.eval()

    total_q_loss = 0.0
    total_q_mae = 0.0
    total_items = 0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["tokens"].shape[0]

        wa = batch["world_assignment"]  # [B, 28, 3]
        if is_train and aug_prob > 0:
            wa = _augment_world_assign(wa, aug_prob, max_depletion, rng)

        with torch.set_grad_enabled(is_train):
            out = model(
                batch["tokens"],
                batch["attention_mask"],
                wa,
                batch["voids"],
            )
            q_pred = out["q"]           # [B, 7]
            q_target = batch["q_per_world"]  # [B, 7]
            legal = batch["legal_mask"].float()  # [B, 7]

            # MSE loss over legal actions only
            diff = (q_pred - q_target) ** 2
            loss = (diff * legal).sum() / legal.sum().clamp(min=1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.q_head.parameters()) + list(model.world_encoder.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

        total_q_loss += float(loss.item())
        mae = ((q_pred - q_target).abs() * legal).sum() / legal.sum().clamp(min=1)
        total_q_mae += float(mae.item())
        total_items += B
        n_batches += 1

    return {
        "q_loss": total_q_loss / max(n_batches, 1),
        "q_mae": total_q_mae / max(n_batches, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True,
                        help="Base checkpoint to fine-tune (v3_consistency_*.pt)")
    parser.add_argument("--train", required=True, nargs="+",
                        help="Training corpus .pt files")
    parser.add_argument("--eval", required=True, nargs="+",
                        help="Eval corpus .pt files")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--aug-prob", type=float, default=0.5,
                        help="Probability per item of applying depletion augmentation")
    parser.add_argument("--max-depletion", type=int, default=3,
                        help="Max dominos to zero per augmented item (1..max_depletion)")
    parser.add_argument("--out", type=str, default="gus/adapters/q_head_aug.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=8192)
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Device: {device}  aug_prob={args.aug_prob}  max_depletion={args.max_depletion}",
          flush=True)

    # Load checkpoint
    print(f"Loading from {args.init}...", flush=True)
    ckpt = torch.load(args.init, weights_only=False, map_location=device)
    margs = ckpt["args"]
    model = StudentTransformerFullVoids(
        d_model=margs["d_model"],
        n_heads=margs["n_heads"],
        n_layers=margs["n_layers"],
        ff_dim=margs.get("ff_dim", 256),
        dropout=0.0,
        d_world=margs.get("d_world", 64),
        q_hidden=margs.get("q_hidden", 256),
        voids_hidden=margs.get("voids_hidden", 64),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # Freeze everything except q_head and world_encoder
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.q_head.parameters():
        p.requires_grad_(True)
    for p in model.world_encoder.parameters():
        p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {n_trainable:,}  (frozen: {n_frozen:,})", flush=True)

    # Datasets
    t0 = time.perf_counter()
    if args.lazy:
        train_ds = JointWorldFullIterable(args.train, shuffle=True,
                                          buffer_size=args.buffer_size)
        eval_ds = JointWorldFullIterable(args.eval, shuffle=False, seed=42)
    else:
        train_ds = JointWorldFullDataset(args.train)
        eval_ds = JointWorldFullDataset(args.eval, seed=42)
    print(f"Datasets loaded in {time.perf_counter()-t0:.1f}s  "
          f"train={len(train_ds)}  eval={len(eval_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False if args.lazy else True,
                              num_workers=args.num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05,
    )

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    best_q_mae = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        tr = _run_epoch(model, train_loader, optimizer, device,
                        args.aug_prob, args.max_depletion, rng)
        ev = _run_epoch(model, eval_loader, None, device,
                        0.0, args.max_depletion, rng)  # eval: no augmentation
        scheduler.step()
        dt = time.perf_counter() - t_epoch

        print(
            f"epoch {epoch+1:3d}/{args.epochs}  dt={dt:.1f}s  "
            f"train q_loss={tr['q_loss']:.4f} q_mae={tr['q_mae']:.3f}  "
            f"eval  q_loss={ev['q_loss']:.4f} q_mae={ev['q_mae']:.3f}",
            flush=True,
        )

        if ev["q_mae"] < best_q_mae:
            best_q_mae = ev["q_mae"]
            torch.save({
                "model_state": model.state_dict(),
                "args": margs,
                "aug_args": vars(args),
                "eval": ev,
                "epoch": epoch + 1,
            }, out_path)
            print(f"  -> saved best (eval q_mae={best_q_mae:.3f})", flush=True)

    print(f"\nFinal best eval q_mae: {best_q_mae:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
