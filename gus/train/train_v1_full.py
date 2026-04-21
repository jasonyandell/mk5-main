"""Train the full 4-head v1 student (belief + V + π_me + world-conditioned Q).

Usage:
    python -u -m gus.train.train_v1_full \
        --train gus/data/corpus_train_1000.pt \
        --eval  gus/data/corpus_eval_100.pt \
        --epochs 20 --batch-size 128 --lr 3e-4 \
        --d-model 128 --n-heads 4 --n-layers 3 --dropout 0.1 \
        --out gus/adapters/v1_full.pt

Each epoch sees every (decision) once paired with a fresh random world
from its joint-world tensor — gives the Q head dense supervision without
materializing the full M-world cross product.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.student import (
    StudentTransformerFull,
    belief_accuracy,
    pi_me_accuracy,
    v1_full_loss,
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_epoch(
    model: StudentTransformerFull,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    loss_weights: dict[str, float],
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    # Accumulators
    totals = defaultdict(float)
    n_batches = 0
    belief_c, belief_t = 0, 0
    pi_c, pi_t = 0, 0
    v_sq_err_sum = 0.0
    v_n = 0
    q_abs_err_sum = 0.0
    q_n = 0

    for batch in loader:
        batch_on_device = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            out = model(
                batch_on_device["tokens"],
                batch_on_device["attention_mask"],
                batch_on_device["world_assignment"],
            )
            loss, per_head = v1_full_loss(out, batch_on_device, weights=loss_weights)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k, v in per_head.items():
            totals[k] += v
        n_batches += 1

        # Belief accuracy
        c, t = belief_accuracy(
            out["belief_logits"],
            batch_on_device["belief_target"],
            batch_on_device["belief_mask"],
        )
        belief_c += c
        belief_t += t

        # π_me accuracy (bot-match)
        c, t = pi_me_accuracy(
            out["pi_me_logits"],
            batch_on_device["legal_mask"],
            batch_on_device["action_taken"],
        )
        pi_c += c
        pi_t += t

        # V MAE against e_q[action_taken]
        B = batch_on_device["e_q"].shape[0]
        idx = torch.arange(B, device=device)
        e_q_at_a = batch_on_device["e_q"][idx, batch_on_device["action_taken"]]
        v_sq_err_sum += float(((out["v"] - e_q_at_a).abs()).sum().item())
        v_n += B

        # Q MAE on legal actions
        q_err = (out["q"] - batch_on_device["q_per_world"]).abs()
        q_mask = batch_on_device["legal_mask"].float()
        q_abs_err_sum += float((q_err * q_mask).sum().item())
        q_n += int(q_mask.sum().item())

    return {
        **{k: v / max(n_batches, 1) for k, v in totals.items()},
        "belief_acc": belief_c / max(belief_t, 1),
        "pi_me_acc": pi_c / max(pi_t, 1),
        "v_mae": v_sq_err_sum / max(v_n, 1),
        "q_mae": q_abs_err_sum / max(q_n, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, nargs="+",
                        help="Path(s) to training corpus .pt — accepts globs or multiple paths")
    parser.add_argument("--eval", required=True, nargs="+",
                        help="Path(s) to eval corpus .pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-world", type=int, default=64)
    parser.add_argument("--q-hidden", type=int, default=256)
    parser.add_argument("--w-belief", type=float, default=1.0)
    parser.add_argument("--w-v", type=float, default=0.5)
    parser.add_argument("--w-pi", type=float, default=0.5)
    parser.add_argument("--w-q", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="gus/adapters/v1_full.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Device: {device}", flush=True)

    t0 = time.perf_counter()
    train_ds = JointWorldFullDataset(args.train)
    eval_ds = JointWorldFullDataset(args.eval, seed=42)
    print(f"Loaded datasets: train={len(train_ds)} decisions, "
          f"eval={len(eval_ds)} decisions (in {time.perf_counter() - t0:.1f}s)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = StudentTransformerFull(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        d_world=args.d_world,
        q_hidden=args.q_hidden,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    loss_weights = {"belief": args.w_belief, "v": args.w_v, "pi_me": args.w_pi, "q": args.w_q}
    print(f"Loss weights: {loss_weights}", flush=True)
    print(f"Baselines: belief-chance=33.33%  π_me-chance=~14%  Q_MAE-scale=~10 Q-pts", flush=True)

    best_score = -1.0  # composite: pi_me_acc on held-out
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        tr = _run_epoch(model, train_loader, optimizer, device, loss_weights)
        ev = _run_epoch(model, eval_loader, None, device, loss_weights)
        dt = time.perf_counter() - t_epoch

        print(
            f"epoch {epoch+1:3d}/{args.epochs}  dt={dt:.1f}s\n"
            f"  train: L={tr['L_total']:.3f}  bel={tr['belief_acc']:.2%}  pi={tr['pi_me_acc']:.2%}  "
            f"vMAE={tr['v_mae']:.2f}  qMAE={tr['q_mae']:.2f}\n"
            f"  eval:  L={ev['L_total']:.3f}  bel={ev['belief_acc']:.2%}  pi={ev['pi_me_acc']:.2%}  "
            f"vMAE={ev['v_mae']:.2f}  qMAE={ev['q_mae']:.2f}",
            flush=True,
        )

        # Composite score for checkpointing: weight π_me, belief, q
        score = 0.5 * ev["pi_me_acc"] + 0.3 * ev["belief_acc"] - 0.05 * ev["q_mae"]
        if score > best_score:
            best_score = score
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "eval": ev,
                "epoch": epoch + 1,
            }, out_path)
            print(f"  -> saved best model (score={score:.4f})", flush=True)

    print(f"\nFinal best composite: {best_score:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
