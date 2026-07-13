"""Train π_opp head — distill oracle action distribution for each opponent seat.

Loads a frozen v3_consistency trunk (encoder + all existing heads) and trains
a new π_opp_head that predicts the oracle's action-softmax for each of the 3
non-me seats (L-opp = rel-seat 1, partner = rel-seat 2, R-opp = rel-seat 3).

Architecture:
  π_opp_head(state_emb [B, D], seat_id [B]) -> logits [B, 7]
  where seat_id ∈ {1, 2, 3} (relative to current player = 0)
  seat_id is embedded via a 3-way embedding (indices 0,1,2 → rel-seats 1,2,3)
  and concatenated with state_emb before projection.

Training target: batch["oracle_softmax_per_seat"][:, rel_seat, :] for each
rel_seat in {1, 2, 3}. Loss is CE (softmax cross-entropy) legal-masked.

Each batch item generates 3 training pairs (one per non-me seat). The relative
seat assignment is: for absolute seat `a`, rel_seat = (a - current_player) % 4.
Row 0 of oracle_softmax_per_seat = me (skipped), rows 1-3 = L-opp/partner/R-opp.

Requires Schema v2 corpus with oracle_softmax_per_seat field populated.

Usage:
    python -u -m gus.train.train_pi_opp \\
        --trunk gus/adapters/v3_consistency_10000g.pt \\
        --train gus/data/corpus_v2_train_*.pt \\
        --eval  gus/data/corpus_v2_eval_*.pt \\
        --epochs 20 --batch-size 256 --lr 2e-4 \\
        --out gus/adapters/pi_opp_1000g.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gus.model.dataset_seq_world import JointWorldFullDataset, JointWorldFullIterable
from gus.model.student import StudentTransformerFullVoids


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# π_opp head
# ---------------------------------------------------------------------------

class PiOppHead(nn.Module):
    """Seat-conditioned opponent policy head.

    Takes state_emb [B, d_model] + seat_id [B] (0=L-opp, 1=partner, 2=R-opp
    in relative seat index space, i.e. rel_seat - 1) and predicts [B, 7] logits.
    """

    def __init__(self, d_model: int, seat_embed_dim: int = 8) -> None:
        super().__init__()
        self.seat_embed = nn.Embedding(3, seat_embed_dim)
        self.proj = nn.Linear(d_model + seat_embed_dim, 7)

    def forward(self, state_emb: torch.Tensor, seat_id: torch.Tensor) -> torch.Tensor:
        """
        state_emb: [B, d_model]
        seat_id:   [B] long, values in {0, 1, 2}
        returns:   [B, 7] logits
        """
        se = self.seat_embed(seat_id)        # [B, seat_embed_dim]
        return self.proj(torch.cat([state_emb, se], dim=-1))  # [B, 7]


# ---------------------------------------------------------------------------
# Wrapper: frozen trunk + trainable π_opp head
# ---------------------------------------------------------------------------

class TrunkWithPiOpp(nn.Module):
    """Frozen StudentTransformerFullVoids trunk + trainable PiOppHead."""

    def __init__(self, trunk: StudentTransformerFullVoids, d_model: int,
                 seat_embed_dim: int = 8) -> None:
        super().__init__()
        self.trunk = trunk
        self.pi_opp_head = PiOppHead(d_model, seat_embed_dim)

        # Freeze everything in trunk
        for p in self.trunk.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        world_assignment: torch.Tensor,
        voids: torch.Tensor,
        seat_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            trunk_out = self.trunk(tokens, attention_mask, world_assignment, voids)
        state_emb = trunk_out["state_emb"]  # [B, d_model]
        pi_opp_logits = self.pi_opp_head(state_emb, seat_id)  # [B, 7]
        return {"pi_opp_logits": pi_opp_logits, "state_emb": state_emb}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def pi_opp_loss(
    logits: torch.Tensor,        # [B, 7]
    target_softmax: torch.Tensor, # [B, 7]  oracle softmax for this seat
    legal_mask: torch.Tensor,    # [B, 7] bool
) -> torch.Tensor:
    """Legal-masked CE loss against oracle softmax target.

    Only legal actions contribute. Target distribution is renormalized over
    legal actions before computing CE.
    """
    # Zero out illegal actions in target and renormalize
    target = target_softmax * legal_mask.float()
    target_sum = target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target = target / target_sum  # [B, 7]

    # Mask logits
    logits_masked = logits.masked_fill(~legal_mask, float("-inf"))
    log_probs = F.log_softmax(logits_masked, dim=-1)  # [B, 7]
    # Zero illegal slots so 0 * 0 = 0, not 0 * (-inf) = NaN
    log_probs = log_probs.masked_fill(~legal_mask, 0.0)

    # CE: -sum_a target[a] * log_prob[a], only over legal actions
    ce = -(target * log_probs).sum(dim=-1)  # [B]
    return ce.mean()


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def _run_epoch(
    model: TrunkWithPiOpp,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    model.trunk.eval()  # trunk always in eval mode (frozen BN/dropout)

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["tokens"].shape[0]

        # oracle_softmax_per_seat: [B, 4, 7] — seats 0=me, 1=L-opp, 2=partner, 3=R-opp
        # We train on seats 1, 2, 3 (3 pairs per item)
        softmax_per_seat = batch["oracle_softmax_per_seat"]  # [B, 4, 7]
        legal_mask = batch["legal_mask"]  # [B, 7]

        loss_accum = torch.tensor(0.0, device=device)
        correct = 0

        for rel_seat in range(1, 4):  # L-opp, partner, R-opp
            seat_id = torch.full((B,), rel_seat - 1, dtype=torch.long, device=device)  # 0,1,2

            with torch.set_grad_enabled(is_train):
                out = model(
                    batch["tokens"],
                    batch["attention_mask"],
                    batch["world_assignment"],
                    batch["voids"],
                    seat_id,
                )
                target = softmax_per_seat[:, rel_seat, :]  # [B, 7]
                loss = pi_opp_loss(out["pi_opp_logits"], target, legal_mask)
                loss_accum = loss_accum + loss

            # Accuracy: does argmax(pi_opp) match argmax(oracle_softmax)?
            pred = out["pi_opp_logits"].masked_fill(~legal_mask, float("-inf")).argmax(dim=-1)
            oracle_best = target.masked_fill(~legal_mask, -1.0).argmax(dim=-1)
            correct += int((pred == oracle_best).sum().item())

        loss_accum = loss_accum / 3.0  # average over 3 seats

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss_accum.backward()
            nn.utils.clip_grad_norm_(model.pi_opp_head.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += float(loss_accum.item())
        total_correct += correct
        total_items += B * 3  # 3 seats per item
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": total_correct / max(total_items, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trunk", required=True,
                        help="Path to frozen trunk checkpoint (v3_consistency_*.pt)")
    parser.add_argument("--train", required=True, nargs="+",
                        help="Schema v2 training corpus .pt files")
    parser.add_argument("--eval", required=True, nargs="+",
                        help="Schema v2 eval corpus .pt files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seat-embed-dim", type=int, default=8)
    parser.add_argument("--out", type=str, default="gus/adapters/pi_opp.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lazy", action="store_true",
                        help="Stream via JointWorldFullIterable (bounded RAM).")
    parser.add_argument("--buffer-size", type=int, default=8192)
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Device: {device}", flush=True)

    # Load trunk
    print(f"Loading trunk from {args.trunk}...", flush=True)
    ckpt = torch.load(args.trunk, weights_only=False, map_location=device)
    trunk_args = ckpt["args"]
    trunk = StudentTransformerFullVoids(
        d_model=trunk_args["d_model"],
        n_heads=trunk_args["n_heads"],
        n_layers=trunk_args["n_layers"],
        ff_dim=trunk_args.get("ff_dim", 256),
        dropout=0.0,
        d_world=trunk_args.get("d_world", 64),
        q_hidden=trunk_args.get("q_hidden", 256),
        voids_hidden=trunk_args.get("voids_hidden", 64),
    ).to(device)
    trunk.load_state_dict(ckpt["model_state"])
    trunk.eval()
    d_model = trunk_args["d_model"]

    model = TrunkWithPiOpp(trunk, d_model, args.seat_embed_dim).to(device)
    n_params = sum(p.numel() for p in model.pi_opp_head.parameters())
    n_frozen = sum(p.numel() for p in model.trunk.parameters())
    print(f"π_opp_head params: {n_params:,}  (trunk frozen: {n_frozen:,})", flush=True)

    # Datasets
    t0 = time.perf_counter()
    if args.lazy:
        train_ds = JointWorldFullIterable(args.train, shuffle=True, buffer_size=args.buffer_size)
        eval_ds  = JointWorldFullIterable(args.eval, shuffle=False, seed=42)
    else:
        train_ds = JointWorldFullDataset(args.train)
        eval_ds  = JointWorldFullDataset(args.eval, seed=42)
    print(f"Datasets loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=False if args.lazy else True,
                              num_workers=args.num_workers)
    eval_loader  = DataLoader(eval_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
        model.pi_opp_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    best_acc = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        tr = _run_epoch(model, train_loader, optimizer, device)
        ev = _run_epoch(model, eval_loader, None, device)
        scheduler.step()
        dt = time.perf_counter() - t_epoch

        print(
            f"epoch {epoch+1:3d}/{args.epochs}  dt={dt:.1f}s  "
            f"train loss={tr['loss']:.4f} acc={tr['acc']:.2%}  "
            f"eval  loss={ev['loss']:.4f} acc={ev['acc']:.2%}",
            flush=True,
        )

        if ev["acc"] > best_acc:
            best_acc = ev["acc"]
            torch.save({
                "pi_opp_head_state": model.pi_opp_head.state_dict(),
                "trunk_path": args.trunk,
                "trunk_args": trunk_args,
                "args": vars(args),
                "eval": ev,
                "epoch": epoch + 1,
            }, out_path)
            print(f"  -> saved best (eval acc={best_acc:.2%})", flush=True)

    print(f"\nFinal best eval acc: {best_acc:.2%}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
