"""Joint co-train: belief + world_encoder + Q_head.

Closes §15's open loop. §15 showed that distribution-target belief training
in ISOLATION lowered KL-vs-truth 0.078 → 0.062 (closed 47% of the gap to
perfect belief), but downstream play didn't improve because the consuming
heads (world_encoder + Q_head) were frozen. This script unfreezes them so
the improvement can propagate.

§21 showed Gus is already at the Bayes ceiling on top-1 belief accuracy
(~39.2%), so any belief improvement must come from shape (calibration), not
argmax. The KL-vs-truth metric is the right evaluator.

Architecture: freeze encoder + voids_encoder + v_head + pi_me. Unfreeze
belief + world_encoder + q_head. Three heads co-train against two targets:
- L_belief_soft: cross-entropy vs empirical marginal over oracle worlds
- L_q: per-world MSE vs oracle q_per_world (one random world per item)

Usage:
    python -u -m gus.train.train_belief_q_joint \\
        --adapter-in gus/adapters/v3_consistency_10000g.pt \\
        --adapter-out gus/adapters/v3_belief_q_joint.pt \\
        --train gus/data/corpus_train_chunk_0-99.pt \\
                gus/data/corpus_train_chunk_100-199.pt \\
                gus/data/corpus_train_chunk_200-299.pt \\
        --eval gus/data/corpus_eval_20.pt \\
        --epochs 15 --batch-size 256 --lr 5e-5 --device mps

Then eval downstream: `python -u gus/eval/lamir1.py --mode q-bootstrap
--adapter <the new adapter> --eval gus/data/corpus_eval_20.pt`. Success
criteria: q-bootstrap regret < 0.685 (§20 baseline) AND belief KL < 0.062
(§15 baseline).
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from gus.model.dataset_seq_world import JointWorldFullDataset, N_DOMINOES, N_SEATS
from gus.model.student import StudentTransformerFullVoids


# ---------------------------------------------------------------------------
# Dataset wrapper: injects [28, 3] soft belief target from sampled worlds
# ---------------------------------------------------------------------------


class JointBeliefQDataset(Dataset):
    """Wraps JointWorldFullDataset; adds precomputed soft belief target."""

    def __init__(self, base: JointWorldFullDataset):
        self.base = base
        self.soft_targets: list[torch.Tensor] = []
        t0 = time.perf_counter()
        for g_idx, d_idx in base.index:
            decision = base.games[g_idx].decisions[d_idx]
            self.soft_targets.append(_world_hands_to_soft_target(decision.world_hands))
        dt = time.perf_counter() - t0
        print(f"  Precomputed soft targets: n={len(self.soft_targets)} in {dt:.1f}s", flush=True)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.base[idx]
        sample["belief_soft_target"] = self.soft_targets[idx]
        return sample


def _world_hands_to_soft_target(world_hands: Tensor) -> Tensor:
    """world_hands: [M, 3, 7] → [28, 3] empirical P(domino in seat)."""
    M = world_hands.shape[0]
    soft = torch.zeros(N_DOMINOES, N_SEATS, dtype=torch.float32)
    for s in range(N_SEATS):
        seat_dominoes = world_hands[:, s, :].reshape(-1).long()
        valid = (seat_dominoes >= 0) & (seat_dominoes < N_DOMINOES)
        seat_dominoes = seat_dominoes[valid]
        counts = torch.bincount(seat_dominoes, minlength=N_DOMINOES).float()
        soft[:, s] = counts / float(M)
    return soft


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def belief_soft_loss(logits: Tensor, soft_target: Tensor, mask: Tensor) -> Tensor:
    """Cross-entropy with distribution target, masked to unseen dominoes."""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logp = torch.log_softmax(logits, dim=-1)
    per_dom = -(soft_target * logp).sum(dim=-1)
    return per_dom[mask].mean()


def belief_kl(logits: Tensor, soft_target: Tensor, mask: Tensor, eps: float = 1e-10) -> Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    q = torch.softmax(logits, dim=-1).clamp_min(eps)
    p = soft_target.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    kl = (p * (p.log() - q.log())).sum(dim=-1)
    return kl[mask].mean()


def q_mse_loss(q_pred: Tensor, q_target: Tensor, legal_mask: Tensor) -> Tensor:
    """MSE between predicted Q and oracle per-world Q on legal actions."""
    legal_f = legal_mask.float()
    sq_err = (q_pred - q_target) ** 2 * legal_f
    n_legal = legal_f.sum().clamp(min=1.0)
    return sq_err.sum() / n_legal


def belief_top1(logits: Tensor, truth: Tensor, mask: Tensor) -> tuple[int, int]:
    preds = logits.argmax(dim=-1)
    if mask.sum() == 0:
        return 0, 0
    return int(((preds == truth) & mask).sum().item()), int(mask.sum().item())


def q_mae(q_pred: Tensor, q_target: Tensor, legal_mask: Tensor) -> tuple[float, int]:
    legal_f = legal_mask.float()
    abs_err = (q_pred - q_target).abs() * legal_f
    n = int(legal_f.sum().item())
    return float(abs_err.sum().item()), n


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------


def _configure_grads(model: StudentTransformerFullVoids) -> tuple[int, int]:
    """Unfreeze belief + world_encoder + q_head. Freeze the rest."""
    unfreeze_prefixes = ("belief.", "world_encoder.", "q_head.")
    n_trainable = 0
    n_frozen = 0
    for name, p in model.named_parameters():
        if any(name.startswith(pfx) for pfx in unfreeze_prefixes):
            p.requires_grad = True
            n_trainable += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    return n_trainable, n_frozen


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def _run_epoch(
    model: StudentTransformerFullVoids,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    weights: dict[str, float],
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    # Keep frozen modules in eval mode regardless.
    if is_train:
        for name, m in model.named_modules():
            if name == "":
                continue
            if not any(name.startswith(pfx.rstrip(".")) for pfx in ("belief", "world_encoder", "q_head")):
                m.eval()

    totals = defaultdict(float)
    n_batches = 0
    b_correct = 0
    b_total = 0
    q_err_sum = 0.0
    q_err_n = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        if is_train:
            out = model(batch["tokens"], batch["attention_mask"], batch["world_assignment"], batch["voids"])
        else:
            with torch.no_grad():
                out = model(batch["tokens"], batch["attention_mask"], batch["world_assignment"], batch["voids"])

        L_b = belief_soft_loss(out["belief_logits"], batch["belief_soft_target"], batch["belief_mask"])
        L_q = q_mse_loss(out["q"], batch["q_per_world"], batch["legal_mask"])
        loss = weights["belief"] * L_b + weights["q"] * L_q

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

        totals["L"] += float(loss.item())
        totals["L_belief"] += float(L_b.item())
        totals["L_q"] += float(L_q.item())
        with torch.no_grad():
            kl = belief_kl(out["belief_logits"], batch["belief_soft_target"], batch["belief_mask"])
            totals["KL"] += float(kl.item())
            c, t = belief_top1(out["belief_logits"], batch["belief_target"], batch["belief_mask"])
            b_correct += c
            b_total += t
            err_sum, err_n = q_mae(out["q"], batch["q_per_world"], batch["legal_mask"])
            q_err_sum += err_sum
            q_err_n += err_n
        n_batches += 1

    return {
        "L": totals["L"] / max(n_batches, 1),
        "L_belief": totals["L_belief"] / max(n_batches, 1),
        "L_q": totals["L_q"] / max(n_batches, 1),
        "KL": totals["KL"] / max(n_batches, 1),
        "top1": b_correct / max(b_total, 1),
        "q_mae": q_err_sum / max(q_err_n, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model(path: str, device: str) -> tuple[StudentTransformerFullVoids, dict]:
    ckpt = torch.load(path, weights_only=False, map_location=device)
    args = ckpt["args"]
    kwargs = {
        "d_model": args["d_model"],
        "n_heads": args["n_heads"],
        "n_layers": args["n_layers"],
        "ff_dim": args.get("ff_dim", 256),
        "dropout": 0.0,
        "d_world": args.get("d_world", 64),
        "q_hidden": args.get("q_hidden", 256),
        "voids_hidden": args.get("voids_hidden", 64),
    }
    model = StudentTransformerFullVoids(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, args


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-in", required=True)
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--alpha-belief", type=float, default=1.0)
    ap.add_argument("--beta-q", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or _pick_device()
    print(f"Device: {device}", flush=True)

    t0 = time.perf_counter()
    print(f"Loading train corpus: {args.train}", flush=True)
    train_base = JointWorldFullDataset(args.train, seed=0)
    print(f"Loading eval  corpus: {args.eval}", flush=True)
    eval_base = JointWorldFullDataset(args.eval, seed=42)
    print(f"  Base loaded in {time.perf_counter()-t0:.1f}s  train={len(train_base)}  eval={len(eval_base)}", flush=True)

    print("Precomputing train soft targets...", flush=True)
    train_ds = JointBeliefQDataset(train_base)
    print("Precomputing eval soft targets...", flush=True)
    eval_ds = JointBeliefQDataset(eval_base)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model, ckpt_args = _load_model(args.adapter_in, device)
    n_train, n_frozen = _configure_grads(model)
    print(f"Trainable: {n_train:,}  Frozen: {n_frozen:,}", flush=True)

    # Baseline
    print("\n=== BASELINE (adapter-in) ===", flush=True)
    base_eval = _run_epoch(model, eval_loader, None, device, {"belief": args.alpha_belief, "q": args.beta_q})
    print(f"  KL={base_eval['KL']:.4f}  top1={base_eval['top1']:.3%}  q_mae={base_eval['q_mae']:.3f}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    weights = {"belief": args.alpha_belief, "q": args.beta_q}

    print(f"\n=== Joint co-train ({args.epochs} epochs, α={args.alpha_belief} β={args.beta_q}) ===", flush=True)

    best_score = -float("inf")  # composite: -KL - q_mae/10 (both lower-better)
    best_state = None
    best_epoch = -1
    curves = []

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        tr = _run_epoch(model, train_loader, optimizer, device, weights)
        ev = _run_epoch(model, eval_loader, None, device, weights)
        dt = time.perf_counter() - t_epoch
        print(
            f"epoch {epoch+1:3d}/{args.epochs}  dt={dt:.1f}s  "
            f"train L={tr['L']:.4f} KL={tr['KL']:.4f} qMAE={tr['q_mae']:.3f}  "
            f"eval KL={ev['KL']:.4f} top1={ev['top1']:.3%} qMAE={ev['q_mae']:.3f}",
            flush=True,
        )
        curves.append({"epoch": epoch+1, **{f"train_{k}": v for k, v in tr.items()},
                       **{f"eval_{k}": v for k, v in ev.items()}})
        score = -ev["KL"] - ev["q_mae"] / 10.0
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest composite score: {best_score:.4f} (epoch {best_epoch})", flush=True)

    print("\n=== AFTER JOINT CO-TRAIN ===", flush=True)
    after = _run_epoch(model, eval_loader, None, device, weights)
    print(f"  KL={after['KL']:.4f}  top1={after['top1']:.3%}  q_mae={after['q_mae']:.3f}", flush=True)

    out_path = Path(args.adapter_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_args = dict(ckpt_args)
    save_args["_joint_cotrain"] = {
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "alpha_belief": args.alpha_belief, "beta_q": args.beta_q,
        "best_epoch": best_epoch, "best_score": best_score,
        "baseline_KL": base_eval["KL"], "final_KL": after["KL"],
        "baseline_qmae": base_eval["q_mae"], "final_qmae": after["q_mae"],
        "baseline_top1": base_eval["top1"], "final_top1": after["top1"],
        "curves": curves,
    }
    torch.save({
        "model_state": model.state_dict(),
        "args": save_args,
        "eval": after,
        "epoch": best_epoch,
    }, out_path)
    print(f"\nSaved: {out_path}", flush=True)

    print("\n=== FINAL SUMMARY ===", flush=True)
    print(f"  belief KL   {base_eval['KL']:.4f} -> {after['KL']:.4f}  (Δ {after['KL']-base_eval['KL']:+.4f})", flush=True)
    print(f"  belief top1 {base_eval['top1']:.3%} -> {after['top1']:.3%}  (Δ {(after['top1']-base_eval['top1'])*100:+.2f} pp)", flush=True)
    print(f"  q_mae       {base_eval['q_mae']:.3f} -> {after['q_mae']:.3f}  (Δ {after['q_mae']-base_eval['q_mae']:+.3f})", flush=True)
    print(f"\nNext: run q-bootstrap eval on the new adapter to see if the calibration propagates.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
