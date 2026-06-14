"""Train the v2 full student with explicit engine-computed void features.

Same 4-head loss as v1_full; adds VoidsEncoder that projects a [24]-dim
void indicator vector into d_model and adds it to the pooled state_emb
before the heads run. Explicit void evidence helps the belief head
especially — that's where the data-size ceiling was hit in v1.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gus.model.dataset_seq_world import JointWorldFullDataset, JointWorldFullIterable
from gus.model.student import (
    StudentTransformerFullVoids,
    StudentTransformerFullVoidsAuction,
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
    model: StudentTransformerFullVoids,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    loss_weights: dict[str, float],
    is_auction: bool = False,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    totals = defaultdict(float)
    n_batches = 0
    belief_c, belief_t = 0, 0
    pi_c, pi_t = 0, 0
    v_abs_sum = 0.0
    v_n = 0
    q_abs_sum = 0.0
    q_n = 0

    for batch in loader:
        batch_on_device = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            if is_auction:
                out = model(
                    batch_on_device["tokens"],
                    batch_on_device["attention_mask"],
                    batch_on_device["world_assignment"],
                    batch_on_device["voids"],
                    batch_on_device["bids"],
                )
            else:
                out = model(
                    batch_on_device["tokens"],
                    batch_on_device["attention_mask"],
                    batch_on_device["world_assignment"],
                    batch_on_device["voids"],
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

        c, t = belief_accuracy(
            out["belief_logits"],
            batch_on_device["belief_target"],
            batch_on_device["belief_mask"],
        )
        belief_c += c
        belief_t += t
        c, t = pi_me_accuracy(
            out["pi_me_logits"],
            batch_on_device["legal_mask"],
            batch_on_device["action_taken"],
        )
        pi_c += c
        pi_t += t
        B = batch_on_device["e_q"].shape[0]
        idx = torch.arange(B, device=device)
        e_q_at_a = batch_on_device["e_q"][idx, batch_on_device["action_taken"]]
        v_abs_sum += float((out["v"] - e_q_at_a).abs().sum().item())
        v_n += B
        q_err = (out["q"] - batch_on_device["q_per_world"]).abs()
        q_mask_f = batch_on_device["legal_mask"].float()
        q_abs_sum += float((q_err * q_mask_f).sum().item())
        q_n += int(q_mask_f.sum().item())

    return {
        **{k: v / max(n_batches, 1) for k, v in totals.items()},
        "belief_acc": belief_c / max(belief_t, 1),
        "pi_me_acc": pi_c / max(pi_t, 1),
        "v_mae": v_abs_sum / max(v_n, 1),
        "q_mae": q_abs_sum / max(q_n, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, nargs="+")
    parser.add_argument("--eval", required=True, nargs="+")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-world", type=int, default=64)
    parser.add_argument("--q-hidden", type=int, default=256)
    parser.add_argument("--voids-hidden", type=int, default=128)
    parser.add_argument("--auction", action="store_true",
                        help="Train the auction-conditioned student (#24): adds a "
                             "BidsEncoder over the [18]-dim auction feature. Requires a "
                             "corpus generated from real auctions (forge.cli.generate_eq_from_snapshots).")
    parser.add_argument("--bids-hidden", type=int, default=64,
                        help="Hidden width of the BidsEncoder MLP (only with --auction).")
    parser.add_argument("--shuffle-bids", action="store_true",
                        help="Capacity control (#24): source each game's auction feature "
                             "from a different game, breaking the auction↔deal correlation. "
                             "Same BidsEncoder capacity, no real auction information.")
    parser.add_argument("--w-belief", type=float, default=1.0)
    parser.add_argument("--w-v", type=float, default=0.5)
    parser.add_argument("--w-pi", type=float, default=0.5)
    parser.add_argument("--w-q", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="gus/adapters/v2_voids.pt")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed torch RNG (model init + per-item world draw) for "
                             "reproducible / multi-seed runs. None = nondeterministic.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lazy", action="store_true",
                        help="Stream chunks via JointWorldFullIterable (bounded RAM). "
                             "Required for >3k-game train corpora on the M5 Max.")
    parser.add_argument("--buffer-size", type=int, default=8192,
                        help="Shuffle buffer size when --lazy (items, not bytes).")
    parser.add_argument("--length-cache", type=str, default=None,
                        help="Path to cache corpus length when --lazy (skips rescan).")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = args.device or _pick_device()
    print(f"Device: {device}", flush=True)

    t0 = time.perf_counter()
    if args.lazy:
        if args.shuffle_bids:
            raise SystemExit("--shuffle-bids is only supported with the non-lazy dataset")
        train_ds = JointWorldFullIterable(
            args.train, shuffle=True, buffer_size=args.buffer_size,
            length_cache_path=args.length_cache,
        )
        eval_ds = JointWorldFullIterable(args.eval, shuffle=False, seed=42)
    else:
        train_ds = JointWorldFullDataset(args.train, shuffle_bids=args.shuffle_bids)
        eval_ds = JointWorldFullDataset(args.eval, seed=42, shuffle_bids=args.shuffle_bids)
    print(f"Loaded datasets: train={len(train_ds)} decisions, "
          f"eval={len(eval_ds)} decisions (in {time.perf_counter() - t0:.1f}s)", flush=True)

    # IterableDataset handles its own shuffling; DataLoader shuffle=False either way.
    # persistent_workers keeps the workers (and their pickled corpus copy) alive
    # across epochs — essential on macOS spawn, where re-spawning would otherwise
    # re-pickle the whole corpus every epoch. The per-item featurization
    # (tokenize/voids/auction/world-sample) is the bottleneck for this small model,
    # so num_workers>0 is the speedup, not GPU.
    persistent = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=False if args.lazy else True,
                               num_workers=args.num_workers,
                               persistent_workers=persistent)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers,
                              persistent_workers=persistent)

    if args.auction:
        model = StudentTransformerFullVoidsAuction(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            d_world=args.d_world,
            q_hidden=args.q_hidden,
            voids_hidden=args.voids_hidden,
            bids_hidden=args.bids_hidden,
        ).to(device)
    else:
        model = StudentTransformerFullVoids(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            d_world=args.d_world,
            q_hidden=args.q_hidden,
            voids_hidden=args.voids_hidden,
        ).to(device)
    print(f"Auction-conditioned: {args.auction}", flush=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    loss_weights = {"belief": args.w_belief, "v": args.w_v, "pi_me": args.w_pi, "q": args.w_q}
    print(f"Loss weights: {loss_weights}", flush=True)

    best_score = -1e9
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()
        tr = _run_epoch(model, train_loader, optimizer, device, loss_weights, is_auction=args.auction)
        ev = _run_epoch(model, eval_loader, None, device, loss_weights, is_auction=args.auction)
        dt = time.perf_counter() - t_epoch

        print(
            f"epoch {epoch+1:3d}/{args.epochs}  dt={dt:.1f}s\n"
            f"  train: L={tr['L_total']:.3f}  bel={tr['belief_acc']:.2%}  pi={tr['pi_me_acc']:.2%}  "
            f"vMAE={tr['v_mae']:.2f}  qMAE={tr['q_mae']:.2f}\n"
            f"  eval:  L={ev['L_total']:.3f}  bel={ev['belief_acc']:.2%}  pi={ev['pi_me_acc']:.2%}  "
            f"vMAE={ev['v_mae']:.2f}  qMAE={ev['q_mae']:.2f}",
            flush=True,
        )

        score = 0.5 * ev["pi_me_acc"] + 0.4 * ev["belief_acc"] - 0.03 * ev["q_mae"]
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
