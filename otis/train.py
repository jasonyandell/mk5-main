"""otis/train.py — one-flag arm trainer (control | treatment).

Mirrors ``champion.margin_net.train`` exactly on the pricing path: same 91-dim
featurization, same 90/5/5 deal-hash split, CrossEntropy on ``bidder_team_pts``
over 43 bins, Adam(lr=1e-3), batch 256, 60 epochs, patience-8 early-stop on VAL
PRICING CE. The two arms share corpus, capacity, seed, data order, and trunk+
pricing initialization (RNG parity — see ``otis.model``).

Treatment adds FIXED-weight auxiliary losses (registered; no tuning):

    total = L_price + 0.5·mean(fate CE) + 0.25·(trick CE) + 0.1·(mean-consistency)

Fate CE is the mean over the five tile heads; consistency ties E[pricing pts] to
Σ_t value_t·P(my team captures t) + E[tricks].
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from champion.margin_net import FEATURE_DIM, exceedance, mean_points
from otis.data import OtisDataset, build_split
from otis.model import N_TILES, OtisNet, consistency_penalty

# Fixed treatment loss weights — registered, one-flag honesty, never tuned.
W_FATE = 0.5
W_TRICK = 0.25
W_CONSISTENCY = 0.1

# Hyperparameters mirrored from champion.margin_net.train (identical both arms).
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 8
MIN_BID = 30  # for reliability of P(pts ≥ 30)


def _pricing_val_metrics(pricing_logits: Tensor, y: Tensor) -> dict:
    """Val pricing CE + MAE(mean-pts) + ECE of P(pts≥30) — the early-stop signal
    (val CE) and its companions."""
    ce = float(nn.functional.cross_entropy(pricing_logits, y).item())
    mae = float((mean_points(pricing_logits) - y.float()).abs().mean().item())
    exc = exceedance(pricing_logits)
    p30 = exc[:, MIN_BID].cpu().numpy()
    hit30 = (y.cpu().numpy() >= MIN_BID).astype(float)
    import numpy as np
    edges = np.linspace(0.0, 1.0, 11)
    num = den = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (p30 >= lo) & (p30 <= hi) if i == 9 else (p30 >= lo) & (p30 < hi)
        if m.sum():
            num += m.sum() * abs(p30[m].mean() - hit30[m].mean())
            den += m.sum()
    ece = float(num / den) if den else float("nan")
    return {"ce": ce, "mae_mean_pts": mae, "ece_p30": ece}


def train(arm: str, seed: int, out: Path, *, source: str = "selfplay",
          device: str = "cpu", epochs: int = EPOCHS) -> dict:
    treatment = arm == "treatment"
    if arm not in ("control", "treatment"):
        raise ValueError(f"arm must be control|treatment, got {arm!r}")

    tr = build_split("train", source)
    va = build_split("val", source)
    print(f"[train] arm={arm} seed={seed} source={source}  "
          f"train={len(tr)} val={len(va)}", flush=True)

    torch.manual_seed(seed)
    model = OtisNet(in_dim=FEATURE_DIM, treatment=treatment).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    price_loss = nn.CrossEntropyLoss()

    ds = OtisDataset(tr)
    gen = torch.Generator().manual_seed(seed)  # identical batch order both arms
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)

    va_X = va.X.to(device)
    va_yp = va.y_price.to(device)

    best_val = float("inf")
    best_state = None
    bad = 0
    t0 = time.time()
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        last_log = time.time()
        for x_b, yp_b, yt_b, yf_b in loader:
            x_b = x_b.to(device)
            out_b = model(x_b)
            loss = price_loss(out_b["pricing"], yp_b.to(device))
            if treatment:
                fate_logits = out_b["fate"]  # [B,5,8]
                yf_b = yf_b.to(device)
                fate_ce = sum(
                    nn.functional.cross_entropy(fate_logits[:, t, :], yf_b[:, t])
                    for t in range(N_TILES)
                ) / N_TILES
                trick_ce = nn.functional.cross_entropy(out_b["trick"], yt_b.to(device))
                cons = consistency_penalty(out_b["pricing"], fate_logits, out_b["trick"])
                loss = loss + W_FATE * fate_ce + W_TRICK * trick_ce + W_CONSISTENCY * cons
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x_b)
            n += len(x_b)
            if time.time() - last_log > 45:
                print(f"[train]   e{epoch} ..{n}/{len(tr)} loss={tot/n:.4f}", flush=True)
                last_log = time.time()
        train_loss = tot / n

        model.eval()
        with torch.no_grad():
            va_logits = model(va_X)["pricing"].cpu()
        vm = _pricing_val_metrics(va_logits, va.y_price)
        print(f"[train]   epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_ce={vm['ce']:.4f}  val_mae={vm['mae_mean_pts']:.3f}  "
              f"val_ece30={vm['ece_p30']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        if vm["ce"] < best_val - 1e-5:
            best_val = vm["ce"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[train]   early stop at epoch {epoch} "
                      f"(no val improvement in {PATIENCE})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feature_dim": FEATURE_DIM,
        "arm": arm,
        "treatment": treatment,
        "seed": seed,
        "source": source,
        "loss_weights": {"fate": W_FATE, "trick": W_TRICK, "consistency": W_CONSISTENCY},
        "hyperparams": {"epochs": epochs, "batch_size": BATCH_SIZE, "lr": LR,
                        "patience": PATIENCE},
    }, out)
    metrics = {
        "arm": arm, "seed": seed, "source": source,
        "best_val_ce": best_val if best_val != float("inf") else float("nan"),
        "epochs_run": epoch, "train_rows": len(tr), "val_rows": len(va),
        "loss_weights": {"fate": W_FATE, "trick": W_TRICK, "consistency": W_CONSISTENCY},
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out.with_suffix(".metrics.json")).write_text(json.dumps(metrics, indent=2))
    print(f"[train] saved {out}  best_val_ce={metrics['best_val_ce']:.4f}  "
          f"epochs={epoch}  {metrics['wall_seconds']}s", flush=True)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="otis v0 arm trainer")
    ap.add_argument("--arm", required=True, choices=["control", "treatment"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="selfplay")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    train(args.arm, args.seed, args.out, source=args.source,
          device=args.device, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
