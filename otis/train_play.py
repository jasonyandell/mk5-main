"""otis/train_play.py — train + gate OtisPlayNet (issue #53, V2-c gate).

Trains the play-time fate/trick net on the per-ply rows from ``otis.play_data`` and
evaluates the registered V2-c gate: held-out VAL per-tile mean fate NLL against the
train-marginal base rate.

  delta_nats = base_rate_nll - model_nll   (positive = model beats the base rate)

Reported overall, per tile, and by trick index (0..6), plus top-1 accuracy vs the
base-rate argmax. Two verbatim gate lines are printed:

  V2C_DELTA_NATS=<overall>
  V2C_BY_TRICK=<comma list, trick 0..6>

Adam lr 1e-3, batch 4096, up to 4 epochs, early-stop on val fate NLL, CPU, seed 42.
Pure CPU — otis does not own the GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from otis.model import N_FATE_CLASSES, N_TILES, N_TRICKS
from otis.play_model import OtisPlayNet

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROWS = ROOT / "scratch" / "otis-night2" / "play_rows"
DEFAULT_CKPT = ROOT / "otis" / "models" / "otis_play_v0.pt"


def _load_split(rows_dir: Path, split: str) -> dict:
    return torch.load(rows_dir / f"{split}.pt", weights_only=False)


def _fate_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean CE over the 5 fate heads. logits [B,5,8], labels [B,5]."""
    return F.cross_entropy(logits.reshape(-1, N_FATE_CLASSES), labels.reshape(-1))


def train_and_eval(
    rows_dir: Path = DEFAULT_ROWS,
    ckpt_path: Path = DEFAULT_CKPT,
    *,
    lr: float = 1e-3,
    batch_size: int = 4096,
    epochs: int = 4,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cpu")

    tr = _load_split(rows_dir, "train")
    va = _load_split(rows_dir, "val")
    Xtr, Ftr, Ttr = tr["X"], tr["y_fate"], tr["y_trick"]
    Xva, Fva, Tva, TIva = va["X"], va["y_fate"], va["y_trick"], va["trick_idx"]
    in_dim = int(Xtr.shape[1])
    print(f"[train_play] train N={len(Xtr)} val N={len(Xva)} feat_dim={in_dim}", flush=True)

    # --- base-rate reference from TRAIN marginals (per tile) -----------------
    base_logp = torch.zeros(N_TILES, N_FATE_CLASSES, dtype=torch.float64)
    for i in range(N_TILES):
        counts = torch.bincount(Ftr[:, i], minlength=N_FATE_CLASSES).to(torch.float64)
        base_logp[i] = torch.log(counts / counts.sum())
    base_argmax = base_logp.argmax(dim=1)  # [5]

    loader = DataLoader(
        TensorDataset(Xtr, Ftr, Ttr), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    net = OtisPlayNet(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def val_fate_nll() -> float:
        net.eval()
        with torch.no_grad():
            out = net(Xva)
            return float(_fate_nll(out["fate"], Fva))

    best_nll = float("inf")
    best_state = None
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        run = 0.0
        nb = 0
        for xb, fb, tb in loader:
            opt.zero_grad()
            out = net(xb)
            loss = _fate_nll(out["fate"], fb) + F.cross_entropy(out["trick"], tb)
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1
        v = val_fate_nll()
        history.append({"epoch": epoch, "train_loss": run / max(nb, 1), "val_fate_nll": v})
        print(f"[train_play] epoch {epoch}  train_loss {run / max(nb,1):.4f}  "
              f"val_fate_nll {v:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if v < best_nll - 1e-5:
            best_nll = v
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            print("[train_play] early stop (val fate NLL did not improve)", flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    # --- V2-c gate on VAL ----------------------------------------------------
    net.eval()
    with torch.no_grad():
        out = net(Xva)
        model_logp = F.log_softmax(out["fate"], dim=-1)  # [N,5,8]

    N = Xva.shape[0]
    ar = torch.arange(N)
    # Per-row, per-tile model NLL and base-rate NLL.
    model_nll_rt = -model_logp[ar[:, None], torch.arange(N_TILES)[None, :], Fva]  # [N,5]
    base_nll_rt = -base_logp[torch.arange(N_TILES)[None, :], Fva.to(torch.long)]   # [N,5] float64
    base_nll_rt = base_nll_rt.to(torch.float64)
    model_nll_rt = model_nll_rt.to(torch.float64)

    model_nll = float(model_nll_rt.mean())
    base_nll = float(base_nll_rt.mean())
    delta_nats = base_nll - model_nll

    per_tile = []
    for i in range(N_TILES):
        m = float(model_nll_rt[:, i].mean())
        b = float(base_nll_rt[:, i].mean())
        per_tile.append({"tile": i, "model_nll": m, "base_nll": b, "delta_nats": b - m})

    by_trick = []
    for t in range(N_TRICKS - 1):  # trick idx 0..6
        sel = TIva == t
        n = int(sel.sum())
        if n == 0:
            by_trick.append({"trick": t, "n": 0, "delta_nats": 0.0})
            continue
        m = float(model_nll_rt[sel].mean())
        b = float(base_nll_rt[sel].mean())
        by_trick.append({"trick": t, "n": n, "model_nll": m, "base_nll": b, "delta_nats": b - m})

    # Top-1 accuracy vs base-rate argmax (fate).
    model_pred = model_logp.argmax(dim=-1)  # [N,5]
    model_acc = float((model_pred == Fva).float().mean())
    base_acc = float((base_argmax[None, :].expand(N, N_TILES) == Fva).float().mean())

    metrics = {
        "feature_dim": in_dim,
        "n_train": int(len(Xtr)),
        "n_val": int(N),
        "val_fate_nll": model_nll,
        "base_rate_nll": base_nll,
        "delta_nats": delta_nats,
        "per_tile": per_tile,
        "by_trick": by_trick,
        "fate_top1_acc": model_acc,
        "base_top1_acc": base_acc,
        "history": history,
        "hparams": {"lr": lr, "batch_size": batch_size, "epochs": epochs, "seed": seed},
        "wall_s": round(time.time() - t0, 1),
    }

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    net.save(ckpt_path)
    metrics_path = ckpt_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[train_play] checkpoint → {ckpt_path}", flush=True)
    print(f"[train_play] metrics → {metrics_path}", flush=True)
    print(f"[train_play] fate top1 acc {model_acc:.4f} vs base {base_acc:.4f}", flush=True)
    print(f"V2C_DELTA_NATS={delta_nats:.6f}", flush=True)
    print("V2C_BY_TRICK=" + ",".join(f"{d['delta_nats']:.6f}" for d in by_trick), flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(DEFAULT_ROWS))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train_and_eval(
        Path(args.rows), Path(args.ckpt),
        lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
    )


if __name__ == "__main__":
    main()
