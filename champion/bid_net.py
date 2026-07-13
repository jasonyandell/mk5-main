"""Bid-strength net: hand -> (9 decl × 13 bid) P(make) table.

Featurization
-------------
28-dim multi-hot hand mask  (which of 28 dominoes are in hand)
+ 7 × 5 = 35 trump-structured features from arena.hand_metrics.evaluate_trump:
    trump_count / 7          (float, normalised)
    has_trump_double         (float 0/1)
    held_count_points / 35   (float, normalised)
    unique_exposed_points / 35
    bid_ceiling / 42         (float, normalised)
Total: 63 dims.

Model
-----
MLP: 63 → 128 → 128 → 117 (= 9 × 13), output reshaped (9, 13), sigmoid.
<1ms CPU forward on a single hand.

Training
--------
Adam + BCE loss on p_make targets from forge.bidding.schema parquet files.
Train / val / test splits from the bidding-results corpus subdirectories.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from arena.hand_metrics import evaluate_trump
from forge.bidding.schema import (
    EVAL_DECLS,
    BID_THRESHOLDS,
    get_pmake_matrix,
    load_file,
)
from gus.model.features import _hand_to_mask

# --------------------------------------------------------------------- #
#  Constants                                                              #
# --------------------------------------------------------------------- #

N_DECLS = len(EVAL_DECLS)    # 9
N_BIDS = len(BID_THRESHOLDS)  # 13
N_TRUMP_FEATS = 5             # per pip-trump
N_PIP_TRUMPS = 7              # pips 0..6

FEATURE_DIM = 28 + N_PIP_TRUMPS * N_TRUMP_FEATS  # 63
OUTPUT_DIM = N_DECLS * N_BIDS                      # 117

# Domino name → ID table (mirrors gus/bidding/evaluate.py)
_DOMINO_NAMES = [f"{a}-{b}" for a in range(7) for b in range(a + 1)]


# --------------------------------------------------------------------- #
#  Hand parsing                                                           #
# --------------------------------------------------------------------- #

def _name_to_id(name: str) -> int:
    a, b = name.split("-")
    hi, lo = max(int(a), int(b)), min(int(a), int(b))
    return _DOMINO_NAMES.index(f"{hi}-{lo}")


def parse_hand_str(hand_str: str) -> tuple[int, ...]:
    """'6-4,5-5,...' → tuple of 7 domino IDs."""
    return tuple(_name_to_id(t.strip()) for t in hand_str.split(",") if t.strip())


# --------------------------------------------------------------------- #
#  Featurization                                                          #
# --------------------------------------------------------------------- #

def featurize_hand(hand: tuple[int, ...]) -> Tensor:
    """28-dim multi-hot + 7×5 trump features → [63] float tensor.

    All values are normalised to roughly [0, 1].
    """
    # 28-dim multi-hot
    mask = _hand_to_mask(list(hand))  # [28] float

    trump_feats: list[float] = []
    for t in range(N_PIP_TRUMPS):
        te = evaluate_trump(hand, t)
        trump_feats.extend([
            te.trump_count / 7.0,
            float(te.has_trump_double),
            te.held_count_points / 35.0,
            te.unique_exposed_points / 35.0,
            te.bid_ceiling / 42.0,
        ])

    structured = torch.tensor(trump_feats, dtype=torch.float32)  # [35]
    return torch.cat([mask, structured])  # [63]


# --------------------------------------------------------------------- #
#  Network                                                                #
# --------------------------------------------------------------------- #

class BidNet(nn.Module):
    """Tiny MLP: in→128→128→(9,13) with sigmoid output."""

    def __init__(self, in_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, OUTPUT_DIM),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, in_dim] → [B, 9, 13] sigmoid probabilities."""
        out = self.net(x)                      # [B, 117]
        return torch.sigmoid(out.view(-1, N_DECLS, N_BIDS))


# --------------------------------------------------------------------- #
#  Dataset                                                                #
# --------------------------------------------------------------------- #

class BiddingDataset(torch.utils.data.Dataset):
    """Loads all parquet files from a directory tree, one row each."""

    def __init__(self, parquet_files: list[Path]) -> None:
        self.samples: list[tuple[Tensor, Tensor]] = []
        for path in parquet_files:
            try:
                df, _seed, hand_str = load_file(path)
            except Exception:
                continue
            hand = parse_hand_str(hand_str)
            if len(hand) != 7:
                continue
            x = featurize_hand(hand)
            y = torch.tensor(get_pmake_matrix(df).flatten(), dtype=torch.float32)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.samples[idx]


# --------------------------------------------------------------------- #
#  Training                                                               #
# --------------------------------------------------------------------- #

def train(
    corpus_dir: Path = Path("data/bidding-results"),
    out_model: Path = Path("champion/bid_net.pt"),
    out_metrics: Path = Path("champion/bid_net_metrics.json"),
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """Train BidNet on the corpus, return metrics dict."""
    import glob

    def _load_split(split: str) -> list[Path]:
        return sorted(
            Path(p) for p in glob.glob(
                str(corpus_dir / split / "*.parquet")
            )
        )

    train_files = _load_split("train")
    val_files = _load_split("val")
    test_files = _load_split("test")

    if not train_files:
        raise FileNotFoundError(
            f"No parquet files found in {corpus_dir}/train — "
            "run forge.cli.bidding_continuous first."
        )

    print(f"Dataset: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    train_ds = BiddingDataset(train_files)
    val_ds = BiddingDataset(val_files)
    test_ds = BiddingDataset(test_files)

    print(f"Loaded: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test rows")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    model = BidNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            pred = model(x_b).view(-1, OUTPUT_DIM)
            loss = loss_fn(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_b)

        train_loss = total_loss / max(len(train_ds), 1)

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_loss = _eval_loss(model, val_ds, device, loss_fn)
            print(
                f"  epoch {epoch:3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}",
                flush=True,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- test evaluation ----
    model.eval()
    metrics = _compute_metrics(model, test_ds, device)
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["train_rows"] = len(train_ds)
    metrics["val_rows"] = len(val_ds)
    metrics["test_rows"] = len(test_ds)
    metrics["epochs"] = epochs
    metrics["corpus_dir"] = str(corpus_dir)

    # Save model
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "feature_dim": FEATURE_DIM}, out_model)
    print(f"Saved model → {out_model}")

    # Save metrics
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {out_metrics}")

    return metrics


def _eval_loss(model: BidNet, ds: BiddingDataset, device: str, loss_fn) -> float:
    if not ds:
        return float("nan")
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    total = 0.0
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        pred = model(x_b).view(-1, OUTPUT_DIM)
        total += loss_fn(pred, y_b).item() * len(x_b)
    return total / len(ds)


def _compute_metrics(model: BidNet, ds: BiddingDataset, device: str) -> dict:
    """Per-cell MAE, ECE, calibration verdict."""
    if not ds:
        return {"test_mae": float("nan"), "ece": float("nan")}

    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    with torch.no_grad():
        for x_b, y_b in loader:
            pred = model(x_b.to(device)).cpu().numpy()   # [B, 9, 13]
            true = y_b.numpy().reshape(-1, N_DECLS, N_BIDS)
            all_pred.append(pred)
            all_true.append(true)

    preds = np.concatenate(all_pred, axis=0)  # [N, 9, 13]
    trues = np.concatenate(all_true, axis=0)

    mae_grid = np.abs(preds - trues).mean(axis=0)  # [9, 13]
    overall_mae = float(mae_grid.mean())

    # Calibration: bin predicted p_make into 10 buckets, compare to empirical mean
    flat_pred = preds.flatten()
    flat_true = trues.flatten()
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_pred_mean: list[float] = []
    bin_true_mean: list[float] = []
    bin_counts: list[int] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (flat_pred >= lo) & (flat_pred < hi)
        if mask.sum() == 0:
            continue
        bin_pred_mean.append(float(flat_pred[mask].mean()))
        bin_true_mean.append(float(flat_true[mask].mean()))
        bin_counts.append(int(mask.sum()))

    # ECE: weighted mean |predicted - empirical| per bin
    total = sum(bin_counts)
    ece = float(sum(
        cnt * abs(p - t) / total
        for p, t, cnt in zip(bin_pred_mean, bin_true_mean, bin_counts)
    )) if total > 0 else float("nan")

    # Per-(decl, bid) MAE table as nested list (serialisable)
    mae_table = mae_grid.tolist()

    # Calibration verdict
    if ece < 0.04:
        calib_verdict = "well-calibrated"
    elif ece < 0.08:
        calib_verdict = "acceptable"
    else:
        calib_verdict = "poorly-calibrated"

    return {
        "test_mae": overall_mae,
        "ece": ece,
        "calibration_verdict": calib_verdict,
        "calibration_bins": {
            "bin_pred_mean": bin_pred_mean,
            "bin_true_mean": bin_true_mean,
            "bin_counts": bin_counts,
        },
        "mae_per_decl_bid": mae_table,
        "eval_decls": EVAL_DECLS,
        "bid_thresholds": BID_THRESHOLDS,
    }


# --------------------------------------------------------------------- #
#  CLI entry point                                                        #
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train BidNet (hand -> p_make table)")
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/bidding-results"))
    ap.add_argument("--out-model", type=Path, default=Path("champion/bid_net.pt"))
    ap.add_argument("--out-metrics", type=Path, default=Path("champion/bid_net_metrics.json"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    metrics = train(
        corpus_dir=args.corpus_dir,
        out_model=args.out_model,
        out_metrics=args.out_metrics,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    print(f"\nTest MAE:  {metrics['test_mae']:.4f}")
    print(f"ECE:       {metrics['ece']:.4f}")
    print(f"Calib:     {metrics['calibration_verdict']}")
