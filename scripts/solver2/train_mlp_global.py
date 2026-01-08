#!/usr/bin/env python3
"""
Value MLP Training - Step 2 (Cross-Seed Generalization)

Train an MLP on global encoding (240 features) and evaluate on held-out seeds.
Target: test loss < 0.02, test MAE < 2 points

Usage:
    python scripts/solver2/train_mlp_global.py
    python scripts/solver2/train_mlp_global.py --epochs 50 --batch-size 65536
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


class ValueMLP(nn.Module):
    """MLP for value prediction with global encoding."""

    def __init__(
        self,
        input_dim: int = 240,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_epoch(
    model: ValueMLP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    n_batches = len(train_loader)

    t0 = time.time()
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(X_batch).squeeze()
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = len(X_batch)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            log(f"  Epoch {epoch}/{total_epochs} batch {batch_idx+1}/{n_batches} ({elapsed:.1f}s)")

    return total_loss / total_samples


def evaluate(
    model: ValueMLP,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model, return (mse_loss, mae)."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            pred = model(X_batch).squeeze()

            batch_size = len(X_batch)
            total_mse += loss_fn(pred, y_batch).item() * batch_size
            total_mae += torch.abs(pred - y_batch).sum().item()
            total_samples += batch_size

    mse = total_mse / total_samples
    mae = total_mae / total_samples * 42  # Denormalize MAE to points
    return mse, mae


def spot_check(
    model: ValueMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    n: int = 20,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Print side-by-side predictions vs ground truth."""
    model.eval()
    indices = np.random.choice(len(X), size=min(n, len(X)), replace=False)

    log("\n" + "=" * 50)
    log("Spot Check (random samples)")
    log("=" * 50)
    log(f"{'DP Value':>10} | {'MLP Value':>10} | {'Error':>6}")
    log("-" * 35)

    total_error = 0.0
    with torch.no_grad():
        for idx in indices:
            x = X[idx:idx+1].to(device)
            true_val = y[idx].item() * 42  # denormalize
            pred_val = model(x).item() * 42
            error = abs(true_val - pred_val)
            total_error += error
            log(f"{true_val:+10.1f} | {pred_val:+10.1f} | {error:6.2f}")

    log("-" * 35)
    log(f"Mean absolute error: {total_error / len(indices):.2f} points")


def load_features_from_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load features and values from parquet file with list-encoded features."""
    log(f"Loading {path}...")
    t0 = time.time()

    df = pd.read_parquet(path)
    n = len(df)
    log(f"  Loaded {n:,} rows ({time.time() - t0:.1f}s)")

    # Features are stored as lists, need to stack them
    t0 = time.time()
    features = np.stack(df["features"].values)
    values = df["value"].values.astype(np.float32)
    log(f"  Stacked features: shape {features.shape} ({time.time() - t0:.1f}s)")

    return features, values


def main():
    parser = argparse.ArgumentParser(description="Train Value MLP on global encoding")
    parser.add_argument("--train-file", type=str, default="data/solver2/train_global.parquet")
    parser.add_argument("--test-file", type=str, default="data/solver2/test_global.parquet")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=65536, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split from train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hidden-dims", type=str, default="256,128,64", help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    args = parser.parse_args()

    # Parse hidden dimensions
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        log(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load data
    log("\n=== Phase 1: Loading Data ===")
    train_path = Path(args.train_file)
    test_path = Path(args.test_file)

    if not train_path.exists():
        log(f"ERROR: Train file not found: {train_path}")
        log("Run preprocess_global.py first!")
        sys.exit(1)

    X_train_full, y_train_full = load_features_from_parquet(train_path)
    X_test, y_test = load_features_from_parquet(test_path)

    log(f"\nMemory: {get_memory_mb():.1f} MB")

    # Split train into train/val
    log("\n=== Phase 2: Preparing Data ===")
    n_train = len(X_train_full)
    indices = np.random.permutation(n_train)
    split_idx = int(n_train * (1 - args.val_split))

    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    X_train = torch.tensor(X_train_full[train_idx])
    y_train = torch.tensor(y_train_full[train_idx])
    X_val = torch.tensor(X_train_full[val_idx])
    y_val = torch.tensor(y_train_full[val_idx])
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    log(f"Train: {len(X_train):,} samples")
    log(f"Val: {len(X_val):,} samples (from train seeds)")
    log(f"Test: {len(X_test_t):,} samples (held-out seeds)")

    # Free numpy arrays
    del X_train_full, y_train_full, X_test, y_test
    import gc
    gc.collect()

    # Create data loaders
    num_workers = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # Create model
    log("\n=== Phase 3: Model Setup ===")
    input_dim = X_train.shape[1]
    model = ValueMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=args.dropout)
    param_count = sum(p.numel() for p in model.parameters())
    log(f"Input dim: {input_dim}")
    log(f"Hidden dims: {hidden_dims}")
    log(f"Dropout: {args.dropout}")
    log(f"Weight decay: {args.weight_decay}")
    log(f"Parameters: {param_count:,}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()

    # Training loop
    log(f"\n=== Phase 4: Training ({args.epochs} epochs) ===")
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, args.epochs
        )

        # Validate
        val_loss, val_mae = evaluate(model, val_loader, loss_fn, device)

        # Test (on held-out seeds)
        test_loss, test_mae = evaluate(model, test_loader, loss_fn, device)

        elapsed = time.time() - t0

        log(
            f"Epoch {epoch:2d}: "
            f"train={train_loss:.6f}, val={val_loss:.6f} (MAE {val_mae:.2f}), "
            f"test={test_loss:.6f} (MAE {test_mae:.2f}) [{elapsed:.1f}s]"
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model for final evaluation
    log("\n=== Phase 5: Final Evaluation ===")
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    final_val_loss, final_val_mae = evaluate(model, val_loader, loss_fn, device)
    final_test_loss, final_test_mae = evaluate(model, test_loader, loss_fn, device)

    log(f"Best validation loss: {final_val_loss:.6f}, MAE: {final_val_mae:.2f} points")
    log(f"Test loss (held-out seeds): {final_test_loss:.6f}, MAE: {final_test_mae:.2f} points")

    # Check success criteria
    log("\n=== Success Criteria ===")
    test_pass = final_test_loss < 0.02
    mae_pass = final_test_mae < 2.0
    log(f"Test loss < 0.02: {'PASS' if test_pass else 'FAIL'} ({final_test_loss:.4f})")
    log(f"Test MAE < 2 points: {'PASS' if mae_pass else 'FAIL'} ({final_test_mae:.2f})")

    # Spot check
    spot_check(model, X_test_t, y_test_t, n=20, device=device)

    # Save model
    output_path = Path("data/solver2/value_mlp_global.pt")
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "val_loss": final_val_loss,
        "test_loss": final_test_loss,
        "test_mae": final_test_mae,
    }, output_path)
    log(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
