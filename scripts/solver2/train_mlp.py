#!/usr/bin/env python3
"""
Value MLP Training - Step 1 (One Seed Sanity Check)

Train an MLP to predict minimax values from game states.
Uses local indices (71 features) for simplicity in single-seed training.

Usage:
    python scripts/solver2/train_mlp.py --file data/solver2/seed_00000000_decl_0.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def log(msg: str) -> None:
    """Print with flush for real-time output."""
    print(msg, flush=True)


def encode_features(states: np.ndarray, decl_id: int) -> np.ndarray:
    """
    Encode packed states into feature vectors (vectorized).

    Features (71 total):
    - remaining[0-3]: 4 × 7 = 28 (binary masks)
    - leader: 4 (one-hot)
    - trick_len: 4 (one-hot)
    - plays p0, p1, p2: 3 × 8 = 24 (one-hot, including 7=none)
    - decl_id: 10 (one-hot)
    - current_player: 1 (0=team0, 1=team1)

    Total: 28 + 4 + 4 + 24 + 10 + 1 = 71 features
    """
    n = len(states)
    features = np.zeros((n, 71), dtype=np.float32)

    # Remaining masks (28 features) - vectorized bit extraction
    for p in range(4):
        mask = (states >> (p * 7)) & 0x7F
        for bit in range(7):
            features[:, p * 7 + bit] = (mask >> bit) & 1

    # Extract fields once
    leader = (states >> 28) & 0x3
    trick_len = (states >> 30) & 0x3
    p0 = (states >> 32) & 0x7
    p1 = (states >> 35) & 0x7
    p2 = (states >> 38) & 0x7

    # One-hot encoding using advanced indexing (much faster than loops)
    row_idx = np.arange(n)
    features[row_idx, 28 + leader] = 1.0      # leader one-hot
    features[row_idx, 32 + trick_len] = 1.0   # trick_len one-hot
    features[row_idx, 36 + p0] = 1.0          # p0 one-hot
    features[row_idx, 44 + p1] = 1.0          # p1 one-hot
    features[row_idx, 52 + p2] = 1.0          # p2 one-hot

    # Declaration (10 features, one-hot)
    features[:, 60 + decl_id] = 1.0

    # Current player team (1 feature: 0=team0, 1=team1)
    current_player = (leader + trick_len) & 0x3
    features[:, 70] = (current_player & 1).astype(np.float32)

    return features


class ValueMLP(nn.Module):
    """Simple MLP for value prediction."""

    def __init__(self, input_dim: int = 71, hidden_dims: list[int] = [128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train(
    model: ValueMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train the model and print losses."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_batches = len(train_loader)

    for epoch in range(epochs):
        t0 = time.time()
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch).squeeze()
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

            # Progress every 100 batches
            if batch_idx % 100 == 0:
                log(f"  Epoch {epoch+1}/{epochs} batch {batch_idx}/{n_batches}")

        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch).squeeze()
                loss = loss_fn(pred, y_batch)
                val_loss += loss.item() * len(X_batch)

        val_loss /= len(val_loader.dataset)
        elapsed = time.time() - t0

        log(f"Epoch {epoch + 1:2d}: train={train_loss:.6f}, val={val_loss:.6f} ({elapsed:.1f}s)")


def spot_check(
    model: ValueMLP,
    X: torch.Tensor,
    y: torch.Tensor,
    n: int = 20,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Print side-by-side predictions vs ground truth."""
    model.eval()
    indices = np.random.choice(len(X), size=n, replace=False)

    log("\n" + "=" * 50)
    log("Spot Check (20 random samples)")
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
    log(f"Mean absolute error: {total_error / n:.2f} points")


def main():
    parser = argparse.ArgumentParser(description="Train Value MLP on solver2 data")
    parser.add_argument("--file", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=65536, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Load data
    t0 = time.time()
    log(f"\nLoading {args.file}...")
    pf = pq.ParquetFile(args.file)
    meta = pf.schema_arrow.metadata or {}
    decl_id = int(meta.get(b"decl_id", b"0").decode())
    log(f"Declaration ID: {decl_id}")

    df = pd.read_parquet(args.file)
    log(f"Loaded {len(df):,} states ({time.time() - t0:.1f}s)")

    # Encode features
    t0 = time.time()
    log("Encoding features...")
    X = encode_features(df["state"].values, decl_id)
    y = df["V"].values.astype(np.float32) / 42.0  # normalize to [-1, 1]

    log(f"Feature shape: {X.shape} ({time.time() - t0:.1f}s)")
    log(f"Value range: [{y.min():.3f}, {y.max():.3f}]")

    # Shuffle and split
    t0 = time.time()
    log("Splitting data...")
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - args.val_split))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, X_val = torch.tensor(X[train_idx]), torch.tensor(X[val_idx])
    y_train, y_val = torch.tensor(y[train_idx]), torch.tensor(y[val_idx])

    log(f"Train: {len(X_train):,}, Val: {len(X_val):,} ({time.time() - t0:.1f}s)")

    # Create data loaders - use multiple workers and pin_memory for GPU
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

    # Create model
    model = ValueMLP(input_dim=X.shape[1])
    param_count = sum(p.numel() for p in model.parameters())
    log(f"\nModel parameters: {param_count:,}")

    # Train
    log(f"\nTraining for {args.epochs} epochs...\n")
    train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Spot check
    spot_check(model, X_val, y_val, n=20, device=device)

    # Save model
    output_path = Path(args.file).stem + "_mlp.pt"
    torch.save(model.state_dict(), output_path)
    log(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
