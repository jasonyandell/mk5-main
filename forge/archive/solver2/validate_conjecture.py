#!/usr/bin/env python3
"""
Validate the three-axis decomposition conjecture from docs/theory/SUIT_STRENGTH.md.

This script tests whether game value decomposes as:
    Φ(σ) ≈ Φ_rank + Φ_void + Φ_control + ε

Uses GPU acceleration via PyTorch for feature extraction and regression.

The experimental plan from §8:
- Phase 1: Void Lattice Alone (variance ratio test)
- Phase 2: Rank Features Alone (linear model R² test)
- Phase 3: Control Features Alone
- Phase 4: Combined Model

Success criteria from Appendix B:
- Conjecture 1 (Void Sufficiency): within-group variance < 10% of total variance
- Conjecture 2 (Linear Decomposition): R² > 0.95
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.solver2.context import build_context
from scripts.solver2.declarations import DECL_ID_TO_NAME, has_trump_power
from scripts.solver2.state import unpack_remaining, compute_level
from scripts.solver2.tables import can_follow, trick_rank, N_DOMINOES


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Precompute lookup tables for can_follow and trick_rank
def build_can_follow_table(decl_id: int) -> torch.Tensor:
    """Build (28, 8) bool table: can_follow[domino_id, led_suit]."""
    table = torch.zeros((N_DOMINOES, 8), dtype=torch.bool)
    for d in range(N_DOMINOES):
        for s in range(8):
            table[d, s] = can_follow(d, s, decl_id)
    return table


def build_trick_rank_table(decl_id: int) -> torch.Tensor:
    """Build (28, 8) int8 table: trick_rank[domino_id, led_suit]."""
    table = torch.zeros((N_DOMINOES, 8), dtype=torch.int8)
    for d in range(N_DOMINOES):
        for s in range(8):
            table[d, s] = trick_rank(d, s, decl_id)
    return table


@torch.no_grad()
def extract_features_gpu(
    states: torch.Tensor,  # (N,) int64
    L: torch.Tensor,  # (4, 7) int8 - local to global domino mapping
    decl_id: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract features from packed states using GPU.

    Returns: (N, F) float32 feature matrix
    """
    n = states.shape[0]
    states = states.to(device)
    L = L.to(device)

    # Build lookup tables
    can_follow_table = build_can_follow_table(decl_id).to(device)
    trick_rank_table = build_trick_rank_table(decl_id).to(device)

    # Unpack remaining masks: (N, 4) each 7-bit
    remaining = unpack_remaining(states)  # (N, 4) int64

    # Extract leader and trick_len
    leader = ((states >> 28) & 0x3).to(torch.int8)  # (N,)
    trick_len = ((states >> 30) & 0x3).to(torch.int8)  # (N,)

    # Compute level (total dominoes remaining)
    level = compute_level(states).float()  # (N,)

    # Leader team (0 or 1)
    leader_team = (leader & 1).float()  # (N,)

    # Expand remaining to per-domino presence: (N, 4, 7) bool
    remaining_expanded = torch.zeros((n, 4, 7), dtype=torch.bool, device=device)
    for local_idx in range(7):
        remaining_expanded[:, :, local_idx] = (remaining & (1 << local_idx)) != 0

    # Get global domino IDs: (N, 4, 7)
    # L is (4, 7), broadcast to (N, 4, 7)
    L_expanded = L.unsqueeze(0).expand(n, -1, -1)  # (N, 4, 7)

    # Compute void patterns: for each (seat, led_suit), is seat void?
    # void[n, seat, suit] = True if no remaining domino can follow suit
    void_pattern = torch.zeros((n, 4, 8), dtype=torch.bool, device=device)

    for seat in range(4):
        for led_suit in range(8):
            # For each local_idx, check if that domino can follow
            can_follow_mask = torch.zeros((n, 7), dtype=torch.bool, device=device)
            for local_idx in range(7):
                domino_ids = L[seat, local_idx]  # scalar
                can_follow_mask[:, local_idx] = can_follow_table[domino_ids, led_suit]

            # Seat is void if no remaining domino can follow
            has_follower = (remaining_expanded[:, seat, :] & can_follow_mask).any(dim=1)
            void_pattern[:, seat, led_suit] = ~has_follower

    # Void counts per seat: (N, 4)
    void_counts = void_pattern.sum(dim=2).float()

    # Rank features: for each seat, sum of trick_rank across suits
    # Also max rank per suit
    rank_sums = torch.zeros((n, 4), dtype=torch.float32, device=device)
    max_ranks = torch.zeros((n, 4, 8), dtype=torch.float32, device=device)

    for seat in range(4):
        for led_suit in range(8):
            for local_idx in range(7):
                domino_id = L[seat, local_idx]
                rank = trick_rank_table[domino_id, led_suit].float()

                # Only count if domino is remaining
                mask = remaining_expanded[:, seat, local_idx]
                rank_sums[:, seat] += mask.float() * rank

                # Update max
                current_max = max_ranks[:, seat, led_suit]
                max_ranks[:, seat, led_suit] = torch.where(
                    mask & (rank > current_max), rank, current_max
                )

    # Team differentials
    # Team 0 = seats 0, 2; Team 1 = seats 1, 3
    rank_diff = (rank_sums[:, 0] + rank_sums[:, 2]) - (rank_sums[:, 1] + rank_sums[:, 3])
    void_diff = (void_counts[:, 0] + void_counts[:, 2]) - (void_counts[:, 1] + void_counts[:, 3])

    # Max rank differential per suit
    max_rank_team0 = (max_ranks[:, 0, :] + max_ranks[:, 2, :]) / 2  # (N, 8)
    max_rank_team1 = (max_ranks[:, 1, :] + max_ranks[:, 3, :]) / 2  # (N, 8)
    max_rank_diff = max_rank_team0 - max_rank_team1  # (N, 8)

    # Build feature matrix - carefully avoiding duplicates and high correlations
    # Only use 7 suits for max_rank_diff (exclude suit 7 which duplicates suit 0 for some decls)
    features = torch.cat([
        # Rank axis (5 features)
        rank_diff.unsqueeze(1) / 100.0,  # normalized rank differential
        max_rank_diff[:, :7] / 30.0,     # max rank diff per suit (7 suits, normalized)

        # Void axis (4 features)
        void_diff.unsqueeze(1) / 8.0,    # void differential
        void_counts / 8.0,               # 4 void counts per seat

        # Control axis (2 features)
        leader_team.unsqueeze(1),        # binary: which team leads
        level.unsqueeze(1) / 28.0,       # normalized level
    ], dim=1)

    return features  # (N, 15)


class LinearModel(nn.Module):
    """Simple linear regression model."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPModel(nn.Module):
    """MLP for nonlinear regression."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 4096,
    lr: float = 0.01,
) -> float:
    """Train model and return R² on full data."""
    device = X.device
    model = model.to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    # Compute R²
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot

    return r2.item()


def compute_r2(X: torch.Tensor, y: torch.Tensor) -> float:
    """Compute R² using closed-form linear regression."""
    # Filter out constant columns
    stds = X.std(dim=0)
    non_const_mask = stds > 1e-6
    if non_const_mask.sum() == 0:
        return 0.0
    X = X[:, non_const_mask]

    # Standardize features for numerical stability
    means = X.mean(dim=0)
    stds = X.std(dim=0)
    stds = torch.where(stds > 1e-6, stds, torch.ones_like(stds))
    X_scaled = (X - means) / stds

    # Add bias term
    ones = torch.ones((X_scaled.shape[0], 1), device=X_scaled.device, dtype=X_scaled.dtype)
    X_aug = torch.cat([ones, X_scaled], dim=1)

    # Normal equations with regularization
    XtX = X_aug.T @ X_aug
    XtX += 1e-4 * torch.eye(XtX.shape[0], device=XtX.device)
    Xty = X_aug.T @ y

    try:
        coeffs = torch.linalg.solve(XtX, Xty)
    except Exception:
        return 0.0

    y_pred = X_aug @ coeffs
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()

    if ss_tot < 1e-10:
        return 0.0

    return (1 - ss_res / ss_tot).item()


def phase1_void_sufficiency(
    features: torch.Tensor,
    values: torch.Tensor,
    void_features_idx: list[int],
) -> dict:
    """
    Phase 1: Test if void features alone explain value.

    Use R² of linear model on void features only.
    """
    X_void = features[:, void_features_idx]
    r2 = compute_r2(X_void, values)

    return {
        "r2": r2,
        "n_features": len(void_features_idx),
    }


def phase2_rank_features(
    features: torch.Tensor,
    values: torch.Tensor,
    rank_features_idx: list[int],
) -> dict:
    """
    Phase 2: Test if rank features alone predict value.
    """
    X_rank = features[:, rank_features_idx]
    r2 = compute_r2(X_rank, values)

    return {
        "r2": r2,
        "n_features": len(rank_features_idx),
    }


def phase3_control_features(
    features: torch.Tensor,
    values: torch.Tensor,
    control_features_idx: list[int],
) -> dict:
    """
    Phase 3: Test if control features predict value.
    """
    X_ctrl = features[:, control_features_idx]
    r2 = compute_r2(X_ctrl, values)

    return {
        "r2": r2,
        "n_features": len(control_features_idx),
    }


def phase4_combined_linear(
    features: torch.Tensor,
    values: torch.Tensor,
) -> dict:
    """
    Phase 4: Combined linear model.
    """
    r2 = compute_r2(features, values)

    return {
        "r2": r2,
        "n_features": features.shape[1],
    }


def phase5_combined_mlp(
    features: torch.Tensor,
    values: torch.Tensor,
    device: torch.device,
    epochs: int = 30,
) -> dict:
    """
    Phase 5: MLP model for nonlinear patterns.
    """
    model = MLPModel(features.shape[1], hidden_dim=128)
    r2 = train_model(model, features, values, epochs=epochs, batch_size=4096, lr=0.001)

    return {
        "r2": r2,
        "model": "MLP(128, 128)",
    }


def load_parquet_data(
    data_dir: Path,
    max_rows: int = 200_000,
    seed_limit: int = 6,
    device: torch.device = torch.device("cpu"),
):
    """Load data from parquet files."""
    files = sorted(data_dir.glob("seed_*.parquet"))[:seed_limit]

    for f in files:
        parts = f.stem.split("_")
        seed = int(parts[1])
        decl_id = int(parts[3])

        table = pq.read_table(f)
        states = torch.tensor(table.column("state").to_numpy(), dtype=torch.int64)
        values = torch.tensor(table.column("V").to_numpy(), dtype=torch.float32)

        # Sample if too large
        if len(states) > max_rows:
            idx = torch.randperm(len(states))[:max_rows]
            states = states[idx]
            values = values[idx]

        yield seed, decl_id, states.to(device), values.to(device)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate three-axis decomposition conjecture")
    parser.add_argument("--max-rows", type=int, default=50_000, help="Max rows per file")
    parser.add_argument("--seed-limit", type=int, default=6, help="Max number of seed files")
    parser.add_argument("--epochs", type=int, default=30, help="MLP training epochs")
    args = parser.parse_args()

    data_dir = Path("data/solver2")
    if not data_dir.exists():
        print(f"Error: {data_dir} not found")
        sys.exit(1)

    device = get_device()
    print("=" * 70)
    print("Three-Axis Decomposition Conjecture Validation")
    print(f"Device: {device}")
    print(f"Settings: max_rows={args.max_rows}, seed_limit={args.seed_limit}, epochs={args.epochs}")
    print("=" * 70)
    print()

    # Feature indices (based on extract_features_gpu output, 18 features total)
    # [0]: rank_diff, [1-7]: max_rank_diff (7 suits), [8]: void_diff, [9-12]: void_counts,
    # [13]: leader_team, [14]: level
    rank_idx = list(range(0, 8))  # rank_diff + 7 max_rank_diff
    void_idx = list(range(8, 13))  # void_diff + 4 void_counts
    control_idx = [13, 14]  # leader_team, level

    all_results = []

    for seed, decl_id, states, values in load_parquet_data(data_dir, max_rows=args.max_rows, seed_limit=args.seed_limit, device=device):
        decl_name = DECL_ID_TO_NAME.get(decl_id, str(decl_id))
        print(f"Processing seed={seed}, decl={decl_name} ({len(states)} states)...")

        # Build context for L matrix
        ctx = build_context(seed, decl_id, device)

        # Extract features
        features = extract_features_gpu(states, ctx.L, decl_id, device)

        # Run phases
        print("    Phase 1 (Void)...", end=" ", flush=True)
        p1 = phase1_void_sufficiency(features, values, void_idx)
        print(f"R²={p1['r2']:.4f}")

        print("    Phase 2 (Rank)...", end=" ", flush=True)
        p2 = phase2_rank_features(features, values, rank_idx)
        print(f"R²={p2['r2']:.4f}")

        print("    Phase 3 (Control)...", end=" ", flush=True)
        p3 = phase3_control_features(features, values, control_idx)
        print(f"R²={p3['r2']:.4f}")

        print("    Phase 4 (Linear)...", end=" ", flush=True)
        p4 = phase4_combined_linear(features, values)
        print(f"R²={p4['r2']:.4f}")

        print(f"    Phase 5 (MLP, {args.epochs} epochs)...", end=" ", flush=True)
        p5 = phase5_combined_mlp(features, values, device, epochs=args.epochs)
        print(f"R²={p5['r2']:.4f}")
        print()

        all_results.append({
            "seed": seed,
            "decl": decl_name,
            "p1": p1["r2"],
            "p2": p2["r2"],
            "p3": p3["r2"],
            "p4": p4["r2"],
            "p5": p5["r2"],
        })

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_p1 = np.mean([r["p1"] for r in all_results])
    avg_p2 = np.mean([r["p2"] for r in all_results])
    avg_p3 = np.mean([r["p3"] for r in all_results])
    avg_p4 = np.mean([r["p4"] for r in all_results])
    avg_p5 = np.mean([r["p5"] for r in all_results])

    print()
    print("Average R² across all (seed, decl) pairs:")
    print(f"  Phase 1 (Void features only):    {avg_p1:.4f}")
    print(f"  Phase 2 (Rank features only):    {avg_p2:.4f}")
    print(f"  Phase 3 (Control features only): {avg_p3:.4f}")
    print(f"  Phase 4 (Linear combined):       {avg_p4:.4f}")
    print(f"  Phase 5 (MLP combined):          {avg_p5:.4f}")
    print()

    # Conjecture evaluation
    print("CONJECTURE EVALUATION:")
    print()

    # What does each axis contribute?
    print("Individual axis contribution:")
    print(f"  Rank alone explains:    {avg_p2*100:.1f}% of variance")
    print(f"  Void alone explains:    {avg_p1*100:.1f}% of variance")
    print(f"  Control alone explains: {avg_p3*100:.1f}% of variance")
    print()

    # Linear combination
    max_individual = max(avg_p1, avg_p2, avg_p3)
    linear_gain = avg_p4 - max_individual
    print(f"Linear combination ({avg_p4*100:.1f}%) vs best individual ({max_individual*100:.1f}%):")
    print(f"  Gain from combining: {linear_gain*100:+.1f}%")
    print()

    # Nonlinear patterns
    mlp_gain = avg_p5 - avg_p4
    print(f"MLP ({avg_p5*100:.1f}%) vs Linear ({avg_p4*100:.1f}%):")
    print(f"  Nonlinear contribution: {mlp_gain*100:+.1f}%")
    print()

    # Final assessment
    print("CONCLUSION:")
    if avg_p4 > 0.95:
        print("  ✓ Linear decomposition holds (R² > 0.95)")
        print("  Game value ≈ Φ_rank + Φ_void + Φ_control")
    elif avg_p5 > 0.95:
        print("  ✓ Nonlinear decomposition holds (MLP R² > 0.95)")
        print("  The three axes are sufficient with nonlinear interactions")
    elif avg_p5 > 0.80:
        print(f"  ◐ Partial support: {avg_p5*100:.0f}% explained by three-axis MLP")
        print("  The framework captures most but not all structure")
    else:
        print(f"  ✗ Weak support: only {avg_p5*100:.0f}% explained")
        print("  Additional features or structure may be needed")

    # Dominant axis
    print()
    if avg_p2 > max(avg_p1, avg_p3) + 0.05:
        print("  Dominant axis: RANK (material strength)")
    elif avg_p1 > max(avg_p2, avg_p3) + 0.05:
        print("  Dominant axis: VOID (interrupt structure)")
    elif avg_p3 > max(avg_p1, avg_p2) + 0.05:
        print("  Dominant axis: CONTROL (tempo/position)")
    else:
        print("  No single dominant axis - balanced contribution")

    print()
    return avg_p5 > 0.80


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
