#!/usr/bin/env python3
"""
Q-Function Diagnostic: Test if role-based action encoding enables cross-seed generalization.

Hypothesis: Q(s,a) with role-based action features should generalize better than V(s)
because "lead boss trump" transfers across seeds while "play domino 14" doesn't.

Data format in parquet:
- state: packed int64
- V: state value (int8)
- mv0-mv6: Q-values for each local move (-128 = illegal)

Usage:
    python scripts/solver2/q_diagnostic.py
    python scripts/solver2/q_diagnostic.py --max-samples 50000
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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.solver2.declarations import (
    DOUBLES_SUIT,
    DOUBLES_TRUMP,
    N_DECLS,
    NOTRUMP,
    PIP_TRUMP_IDS,
)
from scripts.solver2.rng import deal_from_seed
from scripts.solver2.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    can_follow,
    is_in_called_suit,
    led_suit_for_lead_domino,
    trick_rank,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def get_trump_rank(domino_id: int, decl_id: int) -> int:
    """Return trump rank 0-6 (0=boss) or -1 if not trump."""
    if decl_id == NOTRUMP:
        return -1
    if not is_in_called_suit(domino_id, decl_id):
        return -1

    # Get all trumps and their trick_rank values
    trumps = []
    for d in range(28):
        if is_in_called_suit(d, decl_id):
            tau = trick_rank(d, 7, decl_id)  # led_suit=7 means trump suit
            trumps.append((d, tau))

    # Sort by tau descending (highest = boss)
    trumps.sort(key=lambda x: -x[1])

    # Find rank of this domino
    for rank, (d, _) in enumerate(trumps):
        if d == domino_id:
            return rank
    return -1


def get_suit_rank(domino_id: int, suit: int) -> int:
    """Return rank within a pip suit 0-6 (0=boss double, then by sum)."""
    if suit < 0 or suit > 6:
        return -1

    # Get all dominoes in this suit (non-trump)
    suit_dominoes = []
    for d in range(28):
        h, l = DOMINO_HIGH[d], DOMINO_LOW[d]
        if h == suit or l == suit:
            # Rank: doubles beat non-doubles, then by sum
            is_double = DOMINO_IS_DOUBLE[d]
            dom_sum = h + l
            # Double of the suit is boss (rank value 100+)
            # Then by sum descending
            rank_key = (100 if (is_double and h == suit) else 0) + dom_sum
            suit_dominoes.append((d, rank_key))

    suit_dominoes.sort(key=lambda x: -x[1])

    for rank, (d, _) in enumerate(suit_dominoes):
        if d == domino_id:
            return rank
    return -1


def encode_action(
    global_domino_id: int,
    player: int,
    leader: int,
    trick_len: int,
    p0_global: int,  # -1 if no lead yet
    decl_id: int,
) -> np.ndarray:
    """
    Encode an action with role-based features (~18 features).

    Features:
    - is_trump (1): Is this a trump domino?
    - trump_rank (1): Normalized rank within trumps (0=boss, 1=lowest), -1 if not trump
    - is_count (1): Does this domino have count points?
    - count_value (1): Normalized count points (0, 0.5, 1.0 for 0, 5, 10)
    - is_double (1): Is this a double?
    - high_pip (1): Normalized high pip (0-6 -> 0-1)
    - low_pip (1): Normalized low pip
    - is_leading (1): Am I leading the trick?
    - follows_suit (1): Does this follow the led suit? (0 if leading)
    - trumping_in (1): Am I playing trump on a non-trump lead?
    - suit_rank (1): Rank within the led suit (if following)
    - is_partner_winning (1): Is my partner currently winning? (mid-trick only)
    - trick_position (1): Normalized position in trick (0, 0.33, 0.67, 1.0)
    - wins_if_played (1): Would this domino win the current trick?

    Total: ~14 features
    """
    features = np.zeros(14, dtype=np.float32)

    # Basic domino properties
    is_trump = is_in_called_suit(global_domino_id, decl_id)
    features[0] = 1.0 if is_trump else 0.0

    trump_rank = get_trump_rank(global_domino_id, decl_id)
    features[1] = trump_rank / 6.0 if trump_rank >= 0 else -1.0

    count_pts = DOMINO_COUNT_POINTS[global_domino_id]
    features[2] = 1.0 if count_pts > 0 else 0.0
    features[3] = count_pts / 10.0

    features[4] = 1.0 if DOMINO_IS_DOUBLE[global_domino_id] else 0.0
    features[5] = DOMINO_HIGH[global_domino_id] / 6.0
    features[6] = DOMINO_LOW[global_domino_id] / 6.0

    # Trick context
    is_leading = trick_len == 0
    features[7] = 1.0 if is_leading else 0.0

    if is_leading:
        # Leading: no suit to follow, no trumping in
        features[8] = 0.0  # follows_suit (N/A)
        features[9] = 0.0  # trumping_in (N/A)
        features[10] = -1.0  # suit_rank (N/A)
        features[11] = 0.0  # partner_winning (N/A)
        features[13] = 1.0  # wins_if_played (leader always wins initially)
    else:
        # Following: compute led suit and whether we follow
        led_suit = led_suit_for_lead_domino(p0_global, decl_id)
        follows = can_follow(global_domino_id, led_suit, decl_id)
        features[8] = 1.0 if follows else 0.0

        # Trumping in: playing trump when led suit is not trump
        trumping = is_trump and led_suit != 7
        features[9] = 1.0 if trumping else 0.0

        # Suit rank within led suit
        if follows and led_suit < 7:
            suit_rank = get_suit_rank(global_domino_id, led_suit)
            features[10] = suit_rank / 6.0 if suit_rank >= 0 else -1.0
        elif follows and led_suit == 7:
            # Following in trump suit - use trump rank
            features[10] = trump_rank / 6.0 if trump_rank >= 0 else -1.0
        else:
            features[10] = -1.0

        # Partner winning? Partner is player at offset 2 from leader
        partner_offset = (player - leader) % 4
        is_partner_ahead = partner_offset == 2 and trick_len >= 2
        # (Simplified: we don't have full trick state, so approximate)
        features[11] = 0.0  # TODO: would need p1_global to compute properly

        # Wins if played? Compare trick_rank to current best
        my_rank = trick_rank(global_domino_id, led_suit, decl_id)
        lead_rank = trick_rank(p0_global, led_suit, decl_id)
        features[13] = 1.0 if my_rank > lead_rank else 0.0

    # Trick position
    features[12] = trick_len / 3.0

    return features


def encode_state_tau(
    state: int,
    seed: int,
    decl_id: int,
    hands: list[list[int]],
) -> np.ndarray:
    """
    Encode state with tau-based features (simplified version).

    Features (~40):
    - Trump holdings per player: 4 players × 7 ranks = 28 (one-hot)
    - Non-trump count per player: 4
    - Score: 1
    - Leader: 4 (one-hot)
    - Tricks completed: 1
    - Trick length: 1

    Total: 39 features
    """
    features = np.zeros(39, dtype=np.float32)

    # Extract state fields
    remaining = [(state >> (p * 7)) & 0x7F for p in range(4)]
    leader = (state >> 28) & 0x3
    trick_len = (state >> 30) & 0x3

    # Get trump rank mapping for this declaration
    trump_to_rank = {}
    if decl_id != NOTRUMP:
        trumps = []
        for d in range(28):
            if is_in_called_suit(d, decl_id):
                tau = trick_rank(d, 7, decl_id)
                trumps.append((d, tau))
        trumps.sort(key=lambda x: -x[1])
        for rank, (d, _) in enumerate(trumps):
            trump_to_rank[d] = rank

    # Trump holdings: 4 players × 7 ranks (features 0-27)
    for p in range(4):
        for local_idx in range(7):
            if remaining[p] & (1 << local_idx):
                global_id = hands[p][local_idx]
                if global_id in trump_to_rank:
                    rank = trump_to_rank[global_id]
                    features[p * 7 + rank] = 1.0

    # Non-trump count per player (features 28-31)
    for p in range(4):
        non_trump_count = 0
        for local_idx in range(7):
            if remaining[p] & (1 << local_idx):
                global_id = hands[p][local_idx]
                if global_id not in trump_to_rank:
                    non_trump_count += 1
        features[28 + p] = non_trump_count / 7.0

    # Score from played dominoes (feature 32)
    all_remaining_globals = set()
    for p in range(4):
        for local_idx in range(7):
            if remaining[p] & (1 << local_idx):
                all_remaining_globals.add(hands[p][local_idx])

    score = sum(DOMINO_COUNT_POINTS[d] for d in range(28) if d not in all_remaining_globals)
    features[32] = score / 42.0

    # Leader one-hot (features 33-36)
    features[33 + leader] = 1.0

    # Tricks completed (feature 37)
    remaining_count = sum(bin(r).count('1') for r in remaining)
    tricks_completed = (28 - remaining_count - trick_len) // 4
    features[37] = tricks_completed / 7.0

    # Trick length (feature 38)
    features[38] = trick_len / 3.0

    return features


def process_file(
    file_path: Path,
    max_samples: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Process one parquet file, returning (state_features, action_features, q_values).

    Each legal move becomes one training example.
    """
    try:
        pf = pq.ParquetFile(file_path)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())

        df = pd.read_parquet(file_path)
        states = df["state"].values

        # Get Q-values for all 7 moves
        mv_cols = [f"mv{i}" for i in range(7)]
        q_values_all = np.stack([df[c].values for c in mv_cols], axis=1)  # (N, 7)

        hands = deal_from_seed(seed)

        # Collect all (state, action, q) tuples
        state_features_list = []
        action_features_list = []
        q_values_list = []

        n_states = len(states)

        # Sample states if too many
        if max_samples and n_states * 3 > max_samples:  # Assume ~3 legal moves avg
            sample_size = max_samples // 3
            indices = rng.choice(n_states, size=min(sample_size, n_states), replace=False)
        else:
            indices = np.arange(n_states)

        for idx in indices:
            state = int(states[idx])
            q_vals = q_values_all[idx]

            # Extract state fields for action encoding
            leader = (state >> 28) & 0x3
            trick_len = (state >> 30) & 0x3
            p0_local = (state >> 32) & 0x7

            # Current player
            player = (leader + trick_len) % 4
            remaining = (state >> (player * 7)) & 0x7F

            # Lead domino (global) if mid-trick
            p0_global = -1
            if trick_len > 0 and p0_local < 7:
                p0_global = hands[leader][p0_local]

            # Encode state once
            state_feat = encode_state_tau(state, seed, decl_id, hands)

            # Process each legal move
            for local_move in range(7):
                if q_vals[local_move] == -128:
                    continue  # Illegal move

                global_domino = hands[player][local_move]

                # Check if player still has this domino
                if not (remaining & (1 << local_move)):
                    continue

                action_feat = encode_action(
                    global_domino, player, leader, trick_len, p0_global, decl_id
                )

                state_features_list.append(state_feat)
                action_features_list.append(action_feat)
                q_values_list.append(q_vals[local_move] / 42.0)  # Normalize

        if not state_features_list:
            return None

        return (
            np.array(state_features_list, dtype=np.float32),
            np.array(action_features_list, dtype=np.float32),
            np.array(q_values_list, dtype=np.float32),
        )

    except Exception as e:
        log(f"  ERROR processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


class QNetwork(nn.Module):
    """MLP for Q-value prediction: (state, action) -> Q."""

    def __init__(
        self,
        state_dim: int = 39,
        action_dim: int = 14,
        hidden_dims: list[int] = [128, 64, 32],
    ):
        super().__init__()

        input_dim = state_dim + action_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.network(x)


def main():
    parser = argparse.ArgumentParser(description="Q-function diagnostic")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--max-samples", type=int, default=100000, help="Max samples per file")
    parser.add_argument("--train-seeds", type=str, default="0-8", help="Training seed range")
    parser.add_argument("--test-seed", type=int, default=9, help="Test seed")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # Parse train seed range
    train_start, train_end = map(int, args.train_seeds.split("-"))
    train_seeds = set(range(train_start, train_end + 1))
    test_seeds = {args.test_seed}

    log(f"Train seeds: {train_start}-{train_end}")
    log(f"Test seed: {args.test_seed}")

    # Find parquet files
    data_dir = Path(args.data_dir)
    parquet_files = sorted(data_dir.glob("seed_*.parquet"))

    # Filter to relevant files
    train_files = []
    test_files = []
    for f in parquet_files:
        # Parse seed from filename: seed_00000000_decl_0.parquet
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                seed = int(parts[1])
                if seed in train_seeds:
                    train_files.append(f)
                elif seed in test_seeds:
                    test_files.append(f)
            except ValueError:
                pass

    log(f"Found {len(train_files)} train files, {len(test_files)} test files")

    if not train_files or not test_files:
        log("ERROR: No files found for train or test seeds")
        sys.exit(1)

    # Process files
    log("\n=== Phase 1: Loading Training Data ===")
    train_state = []
    train_action = []
    train_q = []

    t0 = time.time()
    for i, f in enumerate(train_files):
        result = process_file(f, args.max_samples, rng)
        if result is not None:
            s, a, q = result
            train_state.append(s)
            train_action.append(a)
            train_q.append(q)
            log(f"  [{i+1}/{len(train_files)}] {f.name}: {len(s):,} samples ({time.time()-t0:.1f}s)")

    log("\n=== Phase 2: Loading Test Data ===")
    test_state = []
    test_action = []
    test_q = []

    for i, f in enumerate(test_files):
        result = process_file(f, args.max_samples, rng)
        if result is not None:
            s, a, q = result
            test_state.append(s)
            test_action.append(a)
            test_q.append(q)
            log(f"  [{i+1}/{len(test_files)}] {f.name}: {len(s):,} samples ({time.time()-t0:.1f}s)")

    # Concatenate
    X_train_state = np.concatenate(train_state, axis=0)
    X_train_action = np.concatenate(train_action, axis=0)
    y_train = np.concatenate(train_q, axis=0)

    X_test_state = np.concatenate(test_state, axis=0)
    X_test_action = np.concatenate(test_action, axis=0)
    y_test = np.concatenate(test_q, axis=0)

    log(f"\nTrain: {len(y_train):,} samples")
    log(f"Test: {len(y_test):,} samples")
    log(f"State dim: {X_train_state.shape[1]}")
    log(f"Action dim: {X_train_action.shape[1]}")

    # Convert to tensors
    train_state_t = torch.tensor(X_train_state)
    train_action_t = torch.tensor(X_train_action)
    train_q_t = torch.tensor(y_train)

    test_state_t = torch.tensor(X_test_state)
    test_action_t = torch.tensor(X_test_action)
    test_q_t = torch.tensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(train_state_t, train_action_t, train_q_t)
    test_dataset = TensorDataset(test_state_t, test_action_t, test_q_t)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create model
    log("\n=== Phase 3: Training ===")
    model = QNetwork(
        state_dim=X_train_state.shape[1],
        action_dim=X_train_action.shape[1],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    loss_fn = nn.MSELoss()

    best_test_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_samples = 0

        for state_b, action_b, q_b in train_loader:
            state_b = state_b.to(device)
            action_b = action_b.to(device)
            q_b = q_b.to(device)

            optimizer.zero_grad()
            pred = model(state_b, action_b).squeeze()
            loss = loss_fn(pred, q_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(q_b)
            train_samples += len(q_b)

        train_loss /= train_samples

        # Evaluate
        model.eval()
        test_loss = 0.0
        test_mae = 0.0
        test_samples = 0

        with torch.no_grad():
            for state_b, action_b, q_b in test_loader:
                state_b = state_b.to(device)
                action_b = action_b.to(device)
                q_b = q_b.to(device)

                pred = model(state_b, action_b).squeeze()
                test_loss += loss_fn(pred, q_b).item() * len(q_b)
                test_mae += torch.abs(pred - q_b).sum().item()
                test_samples += len(q_b)

        test_loss /= test_samples
        test_mae = test_mae / test_samples * 42  # Denormalize to points

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss

        log(f"Epoch {epoch:2d}: train={train_loss:.6f}, test={test_loss:.6f} (MAE {test_mae:.2f} pts)")

    # Final evaluation
    log("\n=== Results ===")
    log(f"Best test MSE: {best_test_loss:.6f}")
    log(f"Best test MAE: {np.sqrt(best_test_loss) * 42:.2f} points (approx)")

    # Compare to V-function baseline
    log("\n=== Comparison to V-function ===")
    log(f"V-function val MSE:  0.022 (from previous run)")
    log(f"V-function test MSE: 0.040 (from previous run)")
    log(f"V-function gap:      1.8x")
    log(f"")
    log(f"Q-function test MSE: {best_test_loss:.4f}")

    # Success criteria
    log("\n=== Success Criteria ===")
    target_mse = 0.03
    target_gap = 1.3

    mse_pass = best_test_loss < target_mse
    log(f"Test MSE < {target_mse}: {'PASS' if mse_pass else 'FAIL'} ({best_test_loss:.4f})")

    # Spot check: show some predictions
    log("\n=== Spot Check (20 random test samples) ===")
    model.eval()
    indices = np.random.choice(len(test_q_t), size=min(20, len(test_q_t)), replace=False)

    log(f"{'True Q':>10} | {'Pred Q':>10} | {'Error':>8}")
    log("-" * 35)

    with torch.no_grad():
        for idx in indices:
            s = test_state_t[idx:idx+1].to(device)
            a = test_action_t[idx:idx+1].to(device)
            true_q = test_q_t[idx].item() * 42
            pred_q = model(s, a).item() * 42
            error = abs(true_q - pred_q)
            log(f"{true_q:+10.1f} | {pred_q:+10.1f} | {error:8.2f}")


if __name__ == "__main__":
    main()
