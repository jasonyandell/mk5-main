#!/usr/bin/env python3
"""
τ-encoding hypothesis validation diagnostic.

Validates whether τ-encoding (power-rank) enables cross-seed generalization
by comparing nearest neighbor value coherence between raw and τ encodings.

Usage:
    python scripts/solver2/tau_diagnostic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.solver2.tables import (
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    N_DOMINOES,
    trick_rank,
)
from scripts.solver2.declarations import (
    DOUBLES_SUIT,
    DOUBLES_TRUMP,
    NOTRUMP,
    PIP_TRUMP_IDS,
    has_trump_power,
)
from scripts.solver2.rng import deal_from_seed


def log(msg: str) -> None:
    print(msg, flush=True)


def deal_from_seed_local(seed: int) -> list[list[int]]:
    """Return 4 hands of 7 unique domino IDs, deterministically from seed."""
    import random
    rng = random.Random(seed)
    dominos = list(range(28))
    rng.shuffle(dominos)
    hands = [sorted(dominos[i * 7 : (i + 1) * 7]) for i in range(4)]
    return hands


def encode_global_features(
    states: np.ndarray,
    seed: int,
    decl_id: int,
) -> np.ndarray:
    """Encode packed states into 240-dim global feature vectors."""
    n = len(states)
    features = np.zeros((n, 240), dtype=np.float32)

    hands = deal_from_seed_local(seed)

    # Count points lookup
    count_points = np.zeros(28, dtype=np.int32)
    DOMINOES = [(h, l) for h in range(7) for l in range(h + 1)]
    for d_id, (h, l) in enumerate(DOMINOES):
        if (h, l) in ((5, 5), (6, 4)):
            count_points[d_id] = 10
        elif (h, l) in ((5, 0), (4, 1), (3, 2)):
            count_points[d_id] = 5

    # Local to global lookup
    local_to_global = np.zeros((4, 8), dtype=np.int32)
    for p in range(4):
        for local_idx in range(7):
            local_to_global[p, local_idx] = hands[p][local_idx]
        local_to_global[p, 7] = -1

    # Extract packed fields
    remaining_local = np.zeros((n, 4), dtype=np.uint8)
    for p in range(4):
        remaining_local[:, p] = (states >> (p * 7)) & 0x7F

    leader = (states >> 28) & 0x3
    trick_len = (states >> 30) & 0x3
    p0 = (states >> 32) & 0x7
    p1 = (states >> 35) & 0x7
    p2 = (states >> 38) & 0x7

    # Feature 0-111: Global remaining masks (4 players × 28 dominoes)
    for p in range(4):
        for local_idx in range(7):
            global_id = hands[p][local_idx]
            has_domino = (remaining_local[:, p] >> local_idx) & 1
            features[:, p * 28 + global_id] = has_domino.astype(np.float32)

    # Feature 112: Score (normalized)
    all_remaining = np.zeros((n, 28), dtype=np.uint8)
    for p in range(4):
        for local_idx in range(7):
            global_id = hands[p][local_idx]
            has_domino = (remaining_local[:, p] >> local_idx) & 1
            all_remaining[:, global_id] = has_domino
    played = 1 - all_remaining
    score = np.sum(played * count_points[np.newaxis, :], axis=1)
    features[:, 112] = score / 42.0

    # Feature 113-116: Leader (one-hot)
    row_idx = np.arange(n)
    features[row_idx, 113 + leader] = 1.0

    # Feature 117-126: Declaration (one-hot)
    features[:, 117 + decl_id] = 1.0

    # Feature 127-238: Trick plays (4 positions × 28 dominoes)
    for trick_pos, (play_local, seat_offset) in enumerate([(p0, 0), (p1, 1), (p2, 2)]):
        seat = (leader + seat_offset) % 4
        mask = play_local < 7
        valid_indices = np.where(mask)[0]
        if len(valid_indices) > 0:
            local_plays = play_local[valid_indices]
            global_plays = local_to_global[seat[valid_indices], local_plays]
            feature_base = 127 + trick_pos * 28
            features[valid_indices, feature_base + global_plays] = 1.0

    # Feature 239: Tricks completed (normalized)
    remaining_count = np.zeros(n, dtype=np.int32)
    for p in range(4):
        for bit in range(7):
            remaining_count += (remaining_local[:, p] >> bit) & 1
    dominoes_played = 28 - remaining_count
    tricks_completed = (dominoes_played - trick_len) // 4
    features[:, 239] = tricks_completed / 7.0

    return features


def compute_tau_for_domino(domino_id: int, decl_id: int) -> int:
    """
    Compute τ (power rank) for a domino given declaration.

    Returns a value encoding the domino's potential power:
    - Trumps: tier 2 + rank within trump suit (0-14)
    - Non-trumps: tier 1 + rank in their own suit (potential power)

    This makes features seed-invariant: "3rd-highest trump" means the same
    thing regardless of which specific domino that is.
    """
    # Check if this is a trump
    if has_trump_power(decl_id):
        if decl_id in PIP_TRUMP_IDS:
            # Pip trump: contains the trump pip
            high, low = DOMINO_HIGH[domino_id], domino_id - (DOMINO_HIGH[domino_id] * (DOMINO_HIGH[domino_id] + 1)) // 2
            # Recalculate low from domino_id
            h = DOMINO_HIGH[domino_id]
            l = domino_id - (h * (h + 1)) // 2
            if h == decl_id or l == decl_id:
                # It's a trump - get its rank
                return trick_rank(domino_id, 7, decl_id)  # led_suit=7 means trump suit
        elif decl_id == DOUBLES_TRUMP:
            if DOMINO_IS_DOUBLE[domino_id]:
                return trick_rank(domino_id, 7, decl_id)

    # Not a trump - compute potential power in its own suit
    # Use high pip as the "suit" to get rank
    own_suit = DOMINO_HIGH[domino_id]
    return trick_rank(domino_id, own_suit, decl_id)


def get_trump_dominoes_and_ranks(decl_id: int) -> list[tuple[int, int]]:
    """
    Return list of (domino_id, rank) for all trumps in this declaration.
    Rank 0 = boss (highest), rank 6 = lowest trump.
    Returns empty list for no-trump.
    """
    if decl_id == NOTRUMP:
        return []

    trumps = []
    for d in range(28):
        if decl_id in PIP_TRUMP_IDS:
            # Pip trump: domino contains the trump pip
            h, l = DOMINO_HIGH[d], d - (DOMINO_HIGH[d] * (DOMINO_HIGH[d] + 1)) // 2
            if h == decl_id or l == decl_id:
                tau = trick_rank(d, 7, decl_id)  # 7 = trump suit
                trumps.append((d, tau))
        elif decl_id == DOUBLES_TRUMP:
            if DOMINO_IS_DOUBLE[d]:
                tau = trick_rank(d, 7, decl_id)
                trumps.append((d, tau))
        elif decl_id == DOUBLES_SUIT:
            if DOMINO_IS_DOUBLE[d]:
                # Doubles as suit, not trump - use their rank
                tau = trick_rank(d, 7, decl_id)
                trumps.append((d, tau))

    # Sort by τ descending (boss first), assign rank 0-6
    trumps.sort(key=lambda x: -x[1])
    return [(d, rank) for rank, (d, _tau) in enumerate(trumps)]


def encode_tau_features(
    states: np.ndarray,
    seed: int,
    decl_id: int,
) -> np.ndarray:
    """
    Encode packed states into τ-based feature vectors using ONE-HOT encoding.

    For each player, encode which trump ranks (0-6) they hold.
    Rank 0 = boss trump, rank 6 = lowest trump.

    This is seed-invariant: "has boss trump" means the same thing regardless
    of which specific domino is the boss in this seed.

    Features:
    - trump_holdings[0-3]: 4 players × 7 one-hot bits = 28 (has rank X?)
    - non_trump_count[0-3]: 4 players × 1 = 4 (how many non-trumps)
    - score: 1
    - leader: 4 (one-hot)
    - decl_id: 10 (one-hot)
    - trick_plays: 3 × 8 = 24 (one-hot trump rank or 7=non-trump, per play)
    - tricks_completed: 1

    Total: 28 + 4 + 1 + 4 + 10 + 24 + 1 = 72 features
    """
    n = len(states)
    hands = deal_from_seed_local(seed)

    # Build trump lookup: domino_id -> rank (0-6), or -1 if not trump
    trump_info = get_trump_dominoes_and_ranks(decl_id)
    domino_to_trump_rank = {}
    for d, rank in trump_info:
        domino_to_trump_rank[d] = rank

    # Local to global lookup
    local_to_global = np.zeros((4, 8), dtype=np.int32)
    for p in range(4):
        for local_idx in range(7):
            local_to_global[p, local_idx] = hands[p][local_idx]
        local_to_global[p, 7] = -1

    # Extract packed fields
    remaining_local = np.zeros((n, 4), dtype=np.uint8)
    for p in range(4):
        remaining_local[:, p] = (states >> (p * 7)) & 0x7F

    leader = (states >> 28) & 0x3
    trick_len = (states >> 30) & 0x3
    p0 = (states >> 32) & 0x7
    p1 = (states >> 35) & 0x7
    p2 = (states >> 38) & 0x7

    # Features: 72 total
    features = np.zeros((n, 72), dtype=np.float32)

    # Features 0-27: Trump holdings (one-hot per player per rank)
    # Features 28-31: Non-trump count per player
    for p in range(4):
        for i in range(n):
            remaining_mask = remaining_local[i, p]
            non_trump_count = 0
            for local_idx in range(7):
                if (remaining_mask >> local_idx) & 1:
                    global_id = hands[p][local_idx]
                    trump_rank = domino_to_trump_rank.get(global_id, -1)
                    if trump_rank >= 0:
                        # One-hot: player p has trump rank trump_rank
                        features[i, p * 7 + trump_rank] = 1.0
                    else:
                        non_trump_count += 1
            features[i, 28 + p] = non_trump_count / 7.0  # Normalized

    # Feature 32: Score
    count_points = np.zeros(28, dtype=np.int32)
    DOMINOES = [(h, l) for h in range(7) for l in range(h + 1)]
    for d_id, (h, l) in enumerate(DOMINOES):
        if (h, l) in ((5, 5), (6, 4)):
            count_points[d_id] = 10
        elif (h, l) in ((5, 0), (4, 1), (3, 2)):
            count_points[d_id] = 5

    all_remaining = np.zeros((n, 28), dtype=np.uint8)
    for p in range(4):
        for local_idx in range(7):
            global_id = hands[p][local_idx]
            has_domino = (remaining_local[:, p] >> local_idx) & 1
            all_remaining[:, global_id] = has_domino
    played = 1 - all_remaining
    score = np.sum(played * count_points[np.newaxis, :], axis=1)
    features[:, 32] = score / 42.0

    # Feature 33-36: Leader (one-hot)
    row_idx = np.arange(n)
    features[row_idx, 33 + leader] = 1.0

    # Feature 37-46: Declaration (one-hot)
    features[:, 37 + decl_id] = 1.0

    # Feature 47-70: Trick plays (one-hot: 8 options per play = trump rank 0-6 or 7=non-trump)
    for trick_pos, (play_local, seat_offset) in enumerate([(p0, 0), (p1, 1), (p2, 2)]):
        seat = (leader + seat_offset) % 4
        for i in range(n):
            if play_local[i] < 7:
                global_id = local_to_global[seat[i], play_local[i]]
                trump_rank = domino_to_trump_rank.get(global_id, 7)  # 7 = non-trump
                feature_base = 47 + trick_pos * 8
                features[i, feature_base + trump_rank] = 1.0

    # Feature 71: Tricks completed
    remaining_count = np.zeros(n, dtype=np.int32)
    for p in range(4):
        for bit in range(7):
            remaining_count += (remaining_local[:, p] >> bit) & 1
    dominoes_played = 28 - remaining_count
    tricks_completed = (dominoes_played - trick_len) // 4
    features[:, 71] = tricks_completed / 7.0

    return features


class ValueMLP(torch.nn.Module):
    """MLP for value prediction."""

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
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, 1))
        layers.append(torch.nn.Tanh())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def find_k_nearest(query: np.ndarray, database: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors using Euclidean distance.
    Returns (indices, distances).
    """
    # Compute distances
    diffs = database - query[np.newaxis, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))

    # Get k smallest
    indices = np.argpartition(distances, k)[:k]
    indices = indices[np.argsort(distances[indices])]

    return indices, distances[indices]


def main():
    log("=" * 60)
    log("τ-Encoding Hypothesis Validation Diagnostic")
    log("=" * 60)

    data_dir = Path("data/solver2")
    model_path = data_dir / "value_mlp_global.pt"

    # Load trained model
    log("\n=== Loading Model ===")
    log(f"  Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    log(f"  Checkpoint keys: {list(checkpoint.keys())}")
    log(f"  hidden_dims: {checkpoint['hidden_dims']}")

    # Infer dropout from state dict structure
    state_dict = checkpoint["model_state_dict"]
    has_dropout = any("Dropout" in str(type(v)) or ".3." in k for k in state_dict.keys() if "network.7" in k)
    # Count layers to infer dropout - if more layers than expected, dropout was used
    n_layers = max(int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('network.'))
    expected_no_dropout = len(checkpoint["hidden_dims"]) * 3 + 2  # (Linear+ReLU+BN)*n + Linear+Tanh
    dropout = 0.3 if n_layers >= expected_no_dropout else 0.0
    log(f"  Inferred dropout: {dropout} (n_layers={n_layers}, expected_no_dropout={expected_no_dropout})")

    model = ValueMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=dropout,
    )
    log(f"  Loading state dict...")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log(f"  ✓ Model loaded: input_dim={checkpoint['input_dim']}, test_loss={checkpoint['test_loss']:.4f}")

    # Find test seed files (seeds 90-99)
    log("\n=== Loading Test Data ===")
    test_files = sorted(data_dir.glob("seed_0000009*.parquet"))[:10]
    log(f"Found {len(test_files)} test seed files")
    for i, f in enumerate(test_files):
        log(f"  [{i+1}] {f.name}")

    # Load and encode test data
    test_data = []  # List of (seed, decl_id, features_global, features_tau, values, states)

    for file_idx, fpath in enumerate(test_files):
        t0 = time.time()
        log(f"\n  [{file_idx+1}/{len(test_files)}] Processing {fpath.name}...")

        log(f"    Reading parquet metadata...")
        pf = pq.ParquetFile(fpath)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())
        log(f"    seed={seed}, decl_id={decl_id}")

        log(f"    Loading dataframe...")
        df = pd.read_parquet(fpath)
        log(f"    Loaded {len(df)} rows")

        # Sample up to 5000 states per file for speed
        if len(df) > 5000:
            log(f"    Sampling 5000 from {len(df)}...")
            df = df.sample(n=5000, random_state=seed)

        states = df["state"].values
        values = df["V"].values.astype(np.float32)

        log(f"    Encoding global features ({len(states)} states)...")
        features_global = encode_global_features(states, seed, decl_id)
        log(f"    Encoding τ features...")
        features_tau = encode_tau_features(states, seed, decl_id)

        test_data.append({
            "seed": seed,
            "decl_id": decl_id,
            "features_global": features_global,
            "features_tau": features_tau,
            "values": values,
            "states": states,
        })
        log(f"    ✓ Done in {time.time() - t0:.1f}s")

    # Get model predictions on global features
    log("\n=== Getting Model Predictions ===")
    all_predictions = []
    all_true_values = []
    all_indices = []  # (file_idx, state_idx)

    for file_idx, data in enumerate(test_data):
        log(f"  [{file_idx+1}/{len(test_data)}] Predicting for seed {data['seed']} decl {data['decl_id']}...")
        features_t = torch.tensor(data["features_global"])
        with torch.no_grad():
            preds = model(features_t).squeeze().numpy() * 42  # denormalize
        all_predictions.extend(preds)
        all_true_values.extend(data["values"])
        all_indices.extend([(file_idx, i) for i in range(len(preds))])
        log(f"    ✓ {len(preds)} predictions")

    all_predictions = np.array(all_predictions)
    all_true_values = np.array(all_true_values)
    errors = np.abs(all_predictions - all_true_values)

    log(f"\n  Total test samples: {len(errors)}")
    log(f"  Mean error: {np.mean(errors):.2f} points")
    log(f"  Max error: {np.max(errors):.2f} points")
    log(f"  Samples with error > 15: {np.sum(errors > 15)}")

    # Find worst predictions (error > 15 points)
    worst_mask = errors > 15
    worst_indices = np.where(worst_mask)[0]

    if len(worst_indices) == 0:
        log("\nNo predictions with error > 15 points. Lowering threshold to 10...")
        worst_mask = errors > 10
        worst_indices = np.where(worst_mask)[0]

    if len(worst_indices) == 0:
        log("No high-error predictions found. Exiting.")
        return

    # Sample up to 20 worst predictions
    if len(worst_indices) > 20:
        np.random.seed(42)
        worst_indices = np.random.choice(worst_indices, size=20, replace=False)

    log(f"\nAnalyzing {len(worst_indices)} worst predictions...")

    # Load training data for neighbor search
    log("\n=== Loading Training Data for Neighbor Search ===")
    train_files = sorted(data_dir.glob("seed_0000000*.parquet"))[:10]  # First 10 train files
    log(f"Found {len(train_files)} train files to load")
    for i, f in enumerate(train_files):
        log(f"  [{i+1}] {f.name}")

    train_features_global = []
    train_features_tau = []
    train_values = []
    train_meta = []  # (seed, decl_id, idx)

    for file_idx, fpath in enumerate(train_files):
        t0 = time.time()
        log(f"\n  [{file_idx+1}/{len(train_files)}] Processing {fpath.name}...")

        pf = pq.ParquetFile(fpath)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())
        log(f"    seed={seed}, decl_id={decl_id}")

        log(f"    Loading dataframe...")
        df = pd.read_parquet(fpath)
        log(f"    Loaded {len(df)} rows")

        # Sample up to 10000 states per file
        if len(df) > 10000:
            log(f"    Sampling 10000 from {len(df)}...")
            df = df.sample(n=10000, random_state=seed)

        states = df["state"].values
        values = df["V"].values.astype(np.float32)

        log(f"    Encoding global features...")
        feat_g = encode_global_features(states, seed, decl_id)
        log(f"    Encoding τ features...")
        feat_t = encode_tau_features(states, seed, decl_id)

        log(f"    Appending to training set...")
        for i in range(len(states)):
            train_features_global.append(feat_g[i])
            train_features_tau.append(feat_t[i])
            train_values.append(values[i])
            train_meta.append((seed, decl_id, i))

        log(f"    ✓ Done in {time.time() - t0:.1f}s (total: {len(train_values)} samples)")

    log(f"\n  Converting to numpy arrays...")
    train_features_global = np.array(train_features_global)
    train_features_tau = np.array(train_features_tau)
    train_values = np.array(train_values)

    log(f"  ✓ Total training samples for search: {len(train_values)}")

    # Analyze each worst prediction
    log("\n" + "=" * 70)
    log("DIAGNOSTIC RESULTS")
    log("=" * 70)

    raw_neighbor_variances = []
    tau_neighbor_variances = []
    raw_neighbor_errors = []
    tau_neighbor_errors = []

    n_to_analyze = min(20, len(worst_indices))
    log(f"\nAnalyzing {n_to_analyze} predictions with highest errors...")

    for rank, worst_idx in enumerate(worst_indices[:n_to_analyze]):
        log(f"\n  [{rank+1}/{n_to_analyze}] Finding neighbors...")
        file_idx, state_idx = all_indices[worst_idx]
        data = test_data[file_idx]

        true_val = all_true_values[worst_idx]
        pred_val = all_predictions[worst_idx]
        error = errors[worst_idx]

        query_global = data["features_global"][state_idx]
        query_tau = data["features_tau"][state_idx]

        # Find K=5 nearest neighbors in both encodings
        raw_indices, raw_dists = find_k_nearest(query_global, train_features_global, k=5)
        tau_indices, tau_dists = find_k_nearest(query_tau, train_features_tau, k=5)

        raw_neighbor_vals = train_values[raw_indices]
        tau_neighbor_vals = train_values[tau_indices]

        raw_mean = np.mean(raw_neighbor_vals)
        raw_var = np.var(raw_neighbor_vals)
        tau_mean = np.mean(tau_neighbor_vals)
        tau_var = np.var(tau_neighbor_vals)

        raw_neighbor_variances.append(raw_var)
        tau_neighbor_variances.append(tau_var)
        raw_neighbor_errors.append(abs(raw_mean - true_val))
        tau_neighbor_errors.append(abs(tau_mean - true_val))

        log(f"\nTest #{rank+1} (seed {data['seed']}, decl {data['decl_id']}):")
        log(f"  True: {true_val:+.1f}, MLP: {pred_val:+.1f}, Error: {error:.1f}")
        log(f"  ")
        log(f"  Raw-nearest (K=5):")
        log(f"    Values: {[f'{v:+.1f}' for v in raw_neighbor_vals]}")
        log(f"    Mean: {raw_mean:+.1f}, Var: {raw_var:.1f}, |Mean-True|: {abs(raw_mean - true_val):.1f}")
        raw_seeds = [train_meta[i][0] for i in raw_indices]
        log(f"    Seeds: {raw_seeds}")
        log(f"  ")
        log(f"  τ-nearest (K=5):")
        log(f"    Values: {[f'{v:+.1f}' for v in tau_neighbor_vals]}")
        log(f"    Mean: {tau_mean:+.1f}, Var: {tau_var:.1f}, |Mean-True|: {abs(tau_mean - true_val):.1f}")
        tau_seeds = [train_meta[i][0] for i in tau_indices]
        log(f"    Seeds: {tau_seeds}")

        # Check if τ found cross-seed matches
        if len(set(tau_seeds)) > 1 or tau_seeds[0] != data['seed']:
            log(f"    ✓ Cross-seed matching achieved!")

    # Phase 2: Analyze worst τ-matches
    log("\n" + "=" * 70)
    log("PHASE 2: ANALYZING WORST τ-MATCHES")
    log("=" * 70)
    log("Looking for patterns in cases where τ-neighbor value differs from true value...")

    # Sort by τ-neighbor error (descending)
    tau_errors_with_idx = [(tau_neighbor_errors[i], i) for i in range(len(tau_neighbor_errors))]
    tau_errors_with_idx.sort(reverse=True)

    # Analyze top 10 worst τ-matches
    log("\nTop 10 worst τ-matches (τ-neighbor value far from true):")

    # Track patterns
    tricks_done_diffs = []
    score_diffs = []
    leader_diffs = []
    void_pattern_diffs = []

    for rank, (tau_err, analysis_idx) in enumerate(tau_errors_with_idx[:10]):
        worst_idx = worst_indices[analysis_idx]
        file_idx, state_idx = all_indices[worst_idx]
        data = test_data[file_idx]

        true_val = all_true_values[worst_idx]
        query_tau = data["features_tau"][state_idx]

        # Find τ-nearest neighbor
        tau_indices, _ = find_k_nearest(query_tau, train_features_tau, k=1)
        neighbor_idx = tau_indices[0]
        neighbor_val = train_values[neighbor_idx]
        neighbor_tau = train_features_tau[neighbor_idx]

        # Extract non-τ features from both
        # New τ-encoding structure (72 features):
        # [0:28] = 4×7 trump one-hot, [28:32] = non-trump counts, [32] = score,
        # [33:37] = leader, [37:47] = decl, [47:71] = trick plays, [71] = tricks_done

        test_tricks_done = query_tau[71]
        neighbor_tricks_done = neighbor_tau[71]
        tricks_diff = abs(test_tricks_done - neighbor_tricks_done) * 7  # denormalize

        test_score = query_tau[32]
        neighbor_score = neighbor_tau[32]
        score_diff = abs(test_score - neighbor_score) * 42  # denormalize

        test_leader = np.argmax(query_tau[33:37])
        neighbor_leader = np.argmax(neighbor_tau[33:37])
        leader_same = test_leader == neighbor_leader

        # Count trump rank mismatches (how many trump positions differ)
        test_trumps = query_tau[0:28]
        neighbor_trumps = neighbor_tau[0:28]
        trump_diff = int(np.sum(np.abs(test_trumps - neighbor_trumps)))  # Hamming distance

        tricks_done_diffs.append(tricks_diff)
        score_diffs.append(score_diff)
        leader_diffs.append(0 if leader_same else 1)
        void_pattern_diffs.append(trump_diff)

        log(f"\n  [{rank+1}] τ-error: {tau_err:.1f} pts (true={true_val:+.1f}, neighbor={neighbor_val:+.1f})")
        log(f"      Tricks done: test={test_tricks_done*7:.1f}, neighbor={neighbor_tricks_done*7:.1f}, diff={tricks_diff:.1f}")
        log(f"      Score: test={test_score*42:.1f}, neighbor={neighbor_score*42:.1f}, diff={score_diff:.1f}")
        log(f"      Leader: test={test_leader}, neighbor={neighbor_leader}, same={leader_same}")
        log(f"      Trump holdings diff: {trump_diff} bits (Hamming distance on 28 one-hot)")

    # Summarize patterns
    log("\n" + "-" * 50)
    log("PATTERN SUMMARY (across 10 worst τ-matches):")
    log("-" * 50)
    log(f"  Tricks done diff:    mean={np.mean(tricks_done_diffs):.2f}, max={max(tricks_done_diffs):.1f}")
    log(f"  Score diff:          mean={np.mean(score_diffs):.2f}, max={max(score_diffs):.1f}")
    log(f"  Leader mismatch:     {sum(leader_diffs)}/10 cases")
    log(f"  Trump holdings diff: mean={np.mean(void_pattern_diffs):.2f}, max={max(void_pattern_diffs)} bits")

    if np.mean(tricks_done_diffs) > 1.5:
        log("\n  → SIGNAL: Tricks done differs significantly - GAME PHASE is critical")
    if np.mean(score_diffs) > 5:
        log("\n  → SIGNAL: Score differs significantly - SCORE STATE is critical")
    if sum(leader_diffs) >= 5:
        log("\n  → SIGNAL: Leader often differs - TURN POSITION is critical")
    if np.mean(void_pattern_diffs) > 2:
        log("\n  → SIGNAL: Trump holdings differ - encoding may need more features")

    # If trump holdings are identical (0 bits diff), check what else differs
    if np.mean(void_pattern_diffs) < 1:
        log("\n  → INSIGHT: Trump holdings are IDENTICAL in worst matches!")
        log("     This means one-hot τ-encoding is working correctly.")
        log("     Problem must be non-trump or trick state related.")
    elif np.mean(void_pattern_diffs) >= 1:
        log("\n  → NOTE: Trump holdings differ slightly. Showing details...")

    # Deeper analysis: compare per-player trump holdings
    log("\n" + "-" * 50)
    log("PHASE 2b: DEEP τ-PROFILE ANALYSIS (One-Hot Encoding)")
    log("-" * 50)
    log("Comparing per-player trump holdings in worst matches...")
    log("Format: [ranks held] where 0=boss, 6=lowest trump")

    for rank, (tau_err, analysis_idx) in enumerate(tau_errors_with_idx[:5]):
        worst_idx = worst_indices[analysis_idx]
        file_idx, state_idx = all_indices[worst_idx]
        data = test_data[file_idx]

        true_val = all_true_values[worst_idx]
        query_tau = data["features_tau"][state_idx]

        tau_indices, _ = find_k_nearest(query_tau, train_features_tau, k=1)
        neighbor_idx = tau_indices[0]
        neighbor_val = train_values[neighbor_idx]
        neighbor_tau = train_features_tau[neighbor_idx]

        log(f"\n  Case {rank+1}: true={true_val:+.1f}, neighbor={neighbor_val:+.1f}")

        # Show per-player trump holdings (one-hot)
        for p in range(4):
            test_ranks = [r for r in range(7) if query_tau[p*7 + r] > 0.5]
            neighbor_ranks = [r for r in range(7) if neighbor_tau[p*7 + r] > 0.5]
            match = "✓" if test_ranks == neighbor_ranks else "✗"
            log(f"      P{p}: test={test_ranks}  neighbor={neighbor_ranks}  {match}")

        # Non-trump counts
        test_nt = [f"{query_tau[28+p]*7:.0f}" for p in range(4)]
        neighbor_nt = [f"{neighbor_tau[28+p]*7:.0f}" for p in range(4)]
        log(f"      Non-trump counts: test={test_nt}  neighbor={neighbor_nt}")

        # Team trump counts
        test_team02 = sum(query_tau[0:7]) + sum(query_tau[14:21])
        test_team13 = sum(query_tau[7:14]) + sum(query_tau[21:28])
        neighbor_team02 = sum(neighbor_tau[0:7]) + sum(neighbor_tau[14:21])
        neighbor_team13 = sum(neighbor_tau[7:14]) + sum(neighbor_tau[21:28])

        log(f"      Team trumps (0+2 vs 1+3):")
        log(f"        Test:     {test_team02:.0f} vs {test_team13:.0f}")
        log(f"        Neighbor: {neighbor_team02:.0f} vs {neighbor_team13:.0f}")

    # Summary statistics
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    raw_var_mean = np.mean(raw_neighbor_variances)
    tau_var_mean = np.mean(tau_neighbor_variances)
    raw_err_mean = np.mean(raw_neighbor_errors)
    tau_err_mean = np.mean(tau_neighbor_errors)

    log(f"\nNeighbor Value Variance (lower = more coherent):")
    log(f"  Raw encoding:  {raw_var_mean:.1f}")
    log(f"  τ-encoding:    {tau_var_mean:.1f}")
    log(f"  Improvement:   {(raw_var_mean - tau_var_mean) / raw_var_mean * 100:.1f}%")

    log(f"\nNeighbor Mean Error vs True Value (lower = better prediction):")
    log(f"  Raw encoding:  {raw_err_mean:.1f} points")
    log(f"  τ-encoding:    {tau_err_mean:.1f} points")
    log(f"  Improvement:   {(raw_err_mean - tau_err_mean) / raw_err_mean * 100:.1f}%")

    log("\n" + "=" * 70)
    log("HYPOTHESIS VALIDATION")
    log("=" * 70)

    if tau_var_mean < raw_var_mean * 0.7 and tau_err_mean < raw_err_mean * 0.7:
        log("✅ HYPOTHESIS CONFIRMED")
        log("τ-encoding shows significantly tighter value clustering.")
        log("Proceed to t42-74vy for full τ-encoding implementation.")
    elif tau_var_mean < raw_var_mean or tau_err_mean < raw_err_mean:
        log("⚠️ HYPOTHESIS PARTIALLY CONFIRMED")
        log("τ-encoding shows improvement but not dramatic.")
        log("Consider refining τ-encoding or investigating other factors.")
    else:
        log("❌ HYPOTHESIS REJECTED")
        log("τ-encoding does not improve neighbor coherence.")
        log("Return to t42-m4wy for replanning.")


if __name__ == "__main__":
    main()
