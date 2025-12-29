#!/usr/bin/env python3
"""
Q-Value Regression Transformer for Texas 42.

Same architecture as classification transformer, but:
- Output: 7 Q-values (regression, not classification)
- Loss: MSE on Q-values for legal moves only
- Labels: mv0-mv6 from parquet (perspective-normalized)

For Value PIMC: sample opponent hands, get Q per sample, average, pick best.

Usage:
    python scripts/solver2/train_q_regression.py --epochs 15 --save-model data/solver2/q_model.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.solver2.declarations import N_DECLS, NOTRUMP
from scripts.solver2.rng import deal_from_seed
from scripts.solver2.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    is_in_called_suit,
    trick_rank,
)


def log(msg: str) -> None:
    elapsed = time.time() - LOG_START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


LOG_START_TIME = time.time()


# =============================================================================
# Trump Rank Computation (same as train_transformer.py)
# =============================================================================

def get_trump_rank(domino_id: int, decl_id: int) -> int:
    if decl_id == NOTRUMP:
        return 7
    if not is_in_called_suit(domino_id, decl_id):
        return 7
    trumps = []
    for d in range(28):
        if is_in_called_suit(d, decl_id):
            tau = trick_rank(d, 7, decl_id)
            trumps.append((d, tau))
    trumps.sort(key=lambda x: -x[1])
    for rank, (d, _) in enumerate(trumps):
        if d == domino_id:
            return rank
    return 7


TRUMP_RANK_TABLE = {}
for _decl in range(N_DECLS):
    for _dom in range(28):
        TRUMP_RANK_TABLE[(_dom, _decl)] = get_trump_rank(_dom, _decl)


# =============================================================================
# Tokenization Constants (same as train_transformer.py)
# =============================================================================

TOKEN_TYPE_CONTEXT = 0
TOKEN_TYPE_PLAYER0 = 1
TOKEN_TYPE_PLAYER1 = 2
TOKEN_TYPE_PLAYER2 = 3
TOKEN_TYPE_PLAYER3 = 4
TOKEN_TYPE_TRICK_P0 = 5
TOKEN_TYPE_TRICK_P1 = 6
TOKEN_TYPE_TRICK_P2 = 7

EMBED_HIGH_PIP = 7
EMBED_LOW_PIP = 7
EMBED_IS_DOUBLE = 2
EMBED_COUNT_VALUE = 3
EMBED_TRUMP_RANK = 8
EMBED_PLAYER_ID = 4
EMBED_IS_CURRENT = 2
EMBED_IS_PARTNER = 2
EMBED_IS_REMAINING = 2
EMBED_DECL = 10
EMBED_LEADER = 4

COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}


# =============================================================================
# Q-Regression Transformer
# =============================================================================

class QRegressionTransformer(nn.Module):
    """
    Same as DominoTransformer but outputs Q-values instead of logits.
    No final classification - just regression targets.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Feature embeddings (compositional)
        self.high_pip_embed = nn.Embedding(EMBED_HIGH_PIP, embed_dim // 6)
        self.low_pip_embed = nn.Embedding(EMBED_LOW_PIP, embed_dim // 6)
        self.is_double_embed = nn.Embedding(EMBED_IS_DOUBLE, embed_dim // 12)
        self.count_value_embed = nn.Embedding(EMBED_COUNT_VALUE, embed_dim // 12)
        self.trump_rank_embed = nn.Embedding(EMBED_TRUMP_RANK, embed_dim // 6)
        self.player_id_embed = nn.Embedding(EMBED_PLAYER_ID, embed_dim // 12)
        self.is_current_embed = nn.Embedding(EMBED_IS_CURRENT, embed_dim // 12)
        self.is_partner_embed = nn.Embedding(EMBED_IS_PARTNER, embed_dim // 12)
        self.is_remaining_embed = nn.Embedding(EMBED_IS_REMAINING, embed_dim // 12)
        self.token_type_embed = nn.Embedding(8, embed_dim // 6)
        self.decl_embed = nn.Embedding(EMBED_DECL, embed_dim // 6)
        self.leader_embed = nn.Embedding(EMBED_LEADER, embed_dim // 12)

        self.input_proj = nn.Linear(self._calc_input_dim(), embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output: 1 Q-value per hand slot
        self.output_proj = nn.Linear(embed_dim, 1)

    def _calc_input_dim(self) -> int:
        return (
            self.embed_dim // 6 +   # high_pip
            self.embed_dim // 6 +   # low_pip
            self.embed_dim // 12 +  # is_double
            self.embed_dim // 12 +  # count_value
            self.embed_dim // 6 +   # trump_rank
            self.embed_dim // 12 +  # player_id
            self.embed_dim // 12 +  # is_current
            self.embed_dim // 12 +  # is_partner
            self.embed_dim // 12 +  # is_remaining
            self.embed_dim // 6 +   # token_type
            self.embed_dim // 6 +   # decl
            self.embed_dim // 12    # leader
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        current_player: torch.Tensor,
    ) -> torch.Tensor:
        """Returns Q-values: (batch, 7)"""
        batch_size = tokens.size(0)
        device = tokens.device

        embeds = []
        embeds.append(self.high_pip_embed(tokens[:, :, 0]))
        embeds.append(self.low_pip_embed(tokens[:, :, 1]))
        embeds.append(self.is_double_embed(tokens[:, :, 2]))
        embeds.append(self.count_value_embed(tokens[:, :, 3]))
        embeds.append(self.trump_rank_embed(tokens[:, :, 4]))
        embeds.append(self.player_id_embed(tokens[:, :, 5]))
        embeds.append(self.is_current_embed(tokens[:, :, 6]))
        embeds.append(self.is_partner_embed(tokens[:, :, 7]))
        embeds.append(self.is_remaining_embed(tokens[:, :, 8]))
        embeds.append(self.token_type_embed(tokens[:, :, 9]))
        embeds.append(self.decl_embed(tokens[:, :, 10]))
        embeds.append(self.leader_embed(tokens[:, :, 11]))

        x = torch.cat(embeds, dim=-1)
        x = self.input_proj(x)

        attn_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract current player's 7 hand tokens
        start_indices = 1 + current_player * 7
        offsets = torch.arange(7, device=device)
        gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        hand_repr = torch.gather(x, dim=1, index=gather_indices)

        # Output Q-values (no activation - regression)
        q_values = self.output_proj(hand_repr).squeeze(-1)

        return q_values


# =============================================================================
# Data Processing
# =============================================================================

def process_file_for_q_regression(
    file_path: Path,
    max_samples: int | None,
    rng: np.random.Generator,
) -> tuple | None:
    """
    Process one parquet file for Q-regression.

    Returns:
        tokens: (N, 32, 12)
        masks: (N, 32)
        current_players: (N,)
        q_targets: (N, 7) - Q-values normalized for perspective
        legal_masks: (N, 7)
        teams: (N,)
    """
    try:
        pf = pq.ParquetFile(file_path)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())

        df = pd.read_parquet(file_path)
        states = df["state"].values.astype(np.int64)

        mv_cols = [f"mv{i}" for i in range(7)]
        q_values_all = np.stack([df[c].values for c in mv_cols], axis=1).astype(np.float32)

        hands = deal_from_seed(seed)
        hands_flat = np.array([hands[p][i] for p in range(4) for i in range(7)])

        n_states = len(states)

        if max_samples and n_states > max_samples:
            indices = rng.choice(n_states, size=max_samples, replace=False)
            states = states[indices]
            q_values_all = q_values_all[indices]

        # Legal masks
        legal_masks = (q_values_all != -128).astype(np.float32)

        has_legal = legal_masks.any(axis=1)
        if not has_legal.any():
            return None

        states = states[has_legal]
        q_values_all = q_values_all[has_legal]
        legal_masks = legal_masks[has_legal]
        n_samples = len(states)

        # Extract state fields
        remaining = np.zeros((n_samples, 4), dtype=np.int64)
        for p in range(4):
            remaining[:, p] = (states >> (p * 7)) & 0x7F

        leader = ((states >> 28) & 0x3).astype(np.int64)
        trick_len = ((states >> 30) & 0x3).astype(np.int64)
        p0_local = ((states >> 32) & 0x7).astype(np.int64)
        p1_local = ((states >> 35) & 0x7).astype(np.int64)
        p2_local = ((states >> 38) & 0x7).astype(np.int64)

        current_player = ((leader + trick_len) % 4).astype(np.int64)
        partner = ((current_player + 2) % 4).astype(np.int64)
        team = current_player % 2

        # Perspective normalize Q-values:
        # Team 0 sees Q as-is (positive Q = good)
        # Team 1 sees -Q (so model always maximizes)
        q_targets = np.where(
            team[:, np.newaxis] == 0,
            q_values_all,
            -q_values_all
        )
        # Replace illegal with 0 (we won't compute loss on them)
        q_targets = np.where(legal_masks > 0, q_targets, 0.0)

        # Precompute domino features
        domino_features = np.zeros((28, 5), dtype=np.int64)
        for i, global_id in enumerate(hands_flat):
            domino_features[i, 0] = DOMINO_HIGH[global_id]
            domino_features[i, 1] = DOMINO_LOW[global_id]
            domino_features[i, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
            domino_features[i, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
            domino_features[i, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]

        # Build tokens
        MAX_TOKENS = 32
        N_FEATURES = 12
        tokens = np.zeros((n_samples, MAX_TOKENS, N_FEATURES), dtype=np.int64)
        masks = np.zeros((n_samples, MAX_TOKENS), dtype=np.float32)

        normalized_leader = ((leader - current_player + 4) % 4).astype(np.int64)

        # Token 0: Context
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_id
        tokens[:, 0, 11] = normalized_leader
        masks[:, 0] = 1.0

        # Tokens 1-28: Hand positions
        for p in range(4):
            normalized_player = ((p - current_player + 4) % 4).astype(np.int64)
            for local_idx in range(7):
                flat_idx = p * 7 + local_idx
                token_idx = 1 + flat_idx

                tokens[:, token_idx, 0] = domino_features[flat_idx, 0]
                tokens[:, token_idx, 1] = domino_features[flat_idx, 1]
                tokens[:, token_idx, 2] = domino_features[flat_idx, 2]
                tokens[:, token_idx, 3] = domino_features[flat_idx, 3]
                tokens[:, token_idx, 4] = domino_features[flat_idx, 4]
                tokens[:, token_idx, 5] = normalized_player
                tokens[:, token_idx, 6] = (normalized_player == 0).astype(np.int64)
                tokens[:, token_idx, 7] = (normalized_player == 2).astype(np.int64)
                tokens[:, token_idx, 8] = ((remaining[:, p] >> local_idx) & 1).astype(np.int64)
                tokens[:, token_idx, 9] = TOKEN_TYPE_PLAYER0 + p
                tokens[:, token_idx, 10] = decl_id
                tokens[:, token_idx, 11] = normalized_leader
                masks[:, token_idx] = 1.0

        # Tokens 29-31: Trick plays
        trick_plays = [p0_local, p1_local, p2_local]
        trick_token_types = [TOKEN_TYPE_TRICK_P0, TOKEN_TYPE_TRICK_P1, TOKEN_TYPE_TRICK_P2]

        for trick_pos in range(3):
            has_play = (trick_len > trick_pos) & (trick_plays[trick_pos] < 7)
            if not has_play.any():
                continue

            token_idx = 29 + trick_pos
            local_idx = trick_plays[trick_pos]
            play_player = (leader + trick_pos) % 4

            for sample_idx in np.where(has_play)[0]:
                pp = int(play_player[sample_idx])
                li = int(local_idx[sample_idx])
                global_id = hands[pp][li]
                cp = int(current_player[sample_idx])
                normalized_pp = (pp - cp + 4) % 4

                tokens[sample_idx, token_idx, 0] = DOMINO_HIGH[global_id]
                tokens[sample_idx, token_idx, 1] = DOMINO_LOW[global_id]
                tokens[sample_idx, token_idx, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
                tokens[sample_idx, token_idx, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
                tokens[sample_idx, token_idx, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]
                tokens[sample_idx, token_idx, 5] = normalized_pp
                tokens[sample_idx, token_idx, 6] = 1 if normalized_pp == 0 else 0
                tokens[sample_idx, token_idx, 7] = 1 if normalized_pp == 2 else 0
                tokens[sample_idx, token_idx, 8] = 0
                tokens[sample_idx, token_idx, 9] = trick_token_types[trick_pos]
                tokens[sample_idx, token_idx, 10] = decl_id
                tokens[sample_idx, token_idx, 11] = int(normalized_leader[sample_idx])
                masks[sample_idx, token_idx] = 1.0

        return (tokens, masks, current_player, q_targets, legal_masks, team)

    except Exception as e:
        log(f"  ERROR processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_data(
    data_dir: Path,
    seed_range: tuple[int, int],
    max_samples_per_file: int | None,
    max_files: int | None,
    rng: np.random.Generator,
    label: str,
) -> tuple:
    """Load data for Q-regression."""
    log(f"\n=== Loading {label} Data (seeds {seed_range[0]}-{seed_range[1]}) ===")

    parquet_files = sorted(data_dir.glob("seed_*.parquet"))

    relevant_files = []
    for f in parquet_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                seed = int(parts[1])
                if seed_range[0] <= seed <= seed_range[1]:
                    relevant_files.append(f)
            except ValueError:
                pass

    if max_files and len(relevant_files) > max_files:
        relevant_files = relevant_files[:max_files]
        log(f"Found {len(relevant_files)} files (limited to {max_files})")
    else:
        log(f"Found {len(relevant_files)} files")

    all_tokens = []
    all_masks = []
    all_players = []
    all_q_targets = []
    all_legal = []
    all_teams = []

    t0 = time.time()
    for i, f in enumerate(relevant_files):
        result = process_file_for_q_regression(f, max_samples_per_file, rng)
        if result is not None:
            tokens, masks, players, q_targets, legal, teams = result
            all_tokens.append(tokens)
            all_masks.append(masks)
            all_players.append(players)
            all_q_targets.append(q_targets)
            all_legal.append(legal)
            all_teams.append(teams)

            if (i + 1) % 10 == 0 or i == len(relevant_files) - 1:
                total_samples = sum(len(t) for t in all_q_targets)
                log(f"  [{i+1}/{len(relevant_files)}] {f.name}: {len(tokens):,} samples "
                    f"(total: {total_samples:,}, {time.time()-t0:.1f}s)")

    if not all_tokens:
        raise ValueError(f"No data loaded for {label}")

    return (
        np.concatenate(all_tokens, axis=0),
        np.concatenate(all_masks, axis=0),
        np.concatenate(all_players, axis=0),
        np.concatenate(all_q_targets, axis=0),
        np.concatenate(all_legal, axis=0),
        np.concatenate(all_teams, axis=0),
    )


# =============================================================================
# Training
# =============================================================================

def compute_metrics(
    model: QRegressionTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Compute MSE loss and accuracy on a data loader."""
    model.eval()
    total_mse = 0.0
    total_correct = 0
    total_samples = 0
    total_legal_moves = 0

    with torch.no_grad():
        for tokens, masks, players, q_targets, legal in loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)
            q_targets = q_targets.to(device)
            legal = legal.to(device)

            q_pred = model(tokens, masks, players)

            # MSE only on legal moves
            legal_mask = (legal > 0)
            mse = ((q_pred - q_targets) ** 2 * legal_mask).sum() / legal_mask.sum()
            total_mse += mse.item() * legal_mask.sum().item()
            total_legal_moves += legal_mask.sum().item()

            # Accuracy: does argmax(pred) == argmax(target)?
            q_pred_masked = q_pred.masked_fill(~legal_mask, float('-inf'))
            q_target_masked = q_targets.masked_fill(~legal_mask, float('-inf'))

            pred_best = q_pred_masked.argmax(dim=-1)
            target_best = q_target_masked.argmax(dim=-1)

            total_correct += (pred_best == target_best).sum().item()
            total_samples += len(tokens)

    avg_mse = total_mse / total_legal_moves if total_legal_moves > 0 else 0
    rmse = np.sqrt(avg_mse)
    acc = total_correct / total_samples if total_samples > 0 else 0
    return rmse, avg_mse, acc


def train_epoch(
    model: QRegressionTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    q_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Train for one epoch using MSE loss on Q-values."""
    model.train()
    total_mse = 0.0
    total_correct = 0
    total_samples = 0
    total_legal_moves = 0

    for tokens, masks, players, q_targets, legal, teams in loader:
        tokens = tokens.to(device)
        masks = masks.to(device)
        players = players.to(device)
        q_targets = q_targets.to(device) / q_scale  # Scale Q-values for training
        legal = legal.to(device)

        optimizer.zero_grad()
        q_pred = model(tokens, masks, players)

        # MSE loss only on legal moves
        legal_mask = (legal > 0)
        mse_per_move = (q_pred - q_targets) ** 2 * legal_mask
        loss = mse_per_move.sum() / legal_mask.sum()

        loss.backward()
        optimizer.step()

        total_mse += loss.item() * legal_mask.sum().item()
        total_legal_moves += legal_mask.sum().item()

        # Accuracy tracking
        with torch.no_grad():
            q_pred_masked = q_pred.masked_fill(~legal_mask, float('-inf'))
            q_target_masked = q_targets.masked_fill(~legal_mask, float('-inf'))
            pred_best = q_pred_masked.argmax(dim=-1)
            target_best = q_target_masked.argmax(dim=-1)
            total_correct += (pred_best == target_best).sum().item()
            total_samples += len(tokens)

    avg_mse = total_mse / total_legal_moves if total_legal_moves > 0 else 0
    rmse = np.sqrt(avg_mse)
    acc = total_correct / total_samples if total_samples > 0 else 0
    return rmse, avg_mse, acc


def main():
    parser = argparse.ArgumentParser(description="Q-value regression transformer")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--q-scale", type=float, default=10.0,
                        help="Scale factor for Q-values (Q-values can be large)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        log("Using CPU")

    log("\n=== Configuration ===")
    log(f"Max samples per file: {args.max_samples or 'all'}")
    log(f"Batch size: {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log(f"Epochs: {args.epochs}")
    log(f"Q-scale: {args.q_scale}")
    log(f"Architecture: {args.n_layers} layers, {args.n_heads} heads, {args.embed_dim} dim")

    data_dir = Path(args.data_dir)

    # Load data
    train_data = load_data(data_dir, (0, 89), args.max_samples, args.max_files, rng, "Train")
    train_tokens, train_masks, train_players, train_q_targets, train_legal, train_teams = train_data

    test_max_files = max(1, args.max_files // 9) if args.max_files else None
    test_data = load_data(data_dir, (90, 99), args.max_samples, test_max_files, rng, "Test")
    test_tokens, test_masks, test_players, test_q_targets, test_legal, test_teams = test_data

    log(f"\n=== Data Summary ===")
    log(f"Train samples: {len(train_q_targets):,}")
    log(f"Test samples: {len(test_q_targets):,}")

    # Analyze Q-value distribution
    legal_q = train_q_targets[train_legal > 0]
    log(f"Q-value range: [{legal_q.min():.1f}, {legal_q.max():.1f}]")
    log(f"Q-value std: {legal_q.std():.2f}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(train_tokens),
        torch.tensor(train_masks),
        torch.tensor(train_players),
        torch.tensor(train_q_targets),
        torch.tensor(train_legal),
        torch.tensor(train_teams),
    )
    test_dataset = TensorDataset(
        torch.tensor(test_tokens),
        torch.tensor(test_masks),
        torch.tensor(test_players),
        torch.tensor(test_q_targets),
        torch.tensor(test_legal),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    # Create model
    log("\n=== Creating Model ===")
    model = QRegressionTransformer(
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    log("\n=== Training ===")
    log(f"{'Epoch':>5} | {'Train RMSE':>10} | {'Train Acc':>9} | {'Test RMSE':>10} | {'Test Acc':>9}")
    log("-" * 60)

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_rmse, train_mse, train_acc = train_epoch(
            model, train_loader, optimizer, device, args.q_scale
        )
        test_rmse, test_mse, test_acc = compute_metrics(model, test_loader, device)

        scheduler.step()

        # Scale RMSE back to original Q-value units
        train_rmse_scaled = train_rmse * args.q_scale
        test_rmse_scaled = np.sqrt(test_mse)  # test uses unscaled Q

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if args.save_model:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'test_acc': test_acc,
                    'test_rmse': test_rmse_scaled,
                    'q_scale': args.q_scale,
                    'args': vars(args),
                }, args.save_model)
                log(f"  → Saved best model to {args.save_model}")

        log(f"{epoch:5d} | {train_rmse_scaled:10.2f} | {train_acc:8.2%} | {test_rmse_scaled:10.2f} | {test_acc:8.2%}")

    # Final results
    log("\n=== Final Results ===")
    log(f"Best test accuracy: {best_test_acc:.2%}")

    # Detailed error analysis
    log("\n=== Q-Value Prediction Analysis ===")
    model.eval()

    all_pred_q = []
    all_target_q = []
    all_legal = []

    with torch.no_grad():
        for batch in test_loader:
            tokens, masks, players, q_targets, legal = batch
            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)

            q_pred = model(tokens, masks, players)

            all_pred_q.append(q_pred.cpu().numpy())
            all_target_q.append(q_targets.numpy())
            all_legal.append(legal.numpy())

    pred_q = np.concatenate(all_pred_q, axis=0)
    target_q = np.concatenate(all_target_q, axis=0)
    legal_mask = np.concatenate(all_legal, axis=0) > 0

    # Error on legal moves
    errors = (pred_q - target_q)[legal_mask]
    log(f"Prediction errors (legal moves):")
    log(f"  Mean error: {errors.mean():.3f}")
    log(f"  Std error: {errors.std():.3f}")
    log(f"  MAE: {np.abs(errors).mean():.3f}")
    log(f"  RMSE: {np.sqrt((errors ** 2).mean()):.3f}")

    # Ranking accuracy by Q-gap
    n_samples = len(pred_q)
    correct = 0
    regret_sum = 0.0

    for i in range(n_samples):
        legal_i = legal_mask[i]
        pred_i = pred_q[i].copy()
        target_i = target_q[i].copy()

        pred_i[~legal_i] = float('-inf')
        target_i[~legal_i] = float('-inf')

        pred_best = np.argmax(pred_i)
        target_best = np.argmax(target_i)

        if pred_best == target_best:
            correct += 1
        else:
            # Regret = Q(optimal) - Q(chosen)
            regret = target_q[i, target_best] - target_q[i, pred_best]
            regret_sum += regret

    log(f"\nRanking accuracy: {correct}/{n_samples} = {correct/n_samples:.2%}")
    log(f"Total regret: {regret_sum:.1f}")
    log(f"Mean regret per error: {regret_sum/(n_samples-correct):.2f}" if correct < n_samples else "No errors")


if __name__ == "__main__":
    main()
