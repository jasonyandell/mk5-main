#!/usr/bin/env python3
"""
Chunked Transformer Training for Texas 42.

Fixes OOM by loading data in chunks (N files at a time) instead of all at once.
Each epoch iterates through all chunks, so we see the full dataset per epoch.

This is the middle ground between:
- train_transformer.py (loads all, OOMs on large datasets)
- train_streaming.py (streams per-sample, too slow due to tokenization overhead)

The chunked approach loads + tokenizes a chunk of files, trains fast on that chunk,
then releases memory and loads the next chunk.

Usage:
    # Quick test
    python scripts/solver2/train_chunked.py --files-per-chunk 5 --samples-per-file 10000 --epochs 3

    # Full training
    python scripts/solver2/train_chunked.py --files-per-chunk 20 --samples-per-file 50000 --epochs 10
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
# Trump Rank (copied from train_transformer.py)
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


TOKEN_TYPE_CONTEXT = 0
TOKEN_TYPE_PLAYER0 = 1
TOKEN_TYPE_TRICK_P0 = 5
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}


# =============================================================================
# Model (same as train_transformer.py)
# =============================================================================

class DominoTransformer(nn.Module):
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

        self.high_pip_embed = nn.Embedding(7, embed_dim // 6)
        self.low_pip_embed = nn.Embedding(7, embed_dim // 6)
        self.is_double_embed = nn.Embedding(2, embed_dim // 12)
        self.count_value_embed = nn.Embedding(3, embed_dim // 12)
        self.trump_rank_embed = nn.Embedding(8, embed_dim // 6)
        self.player_id_embed = nn.Embedding(4, embed_dim // 12)
        self.is_current_embed = nn.Embedding(2, embed_dim // 12)
        self.is_partner_embed = nn.Embedding(2, embed_dim // 12)
        self.is_remaining_embed = nn.Embedding(2, embed_dim // 12)
        self.token_type_embed = nn.Embedding(8, embed_dim // 6)
        self.decl_embed = nn.Embedding(10, embed_dim // 6)
        self.leader_embed = nn.Embedding(4, embed_dim // 12)

        self.input_proj = nn.Linear(self._calc_input_dim(), embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(embed_dim, 1)

    def _calc_input_dim(self) -> int:
        return (
            self.embed_dim // 6 * 4 +
            self.embed_dim // 12 * 6 +
            self.embed_dim // 6 +
            self.embed_dim // 12
        )

    def forward(self, tokens, mask, current_player):
        batch_size = tokens.size(0)
        device = tokens.device

        embeds = [
            self.high_pip_embed(tokens[:, :, 0]),
            self.low_pip_embed(tokens[:, :, 1]),
            self.is_double_embed(tokens[:, :, 2]),
            self.count_value_embed(tokens[:, :, 3]),
            self.trump_rank_embed(tokens[:, :, 4]),
            self.player_id_embed(tokens[:, :, 5]),
            self.is_current_embed(tokens[:, :, 6]),
            self.is_partner_embed(tokens[:, :, 7]),
            self.is_remaining_embed(tokens[:, :, 8]),
            self.token_type_embed(tokens[:, :, 9]),
            self.decl_embed(tokens[:, :, 10]),
            self.leader_embed(tokens[:, :, 11]),
        ]

        x = torch.cat(embeds, dim=-1)
        x = self.input_proj(x)

        attn_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        start_indices = 1 + current_player * 7
        offsets = torch.arange(7, device=device)
        gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        hand_repr = torch.gather(x, dim=1, index=gather_indices)
        logits = self.output_proj(hand_repr).squeeze(-1)

        return logits


# =============================================================================
# Data Processing (Vectorized - from train_transformer.py)
# =============================================================================

def process_file_vectorized(
    file_path: Path,
    max_samples: int | None,
    rng: np.random.Generator,
) -> tuple | None:
    """Process one parquet file with vectorized operations."""
    try:
        pf = pq.ParquetFile(file_path)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())

        df = pd.read_parquet(file_path)
        states = df["state"].values.astype(np.int64)

        mv_cols = [f"mv{i}" for i in range(7)]
        q_values_all = np.stack([df[c].values for c in mv_cols], axis=1)

        hands = deal_from_seed(seed)
        hands_flat = np.array([hands[p][i] for p in range(4) for i in range(7)])

        n_states = len(states)

        if max_samples and n_states > max_samples:
            indices = rng.choice(n_states, size=max_samples, replace=False)
            states = states[indices]
            q_values_all = q_values_all[indices]

        legal_masks = (q_values_all != -128).astype(np.float32)
        has_legal = legal_masks.any(axis=1)
        if not has_legal.any():
            return None

        states = states[has_legal]
        q_values_all = q_values_all[has_legal]
        legal_masks = legal_masks[has_legal]
        n_samples = len(states)

        remaining = np.zeros((n_samples, 4), dtype=np.int64)
        for p in range(4):
            remaining[:, p] = (states >> (p * 7)) & 0x7F

        leader = ((states >> 28) & 0x3).astype(np.int64)
        trick_len = ((states >> 30) & 0x3).astype(np.int64)
        p0_local = ((states >> 32) & 0x7).astype(np.int64)
        p1_local = ((states >> 35) & 0x7).astype(np.int64)
        p2_local = ((states >> 38) & 0x7).astype(np.int64)

        current_player = ((leader + trick_len) % 4).astype(np.int64)

        q_int32 = q_values_all.astype(np.int32)
        team = current_player % 2

        q_for_argmax = np.where(team[:, np.newaxis] == 0, q_int32, -q_int32)
        q_masked = np.where(legal_masks > 0, q_for_argmax, -129)
        targets = q_masked.argmax(axis=1).astype(np.int64)

        domino_features = np.zeros((28, 5), dtype=np.int64)
        for i, global_id in enumerate(hands_flat):
            domino_features[i, 0] = DOMINO_HIGH[global_id]
            domino_features[i, 1] = DOMINO_LOW[global_id]
            domino_features[i, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
            domino_features[i, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
            domino_features[i, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]

        MAX_TOKENS = 32
        N_FEATURES = 12
        tokens = np.zeros((n_samples, MAX_TOKENS, N_FEATURES), dtype=np.int64)
        masks = np.zeros((n_samples, MAX_TOKENS), dtype=np.float32)

        normalized_leader = ((leader - current_player + 4) % 4).astype(np.int64)

        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = decl_id
        tokens[:, 0, 11] = normalized_leader
        masks[:, 0] = 1.0

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

        trick_plays = [p0_local, p1_local, p2_local]
        trick_token_types = [TOKEN_TYPE_TRICK_P0, TOKEN_TYPE_TRICK_P0 + 1, TOKEN_TYPE_TRICK_P0 + 2]

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

        return (tokens, masks, current_player, targets, legal_masks, q_int32, team)

    except Exception as e:
        return None


def load_chunk(
    files: list[Path],
    max_samples_per_file: int | None,
    rng: np.random.Generator,
) -> tuple | None:
    """Load a chunk of files into memory."""
    all_tokens = []
    all_masks = []
    all_players = []
    all_targets = []
    all_legal = []
    all_qvals = []
    all_teams = []

    for f in files:
        result = process_file_vectorized(f, max_samples_per_file, rng)
        if result is not None:
            tokens, masks, players, targets, legal, qvals, teams = result
            all_tokens.append(tokens)
            all_masks.append(masks)
            all_players.append(players)
            all_targets.append(targets)
            all_legal.append(legal)
            all_qvals.append(qvals)
            all_teams.append(teams)

    if not all_tokens:
        return None

    return (
        np.concatenate(all_tokens, axis=0),
        np.concatenate(all_masks, axis=0),
        np.concatenate(all_players, axis=0),
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_legal, axis=0),
        np.concatenate(all_qvals, axis=0),
        np.concatenate(all_teams, axis=0),
    )


# =============================================================================
# Training
# =============================================================================

def train_on_chunk(
    model: DominoTransformer,
    data: tuple,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    temperature: float = 3.0,
    soft_weight: float = 0.7,
) -> tuple[float, float, int]:
    """Train on a chunk of data."""
    tokens, masks, players, targets, legal, qvals, teams = data

    dataset = TensorDataset(
        torch.tensor(tokens),
        torch.tensor(masks),
        torch.tensor(players),
        torch.tensor(targets),
        torch.tensor(legal),
        torch.tensor(qvals),
        torch.tensor(teams),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for batch in loader:
        tok, msk, plr, tgt, leg, qv, tm = batch
        tok = tok.to(device)
        msk = msk.to(device)
        plr = plr.to(device)
        tgt = tgt.to(device)
        leg = leg.to(device)
        qv = qv.to(device).float()
        tm = tm.to(device)

        optimizer.zero_grad()
        logits = model(tok, msk, plr)
        logits_masked = logits.masked_fill(leg == 0, float('-inf'))

        hard_loss = F.cross_entropy(logits_masked, tgt)

        team_sign = torch.where(tm == 0, 1.0, -1.0).unsqueeze(1)
        q_for_soft = qv * team_sign
        q_masked = torch.where(leg > 0, q_for_soft, torch.tensor(float('-inf'), device=device))
        soft_targets = F.softmax(q_masked / temperature, dim=-1)
        soft_targets = soft_targets.clamp(min=1e-8)
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
        log_probs = F.log_softmax(logits_masked, dim=-1)
        log_probs_safe = log_probs.masked_fill(leg == 0, 0.0)
        soft_loss = -(soft_targets * log_probs_safe).sum(dim=-1).mean()

        loss = (1 - soft_weight) * hard_loss + soft_weight * soft_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(tgt)

        with torch.no_grad():
            preds = logits_masked.argmax(dim=-1)
            total_correct += (preds == tgt).sum().item()
            total_samples += len(tgt)

    acc = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss, acc, total_samples


def evaluate_on_chunk(
    model: DominoTransformer,
    data: tuple,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, int]:
    """Evaluate on a chunk of data."""
    tokens, masks, players, targets, legal, qvals, teams = data

    dataset = TensorDataset(
        torch.tensor(tokens),
        torch.tensor(masks),
        torch.tensor(players),
        torch.tensor(targets),
        torch.tensor(legal),
    )
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            tok, msk, plr, tgt, leg = batch
            tok = tok.to(device)
            msk = msk.to(device)
            plr = plr.to(device)
            tgt = tgt.to(device)
            leg = leg.to(device)

            logits = model(tok, msk, plr)
            logits_masked = logits.masked_fill(leg == 0, float('-inf'))

            loss = F.cross_entropy(logits_masked, tgt)
            total_loss += loss.item() * len(tgt)

            preds = logits_masked.argmax(dim=-1)
            total_correct += (preds == tgt).sum().item()
            total_samples += len(tgt)

    acc = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss, acc, total_samples


def get_file_chunks(files: list[Path], chunk_size: int) -> list[list[Path]]:
    """Split files into chunks."""
    chunks = []
    for i in range(0, len(files), chunk_size):
        chunks.append(files[i:i + chunk_size])
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chunked transformer training")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--files-per-chunk", type=int, default=20,
                        help="Number of parquet files to load per chunk")
    parser.add_argument("--samples-per-file", type=int, default=50000,
                        help="Max samples to load per parquet file")
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--soft-weight", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--test-files", type=int, default=10,
                        help="Number of files to use for test set (held in memory)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("Using CPU")

    log("\n=== Configuration ===")
    log(f"Files per chunk: {args.files_per_chunk}")
    log(f"Samples per file: {args.samples_per_file}")
    log(f"Batch size: {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log(f"Epochs: {args.epochs}")
    log(f"Soft weight: {args.soft_weight}")

    data_dir = Path(args.data_dir)

    # Find all parquet files
    parquet_files = sorted(data_dir.glob("seed_*.parquet"))
    log(f"\nFound {len(parquet_files)} parquet files")

    if len(parquet_files) == 0:
        log("ERROR: No parquet files found!")
        return

    # Split into train/test by seed
    train_files = []
    test_files = []

    for f in parquet_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                seed = int(parts[1])
                if seed >= 90:  # Seeds 90-99 for test
                    test_files.append(f)
                else:  # Seeds 0-89 for train
                    train_files.append(f)
            except ValueError:
                pass

    # Shuffle train files
    rng.shuffle(train_files)

    # Limit test files
    test_files = test_files[:args.test_files]

    log(f"Train files: {len(train_files)}")
    log(f"Test files: {len(test_files)}")

    # Split train files into chunks
    train_chunks = get_file_chunks(train_files, args.files_per_chunk)
    log(f"Train chunks: {len(train_chunks)}")

    # Pre-load test data (keep in memory for consistent evaluation)
    log("\nLoading test data...")
    test_data = load_chunk(test_files, args.samples_per_file, rng)
    if test_data is None:
        log("ERROR: Failed to load test data!")
        return
    log(f"Test samples: {len(test_data[3]):,}")

    # Create model
    model = DominoTransformer(
        embed_dim=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.1,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    log("\n=== Training ===")
    log(f"{'Epoch':>5} | {'Chunks':>6} | {'Train Samples':>13} | {'Train Acc':>9} | {'Test Acc':>9}")
    log("-" * 60)

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_samples = 0
        epoch_correct = 0
        epoch_loss = 0.0

        # Shuffle chunks each epoch
        chunk_order = list(range(len(train_chunks)))
        rng.shuffle(chunk_order)

        for chunk_idx, ci in enumerate(chunk_order):
            chunk_files = train_chunks[ci]

            # Load this chunk
            chunk_data = load_chunk(chunk_files, args.samples_per_file, rng)
            if chunk_data is None:
                continue

            # Train on chunk
            loss, acc, n_samples = train_on_chunk(
                model, chunk_data, optimizer, device,
                args.batch_size, args.temperature, args.soft_weight
            )

            epoch_samples += n_samples
            epoch_correct += int(acc * n_samples)
            epoch_loss += loss * n_samples

            # Free memory
            del chunk_data
            gc.collect()

            if (chunk_idx + 1) % 5 == 0:
                log(f"  Epoch {epoch}, chunk {chunk_idx+1}/{len(train_chunks)}: "
                    f"{epoch_samples:,} samples, acc={epoch_correct/epoch_samples:.2%}")

        # End of epoch
        scheduler.step()

        train_acc = epoch_correct / epoch_samples if epoch_samples > 0 else 0

        # Evaluate on test
        test_loss, test_acc, _ = evaluate_on_chunk(
            model, test_data, device, args.batch_size
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if args.save_model:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'test_acc': test_acc,
                    'train_acc': train_acc,
                    'args': vars(args),
                }, args.save_model)
                log(f"  -> Saved best model")

        log(f"{epoch:5d} | {len(train_chunks):6d} | {epoch_samples:13,} | {train_acc:8.2%} | {test_acc:8.2%}")

    log(f"\nBest test accuracy: {best_test_acc:.2%}")

    # Q-Gap Analysis on test set
    log("\n=== Q-Gap Analysis ===")
    model.eval()

    tokens, masks, players, targets, legal, qvals, teams = test_data

    dataset = TensorDataset(
        torch.tensor(tokens),
        torch.tensor(masks),
        torch.tensor(players),
        torch.tensor(targets),
        torch.tensor(legal),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            tok, msk, plr, tgt, leg = batch
            tok = tok.to(device)
            msk = msk.to(device)
            plr = plr.to(device)
            leg = leg.to(device)

            logits = model(tok, msk, plr)
            logits_masked = logits.masked_fill(leg == 0, float('-inf'))
            preds = logits_masked.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)

    preds = np.array(all_preds)
    n_test = len(targets)

    # Compute Q-gaps
    gaps = []
    errors = []

    for i in range(n_test):
        q = qvals[i]
        legal_mask = legal[i]
        t = teams[i]
        pred_idx = preds[i]
        true_idx = targets[i]

        if t == 0:
            legal_q = np.where(legal_mask > 0, q, -999)
            optimal_q = legal_q.max()
            pred_q = q[pred_idx]
            gap = optimal_q - pred_q
        else:
            legal_q = np.where(legal_mask > 0, q, 999)
            optimal_q = legal_q.min()
            pred_q = q[pred_idx]
            gap = pred_q - optimal_q

        gaps.append(gap)
        errors.append(pred_idx != true_idx)

    gaps = np.array(gaps)
    errors = np.array(errors)

    log(f"Total samples: {n_test:,}")
    log(f"Errors: {errors.sum():,} ({errors.mean()*100:.1f}%)")
    log(f"Mean Q-gap: {gaps.mean():.3f}")
    log(f"Median Q-gap: {np.median(gaps):.1f}")

    # Blunder analysis
    blunders = gaps > 10
    log(f"Blunders (gap > 10): {blunders.sum():,} ({blunders.mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
