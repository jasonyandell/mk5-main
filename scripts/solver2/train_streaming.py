#!/usr/bin/env python3
"""
Streaming Transformer Training for Texas 42.

Fixes OOM by streaming data from parquet files instead of loading all into memory.
Uses PyTorch IterableDataset for efficient batching.

Usage:
    python scripts/solver2/train_streaming.py --epochs 10 --samples-per-epoch 1000000
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, IterableDataset

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
# Trump Rank (same as train_transformer.py)
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


# Token types and embedding sizes
TOKEN_TYPE_CONTEXT = 0
TOKEN_TYPE_PLAYER0 = 1
TOKEN_TYPE_TRICK_P0 = 5
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}


# =============================================================================
# Streaming Dataset
# =============================================================================

class StreamingParquetDataset(IterableDataset):
    """
    Streams data from parquet files without loading everything into memory.

    Each worker gets a subset of files to process.
    Samples are yielded one at a time and batched by DataLoader.
    """

    def __init__(
        self,
        data_dir: Path,
        seed_range: tuple[int, int],
        samples_per_file: int = 10000,
        shuffle_files: bool = True,
        rng_seed: int = 42,
    ):
        self.data_dir = data_dir
        self.seed_range = seed_range
        self.samples_per_file = samples_per_file
        self.shuffle_files = shuffle_files
        self.rng_seed = rng_seed

        # Find relevant files
        self.files = self._find_files()

    def _find_files(self) -> list[Path]:
        parquet_files = sorted(self.data_dir.glob("seed_*.parquet"))
        relevant = []
        for f in parquet_files:
            parts = f.stem.split("_")
            if len(parts) >= 2:
                try:
                    seed = int(parts[1])
                    if self.seed_range[0] <= seed <= self.seed_range[1]:
                        relevant.append(f)
                except ValueError:
                    pass
        return relevant

    def __iter__(self) -> Iterator[tuple]:
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single process
            files = self.files
            worker_id = 0
        else:
            # Multi-process: split files among workers
            per_worker = int(math.ceil(len(self.files) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.files))
            files = self.files[start:end]
            worker_id = worker_info.id

        # Shuffle files for this epoch
        rng = np.random.default_rng(self.rng_seed + worker_id)
        if self.shuffle_files:
            files = list(files)
            rng.shuffle(files)

        for f in files:
            yield from self._process_file(f, rng)

    def _process_file(self, file_path: Path, rng: np.random.Generator) -> Iterator[tuple]:
        """Process one parquet file and yield samples."""
        try:
            pf = pq.ParquetFile(file_path)
            meta = pf.schema_arrow.metadata or {}
            seed = int(meta.get(b"seed", b"0").decode())
            decl_id = int(meta.get(b"decl_id", b"0").decode())

            # Read file
            df = pd.read_parquet(file_path)
            states = df["state"].values.astype(np.int64)

            mv_cols = [f"mv{i}" for i in range(7)]
            q_values = np.stack([df[c].values for c in mv_cols], axis=1)

            hands = deal_from_seed(seed)
            hands_flat = np.array([hands[p][i] for p in range(4) for i in range(7)])

            n_states = len(states)

            # Sample indices
            if n_states > self.samples_per_file:
                indices = rng.choice(n_states, size=self.samples_per_file, replace=False)
            else:
                indices = np.arange(n_states)
                rng.shuffle(indices)

            # Precompute domino features
            domino_features = np.zeros((28, 5), dtype=np.int64)
            for i, global_id in enumerate(hands_flat):
                domino_features[i, 0] = DOMINO_HIGH[global_id]
                domino_features[i, 1] = DOMINO_LOW[global_id]
                domino_features[i, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
                domino_features[i, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
                domino_features[i, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]

            for idx in indices:
                sample = self._tokenize_sample(
                    states[idx], q_values[idx], decl_id, hands, hands_flat, domino_features
                )
                if sample is not None:
                    yield sample

        except Exception as e:
            # Skip problematic files
            pass

    def _tokenize_sample(
        self,
        state: int,
        q_vals: np.ndarray,
        decl_id: int,
        hands: list[list[int]],
        hands_flat: np.ndarray,
        domino_features: np.ndarray,
    ) -> tuple | None:
        """Tokenize a single sample."""
        # Legal mask
        legal_mask = (q_vals != -128).astype(np.float32)
        if not legal_mask.any():
            return None

        # Extract state fields
        remaining = [(state >> (p * 7)) & 0x7F for p in range(4)]
        leader = (state >> 28) & 0x3
        trick_len = (state >> 30) & 0x3
        p0_local = (state >> 32) & 0x7
        p1_local = (state >> 35) & 0x7
        p2_local = (state >> 38) & 0x7

        current_player = (leader + trick_len) % 4
        team = current_player % 2

        # Find target (team-aware)
        q_int = q_vals.astype(np.int32)
        if team == 0:
            q_for_argmax = np.where(legal_mask > 0, q_int, -129)
        else:
            q_for_argmax = np.where(legal_mask > 0, -q_int, -129)
        target = q_for_argmax.argmax()

        # Build tokens
        MAX_TOKENS = 32
        N_FEATURES = 12
        tokens = np.zeros((MAX_TOKENS, N_FEATURES), dtype=np.int64)
        mask = np.zeros(MAX_TOKENS, dtype=np.float32)

        normalized_leader = (leader - current_player + 4) % 4

        # Token 0: Context
        tokens[0, 9] = TOKEN_TYPE_CONTEXT
        tokens[0, 10] = decl_id
        tokens[0, 11] = normalized_leader
        mask[0] = 1.0

        # Tokens 1-28: Hand positions
        for p in range(4):
            normalized_player = (p - current_player + 4) % 4
            for local_idx in range(7):
                flat_idx = p * 7 + local_idx
                token_idx = 1 + flat_idx

                tokens[token_idx, 0] = domino_features[flat_idx, 0]
                tokens[token_idx, 1] = domino_features[flat_idx, 1]
                tokens[token_idx, 2] = domino_features[flat_idx, 2]
                tokens[token_idx, 3] = domino_features[flat_idx, 3]
                tokens[token_idx, 4] = domino_features[flat_idx, 4]
                tokens[token_idx, 5] = normalized_player
                tokens[token_idx, 6] = 1 if normalized_player == 0 else 0
                tokens[token_idx, 7] = 1 if normalized_player == 2 else 0
                tokens[token_idx, 8] = (remaining[p] >> local_idx) & 1
                tokens[token_idx, 9] = TOKEN_TYPE_PLAYER0 + p
                tokens[token_idx, 10] = decl_id
                tokens[token_idx, 11] = normalized_leader
                mask[token_idx] = 1.0

        # Tokens 29-31: Trick plays
        trick_plays = [p0_local, p1_local, p2_local]
        trick_token_types = [TOKEN_TYPE_TRICK_P0, TOKEN_TYPE_TRICK_P0 + 1, TOKEN_TYPE_TRICK_P0 + 2]

        for trick_pos in range(trick_len):
            local_idx = trick_plays[trick_pos]
            if local_idx >= 7:
                continue

            token_idx = 29 + trick_pos
            play_player = (leader + trick_pos) % 4
            global_id = hands[play_player][local_idx]
            normalized_pp = (play_player - current_player + 4) % 4

            tokens[token_idx, 0] = DOMINO_HIGH[global_id]
            tokens[token_idx, 1] = DOMINO_LOW[global_id]
            tokens[token_idx, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
            tokens[token_idx, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
            tokens[token_idx, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]
            tokens[token_idx, 5] = normalized_pp
            tokens[token_idx, 6] = 1 if normalized_pp == 0 else 0
            tokens[token_idx, 7] = 1 if normalized_pp == 2 else 0
            tokens[token_idx, 8] = 0
            tokens[token_idx, 9] = trick_token_types[trick_pos]
            tokens[token_idx, 10] = decl_id
            tokens[token_idx, 11] = normalized_leader
            mask[token_idx] = 1.0

        return (
            torch.from_numpy(tokens),
            torch.from_numpy(mask),
            torch.tensor(0, dtype=torch.long),  # current_player normalized to 0
            torch.tensor(target, dtype=torch.long),
            torch.from_numpy(legal_mask),
            torch.from_numpy(q_int.astype(np.float32)),
            torch.tensor(team, dtype=torch.long),
        )


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
            self.embed_dim // 6 * 4 +  # high, low, trump_rank, token_type
            self.embed_dim // 12 * 6 + # is_double, count, player, is_current, is_partner, is_remaining
            self.embed_dim // 6 +      # decl
            self.embed_dim // 12       # leader
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

        # Current player's hand is always at position 0 after normalization
        start_indices = 1 + current_player * 7
        offsets = torch.arange(7, device=device)
        gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        hand_repr = torch.gather(x, dim=1, index=gather_indices)
        logits = self.output_proj(hand_repr).squeeze(-1)

        return logits


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model: DominoTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 3.0,
    soft_weight: float = 0.7,
    max_batches: int | None = None,
) -> tuple[float, float, int]:
    """Train for one epoch."""
    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        tokens, masks, players, targets, legal, qvals, teams = batch
        tokens = tokens.to(device)
        masks = masks.to(device)
        players = players.to(device)
        targets = targets.to(device)
        legal = legal.to(device)
        qvals = qvals.to(device)
        teams = teams.to(device)

        optimizer.zero_grad()
        logits = model(tokens, masks, players)
        logits_masked = logits.masked_fill(legal == 0, float('-inf'))

        # Hard loss
        hard_loss = F.cross_entropy(logits_masked, targets)

        # Soft loss
        team_sign = torch.where(teams == 0, 1.0, -1.0).unsqueeze(1)
        q_for_soft = qvals * team_sign
        q_masked = torch.where(legal > 0, q_for_soft, torch.tensor(float('-inf'), device=device))
        soft_targets = F.softmax(q_masked / temperature, dim=-1)
        soft_targets = soft_targets.clamp(min=1e-8)
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)
        log_probs = F.log_softmax(logits_masked, dim=-1)
        log_probs_safe = log_probs.masked_fill(legal == 0, 0.0)
        soft_loss = -(soft_targets * log_probs_safe).sum(dim=-1).mean()

        loss = (1 - soft_weight) * hard_loss + soft_weight * soft_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)

        with torch.no_grad():
            preds = logits_masked.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += len(targets)

        if (batch_idx + 1) % 100 == 0:
            log(f"    Batch {batch_idx+1}: loss={loss.item():.4f}, acc={total_correct/total_samples:.2%}")

    acc = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss, acc, total_samples


def evaluate(
    model: DominoTransformer,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, int]:
    """Evaluate model."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break

            tokens, masks, players, targets, legal, qvals, teams = batch
            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)
            targets = targets.to(device)
            legal = legal.to(device)

            logits = model(tokens, masks, players)
            logits_masked = logits.masked_fill(legal == 0, float('-inf'))

            loss = F.cross_entropy(logits_masked, targets)
            total_loss += loss.item() * len(targets)

            preds = logits_masked.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += len(targets)

    acc = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss, acc, total_samples


def main():
    parser = argparse.ArgumentParser(description="Streaming transformer training")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples-per-file", type=int, default=50000,
                        help="Max samples to load per parquet file")
    parser.add_argument("--eval-batches", type=int, default=100,
                        help="Batches for evaluation (streaming can't easily get full test set)")
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--soft-weight", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("Using CPU")

    log("\n=== Configuration ===")
    log(f"Batch size: {args.batch_size}")
    log(f"Samples per file: {args.samples_per_file}")
    log(f"Learning rate: {args.lr}")
    log(f"Epochs: {args.epochs}")
    log(f"Soft weight: {args.soft_weight}")

    data_dir = Path(args.data_dir)

    # Create streaming datasets
    train_dataset = StreamingParquetDataset(
        data_dir, seed_range=(0, 89),
        samples_per_file=args.samples_per_file,
        shuffle_files=True,
        rng_seed=args.seed,
    )
    test_dataset = StreamingParquetDataset(
        data_dir, seed_range=(90, 99),
        samples_per_file=args.samples_per_file,
        shuffle_files=False,
        rng_seed=args.seed,
    )

    log(f"Train files: {len(train_dataset.files)}")
    log(f"Test files: {len(test_dataset.files)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

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

    # Training loop
    log("\n=== Training ===")
    log(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Samples':>10} | {'Test Acc':>9}")
    log("-" * 60)

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Recreate dataset with new RNG seed for shuffling
        train_dataset = StreamingParquetDataset(
            data_dir, seed_range=(0, 89),
            samples_per_file=args.samples_per_file,
            shuffle_files=True,
            rng_seed=args.seed + epoch,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        train_loss, train_acc, train_samples = train_epoch(
            model, train_loader, optimizer, device,
            args.temperature, args.soft_weight
        )

        test_loss, test_acc, test_samples = evaluate(
            model, test_loader, device,
            max_batches=args.eval_batches
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
                log(f"  → Saved best model")

        log(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2%} | {train_samples:>10,} | {test_acc:8.2%}")

    log(f"\nBest test accuracy: {best_test_acc:.2%}")


if __name__ == "__main__":
    main()
