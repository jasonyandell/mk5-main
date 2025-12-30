#!/usr/bin/env python3
"""
Train transformer from pre-tokenized data.

Loads numpy arrays saved by pretokenize.py for fast training.
No parquet reading or tokenization overhead - just load and train.

Usage:
    # Train from pre-tokenized data
    python scripts/solver2/train_pretokenized.py --data-dir data/solver2/tokenized --epochs 10

    # Quick test
    python scripts/solver2/train_pretokenized.py --data-dir scratch/tokenized --epochs 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def log(msg: str) -> None:
    elapsed = time.time() - LOG_START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


LOG_START_TIME = time.time()


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
# Data Loading
# =============================================================================

def load_split(data_dir: Path, split: str, max_samples: int | None = None) -> dict:
    """Load pre-tokenized data for a split.

    Uses memory-mapped files to avoid loading entire dataset into RAM.
    """
    split_dir = data_dir / split

    log(f"Loading {split} data from {split_dir}...")
    t0 = time.time()

    # Use memory-mapped mode to avoid loading entire file into RAM
    # This is crucial for large datasets that exceed available memory
    tokens = np.load(split_dir / "tokens.npy", mmap_mode='r')
    masks = np.load(split_dir / "masks.npy", mmap_mode='r')
    players = np.load(split_dir / "players.npy", mmap_mode='r')
    targets = np.load(split_dir / "targets.npy", mmap_mode='r')
    legal = np.load(split_dir / "legal.npy", mmap_mode='r')
    qvals = np.load(split_dir / "qvals.npy", mmap_mode='r')
    teams = np.load(split_dir / "teams.npy", mmap_mode='r')

    n_samples = len(targets)
    if max_samples and max_samples < n_samples:
        # Subsample for memory-constrained training
        log(f"  Subsampling {max_samples:,} from {n_samples:,} samples")
        n_samples = max_samples

    log(f"  Loaded {n_samples:,} samples in {time.time()-t0:.1f}s (memory-mapped)")
    log(f"  Tokens shape: {tokens.shape}, dtype: {tokens.dtype}")

    return {
        "tokens": tokens[:n_samples],
        "masks": masks[:n_samples],
        "players": players[:n_samples],
        "targets": targets[:n_samples],
        "legal": legal[:n_samples],
        "qvals": qvals[:n_samples],
        "teams": teams[:n_samples],
        "n_samples": n_samples,
    }


class Int8Dataset(Dataset):
    """Dataset that stores int8/memmap and converts to tensors on-the-fly.

    This avoids the 8x memory overhead of storing everything as int64.
    Works with memory-mapped files for large datasets.
    """

    def __init__(self, data: dict, include_qvals: bool = True):
        self.tokens = data["tokens"]        # int8 or memmap
        self.masks = data["masks"]          # int8/float32 or memmap
        self.players = data["players"]      # int8 or memmap
        self.targets = data["targets"]      # int8 or memmap
        self.legal = data["legal"]          # int8/float32 or memmap
        self.qvals = data["qvals"] if include_qvals else None
        self.teams = data["teams"] if include_qvals else None
        self.include_qvals = include_qvals
        self.n_samples = data.get("n_samples", len(data["targets"]))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Read from memmap and convert to tensor
        # np.array() forces a copy from memmap, which is needed for PyTorch
        tokens = torch.from_numpy(np.array(self.tokens[idx], dtype=np.int64))
        masks = torch.from_numpy(np.array(self.masks[idx], dtype=np.float32))
        players = torch.tensor(int(self.players[idx]), dtype=torch.long)
        targets = torch.tensor(int(self.targets[idx]), dtype=torch.long)
        legal = torch.from_numpy(np.array(self.legal[idx], dtype=np.float32))

        if self.include_qvals:
            qvals = torch.from_numpy(np.array(self.qvals[idx], dtype=np.float32))
            teams = torch.tensor(int(self.teams[idx]), dtype=torch.long)
            return tokens, masks, players, targets, legal, qvals, teams
        else:
            return tokens, masks, players, targets, legal


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
) -> tuple[float, float, int]:
    """Train for one epoch."""
    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for batch in loader:
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

    acc = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    return avg_loss, acc, total_samples


def evaluate(
    model: DominoTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
    """Evaluate model."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            tokens, masks, players, targets, legal = batch[:5]
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
    parser = argparse.ArgumentParser(description="Train from pre-tokenized data")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with pre-tokenized data (from pretokenize.py)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--soft-weight", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0=main thread, recommended for memmap)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit training samples (for memory-constrained systems)")
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="Limit test samples")
    parser.add_argument("--high-regret-file", type=str, default=None,
                        help="NPZ file with high-regret indices from mine_high_regret.py")
    parser.add_argument("--high-regret-weight", type=float, default=3.0,
                        help="Weight multiplier for high-regret samples (default: 3.0)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("Using CPU")

    log("\n=== Configuration ===")
    log(f"Data dir: {args.data_dir}")
    log(f"Batch size: {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log(f"Epochs: {args.epochs}")
    log(f"Soft weight: {args.soft_weight}")

    data_dir = Path(args.data_dir)

    # Load data
    train_data = load_split(data_dir, "train", args.max_train_samples)
    test_data = load_split(data_dir, "test", args.max_test_samples)

    # Create datasets
    # Use Int8Dataset to avoid 8x memory overhead of int64
    train_dataset = Int8Dataset(train_data, include_qvals=True)
    test_dataset = Int8Dataset(test_data, include_qvals=False)

    # Create weighted sampler if high-regret file provided
    sampler = None
    if args.high_regret_file:
        log(f"\nLoading high-regret indices from {args.high_regret_file}...")
        hr_data = np.load(args.high_regret_file)
        hr_indices = hr_data["high_regret_indices"]
        hr_threshold = float(hr_data["threshold"])
        hr_split = str(hr_data["split"])

        # Verify the file is for training data
        if hr_split != "train":
            log(f"  WARNING: High-regret file is for '{hr_split}' split, not 'train'")
            log(f"  You may want to run mine_high_regret.py on the train split")

        # Filter indices to valid range (in case of subsampling)
        n_train = train_data["n_samples"]
        valid_mask = hr_indices < n_train
        hr_indices = hr_indices[valid_mask]

        log(f"  Threshold: {hr_threshold}")
        log(f"  High-regret samples: {len(hr_indices):,} ({len(hr_indices)/n_train*100:.1f}%)")
        log(f"  Weight multiplier: {args.high_regret_weight}")

        # Create sample weights
        weights = np.ones(n_train, dtype=np.float32)
        weights[hr_indices] = args.high_regret_weight

        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=n_train,
            replacement=True,
        )
        log(f"  Using weighted sampling for training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),  # Don't shuffle if using sampler
        sampler=sampler,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    log("\n=== Training ===")
    log(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>9} | {'Time':>6}")
    log("-" * 55)

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_samples = train_epoch(
            model, train_loader, optimizer, device,
            args.temperature, args.soft_weight
        )

        test_loss, test_acc, test_samples = evaluate(model, test_loader, device)

        scheduler.step()

        epoch_time = time.time() - t0

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

        log(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2%} | {test_acc:8.2%} | {epoch_time:5.1f}s")

    log(f"\nBest test accuracy: {best_test_acc:.2%}")

    # Q-Gap Analysis
    log("\n=== Q-Gap Analysis ===")
    model.eval()

    all_preds = []
    all_targets = []
    all_qvals = []
    all_legal = []
    all_teams = []

    with torch.no_grad():
        for batch in test_loader:
            tokens, masks, players, targets, legal = batch[:5]
            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)
            legal_gpu = legal.to(device)

            logits = model(tokens, masks, players)
            logits_masked = logits.masked_fill(legal_gpu == 0, float('-inf'))
            preds = logits_masked.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    n_test = test_data["n_samples"]

    # Load Q-values and compute gaps in batches (memory efficient)
    qvals = np.array(test_data["qvals"][:n_test], dtype=np.float32)
    legal = np.array(test_data["legal"][:n_test], dtype=np.float32)
    teams = np.array(test_data["teams"][:n_test])

    # Vectorized Q-gap computation
    team_0_mask = (teams == 0)

    # Mask illegal moves: team 0 wants max (so illegal = -inf), team 1 wants min (illegal = +inf)
    q_masked = np.where(
        legal > 0,
        qvals,
        np.where(team_0_mask[:, np.newaxis], -1000.0, 1000.0)
    )

    # Find optimal Q-value per sample
    optimal_q = np.where(
        team_0_mask,
        q_masked.max(axis=1),  # Team 0 wants max
        q_masked.min(axis=1),  # Team 1 wants min
    )

    # Get predicted Q-value
    pred_q = qvals[np.arange(n_test), preds]

    # Compute gap (always positive: how much worse is our choice?)
    gaps = np.where(
        team_0_mask,
        optimal_q - pred_q,  # Team 0: higher is better
        pred_q - optimal_q,  # Team 1: lower is better
    )

    correct = (preds == targets)
    errors = ~correct

    log(f"Total samples: {n_test:,}")
    log(f"Accuracy: {correct.mean()*100:.2f}%")
    log(f"Mean Q-gap: {gaps.mean():.2f}")
    log(f"Median Q-gap: {np.median(gaps):.1f}")

    # Blunder analysis
    blunders = gaps > 10
    log(f"Blunders (gap > 10): {blunders.sum():,} ({blunders.mean()*100:.2f}%)")
    log(f"  Error mean gap: {gaps[errors].mean():.2f}")
    log(f"  Correct mean gap: {gaps[correct].mean():.2f}")


if __name__ == "__main__":
    main()
