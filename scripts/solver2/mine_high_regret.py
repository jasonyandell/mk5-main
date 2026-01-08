#!/usr/bin/env python3
"""
Mine high-regret samples from a trained model.

Runs inference on the dataset and identifies samples where the model
makes suboptimal choices (high Q-gap). These can be oversampled in
subsequent training to reduce blunders.

Usage:
    # Mine high-regret samples from test set
    python scripts/solver2/mine_high_regret.py \
        --model data/solver2/transformer_2m.pt \
        --data-dir data/solver2/tokenized \
        --split test \
        --output scratch/high_regret.npz

    # With custom threshold
    python scripts/solver2/mine_high_regret.py \
        --model data/solver2/transformer_2m.pt \
        --data-dir data/solver2/tokenized \
        --threshold 10 \
        --output scratch/high_regret.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def log(msg: str) -> None:
    elapsed = time.time() - LOG_START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


LOG_START_TIME = time.time()


# =============================================================================
# Model (copied from train_pretokenized.py)
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
    """Load pre-tokenized data for a split."""
    split_dir = data_dir / split

    log(f"Loading {split} data from {split_dir}...")
    t0 = time.time()

    tokens = np.load(split_dir / "tokens.npy", mmap_mode='r')
    masks = np.load(split_dir / "masks.npy", mmap_mode='r')
    players = np.load(split_dir / "players.npy", mmap_mode='r')
    targets = np.load(split_dir / "targets.npy", mmap_mode='r')
    legal = np.load(split_dir / "legal.npy", mmap_mode='r')
    qvals = np.load(split_dir / "qvals.npy", mmap_mode='r')
    teams = np.load(split_dir / "teams.npy", mmap_mode='r')

    n_samples = len(targets)
    if max_samples and max_samples < n_samples:
        log(f"  Subsampling {max_samples:,} from {n_samples:,} samples")
        n_samples = max_samples

    log(f"  Loaded {n_samples:,} samples in {time.time()-t0:.1f}s (memory-mapped)")

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


class InferenceDataset(Dataset):
    """Dataset for inference that returns all fields needed for Q-gap analysis."""

    def __init__(self, data: dict):
        self.tokens = data["tokens"]
        self.masks = data["masks"]
        self.players = data["players"]
        self.targets = data["targets"]
        self.legal = data["legal"]
        self.qvals = data["qvals"]
        self.teams = data["teams"]
        self.n_samples = data["n_samples"]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        tokens = torch.from_numpy(np.array(self.tokens[idx], dtype=np.int64))
        masks = torch.from_numpy(np.array(self.masks[idx], dtype=np.float32))
        players = torch.tensor(int(self.players[idx]), dtype=torch.long)
        targets = torch.tensor(int(self.targets[idx]), dtype=torch.long)
        legal = torch.from_numpy(np.array(self.legal[idx], dtype=np.float32))
        # Keep qvals and teams as int8 for memory efficiency
        qvals = torch.from_numpy(np.array(self.qvals[idx], dtype=np.int8))
        teams = torch.tensor(int(self.teams[idx]), dtype=torch.long)

        return tokens, masks, players, targets, legal, qvals, teams


def compute_qgap_batch(
    pred_indices: np.ndarray,
    target_indices: np.ndarray,
    qvals: np.ndarray,
    legal_masks: np.ndarray,
    teams: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Q-gap for a batch of samples.

    Q-gap = optimal_q - chosen_q (always >= 0)

    For team 0: maximize Q, so optimal = max(legal_q)
    For team 1: minimize Q, so optimal = min(legal_q)

    Returns:
        gaps: Array of Q-gaps (float)
        correct: Array of booleans (pred == target)
    """
    batch_size = len(pred_indices)

    # Convert to float32 for computation
    qvals_f = qvals.astype(np.float32)
    legal_f = legal_masks.astype(np.float32)

    # Mask illegal moves with extreme values
    # For team 0 (maximize): illegal = -inf
    # For team 1 (minimize): illegal = +inf
    team_0_mask = (teams == 0)

    # Create masked Q-values
    q_masked = np.where(
        legal_f > 0,
        qvals_f,
        np.where(team_0_mask[:, np.newaxis], -1000.0, 1000.0)
    )

    # Find optimal Q-value
    optimal_q = np.where(
        team_0_mask,
        q_masked.max(axis=1),  # Team 0 wants max
        q_masked.min(axis=1),  # Team 1 wants min
    )

    # Get predicted Q-value
    pred_q = qvals_f[np.arange(batch_size), pred_indices]

    # Compute gap (always positive: how much worse is our choice?)
    gaps = np.where(
        team_0_mask,
        optimal_q - pred_q,  # Team 0: higher is better
        pred_q - optimal_q,  # Team 1: lower is better
    )

    # Correct predictions
    correct = (pred_indices == target_indices)

    return gaps, correct


def main():
    parser = argparse.ArgumentParser(description="Mine high-regret samples")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with pre-tokenized data")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to analyze (train/test)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for results (.npz)")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Q-gap threshold for 'high regret' (default: 10)")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to analyze")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("Using CPU")

    # Load model
    log(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)

    model = DominoTransformer(
        embed_dim=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    log(f"  Model trained to epoch {checkpoint.get('epoch', '?')}")
    log(f"  Train acc: {checkpoint.get('train_acc', 0):.2%}")
    log(f"  Test acc: {checkpoint.get('test_acc', 0):.2%}")

    # Load data
    data_dir = Path(args.data_dir)
    data = load_split(data_dir, args.split, args.max_samples)
    n_samples = data["n_samples"]

    dataset = InferenceDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep order for index tracking
        num_workers=0,
        pin_memory=True,
    )

    log(f"\n=== Mining High-Regret Samples ===")
    log(f"Threshold: {args.threshold} points")

    # Run inference and compute Q-gaps
    all_preds = []
    all_targets = []
    all_gaps = []
    all_correct = []
    all_qvals = []
    all_legal = []
    all_teams = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            tokens, masks, players, targets, legal, qvals, teams = batch

            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)
            legal_gpu = legal.to(device)

            # Get predictions
            logits = model(tokens, masks, players)
            logits_masked = logits.masked_fill(legal_gpu == 0, float('-inf'))
            preds = logits_masked.argmax(dim=-1).cpu().numpy()

            # Convert to numpy
            targets_np = targets.numpy()
            qvals_np = qvals.numpy()
            legal_np = legal.numpy()
            teams_np = teams.numpy()

            # Compute Q-gaps for this batch
            gaps, correct = compute_qgap_batch(
                preds, targets_np, qvals_np, legal_np, teams_np
            )

            all_preds.extend(preds)
            all_targets.extend(targets_np)
            all_gaps.extend(gaps)
            all_correct.extend(correct)

            if (batch_idx + 1) % 50 == 0:
                n_processed = min((batch_idx + 1) * args.batch_size, n_samples)
                log(f"  Processed {n_processed:,} / {n_samples:,} samples")

    # Convert to numpy arrays
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    gaps = np.array(all_gaps)
    correct = np.array(all_correct)

    # Compute statistics
    log(f"\n=== Results ===")
    log(f"Total samples: {n_samples:,}")

    accuracy = correct.mean()
    log(f"Accuracy: {accuracy:.2%} ({correct.sum():,} correct)")

    log(f"\nQ-Gap Statistics:")
    log(f"  Mean: {gaps.mean():.2f}")
    log(f"  Median: {np.median(gaps):.1f}")
    log(f"  Std: {gaps.std():.2f}")
    log(f"  Min: {gaps.min():.1f}, Max: {gaps.max():.1f}")

    # Distribution of gaps
    log(f"\nQ-Gap Distribution:")
    for threshold in [0, 1, 5, 10, 20, 50]:
        count = (gaps > threshold).sum()
        pct = count / n_samples * 100
        log(f"  Gap > {threshold:2d}: {count:7,} ({pct:5.1f}%)")

    # Find high-regret samples
    high_regret_mask = gaps > args.threshold
    high_regret_indices = np.where(high_regret_mask)[0]
    high_regret_gaps = gaps[high_regret_mask]

    log(f"\nHigh-Regret Samples (gap > {args.threshold}):")
    log(f"  Count: {len(high_regret_indices):,} ({len(high_regret_indices)/n_samples*100:.1f}%)")
    log(f"  Mean gap: {high_regret_gaps.mean():.2f}")

    # Analyze errors vs correct predictions
    errors = ~correct
    log(f"\nErrors Analysis:")
    log(f"  Error count: {errors.sum():,} ({errors.mean()*100:.1f}%)")
    log(f"  Error mean gap: {gaps[errors].mean():.2f}")
    log(f"  Correct mean gap: {gaps[correct].mean():.2f}")

    # Check for suspiciously high gaps on correct predictions
    correct_high_gap = correct & (gaps > args.threshold)
    if correct_high_gap.sum() > 0:
        log(f"\n  WARNING: {correct_high_gap.sum():,} correct predictions have gap > {args.threshold}")
        log(f"  This might indicate a bug in Q-gap computation or target generation")

        # Sample some for inspection
        sample_indices = np.where(correct_high_gap)[0][:5]
        log(f"\n  Sample high-gap correct predictions:")
        for idx in sample_indices:
            q = np.array(data["qvals"][idx])
            legal_mask = np.array(data["legal"][idx])
            team = int(data["teams"][idx])
            pred = preds[idx]
            target = targets[idx]
            gap = gaps[idx]

            legal_q = q[legal_mask > 0]
            log(f"    idx={idx}: pred={pred}, target={target}, team={team}, gap={gap:.1f}")
            log(f"      Q-vals: {q.tolist()}")
            log(f"      Legal:  {legal_mask.tolist()}")
            log(f"      Legal Q: {legal_q.tolist()}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        high_regret_indices=high_regret_indices,
        high_regret_gaps=high_regret_gaps,
        all_gaps=gaps,
        all_correct=correct,
        threshold=args.threshold,
        split=args.split,
        model_path=args.model,
        n_samples=n_samples,
        accuracy=accuracy,
    )

    log(f"\nSaved results to {output_path}")
    log(f"  High-regret indices: {len(high_regret_indices):,}")
    log(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
