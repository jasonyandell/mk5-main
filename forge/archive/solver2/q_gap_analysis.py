#!/usr/bin/env python3
"""
Q-Gap Analysis: Verify prediction quality by measuring Q-value gaps.

For each prediction (correct or not), compute:
  Q-gap = optimal_Q - predicted_Q

This tells us if "errors" are actually bad or just different-but-equal choices.

Target: mean Q-gap under 1 point.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.solver2.train_transformer import (
    DominoTransformer,
    load_data,
    LOG_START_TIME,
)
from scripts.solver2.rng import deal_from_seed


def log(msg: str) -> None:
    elapsed = time.time() - LOG_START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Q-gap analysis")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    data_dir = Path(args.data_dir)

    # Load training data
    train_tokens, train_masks, train_players, train_targets, train_legal = load_data(
        data_dir, (0, 89), args.max_samples, args.max_files, rng, "Train"
    )

    # Load test data
    test_max_files = max(1, args.max_files // 9) if args.max_files else None
    test_tokens, test_masks, test_players, test_targets, test_legal = load_data(
        data_dir, (90, 99), args.max_samples, test_max_files, rng, "Test"
    )

    log(f"\nTrain: {len(train_targets):,}, Test: {len(test_targets):,}")

    # Create and train model
    from torch.utils.data import DataLoader, TensorDataset

    train_dataset = TensorDataset(
        torch.tensor(train_tokens),
        torch.tensor(train_masks),
        torch.tensor(train_players),
        torch.tensor(train_targets),
        torch.tensor(train_legal),
    )
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

    model = DominoTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    log(f"\n=== Training ({args.epochs} epochs) ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for tokens, masks, players, targets, legal in train_loader:
            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)
            targets = targets.to(device)
            legal = legal.to(device)

            optimizer.zero_grad()
            logits = model(tokens, masks, players)
            logits_masked = logits.masked_fill(legal == 0, float('-inf'))
            loss = F.cross_entropy(logits_masked, targets)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            log(f"  Epoch {epoch}: loss = {loss.item():.4f}")

    # Q-gap analysis on test set
    log("\n=== Q-Gap Analysis ===")

    # We need to reload test data with Q-values to compute gaps
    # Load raw parquet files for Q-values
    parquet_files = sorted(data_dir.glob("seed_*.parquet"))
    test_files = [f for f in parquet_files
                  if 90 <= int(f.stem.split("_")[1]) <= 99][:test_max_files]

    all_gaps = []
    all_is_error = []
    blunder_cases = []

    model.eval()
    sample_offset = 0

    for f in test_files:
        pf = pq.ParquetFile(f)
        meta = pf.schema_arrow.metadata or {}
        seed = int(meta.get(b"seed", b"0").decode())
        decl_id = int(meta.get(b"decl_id", b"0").decode())

        df = pd.read_parquet(f)
        states = df["state"].values.astype(np.int64)

        # Get Q-values
        mv_cols = [f"mv{i}" for i in range(7)]
        q_values = np.stack([df[c].values for c in mv_cols], axis=1)  # (N, 7)

        # Filter to match what load_data does
        legal_masks = (q_values != -128).astype(np.float32)
        has_legal = legal_masks.any(axis=1)
        states = states[has_legal]
        q_values = q_values[has_legal]
        legal_masks = legal_masks[has_legal]

        # Sample same way
        n_states = len(states)
        if args.max_samples and n_states > args.max_samples:
            file_rng = np.random.default_rng(args.seed)
            indices = file_rng.choice(n_states, size=args.max_samples, replace=False)
            states = states[indices]
            q_values = q_values[indices]
            legal_masks = legal_masks[indices]

        n_samples = len(states)

        # Get model predictions
        end_offset = sample_offset + n_samples
        if end_offset > len(test_tokens):
            break

        with torch.no_grad():
            tokens = torch.tensor(test_tokens[sample_offset:end_offset]).to(device)
            masks = torch.tensor(test_masks[sample_offset:end_offset]).to(device)
            players = torch.tensor(test_players[sample_offset:end_offset]).to(device)
            legal_t = torch.tensor(test_legal[sample_offset:end_offset]).to(device)
            targets = test_targets[sample_offset:end_offset]

            logits = model(tokens, masks, players)
            logits = logits.masked_fill(legal_t == 0, float('-inf'))
            preds = logits.argmax(dim=-1).cpu().numpy()

        # Compute current_player and team for each sample
        leader = ((states >> 28) & 0x3).astype(np.int64)
        trick_len = ((states >> 30) & 0x3).astype(np.int64)
        current_player = (leader + trick_len) % 4
        team = current_player % 2

        # For each sample, compute Q-gap
        for i in range(n_samples):
            q = q_values[i].astype(np.int32)  # Avoid int8 overflow
            legal = legal_masks[i]
            pred_idx = preds[i]
            true_idx = targets[i]
            t = team[i]

            # Q-values are from Team 0's perspective
            # For Team 0: higher Q is better
            # For Team 1: lower Q is better (they want to minimize Team 0's score)

            if t == 0:
                # Team 0: optimal = max Q, gap = optimal - predicted
                legal_q = np.where(legal > 0, q, -999)
                optimal_q = legal_q.max()
                pred_q = q[pred_idx]
                gap = optimal_q - pred_q
            else:
                # Team 1: optimal = min Q (from their perspective, minimize Team 0's score)
                legal_q = np.where(legal > 0, q, 999)
                optimal_q = legal_q.min()
                pred_q = q[pred_idx]
                # Gap for Team 1: they want low Q, so gap = pred_q - optimal_q
                # (if they predicted higher than optimal, that's bad for them)
                gap = pred_q - optimal_q

            is_error = (pred_idx != true_idx)
            all_gaps.append(gap)
            all_is_error.append(is_error)

            # Track blunder cases (gap > 10)
            if gap > 10:
                blunder_cases.append({
                    'file': f.name,
                    'sample': i,
                    'team': t,
                    'current_player': current_player[i],
                    'q_values': q.tolist(),
                    'legal_mask': legal.tolist(),
                    'optimal_idx': true_idx,
                    'pred_idx': pred_idx,
                    'optimal_q': int(optimal_q),
                    'pred_q': int(pred_q),
                    'gap': int(gap),
                })

        sample_offset = end_offset

    gaps = np.array(all_gaps)
    is_error = np.array(all_is_error)

    log(f"\nTotal samples analyzed: {len(gaps):,}")
    log(f"Errors: {is_error.sum():,} ({is_error.mean()*100:.1f}%)")

    # Overall stats
    log(f"\n--- All Predictions ---")
    log(f"Mean Q-gap: {gaps.mean():.3f}")
    log(f"Median Q-gap: {np.median(gaps):.1f}")
    log(f"Max Q-gap: {gaps.max()}")
    log(f"Min Q-gap: {gaps.min()}")

    # Error analysis
    error_gaps = gaps[is_error]
    if len(error_gaps) > 0:
        log(f"\n--- Errors Only ---")
        log(f"Mean Q-gap: {error_gaps.mean():.3f}")

        # Bucket by gap size
        ties = (error_gaps == 0).sum()
        near = ((error_gaps >= 1) & (error_gaps <= 2)).sum()
        minor = ((error_gaps >= 3) & (error_gaps <= 5)).sum()
        bad = ((error_gaps >= 6) & (error_gaps <= 10)).sum()
        blunder = (error_gaps > 10).sum()

        total_errors = len(error_gaps)
        log(f"\nError buckets:")
        log(f"  Ties (gap=0):    {ties:5d} ({ties/total_errors*100:5.1f}%)")
        log(f"  Near (1-2):      {near:5d} ({near/total_errors*100:5.1f}%)")
        log(f"  Minor (3-5):     {minor:5d} ({minor/total_errors*100:5.1f}%)")
        log(f"  Bad (6-10):      {bad:5d} ({bad/total_errors*100:5.1f}%)")
        log(f"  Blunder (>10):   {blunder:5d} ({blunder/total_errors*100:5.1f}%)")

    # Verdict
    log(f"\n=== Verdict ===")
    mean_gap = gaps.mean()
    if mean_gap < 1.0:
        log(f"✓ SOLID: Mean Q-gap {mean_gap:.3f} < 1.0 target")
    else:
        log(f"✗ CONCERN: Mean Q-gap {mean_gap:.3f} >= 1.0 target")

    # Show blunder cases if any
    if blunder_cases:
        log(f"\n=== Blunder Cases (first 10) ===")
        for case in blunder_cases[:10]:
            log(f"\nFile: {case['file']}, Sample: {case['sample']}")
            log(f"  Team: {case['team']}, Player: {case['current_player']}")
            log(f"  Q-values: {case['q_values']}")
            log(f"  Legal mask: {case['legal_mask']}")
            log(f"  Optimal idx: {case['optimal_idx']}, Q={case['optimal_q']}")
            log(f"  Predicted idx: {case['pred_idx']}, Q={case['pred_q']}")
            log(f"  Gap: {case['gap']}")


if __name__ == "__main__":
    main()
