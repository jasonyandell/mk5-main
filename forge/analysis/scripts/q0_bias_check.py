#!/usr/bin/env python3
"""Check if the model has a systematic bias against Q[0].

Simple test: Compare average Q[0] to average Q[other] across all validation
samples, for legal actions only.

If there's positional bias, Q[0] will be systematically lower.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

from forge.ml.data import DominoDataset
from forge.ml.module import DominoLightningModule

MODEL_PATH = Path(PROJECT_ROOT) / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
DATA_PATH = Path(PROJECT_ROOT) / "data/tokenized-full"


def load_model(device: str = "cuda") -> DominoLightningModule:
    """Load the trained Q-value model."""
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    if 'rng_state' in checkpoint:
        del checkpoint['rng_state']
    hparams = checkpoint.get('hyper_parameters', {})
    model = DominoLightningModule(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    return model


def analyze_q_bias():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(device)

    val_dataset = DominoDataset(str(DATA_PATH), split='val', mmap=False)
    print(f"Validation samples: {len(val_dataset):,}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Collect Q-value statistics per slot
    # For each slot: [sum of Q when legal, count when legal]
    q_stats = {i: {'sum': 0.0, 'count': 0} for i in range(28)}

    # Also track Q-value rank: when slot is legal, what rank does it get?
    rank_stats = {i: [] for i in range(28)}

    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyzing Q-values"):
            tokens, masks, players, targets, legal, qvals, teams, values = batch

            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)
            legal = legal.to(device)

            model_q, _ = model(tokens, masks, players)

            # Move to CPU for analysis
            model_q = model_q.cpu().numpy()
            legal_np = legal.cpu().numpy()

            batch_size = len(model_q)
            for i in range(batch_size):
                q = model_q[i]
                leg = legal_np[i]

                legal_indices = np.where(leg > 0)[0]
                if len(legal_indices) == 0:
                    continue

                # Get Q-values for legal actions
                legal_qs = q[legal_indices]

                # Rank legal actions by Q-value (descending)
                ranks = np.argsort(-legal_qs)  # indices that sort descending

                for slot in legal_indices:
                    q_stats[slot]['sum'] += q[slot]
                    q_stats[slot]['count'] += 1

                    # What rank does this slot have among legal actions?
                    slot_in_legal = np.where(legal_indices == slot)[0][0]
                    rank = np.where(ranks == slot_in_legal)[0][0]  # 0 = best
                    rank_stats[slot].append(rank)

            n_samples += batch_size
            if n_samples >= 100000:  # Sample first 100k for speed
                break

    print("\n" + "=" * 70)
    print("Q-VALUE BIAS ANALYSIS")
    print("=" * 70)

    print("\n### Average Q-value by slot (when legal) ###")
    print("Slot  Avg_Q      Count      Avg_Rank")
    print("-" * 50)

    for slot in range(28):
        if q_stats[slot]['count'] > 0:
            avg_q = q_stats[slot]['sum'] / q_stats[slot]['count']
            avg_rank = np.mean(rank_stats[slot]) if rank_stats[slot] else 0
            print(f"{slot:4d}  {avg_q:8.3f}   {q_stats[slot]['count']:8d}   {avg_rank:.3f}")

    # Summary statistics
    print("\n### Summary ###")
    slot0_avg = q_stats[0]['sum'] / q_stats[0]['count'] if q_stats[0]['count'] > 0 else 0
    other_avgs = []
    for slot in range(1, 28):
        if q_stats[slot]['count'] > 0:
            other_avgs.append(q_stats[slot]['sum'] / q_stats[slot]['count'])

    print(f"Slot 0 average Q: {slot0_avg:.3f}")
    print(f"Slots 1-27 average Q: {np.mean(other_avgs):.3f}")
    print(f"Difference: {slot0_avg - np.mean(other_avgs):.3f}")

    # Rank analysis
    slot0_avg_rank = np.mean(rank_stats[0]) if rank_stats[0] else 0
    other_ranks = []
    for slot in range(1, 28):
        if rank_stats[slot]:
            other_ranks.append(np.mean(rank_stats[slot]))

    print(f"\nSlot 0 average rank (0=best): {slot0_avg_rank:.3f}")
    print(f"Slots 1-27 average rank: {np.mean(other_ranks):.3f}")


if __name__ == "__main__":
    analyze_q_bias()
