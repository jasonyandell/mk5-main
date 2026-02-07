#!/usr/bin/env python3
"""Compare model Q-values to oracle Q-values by slot.

If model has positional bias, correlation(model_q[0], oracle_q[0]) will be
lower than correlation for other slots.

If model is content-based, correlations should be similar across slots.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
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


def analyze_correlation():
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

    # Collect paired (model_q, oracle_q) for each slot when legal
    # Action space is 7 (one per domino in hand), NOT 28
    slot_data = {i: {'model': [], 'oracle': []} for i in range(7)}

    n_samples = 0
    max_samples = 100000

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting Q-values"):
            tokens, masks, players, targets, legal, qvals, teams, values = batch

            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)

            model_q, _ = model(tokens, masks, players)

            # Move to CPU for analysis
            model_q = model_q.cpu().numpy()
            oracle_q = qvals.numpy()
            legal_np = legal.numpy()
            teams_np = teams.numpy()

            batch_size = len(model_q)
            for i in range(batch_size):
                # Team sign for oracle Q
                team_sign = 1.0 if teams_np[i] == 0 else -1.0

                for slot in range(7):  # 7 slots per hand
                    if legal_np[i, slot] > 0:
                        slot_data[slot]['model'].append(float(model_q[i, slot]))
                        slot_data[slot]['oracle'].append(float(oracle_q[i, slot]) * team_sign)

            n_samples += batch_size
            if n_samples >= max_samples:
                break

    print("\n" + "=" * 70)
    print("MODEL-ORACLE Q-VALUE CORRELATION BY SLOT")
    print("=" * 70)

    print("\nSlot   N_samples   r(model,oracle)   mean_model   mean_oracle   model_bias")
    print("-" * 80)

    correlations = []
    for slot in range(7):  # 7 hand slots
        model_vals = np.array(slot_data[slot]['model'])
        oracle_vals = np.array(slot_data[slot]['oracle'])

        if len(model_vals) >= 100:  # Need enough data
            r, p = pearsonr(model_vals, oracle_vals)
            mean_model = np.mean(model_vals)
            mean_oracle = np.mean(oracle_vals)
            bias = mean_model - mean_oracle

            correlations.append((slot, r, len(model_vals)))

            print(f"{slot:4d}   {len(model_vals):8d}     r={r:.4f}       "
                  f"{mean_model:7.2f}      {mean_oracle:7.2f}      {bias:+.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if correlations:
        slot0_corr = next((c[1] for c in correlations if c[0] == 0), None)
        other_corrs = [c[1] for c in correlations if c[0] > 0]

        print(f"\nSlot 0 correlation:     r = {slot0_corr:.4f}")
        print(f"Slots 1-6 mean corr:    r = {np.mean(other_corrs):.4f}")
        print(f"Slots 1-6 min corr:     r = {np.min(other_corrs):.4f}")
        print(f"Slots 1-6 max corr:     r = {np.max(other_corrs):.4f}")

        print("\n### Interpretation ###")
        if slot0_corr < np.mean(other_corrs) - 0.05:
            print("⚠️  Slot 0 correlation is LOWER than others -> suggests POSITIONAL BIAS")
        elif slot0_corr > np.mean(other_corrs) + 0.05:
            print("✓  Slot 0 correlation is HIGHER than others -> no bias detected")
        else:
            print("✓  Slot 0 correlation is SIMILAR to others -> CONTENT-BASED (no positional bias)")


if __name__ == "__main__":
    analyze_correlation()
