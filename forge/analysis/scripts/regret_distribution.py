#!/usr/bin/env python3
"""Compute regret distribution across the full validation set.

Regret = oracle_best_q - oracle_q[model_pick]

Where model_pick is model_q.argmax() and oracle_q is ground truth Q-values.

Outputs:
- Summary statistics (mean, median, percentiles, max, zero-regret rate)
- High-regret states (regret > 5) saved to parquet for inspection
- Histogram of regret distribution
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

from forge.ml.data import DominoDataset
from forge.ml.metrics import compute_qgaps_per_sample
from forge.ml.module import DominoLightningModule

# Configuration
MODEL_PATH = Path(PROJECT_ROOT) / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
DATA_PATH = Path(PROJECT_ROOT) / "data/tokenized-full"
OUTPUT_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"
BATCH_SIZE = 512
NUM_WORKERS = 8
HIGH_REGRET_THRESHOLD = 5.0


def load_model(device: str = "cuda") -> DominoLightningModule:
    """Load the trained Q-value model."""
    print(f"Loading model from {MODEL_PATH}")

    # Load checkpoint - we need to handle RNG state separately for inference
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)

    # Remove RNG state to avoid loading issues during inference
    if 'rng_state' in checkpoint:
        del checkpoint['rng_state']

    # Get hyperparameters and create model
    hparams = checkpoint.get('hyper_parameters', {})
    model = DominoLightningModule(**hparams)

    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    return model


def compute_regret_distribution(
    model: DominoLightningModule,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, list[dict]]:
    """
    Compute regret for all validation samples.

    Returns:
        regrets: Array of regret values for each sample
        high_regret_samples: List of dicts with high-regret sample info
    """
    all_regrets = []
    high_regret_samples = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing regret"):
            tokens, masks, players, targets, legal, qvals, teams, values = batch

            # Move to device
            tokens = tokens.to(device)
            masks = masks.to(device)
            players = players.to(device)
            legal = legal.to(device)
            qvals = qvals.to(device)
            teams = teams.to(device)

            # Get model predictions
            model_q, model_v = model(tokens, masks, players)

            # Compute per-sample regret using existing metric
            regrets = compute_qgaps_per_sample(model_q, qvals, legal, teams)
            regrets_np = regrets.cpu().numpy()
            all_regrets.append(regrets_np)

            # Collect high-regret samples for inspection
            batch_size = len(regrets_np)
            for i in range(batch_size):
                if regrets_np[i] > HIGH_REGRET_THRESHOLD:
                    # Get model's prediction and oracle best
                    model_q_masked = model_q[i].clone()
                    model_q_masked[legal[i] == 0] = float('-inf')
                    model_pick = model_q_masked.argmax().item()

                    # Team-signed qvals for finding oracle best
                    team_sign = 1.0 if teams[i].item() == 0 else -1.0
                    q_signed = qvals[i].cpu().numpy() * team_sign
                    q_legal = np.where(legal[i].cpu().numpy() > 0, q_signed, -np.inf)
                    oracle_best = int(np.argmax(q_legal))

                    high_regret_samples.append({
                        "sample_idx": sample_idx + i,
                        "regret": float(regrets_np[i]),
                        "model_pick": model_pick,
                        "oracle_best": oracle_best,
                        "model_q_picked": float(model_q[i, model_pick].cpu()),
                        "oracle_q_picked": float(qvals[i, model_pick].cpu()) * team_sign,
                        "oracle_q_best": float(qvals[i, oracle_best].cpu()) * team_sign,
                        "team": int(teams[i].item()),
                        "player": int(players[i].item()),
                        "n_legal": int(legal[i].sum().item()),
                        "oracle_value": float(values[i].item()),
                        # Store full Q-value arrays for debugging
                        "model_q": [float(x) for x in model_q[i].cpu().numpy()],
                        "oracle_q": [float(x) for x in qvals[i].cpu().numpy()],
                        "legal_mask": [int(x) for x in legal[i].cpu().numpy()],
                    })

            sample_idx += batch_size

    return np.concatenate(all_regrets), high_regret_samples


def print_summary_stats(regrets: np.ndarray) -> dict:
    """Print and return summary statistics."""
    stats = {
        "n_samples": len(regrets),
        "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)),
        "p95_regret": float(np.percentile(regrets, 95)),
        "p99_regret": float(np.percentile(regrets, 99)),
        "max_regret": float(np.max(regrets)),
        "zero_regret_pct": float(100 * np.mean(regrets == 0)),
        "near_zero_regret_pct": float(100 * np.mean(regrets < 0.01)),
    }

    print("\n" + "=" * 60)
    print("REGRET DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"Total samples:          {stats['n_samples']:,}")
    print(f"Mean regret:            {stats['mean_regret']:.4f} points")
    print(f"Median regret:          {stats['median_regret']:.4f} points")
    print(f"95th percentile:        {stats['p95_regret']:.4f} points")
    print(f"99th percentile:        {stats['p99_regret']:.4f} points")
    print(f"Max regret:             {stats['max_regret']:.4f} points")
    print(f"Zero regret:            {stats['zero_regret_pct']:.2f}%")
    print(f"Near-zero (<0.01):      {stats['near_zero_regret_pct']:.2f}%")
    print("=" * 60)

    # Additional buckets
    buckets = [0, 1, 2, 5, 10, 20, 42]
    print("\nRegret distribution buckets:")
    for i in range(len(buckets) - 1):
        low, high = buckets[i], buckets[i + 1]
        pct = 100 * np.mean((regrets >= low) & (regrets < high))
        print(f"  [{low:2d}, {high:2d}): {pct:6.2f}%")
    pct_high = 100 * np.mean(regrets >= buckets[-1])
    print(f"  [{buckets[-1]:2d}, ∞):  {pct_high:6.2f}%")

    return stats


def plot_histogram(regrets: np.ndarray, output_path: Path) -> None:
    """Create histogram of regret distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Full distribution with log scale on y-axis
    ax1 = axes[0]
    bins = np.linspace(0, max(10, np.percentile(regrets, 99.9)), 50)
    ax1.hist(regrets, bins=bins, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Regret (points)', fontsize=12)
    ax1.set_ylabel('Count (log scale)', fontsize=12)
    ax1.set_title('Regret Distribution (Full)', fontsize=14)
    ax1.axvline(np.mean(regrets), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(regrets):.3f}')
    ax1.axvline(np.median(regrets), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(regrets):.3f}')
    ax1.legend()

    # Right: Zoomed in on low-regret region
    ax2 = axes[1]
    regrets_low = regrets[regrets < 2]
    bins_low = np.linspace(0, 2, 40)
    ax2.hist(regrets_low, bins=bins_low, color='forestgreen', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Regret (points)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Regret < 2 ({100*len(regrets_low)/len(regrets):.1f}% of samples)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogram saved to {output_path}")


def plot_cumulative(regrets: np.ndarray, output_path: Path) -> None:
    """Create cumulative distribution plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_regrets = np.sort(regrets)
    cumulative = np.arange(1, len(sorted_regrets) + 1) / len(sorted_regrets)

    ax.plot(sorted_regrets, cumulative, color='steelblue', linewidth=2)
    ax.set_xlabel('Regret (points)', fontsize=12)
    ax.set_ylabel('Cumulative Fraction', fontsize=12)
    ax.set_title('Cumulative Distribution of Regret', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark key percentiles
    for pct in [50, 90, 95, 99]:
        val = np.percentile(regrets, pct)
        ax.axhline(pct/100, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(val, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(f'p{pct}: {val:.2f}', xy=(val, pct/100),
                    xytext=(val + 0.5, pct/100 - 0.05), fontsize=10)

    ax.set_xlim(0, min(15, np.percentile(regrets, 99.9)))
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Cumulative plot saved to {output_path}")


def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(exist_ok=True)

    # Load model
    model = load_model(device)

    # Load validation dataset
    print(f"Loading validation data from {DATA_PATH}")
    val_dataset = DominoDataset(str(DATA_PATH), split='val', mmap=False)
    print(f"Validation samples: {len(val_dataset):,}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Compute regret distribution
    regrets, high_regret_samples = compute_regret_distribution(model, val_loader, device)

    # Summary statistics
    stats = print_summary_stats(regrets)

    # Save stats to JSON
    import json
    stats_path = OUTPUT_DIR / "tables/regret_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    # Save high-regret samples to parquet
    if high_regret_samples:
        high_regret_df = pd.DataFrame(high_regret_samples)
        parquet_path = OUTPUT_DIR / "tables/high_regret_samples.parquet"
        high_regret_df.to_parquet(parquet_path, index=False)
        print(f"High-regret samples ({len(high_regret_df):,}) saved to {parquet_path}")

        # Also show top 10
        print("\nTop 10 highest regret samples:")
        top10 = high_regret_df.nlargest(10, 'regret')
        for _, row in top10.iterrows():
            print(f"  idx={row['sample_idx']:6d}  regret={row['regret']:.2f}  "
                  f"pick={row['model_pick']} vs best={row['oracle_best']}  "
                  f"q_picked={row['oracle_q_picked']:.1f} vs q_best={row['oracle_q_best']:.1f}")
    else:
        print(f"\nNo samples with regret > {HIGH_REGRET_THRESHOLD}")

    # Plot histogram
    plot_histogram(regrets, OUTPUT_DIR / "figures/regret_histogram.png")

    # Plot cumulative distribution
    plot_cumulative(regrets, OUTPUT_DIR / "figures/regret_cumulative.png")

    # Save full regret array for future analysis
    np.save(OUTPUT_DIR / "tables/regret_values.npy", regrets)
    print(f"\nFull regret array saved to {OUTPUT_DIR}/tables/regret_values.npy")


if __name__ == "__main__":
    main()
