#!/usr/bin/env python3
"""
Investigation 19: LayerNorm Statistics by Position

Hypothesis: LayerNorm statistics differ systematically for position 1 (slot 0)
vs other positions, potentially causing the observed bias.

Method:
1. Load the Q-value model
2. Manually forward through layers to capture pre-LayerNorm activations
3. Run inference on validation samples
4. Compute per-position statistics incrementally (mean, variance, effective scale)
"""

import sys
from pathlib import Path

PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import warnings
warnings.filterwarnings("ignore", message=".*nested tensors.*prototype.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from forge.ml.module import DominoLightningModule
from forge.ml.data import DominoDataset


class RunningStats:
    """Welford's online algorithm for computing running mean and variance."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)

    def update(self, batch):
        """Update with a batch of samples."""
        for x in batch:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_variance(self):
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.M2 / self.n

    def get_std(self):
        return np.sqrt(self.get_variance())


def compute_prenorm_stats(model, tokens, mask, current_player):
    """
    Forward pass through DominoTransformer that computes pre-LayerNorm statistics.

    Returns:
        dict of layer_name -> dict with:
            - 'pos_stds': (seq_len,) array of per-position stds (averaged per-sample)
            - 'pos_means': (seq_len,) array of per-position means
    """
    device = tokens.device
    dom_model = model.model  # The DominoTransformer
    batch_size = tokens.shape[0]

    # Build embeddings from token features
    embeds = [
        dom_model.high_pip_embed(tokens[:, :, 0]),
        dom_model.low_pip_embed(tokens[:, :, 1]),
        dom_model.is_double_embed(tokens[:, :, 2]),
        dom_model.count_value_embed(tokens[:, :, 3]),
        dom_model.trump_rank_embed(tokens[:, :, 4]),
        dom_model.player_id_embed(tokens[:, :, 5]),
        dom_model.is_current_embed(tokens[:, :, 6]),
        dom_model.is_partner_embed(tokens[:, :, 7]),
        dom_model.is_remaining_embed(tokens[:, :, 8]),
        dom_model.token_type_embed(tokens[:, :, 9]),
        dom_model.decl_embed(tokens[:, :, 10]),
        dom_model.leader_embed(tokens[:, :, 11]),
    ]

    x = torch.cat(embeds, dim=-1)
    x = dom_model.input_proj(x)

    # Attention mask
    attn_mask = (mask == 0)

    # Compute stats at each layer
    layer_stats = {}
    seq_len = x.shape[1]

    # Manually iterate through transformer encoder layers
    for i, layer in enumerate(dom_model.transformer.layers):
        # Capture pre-norm1 stats
        # Per-sample, per-position std across embedding dim
        pre_norm1 = x.detach()
        pos_stds_norm1 = pre_norm1.std(dim=2).cpu().numpy()  # (batch, seq_len)
        pos_means_norm1 = pre_norm1.mean(dim=2).cpu().numpy()  # (batch, seq_len)
        layer_stats[f"layer_{i}_norm1"] = {
            'pos_stds': pos_stds_norm1,
            'pos_means': pos_means_norm1,
        }

        # Self-attention
        attn_out = layer.self_attn(
            x, x, x,
            key_padding_mask=attn_mask,
            need_weights=False
        )[0]
        x_after_attn = x + layer.dropout1(attn_out)
        x_normed1 = layer.norm1(x_after_attn)

        # Capture pre-norm2 stats
        pre_norm2 = x_after_attn.detach()
        pos_stds_norm2 = pre_norm2.std(dim=2).cpu().numpy()
        pos_means_norm2 = pre_norm2.mean(dim=2).cpu().numpy()
        layer_stats[f"layer_{i}_norm2"] = {
            'pos_stds': pos_stds_norm2,
            'pos_means': pos_means_norm2,
        }

        # FFN
        ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_normed1))))
        x_after_ff = x_normed1 + layer.dropout2(ff_out)
        x = layer.norm2(x_after_ff)

    return layer_stats


def main():
    # Configuration
    CHECKPOINT = Path(PROJECT_ROOT) / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
    DATA_PATH = Path(PROJECT_ROOT) / "data/tokenized-full"
    N_SAMPLES = 5000  # Number of samples to analyze
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {DEVICE}")
    print(f"Loading model from: {CHECKPOINT}")

    # Load checkpoint manually to avoid RNG state restore issues
    checkpoint = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=False)
    hparams = checkpoint['hyper_parameters']

    # Create model with same hyperparameters
    model = DominoLightningModule(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(DEVICE)

    print(f"Model architecture:")
    print(f"  embed_dim: {model.hparams.embed_dim}")
    print(f"  n_layers: {model.hparams.n_layers}")
    print(f"  n_heads: {model.hparams.n_heads}")

    # Load validation data
    print(f"\nLoading validation data from: {DATA_PATH}")
    val_dataset = DominoDataset(str(DATA_PATH), "val", mmap=True)
    print(f"Validation set size: {len(val_dataset)}")

    # Run inference
    n_samples = min(N_SAMPLES, len(val_dataset))
    print(f"\nRunning inference on {n_samples} samples...")

    batch_size = 64
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Initialize running statistics accumulators
    # We'll use simple averaging since we're collecting per-batch stats
    all_stds = defaultdict(list)  # layer_name -> list of (batch, seq_len) arrays
    all_means = defaultdict(list)

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            # Gather batch
            batch_tokens = []
            batch_masks = []
            batch_players = []

            for i in range(start_idx, end_idx):
                tokens, masks, players, targets, legal, qvals, teams, values = val_dataset[i]
                batch_tokens.append(tokens)
                batch_masks.append(masks)
                batch_players.append(players)

            tokens = torch.stack(batch_tokens).to(DEVICE)
            masks = torch.stack(batch_masks).to(DEVICE)
            players = torch.stack(batch_players).to(DEVICE)

            # Compute stats for this batch
            batch_stats = compute_prenorm_stats(model, tokens, masks, players)

            for layer_name, stats in batch_stats.items():
                all_stds[layer_name].append(stats['pos_stds'])
                all_means[layer_name].append(stats['pos_means'])

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {end_idx}/{n_samples} samples")

    print(f"\nAnalyzing LayerNorm statistics by position...")

    results = {}

    for layer_name in sorted(all_stds.keys()):
        # Concatenate all batches and compute final stats
        stds = np.vstack(all_stds[layer_name])  # (n_samples, seq_len)
        means = np.vstack(all_means[layer_name])

        # Average across samples
        pos_stds = stds.mean(axis=0)
        pos_means = means.mean(axis=0)

        results[layer_name] = {
            'stds': pos_stds,
            'means': pos_means,
        }

        print(f"\n{layer_name}:")
        print(f"  Shape: ({stds.shape[0]}, {stds.shape[1]})")

        # Focus on positions 1-7 (P0's hand slots)
        print(f"\n  Position statistics (focusing on P0's hand positions 1-7):")
        print(f"  {'Pos':>4} {'Mean':>10} {'Std':>10}")
        print(f"  {'-'*4} {'-'*10} {'-'*10}")

        seq_len = len(pos_stds)
        for pos in range(min(8, seq_len)):
            label = "ctx" if pos == 0 else f"s{pos-1}"
            print(f"  {label:>4} {pos_means[pos]:>10.4f} {pos_stds[pos]:>10.4f}")

        # Compare slot 0 (position 1) vs slots 1-6 (positions 2-7)
        slot0_mean = pos_means[1]
        slots_1_6_mean = pos_means[2:8].mean()
        slot0_std = pos_stds[1]
        slots_1_6_std = pos_stds[2:8].mean()

        print(f"\n  Slot 0 (pos 1) vs Slots 1-6 (pos 2-7):")
        print(f"    Mean: slot0={slot0_mean:.4f}, others={slots_1_6_mean:.4f}, ratio={slot0_mean/slots_1_6_mean:.3f}")
        print(f"    Std:  slot0={slot0_std:.4f}, others={slots_1_6_std:.4f}, ratio={slot0_std/slots_1_6_std:.3f}")

    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY: LayerNorm Statistics Comparison")
    print("="*80)

    print("\n{:<25} {:>12} {:>12} {:>10}".format("Layer", "Slot0 Std", "Slots1-6 Std", "Ratio"))
    print("-"*62)

    for layer_name in sorted(results.keys()):
        stats = results[layer_name]
        slot0_std = stats['stds'][1]
        slots_1_6_std = stats['stds'][2:8].mean()
        ratio = slot0_std / slots_1_6_std
        print(f"{layer_name:<25} {slot0_std:>12.4f} {slots_1_6_std:>12.4f} {ratio:>10.3f}")

    print("\n{:<25} {:>12} {:>12} {:>10}".format("Layer", "Slot0 Mean", "Slots1-6 Mean", "Ratio"))
    print("-"*62)

    for layer_name in sorted(results.keys()):
        stats = results[layer_name]
        slot0_mean = stats['means'][1]
        slots_1_6_mean = stats['means'][2:8].mean()
        ratio = slot0_mean / slots_1_6_mean if slots_1_6_mean != 0 else float('inf')
        print(f"{layer_name:<25} {slot0_mean:>12.4f} {slots_1_6_mean:>12.4f} {ratio:>10.3f}")

    # Write results to file
    output_path = Path(PROJECT_ROOT) / "forge/analysis/bias/19-layernorm-statistics.md"
    write_report(output_path, results, n_samples)
    print(f"\nReport written to: {output_path}")


def write_report(output_path: Path, results: dict, n_samples: int):
    """Write analysis report to markdown file."""

    lines = [
        "# 19: LayerNorm Statistics by Position",
        "",
        "## Question",
        "Do LayerNorm statistics differ systematically for position 1 (slot 0) vs other positions?",
        "",
        "## Method",
        "- Loaded Q-value model: `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`",
        f"- Ran inference on {n_samples} validation samples",
        "- Manually forwarded through each transformer layer to capture pre-normalization activations",
        "- Computed per-position statistics:",
        "  - Mean activation value (averaged across embedding dim, then samples)",
        "  - Standard deviation (across embedding dim per sample, then averaged - what LayerNorm divides by)",
        "",
        "## Position Layout",
        "```",
        "Position 0:     Context token",
        "Positions 1-7:  P0's hand (slot 0 at position 1)",
        "Positions 8-14: P1's hand",
        "Positions 15-21: P2's hand",
        "Positions 22-28: P3's hand",
        "```",
        "",
        "## Results",
        "",
        "### Standard Deviation by Position (LayerNorm Scaling Factor)",
        "",
        "Higher std means LayerNorm scales down the activation more aggressively.",
        "",
        "| Layer | Slot 0 Std | Slots 1-6 Std | Ratio |",
        "|-------|------------|---------------|-------|",
    ]

    # Std comparison table
    for layer_name in sorted(results.keys()):
        stats = results[layer_name]
        slot0_std = stats['stds'][1]
        slots_1_6_std = stats['stds'][2:8].mean()
        ratio = slot0_std / slots_1_6_std
        lines.append(f"| {layer_name} | {slot0_std:.4f} | {slots_1_6_std:.4f} | {ratio:.3f} |")

    lines.extend([
        "",
        "### Mean Activation by Position",
        "",
        "| Layer | Slot 0 Mean | Slots 1-6 Mean | Ratio |",
        "|-------|-------------|----------------|-------|",
    ])

    # Mean comparison table
    for layer_name in sorted(results.keys()):
        stats = results[layer_name]
        slot0_mean = stats['means'][1]
        slots_1_6_mean = stats['means'][2:8].mean()
        ratio = slot0_mean / slots_1_6_mean if abs(slots_1_6_mean) > 1e-6 else float('nan')
        lines.append(f"| {layer_name} | {slot0_mean:.4f} | {slots_1_6_mean:.4f} | {ratio:.3f} |")

    lines.extend([
        "",
        "### Per-Position Detail (Last Layer norm2)",
        "",
    ])

    last_norm2 = "layer_5_norm2"
    if last_norm2 in results:
        stats = results[last_norm2]
        lines.extend([
            "| Position | Role | Mean | Std |",
            "|----------|------|------|-----|",
        ])
        labels = ["context", "slot0", "slot1", "slot2", "slot3", "slot4", "slot5", "slot6"]
        for pos in range(min(8, len(stats['means']))):
            lines.append(f"| {pos} | {labels[pos]} | {stats['means'][pos]:.4f} | {stats['stds'][pos]:.4f} |")

    # Compute and add interpretation
    if last_norm2 in results:
        stats = results[last_norm2]
        slot0_std = stats['stds'][1]
        slots_1_6_std = stats['stds'][2:8].mean()
        ratio = slot0_std / slots_1_6_std

        # Determine if this could explain the bias
        if abs(ratio - 1.0) > 0.05:  # More than 5% difference
            interpretation = f"**YES**: Slot 0 has {abs(ratio-1)*100:.1f}% {'higher' if ratio > 1 else 'lower'} std than slots 1-6."
            if ratio > 1:
                explanation = "Higher std means LayerNorm applies stronger normalization, potentially compressing the signal."
            else:
                explanation = "Lower std means slot 0 activations are already more compressed before LayerNorm."
        else:
            interpretation = f"**NO**: Slot 0 std ratio ({ratio:.3f}) is within 5% of slots 1-6."
            explanation = "LayerNorm treats slot 0 similarly to other slots."

        lines.extend([
            "",
            "## Interpretation",
            "",
            f"**Do LayerNorm statistics differ for slot 0?** {interpretation}",
            "",
            explanation,
            "",
        ])

    # Add conclusion
    lines.extend([
        "## Conclusion",
        "",
    ])

    if last_norm2 in results:
        stats = results[last_norm2]
        slot0_std = stats['stds'][1]
        slots_1_6_std = stats['stds'][2:8].mean()
        ratio = slot0_std / slots_1_6_std

        if abs(ratio - 1.0) > 0.05:
            lines.append(f"LayerNorm statistics show a {abs(ratio-1)*100:.1f}% difference for slot 0 vs other slots.")
            lines.append("This could contribute to the observed positional bias, but the effect size")
            lines.append("should be compared against the 19x accuracy degradation (65% vs 98.7% variance explained).")
        else:
            lines.append("LayerNorm statistics are essentially identical for slot 0 vs other slots.")
            lines.append("LayerNorm normalization is NOT the cause of the positional bias.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
