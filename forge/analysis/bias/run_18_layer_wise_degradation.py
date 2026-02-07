"""
Investigation 18: Layer-wise degradation analysis for slot 0 bias.

Hypothesis: The slot 0 bias accumulates across transformer layers.
The model shows slot 0 has r=0.81 correlation with oracle while slots 1-6 have r=0.99+.

Approach:
1. Hook the model to extract embeddings AFTER EACH transformer layer
2. For each layer, project the hand embeddings through the output head to get Q-values
3. Correlate position 1 (slot 0) embedding's predicted Q with final Q[0] oracle value
4. Correlate positions 2-7 (slots 1-6) embeddings' predicted Qs with final Q[1-6] oracle values
5. Plot the degradation curve: Does position 1's predictive power decrease layer-by-layer?

Key questions:
- Is the degradation gradual or does it happen at a specific layer?
- Does it correlate with any specific attention patterns?
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

# Suppress nested tensor warning
warnings.filterwarnings("ignore", message=".*nested tensors.*prototype.*")

PROJECT_ROOT = Path("/home/jason/v2/mk5-tailwind")
sys.path.insert(0, str(PROJECT_ROOT))

from forge.ml.module import DominoLightningModule


def extract_layerwise_embeddings(
    model: DominoLightningModule,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    current_player: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Extract embeddings at each layer of the transformer.

    Returns:
        Dict mapping layer name to hand embeddings (batch, 7, embed_dim)
        Keys: 'input', 'layer_0', 'layer_1', ..., 'layer_N-1'
    """
    device = tokens.device

    # Build embeddings from token features
    embeds = [
        model.model.high_pip_embed(tokens[:, :, 0]),
        model.model.low_pip_embed(tokens[:, :, 1]),
        model.model.is_double_embed(tokens[:, :, 2]),
        model.model.count_value_embed(tokens[:, :, 3]),
        model.model.trump_rank_embed(tokens[:, :, 4]),
        model.model.player_id_embed(tokens[:, :, 5]),
        model.model.is_current_embed(tokens[:, :, 6]),
        model.model.is_partner_embed(tokens[:, :, 7]),
        model.model.is_remaining_embed(tokens[:, :, 8]),
        model.model.token_type_embed(tokens[:, :, 9]),
        model.model.decl_embed(tokens[:, :, 10]),
        model.model.leader_embed(tokens[:, :, 11]),
    ]

    x = torch.cat(embeds, dim=-1)
    x = model.model.input_proj(x)

    # Prepare attention mask
    attn_mask = (mask == 0)

    # Helper to extract hand embeddings
    def extract_hand(tensor: torch.Tensor) -> torch.Tensor:
        """Extract hand representations for current player."""
        start_indices = 1 + current_player * 7
        offsets = torch.arange(7, device=device)
        gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, model.model.embed_dim)
        return torch.gather(tensor, dim=1, index=gather_indices)

    layerwise_embeddings = {}

    # Input embeddings (after projection, before transformer)
    layerwise_embeddings['input'] = extract_hand(x)

    # Process through each transformer layer
    for layer_idx, layer in enumerate(model.model.transformer.layers):
        # Self-attention
        attn_output, _ = layer.self_attn(
            x, x, x,
            key_padding_mask=attn_mask,
            need_weights=False,
        )
        x = layer.norm1(x + layer.dropout1(attn_output))

        # Feedforward
        ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(ff_output))

        # Store hand embeddings after this layer
        layerwise_embeddings[f'layer_{layer_idx}'] = extract_hand(x)

    return layerwise_embeddings


def compute_predicted_q(
    model: DominoLightningModule,
    hand_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Project hand embeddings through the output head to get predicted Q-values.

    Args:
        hand_embeddings: (batch, 7, embed_dim)

    Returns:
        predicted_q: (batch, 7) - Q-value predictions
    """
    return model.model.output_proj(hand_embeddings).squeeze(-1)


def compute_slot_correlations(
    predicted_q: np.ndarray,
    oracle_q: np.ndarray,
    legal_mask: np.ndarray
) -> Dict[int, float]:
    """
    Compute correlation between predicted and oracle Q-values for each slot.

    Args:
        predicted_q: (n_samples, 7)
        oracle_q: (n_samples, 7)
        legal_mask: (n_samples, 7)

    Returns:
        Dict mapping slot index to correlation coefficient
    """
    correlations = {}

    for slot in range(7):
        # Only include samples where this slot is legal
        mask = legal_mask[:, slot] > 0
        if mask.sum() < 100:
            correlations[slot] = np.nan
            continue

        pred = predicted_q[mask, slot]
        oracle = oracle_q[mask, slot]

        r, _ = stats.pearsonr(pred, oracle)
        correlations[slot] = r

    return correlations


def main():
    print("=" * 70)
    print("Investigation 18: Layer-wise Degradation Analysis")
    print("=" * 70)
    print()

    # Load model
    checkpoint_path = PROJECT_ROOT / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
    print(f"Loading model from: {checkpoint_path}")

    model = DominoLightningModule.load_from_checkpoint(str(checkpoint_path), weights_only=False)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    n_layers = len(model.model.transformer.layers)
    embed_dim = model.model.embed_dim
    print(f"Model: {n_layers} layers, {embed_dim} embed_dim")

    # Load validation data
    data_path = PROJECT_ROOT / "data/tokenized-full/val"
    print(f"\nLoading validation data from: {data_path}")

    tokens = np.load(data_path / "tokens.npy")
    masks = np.load(data_path / "masks.npy")
    players = np.load(data_path / "players.npy")
    qvals = np.load(data_path / "qvals.npy")
    legal = np.load(data_path / "legal.npy")
    teams = np.load(data_path / "teams.npy")

    n_samples = len(tokens)
    print(f"Total validation samples: {n_samples:,}")

    # Process in batches
    batch_size = 1024

    # Storage for all layer results
    all_predicted_q = {f'layer_{i}': [] for i in range(n_layers)}
    all_predicted_q['input'] = []

    print(f"\nProcessing {n_samples:,} samples in batches of {batch_size}...")

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            batch_tokens = torch.from_numpy(tokens[start_idx:end_idx].astype(np.int64)).to(device)
            batch_masks = torch.from_numpy(masks[start_idx:end_idx].astype(np.float32)).to(device)
            batch_players = torch.from_numpy(players[start_idx:end_idx].astype(np.int64)).to(device)
            batch_teams = teams[start_idx:end_idx]

            # Extract layer-wise embeddings
            layerwise_embeds = extract_layerwise_embeddings(model, batch_tokens, batch_masks, batch_players)

            # For each layer, project to Q-values
            for layer_name, embeds in layerwise_embeds.items():
                pred_q = compute_predicted_q(model, embeds)

                # Adjust for team perspective (model predicts from current player's perspective)
                team_sign = np.where(batch_teams == 0, 1.0, -1.0)
                pred_q_np = pred_q.cpu().numpy() * team_sign[:, np.newaxis]

                all_predicted_q[layer_name].append(pred_q_np)

            if (start_idx // batch_size) % 50 == 0:
                print(f"  Processed {end_idx:,} / {n_samples:,} samples...")

    # Concatenate all batches
    print("\nConcatenating results...")
    for layer_name in all_predicted_q:
        all_predicted_q[layer_name] = np.concatenate(all_predicted_q[layer_name], axis=0)

    # Oracle Q-values (already in team 0 perspective)
    oracle_q = qvals.astype(np.float32)

    # Compute correlations for each layer and each slot
    print("\n" + "=" * 70)
    print("LAYER-WISE CORRELATION ANALYSIS")
    print("=" * 70)

    layer_names = ['input'] + [f'layer_{i}' for i in range(n_layers)]
    results = {}

    print("\nCorrelation (r) between predicted Q and oracle Q by layer and slot:")
    print("-" * 70)
    header = "Layer".ljust(12) + " ".join([f"Slot {i}".center(8) for i in range(7)]) + " | Slot0-Mean"
    print(header)
    print("-" * 70)

    for layer_name in layer_names:
        pred_q = all_predicted_q[layer_name]
        correlations = compute_slot_correlations(pred_q, oracle_q, legal)
        results[layer_name] = correlations

        # Format output
        corr_str = " ".join([f"{correlations[i]:.4f}" if not np.isnan(correlations[i]) else "   N/A " for i in range(7)])

        # Compute slot 0 vs mean of slots 1-6
        slot0_r = correlations[0]
        slots16_mean = np.nanmean([correlations[i] for i in range(1, 7)])
        diff = slot0_r - slots16_mean if not np.isnan(slot0_r) else np.nan

        print(f"{layer_name.ljust(12)} {corr_str} | {diff:+.4f}" if not np.isnan(diff) else f"{layer_name.ljust(12)} {corr_str} |   N/A")

    # Compute degradation metrics
    print("\n" + "=" * 70)
    print("DEGRADATION ANALYSIS")
    print("=" * 70)

    # Track slot 0 and slots 1-6 mean correlation across layers
    slot0_corrs = [results[layer][0] for layer in layer_names]
    slots16_means = [np.nanmean([results[layer][i] for i in range(1, 7)]) for layer in layer_names]

    print("\nSlot 0 correlation trajectory:")
    for i, layer in enumerate(layer_names):
        print(f"  {layer}: r = {slot0_corrs[i]:.4f}")

    print("\nSlots 1-6 mean correlation trajectory:")
    for i, layer in enumerate(layer_names):
        print(f"  {layer}: r = {slots16_means[i]:.4f}")

    # Identify where degradation happens
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Check if degradation is gradual or sudden
    slot0_deltas = np.diff(slot0_corrs)
    slots16_deltas = np.diff(slots16_means)

    print("\nChange in correlation per layer (delta r):")
    print("-" * 50)
    print("Transition".ljust(20) + "Slot 0".center(12) + "Slots 1-6".center(12) + "Gap Change")
    print("-" * 50)

    for i, (layer_a, layer_b) in enumerate(zip(layer_names[:-1], layer_names[1:])):
        delta0 = slot0_deltas[i]
        delta16 = slots16_deltas[i]
        gap_change = delta0 - delta16
        print(f"{layer_a} -> {layer_b}".ljust(20) + f"{delta0:+.4f}".center(12) + f"{delta16:+.4f}".center(12) + f"{gap_change:+.4f}")

    # Find layer with largest gap increase
    gap_changes = slot0_deltas - slots16_deltas
    worst_transition_idx = np.argmin(gap_changes)

    print(f"\nLargest degradation: {layer_names[worst_transition_idx]} -> {layer_names[worst_transition_idx + 1]}")
    print(f"  Gap change: {gap_changes[worst_transition_idx]:+.4f}")

    # Final slot 0 vs slots 1-6 comparison
    final_slot0 = slot0_corrs[-1]
    final_slots16 = slots16_means[-1]
    initial_slot0 = slot0_corrs[0]
    initial_slots16 = slots16_means[0]

    print(f"\nInitial (input layer):")
    print(f"  Slot 0: r = {initial_slot0:.4f}")
    print(f"  Slots 1-6: r = {initial_slots16:.4f}")
    print(f"  Gap: {initial_slot0 - initial_slots16:+.4f}")

    print(f"\nFinal (after all layers):")
    print(f"  Slot 0: r = {final_slot0:.4f}")
    print(f"  Slots 1-6: r = {final_slots16:.4f}")
    print(f"  Gap: {final_slot0 - final_slots16:+.4f}")

    total_slot0_degradation = final_slot0 - initial_slot0
    total_slots16_degradation = final_slots16 - initial_slots16

    print(f"\nTotal degradation:")
    print(f"  Slot 0: {total_slot0_degradation:+.4f}")
    print(f"  Slots 1-6: {total_slots16_degradation:+.4f}")

    # Save results
    output_dir = PROJECT_ROOT / "forge/analysis/bias"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot degradation curves
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Correlation by layer
    ax = axes[0]
    x = np.arange(len(layer_names))
    ax.plot(x, slot0_corrs, 'ro-', label='Slot 0', linewidth=2, markersize=8)
    ax.plot(x, slots16_means, 'bs-', label='Slots 1-6 (mean)', linewidth=2, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Input'] + [f'L{i}' for i in range(n_layers)], rotation=45)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation with Oracle (r)')
    ax.set_title('Correlation Trajectory by Layer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.0)

    # Plot 2: Gap between slot 0 and slots 1-6
    ax = axes[1]
    gaps = np.array(slot0_corrs) - np.array(slots16_means)
    ax.bar(x, gaps, color=['red' if g < 0 else 'green' for g in gaps], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Input'] + [f'L{i}' for i in range(n_layers)], rotation=45)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Gap (Slot 0 - Slots 1-6)')
    ax.set_title('Correlation Gap by Layer')
    ax.grid(True, alpha=0.3)

    # Plot 3: Per-slot correlations at each layer
    ax = axes[2]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    for slot in range(7):
        slot_corrs = [results[layer][slot] for layer in layer_names]
        label = f'Slot {slot}' + (' (biased)' if slot == 0 else '')
        ax.plot(x, slot_corrs, marker='o', label=label, color=colors[slot],
                linewidth=2 if slot == 0 else 1,
                alpha=1.0 if slot == 0 else 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Input'] + [f'L{i}' for i in range(n_layers)], rotation=45)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation with Oracle (r)')
    ax.set_title('Per-Slot Correlation Trajectory')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.0)

    plt.tight_layout()
    fig_path = figures_dir / '18_layer_wise_degradation.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to: {fig_path}")

    # Save numerical results
    results_dict = {
        'n_samples': n_samples,
        'n_layers': n_layers,
        'layer_names': layer_names,
        'slot0_correlations': slot0_corrs,
        'slots16_mean_correlations': slots16_means,
        'per_slot_correlations': {layer: {int(k): float(v) for k, v in results[layer].items()} for layer in layer_names},
        'slot0_deltas': slot0_deltas.tolist(),
        'slots16_deltas': slots16_deltas.tolist(),
        'gap_changes': gap_changes.tolist(),
        'worst_transition': f"{layer_names[worst_transition_idx]} -> {layer_names[worst_transition_idx + 1]}",
    }

    json_path = output_dir / "18_layer_wise_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved results to: {json_path}")

    # Generate markdown report
    report_path = output_dir / "18-layer-wise-degradation.md"

    with open(report_path, 'w') as f:
        f.write("# Investigation 18: Layer-wise Degradation Analysis\n\n")

        f.write("## Question\n")
        f.write("Does the slot 0 bias accumulate across transformer layers, or is it introduced at a specific layer?\n\n")

        f.write("## Context\n")
        f.write("The model shows slot 0 has r=0.81 correlation with oracle while slots 1-6 have r=0.99+.\n")
        f.write("Previous investigations (01-14) ruled out attention masking, tokenization, positional encoding, ")
        f.write("and output head bias. The remaining hypothesis is that the degradation occurs within the ")
        f.write("transformer encoder layers themselves.\n\n")

        f.write("## Method\n")
        f.write("1. Extract embeddings after each transformer layer\n")
        f.write("2. Project each layer's hand embeddings through the output head to get Q-values\n")
        f.write("3. Correlate predicted Q-values with oracle Q-values for each slot\n")
        f.write("4. Track how the correlation gap (slot 0 vs slots 1-6) changes layer by layer\n\n")

        f.write(f"**Model**: {n_layers} layers, {embed_dim} embed_dim\n")
        f.write(f"**Validation samples**: {n_samples:,}\n\n")

        f.write("## Results\n\n")

        f.write("### Correlation by Layer\n\n")
        f.write("| Layer | Slot 0 (r) | Slots 1-6 mean (r) | Gap |\n")
        f.write("|-------|------------|-------------------|-----|\n")
        for i, layer in enumerate(layer_names):
            gap = slot0_corrs[i] - slots16_means[i]
            f.write(f"| {layer} | {slot0_corrs[i]:.4f} | {slots16_means[i]:.4f} | {gap:+.4f} |\n")

        f.write("\n### Degradation per Layer Transition\n\n")
        f.write("| Transition | Slot 0 delta | Slots 1-6 delta | Gap change |\n")
        f.write("|------------|--------------|-----------------|------------|\n")
        for i, (layer_a, layer_b) in enumerate(zip(layer_names[:-1], layer_names[1:])):
            f.write(f"| {layer_a} -> {layer_b} | {slot0_deltas[i]:+.4f} | {slots16_deltas[i]:+.4f} | {gap_changes[i]:+.4f} |\n")

        f.write("\n### Per-Slot Correlations\n\n")
        f.write("| Layer | " + " | ".join([f"Slot {i}" for i in range(7)]) + " |\n")
        f.write("|-------|" + "|".join(["-----" for _ in range(7)]) + "|\n")
        for layer in layer_names:
            row = f"| {layer} |"
            for slot in range(7):
                row += f" {results[layer][slot]:.4f} |"
            f.write(row + "\n")

        f.write("\n## Key Findings\n\n")

        # Determine the pattern
        if initial_slot0 - initial_slots16 < -0.05:
            f.write("**The gap is already present at the input layer** (before any transformer processing).\n\n")
            f.write(f"- Input layer gap: {initial_slot0 - initial_slots16:+.4f}\n")
            f.write(f"- Final layer gap: {final_slot0 - final_slots16:+.4f}\n\n")

            if abs(gap_changes).max() < 0.01:
                f.write("The transformer layers **do not significantly change** the gap.\n")
                f.write("The bias is **introduced before the transformer**, likely in:\n")
                f.write("- The input embedding layer\n")
                f.write("- The input projection layer\n")
                f.write("- Training data distribution\n")
            else:
                f.write(f"The largest gap change occurs at: {layer_names[worst_transition_idx]} -> {layer_names[worst_transition_idx + 1]} ({gap_changes[worst_transition_idx]:+.4f})\n")
        else:
            if np.argmin(gap_changes) == 0:
                f.write("**The degradation happens early** - primarily in the first transformer layer.\n\n")
            else:
                f.write(f"**The degradation is gradual** across layers, with the largest drop at layer {worst_transition_idx + 1}.\n\n")

        # Variance explained
        f.write("\n### Variance Explained\n\n")
        f.write("| Slot | Final r | r^2 (variance explained) |\n")
        f.write("|------|---------|-------------------------|\n")
        final_layer = layer_names[-1]
        for slot in range(7):
            r = results[final_layer][slot]
            r2 = r**2 * 100
            marker = " <-- biased" if slot == 0 else ""
            f.write(f"| Slot {slot} | {r:.4f} | {r2:.1f}%{marker} |\n")

        f.write("\n## Conclusion\n\n")

        # Summarize
        if initial_slot0 - initial_slots16 < -0.05:
            f.write("The slot 0 bias is **present from the input layer**, indicating the root cause is ")
            f.write("**upstream of the transformer** - either in how position 1 (slot 0) is embedded differently ")
            f.write("from positions 2-7 (slots 1-6), or in training data distribution bias.\n\n")
            f.write("The transformer layers do not significantly amplify or reduce this bias.\n")
        else:
            f.write("The slot 0 bias **accumulates across transformer layers**, suggesting the architecture ")
            f.write("or attention patterns systematically degrade information for position 1 relative to ")
            f.write("positions 2-7.\n")

        f.write(f"\n## Visualizations\n\n")
        f.write(f"- `figures/18_layer_wise_degradation.png` - Correlation trajectory and gap analysis\n")

    print(f"Saved report to: {report_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results_dict


if __name__ == "__main__":
    main()
