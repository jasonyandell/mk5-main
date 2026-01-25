"""
17-attention-head-specialization.py - Per-head analysis for position 1 treatment

Investigates whether specific attention heads consistently treat position 1
(hand slot 0) differently, which could explain the slot 0 bias (r=0.81 vs r=0.99+).

Key questions:
1. Do specific heads act as "position 1 suppressors"?
2. Is there consistent position-specific behavior across samples?
3. Which heads show strongest differential treatment?

Output:
- forge/analysis/bias/figures/17_*.png (visualizations)
- forge/analysis/bias/17-attention-head-specialization.md (findings)
"""

import sys
from pathlib import Path

PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy import stats

from forge.ml.module import DominoLightningModule

# === Configuration ===
MODEL_PATH = Path(PROJECT_ROOT) / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
DATA_DIR = Path(PROJECT_ROOT) / "data/tokenized-full/val"
OUTPUT_DIR = Path(PROJECT_ROOT) / "forge/analysis/bias/figures"
N_SAMPLES = 2000  # More samples for statistical power

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """Load model for inference."""
    print(f"Loading model from {MODEL_PATH}")
    model = DominoLightningModule.load_from_checkpoint(
        str(MODEL_PATH),
        map_location='cpu',
        weights_only=False,
    )
    model.eval()
    return model


def load_validation_data(n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load validation data."""
    print(f"Loading validation data from {DATA_DIR}")

    tokens = torch.from_numpy(np.load(DATA_DIR / "tokens.npy")[:n_samples])
    masks = torch.from_numpy(np.load(DATA_DIR / "masks.npy")[:n_samples])
    players = torch.from_numpy(np.load(DATA_DIR / "players.npy")[:n_samples])

    print(f"Loaded {len(tokens)} samples")
    return tokens, masks, players


def extract_per_head_attention(
    model: DominoLightningModule,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    players: torch.Tensor,
    batch_size: int = 64
) -> Dict[str, torch.Tensor]:
    """Extract attention weights per-head for all samples.

    Returns dict mapping layer names to tensors of shape (n_samples, n_heads, seq_len, seq_len).
    """
    device = next(model.parameters()).device
    all_attention_weights = defaultdict(list)

    n_samples = len(tokens)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"Running inference on {n_samples} samples...")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)

            batch_tokens = tokens[start:end].to(device).long()
            batch_masks = masks[start:end].to(device)

            # Get embeddings
            domino_model = model.model

            embeds = [
                domino_model.high_pip_embed(batch_tokens[:, :, 0]),
                domino_model.low_pip_embed(batch_tokens[:, :, 1]),
                domino_model.is_double_embed(batch_tokens[:, :, 2]),
                domino_model.count_value_embed(batch_tokens[:, :, 3]),
                domino_model.trump_rank_embed(batch_tokens[:, :, 4]),
                domino_model.player_id_embed(batch_tokens[:, :, 5]),
                domino_model.is_current_embed(batch_tokens[:, :, 6]),
                domino_model.is_partner_embed(batch_tokens[:, :, 7]),
                domino_model.is_remaining_embed(batch_tokens[:, :, 8]),
                domino_model.token_type_embed(batch_tokens[:, :, 9]),
                domino_model.decl_embed(batch_tokens[:, :, 10]),
                domino_model.leader_embed(batch_tokens[:, :, 11]),
            ]

            x = torch.cat(embeds, dim=-1)
            x = domino_model.input_proj(x)

            attn_mask = (batch_masks == 0)

            for layer_idx, layer in enumerate(domino_model.transformer.layers):
                attn_output, attn_weights = layer.self_attn(
                    x, x, x,
                    key_padding_mask=attn_mask,
                    need_weights=True,
                    average_attn_weights=False  # Get per-head weights
                )

                # attn_weights shape: (batch, n_heads, seq_len, seq_len)
                all_attention_weights[f"layer_{layer_idx}"].append(attn_weights.cpu())

                # Continue through layer
                x = layer.norm1(x + layer.dropout1(attn_output))
                ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(ff_output))

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1}/{n_batches} batches")

    # Concatenate all batches
    result = {}
    for layer_name, weight_list in all_attention_weights.items():
        result[layer_name] = torch.cat(weight_list, dim=0)

    return result


def analyze_per_head_position_treatment(
    attention_weights: Dict[str, torch.Tensor],
    players: torch.Tensor
) -> Dict:
    """Analyze how each head treats position 1 vs positions 2-7.

    For each head, compute:
    - Average attention TO position 1 (slot 0) vs positions 2-7 (slots 1-6)
    - Average attention FROM position 1 vs positions 2-7
    - Statistical significance of differences

    Note: The hand positions depend on current_player:
    - P0: positions 1-7
    - P1: positions 8-14
    - P2: positions 15-21
    - P3: positions 22-28
    """

    results = {}
    players_np = players.numpy()

    for layer_name, weights in attention_weights.items():
        n_samples, n_heads, seq_len, _ = weights.shape
        weights_np = weights.numpy()

        # Per-head statistics
        head_stats = []

        for head_idx in range(n_heads):
            # Collect attention patterns for this head, adjusting for player
            attn_to_slot0_list = []
            attn_to_slots16_list = []
            attn_from_slot0_list = []
            attn_from_slots16_list = []

            for sample_idx in range(n_samples):
                player = int(players_np[sample_idx])
                hand_start = 1 + player * 7  # Position of slot 0 for this player

                # Make sure hand positions are valid
                if hand_start + 7 > seq_len:
                    continue

                head_attn = weights_np[sample_idx, head_idx]  # (seq_len, seq_len)

                # Position 1 = hand_start (slot 0 for current player)
                pos1 = hand_start
                pos2_7 = list(range(hand_start + 1, hand_start + 7))  # slots 1-6

                # Attention TO position 1 from all other hand positions
                # Query from slots 1-6, key to slot 0
                attn_to_pos1 = head_attn[pos2_7, pos1].mean()  # How much slots 1-6 attend TO slot 0

                # Attention TO positions 2-7 from all other hand positions
                # This is trickier - we want attention between non-slot-0 positions
                attn_to_pos2_7 = head_attn[pos2_7, :][:, pos2_7].mean()  # Internal attention among slots 1-6

                # Attention FROM position 1 to all other hand positions
                attn_from_pos1 = head_attn[pos1, pos2_7].mean()  # How much slot 0 attends to slots 1-6

                # Attention FROM positions 2-7 to slot 0
                attn_from_pos2_7 = head_attn[pos2_7, pos1].mean()  # Already computed above

                attn_to_slot0_list.append(attn_to_pos1)
                attn_to_slots16_list.append(attn_to_pos2_7)
                attn_from_slot0_list.append(attn_from_pos1)
                attn_from_slots16_list.append(attn_from_pos2_7)

            attn_to_slot0 = np.array(attn_to_slot0_list)
            attn_to_slots16 = np.array(attn_to_slots16_list)
            attn_from_slot0 = np.array(attn_from_slot0_list)
            attn_from_slots16 = np.array(attn_from_slots16_list)

            # Statistics
            to_ratio = attn_to_slot0.mean() / (attn_to_slots16.mean() + 1e-8)
            from_ratio = attn_from_slot0.mean() / (attn_from_slots16.mean() + 1e-8)

            # T-test: is attention TO slot 0 different from TO slots 1-6?
            t_stat_to, p_value_to = stats.ttest_ind(attn_to_slot0, attn_to_slots16)
            t_stat_from, p_value_from = stats.ttest_ind(attn_from_slot0, attn_from_slots16)

            head_stats.append({
                'head_idx': head_idx,
                'attn_to_slot0_mean': attn_to_slot0.mean(),
                'attn_to_slot0_std': attn_to_slot0.std(),
                'attn_to_slots16_mean': attn_to_slots16.mean(),
                'attn_to_slots16_std': attn_to_slots16.std(),
                'attn_from_slot0_mean': attn_from_slot0.mean(),
                'attn_from_slot0_std': attn_from_slot0.std(),
                'attn_from_slots16_mean': attn_from_slots16.mean(),
                'attn_from_slots16_std': attn_from_slots16.std(),
                'to_ratio': to_ratio,
                'from_ratio': from_ratio,
                't_stat_to': t_stat_to,
                'p_value_to': p_value_to,
                't_stat_from': t_stat_from,
                'p_value_from': p_value_from,
                'is_suppressor': to_ratio < 0.8 and p_value_to < 0.01,  # Significantly lower attention to slot 0
            })

        results[layer_name] = {
            'n_heads': n_heads,
            'head_stats': head_stats,
        }

    return results


def identify_specialized_heads(results: Dict) -> List[Dict]:
    """Identify heads that consistently treat position 1 differently."""

    specialized_heads = []

    for layer_name, layer_data in results.items():
        for head_stat in layer_data['head_stats']:
            # Check for significant differential treatment
            if head_stat['p_value_to'] < 0.01:  # Significant difference
                effect_size = abs(head_stat['attn_to_slot0_mean'] - head_stat['attn_to_slots16_mean'])

                specialized_heads.append({
                    'layer': layer_name,
                    'head': head_stat['head_idx'],
                    'to_ratio': head_stat['to_ratio'],
                    'from_ratio': head_stat['from_ratio'],
                    'effect_size': effect_size,
                    'p_value_to': head_stat['p_value_to'],
                    'is_suppressor': head_stat['is_suppressor'],
                    'direction': 'suppressor' if head_stat['to_ratio'] < 1 else 'amplifier',
                })

    # Sort by effect size
    specialized_heads.sort(key=lambda x: x['effect_size'], reverse=True)

    return specialized_heads


def plot_per_head_ratios(results: Dict, output_dir: Path):
    """Visualize attention ratio (to slot 0 / to slots 1-6) for each head."""

    n_layers = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (layer_name, layer_data) in enumerate(results.items()):
        ax = axes[idx]

        head_indices = [s['head_idx'] for s in layer_data['head_stats']]
        to_ratios = [s['to_ratio'] for s in layer_data['head_stats']]
        from_ratios = [s['from_ratio'] for s in layer_data['head_stats']]

        x = np.arange(len(head_indices))
        width = 0.35

        bars1 = ax.bar(x - width/2, to_ratios, width, label='Attn TO ratio', alpha=0.7, color='blue')
        bars2 = ax.bar(x + width/2, from_ratios, width, label='Attn FROM ratio', alpha=0.7, color='orange')

        # Reference line at 1.0 (equal treatment)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equal treatment')
        ax.axhline(y=0.8, color='gray', linestyle=':', linewidth=1)
        ax.axhline(y=1.2, color='gray', linestyle=':', linewidth=1)

        ax.set_xlabel('Head Index')
        ax.set_ylabel('Ratio (Slot 0 / Slots 1-6)')
        ax.set_title(f'{layer_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(head_indices)

        if idx == 0:
            ax.legend(loc='upper right')

    # Hide unused axes
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Per-Head Attention Ratio: Slot 0 vs Slots 1-6\n(Ratio < 1 = suppressor, > 1 = amplifier)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / '17_per_head_ratios.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-head ratios to {output_dir / '17_per_head_ratios.png'}")


def plot_suppressor_heatmap(results: Dict, output_dir: Path):
    """Create heatmap showing which heads are suppressors."""

    layer_names = list(results.keys())
    n_heads = results[layer_names[0]]['n_heads']

    # Create matrix: layers x heads
    suppressor_matrix = np.zeros((len(layer_names), n_heads))
    ratio_matrix = np.zeros((len(layer_names), n_heads))

    for layer_idx, layer_name in enumerate(layer_names):
        for head_stat in results[layer_name]['head_stats']:
            head_idx = head_stat['head_idx']
            ratio_matrix[layer_idx, head_idx] = head_stat['to_ratio']
            if head_stat['is_suppressor']:
                suppressor_matrix[layer_idx, head_idx] = 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Ratio heatmap
    ax = axes[0]
    im = ax.imshow(ratio_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0.5, vmax=1.5)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'H{i}' for i in range(n_heads)])
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names)
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Attention TO Ratio (Slot 0 / Slots 1-6)\n(Blue < 1 = suppressor, Red > 1 = amplifier)')
    plt.colorbar(im, ax=ax)

    # Annotate with values
    for i in range(len(layer_names)):
        for j in range(n_heads):
            text = ax.text(j, i, f'{ratio_matrix[i, j]:.2f}',
                          ha='center', va='center', fontsize=8,
                          color='white' if ratio_matrix[i, j] < 0.8 or ratio_matrix[i, j] > 1.2 else 'black')

    # Suppressor binary heatmap
    ax = axes[1]
    im = ax.imshow(suppressor_matrix, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'H{i}' for i in range(n_heads)])
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names)
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title('Position 1 Suppressors\n(Ratio < 0.8 and p < 0.01)')
    plt.colorbar(im, ax=ax)

    # Count suppressors
    n_suppressors = int(suppressor_matrix.sum())
    total_heads = suppressor_matrix.size

    plt.suptitle(f'Attention Head Specialization Analysis\n({n_suppressors}/{total_heads} heads are slot 0 suppressors)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / '17_suppressor_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved suppressor heatmap to {output_dir / '17_suppressor_heatmap.png'}")


def plot_detailed_head_analysis(results: Dict, output_dir: Path):
    """Create detailed view of most extreme heads."""

    # Find the most extreme suppressors and amplifiers
    all_heads = []
    for layer_name, layer_data in results.items():
        for head_stat in layer_data['head_stats']:
            all_heads.append({
                'layer': layer_name,
                **head_stat
            })

    # Sort by to_ratio
    all_heads.sort(key=lambda x: x['to_ratio'])

    # Take 4 most suppressing and 4 most amplifying
    extreme_heads = all_heads[:4] + all_heads[-4:]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, head_info in enumerate(extreme_heads):
        ax = axes[idx]

        # Bar chart comparing to/from slot 0 vs slots 1-6
        categories = ['To Slot 0', 'To Slots 1-6', 'From Slot 0', 'From Slots 1-6']
        values = [
            head_info['attn_to_slot0_mean'],
            head_info['attn_to_slots16_mean'],
            head_info['attn_from_slot0_mean'],
            head_info['attn_from_slots16_mean'],
        ]
        errors = [
            head_info['attn_to_slot0_std'],
            head_info['attn_to_slots16_std'],
            head_info['attn_from_slot0_std'],
            head_info['attn_from_slots16_std'],
        ]

        colors = ['red', 'blue', 'orange', 'green']
        ax.bar(range(4), values, yerr=errors, color=colors, alpha=0.7, capsize=5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['To\nSlot 0', 'To\nSlots 1-6', 'From\nSlot 0', 'From\nSlots 1-6'], fontsize=8)
        ax.set_ylabel('Attention')

        direction = 'SUPPRESSOR' if head_info['to_ratio'] < 1 else 'AMPLIFIER'
        ax.set_title(f"{head_info['layer']} Head {head_info['head_idx']}\n{direction} (ratio={head_info['to_ratio']:.2f})")

    plt.suptitle('Most Extreme Attention Heads\n(4 strongest suppressors, 4 strongest amplifiers)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / '17_extreme_heads.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved extreme heads analysis to {output_dir / '17_extreme_heads.png'}")


def plot_layer_aggregate(results: Dict, output_dir: Path):
    """Plot aggregate statistics per layer."""

    layer_names = list(results.keys())

    mean_to_ratios = []
    mean_from_ratios = []
    n_suppressors = []

    for layer_name in layer_names:
        layer_data = results[layer_name]

        to_ratios = [s['to_ratio'] for s in layer_data['head_stats']]
        from_ratios = [s['from_ratio'] for s in layer_data['head_stats']]
        suppressors = sum(1 for s in layer_data['head_stats'] if s['is_suppressor'])

        mean_to_ratios.append(np.mean(to_ratios))
        mean_from_ratios.append(np.mean(from_ratios))
        n_suppressors.append(suppressors)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Mean TO ratio per layer
    ax = axes[0]
    ax.bar(range(len(layer_names)), mean_to_ratios, color='blue', alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels([l.replace('layer_', 'L') for l in layer_names])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean TO Ratio')
    ax.set_title('Mean Attention TO Ratio\n(Slot 0 / Slots 1-6)')

    # Mean FROM ratio per layer
    ax = axes[1]
    ax.bar(range(len(layer_names)), mean_from_ratios, color='orange', alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels([l.replace('layer_', 'L') for l in layer_names])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean FROM Ratio')
    ax.set_title('Mean Attention FROM Ratio\n(Slot 0 / Slots 1-6)')

    # Number of suppressors per layer
    ax = axes[2]
    ax.bar(range(len(layer_names)), n_suppressors, color='red', alpha=0.7)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels([l.replace('layer_', 'L') for l in layer_names])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Number of Suppressor Heads')
    ax.set_title('Position 1 Suppressors per Layer\n(TO ratio < 0.8, p < 0.01)')

    plt.tight_layout()
    plt.savefig(output_dir / '17_layer_aggregate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved layer aggregate to {output_dir / '17_layer_aggregate.png'}")


def generate_report(
    results: Dict,
    specialized_heads: List[Dict],
    output_dir: Path
):
    """Generate markdown report."""

    report_path = output_dir.parent / '17-attention-head-specialization.md'

    with open(report_path, 'w') as f:
        f.write("# 17: Attention Head Specialization Analysis\n\n")

        f.write("## Question\n")
        f.write("Do specific attention heads consistently treat position 1 (hand slot 0) differently?\n\n")

        f.write("## Background\n")
        f.write("The model shows systematic bias against slot 0:\n")
        f.write("- Slot 0: r=0.81 correlation with oracle\n")
        f.write("- Slots 1-6: r=0.99+ correlation with oracle\n\n")
        f.write("This analysis examines whether individual attention heads specialize on\n")
        f.write("ignoring or suppressing information flow to/from position 1.\n\n")

        f.write("## Method\n")
        f.write(f"- Model: `{MODEL_PATH.name}`\n")
        f.write(f"- Samples: {N_SAMPLES} validation examples\n")
        f.write("- For each (layer, head) pair:\n")
        f.write("  - Computed attention TO slot 0 from slots 1-6\n")
        f.write("  - Computed attention TO slots 1-6 (internal)\n")
        f.write("  - Computed attention FROM slot 0 to slots 1-6\n")
        f.write("  - Calculated ratio and statistical significance\n")
        f.write("- A head is classified as a 'suppressor' if:\n")
        f.write("  - TO ratio < 0.8 (slots 1-6 attend to slot 0 less than to each other)\n")
        f.write("  - p-value < 0.01 (statistically significant)\n\n")

        f.write("## Position Layout Reminder\n")
        f.write("```\n")
        f.write("For current player P:\n")
        f.write("  Position 1 + P*7 = Slot 0 (first domino in hand)\n")
        f.write("  Positions 2-7 + P*7 = Slots 1-6 (remaining dominoes)\n")
        f.write("```\n\n")

        f.write("## Results\n\n")

        # Per-layer summary table
        f.write("### Per-Layer Summary\n\n")
        f.write("| Layer | Mean TO Ratio | Mean FROM Ratio | Suppressors |\n")
        f.write("|-------|---------------|-----------------|-------------|\n")

        for layer_name, layer_data in results.items():
            to_ratios = [s['to_ratio'] for s in layer_data['head_stats']]
            from_ratios = [s['from_ratio'] for s in layer_data['head_stats']]
            n_suppressors = sum(1 for s in layer_data['head_stats'] if s['is_suppressor'])
            n_heads = layer_data['n_heads']

            f.write(f"| {layer_name} | {np.mean(to_ratios):.3f} | {np.mean(from_ratios):.3f} | {n_suppressors}/{n_heads} |\n")

        f.write("\n")

        # Per-head detailed table
        f.write("### Per-Head TO Ratios\n\n")
        f.write("| Layer | ")
        n_heads = results[list(results.keys())[0]]['n_heads']
        for h in range(n_heads):
            f.write(f"H{h} | ")
        f.write("\n")
        f.write("|-------|" + "----|" * n_heads + "\n")

        for layer_name, layer_data in results.items():
            f.write(f"| {layer_name} | ")
            for head_stat in layer_data['head_stats']:
                ratio = head_stat['to_ratio']
                marker = " **" if head_stat['is_suppressor'] else ""
                marker_end = "**" if head_stat['is_suppressor'] else ""
                f.write(f"{marker}{ratio:.2f}{marker_end} | ")
            f.write("\n")

        f.write("\n**Bold** indicates suppressor heads (ratio < 0.8, p < 0.01)\n\n")

        # Identified suppressors
        f.write("### Identified Position 1 Suppressors\n\n")
        suppressors = [h for h in specialized_heads if h['is_suppressor']]

        if suppressors:
            f.write(f"Found **{len(suppressors)} suppressor heads** that significantly reduce attention to slot 0:\n\n")
            f.write("| Layer | Head | TO Ratio | Effect Size | p-value |\n")
            f.write("|-------|------|----------|-------------|----------|\n")

            for head_info in suppressors[:10]:  # Top 10
                f.write(f"| {head_info['layer']} | {head_info['head']} | {head_info['to_ratio']:.3f} | {head_info['effect_size']:.4f} | {head_info['p_value_to']:.2e} |\n")
        else:
            f.write("No significant suppressor heads identified.\n\n")

        # Amplifiers
        f.write("\n### Identified Position 1 Amplifiers\n\n")
        amplifiers = [h for h in specialized_heads if h['direction'] == 'amplifier']

        if amplifiers:
            f.write(f"Found **{len(amplifiers)} amplifier heads** that increase attention to slot 0:\n\n")
            f.write("| Layer | Head | TO Ratio | Effect Size | p-value |\n")
            f.write("|-------|------|----------|-------------|----------|\n")

            for head_info in amplifiers[:10]:
                f.write(f"| {head_info['layer']} | {head_info['head']} | {head_info['to_ratio']:.3f} | {head_info['effect_size']:.4f} | {head_info['p_value_to']:.2e} |\n")
        else:
            f.write("No significant amplifier heads identified.\n\n")

        # Interpretation
        f.write("\n## Interpretation\n\n")

        total_heads = sum(layer_data['n_heads'] for layer_data in results.values())
        total_suppressors = len(suppressors)
        total_amplifiers = len(amplifiers)

        f.write(f"Out of {total_heads} total attention heads:\n")
        f.write(f"- **{total_suppressors}** are position 1 suppressors\n")
        f.write(f"- **{total_amplifiers}** are position 1 amplifiers\n")
        f.write(f"- **{total_heads - total_suppressors - total_amplifiers}** show no significant differential treatment\n\n")

        if total_suppressors > total_amplifiers:
            f.write("**Net effect: More heads suppress slot 0 than amplify it.**\n\n")
            f.write("This imbalance could contribute to degraded Q-value predictions for slot 0,\n")
            f.write("as the first domino position receives systematically less attention information\n")
            f.write("from other positions during representation learning.\n\n")
        elif total_suppressors < total_amplifiers:
            f.write("**Net effect: More heads amplify slot 0 than suppress it.**\n\n")
            f.write("Attention patterns alone do not explain the slot 0 bias.\n")
            f.write("The root cause must lie elsewhere (training distribution, output head, etc.).\n\n")
        else:
            f.write("**Net effect: Balanced suppression and amplification.**\n\n")
            f.write("No systematic attention imbalance detected.\n\n")

        # Layer-by-layer interpretation
        f.write("### Layer-by-Layer Pattern\n\n")
        for layer_name, layer_data in results.items():
            n_suppressors = sum(1 for s in layer_data['head_stats'] if s['is_suppressor'])
            mean_ratio = np.mean([s['to_ratio'] for s in layer_data['head_stats']])

            if n_suppressors > 0:
                f.write(f"- **{layer_name}**: {n_suppressors} suppressors, mean ratio = {mean_ratio:.3f}\n")
            else:
                f.write(f"- {layer_name}: No suppressors, mean ratio = {mean_ratio:.3f}\n")

        f.write("\n## Visualizations\n\n")
        f.write("- `figures/17_per_head_ratios.png` - TO/FROM ratios for each head\n")
        f.write("- `figures/17_suppressor_heatmap.png` - Layer x Head suppressor map\n")
        f.write("- `figures/17_extreme_heads.png` - Most extreme suppressors and amplifiers\n")
        f.write("- `figures/17_layer_aggregate.png` - Per-layer aggregate statistics\n\n")

        f.write("## Conclusion\n\n")

        if total_suppressors > total_amplifiers * 1.5:
            f.write("**CONFIRMED: Specific attention heads specialize in suppressing slot 0.**\n\n")
            f.write("This provides a plausible mechanism for the observed Q-value bias:\n")
            f.write("1. Multiple heads learn to attend less to slot 0\n")
            f.write("2. This creates information asymmetry in the representation\n")
            f.write("3. The Q-value head extracts degraded information for slot 0\n")
            f.write("4. Result: r=0.81 vs r=0.99+ correlation with oracle\n")
        elif total_suppressors > 0:
            f.write("**PARTIAL: Some attention heads suppress slot 0, but effect is limited.**\n\n")
            f.write("The attention patterns show some differential treatment, but\n")
            f.write("this alone may not fully explain the slot 0 bias.\n")
            f.write("Other factors (training distribution, domino ordering) likely contribute.\n")
        else:
            f.write("**NOT CONFIRMED: No systematic attention head suppression of slot 0.**\n\n")
            f.write("Attention patterns do not explain the slot 0 bias.\n")
            f.write("Root cause likely in training data distribution (see 12-domino-ordering.md).\n")

    print(f"Saved report to {report_path}")


def main():
    """Main analysis pipeline."""

    print("=" * 60)
    print("Attention Head Specialization Analysis")
    print("=" * 60)

    # Load model
    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # Print architecture info
    n_layers = len(model.model.transformer.layers)
    n_heads = model.model.transformer.layers[0].self_attn.num_heads
    print(f"Model: {n_layers} layers, {n_heads} heads per layer")

    # Load data
    tokens, masks, players = load_validation_data(N_SAMPLES)

    # Extract attention weights
    print("\n" + "=" * 60)
    print("Extracting Attention Weights")
    print("=" * 60)
    attention_weights = extract_per_head_attention(model, tokens, masks, players)

    # Analyze per-head position treatment
    print("\n" + "=" * 60)
    print("Analyzing Per-Head Position Treatment")
    print("=" * 60)
    results = analyze_per_head_position_treatment(attention_weights, players)

    # Identify specialized heads
    print("\n" + "=" * 60)
    print("Identifying Specialized Heads")
    print("=" * 60)
    specialized_heads = identify_specialized_heads(results)

    suppressors = [h for h in specialized_heads if h['is_suppressor']]
    amplifiers = [h for h in specialized_heads if h['direction'] == 'amplifier']

    print(f"\nFound {len(suppressors)} suppressor heads")
    print(f"Found {len(amplifiers)} amplifier heads")

    if suppressors:
        print("\nTop suppressors:")
        for head in suppressors[:5]:
            print(f"  {head['layer']} H{head['head']}: ratio={head['to_ratio']:.3f}, p={head['p_value_to']:.2e}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    plot_per_head_ratios(results, OUTPUT_DIR)
    plot_suppressor_heatmap(results, OUTPUT_DIR)
    plot_detailed_head_analysis(results, OUTPUT_DIR)
    plot_layer_aggregate(results, OUTPUT_DIR)

    # Generate report
    print("\n" + "=" * 60)
    print("Generating Report")
    print("=" * 60)
    generate_report(results, specialized_heads, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
