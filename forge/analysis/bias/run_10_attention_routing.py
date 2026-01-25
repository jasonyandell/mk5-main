"""
10-attention-routing-deep.py - Deep investigation of attention patterns

Investigates whether position 0 is systematically isolated in attention patterns,
which could explain the slot 0 bias (r=0.81 vs r=0.99+ for slots 1-6).

Approach:
1. Load the model and hook into attention layers
2. Run inference on validation data
3. Extract attention weights from ALL layers
4. Visualize attention received/given by each position
5. Quantify isolation of position 0

Output:
- forge/analysis/bias/figures/10_*.png (visualizations)
- forge/analysis/bias/10-attention-routing-deep.md (findings)
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

from forge.ml.module import DominoLightningModule

# === Configuration ===
MODEL_PATH = Path(PROJECT_ROOT) / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
DATA_DIR = Path(PROJECT_ROOT) / "data/tokenized-full/val"
OUTPUT_DIR = Path(PROJECT_ROOT) / "forge/analysis/bias/figures"
N_SAMPLES = 1000  # Number of validation samples to analyze

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class AttentionExtractor:
    """Hook-based attention weight extractor for TransformerEncoder."""

    def __init__(self, model: nn.Module):
        self.attention_weights: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        """Register forward hooks on attention layers."""
        # TransformerEncoder contains TransformerEncoderLayers
        # Each layer has a MultiheadAttention module
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._get_attention_hook(name))
                self.hooks.append(hook)

    def _get_attention_hook(self, name: str):
        """Create a hook function for capturing attention weights."""
        def hook(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            # attn_weights shape: (batch, tgt_len, src_len) when average_attn_weights=True
            # or (batch, num_heads, tgt_len, src_len) when average_attn_weights=False
            if len(output) > 1 and output[1] is not None:
                attn_weights = output[1].detach().cpu()
                self.attention_weights[name].append(attn_weights)
        return hook

    def clear(self):
        """Clear accumulated attention weights."""
        self.attention_weights.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()


def load_model_with_attention_output():
    """Load model and configure for attention weight output."""
    print(f"Loading model from {MODEL_PATH}")
    # Workaround for PyTorch 2.6+ weights_only default change
    # The checkpoint contains RNG state with numpy arrays which requires unsafe loading
    # This is a trusted local checkpoint so weights_only=False is safe
    model = DominoLightningModule.load_from_checkpoint(
        str(MODEL_PATH),
        map_location='cpu',  # Load to CPU first
        weights_only=False,  # Required for checkpoints with numpy RNG state
    )
    model.eval()

    # Configure MultiheadAttention modules to return attention weights
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            # Need to set need_weights=True in forward calls
            # This is done via a wrapper since we can't modify the forward call directly
            pass

    return model


def patch_mha_for_weights(model: nn.Module):
    """Patch MultiheadAttention modules to always return attention weights."""
    # Find all TransformerEncoderLayer modules and patch their self_attn
    for name, module in model.named_modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            # Store original forward
            original_self_attn_forward = module.self_attn.forward

            def make_patched_forward(orig_forward):
                def patched_forward(*args, **kwargs):
                    kwargs['need_weights'] = True
                    kwargs['average_attn_weights'] = False  # Get per-head weights
                    return orig_forward(*args, **kwargs)
                return patched_forward

            module.self_attn.forward = make_patched_forward(original_self_attn_forward)


def load_validation_data(n_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load validation data."""
    print(f"Loading validation data from {DATA_DIR}")

    tokens = torch.from_numpy(np.load(DATA_DIR / "tokens.npy")[:n_samples])
    masks = torch.from_numpy(np.load(DATA_DIR / "masks.npy")[:n_samples])
    players = torch.from_numpy(np.load(DATA_DIR / "players.npy")[:n_samples])

    print(f"Loaded {len(tokens)} samples")
    print(f"Token shape: {tokens.shape}")  # (batch, seq_len, features)
    print(f"Mask shape: {masks.shape}")

    return tokens, masks, players


def run_inference_with_hooks(
    model: DominoLightningModule,
    tokens: torch.Tensor,
    masks: torch.Tensor,
    players: torch.Tensor,
    batch_size: int = 64
) -> Dict[str, torch.Tensor]:
    """Run inference and collect attention weights from all layers."""

    # We need to manually forward through the transformer to get attention weights
    # Since nn.TransformerEncoder doesn't expose them directly

    device = next(model.parameters()).device
    all_attention_weights = defaultdict(list)

    n_samples = len(tokens)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"Running inference on {n_samples} samples in {n_batches} batches...")

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)

            batch_tokens = tokens[start:end].to(device).long()  # Embeddings require long tensors
            batch_masks = masks[start:end].to(device)
            batch_players = players[start:end].to(device).long()

            # Get embeddings (replicate model forward logic up to transformer)
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

            # Manually call each encoder layer to get attention weights
            attn_mask = (batch_masks == 0)  # True where padding

            for layer_idx, layer in enumerate(domino_model.transformer.layers):
                # Call self-attention manually
                attn_output, attn_weights = layer.self_attn(
                    x, x, x,
                    key_padding_mask=attn_mask,
                    need_weights=True,
                    average_attn_weights=False  # Get per-head weights
                )

                # Store attention weights (batch, heads, seq, seq)
                all_attention_weights[f"layer_{layer_idx}"].append(attn_weights.cpu())

                # Continue through the layer (residual + feedforward)
                x = layer.norm1(x + layer.dropout1(attn_output))
                ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(ff_output))

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{n_batches} batches")

    # Concatenate all batches
    result = {}
    for layer_name, weight_list in all_attention_weights.items():
        result[layer_name] = torch.cat(weight_list, dim=0)
        print(f"  {layer_name}: {result[layer_name].shape}")

    return result


def analyze_attention_routing(attention_weights: Dict[str, torch.Tensor]) -> Dict:
    """Analyze attention patterns to quantify position isolation."""

    results = {}

    for layer_name, weights in attention_weights.items():
        # weights shape: (n_samples, n_heads, seq_len, seq_len)
        n_samples, n_heads, seq_len, _ = weights.shape

        # Average over heads and samples
        # Shape after mean: (seq_len, seq_len)
        mean_attn = weights.mean(dim=(0, 1)).numpy()

        # Attention RECEIVED by each position (sum over source positions, column sum)
        attn_received = mean_attn.sum(axis=0)  # How much other positions attend TO each position

        # Attention GIVEN by each position (sum over target positions, row sum)
        attn_given = mean_attn.sum(axis=1)  # How much each position attends TO others

        # Per-head analysis
        per_head_attn = weights.mean(dim=0).numpy()  # (n_heads, seq_len, seq_len)

        results[layer_name] = {
            'mean_attn': mean_attn,
            'attn_received': attn_received,
            'attn_given': attn_given,
            'per_head_attn': per_head_attn,
            'n_heads': n_heads,
            'seq_len': seq_len,
        }

        # Print summary statistics for position 0 vs others
        # Focus on hand positions (1-7 for P0, 8-14 for P1, etc. but actual index depends on current player)
        # Position 0 is the context token
        print(f"\n{layer_name}:")
        print(f"  Attention received by position 0 (context): {attn_received[0]:.4f}")
        print(f"  Attention received by positions 1-7 (mean): {attn_received[1:8].mean():.4f}")
        print(f"  Attention given by position 0: {attn_given[0]:.4f}")
        print(f"  Attention given by positions 1-7 (mean): {attn_given[1:8].mean():.4f}")

        # Self-attention (diagonal)
        diag = np.diag(mean_attn)
        print(f"  Self-attention position 0: {diag[0]:.4f}")
        print(f"  Self-attention positions 1-7 (mean): {diag[1:8].mean():.4f}")

    return results


def plot_attention_heatmaps(
    results: Dict,
    output_dir: Path
):
    """Create heatmap visualizations of attention patterns."""

    n_layers = len(results)

    # Plot 1: Mean attention matrix per layer
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    for idx, (layer_name, data) in enumerate(results.items()):
        ax = axes[idx]
        sns.heatmap(
            data['mean_attn'][:32, :32],  # Focus on first 32 positions
            ax=ax,
            cmap='viridis',
            vmin=0,
            xticklabels=8,
            yticklabels=8
        )
        ax.set_title(f'{layer_name}\nMean Attention')
        ax.set_xlabel('Key (attended to)')
        ax.set_ylabel('Query (attends from)')

    plt.tight_layout()
    plt.savefig(output_dir / '10_attention_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention heatmaps to {output_dir / '10_attention_heatmaps.png'}")

    # Plot 2: Attention received/given by position
    fig, axes = plt.subplots(2, n_layers, figsize=(6 * n_layers, 8))
    if n_layers == 1:
        axes = axes.reshape(-1, 1)

    for idx, (layer_name, data) in enumerate(results.items()):
        seq_len = data['seq_len']
        positions = np.arange(min(32, seq_len))

        # Attention received
        ax = axes[0, idx]
        attn_recv = data['attn_received'][:32]
        colors = ['red' if i == 0 else 'blue' for i in positions]
        ax.bar(positions, attn_recv, color=colors, alpha=0.7)
        ax.axhline(y=attn_recv[1:8].mean(), color='green', linestyle='--', label='Mean (pos 1-7)')
        ax.set_title(f'{layer_name}\nAttention Received')
        ax.set_xlabel('Position')
        ax.set_ylabel('Total attention received')
        ax.legend()

        # Attention given
        ax = axes[1, idx]
        attn_given = data['attn_given'][:32]
        ax.bar(positions, attn_given, color=colors, alpha=0.7)
        ax.axhline(y=attn_given[1:8].mean(), color='green', linestyle='--', label='Mean (pos 1-7)')
        ax.set_title(f'{layer_name}\nAttention Given')
        ax.set_xlabel('Position')
        ax.set_ylabel('Total attention given')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / '10_attention_by_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention by position to {output_dir / '10_attention_by_position.png'}")

    # Plot 3: Per-head attention patterns for first layer
    first_layer = list(results.keys())[0]
    data = results[first_layer]
    n_heads = data['n_heads']

    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for head_idx in range(n_heads):
        ax = axes[head_idx]
        head_attn = data['per_head_attn'][head_idx, :32, :32]
        sns.heatmap(head_attn, ax=ax, cmap='viridis', vmin=0, cbar=False,
                    xticklabels=8, yticklabels=8)
        ax.set_title(f'Head {head_idx}')

    # Hide unused axes
    for idx in range(n_heads, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'{first_layer}: Per-Head Attention Patterns', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / '10_per_head_attention_layer0.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-head attention to {output_dir / '10_per_head_attention_layer0.png'}")

    # Plot 4: Focus on position 0 isolation - attention to/from position 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Aggregate across layers
    all_attn_from_0 = []
    all_attn_to_0 = []

    for layer_name, data in results.items():
        mean_attn = data['mean_attn']
        all_attn_from_0.append(mean_attn[0, :32])  # Row 0: attention FROM position 0
        all_attn_to_0.append(mean_attn[:32, 0])    # Column 0: attention TO position 0

    all_attn_from_0 = np.array(all_attn_from_0)
    all_attn_to_0 = np.array(all_attn_to_0)

    positions = np.arange(32)
    layer_names = list(results.keys())

    # Attention FROM position 0 to others
    ax = axes[0]
    for layer_idx, layer_name in enumerate(layer_names):
        ax.plot(positions, all_attn_from_0[layer_idx], label=layer_name, marker='o', markersize=3)
    ax.set_title('Attention FROM Position 0\n(How much pos 0 attends to others)')
    ax.set_xlabel('Target Position')
    ax.set_ylabel('Attention Weight')
    ax.legend()
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Self')

    # Attention TO position 0 from others
    ax = axes[1]
    for layer_idx, layer_name in enumerate(layer_names):
        ax.plot(positions, all_attn_to_0[layer_idx], label=layer_name, marker='o', markersize=3)
    ax.set_title('Attention TO Position 0\n(How much others attend to pos 0)')
    ax.set_xlabel('Source Position')
    ax.set_ylabel('Attention Weight')
    ax.legend()
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Self')

    plt.tight_layout()
    plt.savefig(output_dir / '10_position_0_isolation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved position 0 isolation analysis to {output_dir / '10_position_0_isolation.png'}")


def quantify_isolation(results: Dict) -> Dict:
    """Compute metrics quantifying position 0 isolation."""

    metrics = {}

    for layer_name, data in results.items():
        mean_attn = data['mean_attn']

        # Isolation metric 1: Ratio of attention received by pos 0 vs pos 1-7 mean
        recv_0 = data['attn_received'][0]
        recv_1_7 = data['attn_received'][1:8].mean()
        recv_ratio = recv_0 / recv_1_7 if recv_1_7 > 0 else 0

        # Isolation metric 2: Ratio of attention given by pos 0 vs pos 1-7 mean
        given_0 = data['attn_given'][0]
        given_1_7 = data['attn_given'][1:8].mean()
        given_ratio = given_0 / given_1_7 if given_1_7 > 0 else 0

        # Isolation metric 3: Cross-attention between pos 0 and pos 1-7
        # How much does pos 0 attend to pos 1-7?
        attn_0_to_1_7 = mean_attn[0, 1:8].mean()
        # How much do pos 1-7 attend to pos 0?
        attn_1_7_to_0 = mean_attn[1:8, 0].mean()
        # Compare to pos 1-7 internal attention
        attn_1_7_internal = mean_attn[1:8, 1:8].mean()

        metrics[layer_name] = {
            'recv_ratio': recv_ratio,
            'given_ratio': given_ratio,
            'attn_0_to_1_7': attn_0_to_1_7,
            'attn_1_7_to_0': attn_1_7_to_0,
            'attn_1_7_internal': attn_1_7_internal,
            'isolation_score': (attn_1_7_internal / attn_0_to_1_7 if attn_0_to_1_7 > 0 else float('inf')),
        }

        print(f"\n{layer_name} isolation metrics:")
        print(f"  Attention received ratio (pos 0 / pos 1-7): {recv_ratio:.3f}")
        print(f"  Attention given ratio (pos 0 / pos 1-7): {given_ratio:.3f}")
        print(f"  Attention from pos 0 to pos 1-7: {attn_0_to_1_7:.4f}")
        print(f"  Attention from pos 1-7 to pos 0: {attn_1_7_to_0:.4f}")
        print(f"  Internal attention within pos 1-7: {attn_1_7_internal:.4f}")
        print(f"  Isolation score (internal/from_0): {metrics[layer_name]['isolation_score']:.2f}")

    return metrics


def plot_hand_slot_attention(
    results: Dict,
    output_dir: Path
):
    """Analyze attention patterns specifically among hand slot positions (1-7).

    The bias is about slot 0 IN THE OUTPUT (first domino in hand), not position 0 in the sequence.
    Hand positions are 1-7 for P0 (the player we care about when they play).
    """

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for idx, (layer_name, data) in enumerate(results.items()):
        ax = axes[idx]

        # Extract hand slot attention (positions 1-7 for P0's hand)
        # These are the 7 output slots
        hand_attn = data['mean_attn'][1:8, 1:8]

        # Create heatmap with slot labels
        slot_labels = [f'Slot {i}' for i in range(7)]
        sns.heatmap(
            hand_attn,
            ax=ax,
            cmap='viridis',
            annot=True,
            fmt='.3f',
            xticklabels=slot_labels,
            yticklabels=slot_labels,
        )
        ax.set_title(f'{layer_name}\nHand Slot Attention (P0 positions 1-7)')
        ax.set_xlabel('Key (attended to)')
        ax.set_ylabel('Query (attends from)')

        # Print statistics
        print(f"\n{layer_name} - Hand slot (output position) analysis:")
        for slot in range(7):
            recv = hand_attn[:, slot].sum()
            given = hand_attn[slot, :].sum()
            print(f"  Slot {slot}: received={recv:.4f}, given={given:.4f}")

    plt.tight_layout()
    plt.savefig(output_dir / '10_hand_slot_attention.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved hand slot attention to {output_dir / '10_hand_slot_attention.png'}")

    # Plot: Compare slot 0 vs slots 1-6 attention
    fig, ax = plt.subplots(figsize=(10, 6))

    slot_0_recv = []
    slot_1_6_recv = []
    layer_labels = []

    for layer_name, data in results.items():
        hand_attn = data['mean_attn'][1:8, 1:8]
        slot_0_recv.append(hand_attn[:, 0].sum())
        slot_1_6_recv.append(hand_attn[:, 1:7].sum(axis=0).mean())
        layer_labels.append(layer_name)

    x = np.arange(len(layer_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, slot_0_recv, width, label='Slot 0 (biased)', color='red', alpha=0.7)
    bars2 = ax.bar(x + width/2, slot_1_6_recv, width, label='Slots 1-6 (mean)', color='blue', alpha=0.7)

    ax.set_ylabel('Total Attention Received')
    ax.set_title('Attention Received: Slot 0 vs Slots 1-6\n(Among Hand Positions)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / '10_slot0_vs_others_attention.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved slot comparison to {output_dir / '10_slot0_vs_others_attention.png'}")


def generate_report(results: Dict, metrics: Dict, output_dir: Path):
    """Generate markdown report with findings."""

    report_path = output_dir.parent / '10-attention-routing-deep.md'

    with open(report_path, 'w') as f:
        f.write("# 10: Deep Attention Routing Analysis\n\n")
        f.write("## Question\n")
        f.write("Is position 0 (or hand slot 0) systematically isolated in attention patterns?\n\n")

        f.write("## Method\n")
        f.write(f"- Loaded model: `{MODEL_PATH.name}`\n")
        f.write(f"- Ran inference on {N_SAMPLES} validation samples\n")
        f.write("- Extracted attention weights from all transformer layers\n")
        f.write("- Analyzed attention received/given by each position\n\n")

        f.write("## Model Architecture\n")
        for layer_name, data in results.items():
            f.write(f"- {layer_name}: {data['n_heads']} heads, sequence length {data['seq_len']}\n")
        f.write("\n")

        f.write("## Position Layout\n")
        f.write("```\n")
        f.write("Position 0:     Context token\n")
        f.write("Positions 1-7:  P0's hand (slot 0-6 for current player when P0)\n")
        f.write("Positions 8-14: P1's hand\n")
        f.write("Positions 15-21: P2's hand\n")
        f.write("Positions 22-28: P3's hand\n")
        f.write("Positions 29+:  Trick history\n")
        f.write("```\n\n")

        f.write("## Results\n\n")

        f.write("### Attention Flow Summary\n\n")
        f.write("| Layer | Pos 0 Recv | Pos 1-7 Recv | Ratio | Pos 0 Given | Pos 1-7 Given | Ratio |\n")
        f.write("|-------|------------|--------------|-------|-------------|---------------|-------|\n")

        for layer_name, data in results.items():
            recv_0 = data['attn_received'][0]
            recv_1_7 = data['attn_received'][1:8].mean()
            given_0 = data['attn_given'][0]
            given_1_7 = data['attn_given'][1:8].mean()
            f.write(f"| {layer_name} | {recv_0:.3f} | {recv_1_7:.3f} | {recv_0/recv_1_7:.2f} | {given_0:.3f} | {given_1_7:.3f} | {given_0/given_1_7:.2f} |\n")

        f.write("\n### Isolation Metrics\n\n")
        f.write("| Layer | Attn 0->1-7 | Attn 1-7->0 | Internal 1-7 | Isolation Score |\n")
        f.write("|-------|-------------|-------------|--------------|----------------|\n")

        for layer_name, m in metrics.items():
            f.write(f"| {layer_name} | {m['attn_0_to_1_7']:.4f} | {m['attn_1_7_to_0']:.4f} | {m['attn_1_7_internal']:.4f} | {m['isolation_score']:.2f} |\n")

        f.write("\n### Hand Slot Analysis (Key for Output Bias)\n\n")
        f.write("The bias manifests in **output slots** (the 7 Q-value predictions for a player's hand).\n")
        f.write("For P0, these correspond to sequence positions 1-7.\n\n")

        f.write("| Layer | Slot 0 Recv | Slots 1-6 Recv (mean) | Ratio |\n")
        f.write("|-------|-------------|----------------------|-------|\n")

        for layer_name, data in results.items():
            hand_attn = data['mean_attn'][1:8, 1:8]
            slot_0 = hand_attn[:, 0].sum()
            slots_1_6 = hand_attn[:, 1:7].sum(axis=0).mean()
            f.write(f"| {layer_name} | {slot_0:.4f} | {slots_1_6:.4f} | {slot_0/slots_1_6:.2f} |\n")

        f.write("\n## Visualizations\n\n")
        f.write("- `figures/10_attention_heatmaps.png` - Mean attention matrices per layer\n")
        f.write("- `figures/10_attention_by_position.png` - Attention received/given by position\n")
        f.write("- `figures/10_per_head_attention_layer0.png` - Per-head patterns in first layer\n")
        f.write("- `figures/10_position_0_isolation.png` - Analysis of position 0 connectivity\n")
        f.write("- `figures/10_hand_slot_attention.png` - Attention among hand slots (1-7)\n")
        f.write("- `figures/10_slot0_vs_others_attention.png` - Slot 0 vs slots 1-6 comparison\n\n")

        f.write("## Interpretation\n\n")

        # Compute summary statistics
        last_layer = list(results.keys())[-1]
        last_metrics = metrics[last_layer]
        last_data = results[last_layer]

        hand_attn = last_data['mean_attn'][1:8, 1:8]
        slot_0_recv = hand_attn[:, 0].sum()
        slots_1_6_recv_mean = hand_attn[:, 1:7].sum(axis=0).mean()
        ratio = slot_0_recv / slots_1_6_recv_mean

        if ratio < 0.9:
            f.write(f"**Finding: Slot 0 receives {(1-ratio)*100:.1f}% LESS attention than slots 1-6.**\n\n")
            f.write("This attention deficit provides a plausible mechanism for the slot 0 bias:\n")
            f.write("- Slot 0 representations receive less information from other positions\n")
            f.write("- The Q-value head extracts from position 1 (slot 0) with impoverished representations\n")
            f.write("- This manifests as degraded correlation with oracle (r=0.81 vs r=0.99+)\n\n")
        elif ratio > 1.1:
            f.write(f"**Finding: Slot 0 receives {(ratio-1)*100:.1f}% MORE attention than slots 1-6.**\n\n")
            f.write("Attention routing does NOT explain the slot 0 bias - attention is actually higher.\n")
            f.write("The bias must originate elsewhere (output projection, training dynamics, etc.).\n\n")
        else:
            f.write(f"**Finding: Attention to slot 0 is similar to slots 1-6 (ratio = {ratio:.2f}).**\n\n")
            f.write("Attention routing does NOT explain the slot 0 bias.\n")
            f.write("The bias must originate elsewhere (output projection, training dynamics, etc.).\n\n")

        f.write("## Conclusion\n\n")
        if ratio < 0.9:
            f.write("Attention routing analysis reveals systematic under-attention to hand slot 0.\n")
            f.write("This provides a plausible mechanism for the observed Q-value prediction bias.\n")
        else:
            f.write("Attention routing analysis does NOT reveal systematic isolation of slot 0.\n")
            f.write("The bias likely originates from other architectural or training factors.\n")

    print(f"Saved report to {report_path}")


def main():
    """Main analysis pipeline."""

    print("=" * 60)
    print("Deep Attention Routing Analysis")
    print("=" * 60)

    # Load model
    model = load_model_with_attention_output()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")

    # Print model architecture info
    print(f"\nModel architecture:")
    print(f"  Embed dim: {model.model.embed_dim}")
    print(f"  Num layers: {len(model.model.transformer.layers)}")
    if hasattr(model.model.transformer.layers[0].self_attn, 'num_heads'):
        print(f"  Num heads: {model.model.transformer.layers[0].self_attn.num_heads}")

    # Load data
    tokens, masks, players = load_validation_data(N_SAMPLES)

    # Run inference and collect attention weights
    attention_weights = run_inference_with_hooks(model, tokens, masks, players)

    # Analyze attention patterns
    print("\n" + "=" * 60)
    print("Attention Analysis Results")
    print("=" * 60)
    results = analyze_attention_routing(attention_weights)

    # Quantify isolation
    print("\n" + "=" * 60)
    print("Isolation Metrics")
    print("=" * 60)
    metrics = quantify_isolation(results)

    # Generate visualizations
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    plot_attention_heatmaps(results, OUTPUT_DIR)
    plot_hand_slot_attention(results, OUTPUT_DIR)

    # Generate report
    print("\n" + "=" * 60)
    print("Generating Report")
    print("=" * 60)
    generate_report(results, metrics, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
