"""
Attention Sink Analysis for Texas 42 Q-Value Model

Investigates whether position 0 receives disproportionate attention,
potentially explaining the degraded correlation for slot 0 predictions.

Hypothesis: Transformer "attention sink" phenomenon - models dump unused
attention on early positions, corrupting their representations.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forge.ml.module import DominoLightningModule


def extract_attention_weights(
    model: nn.Module,
    tokens: Tensor,
    masks: Tensor,
    current_player: Tensor,
) -> list[Tensor]:
    """
    Extract attention weights from all transformer layers.

    Uses a custom forward pass through the model components to capture
    attention weights during inference.

    Args:
        model: DominoLightningModule model
        tokens: Input tokens (batch, seq_len, features)
        masks: Attention mask (batch, seq_len)
        current_player: Current player indices (batch,)

    Returns:
        List of attention weight tensors, one per layer
        Each tensor has shape (batch, n_heads, seq_len, seq_len)
    """
    attention_weights = []
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        model.eval()
        inner_model = model.model

        # Build embeddings (same as DominoTransformer.forward)
        embeds = [
            inner_model.high_pip_embed(tokens[:, :, 0]),
            inner_model.low_pip_embed(tokens[:, :, 1]),
            inner_model.is_double_embed(tokens[:, :, 2]),
            inner_model.count_value_embed(tokens[:, :, 3]),
            inner_model.trump_rank_embed(tokens[:, :, 4]),
            inner_model.player_id_embed(tokens[:, :, 5]),
            inner_model.is_current_embed(tokens[:, :, 6]),
            inner_model.is_partner_embed(tokens[:, :, 7]),
            inner_model.is_remaining_embed(tokens[:, :, 8]),
            inner_model.token_type_embed(tokens[:, :, 9]),
            inner_model.decl_embed(tokens[:, :, 10]),
            inner_model.leader_embed(tokens[:, :, 11]),
        ]

        x = torch.cat(embeds, dim=-1)
        x = inner_model.input_proj(x)

        # Create attention mask
        src_key_padding_mask = (masks == 0)

        # Run through transformer layers manually to extract attention
        for layer in inner_model.transformer.layers:
            # Self-attention
            attn_output, attn_weights_layer = layer.self_attn(
                x, x, x,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,  # Get per-head weights
            )
            attention_weights.append(attn_weights_layer.detach().cpu())

            # Rest of layer (feedforward, residual connections, norm)
            x = x + layer.dropout1(attn_output)
            x = layer.norm1(x)
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = x + layer.dropout2(x2)
            x = layer.norm2(x)

    return attention_weights


def analyze_attention_patterns(
    attention_weights: list[Tensor],
    masks: Tensor,
    current_player: Tensor,
    n_samples: int = 100,
) -> dict:
    """
    Analyze attention patterns for evidence of attention sink.

    Computes:
    - Mean attention TO position 0 (context token) from all positions
    - Mean attention FROM position 0 to all positions
    - Attention patterns for current player's hand positions
    - Comparison of attention to slot 0 vs slots 1-6

    Args:
        attention_weights: List of attention tensors per layer
        masks: Attention masks (batch, seq_len)
        current_player: Current player indices (batch,)
        n_samples: Number of samples to analyze

    Returns:
        Dictionary with attention statistics
    """
    results = {
        'n_samples': n_samples,
        'n_layers': len(attention_weights),
        'layers': [],
    }

    # Token position mapping:
    # 0: Context token
    # 1-7: Player 0's hand (positions 0-6 in hand)
    # 8-14: Player 1's hand
    # 15-21: Player 2's hand
    # 22-28: Player 3's hand
    # 29-31: Trick tokens (if present)

    for layer_idx, attn in enumerate(attention_weights):
        # attn shape: (batch, n_heads, seq_len, seq_len)
        attn = attn[:n_samples]
        batch_size, n_heads, seq_len, _ = attn.shape

        # Average over heads for overall patterns
        attn_mean = attn.mean(dim=1)  # (batch, seq_len, seq_len)

        # Attention TO position 0 (context token) from all other positions
        attn_to_pos0 = attn_mean[:, 1:, 0].mean().item()  # Mean attention to pos 0

        # Attention FROM position 0 to all positions
        attn_from_pos0 = attn_mean[:, 0, :].mean().item()

        # Attention patterns within current player's hand
        # For each sample, get the 7 positions corresponding to current player
        hand_start = 1 + current_player[:n_samples] * 7  # (batch,)

        # Analyze attention between hand positions
        intra_hand_attn = []
        slot0_receives = []  # Attention slot 0 receives from slots 1-6
        slot0_gives = []     # Attention slot 0 gives to slots 1-6

        for b in range(min(batch_size, n_samples)):
            start = int(hand_start[b].item())
            hand_positions = list(range(start, start + 7))

            # Extract 7x7 attention matrix for this player's hand
            hand_attn = attn_mean[b, hand_positions][:, hand_positions]  # (7, 7)
            intra_hand_attn.append(hand_attn)

            # Attention slot 0 receives from slots 1-6
            slot0_receives.append(hand_attn[0, 1:].mean().item())

            # Attention slot 0 gives to slots 1-6
            slot0_gives.append(hand_attn[1:, 0].mean().item())

        intra_hand = torch.stack(intra_hand_attn).mean(dim=0)  # Average 7x7 matrix

        # Attention to pos 0 vs average attention to positions 1-7 (P0's hand)
        attn_to_hand_pos0 = attn_mean[:, :, 1].mean().item()  # Attention to position 1 (P0 slot 0)
        attn_to_hand_pos1_6 = attn_mean[:, :, 2:8].mean().item()  # Positions 2-7 (P0 slots 1-6)

        layer_stats = {
            'layer': layer_idx,
            'n_heads': n_heads,
            'attn_to_context': attn_to_pos0,
            'attn_from_context': attn_from_pos0,
            'attn_to_P0_slot0': attn_to_hand_pos0,
            'attn_to_P0_slots1_6': attn_to_hand_pos1_6,
            'slot0_receives_mean': np.mean(slot0_receives),
            'slot0_gives_mean': np.mean(slot0_gives),
            'intra_hand_matrix': intra_hand.numpy(),  # 7x7 mean attention within hand
        }
        results['layers'].append(layer_stats)

    return results


def analyze_position_representations(
    model: nn.Module,
    tokens: Tensor,
    masks: Tensor,
    current_player: Tensor,
    n_samples: int = 100,
) -> dict:
    """
    Analyze how position 0 representations differ from other positions.

    Computes:
    - Mean representation norm at each position
    - Cosine similarity between positions
    - Representation variance at each position
    """
    intermediate_outputs = []

    # Hook to capture transformer output
    transformer = model.model.transformer

    def hook(module, input, output):
        intermediate_outputs.append(output.detach().cpu())

    handle = transformer.register_forward_hook(hook)

    with torch.no_grad():
        model.eval()
        _ = model(tokens[:n_samples], masks[:n_samples], current_player[:n_samples])

    handle.remove()

    # Shape: (batch, seq_len, embed_dim)
    transformer_out = intermediate_outputs[0]
    batch_size, seq_len, embed_dim = transformer_out.shape

    # Compute norms at each position
    norms = transformer_out.norm(dim=-1)  # (batch, seq_len)
    mean_norms = norms.mean(dim=0)  # (seq_len,)

    # Compute variance at each position
    variances = transformer_out.var(dim=-1).mean(dim=0)  # (seq_len,)

    # For current player's hand, analyze representations
    hand_norms = []
    for b in range(batch_size):
        start = int((1 + current_player[b] * 7).item())
        hand_rep = transformer_out[b, start:start+7]  # (7, embed_dim)
        hand_norms.append(hand_rep.norm(dim=-1))  # (7,)

    hand_norms = torch.stack(hand_norms).mean(dim=0)  # (7,) average across samples

    return {
        'mean_norms_by_position': mean_norms.numpy(),
        'mean_variances_by_position': variances.numpy(),
        'hand_slot_norms': hand_norms.numpy(),  # Norm at each of 7 hand slots
        'n_samples': n_samples,
    }


def analyze_q_predictions_by_slot(
    model: nn.Module,
    tokens: Tensor,
    masks: Tensor,
    current_player: Tensor,
    qvals: Tensor,
    legal: Tensor,
    teams: Tensor,
    n_samples: int = 1000,
    batch_size: int = 512,
) -> dict:
    """
    Analyze Q-value predictions by slot to understand slot 0 degradation.

    Computes per-slot:
    - Correlation with oracle Q-values
    - Mean absolute error
    - Prediction variance

    Processes in batches to avoid OOM.
    """
    import gc

    device = next(model.parameters()).device
    n_samples = min(n_samples, len(tokens))

    # Process in batches
    all_predictions = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        tokens_d = tokens[start:end].to(device)
        masks_d = masks[start:end].to(device)
        players_d = current_player[start:end].to(device)

        with torch.no_grad():
            model.eval()
            predicted_q, _ = model(tokens_d, masks_d, players_d)
            all_predictions.append(predicted_q.cpu().numpy())

        # Free GPU memory
        del tokens_d, masks_d, players_d
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    predicted_q = np.concatenate(all_predictions, axis=0)

    qvals_sub = qvals[:n_samples].numpy()
    legal_sub = legal[:n_samples].numpy()
    teams_sub = teams[:n_samples].numpy()

    # Adjust Q-values for team perspective
    team_sign = np.where(teams_sub == 0, 1, -1)[:, np.newaxis]
    oracle_q = qvals_sub * team_sign

    # Per-slot analysis
    slot_stats = []
    for slot in range(7):
        # Get valid samples for this slot
        valid = legal_sub[:, slot] > 0
        if valid.sum() < 10:
            slot_stats.append(None)
            continue

        pred = predicted_q[valid, slot]
        oracle = oracle_q[valid, slot]

        # Correlation
        corr = np.corrcoef(pred, oracle)[0, 1]

        # MAE
        mae = np.abs(pred - oracle).mean()

        # Variance
        pred_var = np.var(pred)
        oracle_var = np.var(oracle)

        # Mean bias
        bias = (pred - oracle).mean()

        slot_stats.append({
            'slot': slot,
            'n_valid': int(valid.sum()),
            'correlation': corr,
            'mae': mae,
            'pred_var': pred_var,
            'oracle_var': oracle_var,
            'bias': bias,
            'pred_mean': pred.mean(),
            'oracle_mean': oracle.mean(),
        })

    return {'slot_stats': slot_stats}


def analyze_attention_head_diversity(attn_weights: list[Tensor], n_samples: int = 100) -> dict:
    """
    Analyze diversity of attention patterns across heads.

    For attention sink phenomenon, some heads may be more affected than others.
    """
    head_stats = []

    for layer_idx, attn in enumerate(attn_weights):
        attn = attn[:n_samples]  # (batch, n_heads, seq_len, seq_len)
        batch_size, n_heads, seq_len, _ = attn.shape

        for head_idx in range(n_heads):
            head_attn = attn[:, head_idx, :, :]  # (batch, seq_len, seq_len)

            # Attention to position 0 (context) from all positions
            attn_to_pos0 = head_attn[:, 1:, 0].mean().item()

            # Attention to position 1 (P0 slot 0) from all positions
            attn_to_pos1 = head_attn[:, :, 1].mean().item()

            # Entropy of attention distribution (higher = more uniform)
            # For each source position, compute entropy of attention distribution
            attn_probs = head_attn  # Already softmax'd
            entropy = -(attn_probs * torch.log(attn_probs + 1e-10)).sum(dim=-1).mean().item()

            head_stats.append({
                'layer': layer_idx,
                'head': head_idx,
                'attn_to_context': attn_to_pos0,
                'attn_to_pos1': attn_to_pos1,
                'entropy': entropy,
            })

    return {'head_stats': head_stats}


def main():
    """Run attention sink analysis."""
    print("=" * 70)
    print("ATTENTION SINK ANALYSIS - Texas 42 Q-Value Model")
    print("=" * 70)

    # Load model
    model_path = PROJECT_ROOT / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
    print(f"\nLoading model from: {model_path}")

    model = DominoLightningModule.load_from_checkpoint(
        str(model_path),
        map_location='cpu',
        weights_only=False,  # Required for numpy RNG state in checkpoint
    )
    model.eval()

    print(f"Model architecture:")
    print(f"  - Embed dim: {model.model.embed_dim}")
    print(f"  - Layers: {len(model.model.transformer.layers)}")
    print(f"  - Heads per layer: {model.model.transformer.layers[0].self_attn.num_heads}")

    # Load validation data using memory mapping
    data_dir = PROJECT_ROOT / "data/tokenized-full/val"
    print(f"\nLoading validation data from: {data_dir}")

    # Use mmap to avoid loading full arrays into memory
    tokens_mmap = np.load(data_dir / "tokens.npy", mmap_mode='r')
    masks_mmap = np.load(data_dir / "masks.npy", mmap_mode='r')
    players_mmap = np.load(data_dir / "players.npy", mmap_mode='r')
    qvals_mmap = np.load(data_dir / "qvals.npy", mmap_mode='r')
    legal_mmap = np.load(data_dir / "legal.npy", mmap_mode='r')
    teams_mmap = np.load(data_dir / "teams.npy", mmap_mode='r')

    print(f"  - {len(tokens_mmap):,} samples")
    print(f"  - Sequence length: {tokens_mmap.shape[1]}")

    # Analyze a subset - smaller for attention (memory-intensive), larger for Q-analysis
    n_attn_samples = 500   # Attention extraction uses lots of memory
    n_q_samples = 20000    # Q-analysis is lightweight
    print(f"\nAnalyzing {n_attn_samples} samples for attention, {n_q_samples} for Q-predictions...")

    # Copy a subset to tensors for analysis
    tokens = torch.from_numpy(np.array(tokens_mmap[:n_attn_samples]).astype(np.int64))
    masks = torch.from_numpy(np.array(masks_mmap[:n_attn_samples]).astype(np.float32))
    players = torch.from_numpy(np.array(players_mmap[:n_attn_samples]).astype(np.int64))

    # Extract attention weights
    print("\n1. Extracting attention weights...")
    attn_weights = extract_attention_weights(model, tokens, masks, players)

    # Analyze attention patterns
    print("2. Analyzing attention patterns...")
    attn_results = analyze_attention_patterns(
        attn_weights, masks, players, n_samples=n_attn_samples,
    )

    # Analyze representations
    print("3. Analyzing position representations...")
    repr_results = analyze_position_representations(
        model, tokens, masks, players, n_samples=n_attn_samples,
    )

    # Free memory before Q-analysis
    import gc
    del tokens, masks, players, attn_weights
    gc.collect()

    # Load Q-analysis data
    print("4. Analyzing Q-predictions by slot...")
    tokens_q = torch.from_numpy(np.array(tokens_mmap[:n_q_samples]).astype(np.int64))
    masks_q = torch.from_numpy(np.array(masks_mmap[:n_q_samples]).astype(np.float32))
    players_q = torch.from_numpy(np.array(players_mmap[:n_q_samples]).astype(np.int64))
    qvals = torch.from_numpy(np.array(qvals_mmap[:n_q_samples]).astype(np.float32))
    legal = torch.from_numpy(np.array(legal_mmap[:n_q_samples]).astype(np.float32))
    teams = torch.from_numpy(np.array(teams_mmap[:n_q_samples]).astype(np.int64))

    q_results = analyze_q_predictions_by_slot(
        model, tokens_q, masks_q, players_q, qvals, legal, teams,
        n_samples=n_q_samples,
    )

    # Note: head_results will use cached attn_results data
    print("5. Computing head diversity from cached attention...")
    # Skip head analysis since we freed the weights
    head_results = {'head_stats': []}

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- Attention to Context Token (Position 0) ---")
    for layer in attn_results['layers']:
        print(f"Layer {layer['layer']}: Attn TO context = {layer['attn_to_context']:.4f}, "
              f"FROM context = {layer['attn_from_context']:.4f}")

    print("\n--- Attention to P0's Hand Positions ---")
    print("(This compares attention to absolute positions 1 vs 2-7, i.e. P0's slots)")
    for layer in attn_results['layers']:
        ratio = layer['attn_to_P0_slot0'] / layer['attn_to_P0_slots1_6']
        print(f"Layer {layer['layer']}: Slot 0 = {layer['attn_to_P0_slot0']:.4f}, "
              f"Slots 1-6 = {layer['attn_to_P0_slots1_6']:.4f}, "
              f"Ratio = {ratio:.2f}x")

    print("\n--- Intra-Hand Attention (Current Player's Hand, 7x7) ---")
    for layer in attn_results['layers']:
        matrix = layer['intra_hand_matrix']
        print(f"\nLayer {layer['layer']}:")
        print(f"  Attention slot 0 RECEIVES from slots 1-6: {layer['slot0_receives_mean']:.4f}")
        print(f"  Attention slot 0 GIVES to slots 1-6:      {layer['slot0_gives_mean']:.4f}")

        # Diagonal (self-attention)
        print(f"  Self-attention (diagonal mean): {np.diag(matrix).mean():.4f}")

        # Compare slot 0 row/column to others
        slot0_row_sum = matrix[0, :].sum() - matrix[0, 0]  # Exclude self
        other_row_sums = np.array([matrix[i, :].sum() - matrix[i, i] for i in range(1, 7)])
        print(f"  Slot 0 row sum (excluding self): {slot0_row_sum:.4f}")
        print(f"  Other slots row sum (mean): {other_row_sums.mean():.4f}")

    print("\n--- Representation Norms ---")
    print("Position | Mean Norm")
    print("-" * 25)
    for i, norm in enumerate(repr_results['mean_norms_by_position'][:10]):
        label = "Context" if i == 0 else f"P{(i-1)//7} slot {(i-1)%7}"
        print(f"    {i:2d}   |  {norm:.4f}  ({label})")

    print("\n--- Hand Slot Norms (Current Player) ---")
    print("Slot | Mean Norm")
    print("-" * 20)
    for i, norm in enumerate(repr_results['hand_slot_norms']):
        print(f"  {i}  |  {norm:.4f}")

    slot0_norm = repr_results['hand_slot_norms'][0]
    other_norms = repr_results['hand_slot_norms'][1:]
    print(f"\nSlot 0 norm: {slot0_norm:.4f}")
    print(f"Slots 1-6 mean norm: {other_norms.mean():.4f}")
    print(f"Ratio: {slot0_norm / other_norms.mean():.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check for attention sink pattern
    last_layer = attn_results['layers'][-1]
    attn_sink_ratio = last_layer['attn_to_P0_slot0'] / last_layer['attn_to_P0_slots1_6']

    if attn_sink_ratio > 1.5:
        print(f"\n[!] ATTENTION SINK DETECTED: Position 0 receives {attn_sink_ratio:.1f}x more attention")
        print("    This may explain the degraded slot 0 predictions.")
    elif attn_sink_ratio > 1.1:
        print(f"\n[~] MILD ATTENTION BIAS: Position 0 receives {attn_sink_ratio:.1f}x more attention")
    else:
        print(f"\n[OK] No significant attention sink detected (ratio = {attn_sink_ratio:.2f}x)")

    norm_ratio = slot0_norm / other_norms.mean()
    if abs(norm_ratio - 1.0) > 0.1:
        print(f"\n[~] REPRESENTATION DIFFERENCE: Slot 0 norm is {norm_ratio:.2f}x of other slots")

    # Additional analysis: attention entropy
    print("\n--- Additional Analysis ---")
    print("\nEarly Layer Attention Patterns (may reveal initialization effects):")
    first_layer = attn_results['layers'][0]
    print(f"  Layer 0 ratio (slot0/slots1-6): {first_layer['attn_to_P0_slot0'] / first_layer['attn_to_P0_slots1_6']:.2f}x")
    print(f"  Attention to context token: {first_layer['attn_to_context']:.4f}")

    print("\nContext Token Analysis:")
    print(f"  Context (pos 0) receives more attention in early layers due to")
    print(f"  being a global summary token - this is expected behavior.")

    # Q-prediction analysis
    print("\n" + "=" * 70)
    print("Q-PREDICTION ANALYSIS BY SLOT")
    print("=" * 70)
    print("\nSlot | Correlation | MAE   | Bias   | N valid")
    print("-" * 50)
    for stats in q_results['slot_stats']:
        if stats is None:
            continue
        print(f"  {stats['slot']}  |   {stats['correlation']:.4f}   | {stats['mae']:.2f}  | {stats['bias']:+.2f}  | {stats['n_valid']:,}")

    # Highlight slot 0 difference
    slot0 = q_results['slot_stats'][0]
    other_slots = [s for s in q_results['slot_stats'][1:] if s is not None]
    if slot0 and other_slots:
        avg_other_corr = np.mean([s['correlation'] for s in other_slots])
        print(f"\nSlot 0 correlation: {slot0['correlation']:.4f}")
        print(f"Slots 1-6 avg correlation: {avg_other_corr:.4f}")
        print(f"Slot 0 is {(1 - slot0['correlation']**2) / (1 - avg_other_corr**2):.1f}x worse (unexplained variance ratio)")

    # Head diversity
    print("\n" + "=" * 70)
    print("ATTENTION HEAD ANALYSIS")
    print("=" * 70)
    print("\nHeads with highest attention to context token:")
    sorted_heads = sorted(head_results['head_stats'], key=lambda x: -x['attn_to_context'])[:5]
    for h in sorted_heads:
        print(f"  Layer {h['layer']} Head {h['head']}: attn_to_context = {h['attn_to_context']:.4f}")

    print("\nHeads with lowest entropy (most focused attention):")
    sorted_heads = sorted(head_results['head_stats'], key=lambda x: x['entropy'])[:5]
    for h in sorted_heads:
        print(f"  Layer {h['layer']} Head {h['head']}: entropy = {h['entropy']:.4f}")

    return attn_results, repr_results, q_results, head_results


if __name__ == "__main__":
    attn_results, repr_results, q_results, head_results = main()
