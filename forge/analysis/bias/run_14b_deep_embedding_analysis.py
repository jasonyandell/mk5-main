"""
Investigation 14b: Deep dive into transformer encoder embeddings.

The initial embedding analysis (14) found only small effect sizes in encoder output.
This doesn't explain the large r=0.81 vs r=0.99+ difference.

This analysis digs deeper:
1. Analyze embeddings at each transformer layer
2. Check if attention patterns differ for position 1
3. Examine if specific embedding dimensions are problematic
4. Look at correlation between embedding stats and Q-value error
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import pearsonr

# Suppress nested tensor warning
warnings.filterwarnings("ignore", message=".*nested tensors.*prototype.*")

PROJECT_ROOT = Path("/home/jason/v2/mk5-tailwind")
sys.path.insert(0, str(PROJECT_ROOT))

from forge.ml.module import DominoLightningModule


def extract_layer_outputs(model, tokens, mask, current_player):
    """
    Extract embeddings at each transformer layer.

    Returns:
        layer_outputs: list of (batch, seq_len, embed_dim) for each layer
        hand_reprs: list of (batch, 7, embed_dim) - hand positions at each layer
    """
    device = tokens.device

    # Build embeddings from token features (same as forward)
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

    attn_mask = (mask == 0)

    # Store input embedding
    layer_outputs = [x.clone()]

    # Run through each encoder layer
    for layer in model.model.transformer.layers:
        x = layer(x, src_key_padding_mask=attn_mask)
        layer_outputs.append(x.clone())

    # Extract hand positions at each layer
    start_indices = 1 + current_player * 7
    offsets = torch.arange(7, device=device)
    gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
    gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, model.model.embed_dim)

    hand_reprs = []
    for layer_out in layer_outputs:
        hand_repr = torch.gather(layer_out, dim=1, index=gather_indices)
        hand_reprs.append(hand_repr)

    return layer_outputs, hand_reprs


def extract_embeddings_and_qvals(model, tokens, mask, current_player, qvals, legal, teams):
    """
    Extract encoder output hand embeddings and corresponding Q-value prediction errors.

    Returns:
        hand_repr: (batch, 7, embed_dim)
        pred_q: (batch, 7) - predicted Q-values
        oracle_q: (batch, 7) - oracle Q-values (from current player perspective)
        legal_mask: (batch, 7) - legal action mask
        q_error: (batch, 7) - absolute Q-value prediction error
    """
    device = tokens.device

    # Build embeddings
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

    attn_mask = (mask == 0)
    encoder_out = model.model.transformer(x, src_key_padding_mask=attn_mask)

    # Extract hand representations
    start_indices = 1 + current_player * 7
    offsets = torch.arange(7, device=device)
    gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
    gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, model.model.embed_dim)

    hand_repr = torch.gather(encoder_out, dim=1, index=gather_indices)

    # Get predicted Q-values
    pred_q = model.model.output_proj(hand_repr).squeeze(-1)

    # Oracle Q-values from current player perspective
    team_sign = torch.where(teams == 0, 1.0, -1.0).unsqueeze(1)
    oracle_q = qvals * team_sign

    # Legal mask
    legal_mask = legal > 0

    # Q-value error
    q_error = torch.abs(pred_q - oracle_q)

    return hand_repr, pred_q, oracle_q, legal_mask, q_error


def main():
    print("=" * 70)
    print("Investigation 14b: Deep Embedding Analysis")
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
    print(f"Model has {n_layers} transformer layers, embed_dim={embed_dim}")

    # Load validation data
    data_path = PROJECT_ROOT / "data/tokenized-full/val"
    print(f"Loading validation data from: {data_path}")

    tokens = np.load(data_path / "tokens.npy")
    masks = np.load(data_path / "masks.npy")
    players = np.load(data_path / "players.npy")
    qvals = np.load(data_path / "qvals.npy")
    legal = np.load(data_path / "legal.npy")
    teams = np.load(data_path / "teams.npy")

    n_samples = len(tokens)
    print(f"Total validation samples: {n_samples:,}")

    # Part 1: Layer-by-layer norm analysis
    print("\n" + "=" * 70)
    print("PART 1: Layer-by-layer embedding norm analysis")
    print("=" * 70)

    batch_size = 2048
    layer_slot_norms = [[] for _ in range(n_layers + 1)]

    print(f"\nProcessing {min(50000, n_samples):,} samples...")

    with torch.no_grad():
        for start_idx in range(0, min(50000, n_samples), batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            batch_tokens = torch.from_numpy(tokens[start_idx:end_idx].astype(np.int64)).to(device)
            batch_masks = torch.from_numpy(masks[start_idx:end_idx].astype(np.float32)).to(device)
            batch_players = torch.from_numpy(players[start_idx:end_idx].astype(np.int64)).to(device)

            _, hand_reprs = extract_layer_outputs(model, batch_tokens, batch_masks, batch_players)

            for layer_idx, hand_repr in enumerate(hand_reprs):
                # Per-slot norms: (batch, 7)
                slot_norms = torch.norm(hand_repr, p=2, dim=-1).cpu().numpy()
                layer_slot_norms[layer_idx].append(slot_norms)

    # Concatenate
    layer_slot_norms = [np.concatenate(x, axis=0) for x in layer_slot_norms]

    print("\nPer-layer, per-slot mean L2 norms:")
    print("-" * 70)
    print("Layer   | Slot 0  | Slot 1  | Slot 2  | Slot 3  | Slot 4  | Slot 5  | Slot 6  | Slot0 vs Mean")
    print("-" * 70)

    for layer_idx in range(n_layers + 1):
        norms = layer_slot_norms[layer_idx]  # (n_samples, 7)
        slot_means = norms.mean(axis=0)
        mean_1_6 = slot_means[1:].mean()
        diff_pct = 100 * (slot_means[0] - mean_1_6) / mean_1_6

        layer_name = f"Input" if layer_idx == 0 else f"L{layer_idx}"
        print(f"{layer_name:7s} | {slot_means[0]:.4f} | {slot_means[1]:.4f} | {slot_means[2]:.4f} | "
              f"{slot_means[3]:.4f} | {slot_means[4]:.4f} | {slot_means[5]:.4f} | {slot_means[6]:.4f} | "
              f"{diff_pct:+.2f}%")

    # Part 2: Correlation between Q-error and embedding properties
    print("\n" + "=" * 70)
    print("PART 2: Q-value error correlation with embedding properties")
    print("=" * 70)

    all_q_error = []
    all_embed_norm = []
    all_embed_var = []
    all_slots = []

    print(f"\nProcessing all {n_samples:,} samples...")

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            batch_tokens = torch.from_numpy(tokens[start_idx:end_idx].astype(np.int64)).to(device)
            batch_masks = torch.from_numpy(masks[start_idx:end_idx].astype(np.float32)).to(device)
            batch_players = torch.from_numpy(players[start_idx:end_idx].astype(np.int64)).to(device)
            batch_qvals = torch.from_numpy(qvals[start_idx:end_idx].astype(np.float32)).to(device)
            batch_legal = torch.from_numpy(legal[start_idx:end_idx].astype(np.float32)).to(device)
            batch_teams = torch.from_numpy(teams[start_idx:end_idx].astype(np.int64)).to(device)

            hand_repr, pred_q, oracle_q, legal_mask, q_error = extract_embeddings_and_qvals(
                model, batch_tokens, batch_masks, batch_players, batch_qvals, batch_legal, batch_teams
            )

            # Embedding stats per slot
            embed_norm = torch.norm(hand_repr, p=2, dim=-1)  # (batch, 7)
            embed_var = torch.var(hand_repr, dim=-1)  # (batch, 7)

            # Only consider legal actions
            for slot in range(7):
                mask = legal_mask[:, slot].cpu().numpy()
                if mask.sum() > 0:
                    all_q_error.extend(q_error[mask, slot].cpu().numpy().tolist())
                    all_embed_norm.extend(embed_norm[mask, slot].cpu().numpy().tolist())
                    all_embed_var.extend(embed_var[mask, slot].cpu().numpy().tolist())
                    all_slots.extend([slot] * mask.sum())

            if (start_idx // batch_size) % 100 == 0:
                print(f"  Processed {end_idx:,} / {n_samples:,}...")

    all_q_error = np.array(all_q_error)
    all_embed_norm = np.array(all_embed_norm)
    all_embed_var = np.array(all_embed_var)
    all_slots = np.array(all_slots)

    print(f"\nTotal legal actions analyzed: {len(all_q_error):,}")

    # Correlation of embedding properties with Q-error
    r_norm, p_norm = pearsonr(all_embed_norm, all_q_error)
    r_var, p_var = pearsonr(all_embed_var, all_q_error)

    print(f"\nCorrelation of embedding properties with |Q-error|:")
    print(f"  Embedding norm: r = {r_norm:.4f}, p = {p_norm:.2e}")
    print(f"  Embedding var:  r = {r_var:.4f}, p = {p_var:.2e}")

    # Part 3: Per-slot Q-error statistics
    print("\n" + "=" * 70)
    print("PART 3: Per-slot Q-value error statistics")
    print("=" * 70)

    print("\nPer-slot Q-error (MAE in points):")
    print("-" * 50)
    for slot in range(7):
        mask = all_slots == slot
        slot_errors = all_q_error[mask]
        mae = np.mean(slot_errors)
        std = np.std(slot_errors)
        n = len(slot_errors)
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: MAE = {mae:.3f} +/- {std:.3f} (n={n:,}){marker}")

    # Part 4: Correlation between oracle Q and predicted Q per slot
    print("\n" + "=" * 70)
    print("PART 4: Oracle vs Predicted Q-value correlation per slot")
    print("=" * 70)

    # Reload and compute correlations per slot
    all_pred_q = {s: [] for s in range(7)}
    all_oracle_q = {s: [] for s in range(7)}

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            batch_tokens = torch.from_numpy(tokens[start_idx:end_idx].astype(np.int64)).to(device)
            batch_masks = torch.from_numpy(masks[start_idx:end_idx].astype(np.float32)).to(device)
            batch_players = torch.from_numpy(players[start_idx:end_idx].astype(np.int64)).to(device)
            batch_qvals = torch.from_numpy(qvals[start_idx:end_idx].astype(np.float32)).to(device)
            batch_legal = torch.from_numpy(legal[start_idx:end_idx].astype(np.float32)).to(device)
            batch_teams = torch.from_numpy(teams[start_idx:end_idx].astype(np.int64)).to(device)

            hand_repr, pred_q, oracle_q, legal_mask, _ = extract_embeddings_and_qvals(
                model, batch_tokens, batch_masks, batch_players, batch_qvals, batch_legal, batch_teams
            )

            for slot in range(7):
                mask = legal_mask[:, slot].cpu().numpy()
                if mask.sum() > 0:
                    all_pred_q[slot].extend(pred_q[mask, slot].cpu().numpy().tolist())
                    all_oracle_q[slot].extend(oracle_q[mask, slot].cpu().numpy().tolist())

    print("\nPer-slot Pearson correlation (model vs oracle Q-values):")
    print("-" * 50)
    for slot in range(7):
        pred = np.array(all_pred_q[slot])
        oracle = np.array(all_oracle_q[slot])
        r, p = pearsonr(pred, oracle)
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: r = {r:.4f} (n={len(pred):,}){marker}")

    # Part 5: Check if slot 0 has different oracle Q distribution
    print("\n" + "=" * 70)
    print("PART 5: Oracle Q-value distribution per slot")
    print("=" * 70)

    print("\nPer-slot oracle Q-value statistics:")
    print("-" * 50)
    for slot in range(7):
        oracle = np.array(all_oracle_q[slot])
        mean_q = np.mean(oracle)
        std_q = np.std(oracle)
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"  Slot {slot}: mean = {mean_q:.2f}, std = {std_q:.2f}{marker}")

    # Check if slot 0 is at a disadvantage in the training data distribution
    slot0_mean = np.mean(all_oracle_q[0])
    slots16_mean = np.mean([np.mean(all_oracle_q[s]) for s in range(1, 7)])
    print(f"\n  Slot 0 mean vs Slots 1-6 mean: {slot0_mean:.2f} vs {slots16_mean:.2f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Compute final statistics
    slot0_r = pearsonr(all_pred_q[0], all_oracle_q[0])[0]
    slots16_r = np.mean([pearsonr(all_pred_q[s], all_oracle_q[s])[0] for s in range(1, 7)])

    slot0_mae = np.mean(all_q_error[all_slots == 0])
    slots16_mae = np.mean([np.mean(all_q_error[all_slots == s]) for s in range(1, 7)])

    print(f"\nSlot 0 correlation: r = {slot0_r:.4f}")
    print(f"Slots 1-6 avg correlation: r = {slots16_r:.4f}")
    print(f"Correlation gap: {slots16_r - slot0_r:.4f}")

    print(f"\nSlot 0 MAE: {slot0_mae:.3f} pts")
    print(f"Slots 1-6 avg MAE: {slots16_mae:.3f} pts")
    print(f"MAE gap: {slot0_mae - slots16_mae:.3f} pts")


if __name__ == "__main__":
    main()
