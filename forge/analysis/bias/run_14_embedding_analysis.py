"""
Investigation 14: Transformer encoder embedding analysis for positional bias.

Hypothesis: The transformer encoder produces systematically different embeddings for
position 1 (slot 0's source) compared to positions 2-7 (slots 1-6).

Context: The model shows slot 0 has r=0.81 correlation with oracle while slots 1-6 have r=0.99+.
Previous investigations ruled out output head bias - the bias must originate in the transformer
encoder's embeddings.

Analysis:
1. Load the model and run inference on validation samples
2. Extract transformer encoder output (before output projection)
3. For each player's hand block, compare position 1 (slot 0) to positions 2-7 (slots 1-6):
   - L2 norm of embeddings
   - Variance across embedding dimensions
   - Cosine similarity between positions
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

# Suppress nested tensor warning
warnings.filterwarnings("ignore", message=".*nested tensors.*prototype.*")

PROJECT_ROOT = Path("/home/jason/v2/mk5-tailwind")
sys.path.insert(0, str(PROJECT_ROOT))

from forge.ml.module import DominoLightningModule


def extract_encoder_output(model, tokens, mask, current_player):
    """
    Run forward pass but extract the transformer encoder output
    before the output projection.

    Returns:
        encoder_out: (batch, seq_len, embed_dim) - transformer encoder output
        hand_repr: (batch, 7, embed_dim) - hand position embeddings for current player
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

    # Apply transformer with attention mask
    attn_mask = (mask == 0)
    encoder_out = model.model.transformer(x, src_key_padding_mask=attn_mask)

    # Extract hand representations for current player
    # Player's 7 dominoes start at index 1 + player_id * 7
    start_indices = 1 + current_player * 7
    offsets = torch.arange(7, device=device)
    gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
    gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, model.model.embed_dim)

    hand_repr = torch.gather(encoder_out, dim=1, index=gather_indices)

    return encoder_out, hand_repr


def analyze_embeddings(hand_repr):
    """
    Analyze embedding statistics for slot 0 (position 1) vs slots 1-6 (positions 2-7).

    Args:
        hand_repr: (batch, 7, embed_dim) - hand position embeddings

    Returns:
        dict with analysis results
    """
    batch_size = hand_repr.shape[0]
    embed_dim = hand_repr.shape[2]

    # Slot 0 embedding (position 1 in input sequence)
    slot0_embed = hand_repr[:, 0, :]  # (batch, embed_dim)

    # Slots 1-6 embeddings (positions 2-7 in input sequence)
    slots1_6_embed = hand_repr[:, 1:, :]  # (batch, 6, embed_dim)

    # 1. L2 norm analysis
    slot0_norm = torch.norm(slot0_embed, p=2, dim=-1)  # (batch,)
    slots1_6_norm = torch.norm(slots1_6_embed, p=2, dim=-1)  # (batch, 6)

    # 2. Variance across embedding dimensions
    slot0_var = torch.var(slot0_embed, dim=-1)  # (batch,)
    slots1_6_var = torch.var(slots1_6_embed, dim=-1)  # (batch, 6)

    # 3. Cosine similarity between positions
    # Compute pairwise cosine similarity within the hand
    # Normalize embeddings
    hand_norm = F.normalize(hand_repr, p=2, dim=-1)  # (batch, 7, embed_dim)

    # Cosine similarity matrix: (batch, 7, 7)
    cos_sim = torch.bmm(hand_norm, hand_norm.transpose(1, 2))

    # Similarity of slot 0 to other slots
    slot0_to_others = cos_sim[:, 0, 1:]  # (batch, 6) - slot 0 vs slots 1-6

    # Average similarity among slots 1-6
    # Upper triangular part of the 6x6 submatrix
    slots1_6_sim_matrix = cos_sim[:, 1:, 1:]  # (batch, 6, 6)
    triu_indices = torch.triu_indices(6, 6, offset=1)
    slots1_6_pairwise = slots1_6_sim_matrix[:, triu_indices[0], triu_indices[1]]  # (batch, 15)

    # Convert to numpy for stats
    results = {
        'slot0_norm': slot0_norm.cpu().numpy(),
        'slots1_6_norm_mean': slots1_6_norm.mean(dim=1).cpu().numpy(),
        'slots1_6_norm_all': slots1_6_norm.cpu().numpy(),

        'slot0_var': slot0_var.cpu().numpy(),
        'slots1_6_var_mean': slots1_6_var.mean(dim=1).cpu().numpy(),
        'slots1_6_var_all': slots1_6_var.cpu().numpy(),

        'slot0_to_others_mean': slot0_to_others.mean(dim=1).cpu().numpy(),
        'slot0_to_others_all': slot0_to_others.cpu().numpy(),
        'slots1_6_pairwise_mean': slots1_6_pairwise.mean(dim=1).cpu().numpy(),

        # Per-slot metrics for detailed analysis
        'per_slot_norm': torch.norm(hand_repr, p=2, dim=-1).cpu().numpy(),  # (batch, 7)
        'per_slot_var': torch.var(hand_repr, dim=-1).cpu().numpy(),  # (batch, 7)
    }

    return results


def statistical_tests(results):
    """
    Run statistical tests comparing slot 0 to slots 1-6.
    """
    tests = {}

    # 1. L2 norm: paired t-test (slot 0 norm vs mean of slots 1-6 norm)
    t_norm, p_norm = stats.ttest_rel(results['slot0_norm'], results['slots1_6_norm_mean'])
    tests['norm_ttest'] = {'t': t_norm, 'p': p_norm}

    # Cohen's d for norm difference
    diff_norm = results['slot0_norm'] - results['slots1_6_norm_mean']
    cohens_d_norm = np.mean(diff_norm) / np.std(diff_norm)
    tests['norm_cohens_d'] = cohens_d_norm

    # 2. Variance: paired t-test
    t_var, p_var = stats.ttest_rel(results['slot0_var'], results['slots1_6_var_mean'])
    tests['var_ttest'] = {'t': t_var, 'p': p_var}

    diff_var = results['slot0_var'] - results['slots1_6_var_mean']
    cohens_d_var = np.mean(diff_var) / np.std(diff_var)
    tests['var_cohens_d'] = cohens_d_var

    # 3. Cosine similarity: paired t-test
    # Compare: slot 0's similarity to others vs slots 1-6's pairwise similarity
    t_sim, p_sim = stats.ttest_rel(
        results['slot0_to_others_mean'],
        results['slots1_6_pairwise_mean']
    )
    tests['sim_ttest'] = {'t': t_sim, 'p': p_sim}

    diff_sim = results['slot0_to_others_mean'] - results['slots1_6_pairwise_mean']
    cohens_d_sim = np.mean(diff_sim) / np.std(diff_sim)
    tests['sim_cohens_d'] = cohens_d_sim

    # 4. One-way ANOVA across all 7 slots for each metric
    per_slot_norm = results['per_slot_norm']  # (n_samples, 7)
    f_norm, p_anova_norm = stats.f_oneway(*[per_slot_norm[:, i] for i in range(7)])
    tests['anova_norm'] = {'F': f_norm, 'p': p_anova_norm}

    per_slot_var = results['per_slot_var']  # (n_samples, 7)
    f_var, p_anova_var = stats.f_oneway(*[per_slot_var[:, i] for i in range(7)])
    tests['anova_var'] = {'F': f_var, 'p': p_anova_var}

    return tests


def main():
    print("=" * 70)
    print("Investigation 14: Transformer Encoder Embedding Analysis")
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

    # Load validation data
    data_path = PROJECT_ROOT / "data/tokenized-full/val"
    print(f"Loading validation data from: {data_path}")

    tokens = np.load(data_path / "tokens.npy")
    masks = np.load(data_path / "masks.npy")
    players = np.load(data_path / "players.npy")

    n_samples = len(tokens)
    print(f"Total validation samples: {n_samples:,}")

    # Process in batches
    batch_size = 2048
    all_results = []

    print(f"\nProcessing {n_samples:,} samples in batches of {batch_size}...")

    with torch.no_grad():
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)

            batch_tokens = torch.from_numpy(tokens[start_idx:end_idx].astype(np.int64)).to(device)
            batch_masks = torch.from_numpy(masks[start_idx:end_idx].astype(np.float32)).to(device)
            batch_players = torch.from_numpy(players[start_idx:end_idx].astype(np.int64)).to(device)

            # Extract encoder output
            _, hand_repr = extract_encoder_output(model, batch_tokens, batch_masks, batch_players)

            # Analyze this batch
            batch_results = analyze_embeddings(hand_repr)
            all_results.append(batch_results)

            if (start_idx // batch_size) % 50 == 0:
                print(f"  Processed {end_idx:,} / {n_samples:,} samples...")

    print(f"\nConcatenating results from {len(all_results)} batches...")

    # Concatenate results
    combined = {}
    for key in all_results[0].keys():
        combined[key] = np.concatenate([r[key] for r in all_results], axis=0)

    # Run statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    tests = statistical_tests(combined)

    # Print summary statistics
    print("\n1. L2 NORM OF EMBEDDINGS")
    print("-" * 40)
    slot0_norm_mean = np.mean(combined['slot0_norm'])
    slot0_norm_std = np.std(combined['slot0_norm'])
    slots16_norm_mean = np.mean(combined['slots1_6_norm_mean'])
    slots16_norm_std = np.std(combined['slots1_6_norm_mean'])

    print(f"   Slot 0:     mean = {slot0_norm_mean:.4f}, std = {slot0_norm_std:.4f}")
    print(f"   Slots 1-6:  mean = {slots16_norm_mean:.4f}, std = {slots16_norm_std:.4f}")
    print(f"   Difference: {slot0_norm_mean - slots16_norm_mean:.4f} ({100*(slot0_norm_mean - slots16_norm_mean)/slots16_norm_mean:+.2f}%)")
    print(f"   t-test: t={tests['norm_ttest']['t']:.2f}, p={tests['norm_ttest']['p']:.2e}")
    print(f"   Cohen's d = {tests['norm_cohens_d']:.3f}")
    print(f"   ANOVA F={tests['anova_norm']['F']:.2f}, p={tests['anova_norm']['p']:.2e}")

    # Per-slot norm breakdown
    print("\n   Per-slot mean norms:")
    per_slot_norm = combined['per_slot_norm']
    for slot in range(7):
        slot_mean = np.mean(per_slot_norm[:, slot])
        slot_std = np.std(per_slot_norm[:, slot])
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"     Slot {slot}: {slot_mean:.4f} +/- {slot_std:.4f}{marker}")

    print("\n2. VARIANCE ACROSS EMBEDDING DIMENSIONS")
    print("-" * 40)
    slot0_var_mean = np.mean(combined['slot0_var'])
    slot0_var_std = np.std(combined['slot0_var'])
    slots16_var_mean = np.mean(combined['slots1_6_var_mean'])
    slots16_var_std = np.std(combined['slots1_6_var_mean'])

    print(f"   Slot 0:     mean = {slot0_var_mean:.6f}, std = {slot0_var_std:.6f}")
    print(f"   Slots 1-6:  mean = {slots16_var_mean:.6f}, std = {slots16_var_std:.6f}")
    print(f"   Difference: {slot0_var_mean - slots16_var_mean:.6f} ({100*(slot0_var_mean - slots16_var_mean)/slots16_var_mean:+.2f}%)")
    print(f"   t-test: t={tests['var_ttest']['t']:.2f}, p={tests['var_ttest']['p']:.2e}")
    print(f"   Cohen's d = {tests['var_cohens_d']:.3f}")
    print(f"   ANOVA F={tests['anova_var']['F']:.2f}, p={tests['anova_var']['p']:.2e}")

    # Per-slot variance breakdown
    print("\n   Per-slot mean variance:")
    per_slot_var = combined['per_slot_var']
    for slot in range(7):
        slot_mean = np.mean(per_slot_var[:, slot])
        slot_std = np.std(per_slot_var[:, slot])
        marker = " <-- SLOT 0" if slot == 0 else ""
        print(f"     Slot {slot}: {slot_mean:.6f} +/- {slot_std:.6f}{marker}")

    print("\n3. COSINE SIMILARITY ANALYSIS")
    print("-" * 40)
    slot0_sim_mean = np.mean(combined['slot0_to_others_mean'])
    slot0_sim_std = np.std(combined['slot0_to_others_mean'])
    slots16_sim_mean = np.mean(combined['slots1_6_pairwise_mean'])
    slots16_sim_std = np.std(combined['slots1_6_pairwise_mean'])

    print(f"   Slot 0 to others:     mean = {slot0_sim_mean:.4f}, std = {slot0_sim_std:.4f}")
    print(f"   Slots 1-6 pairwise:   mean = {slots16_sim_mean:.4f}, std = {slots16_sim_std:.4f}")
    print(f"   Difference: {slot0_sim_mean - slots16_sim_mean:.4f}")
    print(f"   t-test: t={tests['sim_ttest']['t']:.2f}, p={tests['sim_ttest']['p']:.2e}")
    print(f"   Cohen's d = {tests['sim_cohens_d']:.3f}")

    # Detailed slot 0 similarity to each other slot
    print("\n   Slot 0 cosine similarity to each slot:")
    slot0_to_others_all = combined['slot0_to_others_all']  # (n_samples, 6)
    for i in range(6):
        other_slot = i + 1
        sim_mean = np.mean(slot0_to_others_all[:, i])
        sim_std = np.std(slot0_to_others_all[:, i])
        print(f"     Slot 0 <-> Slot {other_slot}: {sim_mean:.4f} +/- {sim_std:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Determine if there's a significant difference
    significant = any([
        tests['norm_ttest']['p'] < 0.001,
        tests['var_ttest']['p'] < 0.001,
        tests['sim_ttest']['p'] < 0.001,
    ])

    large_effect = any([
        abs(tests['norm_cohens_d']) > 0.5,
        abs(tests['var_cohens_d']) > 0.5,
        abs(tests['sim_cohens_d']) > 0.5,
    ])

    if significant and large_effect:
        print("\nSLOT 0 EMBEDDINGS ARE STATISTICALLY DIFFERENT from slots 1-6.")
        print("This confirms the transformer encoder as the source of positional bias.")
    elif significant:
        print("\nStatistically significant but SMALL EFFECT SIZE detected.")
        print("Slot 0 embeddings are measurably different but the effect is small.")
    else:
        print("\nNo significant systematic difference found in slot 0 embeddings.")
        print("The positional bias must originate elsewhere in the architecture.")

    # Save results
    output_dir = PROJECT_ROOT / "forge/analysis/bias"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed statistics
    stats_dict = {
        'n_samples': n_samples,
        'slot0_norm_mean': float(slot0_norm_mean),
        'slot0_norm_std': float(slot0_norm_std),
        'slots16_norm_mean': float(slots16_norm_mean),
        'slots16_norm_std': float(slots16_norm_std),
        'norm_ttest_t': float(tests['norm_ttest']['t']),
        'norm_ttest_p': float(tests['norm_ttest']['p']),
        'norm_cohens_d': float(tests['norm_cohens_d']),
        'slot0_var_mean': float(slot0_var_mean),
        'slot0_var_std': float(slot0_var_std),
        'slots16_var_mean': float(slots16_var_mean),
        'slots16_var_std': float(slots16_var_std),
        'var_ttest_t': float(tests['var_ttest']['t']),
        'var_ttest_p': float(tests['var_ttest']['p']),
        'var_cohens_d': float(tests['var_cohens_d']),
        'slot0_sim_mean': float(slot0_sim_mean),
        'slot0_sim_std': float(slot0_sim_std),
        'slots16_sim_mean': float(slots16_sim_mean),
        'slots16_sim_std': float(slots16_sim_std),
        'sim_ttest_t': float(tests['sim_ttest']['t']),
        'sim_ttest_p': float(tests['sim_ttest']['p']),
        'sim_cohens_d': float(tests['sim_cohens_d']),
        'anova_norm_F': float(tests['anova_norm']['F']),
        'anova_norm_p': float(tests['anova_norm']['p']),
        'anova_var_F': float(tests['anova_var']['F']),
        'anova_var_p': float(tests['anova_var']['p']),
    }

    # Per-slot statistics
    for slot in range(7):
        stats_dict[f'slot{slot}_norm_mean'] = float(np.mean(per_slot_norm[:, slot]))
        stats_dict[f'slot{slot}_norm_std'] = float(np.std(per_slot_norm[:, slot]))
        stats_dict[f'slot{slot}_var_mean'] = float(np.mean(per_slot_var[:, slot]))
        stats_dict[f'slot{slot}_var_std'] = float(np.std(per_slot_var[:, slot]))

    import json
    with open(output_dir / "14_embedding_stats.json", "w") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"\nStatistics saved to: {output_dir / '14_embedding_stats.json'}")

    return stats_dict, tests


if __name__ == "__main__":
    main()
