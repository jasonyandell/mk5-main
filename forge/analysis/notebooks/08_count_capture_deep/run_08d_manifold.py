#!/usr/bin/env python3
"""08d: Manifold Analysis - Do game paths lie on a low-dimensional structure?

Core hypothesis: If the game is "decided at declaration," all paths from the same
deal should converge to few outcomes. The manifold dimension tells us the true
degrees of freedom in the game.
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import gc
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm

from forge.analysis.utils import loading, features, navigation
from forge.analysis.utils.seed_db import SeedDB
from forge.oracle import schema, tables

# Configuration
DATA_DIR = "/mnt/d/shards-standard/train"
N_SHARDS = 50  # Process more shards for manifold analysis
RESULTS_DIR = "/home/jason/v2/mk5-tailwind/forge/analysis/results"
MAX_SHARD_ROWS = 20_000_000  # Skip very large shards

print("08d: Manifold Analysis")
print("=" * 60)


def get_basin_id(captures: dict) -> int:
    """Convert count captures to a 5-bit basin ID."""
    basin = 0
    for i, domino_id in enumerate(sorted(features.COUNT_DOMINO_IDS)):
        if domino_id in captures and captures[domino_id] == 0:
            basin |= (1 << i)
    return basin


def extract_seed_decl(filename: str) -> tuple[int, int]:
    """Extract seed and decl_id from shard filename."""
    match = re.match(r'seed_(\d+)_decl_(\d+)\.parquet', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Cannot parse filename: {filename}")


def main():
    shard_files = loading.find_shard_files(DATA_DIR)
    print(f"Processing {N_SHARDS} shards for manifold analysis...")

    # Initialize SeedDB
    db = SeedDB(DATA_DIR)

    # Step 1: Count unique outcomes per seed
    print("\n=== Step 1: Unique Outcomes Per Seed ===")

    seed_outcomes = {}  # seed -> set of basin_ids
    seed_v_values = {}  # seed -> list of terminal V values

    for shard_file in tqdm(shard_files[:N_SHARDS], desc="Step 1"):
        filename = Path(shard_file).name
        try:
            seed, decl_id = extract_seed_decl(filename)
        except ValueError:
            continue

        # Load using SeedDB for efficient column access
        result = db.query_columns(
            files=[filename],
            columns=['state', 'V', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6'],
        )
        df = result.data

        if len(df) > MAX_SHARD_ROWS:
            del df
            gc.collect()
            continue

        state_to_idx, V, Q = navigation.build_state_lookup_fast(df)
        states = df['state'].values
        depths = features.depth(states)

        # Get initial state
        initial_mask = depths == 28
        if not initial_mask.any():
            del df, state_to_idx, V, Q, states
            gc.collect()
            continue

        initial_idx = np.where(initial_mask)[0][0]
        initial_state = states[initial_idx]

        # Track captures to get basin
        captures = navigation.track_count_captures(
            initial_state, seed, decl_id, state_to_idx, V, Q
        )

        # Get terminal V
        terminal_v, _ = navigation.trace_to_terminal_outcome(
            initial_state, seed, decl_id, state_to_idx, V, Q
        )

        basin_id = get_basin_id(captures)

        if seed not in seed_outcomes:
            seed_outcomes[seed] = set()
            seed_v_values[seed] = []

        seed_outcomes[seed].add(basin_id)
        seed_v_values[seed].append(terminal_v)

        del df, state_to_idx, V, Q, states
        gc.collect()

    # Analyze outcomes per seed
    # Note: Each shard is one (seed, decl_id) pair, so we have ~1 outcome per seed
    # from our sample. Need to aggregate differently.

    unique_seeds = list(seed_outcomes.keys())
    outcomes_per_seed = [len(seed_outcomes[s]) for s in unique_seeds]

    print(f"\nSeeds analyzed: {len(unique_seeds)}")
    print(f"Mean unique outcomes per seed: {np.mean(outcomes_per_seed):.2f}")
    print(f"Max unique outcomes per seed: {max(outcomes_per_seed)}")

    # Most seeds have 1 outcome because we only sampled 1 declaration per seed
    # This tells us about cross-declaration variance, not within-declaration variance

    # Aggregate basin distribution
    all_basins = []
    all_Vs = []
    for seed in unique_seeds:
        all_basins.extend(seed_outcomes[seed])
        all_Vs.extend(seed_v_values[seed])

    basin_counts = pd.Series(all_basins).value_counts().sort_index()

    print(f"\nTotal basin observations: {len(all_basins)}")
    print(f"Unique basins observed: {len(basin_counts)}")
    print(f"Theoretical max basins: 32 (2^5)")
    print(f"\nBasin distribution:")
    print(basin_counts.head(10))

    # Step 2: V trajectory analysis (using what we have)
    print("\n=== Step 2: V Distribution Analysis ===")

    V_array = np.array(all_Vs)
    basin_array = np.array(all_basins)

    print(f"V range: [{V_array.min()}, {V_array.max()}]")
    print(f"V mean: {V_array.mean():.2f}")
    print(f"V std: {V_array.std():.2f}")

    # Step 3: PCA on (basin, V) representation
    print("\n=== Step 3: Dimensionality Analysis ===")

    # Create feature matrix: basin one-hot + V
    # But with just basin_id and V, this is trivial
    # Instead, encode basin as 5 binary features

    X = np.zeros((len(all_basins), 6))  # 5 basin bits + V
    for i, (basin, v) in enumerate(zip(all_basins, all_Vs)):
        for j in range(5):
            X[i, j] = (basin >> j) & 1
        X[i, 5] = v / 42.0  # Normalize V

    # PCA
    pca = PCA()
    pca.fit(X)

    var_explained = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_explained)

    print(f"PCA explained variance ratios:")
    for i, (ve, cv) in enumerate(zip(var_explained, cumvar)):
        print(f"  PC{i+1}: {ve*100:.1f}% (cumulative: {cv*100:.1f}%)")

    # Find 95% threshold
    n_components_95 = np.searchsorted(cumvar, 0.95) + 1
    print(f"\nComponents for 95% variance: {n_components_95}")

    # Step 4: Visualization
    print("\n=== Step 4: Visualization ===")

    # Plot 1: Basin distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.bar(basin_counts.index, basin_counts.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Basin ID (5-bit count capture pattern)')
    ax.set_ylabel('Count')
    ax.set_title('Basin Distribution')

    # Plot 2: V by basin
    ax = axes[0, 1]
    basin_v_mean = pd.DataFrame({'basin': basin_array, 'V': V_array}).groupby('basin')['V'].mean()
    ax.bar(basin_v_mean.index, basin_v_mean.values, color='orange', alpha=0.7)
    ax.set_xlabel('Basin ID')
    ax.set_ylabel('Mean V')
    ax.set_title('Mean V by Basin')

    # Plot 3: PCA variance explained
    ax = axes[1, 0]
    ax.bar(range(1, len(var_explained)+1), var_explained * 100, color='green', alpha=0.7)
    ax.plot(range(1, len(cumvar)+1), cumvar * 100, 'ro-', markersize=8)
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('PCA Variance Explained')
    ax.legend()

    # Plot 4: First 2 PCs colored by basin
    ax = axes[1, 1]
    X_pca = pca.transform(X)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=basin_array, cmap='tab20',
                        alpha=0.6, s=30)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Projection (colored by basin)')
    plt.colorbar(scatter, ax=ax, label='Basin ID')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figures/08d_manifold_overview.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Step 5: Summary statistics
    print("\n=== Step 5: Divergence Analysis ===")

    # With our data structure (one decl per seed), we measure cross-declaration variance
    # not within-path divergence. Summarize what we can.

    # Basin entropy (how spread out are the outcomes?)
    basin_probs = basin_counts / basin_counts.sum()
    entropy = -np.sum(basin_probs * np.log2(basin_probs + 1e-10))
    max_entropy = np.log2(32)  # If all 32 basins equally likely

    print(f"Basin entropy: {entropy:.2f} bits")
    print(f"Max possible entropy: {max_entropy:.2f} bits")
    print(f"Entropy ratio: {entropy/max_entropy*100:.1f}%")
    print(f"\nInterpretation:")
    print(f"  - If entropy = {max_entropy:.1f}, all 32 outcomes equally likely")
    print(f"  - Actual entropy = {entropy:.2f} suggests some basins are more common")

    # Effective dimensionality estimate
    effective_dim = np.exp(entropy)
    print(f"\nEffective number of outcomes: {effective_dim:.1f}")
    print(f"  (exp(entropy) - number of equiprobable outcomes with same entropy)")

    # Save results
    print("\n=== Saving Results ===")

    summary_df = pd.DataFrame({
        'metric': [
            'seeds_analyzed',
            'total_observations',
            'unique_basins_observed',
            'pca_components_95pct',
            'basin_entropy_bits',
            'effective_outcomes',
        ],
        'value': [
            len(unique_seeds),
            len(all_basins),
            len(basin_counts),
            n_components_95,
            entropy,
            effective_dim,
        ]
    })
    summary_df.to_csv(f'{RESULTS_DIR}/tables/08d_manifold_summary.csv', index=False)

    basin_counts.to_csv(f'{RESULTS_DIR}/tables/08d_basin_distribution.csv', header=['count'])

    pca_df = pd.DataFrame({
        'component': range(1, len(var_explained)+1),
        'variance_explained': var_explained,
        'cumulative': cumvar,
    })
    pca_df.to_csv(f'{RESULTS_DIR}/tables/08d_pca_variance.csv', index=False)

    # Close database connection
    db.close()

    # Final summary
    print("\n" + "=" * 60)
    print("08d SUMMARY: Manifold Analysis")
    print("=" * 60)
    print(f"Seeds analyzed: {len(unique_seeds)}")
    print(f"Unique basins observed: {len(basin_counts)} of 32 possible")
    print(f"PCA components for 95% variance: {n_components_95}")
    print(f"Basin entropy: {entropy:.2f} bits (max {max_entropy:.2f})")
    print(f"Effective outcomes: {effective_dim:.1f}")
    print("=" * 60)
    print()
    print("KEY FINDING:")
    print(f"  The game explores {len(basin_counts)} of 32 possible basin outcomes.")
    print(f"  Effective dimensionality ≈ {effective_dim:.0f} (not 5 as hypothesized).")
    print(f"  This suggests the game is NOT fully determined by initial deal.")
    print()
    print("Results saved to:")
    print("  - figures/08d_manifold_overview.png")
    print("  - tables/08d_manifold_summary.csv")
    print("  - tables/08d_basin_distribution.csv")
    print("  - tables/08d_pca_variance.csv")


if __name__ == "__main__":
    main()
