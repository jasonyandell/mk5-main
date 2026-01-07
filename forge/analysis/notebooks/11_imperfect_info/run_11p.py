#!/usr/bin/env python3
"""
11p: Path Similarity Analysis (DTW)

Question: How similar are PV trajectories across opponent configs?
Method: Compare V distributions at each depth level across configs
What It Reveals: Path stability given hand

Memory-efficient approach: Instead of full PV tracing (too memory intensive),
we sample V distributions at each depth level and compare the "depth trajectory"
across opponent configurations.
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr

from forge.analysis.utils import features

DATA_DIR = Path("/mnt/d/shards-marginalized/train")
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Full analysis
np.random.seed(42)

# Depth levels to sample
DEPTH_LEVELS = [28, 24, 20, 16, 12, 8, 4, 1]


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute DTW distance, normalized by path length."""
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return float('inf')

    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(seq1[i-1]) - float(seq2[j-1]))
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],
                dtw_matrix[i, j-1],
                dtw_matrix[i-1, j-1]
            )

    return dtw_matrix[n, m] / (n + m)


def load_depth_v_profile(path: Path) -> dict | None:
    """Load V distribution stats at each depth level.

    Returns dict mapping depth -> (mean_V, std_V, n_states)
    """
    try:
        pf = pq.ParquetFile(path)

        depth_stats = {d: [] for d in DEPTH_LEVELS}

        for batch in pf.iter_batches(batch_size=100000, columns=['state', 'V']):
            states = batch['state'].to_numpy()
            V = batch['V'].to_numpy()
            depths = features.depth(states)

            for d in DEPTH_LEVELS:
                mask = depths == d
                if mask.any():
                    depth_stats[d].extend(V[mask].tolist())

        # Compute stats
        profile = {}
        for d in DEPTH_LEVELS:
            vs = depth_stats[d]
            if len(vs) > 0:
                profile[d] = {
                    'mean_v': float(np.mean(vs)),
                    'std_v': float(np.std(vs)),
                    'n_states': len(vs)
                }

        return profile if len(profile) > 2 else None

    except Exception:
        return None


def compare_profiles(p1: dict, p2: dict) -> dict:
    """Compare two depth-V profiles."""
    common_depths = set(p1.keys()) & set(p2.keys())
    if len(common_depths) < 2:
        return None

    # Extract mean V sequences (sorted by depth, descending = game progression)
    sorted_depths = sorted(common_depths, reverse=True)
    v1 = np.array([p1[d]['mean_v'] for d in sorted_depths])
    v2 = np.array([p2[d]['mean_v'] for d in sorted_depths])

    # DTW distance
    dtw_dist = dtw_distance(v1, v2)

    # Pearson correlation
    if np.std(v1) > 0 and np.std(v2) > 0:
        corr, _ = pearsonr(v1, v2)
    else:
        corr = 1.0 if np.allclose(v1, v2) else 0.0

    # Mean absolute difference
    mad = np.mean(np.abs(v1 - v2))

    # Root (depth 28) difference
    root_diff = abs(p1.get(28, {}).get('mean_v', 0) - p2.get(28, {}).get('mean_v', 0))

    # Terminal (depth 1) difference
    term_diff = abs(p1.get(1, {}).get('mean_v', 0) - p2.get(1, {}).get('mean_v', 0))

    return {
        'dtw_distance': dtw_dist,
        'correlation': corr,
        'mean_abs_diff': mad,
        'root_v_diff': root_diff,
        'terminal_v_diff': term_diff,
        'n_depths': len(common_depths),
    }


def analyze_path_similarity(base_seed: int) -> dict | None:
    """Analyze path similarity for one base seed across opponent configs."""
    decl_id = base_seed % 10

    profiles = []
    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None

        profile = load_depth_v_profile(path)
        if profile is None:
            return None
        profiles.append(profile)
        gc.collect()

    if len(profiles) != 3:
        return None

    # Compare all pairs
    comparisons = []
    for i in range(3):
        for j in range(i + 1, 3):
            comp = compare_profiles(profiles[i], profiles[j])
            if comp:
                comparisons.append(comp)

    if not comparisons:
        return None

    # Aggregate metrics
    mean_dtw = np.mean([c['dtw_distance'] for c in comparisons])
    mean_corr = np.mean([c['correlation'] for c in comparisons])
    min_corr = min(c['correlation'] for c in comparisons)
    mean_mad = np.mean([c['mean_abs_diff'] for c in comparisons])
    root_spread = max(c['root_v_diff'] for c in comparisons)
    terminal_spread = max(c['terminal_v_diff'] for c in comparisons)

    # Extract root V values
    root_vs = [p.get(28, {}).get('mean_v', 0) for p in profiles]
    terminal_vs = [p.get(1, {}).get('mean_v', 0) for p in profiles]

    return {
        'base_seed': base_seed,
        'mean_dtw': mean_dtw,
        'mean_corr': mean_corr,
        'min_corr': min_corr,
        'mean_abs_diff': mean_mad,
        'root_v_spread': max(root_vs) - min(root_vs),
        'terminal_v_spread': max(terminal_vs) - min(terminal_vs),
        'root_v_mean': np.mean(root_vs),
        'terminal_v_mean': np.mean(terminal_vs),
    }


def main():
    print("=" * 60)
    print("PATH SIMILARITY ANALYSIS (DTW)")
    print("Depth-based V trajectory comparison")
    print("=" * 60)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds (preliminary)...")

    # Collect data
    all_results = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        result = analyze_path_similarity(base_seed)
        if result:
            all_results.append(result)

    print(f"\n✓ Analyzed {len(all_results)} hands")

    if len(all_results) == 0:
        print("No results")
        return

    df = pd.DataFrame(all_results)

    # Summary statistics
    print("\n" + "=" * 60)
    print("PATH SIMILARITY SUMMARY")
    print("=" * 60)

    print(f"\n  DTW Distance (depth trajectory):")
    print(f"    Mean: {df['mean_dtw'].mean():.2f}")
    print(f"    Median: {df['mean_dtw'].median():.2f}")
    print(f"    Std: {df['mean_dtw'].std():.2f}")

    print(f"\n  Trajectory Correlation:")
    print(f"    Mean: {df['mean_corr'].mean():.3f}")
    print(f"    Median: {df['mean_corr'].median():.3f}")
    print(f"    Min: {df['min_corr'].min():.3f}")

    print(f"\n  Mean Absolute V Difference:")
    print(f"    Mean: {df['mean_abs_diff'].mean():.2f} points")
    print(f"    Median: {df['mean_abs_diff'].median():.2f} points")

    # Stability categories
    print("\n" + "=" * 60)
    print("STABILITY CATEGORIES")
    print("=" * 60)

    high_sim = df[df['min_corr'] > 0.9]
    med_sim = df[(df['min_corr'] > 0.7) & (df['min_corr'] <= 0.9)]
    low_sim = df[df['min_corr'] <= 0.7]

    print(f"\n  High stability (corr > 0.9): {len(high_sim)} ({len(high_sim)/len(df)*100:.1f}%)")
    print(f"  Medium stability (0.7-0.9): {len(med_sim)} ({len(med_sim)/len(df)*100:.1f}%)")
    print(f"  Low stability (corr ≤ 0.7): {len(low_sim)} ({len(low_sim)/len(df)*100:.1f}%)")

    # Root vs terminal analysis
    print("\n" + "=" * 60)
    print("ROOT VS TERMINAL VALUE ANALYSIS")
    print("=" * 60)

    print(f"\n  Root V (depth 28) spread:")
    print(f"    Mean spread: {df['root_v_spread'].mean():.1f} points")
    print(f"    Max spread: {df['root_v_spread'].max():.0f} points")

    print(f"\n  Terminal V (depth 1) spread:")
    print(f"    Mean spread: {df['terminal_v_spread'].mean():.1f} points")
    print(f"    Max spread: {df['terminal_v_spread'].max():.0f} points")

    # Correlations
    print("\n" + "=" * 60)
    print("CORRELATIONS")
    print("=" * 60)

    print(f"\n  DTW vs root V spread: {df['mean_dtw'].corr(df['root_v_spread']):+.3f}")
    print(f"  Corr vs root V spread: {df['mean_corr'].corr(df['root_v_spread']):+.3f}")
    print(f"  DTW vs terminal V spread: {df['mean_dtw'].corr(df['terminal_v_spread']):+.3f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables_dir / "11p_path_similarity_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    # Summary table
    summary = pd.DataFrame([{
        'n_hands': len(df),
        'mean_dtw': df['mean_dtw'].mean(),
        'median_dtw': df['mean_dtw'].median(),
        'mean_corr': df['mean_corr'].mean(),
        'median_corr': df['mean_corr'].median(),
        'high_stability_pct': len(high_sim) / len(df) * 100,
        'med_stability_pct': len(med_sim) / len(df) * 100,
        'low_stability_pct': len(low_sim) / len(df) * 100,
        'mean_root_v_spread': df['root_v_spread'].mean(),
        'mean_terminal_v_spread': df['terminal_v_spread'].mean(),
    }])
    summary.to_csv(tables_dir / "11p_path_similarity_summary.csv", index=False)
    print("✓ Saved summary")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: DTW distribution
    ax1 = axes[0, 0]
    ax1.hist(df['mean_dtw'], bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(df['mean_dtw'].median(), color='red', linestyle='--',
                label=f'Median: {df["mean_dtw"].median():.1f}')
    ax1.set_xlabel('Mean DTW Distance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('V-Trajectory DTW Distance Distribution')
    ax1.legend()

    # Top right: Correlation distribution
    ax2 = axes[0, 1]
    ax2.hist(df['mean_corr'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(0.9, color='red', linestyle='--', label='High stability')
    ax2.axvline(0.7, color='orange', linestyle='--', label='Medium stability')
    ax2.set_xlabel('Mean Trajectory Correlation')
    ax2.set_ylabel('Frequency')
    ax2.set_title('V-Trajectory Correlation Distribution')
    ax2.legend()

    # Bottom left: DTW vs root V spread
    ax3 = axes[1, 0]
    ax3.scatter(df['root_v_spread'], df['mean_dtw'], alpha=0.6, s=40)
    ax3.set_xlabel('Root V Spread (across configs)')
    ax3.set_ylabel('Mean DTW Distance')
    r = df['mean_dtw'].corr(df['root_v_spread'])
    ax3.set_title(f'Path Distance vs Initial Value Spread (r={r:.2f})')

    # Bottom right: Correlation vs root V spread colored by stability
    ax4 = axes[1, 1]
    colors = ['green' if x > 0.9 else 'orange' if x > 0.7 else 'red'
              for x in df['min_corr']]
    ax4.scatter(df['root_v_spread'], df['mean_corr'], c=colors, alpha=0.6, s=40)
    ax4.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    ax4.axhline(0.7, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Root V Spread')
    ax4.set_ylabel('Mean Correlation')
    ax4.set_title('Path Similarity vs Initial Value Spread')

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11p_path_similarity.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS (PRELIMINARY)")
    print("=" * 60)

    high_pct = len(high_sim) / len(df) * 100
    print(f"\n1. PATH STABILITY:")
    print(f"   {high_pct:.1f}% of hands have highly stable trajectories (corr > 0.9)")
    print(f"   Mean trajectory correlation: {df['mean_corr'].mean():.3f}")

    if high_pct > 50:
        print(f"   → V trajectories are STABLE across opponent configurations")
    elif high_pct > 30:
        print(f"   → V trajectories show MODERATE stability")
    else:
        print(f"   → V trajectories DIVERGE significantly based on opponent hands")

    corr = df['mean_dtw'].corr(df['root_v_spread'])
    print(f"\n2. DTW VS VALUE SPREAD:")
    print(f"   Correlation: {corr:+.3f}")
    if corr > 0.3:
        print(f"   → Hands with higher V variance have more divergent paths")
    else:
        print(f"   → Path stability is largely independent of value spread")

    print(f"\n3. PRACTICAL IMPLICATION:")
    if df['mean_corr'].mean() > 0.8:
        print(f"   The V progression through the game is similar across opponent configs")
        print(f"   → Game trajectory is predictable from your hand alone")
    else:
        print(f"   V progression varies significantly based on opponent hands")
        print(f"   → Opponent inference could improve play quality")


if __name__ == "__main__":
    main()
