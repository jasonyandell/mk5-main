#!/usr/bin/env python3
"""
11o: Robust vs Fragile Moves

Question: Which moves are "always good" vs "depends on opponent hands"?
Method: Classify moves by Q-variance and best-move consistency across opponent configs
What It Reveals: Safe play vs speculative play

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from forge.analysis.utils import features
from forge.oracle import schema, tables
from forge.oracle.rng import deal_from_seed

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results")
N_BASE_SEEDS = 201  # Full analysis
np.random.seed(42)


def load_common_states_for_seed(base_seed: int, max_states: int = 10000) -> dict | None:
    """Load states that appear in all 3 opponent configs and get their Q values.

    Returns dict mapping state -> [(Q0, Q1, Q2, Q3, Q4, Q5, Q6), ...]
    where each inner list has 3 entries (one per opp config)
    """
    decl_id = base_seed % 10

    # Collect states and Q values from each config
    config_data = []

    for opp_seed in range(3):
        path = DATA_DIR / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        if not path.exists():
            return None

        try:
            pf = pq.ParquetFile(path)
            states_q = {}

            for batch in pf.iter_batches(batch_size=50000, columns=['state', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']):
                states = batch['state'].to_numpy()
                Q = np.column_stack([
                    batch['q0'].to_numpy(),
                    batch['q1'].to_numpy(),
                    batch['q2'].to_numpy(),
                    batch['q3'].to_numpy(),
                    batch['q4'].to_numpy(),
                    batch['q5'].to_numpy(),
                    batch['q6'].to_numpy()
                ])

                for i, s in enumerate(states):
                    if s not in states_q:
                        states_q[s] = Q[i]

                if len(states_q) > max_states * 3:  # Allow some extra
                    break

            config_data.append(states_q)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    if len(config_data) != 3:
        return None

    # Find common states
    common = set(config_data[0].keys()) & set(config_data[1].keys()) & set(config_data[2].keys())

    if len(common) == 0:
        return None

    # Limit to max_states
    common = list(common)[:max_states]

    # Build result
    result = {}
    for s in common:
        result[s] = [config_data[0][s], config_data[1][s], config_data[2][s]]

    return result


def classify_moves(state_data: dict) -> dict:
    """Classify each legal action at each state as robust or fragile.

    Returns statistics about robust vs fragile moves.
    """
    stats = {
        'n_states': 0,
        'n_actions': 0,
        'n_robust_best': 0,    # Best move in all configs
        'n_fragile_best': 0,   # Best move varies
        'robust_q_variance': [],  # Q variance for robust best moves
        'fragile_q_variance': [], # Q variance for fragile moves
        'robust_by_depth': defaultdict(int),
        'fragile_by_depth': defaultdict(int),
        'slot_stats': defaultdict(lambda: {'robust': 0, 'fragile': 0, 'q_var': []}),
    }

    for state, q_configs in state_data.items():
        # q_configs is list of 3 Q arrays (7 values each)
        Q0, Q1, Q2 = np.array(q_configs[0]), np.array(q_configs[1]), np.array(q_configs[2])

        # Find legal actions (Q != -128)
        legal_mask = (Q0 != -128) & (Q1 != -128) & (Q2 != -128)
        n_legal = legal_mask.sum()

        if n_legal == 0:
            continue

        stats['n_states'] += 1
        stats['n_actions'] += n_legal

        # Get depth
        depth = features.depth(np.array([state]))[0]

        # Get best action in each config
        Q0_masked = np.where(legal_mask, Q0, -999)
        Q1_masked = np.where(legal_mask, Q1, -999)
        Q2_masked = np.where(legal_mask, Q2, -999)

        best0 = np.argmax(Q0_masked)
        best1 = np.argmax(Q1_masked)
        best2 = np.argmax(Q2_masked)

        # Check if best move is consistent
        if best0 == best1 == best2:
            stats['n_robust_best'] += 1
            stats['robust_by_depth'][depth] += 1
            # Q variance of the robust best move
            q_values = [Q0[best0], Q1[best0], Q2[best0]]
            q_var = np.std(q_values)
            stats['robust_q_variance'].append(q_var)
        else:
            stats['n_fragile_best'] += 1
            stats['fragile_by_depth'][depth] += 1
            # Average Q variance across the "best" moves
            q_values = []
            for b, Q in [(best0, Q0), (best1, Q1), (best2, Q2)]:
                q_values.append(Q[b])
            q_var = np.std(q_values)
            stats['fragile_q_variance'].append(q_var)

        # Per-slot analysis
        legal_indices = np.where(legal_mask)[0]
        for slot in legal_indices:
            q_vals = [Q0[slot], Q1[slot], Q2[slot]]
            q_var = np.std(q_vals)

            # Is this slot robust (same rank across configs)?
            rank0 = np.sum(Q0_masked > Q0[slot])  # How many better?
            rank1 = np.sum(Q1_masked > Q1[slot])
            rank2 = np.sum(Q2_masked > Q2[slot])

            is_robust = (rank0 == rank1 == rank2)

            if is_robust:
                stats['slot_stats'][slot]['robust'] += 1
            else:
                stats['slot_stats'][slot]['fragile'] += 1
            stats['slot_stats'][slot]['q_var'].append(q_var)

    return stats


def main():
    print("=" * 60)
    print("ROBUST VS FRAGILE MOVES ANALYSIS")
    print("=" * 60)

    # Get available base seeds
    files = sorted(DATA_DIR.glob("seed_*_opp0_decl_*.parquet"))
    base_seeds = [int(f.stem.split('_')[1]) for f in files]

    print(f"Total base seeds available: {len(base_seeds)}")
    sample_seeds = base_seeds[:N_BASE_SEEDS]
    print(f"Analyzing {len(sample_seeds)} base seeds...")

    # Aggregate stats
    all_stats = {
        'n_states': 0,
        'n_actions': 0,
        'n_robust_best': 0,
        'n_fragile_best': 0,
        'robust_q_variance': [],
        'fragile_q_variance': [],
        'robust_by_depth': defaultdict(int),
        'fragile_by_depth': defaultdict(int),
        'slot_stats': defaultdict(lambda: {'robust': 0, 'fragile': 0, 'q_var': []}),
    }

    per_seed_data = []

    for base_seed in tqdm(sample_seeds, desc="Processing"):
        state_data = load_common_states_for_seed(base_seed, max_states=5000)
        if state_data is None:
            continue

        stats = classify_moves(state_data)

        # Aggregate
        all_stats['n_states'] += stats['n_states']
        all_stats['n_actions'] += stats['n_actions']
        all_stats['n_robust_best'] += stats['n_robust_best']
        all_stats['n_fragile_best'] += stats['n_fragile_best']
        all_stats['robust_q_variance'].extend(stats['robust_q_variance'])
        all_stats['fragile_q_variance'].extend(stats['fragile_q_variance'])

        for d, c in stats['robust_by_depth'].items():
            all_stats['robust_by_depth'][d] += c
        for d, c in stats['fragile_by_depth'].items():
            all_stats['fragile_by_depth'][d] += c

        for slot, s in stats['slot_stats'].items():
            all_stats['slot_stats'][slot]['robust'] += s['robust']
            all_stats['slot_stats'][slot]['fragile'] += s['fragile']
            all_stats['slot_stats'][slot]['q_var'].extend(s['q_var'])

        # Per-seed summary
        total = stats['n_robust_best'] + stats['n_fragile_best']
        if total > 0:
            robust_rate = stats['n_robust_best'] / total
        else:
            robust_rate = 0

        per_seed_data.append({
            'base_seed': base_seed,
            'n_states': stats['n_states'],
            'n_robust': stats['n_robust_best'],
            'n_fragile': stats['n_fragile_best'],
            'robust_rate': robust_rate,
            'mean_robust_q_var': np.mean(stats['robust_q_variance']) if stats['robust_q_variance'] else 0,
            'mean_fragile_q_var': np.mean(stats['fragile_q_variance']) if stats['fragile_q_variance'] else 0,
        })

        # Memory cleanup
        del state_data, stats
        gc.collect()

    print(f"\n✓ Analyzed {len(per_seed_data)} seeds with {all_stats['n_states']} common states")

    if all_stats['n_states'] == 0:
        print("No common states found")
        return

    # Summary statistics
    print("\n" + "=" * 60)
    print("BEST MOVE CLASSIFICATION")
    print("=" * 60)

    total_best = all_stats['n_robust_best'] + all_stats['n_fragile_best']
    robust_pct = all_stats['n_robust_best'] / total_best * 100
    fragile_pct = all_stats['n_fragile_best'] / total_best * 100

    print(f"\n  Total states analyzed: {total_best:,}")
    print(f"  Robust best moves: {all_stats['n_robust_best']:,} ({robust_pct:.1f}%)")
    print(f"  Fragile best moves: {all_stats['n_fragile_best']:,} ({fragile_pct:.1f}%)")

    # Q variance comparison
    print("\n" + "=" * 60)
    print("Q-VALUE VARIANCE COMPARISON")
    print("=" * 60)

    robust_q_mean = np.mean(all_stats['robust_q_variance']) if all_stats['robust_q_variance'] else 0
    fragile_q_mean = np.mean(all_stats['fragile_q_variance']) if all_stats['fragile_q_variance'] else 0
    robust_q_std = np.std(all_stats['robust_q_variance']) if all_stats['robust_q_variance'] else 0
    fragile_q_std = np.std(all_stats['fragile_q_variance']) if all_stats['fragile_q_variance'] else 0

    print(f"\n  Robust moves - Q variance: {robust_q_mean:.2f} ± {robust_q_std:.2f}")
    print(f"  Fragile moves - Q variance: {fragile_q_mean:.2f} ± {fragile_q_std:.2f}")

    if robust_q_mean > 0:
        ratio = fragile_q_mean / robust_q_mean
        print(f"\n  → Fragile moves have {ratio:.1f}x more Q-variance than robust moves")

    # By depth
    print("\n" + "=" * 60)
    print("ROBUSTNESS BY GAME DEPTH")
    print("=" * 60)

    depth_data = []
    for d in sorted(set(all_stats['robust_by_depth'].keys()) | set(all_stats['fragile_by_depth'].keys())):
        r = all_stats['robust_by_depth'].get(d, 0)
        f = all_stats['fragile_by_depth'].get(d, 0)
        total = r + f
        pct = r / total * 100 if total > 0 else 0
        depth_data.append({'depth': d, 'robust': r, 'fragile': f, 'total': total, 'robust_pct': pct})

    depth_df = pd.DataFrame(depth_data)

    print("\n  Depth | Robust | Fragile | Total | Robust %")
    print("  " + "-" * 45)

    # Group by depth ranges
    for dmin, dmax, label in [(0, 4, "Endgame (0-4)"), (5, 8, "Late (5-8)"),
                               (9, 16, "Mid (9-16)"), (17, 28, "Early (17+)")]:
        subset = depth_df[(depth_df['depth'] >= dmin) & (depth_df['depth'] <= dmax)]
        if len(subset) > 0:
            r = subset['robust'].sum()
            f = subset['fragile'].sum()
            t = r + f
            pct = r / t * 100 if t > 0 else 0
            print(f"  {label:<15} | {r:>6,} | {f:>7,} | {t:>5,} | {pct:>6.1f}%")

    # By slot
    print("\n" + "=" * 60)
    print("ROBUSTNESS BY ACTION SLOT")
    print("=" * 60)

    slot_data = []
    for slot in range(7):
        s = all_stats['slot_stats'][slot]
        r = s['robust']
        f = s['fragile']
        total = r + f
        pct = r / total * 100 if total > 0 else 0
        mean_q_var = np.mean(s['q_var']) if s['q_var'] else 0
        slot_data.append({
            'slot': slot,
            'robust': r,
            'fragile': f,
            'total': total,
            'robust_pct': pct,
            'mean_q_var': mean_q_var
        })

    print("\n  Slot | Robust | Fragile | Robust % | Mean Q-Var")
    print("  " + "-" * 50)
    for s in slot_data:
        print(f"    {s['slot']} | {s['robust']:>6,} | {s['fragile']:>7,} | {s['robust_pct']:>6.1f}%  | {s['mean_q_var']:>5.1f}")

    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Per-seed data
    df_seeds = pd.DataFrame(per_seed_data)
    df_seeds.to_csv(tables_dir / "11o_robust_fragile_by_seed.csv", index=False)
    print("✓ Saved per-seed data")

    # Summary
    summary = pd.DataFrame([{
        'n_states': all_stats['n_states'],
        'n_robust_best': all_stats['n_robust_best'],
        'n_fragile_best': all_stats['n_fragile_best'],
        'robust_pct': robust_pct,
        'robust_q_var_mean': robust_q_mean,
        'fragile_q_var_mean': fragile_q_mean,
    }])
    summary.to_csv(tables_dir / "11o_robust_fragile_summary.csv", index=False)
    print("✓ Saved summary")

    # By depth
    depth_df.to_csv(tables_dir / "11o_robust_by_depth.csv", index=False)
    print("✓ Saved depth analysis")

    # By slot
    pd.DataFrame(slot_data).to_csv(tables_dir / "11o_robust_by_slot.csv", index=False)
    print("✓ Saved slot analysis")

    # Visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Robust vs Fragile pie chart
    ax1 = axes[0, 0]
    ax1.pie([all_stats['n_robust_best'], all_stats['n_fragile_best']],
            labels=['Robust', 'Fragile'],
            colors=['green', 'red'],
            autopct='%1.1f%%',
            startangle=90)
    ax1.set_title('Best Move Classification')

    # Top right: Robustness by depth
    ax2 = axes[0, 1]
    if len(depth_df) > 0:
        ax2.bar(depth_df['depth'], depth_df['robust_pct'], color='green', alpha=0.7)
        ax2.set_xlabel('Depth (dominoes remaining)')
        ax2.set_ylabel('Robust Move %')
        ax2.set_title('Robustness by Game Depth')
        ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5)

    # Bottom left: Q-variance distribution
    ax3 = axes[1, 0]
    if all_stats['robust_q_variance']:
        ax3.hist(all_stats['robust_q_variance'], bins=30, alpha=0.5, label='Robust', color='green')
    if all_stats['fragile_q_variance']:
        ax3.hist(all_stats['fragile_q_variance'], bins=30, alpha=0.5, label='Fragile', color='red')
    ax3.set_xlabel('Q-value Standard Deviation')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Q-Variance: Robust vs Fragile Moves')
    ax3.legend()

    # Bottom right: Robustness by slot
    ax4 = axes[1, 1]
    slots = [s['slot'] for s in slot_data]
    robust_pcts = [s['robust_pct'] for s in slot_data]
    ax4.bar(slots, robust_pcts, color='green', alpha=0.7)
    ax4.set_xlabel('Action Slot')
    ax4.set_ylabel('Robust Move %')
    ax4.set_title('Robustness by Action Slot')
    ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()

    fig_path = RESULTS_DIR / "figures" / "11o_robust_vs_fragile.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved figure: {fig_path}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print(f"\n1. BEST MOVE CONSISTENCY:")
    print(f"   Robust (same best move in all configs): {robust_pct:.1f}%")
    print(f"   Fragile (best move varies): {fragile_pct:.1f}%")

    print(f"\n2. Q-VARIANCE BY MOVE TYPE:")
    print(f"   Robust moves: σ(Q) = {robust_q_mean:.2f}")
    print(f"   Fragile moves: σ(Q) = {fragile_q_mean:.2f}")
    if robust_q_mean > 0:
        print(f"   → Fragile moves have {fragile_q_mean/robust_q_mean:.1f}x more Q-uncertainty")

    print(f"\n3. DEPTH PATTERN:")
    endgame = depth_df[(depth_df['depth'] >= 0) & (depth_df['depth'] <= 4)]
    early = depth_df[depth_df['depth'] >= 17]
    if len(endgame) > 0:
        eg_pct = endgame['robust'].sum() / (endgame['robust'].sum() + endgame['fragile'].sum()) * 100
        print(f"   Endgame (depth 0-4): {eg_pct:.1f}% robust")
    if len(early) > 0:
        early_pct = early['robust'].sum() / (early['robust'].sum() + early['fragile'].sum()) * 100
        print(f"   Early game (depth 17+): {early_pct:.1f}% robust")


if __name__ == "__main__":
    main()
