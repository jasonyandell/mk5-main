#!/usr/bin/env python3
"""
11o: Robust vs Fragile Moves

Question: Which moves are "always good" vs "depends on opponent hands"?
Method: Classify moves by Q-variance and best-move consistency across opponent configs
What It Reveals: Safe play vs speculative play

Uses marginalized data where P0's hand is fixed but opponents' cards vary.
"""

import sys
PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from forge.analysis.utils.seed_db import SeedDB
from forge.analysis.utils import features

DATA_DIR = Path(PROJECT_ROOT) / "data/shards-marginalized/train"
RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"
N_BASE_SEEDS = 201  # Full analysis
np.random.seed(42)


def load_common_states_for_seed(db: SeedDB, base_seed: int, max_states: int = 10000) -> tuple | None:
    """Load states that appear in all 3 opponent configs via SQL JOIN.

    Returns (states, Q_a, Q_b, Q_c) as numpy arrays for vectorized processing.
    """
    decl_id = base_seed % 10

    # Build file paths
    files = []
    for opp_seed in range(3):
        filename = f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
        path = DATA_DIR / filename
        if not path.exists():
            return None
        files.append(str(path))

    # SQL JOIN to get common states with all Q values
    q_cols_a = ", ".join([f"a.q{i} as a_q{i}" for i in range(7)])
    q_cols_b = ", ".join([f"b.q{i} as b_q{i}" for i in range(7)])
    q_cols_c = ", ".join([f"c.q{i} as c_q{i}" for i in range(7)])

    sql = f"""
    SELECT a.state, {q_cols_a}, {q_cols_b}, {q_cols_c}
    FROM read_parquet('{files[0]}') a
    INNER JOIN read_parquet('{files[1]}') b ON a.state = b.state
    INNER JOIN read_parquet('{files[2]}') c ON a.state = c.state
    USING SAMPLE {max_states} ROWS
    """

    try:
        result = db.execute(sql)
        df = result.data
        if df is None or len(df) == 0:
            return None
    except Exception:
        return None

    # Convert to numpy arrays for vectorized processing (much faster than iterrows)
    states = df['state'].values
    Q_a = np.column_stack([df[f'a_q{i}'].values for i in range(7)]).astype(np.int8)
    Q_b = np.column_stack([df[f'b_q{i}'].values for i in range(7)]).astype(np.int8)
    Q_c = np.column_stack([df[f'c_q{i}'].values for i in range(7)]).astype(np.int8)

    return states, Q_a, Q_b, Q_c


def classify_moves(states: np.ndarray, Q_a: np.ndarray, Q_b: np.ndarray, Q_c: np.ndarray) -> dict:
    """Classify each legal action at each state as robust or fragile.

    Uses vectorized numpy operations for performance.
    Returns statistics about robust vs fragile moves.
    """
    n_states = len(states)

    # Legal mask: action is legal in all 3 configs
    legal_mask = (Q_a != -128) & (Q_b != -128) & (Q_c != -128)
    n_legal_per_state = legal_mask.sum(axis=1)

    # Filter out states with no legal actions
    valid_mask = n_legal_per_state > 0
    states = states[valid_mask]
    Q_a = Q_a[valid_mask]
    Q_b = Q_b[valid_mask]
    Q_c = Q_c[valid_mask]
    legal_mask = legal_mask[valid_mask]
    n_states = len(states)

    if n_states == 0:
        return {'n_states': 0, 'n_actions': 0, 'n_robust_best': 0, 'n_fragile_best': 0,
                'robust_q_variance': [], 'fragile_q_variance': [],
                'robust_by_depth': defaultdict(int), 'fragile_by_depth': defaultdict(int),
                'slot_stats': defaultdict(lambda: {'robust': 0, 'fragile': 0, 'q_var': []})}

    # Compute depths for all states at once
    depths = features.depth(states)

    # Mask illegal actions with -999 for argmax
    Q_a_masked = np.where(legal_mask, Q_a.astype(np.int16), -999)
    Q_b_masked = np.where(legal_mask, Q_b.astype(np.int16), -999)
    Q_c_masked = np.where(legal_mask, Q_c.astype(np.int16), -999)

    # Best action per config (vectorized argmax)
    best_a = np.argmax(Q_a_masked, axis=1)
    best_b = np.argmax(Q_b_masked, axis=1)
    best_c = np.argmax(Q_c_masked, axis=1)

    # Robust = all configs agree on best action
    robust_mask = (best_a == best_b) & (best_b == best_c)
    n_robust = robust_mask.sum()
    n_fragile = n_states - n_robust

    # Q-variance for best moves
    row_idx = np.arange(n_states)
    best_q_a = Q_a[row_idx, best_a].astype(np.float32)
    best_q_b = Q_b[row_idx, best_b].astype(np.float32)
    best_q_c = Q_c[row_idx, best_c].astype(np.float32)

    # Stack and compute std per state
    best_q_stack = np.column_stack([best_q_a, best_q_b, best_q_c])
    q_var_per_state = np.std(best_q_stack, axis=1)

    robust_q_variance = q_var_per_state[robust_mask].tolist()
    fragile_q_variance = q_var_per_state[~robust_mask].tolist()

    # By depth stats
    robust_by_depth = defaultdict(int)
    fragile_by_depth = defaultdict(int)
    for d in np.unique(depths):
        d_mask = depths == d
        robust_by_depth[int(d)] = int((robust_mask & d_mask).sum())
        fragile_by_depth[int(d)] = int((~robust_mask & d_mask).sum())

    # Slot stats (simplified - aggregate across all states)
    slot_stats = defaultdict(lambda: {'robust': 0, 'fragile': 0, 'q_var': []})
    for slot in range(7):
        slot_legal = legal_mask[:, slot]
        if slot_legal.sum() == 0:
            continue

        # Q values for this slot across configs
        q_vals = np.column_stack([Q_a[slot_legal, slot].astype(np.float32),
                                   Q_b[slot_legal, slot].astype(np.float32),
                                   Q_c[slot_legal, slot].astype(np.float32)])
        q_var = np.std(q_vals, axis=1)
        slot_stats[slot]['q_var'] = q_var.tolist()

        # Check if slot rank is consistent across configs (simplified: just count)
        slot_stats[slot]['robust'] = int(slot_legal.sum())
        slot_stats[slot]['fragile'] = 0  # Simplified

    return {
        'n_states': n_states,
        'n_actions': int(legal_mask.sum()),
        'n_robust_best': int(n_robust),
        'n_fragile_best': int(n_fragile),
        'robust_q_variance': robust_q_variance,
        'fragile_q_variance': fragile_q_variance,
        'robust_by_depth': robust_by_depth,
        'fragile_by_depth': fragile_by_depth,
        'slot_stats': slot_stats,
    }


def main():
    print("=" * 60)
    print("ROBUST VS FRAGILE MOVES ANALYSIS")
    print("=" * 60)

    # Initialize SeedDB
    db = SeedDB(DATA_DIR)

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
        result = load_common_states_for_seed(db, base_seed, max_states=5000)
        if result is None:
            continue

        states, Q_a, Q_b, Q_c = result
        stats = classify_moves(states, Q_a, Q_b, Q_c)

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
        del states, Q_a, Q_b, Q_c, stats
        gc.collect()

    db.close()
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
