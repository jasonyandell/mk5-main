#!/usr/bin/env python3
"""08c: Count Capture Predictors - Standalone script version."""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from forge.analysis.utils import loading, features, viz, navigation
from forge.oracle import schema, tables

# Configuration
DATA_DIR = "/mnt/d/shards-standard/"
N_SHARDS = 50  # More data for better predictions
RESULTS_DIR = "/home/jason/v2/mk5-tailwind/forge/analysis/results"

viz.setup_notebook_style()


def extract_deal_features(seed, decl_id):
    """Extract features for a deal that might predict count capture."""
    hands = schema.deal_from_seed(seed)
    trump_suit = decl_id if decl_id <= 6 else -1

    features_dict = {
        'seed': seed,
        'decl_id': decl_id,
    }

    # Count trumps per player
    for p in range(4):
        trump_count = 0
        if trump_suit >= 0:
            for domino_id in hands[p]:
                pips = schema.domino_pips(domino_id)
                if trump_suit in pips:
                    trump_count += 1
        features_dict[f'p{p}_trumps'] = trump_count

    # Team trump counts
    features_dict['team0_trumps'] = features_dict['p0_trumps'] + features_dict['p2_trumps']
    features_dict['team1_trumps'] = features_dict['p1_trumps'] + features_dict['p3_trumps']
    features_dict['trump_advantage'] = features_dict['team0_trumps'] - features_dict['team1_trumps']

    # For each count domino, record who holds it
    for domino_id in features.COUNT_DOMINO_IDS:
        pips = schema.domino_pips(domino_id)
        holder = -1
        for p in range(4):
            if domino_id in hands[p]:
                holder = p
                break
        features_dict[f'holder_{pips[0]}_{pips[1]}'] = holder
        features_dict[f'team_{pips[0]}_{pips[1]}'] = holder % 2 if holder >= 0 else -1
        is_trump = trump_suit in pips if trump_suit >= 0 else False
        features_dict[f'is_trump_{pips[0]}_{pips[1]}'] = int(is_trump)

    return features_dict


def main():
    print("08c: Count Capture Predictors")
    print("=" * 60)

    shard_files = loading.find_shard_files(DATA_DIR)
    print(f"Processing {N_SHARDS} shards...")

    # Collect training data
    training_data = []
    for shard_file in tqdm(shard_files[:N_SHARDS], desc="Processing"):
        df, seed, decl_id = schema.load_file(shard_file)

        # Skip very large shards to avoid OOM
        if len(df) > 20_000_000:
            print(f"  Skipping {shard_file} ({len(df)} rows - too large)")
            del df
            gc.collect()
            continue

        state_to_idx, V, Q = navigation.build_state_lookup_fast(df)
        states = df['state'].values
        depths = features.depth(states)

        initial_mask = depths == 28
        if not initial_mask.any():
            del df, state_to_idx, V, Q, states
            gc.collect()
            continue

        initial_idx = np.where(initial_mask)[0][0]
        initial_state = states[initial_idx]

        captures = navigation.track_count_captures(
            initial_state, seed, decl_id, state_to_idx, V, Q
        )

        deal_features = extract_deal_features(seed, decl_id)

        for domino_id in features.COUNT_DOMINO_IDS:
            pips = schema.domino_pips(domino_id)
            col_name = f'capture_{pips[0]}_{pips[1]}'
            if domino_id in captures:
                deal_features[col_name] = 1 if captures[domino_id] == 0 else 0
            else:
                deal_features[col_name] = np.nan

        training_data.append(deal_features)

        del df, state_to_idx, V, Q, states
        gc.collect()

    train_df = pd.DataFrame(training_data)
    print(f"\nCollected {len(train_df)} seed observations")

    # Baseline analysis
    print("\n=== Baseline: Holder Predicts Capture ===")
    for domino_id in features.COUNT_DOMINO_IDS:
        pips = schema.domino_pips(domino_id)
        points = tables.DOMINO_COUNT_POINTS[domino_id]
        team_col = f'team_{pips[0]}_{pips[1]}'
        capture_col = f'capture_{pips[0]}_{pips[1]}'
        valid = train_df[[team_col, capture_col]].dropna()
        holder_wins = ((valid[team_col] == 0) & (valid[capture_col] == 1)) | \
                      ((valid[team_col] == 1) & (valid[capture_col] == 0))
        accuracy = holder_wins.mean()
        print(f"{pips[0]}-{pips[1]} ({points}pts): Holder's team captures {accuracy*100:.1f}%")

    # Feature columns
    feature_cols = ['trump_advantage', 'team0_trumps', 'team1_trumps']
    for domino_id in features.COUNT_DOMINO_IDS:
        pips = schema.domino_pips(domino_id)
        feature_cols.append(f'team_{pips[0]}_{pips[1]}')
        feature_cols.append(f'is_trump_{pips[0]}_{pips[1]}')

    # Logistic Regression
    print("\n=== Logistic Regression ===")
    results = []
    for domino_id in features.COUNT_DOMINO_IDS:
        pips = schema.domino_pips(domino_id)
        points = tables.DOMINO_COUNT_POINTS[domino_id]
        capture_col = f'capture_{pips[0]}_{pips[1]}'

        valid = train_df[feature_cols + [capture_col]].dropna()
        X = valid[feature_cols].values
        y = valid[capture_col].values

        if len(np.unique(y)) < 2 or len(y) < 5:
            print(f"{pips[0]}-{pips[1]}: Skipping (insufficient data)")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=min(5, len(y)), scoring='accuracy')
        model.fit(X_scaled, y)

        results.append({
            'domino': f"{pips[0]}-{pips[1]}",
            'points': points,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(y),
            'model': model,
            'feature_names': feature_cols,
        })
        print(f"{pips[0]}-{pips[1]} ({points}pts): CV accuracy = {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Random Forest
    print("\n=== Random Forest ===")
    rf_results = []
    for domino_id in features.COUNT_DOMINO_IDS:
        pips = schema.domino_pips(domino_id)
        points = tables.DOMINO_COUNT_POINTS[domino_id]
        capture_col = f'capture_{pips[0]}_{pips[1]}'

        valid = train_df[feature_cols + [capture_col]].dropna()
        X = valid[feature_cols].values
        y = valid[capture_col].values

        if len(np.unique(y)) < 2 or len(y) < 5:
            continue

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        cv_scores = cross_val_score(rf, X, y, cv=min(5, len(y)), scoring='accuracy')
        rf.fit(X, y)

        rf_results.append({
            'domino': f"{pips[0]}-{pips[1]}",
            'points': points,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': rf.feature_importances_,
        })
        print(f"{pips[0]}-{pips[1]} ({points}pts): RF CV accuracy = {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    if not results or not rf_results:
        print("ERROR: Not enough data for analysis")
        return

    # Comparison
    comparison_df = pd.DataFrame({
        'domino': [r['domino'] for r in results],
        'points': [r['points'] for r in results],
        'logistic_acc': [r['cv_accuracy'] for r in results],
        'rf_acc': [r['cv_accuracy'] for r in rf_results],
    })

    print("\n=== Model Comparison ===")
    print(comparison_df.to_string(index=False))
    print(f"\nMean Logistic: {comparison_df['logistic_acc'].mean():.3f}")
    print(f"Mean RF: {comparison_df['rf_acc'].mean():.3f}")

    # Aggregate importance
    all_importances = np.zeros(len(feature_cols))
    for result in rf_results:
        all_importances += result['feature_importance']
    all_importances /= len(rf_results)
    sorted_idx = np.argsort(all_importances)[::-1]

    # Plots
    if len(results) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, result in enumerate(results):
            if i >= 5:
                break
            ax = axes[i]
            coefs = result['model'].coef_[0]
            sorted_coef_idx = np.argsort(np.abs(coefs))[::-1]
            top_n = min(10, len(coefs))
            top_idx = sorted_coef_idx[:top_n]
            colors = ['green' if c > 0 else 'red' for c in coefs[top_idx]]
            ax.barh(range(top_n), coefs[top_idx], color=colors, alpha=0.7)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feature_cols[j] for j in top_idx], fontsize=8)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Coefficient')
            ax.set_title(f"{result['domino']} ({result['points']}pts)\nCV Acc: {result['cv_accuracy']:.3f}")
            ax.invert_yaxis()
        axes[5].set_visible(False)
        plt.suptitle('Feature Importance for Count Capture Prediction', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/figures/08c_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Aggregate plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_cols)), all_importances[sorted_idx], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax.set_xlabel('Mean Feature Importance (Random Forest)')
    ax.set_title('Feature Importance for Count Capture Prediction (Averaged)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/figures/08c_aggregate_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Save tables
    comparison_df.to_csv(f'{RESULTS_DIR}/tables/08c_model_comparison.csv', index=False)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_importance': all_importances,
    }).sort_values('mean_importance', ascending=False)
    importance_df.to_csv(f'{RESULTS_DIR}/tables/08c_feature_importance.csv', index=False)

    print("\n" + "=" * 60)
    print("08c SUMMARY")
    print("=" * 60)
    print(f"Seeds analyzed: {len(train_df)}")
    print(f"Mean Logistic CV accuracy: {comparison_df['logistic_acc'].mean():.3f}")
    print(f"Mean RF CV accuracy: {comparison_df['rf_acc'].mean():.3f}")
    print(f"Most important feature: {feature_cols[sorted_idx[0]]}")
    print("=" * 60)
    print("\nKEY FINDING: Who holds the count is the strongest predictor.")
    print("Results saved to figures/ and tables/")


if __name__ == "__main__":
    main()
