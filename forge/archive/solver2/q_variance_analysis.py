#!/usr/bin/env python3
"""
Q-Variance Analysis: Find positions where opponent distribution matters.

For each test position:
1. Sample 20 opponent hand configurations
2. Run Q-regression model on each
3. Compute Q-variance per move
4. Correlate with hand strength metrics

Goal: Are there positions where PIMC could help?
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.solver2.declarations import N_DECLS, NOTRUMP
from scripts.solver2.rng import deal_from_seed
from scripts.solver2.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    is_in_called_suit,
    trick_rank,
)
from scripts.solver2.train_q_regression import QRegressionTransformer

LOG_START_TIME = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - LOG_START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


# Trump rank table (same as training)
def get_trump_rank(domino_id: int, decl_id: int) -> int:
    if decl_id == NOTRUMP:
        return 7
    if not is_in_called_suit(domino_id, decl_id):
        return 7
    trumps = []
    for d in range(28):
        if is_in_called_suit(d, decl_id):
            tau = trick_rank(d, 7, decl_id)
            trumps.append((d, tau))
    trumps.sort(key=lambda x: -x[1])
    for rank, (d, _) in enumerate(trumps):
        if d == domino_id:
            return rank
    return 7


TRUMP_RANK_TABLE = {}
for _decl in range(N_DECLS):
    for _dom in range(28):
        TRUMP_RANK_TABLE[(_dom, _decl)] = get_trump_rank(_dom, _decl)

# Token types
TOKEN_TYPE_CONTEXT = 0
TOKEN_TYPE_PLAYER0 = 1
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}


def get_initial_position_mask(states: np.ndarray) -> np.ndarray:
    """Vectorized check for initial positions (all remaining=0x7F, trick_len=0, no play)."""
    remaining_ok = np.ones(len(states), dtype=bool)
    for p in range(4):
        remaining = (states >> (p * 7)) & 0x7F
        remaining_ok &= (remaining == 0x7F)
    trick_len = (states >> 30) & 0x3
    trick_ok = (trick_len == 0)
    p0 = (states >> 32) & 0x7
    p0_ok = (p0 == 7)
    return remaining_ok & trick_ok & p0_ok


def sample_opponent_hands(current_player: int, my_hand: list[int], rng) -> list[list[int]]:
    """Generate one sampled configuration of opponent hands."""
    all_dominoes = set(range(28))
    my_set = set(my_hand)
    opponent_pool = list(all_dominoes - my_set)
    rng.shuffle(opponent_pool)

    hands = [None] * 4
    hands[current_player] = my_hand
    opp_idx = 0
    for p in range(4):
        if p == current_player:
            continue
        hands[p] = sorted(opponent_pool[opp_idx:opp_idx + 7])
        opp_idx += 7
    return hands


def tokenize_for_pimc(
    decl_id: int,
    leader: int,
    current_player: int,
    hands: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize an initial position for PIMC inference."""
    MAX_TOKENS = 32
    N_FEATURES = 12

    tokens = np.zeros((MAX_TOKENS, N_FEATURES), dtype=np.int64)
    mask = np.zeros(MAX_TOKENS, dtype=np.float32)

    normalized_leader = (leader - current_player + 4) % 4

    # Token 0: Context
    tokens[0, 9] = TOKEN_TYPE_CONTEXT
    tokens[0, 10] = decl_id
    tokens[0, 11] = normalized_leader
    mask[0] = 1.0

    # Tokens 1-28: Hand positions
    for p in range(4):
        normalized_player = (p - current_player + 4) % 4
        for local_idx in range(7):
            token_idx = 1 + p * 7 + local_idx
            global_id = hands[p][local_idx]

            tokens[token_idx, 0] = DOMINO_HIGH[global_id]
            tokens[token_idx, 1] = DOMINO_LOW[global_id]
            tokens[token_idx, 2] = 1 if DOMINO_IS_DOUBLE[global_id] else 0
            tokens[token_idx, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]]
            tokens[token_idx, 4] = TRUMP_RANK_TABLE[(global_id, decl_id)]
            tokens[token_idx, 5] = normalized_player
            tokens[token_idx, 6] = 1 if normalized_player == 0 else 0
            tokens[token_idx, 7] = 1 if normalized_player == 2 else 0
            tokens[token_idx, 8] = 1  # is_remaining (all 7 for initial position)
            tokens[token_idx, 9] = TOKEN_TYPE_PLAYER0 + p
            tokens[token_idx, 10] = decl_id
            tokens[token_idx, 11] = normalized_leader
            mask[token_idx] = 1.0

    return tokens, mask


def compute_hand_metrics(hand: list[int], decl_id: int) -> dict:
    """Compute hand strength metrics."""
    trump_count = 0
    has_boss = False
    count_points = 0
    double_count = 0

    for d in hand:
        # Trump count
        if TRUMP_RANK_TABLE[(d, decl_id)] < 7:
            trump_count += 1
            if TRUMP_RANK_TABLE[(d, decl_id)] == 0:
                has_boss = True

        # Count points
        count_points += DOMINO_COUNT_POINTS[d]

        # Doubles
        if DOMINO_IS_DOUBLE[d]:
            double_count += 1

    return {
        'trump_count': trump_count,
        'has_boss': has_boss,
        'count_points': count_points,
        'double_count': double_count,
    }


def load_initial_positions(data_dir: Path, max_positions: int, seed_range: tuple[int, int]) -> list[dict]:
    """Load initial positions from parquet files (memory efficient)."""
    parquet_files = sorted(data_dir.glob("seed_*.parquet"))

    relevant_files = []
    for f in parquet_files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            try:
                seed = int(parts[1])
                if seed_range[0] <= seed <= seed_range[1]:
                    relevant_files.append(f)
            except ValueError:
                pass

    positions = []

    for f in relevant_files:
        if len(positions) >= max_positions:
            break

        try:
            pf = pq.ParquetFile(f)
            meta = pf.schema_arrow.metadata or {}
            seed = int(meta.get(b"seed", b"0").decode())
            decl_id = int(meta.get(b"decl_id", b"0").decode())

            # Read only state column first to find initial positions
            states = pq.read_table(f, columns=['state']).column('state').to_numpy().astype(np.int64)

            # Find initial positions
            initial_mask = get_initial_position_mask(states)
            initial_indices = np.where(initial_mask)[0]

            if len(initial_indices) == 0:
                del states
                continue

            # Only read Q-values for the positions we need
            needed_indices = initial_indices[:max_positions - len(positions)]

            # Read full row data only for needed indices
            table = pq.read_table(f)
            hands = deal_from_seed(seed)

            for idx in needed_indices:
                state = states[idx]
                leader = (state >> 28) & 0x3
                current_player = leader

                q_values = np.array([
                    table.column(f'mv{i}')[int(idx)].as_py()
                    for i in range(7)
                ], dtype=np.float32)

                positions.append({
                    'state': state,
                    'seed': seed,
                    'decl_id': decl_id,
                    'hands': hands,
                    'current_player': current_player,
                    'q_values': q_values,
                })

                if len(positions) >= max_positions:
                    break

            del states, table

        except Exception as e:
            log(f"  Error loading {f}: {e}")
            continue

    return positions


def main():
    parser = argparse.ArgumentParser(description="Q-Variance Analysis")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--model-path", type=str, default="data/solver2/q_model.pt")
    parser.add_argument("--n-positions", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Load model
    log(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_args = checkpoint['args']

    model = QRegressionTransformer(
        embed_dim=model_args['embed_dim'],
        n_heads=model_args['n_heads'],
        n_layers=model_args['n_layers'],
        ff_dim=model_args['ff_dim'],
        dropout=0.0,  # No dropout for inference
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    log(f"Model loaded (epoch {checkpoint['epoch']}, test_acc={checkpoint['test_acc']:.2%})")

    # Load positions
    log(f"Loading {args.n_positions} initial positions...")
    positions = load_initial_positions(Path(args.data_dir), args.n_positions, (90, 99))
    log(f"Loaded {len(positions)} positions")

    if len(positions) == 0:
        log("No positions found!")
        return

    # SANITY CHECK: Model accuracy on TRUE hands (no sampling)
    log("\n=== SANITY CHECK: Model on TRUE hands ===")
    true_hand_correct = 0
    true_hand_regret = 0.0
    for pos in positions:
        decl_id = pos['decl_id']
        current_player = pos['current_player']
        true_hands = pos['hands']
        true_q = pos['q_values']
        legal_mask = (true_q != -128)

        # Tokenize with TRUE hands
        tokens, mask = tokenize_for_pimc(decl_id, current_player, current_player, true_hands)
        tokens_t = torch.tensor(tokens[np.newaxis], dtype=torch.long, device=device)
        masks_t = torch.tensor(mask[np.newaxis], dtype=torch.float32, device=device)
        players_t = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            q_pred = model(tokens_t, masks_t, players_t)[0].cpu().numpy()

        pred_best = np.argmax(np.where(legal_mask, q_pred, -np.inf))

        # Handle ties: check if predicted move has optimal Q-value
        legal_q = np.where(legal_mask, true_q, -np.inf)
        max_q = legal_q.max()
        optimal_indices = np.where(legal_q == max_q)[0]

        if pred_best in optimal_indices:
            true_hand_correct += 1

        # Regret = optimal_q - chosen_q
        regret = max_q - true_q[pred_best]
        true_hand_regret += regret

    log(f"Model accuracy on TRUE hands: {true_hand_correct}/{len(positions)} = {true_hand_correct/len(positions):.1%}")
    log(f"Mean regret on TRUE hands: {true_hand_regret/len(positions):.2f}")
    log("(Handles ties correctly - any optimal move counts as correct)")

    # Analyze each position
    results = []

    log(f"\nAnalyzing Q-variance across {args.n_samples} opponent samples per position...")

    for i, pos in enumerate(positions):
        decl_id = pos['decl_id']
        current_player = pos['current_player']
        true_hands = pos['hands']
        my_hand = true_hands[current_player]
        true_q = pos['q_values']

        # Legal mask
        legal_mask = (true_q != -128)

        # Hand metrics
        metrics = compute_hand_metrics(my_hand, decl_id)

        # Sample opponent hands and get Q predictions
        all_tokens = []
        all_masks = []

        for _ in range(args.n_samples):
            sampled_hands = sample_opponent_hands(current_player, my_hand, rng)
            tokens, mask = tokenize_for_pimc(decl_id, current_player, current_player, sampled_hands)
            all_tokens.append(tokens)
            all_masks.append(mask)

        # Batch inference
        tokens_batch = torch.tensor(np.stack(all_tokens), dtype=torch.long, device=device)
        masks_batch = torch.tensor(np.stack(all_masks), dtype=torch.float32, device=device)
        players_batch = torch.zeros(args.n_samples, dtype=torch.long, device=device)

        with torch.no_grad():
            q_pred = model(tokens_batch, masks_batch, players_batch)  # (n_samples, 7)

        q_pred = q_pred.cpu().numpy()

        # Compute variance per move (only for legal moves)
        q_mean = q_pred.mean(axis=0)
        q_std = q_pred.std(axis=0)
        q_range = q_pred.max(axis=0) - q_pred.min(axis=0)

        # Mask illegal moves
        q_mean[~legal_mask] = np.nan
        q_std[~legal_mask] = np.nan
        q_range[~legal_mask] = np.nan

        # Average variance across legal moves
        avg_std = np.nanmean(q_std)
        avg_range = np.nanmean(q_range)
        max_std = np.nanmax(q_std)

        # Does sampling change the best move?
        best_per_sample = np.argmax(np.where(legal_mask, q_pred, -np.inf), axis=1)
        n_unique_best = len(np.unique(best_per_sample))

        # Best move by average Q
        avg_best = np.nanargmax(np.where(legal_mask, q_mean, -np.inf))

        # True optimal (handle ties)
        legal_true_q = np.where(legal_mask, true_q, -np.inf)
        max_true_q = legal_true_q.max()
        optimal_indices = np.where(legal_true_q == max_true_q)[0]
        true_best = optimal_indices[0]  # First optimal for display

        # Regret for PIMC choice
        pimc_regret = max_true_q - true_q[avg_best]

        results.append({
            'position': i,
            'decl_id': decl_id,
            'trump_count': metrics['trump_count'],
            'has_boss': metrics['has_boss'],
            'count_points': metrics['count_points'],
            'double_count': metrics['double_count'],
            'n_legal': legal_mask.sum(),
            'avg_q_std': avg_std,
            'avg_q_range': avg_range,
            'max_q_std': max_std,
            'n_unique_best': n_unique_best,
            'avg_best': avg_best,
            'true_best': true_best,
            'avg_correct': avg_best in optimal_indices,  # Handle ties
            'pimc_regret': pimc_regret,
        })

        if (i + 1) % 20 == 0:
            log(f"  Processed {i+1}/{len(positions)}")

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)

    log(f"\nPositions analyzed: {len(df)}")
    log(f"PIMC Accuracy (avg Q picks true optimal): {df['avg_correct'].mean():.1%}")
    log(f"PIMC Mean Regret: {df['pimc_regret'].mean():.2f}")

    log(f"\n--- Q-Variance Statistics ---")
    log(f"Mean Q std across positions: {df['avg_q_std'].mean():.2f}")
    log(f"Mean Q range across positions: {df['avg_q_range'].mean():.2f}")
    log(f"Max Q std in any position: {df['max_q_std'].max():.2f}")

    log(f"\n--- Best Move Stability ---")
    log(f"Positions with only 1 unique best (stable): {(df['n_unique_best']==1).mean():.1%}")
    log(f"Positions with 2+ unique best: {(df['n_unique_best']>=2).mean():.1%}")
    log(f"Positions with 3+ unique best: {(df['n_unique_best']>=3).mean():.1%}")

    log(f"\n--- Correlation with Hand Metrics ---")
    for metric in ['trump_count', 'has_boss', 'count_points', 'double_count', 'n_legal']:
        corr = df['avg_q_std'].corr(df[metric])
        log(f"  {metric:15s} vs Q-std: r={corr:+.3f}")

    # High variance positions
    high_var = df[df['avg_q_std'] > df['avg_q_std'].median()]
    low_var = df[df['avg_q_std'] <= df['avg_q_std'].median()]

    log(f"\n--- High vs Low Variance Positions ---")
    log(f"High variance (above median):")
    log(f"  Accuracy: {high_var['avg_correct'].mean():.1%}")
    log(f"  Mean trump count: {high_var['trump_count'].mean():.2f}")
    log(f"  Mean n_legal: {high_var['n_legal'].mean():.1f}")
    log(f"  Has boss: {high_var['has_boss'].mean():.1%}")

    log(f"Low variance (below median):")
    log(f"  Accuracy: {low_var['avg_correct'].mean():.1%}")
    log(f"  Mean trump count: {low_var['trump_count'].mean():.2f}")
    log(f"  Mean n_legal: {low_var['n_legal'].mean():.1f}")
    log(f"  Has boss: {low_var['has_boss'].mean():.1%}")

    # Positions where best move changes
    changes = df[df['n_unique_best'] >= 2]
    stable = df[df['n_unique_best'] == 1]

    log(f"\n--- Positions Where Best Move Changes (n={len(changes)}) ---")
    if len(changes) > 0:
        log(f"  Accuracy: {changes['avg_correct'].mean():.1%}")
        log(f"  Mean Q-std: {changes['avg_q_std'].mean():.2f}")
    log(f"Stable positions (n={len(stable)}):")
    if len(stable) > 0:
        log(f"  Accuracy: {stable['avg_correct'].mean():.1%}")
        log(f"  Mean Q-std: {stable['avg_q_std'].mean():.2f}")

    # Show a few high-variance examples
    log(f"\n--- Top 5 Highest Variance Positions ---")
    top5 = df.nlargest(5, 'avg_q_std')
    for _, row in top5.iterrows():
        log(f"  Pos {row['position']}: std={row['avg_q_std']:.2f}, range={row['avg_q_range']:.2f}, "
            f"trumps={row['trump_count']}, n_unique_best={row['n_unique_best']}, correct={row['avg_correct']}")


if __name__ == "__main__":
    main()
