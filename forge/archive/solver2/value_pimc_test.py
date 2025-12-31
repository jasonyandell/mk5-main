#!/usr/bin/env python3
"""
Value PIMC Test with Classification Model.

Use the existing sw=0.7 classification model (85% accuracy):
1. Sample opponent hands
2. Get raw logits for each sample
3. Average logits per move
4. Pick move with highest average logit
5. Compare to single-sample baseline (TRUE hands)
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
from scripts.solver2.train_transformer import DominoTransformer

LOG_START_TIME = time.time()


def log(msg: str) -> None:
    elapsed = time.time() - LOG_START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


# Trump rank table
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

TOKEN_TYPE_CONTEXT = 0
TOKEN_TYPE_PLAYER0 = 1
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}


def get_initial_position_mask(states: np.ndarray) -> np.ndarray:
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


def tokenize_position(
    decl_id: int,
    leader: int,
    current_player: int,
    hands: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize an initial position."""
    MAX_TOKENS = 32
    N_FEATURES = 12

    tokens = np.zeros((MAX_TOKENS, N_FEATURES), dtype=np.int64)
    mask = np.zeros(MAX_TOKENS, dtype=np.float32)

    normalized_leader = (leader - current_player + 4) % 4

    tokens[0, 9] = TOKEN_TYPE_CONTEXT
    tokens[0, 10] = decl_id
    tokens[0, 11] = normalized_leader
    mask[0] = 1.0

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
            tokens[token_idx, 8] = 1  # is_remaining
            tokens[token_idx, 9] = TOKEN_TYPE_PLAYER0 + p
            tokens[token_idx, 10] = decl_id
            tokens[token_idx, 11] = normalized_leader
            mask[token_idx] = 1.0

    return tokens, mask


def load_initial_positions(data_dir: Path, max_positions: int, seed_range: tuple[int, int]) -> list[dict]:
    """Load initial positions from parquet files."""
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

            states = pq.read_table(f, columns=['state']).column('state').to_numpy().astype(np.int64)
            initial_mask = get_initial_position_mask(states)
            initial_indices = np.where(initial_mask)[0]

            if len(initial_indices) == 0:
                del states
                continue

            needed_indices = initial_indices[:max_positions - len(positions)]
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
    parser = argparse.ArgumentParser(description="Value PIMC with Classification Model")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--model-path", type=str, default="data/solver2/transformer_best.pt")
    parser.add_argument("--n-positions", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Load classification model
    log(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_args = checkpoint.get('args', {})

    model = DominoTransformer(
        embed_dim=model_args.get('embed_dim', 64),
        n_heads=model_args.get('n_heads', 4),
        n_layers=model_args.get('n_layers', 2),
        ff_dim=model_args.get('ff_dim', 128),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    log(f"Model loaded (epoch {checkpoint.get('epoch', '?')}, test_acc={checkpoint.get('test_acc', 0):.2%})")

    # Load positions
    log(f"Loading {args.n_positions} initial positions...")
    positions = load_initial_positions(Path(args.data_dir), args.n_positions, (90, 99))
    log(f"Loaded {len(positions)} positions")

    if len(positions) == 0:
        log("No positions found!")
        return

    # Results storage
    baseline_correct = 0
    baseline_regret = 0.0
    pimc_correct = 0
    pimc_regret = 0.0

    log(f"\n=== Testing {len(positions)} positions with {args.n_samples} samples each ===\n")

    for i, pos in enumerate(positions):
        decl_id = pos['decl_id']
        current_player = pos['current_player']
        true_hands = pos['hands']
        my_hand = true_hands[current_player]
        true_q = pos['q_values']

        legal_mask = (true_q != -128)
        legal_q = np.where(legal_mask, true_q, -np.inf)
        max_q = legal_q.max()
        optimal_indices = np.where(legal_q == max_q)[0]

        # === BASELINE: Single sample with TRUE hands ===
        tokens_true, mask_true = tokenize_position(decl_id, current_player, current_player, true_hands)
        t = torch.tensor(tokens_true[np.newaxis], dtype=torch.long, device=device)
        m = torch.tensor(mask_true[np.newaxis], dtype=torch.float32, device=device)
        p = torch.tensor([0], dtype=torch.long, device=device)

        with torch.no_grad():
            logits_true = model(t, m, p)[0].cpu().numpy()

        # Mask illegal moves
        logits_true_masked = np.where(legal_mask, logits_true, -np.inf)
        baseline_choice = np.argmax(logits_true_masked)

        if baseline_choice in optimal_indices:
            baseline_correct += 1
        baseline_regret += max_q - true_q[baseline_choice]

        # === VALUE PIMC: Average logits across samples ===
        all_tokens = []
        all_masks = []

        for _ in range(args.n_samples):
            sampled_hands = sample_opponent_hands(current_player, my_hand, rng)
            tokens, mask = tokenize_position(decl_id, current_player, current_player, sampled_hands)
            all_tokens.append(tokens)
            all_masks.append(mask)

        tokens_batch = torch.tensor(np.stack(all_tokens), dtype=torch.long, device=device)
        masks_batch = torch.tensor(np.stack(all_masks), dtype=torch.float32, device=device)
        players_batch = torch.zeros(args.n_samples, dtype=torch.long, device=device)

        with torch.no_grad():
            logits_samples = model(tokens_batch, masks_batch, players_batch).cpu().numpy()

        # Average logits across samples
        avg_logits = logits_samples.mean(axis=0)
        avg_logits_masked = np.where(legal_mask, avg_logits, -np.inf)
        pimc_choice = np.argmax(avg_logits_masked)

        if pimc_choice in optimal_indices:
            pimc_correct += 1
        pimc_regret += max_q - true_q[pimc_choice]

        if (i + 1) % 10 == 0:
            log(f"  [{i+1}/{len(positions)}] Baseline: {baseline_correct}/{i+1} ({baseline_correct/(i+1):.1%}), "
                f"PIMC: {pimc_correct}/{i+1} ({pimc_correct/(i+1):.1%})")

    # Track when PIMC differs from baseline
    differs_count = 0
    differs_better = 0
    differs_worse = 0

    # Re-run to count differences
    rng2 = np.random.default_rng(args.seed)  # Reset RNG
    for i, pos in enumerate(positions):
        decl_id = pos['decl_id']
        current_player = pos['current_player']
        true_hands = pos['hands']
        my_hand = true_hands[current_player]
        true_q = pos['q_values']
        legal_mask = (true_q != -128)

        # Baseline
        tokens_true, mask_true = tokenize_position(decl_id, current_player, current_player, true_hands)
        t = torch.tensor(tokens_true[np.newaxis], dtype=torch.long, device=device)
        m = torch.tensor(mask_true[np.newaxis], dtype=torch.float32, device=device)
        p = torch.tensor([0], dtype=torch.long, device=device)
        with torch.no_grad():
            logits_true = model(t, m, p)[0].cpu().numpy()
        logits_true_masked = np.where(legal_mask, logits_true, -np.inf)
        baseline_choice = np.argmax(logits_true_masked)

        # PIMC
        all_tokens = []
        all_masks = []
        for _ in range(args.n_samples):
            sampled_hands = sample_opponent_hands(current_player, my_hand, rng2)
            tokens, mask = tokenize_position(decl_id, current_player, current_player, sampled_hands)
            all_tokens.append(tokens)
            all_masks.append(mask)
        tokens_batch = torch.tensor(np.stack(all_tokens), dtype=torch.long, device=device)
        masks_batch = torch.tensor(np.stack(all_masks), dtype=torch.float32, device=device)
        players_batch = torch.zeros(args.n_samples, dtype=torch.long, device=device)
        with torch.no_grad():
            logits_samples = model(tokens_batch, masks_batch, players_batch).cpu().numpy()
        avg_logits = logits_samples.mean(axis=0)
        avg_logits_masked = np.where(legal_mask, avg_logits, -np.inf)
        pimc_choice = np.argmax(avg_logits_masked)

        if pimc_choice != baseline_choice:
            differs_count += 1
            baseline_q = true_q[baseline_choice]
            pimc_q = true_q[pimc_choice]
            if pimc_q > baseline_q:
                differs_better += 1
            elif pimc_q < baseline_q:
                differs_worse += 1

    # Final results
    n = len(positions)
    log("\n" + "="*60)
    log("RESULTS")
    log("="*60)

    log(f"\nPIMC differs from baseline: {differs_count}/{n} ({differs_count/n:.1%})")
    if differs_count > 0:
        log(f"  When different: {differs_better} better, {differs_worse} worse, {differs_count - differs_better - differs_worse} tied")

    log(f"\nPositions: {n}")
    log(f"Samples per position: {args.n_samples}")

    log(f"\n{'Method':<20} {'Accuracy':<12} {'Mean Regret':<12}")
    log("-" * 44)
    log(f"{'Baseline (TRUE)':<20} {baseline_correct/n:.1%}{'':>5} {baseline_regret/n:.2f}")
    log(f"{'Value PIMC':<20} {pimc_correct/n:.1%}{'':>5} {pimc_regret/n:.2f}")

    # Improvement
    acc_delta = (pimc_correct - baseline_correct) / n * 100
    regret_delta = baseline_regret/n - pimc_regret/n
    log(f"\nDelta: {acc_delta:+.1f}% accuracy, {regret_delta:+.2f} regret")

    if pimc_correct > baseline_correct:
        log("\n✓ Value PIMC improves over baseline!")
    elif pimc_correct == baseline_correct:
        log("\n~ Value PIMC matches baseline")
    else:
        log("\n✗ Value PIMC worse than baseline")


if __name__ == "__main__":
    main()
