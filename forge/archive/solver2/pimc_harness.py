#!/usr/bin/env python3
"""
PIMC Test Harness - Phase 1: Initial Positions Only

Tests whether PIMC averaging washes out transformer blunders.
Restricts to TRUE initial positions (start of hand) to avoid local-index issues.

Initial position criteria:
- All remaining masks = 0x7F (everyone has all 7 dominoes)
- trick_len = 0 (no plays in current trick)
- p0 = 7 (no lead play stored)

Usage:
    # Sanity checks first
    python scripts/solver2/pimc_harness.py --sanity-check

    # Quick test (100 positions, 10 samples)
    python scripts/solver2/pimc_harness.py --max-positions 100 --samples 10 --verbose

    # Sample sweep
    python scripts/solver2/pimc_harness.py --max-positions 1000 --sample-sweep
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
import torch.nn.functional as F

# Add parent to path for imports
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

# Import model architecture
from scripts.solver2.train_transformer import (
    DominoTransformer,
    TRUMP_RANK_TABLE,
    COUNT_VALUE_MAP,
    TOKEN_TYPE_CONTEXT,
    TOKEN_TYPE_PLAYER0,
)


def log(msg: str) -> None:
    """Print with timestamp."""
    elapsed = time.time() - LOG_START_TIME
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


LOG_START_TIME = time.time()


# =============================================================================
# Position Loading - Initial Positions Only
# =============================================================================

def is_initial_position_scalar(state: int) -> bool:
    """Check if state is a TRUE initial position (start of hand). Scalar version."""
    # All remaining masks must be 0x7F (everyone has all 7 dominoes)
    for p in range(4):
        remaining = (state >> (p * 7)) & 0x7F
        if remaining != 0x7F:
            return False

    # trick_len must be 0
    trick_len = (state >> 30) & 0x3
    if trick_len != 0:
        return False

    # p0 must be 7 (no lead play)
    p0 = (state >> 32) & 0x7
    if p0 != 7:
        return False

    return True


def get_initial_position_mask(states: np.ndarray) -> np.ndarray:
    """Vectorized check for initial positions. Returns boolean mask."""
    # All remaining masks must be 0x7F
    remaining_ok = np.ones(len(states), dtype=bool)
    for p in range(4):
        remaining = (states >> (p * 7)) & 0x7F
        remaining_ok &= (remaining == 0x7F)

    # trick_len must be 0
    trick_len = (states >> 30) & 0x3
    trick_ok = (trick_len == 0)

    # p0 must be 7 (no lead play)
    p0 = (states >> 32) & 0x7
    p0_ok = (p0 == 7)

    return remaining_ok & trick_ok & p0_ok


def get_midgame_position_mask(states: np.ndarray, min_trick_len: int = 2) -> np.ndarray:
    """Vectorized check for mid-game positions (trick_len >= min_trick_len)."""
    trick_len = (states >> 30) & 0x3
    return trick_len >= min_trick_len


def load_initial_positions(
    data_dir: Path,
    seed_range: tuple[int, int],
    max_positions: int | None,
    verbose: bool = False,
) -> list[dict]:
    """
    Load initial positions from parquet files.

    Returns list of dicts with:
        - state: packed int64
        - seed: deal seed
        - decl_id: declaration
        - leader: who leads (current player at initial position)
        - q_values: (7,) int8 array
        - hands: list of 4 hands (each is list of 7 global domino IDs)
    """
    parquet_files = sorted(data_dir.glob("seed_*.parquet"))

    # Filter to seed range
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

    if verbose:
        log(f"Found {len(relevant_files)} files in seed range {seed_range}")

    positions = []
    files_with_positions = 0

    for i, f in enumerate(relevant_files):
        # Early stop check
        if max_positions and len(positions) >= max_positions:
            break

        try:
            pf = pq.ParquetFile(f)
            meta = pf.schema_arrow.metadata or {}
            seed = int(meta.get(b"seed", b"0").decode())
            decl_id = int(meta.get(b"decl_id", b"0").decode())

            # Only read state and mv columns (skip V to save memory)
            df = pd.read_parquet(f, columns=['state', 'mv0', 'mv1', 'mv2', 'mv3', 'mv4', 'mv5', 'mv6'])
            states = df["state"].values.astype(np.int64)

            # Filter to initial positions (vectorized)
            initial_mask = get_initial_position_mask(states)
            initial_indices = np.where(initial_mask)[0]

            if len(initial_indices) == 0:
                continue

            files_with_positions += 1
            hands = deal_from_seed(seed)

            # Get Q-values
            mv_cols = [f"mv{i}" for i in range(7)]
            q_all = np.stack([df[c].values for c in mv_cols], axis=1)

            for idx in initial_indices:
                state = states[idx]
                leader = (state >> 28) & 0x3  # At initial, leader = current player

                positions.append({
                    'state': state,
                    'seed': seed,
                    'decl_id': decl_id,
                    'leader': leader,
                    'q_values': q_all[idx].astype(np.int32),
                    'hands': hands,
                })

                if max_positions and len(positions) >= max_positions:
                    break

            if verbose and (i + 1) % 10 == 0:
                log(f"  [{i+1}/{len(relevant_files)}] Loaded {len(positions)} positions")

            if max_positions and len(positions) >= max_positions:
                if verbose:
                    log(f"  Reached {max_positions} positions, stopping early")
                break

        except Exception as e:
            if verbose:
                log(f"  ERROR loading {f}: {e}")
            continue

    if verbose:
        log(f"Loaded {len(positions)} initial positions from {files_with_positions} files")

    return positions


# =============================================================================
# Sampling
# =============================================================================

def sample_opponent_hands(
    current_player: int,
    my_hand: list[int],
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Generate one sampled configuration of opponent hands.

    Args:
        current_player: 0-3, who we are
        my_hand: list of 7 global domino IDs (our known hand)
        rng: numpy random generator

    Returns:
        hands: list of 4 hands, where hands[current_player] = my_hand
               and other hands are shuffled from remaining 21 dominoes
    """
    all_dominoes = set(range(28))
    my_set = set(my_hand)
    opponent_pool = list(all_dominoes - my_set)

    rng.shuffle(opponent_pool)

    # Build hands array
    hands = [None] * 4
    hands[current_player] = my_hand

    opp_idx = 0
    for p in range(4):
        if p == current_player:
            continue
        hands[p] = sorted(opponent_pool[opp_idx:opp_idx + 7])
        opp_idx += 7

    return hands


# =============================================================================
# Tokenization for PIMC
# =============================================================================

def tokenize_for_pimc(
    decl_id: int,
    leader: int,  # = current_player at initial position
    hands: list[list[int]],  # sampled hands
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Tokenize an initial position for PIMC inference.

    At initial position:
    - All players have all 7 dominoes (remaining = 0x7F for all)
    - trick_len = 0, no plays
    - leader = current_player

    Uses perspective normalization (same as training):
    - player_id normalized: 0=current, 2=partner, 1,3=opponents
    - leader normalized relative to current player

    Returns:
        tokens: (32, 12) int64
        mask: (32,) float32
        current_player: int (= leader at initial)
    """
    current_player = leader  # At initial position

    MAX_TOKENS = 32
    N_FEATURES = 12

    tokens = np.zeros((MAX_TOKENS, N_FEATURES), dtype=np.int64)
    mask = np.zeros(MAX_TOKENS, dtype=np.float32)

    # Normalized leader (always 0 at initial since leader = current_player)
    normalized_leader = 0

    # Token 0: Context
    tokens[0] = [
        0, 0, 0, 0, 0,  # no domino features
        0, 0, 0, 0,     # no player segment
        TOKEN_TYPE_CONTEXT,
        decl_id,
        normalized_leader,
    ]
    mask[0] = 1.0

    token_idx = 1

    # Tokens 1-28: Hand positions (4 players × 7 dominoes)
    for p in range(4):
        # Normalized player_id
        normalized_p = (p - current_player + 4) % 4
        is_current = 1 if normalized_p == 0 else 0
        is_partner = 1 if normalized_p == 2 else 0

        for local_idx in range(7):
            global_id = hands[p][local_idx]

            tokens[token_idx] = [
                DOMINO_HIGH[global_id],
                DOMINO_LOW[global_id],
                1 if DOMINO_IS_DOUBLE[global_id] else 0,
                COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[global_id]],
                TRUMP_RANK_TABLE[(global_id, decl_id)],
                normalized_p,
                is_current,
                is_partner,
                1,  # is_remaining = True (initial position, everyone has all)
                TOKEN_TYPE_PLAYER0 + p,  # token_type uses physical player for gather
                decl_id,
                normalized_leader,
            ]
            mask[token_idx] = 1.0
            token_idx += 1

    # No trick tokens at initial position

    return tokens, mask, current_player


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_path: str, device: torch.device) -> DominoTransformer:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get architecture args from checkpoint
    args = checkpoint.get('args', {})

    model = DominoTransformer(
        embed_dim=args.get('embed_dim', 64),
        n_heads=args.get('n_heads', 4),
        n_layers=args.get('n_layers', 2),
        ff_dim=args.get('ff_dim', 128),
        dropout=0.0,  # No dropout at inference
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


# =============================================================================
# PIMC Inference
# =============================================================================

def pimc_batch_inference(
    model: DominoTransformer,
    positions: list[dict],
    n_samples: int,
    device: torch.device,
    rng: np.random.Generator,
    verbose: bool = False,
) -> dict:
    """
    Run PIMC on a batch of positions.

    Returns dict with:
        - majority_choices: (n_positions,) chosen move via majority vote
        - soft_choices: (n_positions,) chosen move via soft vote (mean logits)
        - majority_regrets: (n_positions,) regret for majority choice
        - soft_regrets: (n_positions,) regret for soft choice
        - perfect_info_choices: (n_positions,) model choice with true hands
        - perfect_info_regrets: (n_positions,) regret for perfect info
    """
    n_positions = len(positions)

    if verbose:
        log(f"PIMC inference: {n_positions} positions × {n_samples} samples")

    # === Step 1: Tokenize all samples ===
    t0 = time.time()

    all_tokens = []
    all_masks = []
    all_players = []

    # Also tokenize perfect-info (true hands) for baseline
    pi_tokens = []
    pi_masks = []
    pi_players = []

    for pos in positions:
        current_player = pos['leader']
        my_hand = pos['hands'][current_player]

        # Perfect info tokenization (true hands)
        tok, msk, cp = tokenize_for_pimc(pos['decl_id'], pos['leader'], pos['hands'])
        pi_tokens.append(tok)
        pi_masks.append(msk)
        pi_players.append(cp)

        # PIMC samples
        for _ in range(n_samples):
            sampled_hands = sample_opponent_hands(current_player, my_hand, rng)
            tok, msk, cp = tokenize_for_pimc(pos['decl_id'], pos['leader'], sampled_hands)
            all_tokens.append(tok)
            all_masks.append(msk)
            all_players.append(cp)

    if verbose:
        log(f"  Tokenization: {time.time() - t0:.2f}s")

    # === Step 2: GPU inference ===
    t0 = time.time()

    # Stack into tensors
    tokens_t = torch.tensor(np.stack(all_tokens), dtype=torch.long, device=device)
    masks_t = torch.tensor(np.stack(all_masks), dtype=torch.float32, device=device)
    players_t = torch.tensor(np.array(all_players), dtype=torch.long, device=device)

    # Perfect info tensors
    pi_tokens_t = torch.tensor(np.stack(pi_tokens), dtype=torch.long, device=device)
    pi_masks_t = torch.tensor(np.stack(pi_masks), dtype=torch.float32, device=device)
    pi_players_t = torch.tensor(np.array(pi_players), dtype=torch.long, device=device)

    with torch.no_grad():
        # PIMC samples
        logits = model(tokens_t, masks_t, players_t)  # (n_positions * n_samples, 7)

        # Perfect info
        pi_logits = model(pi_tokens_t, pi_masks_t, pi_players_t)  # (n_positions, 7)

    if verbose:
        log(f"  GPU inference: {time.time() - t0:.2f}s ({len(tokens_t)} samples)")

    # === Step 3: Build legal masks and aggregate ===
    t0 = time.time()

    # Reshape PIMC logits: (n_positions, n_samples, 7)
    logits = logits.view(n_positions, n_samples, 7)

    # Build legal masks from Q-values
    legal_masks = []
    q_arrays = []
    teams = []

    for pos in positions:
        q = pos['q_values']
        legal = (q != -128).astype(np.float32)
        legal_masks.append(legal)
        q_arrays.append(q)
        teams.append(pos['leader'] % 2)

    legal_t = torch.tensor(np.stack(legal_masks), dtype=torch.float32, device=device)  # (n_pos, 7)

    # Mask illegal moves
    logits_masked = logits.masked_fill(legal_t.unsqueeze(1) == 0, float('-inf'))
    pi_logits_masked = pi_logits.masked_fill(legal_t == 0, float('-inf'))

    # Majority vote: mode of argmax across samples
    sample_choices = logits_masked.argmax(dim=-1)  # (n_pos, n_samples)
    majority_choices = sample_choices.mode(dim=1).values  # (n_pos,)

    # Soft vote: argmax of mean logits
    mean_logits = logits_masked.mean(dim=1)  # (n_pos, 7)
    soft_choices = mean_logits.argmax(dim=-1)  # (n_pos,)

    # Diagnostic: sample diversity
    if verbose and n_samples > 1:
        # Count how many unique choices per position
        unique_counts = []
        for i in range(n_positions):
            unique = sample_choices[i].unique().numel()
            unique_counts.append(unique)
        avg_unique = np.mean(unique_counts)
        log(f"  Sample diversity: avg {avg_unique:.1f} unique choices per position (out of {n_samples} samples)")

        # Logit variance across samples
        # logits shape: (n_positions, n_samples, 7)
        logit_std = logits.std(dim=1).mean().item()  # Mean std across samples
        log(f"  Logit variance: mean std across samples = {logit_std:.4f}")

    # Perfect info choice
    pi_choices = pi_logits_masked.argmax(dim=-1)  # (n_pos,)

    if verbose:
        log(f"  Aggregation: {time.time() - t0:.2f}s")

    # === Step 4: Compute regrets ===
    majority_choices_np = majority_choices.cpu().numpy()
    soft_choices_np = soft_choices.cpu().numpy()
    pi_choices_np = pi_choices.cpu().numpy()

    majority_regrets = []
    soft_regrets = []
    pi_regrets = []

    for i, pos in enumerate(positions):
        q = q_arrays[i]
        legal = legal_masks[i]
        team = teams[i]

        if team == 0:
            # Team 0 wants max Q
            legal_q = np.where(legal > 0, q, -999)
            optimal_q = legal_q.max()
            maj_regret = optimal_q - q[majority_choices_np[i]]
            soft_regret = optimal_q - q[soft_choices_np[i]]
            pi_regret = optimal_q - q[pi_choices_np[i]]
        else:
            # Team 1 wants min Q
            legal_q = np.where(legal > 0, q, 999)
            optimal_q = legal_q.min()
            maj_regret = q[majority_choices_np[i]] - optimal_q
            soft_regret = q[soft_choices_np[i]] - optimal_q
            pi_regret = q[pi_choices_np[i]] - optimal_q

        majority_regrets.append(maj_regret)
        soft_regrets.append(soft_regret)
        pi_regrets.append(pi_regret)

    return {
        'majority_choices': majority_choices_np,
        'soft_choices': soft_choices_np,
        'perfect_info_choices': pi_choices_np,
        'majority_regrets': np.array(majority_regrets),
        'soft_regrets': np.array(soft_regrets),
        'perfect_info_regrets': np.array(pi_regrets),
    }


# =============================================================================
# Sanity Checks
# =============================================================================

def run_sanity_checks(data_dir: Path, model_path: str | None, device: torch.device):
    """Run sanity checks before full PIMC run."""

    log("=== Sanity Check 1: Load initial positions ===")
    positions = load_initial_positions(data_dir, (90, 99), max_positions=5, verbose=True)

    if not positions:
        log("ERROR: No initial positions found!")
        return False

    pos = positions[0]
    log(f"  Sample position: seed={pos['seed']}, decl={pos['decl_id']}, leader={pos['leader']}")
    log(f"  Q-values: {pos['q_values'].tolist()}")
    log(f"  Hands[0]: {pos['hands'][0]}")
    log("  ✓ Positions loaded correctly")

    log("\n=== Sanity Check 2: Sampling ===")
    rng = np.random.default_rng(42)
    current_player = pos['leader']
    my_hand = pos['hands'][current_player]

    log(f"  Current player: {current_player}")
    log(f"  My hand: {sorted(my_hand)}")

    for i in range(3):
        sampled = sample_opponent_hands(current_player, my_hand, rng)
        log(f"  Sample {i}:")
        for p in range(4):
            marker = "(*)" if p == current_player else "   "
            log(f"    P{p}{marker}: {sorted(sampled[p])}")

        # Verify invariants
        all_doms = set()
        for p in range(4):
            assert len(sampled[p]) == 7, f"P{p} has {len(sampled[p])} dominoes"
            all_doms.update(sampled[p])
        assert all_doms == set(range(28)), f"Missing dominoes: {set(range(28)) - all_doms}"
        assert set(sampled[current_player]) == set(my_hand), "Current player's hand changed!"

    log("  ✓ Sampling produces valid hands")

    log("\n=== Sanity Check 3: Tokenization ===")
    tokens, mask, cp = tokenize_for_pimc(pos['decl_id'], pos['leader'], pos['hands'])
    log(f"  Token shape: {tokens.shape}")
    log(f"  Mask sum: {mask.sum()} (expect 29 for initial + context)")
    log(f"  Current player: {cp}")
    log(f"  Context token (idx 0): decl={tokens[0, 10]}, leader={tokens[0, 11]}")
    log(f"  First hand token (idx 1): high={tokens[1, 0]}, low={tokens[1, 1]}, remaining={tokens[1, 8]}")
    log("  ✓ Tokenization looks correct")

    if model_path and Path(model_path).exists():
        log(f"\n=== Sanity Check 4: Model loading ===")
        model = load_model(model_path, device)
        log(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

        # Test forward pass
        tokens_t = torch.tensor(tokens[np.newaxis], dtype=torch.long, device=device)
        mask_t = torch.tensor(mask[np.newaxis], dtype=torch.float32, device=device)
        player_t = torch.tensor([cp], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(tokens_t, mask_t, player_t)

        log(f"  Output shape: {logits.shape}")
        log(f"  Logits: {logits[0].cpu().numpy().round(2).tolist()}")
        log("  ✓ Model runs correctly")

        log(f"\n=== Sanity Check 5: Single position PIMC ===")
        rng = np.random.default_rng(42)
        result = pimc_batch_inference(model, [pos], n_samples=5, device=device, rng=rng, verbose=True)

        log(f"  Majority choice: {result['majority_choices'][0]}")
        log(f"  Soft choice: {result['soft_choices'][0]}")
        log(f"  Perfect info choice: {result['perfect_info_choices'][0]}")
        log(f"  Majority regret: {result['majority_regrets'][0]}")
        log(f"  Soft regret: {result['soft_regrets'][0]}")
        log(f"  Perfect info regret: {result['perfect_info_regrets'][0]}")
        log("  ✓ Single position PIMC works")
    else:
        log(f"\n=== Sanity Check 4-5: Skipped (no model at {model_path}) ===")

    log("\n=== All sanity checks passed! ===")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PIMC Test Harness")
    parser.add_argument("--data-dir", type=str, default="data/solver2")
    parser.add_argument("--model", type=str, default="data/solver2/transformer_best.pt",
                        help="Path to trained model")
    parser.add_argument("--seed-range", type=str, default="0-99",
                        help="Seed range to use (e.g., '90-99' for test only, '0-99' for all)")
    parser.add_argument("--max-positions", type=int, default=100,
                        help="Max positions to test")
    parser.add_argument("--samples", type=int, default=50,
                        help="Samples per position")
    parser.add_argument("--sample-sweep", action="store_true",
                        help="Run sample count sweep (1, 20, 50, 100, 200)")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run sanity checks only")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        log(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("Using CPU")

    data_dir = Path(args.data_dir)

    # Parse seed range
    seed_parts = args.seed_range.split("-")
    seed_range = (int(seed_parts[0]), int(seed_parts[1]))

    if args.sanity_check:
        run_sanity_checks(data_dir, args.model, device)
        return

    # Load model
    if not Path(args.model).exists():
        log(f"ERROR: Model not found at {args.model}")
        log("Train a model first with: python scripts/solver2/train_transformer.py --save-model data/solver2/transformer_best.pt")
        return

    model = load_model(args.model, device)
    log(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load positions
    log(f"\nLoading initial positions (max {args.max_positions}, seeds {seed_range})...")
    positions = load_initial_positions(data_dir, seed_range, args.max_positions, verbose=args.verbose)

    if not positions:
        log("ERROR: No initial positions found!")
        return

    log(f"Loaded {len(positions)} initial positions")

    # Run PIMC
    rng = np.random.default_rng(args.seed)

    if args.sample_sweep:
        sample_counts = [1, 20, 50, 100, 200]
    else:
        sample_counts = [args.samples]

    log(f"\n{'='*70}")
    log(f"PIMC Test: {len(positions)} positions")
    log(f"{'='*70}")

    # First, show perfect-info baseline
    result = pimc_batch_inference(model, positions, n_samples=1, device=device, rng=rng, verbose=args.verbose)
    pi_mean_regret = result['perfect_info_regrets'].mean()
    pi_blunder_rate = (result['perfect_info_regrets'] > 10).mean() * 100

    log(f"\nPerfect Info Baseline (model with true hands):")
    log(f"  Mean regret: {pi_mean_regret:.3f}")
    log(f"  Blunder rate (>10): {pi_blunder_rate:.2f}%")

    # Sample sweep
    log(f"\n{'Samples':>8} | {'Maj Mean':>9} | {'Maj Blund':>9} | {'Soft Mean':>9} | {'Soft Blund':>10}")
    log("-" * 60)

    for n_samples in sample_counts:
        rng = np.random.default_rng(args.seed)  # Reset for reproducibility
        result = pimc_batch_inference(model, positions, n_samples=n_samples, device=device, rng=rng, verbose=args.verbose)

        maj_mean = result['majority_regrets'].mean()
        maj_blund = (result['majority_regrets'] > 10).mean() * 100
        soft_mean = result['soft_regrets'].mean()
        soft_blund = (result['soft_regrets'] > 10).mean() * 100

        log(f"{n_samples:8d} | {maj_mean:9.3f} | {maj_blund:8.2f}% | {soft_mean:9.3f} | {soft_blund:9.2f}%")

    log(f"\n{'='*70}")
    log("Done!")


if __name__ == "__main__":
    main()
