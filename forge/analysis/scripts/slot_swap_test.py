#!/usr/bin/env python3
"""Test whether model bias follows card content or output slot position.

Experiment:
1. Find a failure case where oracle_best=0 but model picks something else
2. Swap the token features of domino 0 and domino 3 in the input
3. Run inference before and after
4. If model now picks slot 0 (with domino 3's features) -> content-based
5. If model still avoids slot 0 -> positional output bias

This tests: is the model learning "avoid output 0" vs "avoid low-pip dominoes"?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = "/home/jason/v2/mk5-tailwind"
sys.path.insert(0, PROJECT_ROOT)

from forge.ml.module import DominoLightningModule

# Token feature indices
FEAT_HIGH_PIP = 0
FEAT_LOW_PIP = 1
FEAT_IS_DOUBLE = 2
FEAT_COUNT_VALUE = 3
FEAT_TRUMP_RANK = 4
FEAT_NORMALIZED_PLAYER = 5
FEAT_IS_CURRENT = 6
FEAT_IS_PARTNER = 7
FEAT_IS_REMAINING = 8
FEAT_TOKEN_TYPE = 9
FEAT_DECL_ID = 10
FEAT_NORMALIZED_LEADER = 11

MODEL_PATH = Path(PROJECT_ROOT) / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
RESULTS_DIR = Path(PROJECT_ROOT) / "forge/analysis/results"


def load_model(device: str = "cuda") -> DominoLightningModule:
    """Load the trained Q-value model."""
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    if 'rng_state' in checkpoint:
        del checkpoint['rng_state']
    hparams = checkpoint.get('hyper_parameters', {})
    model = DominoLightningModule(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    return model


def find_domino_position(tokens: np.ndarray, domino_id: int) -> int | None:
    """Find which token position contains a specific domino for current player."""
    # Domino IDs map to (high, low) pips
    # ID 0 = (0,0), ID 1 = (1,0), ID 2 = (1,1), ID 3 = (2,0), etc.
    # Formula: high_pip = floor((sqrt(1+8*id) - 1) / 2), then low = id - high*(high+1)/2

    # Build lookup: domino_id -> (high, low)
    domino_to_pips = {}
    idx = 0
    for high in range(7):
        for low in range(high + 1):
            domino_to_pips[idx] = (high, low)
            idx += 1

    target_high, target_low = domino_to_pips[domino_id]

    # Search hand tokens (positions 1-28) for current player's remaining dominoes
    for pos in range(1, 29):
        if tokens[pos, FEAT_IS_CURRENT] == 1 and tokens[pos, FEAT_IS_REMAINING] == 1:
            high = tokens[pos, FEAT_HIGH_PIP]
            low = tokens[pos, FEAT_LOW_PIP]
            if high == target_high and low == target_low:
                return pos
    return None


def get_hand_dominoes(tokens: np.ndarray) -> list[tuple[int, int, int]]:
    """Get all dominoes in current player's remaining hand.

    Returns list of (token_position, domino_id, (high, low))
    """
    # Build lookup: (high, low) -> domino_id
    pips_to_id = {}
    idx = 0
    for high in range(7):
        for low in range(high + 1):
            pips_to_id[(high, low)] = idx
            idx += 1

    hand = []
    for pos in range(1, 29):
        if tokens[pos, FEAT_IS_CURRENT] == 1 and tokens[pos, FEAT_IS_REMAINING] == 1:
            high = int(tokens[pos, FEAT_HIGH_PIP])
            low = int(tokens[pos, FEAT_LOW_PIP])
            dom_id = pips_to_id[(high, low)]
            hand.append((pos, dom_id, (high, low)))
    return hand


def run_slot_swap_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(device)

    # Load high-regret samples
    hr_df = pd.read_parquet(RESULTS_DIR / "tables/high_regret_samples.parquet")

    # Find cases where oracle_best=0 but model picked differently
    slot0_failures = hr_df[(hr_df['oracle_best'] == 0) & (hr_df['model_pick'] != 0)]
    print(f"Found {len(slot0_failures)} cases where oracle_best=0 but model didn't pick it")

    if len(slot0_failures) == 0:
        print("No slot 0 failures found!")
        return

    # Load validation tokens
    tokens_all = np.load(Path(PROJECT_ROOT) / "data/tokenized-full/val/tokens.npy")
    masks_all = np.load(Path(PROJECT_ROOT) / "data/tokenized-full/val/masks.npy")
    players_all = np.load(Path(PROJECT_ROOT) / "data/tokenized-full/val/players.npy")

    results = []
    tested = 0
    debug_count = 0

    for _, row in slot0_failures.iterrows():
        if tested >= 50:  # Test 50 cases
            break

        sample_idx = row['sample_idx']

        # Get original tokens
        tokens = tokens_all[sample_idx].copy()
        mask = masks_all[sample_idx].copy()
        player = players_all[sample_idx]

        # Get current player's hand
        hand = get_hand_dominoes(tokens)

        # Debug: show first few hands
        if debug_count < 10:
            print(f"\nDebug sample {sample_idx}:")
            print(f"  Hand has {len(hand)} dominoes: {[d[1] for d in hand]}")
            print(f"  oracle_best={row['oracle_best']}, model_pick={row['model_pick']}")
            legal_mask = row['legal_mask']
            legal_actions = [i for i, x in enumerate(legal_mask) if x > 0]
            print(f"  Legal actions: {legal_actions}")
            print(f"  Is action 0 legal? {legal_mask[0]}")
            debug_count += 1
            if debug_count >= 10:
                print("\n*** Stopping after 10 debug samples ***")
                return

        # Find domino 0 and another domino to swap with
        pos_dom0 = None
        swap_candidate = None
        for pos, dom_id, pips in hand:
            if dom_id == 0:
                pos_dom0 = pos
            elif swap_candidate is None and dom_id > 0:
                swap_candidate = (pos, dom_id, pips)

        if pos_dom0 is None or swap_candidate is None:
            if debug_count < 10:
                print(f"  -> Skipping: pos_dom0={pos_dom0}, swap_candidate={swap_candidate}")
            continue  # Can't do swap test

        pos_other, other_id, other_pips = swap_candidate
        tested += 1

        # Build legal mask from stored data
        legal = np.array(row['legal_mask'], dtype=np.float32)

        # Run original inference
        tokens_t = torch.from_numpy(tokens).unsqueeze(0).to(device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
        player_t = torch.tensor([player]).to(device)

        with torch.no_grad():
            model_q_orig, _ = model(tokens_t, mask_t, player_t)

        q_orig = model_q_orig[0].cpu().numpy()
        q_masked_orig = np.where(legal > 0, q_orig, -np.inf)
        pred_orig = int(np.argmax(q_masked_orig))

        # Now swap the token features for dominoes 0 and other
        tokens_swapped = tokens.copy()

        # Swap the card-identifying features (pips, double status, trump rank)
        swap_features = [FEAT_HIGH_PIP, FEAT_LOW_PIP, FEAT_IS_DOUBLE, FEAT_COUNT_VALUE, FEAT_TRUMP_RANK]
        for feat in swap_features:
            tokens_swapped[pos_dom0, feat], tokens_swapped[pos_other, feat] = \
                tokens[pos_other, feat], tokens[pos_dom0, feat]

        # Also swap legal mask entries (0 <-> other_id)
        legal_swapped = legal.copy()
        legal_swapped[0], legal_swapped[other_id] = legal[other_id], legal[0]

        # Run swapped inference
        tokens_swapped_t = torch.from_numpy(tokens_swapped).unsqueeze(0).to(device)

        with torch.no_grad():
            model_q_swapped, _ = model(tokens_swapped_t, mask_t, player_t)

        q_swapped = model_q_swapped[0].cpu().numpy()
        q_masked_swapped = np.where(legal_swapped > 0, q_swapped, -np.inf)
        pred_swapped = int(np.argmax(q_masked_swapped))

        # Interpretation:
        # - If model picks slot 0 after swap -> it now values the features that moved there (CONTENT-BASED)
        # - If model picks slot other_id after swap -> it follows where dom0's features went (FOLLOWS CARD)
        # - If model still avoids slot 0 regardless -> POSITIONAL OUTPUT BIAS

        result = {
            'sample_idx': sample_idx,
            'swap_with': other_id,
            'orig_pick': pred_orig,
            'swapped_pick': pred_swapped,
            'q_orig_0': q_orig[0],
            'q_orig_other': q_orig[other_id],
            'q_swapped_0': q_swapped[0],
            'q_swapped_other': q_swapped[other_id],
        }
        results.append(result)

        if tested <= 10:  # Print first 10 details
            print(f"\nSample {sample_idx} (swap dom0 <-> dom{other_id}):")
            print(f"  Original:  pick={pred_orig}, Q[0]={q_orig[0]:.2f}, Q[{other_id}]={q_orig[other_id]:.2f}")
            print(f"  Swapped:   pick={pred_swapped}, Q[0]={q_swapped[0]:.2f}, Q[{other_id}]={q_swapped[other_id]:.2f}")

            if pred_swapped == 0:
                print(f"  -> Model picks slot 0 (now has dom{other_id} features): CONTENT-BASED")
            elif pred_swapped == other_id:
                print(f"  -> Model picks slot {other_id} (now has dom0 features): FOLLOWS THE CARD")
            else:
                print(f"  -> Model picks {pred_swapped} (neither): INCONCLUSIVE")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results:
        picks_slot0 = sum(1 for r in results if r['swapped_pick'] == 0)
        picks_other_slot = sum(1 for r in results if r['swapped_pick'] == r['swap_with'])
        picks_neither = len(results) - picks_slot0 - picks_other_slot

        print(f"After swapping domino 0 with another domino's features ({len(results)} tests):")
        print(f"  Picks slot 0 (CONTENT-BASED - values the features):    {picks_slot0:3d} ({100*picks_slot0/len(results):.1f}%)")
        print(f"  Picks swapped slot (FOLLOWS THE CARD):                 {picks_other_slot:3d} ({100*picks_other_slot/len(results):.1f}%)")
        print(f"  Picks other (inconclusive):                            {picks_neither:3d} ({100*picks_neither/len(results):.1f}%)")

        # Check Q-value changes
        avg_q0_change = np.mean([r['q_swapped_0'] - r['q_orig_0'] for r in results])
        avg_other_change = np.mean([r['q_swapped_other'] - r['q_orig_other'] for r in results])
        print(f"\nQ-value changes after swap:")
        print(f"  Avg Q[0] change:     {avg_q0_change:+.2f} (slot now has other domino's features)")
        print(f"  Avg Q[other] change: {avg_other_change:+.2f} (slot now has dom0's features)")


if __name__ == "__main__":
    run_slot_swap_experiment()
