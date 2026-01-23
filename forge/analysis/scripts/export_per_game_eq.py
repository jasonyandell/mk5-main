#!/usr/bin/env python3
"""Export per-game E[Q] data for 3D visualization.

Creates JSONL with one record per game containing:
- Player-to-domino mapping (who has what)
- 28x28 E[Q] matrix (dominoes × moves), interpolated for continuity
- Active mask showing which player is deciding at each move
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import json
import numpy as np
import torch
from pathlib import Path
# No scipy needed - using simple step-hold

from forge.oracle import schema

# Trump declaration names
DECL_NAMES = {
    0: 'blanks', 1: 'ones', 2: 'twos', 3: 'threes',
    4: 'fours', 5: 'fives', 6: 'sixes',
    7: 'doubles', 8: 'doubles-suit', 9: 'no-trump'
}


def pips_to_domino_id(high: int, low: int) -> int:
    """Convert (high, low) pips to domino ID (0-27)."""
    h, l = max(high, low), min(high, low)
    return h * (h + 1) // 2 + l


def extract_hand_dominoes(tokens, length) -> list[int]:
    """Extract list of domino IDs from transcript tokens."""
    FEAT_HIGH = 0
    FEAT_LOW = 1
    FEAT_TOKEN_TYPE = 7
    TOKEN_TYPE_HAND = 1

    hand = []
    for i in range(length):
        if tokens[i, FEAT_TOKEN_TYPE] == TOKEN_TYPE_HAND:
            high = tokens[i, FEAT_HIGH].item()
            low = tokens[i, FEAT_LOW].item()
            domino_id = pips_to_domino_id(high, low)
            hand.append(domino_id)
    return hand


def step_hold_gaps(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill gaps using step-hold (carry forward last known value).

    Args:
        values: Array with some valid values
        valid_mask: Boolean mask where True = valid value

    Returns:
        Array with gaps filled by carrying forward last known value
    """
    result = values.copy()

    if valid_mask.sum() == 0:
        return result

    # Carry forward: fill each gap with the last known value
    last_valid = np.nan
    for i in range(len(values)):
        if valid_mask[i]:
            last_valid = values[i]
            result[i] = values[i]
        else:
            result[i] = last_valid

    return result


def process_game(data: dict, game_idx: int) -> dict | None:
    """Process a single game into visualization format.

    Returns dict with:
        - game_id: str
        - players: list of {id, dominoes: [{id, pips, slot}]}
        - eq_matrix: 28x28 (dominoes grouped by player × moves)
        - active_player: list of 28 ints (which player is active at each move)
        - domino_played: 28x28 bool (True if domino was played at that move)
    """
    game_mask = data['game_idx'] == game_idx
    decision_indices = torch.where(game_mask)[0]

    if len(decision_indices) != 28:
        print(f"  Game {game_idx}: {len(decision_indices)} decisions (expected 28), skipping")
        return None

    n_dominoes = 28
    n_moves = 28

    # Extract trump declaration from first token of first decision
    first_dec_idx = decision_indices[0]
    trump_id = data['transcript_tokens'][first_dec_idx, 0, 6].item()
    trump_name = DECL_NAMES.get(trump_id, f'unknown-{trump_id}')

    # Initialize structures
    # eq_by_domino[domino_id] = {move_idx: eq_value}
    eq_by_domino = {d: {} for d in range(n_dominoes)}
    active_player = []  # Which player is active at each move

    # Track which dominoes each player has (from first appearance)
    player_dominoes = {p: set() for p in range(4)}
    domino_to_player = {}

    # Track when each domino was played
    domino_played_at = {}  # domino_id -> move_idx

    for move_idx, dec_idx in enumerate(decision_indices):
        player = data['player'][dec_idx].item()
        active_player.append(player)

        # Get this player's hand at this decision
        tokens = data['transcript_tokens'][dec_idx]
        length = data['transcript_lengths'][dec_idx].item()
        hand = extract_hand_dominoes(tokens, length)

        # Record player ownership
        for domino_id in hand:
            if domino_id not in domino_to_player:
                domino_to_player[domino_id] = player
                player_dominoes[player].add(domino_id)

        # Get E[Q] values for legal moves
        eq_values = data['e_q_mean'][dec_idx]
        legal_mask = data['legal_mask'][dec_idx]

        for slot, domino_id in enumerate(hand):
            if slot < 7 and legal_mask[slot].item():
                eq_by_domino[domino_id][move_idx] = eq_values[slot].item()

        # Track which domino was played
        action = data['action_taken'][dec_idx].item()
        if 0 <= action < len(hand):
            played_domino = hand[action]
            domino_played_at[played_domino] = move_idx

    # Build player structure with dominoes in consistent order
    players = []
    domino_order = []  # Global order: P0's 7, then P1's 7, etc.

    for p in range(4):
        p_dominoes = sorted(player_dominoes[p])
        # Pad to 7 if needed (shouldn't happen in valid games)
        while len(p_dominoes) < 7:
            p_dominoes.append(-1)

        player_info = {
            "id": p,
            "dominoes": []
        }
        for slot, dom_id in enumerate(p_dominoes[:7]):
            if dom_id >= 0:
                pips = schema.domino_pips(dom_id)
                player_info["dominoes"].append({
                    "id": dom_id,
                    "pips": f"{pips[0]}-{pips[1]}",
                    "slot": slot
                })
                domino_order.append(dom_id)
            else:
                player_info["dominoes"].append(None)
                domino_order.append(-1)

        players.append(player_info)

    # Build 28x28 E[Q] matrix (rows = dominoes in player order, cols = moves)
    eq_matrix_raw = np.full((n_dominoes, n_moves), np.nan)

    for row_idx, dom_id in enumerate(domino_order):
        if dom_id < 0:
            continue
        for move_idx, eq_val in eq_by_domino[dom_id].items():
            eq_matrix_raw[row_idx, move_idx] = eq_val

    # Step-hold gaps for each domino (carry forward last known value)
    eq_matrix = np.full((n_dominoes, n_moves), np.nan)

    for row_idx, dom_id in enumerate(domino_order):
        if dom_id < 0:
            continue

        valid_mask = ~np.isnan(eq_matrix_raw[row_idx])

        if valid_mask.sum() > 0:
            eq_matrix[row_idx] = step_hold_gaps(eq_matrix_raw[row_idx], valid_mask)

            # Truncate after domino was played
            if dom_id in domino_played_at:
                played_move = domino_played_at[dom_id]
                eq_matrix[row_idx, played_move + 1:] = np.nan

    # Build domino_played mask
    domino_played = np.zeros((n_dominoes, n_moves), dtype=bool)
    for row_idx, dom_id in enumerate(domino_order):
        if dom_id in domino_played_at:
            domino_played[row_idx, domino_played_at[dom_id]] = True

    return {
        "game_id": f"game_{game_idx:04d}",
        "game_idx": game_idx,
        "trump_id": trump_id,
        "trump_name": trump_name,
        "players": players,
        "domino_order": domino_order,
        "eq_matrix": eq_matrix.tolist(),
        "active_player": active_player,
        "domino_played": domino_played.tolist(),
    }


def main():
    data_path = Path("/home/jason/v2/mk5-tailwind/forge/data/eq_v2.2_250g.pt")
    output_path = Path("/home/jason/v2/mk5-tailwind/forge/analysis/results/data/27b_eq_per_game.jsonl")

    print(f"Loading {data_path}...")
    data = torch.load(data_path, weights_only=False)

    n_games = data['game_idx'].max().item() + 1
    print(f"Processing {n_games} games...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    valid_games = 0
    with open(output_path, 'w') as f:
        for game_idx in range(n_games):
            if game_idx % 50 == 0:
                print(f"  {game_idx}/{n_games}...")

            result = process_game(data, game_idx)
            if result:
                # Convert NaN to null for JSON
                eq_matrix = result["eq_matrix"]
                for row in eq_matrix:
                    for i in range(len(row)):
                        if row[i] is not None and np.isnan(row[i]):
                            row[i] = None

                f.write(json.dumps(result) + '\n')
                valid_games += 1

    print(f"✓ Saved {valid_games} games to {output_path}")


if __name__ == "__main__":
    main()
