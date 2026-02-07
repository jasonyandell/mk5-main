#!/usr/bin/env python3
"""Export per-game E[Q] PDF data from exp-100games-2M format.

New format contains:
- transcript_tokens: [28, 36, 8] - Full sequence at each decision
- e_q_mean/e_q_var: [28, 7] - Mean and variance per slot
- e_q_pdf: [28, 7, 85] - Full PDF per slot
- legal_mask: [28, 7] - Legal actions
- action_taken: [28] - Original slot index played
- metadata: {seed, decl_id, ...}
"""

import sys
sys.path.insert(0, "/home/jason/v2/mk5-tailwind")

import json
import numpy as np
import torch
from pathlib import Path

from forge.oracle import schema, tables
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import led_suit_for_lead_domino, trick_rank

# Trump declaration names
DECL_NAMES = {
    0: 'blanks', 1: 'ones', 2: 'twos', 3: 'threes',
    4: 'fours', 5: 'fives', 6: 'sixes',
    7: 'doubles', 8: 'doubles-suit', 9: 'no-trump'
}


def determine_trick_winner(trick_plays: list, decl_id: int) -> int:
    """Determine winner of a complete trick."""
    if len(trick_plays) != 4:
        raise ValueError("Trick must have 4 plays")

    lead_player, lead_domino_id = trick_plays[0]
    led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)

    best_idx = 0
    best_rank = trick_rank(lead_domino_id, led_suit, decl_id)

    for i in range(1, 4):
        _, domino_id = trick_plays[i]
        rank = trick_rank(domino_id, led_suit, decl_id)
        if rank > best_rank:
            best_idx = i
            best_rank = rank

    return trick_plays[best_idx][0]


def process_game(data: dict) -> dict:
    """Process a single game file into visualization format."""
    metadata = data['metadata']
    seed = metadata['seed']
    decl_id = metadata['decl_id']

    # Reconstruct hands from seed
    hands = deal_from_seed(seed)

    n_dominoes = 28
    n_moves = 28
    n_bins = 85  # -42 to +42

    # Build player structure
    players = []
    domino_order = []

    for p in range(4):
        player_hand = hands[p]
        player_info = {
            "id": p,
            "dominoes": []
        }
        for slot, dom_id in enumerate(player_hand):
            pips = schema.domino_pips(dom_id)
            player_info["dominoes"].append({
                "id": dom_id,
                "pips": f"{pips[0]}-{pips[1]}",
                "slot": slot
            })
            domino_order.append(dom_id)
        players.append(player_info)

    # Extract tensors
    e_q_mean = data['e_q_mean'].numpy()  # [28, 7]
    e_q_var = data['e_q_var'].numpy()    # [28, 7] - variance of E[Q] estimate
    e_q_pdf = data['e_q_pdf'].numpy()    # [28, 7, 85]
    legal_mask = data['legal_mask'].numpy()  # [28, 7]
    action_taken = data['action_taken'].numpy()  # [28]
    n_samples = data['n_samples'].numpy()  # [28] - samples per decision
    converged = data['converged'].numpy()  # [28] - convergence status

    # Use sparse representation: only store non-zero entries
    pdf_data = []  # List of {domino_idx, move_idx, pdf: [85 floats]}
    active_player = []
    domino_played = np.zeros((n_dominoes, n_moves), dtype=bool)

    # Score tracking
    team_scores = [0, 0]  # Team 0 (P0+P2), Team 1 (P1+P3)
    score_history = []  # Score after each move
    trick_plays = []  # Current trick's plays: [(player, domino_id), ...]
    trick_winners = []  # Winner of each completed trick

    # Replay the game to determine active player at each decision
    current_player = 0  # P0 leads first

    for move_idx in range(28):
        player = current_player
        active_player.append(int(player))

        legal = legal_mask[move_idx]  # [7] bool
        pdf = e_q_pdf[move_idx]       # [7, 85]
        mean = e_q_mean[move_idx]     # [7]
        var = e_q_var[move_idx]       # [7] - variance of E[Q]
        action = action_taken[move_idx]
        samples = int(n_samples[move_idx])
        conv = bool(converged[move_idx])

        # Compute win probability threshold based on team
        # Offense (P0/P2): win if E[Q] >= 18 (bins 60-84)
        # Defense (P1/P3): win if E[Q] > -18 (bins 25-84, since bin 24 = -18)
        is_offense = player in [0, 2]
        win_bin_start = 60 if is_offense else 25

        # Store PDF for each legal action
        for orig_slot in range(7):
            if not legal[orig_slot]:
                continue

            dom_id = hands[player][orig_slot]
            global_idx = player * 7 + orig_slot  # position in domino_order

            slot_pdf = pdf[orig_slot]
            slot_mean = float(mean[orig_slot])
            slot_var = float(var[orig_slot])
            slot_std = float(np.sqrt(slot_var)) if slot_var > 0 else 0
            win_prob = float(slot_pdf[win_bin_start:].sum())

            pdf_data.append({
                "d": global_idx,  # domino index in domino_order
                "m": move_idx,    # move index
                "pdf": slot_pdf.tolist(),  # 85-element array
                "mean": slot_mean,
                "std": slot_std,  # standard deviation of E[Q] estimate
                "win": win_prob,  # P(win) for this action
                "samples": samples,  # Monte Carlo samples used
                "converged": conv,   # whether estimate converged
            })

        # Mark which domino was played and track for scoring
        if 0 <= action < 7:
            global_idx = player * 7 + action
            domino_played[global_idx, move_idx] = True

            # Track play for trick scoring
            dom_id = hands[player][action]
            trick_plays.append((player, dom_id))

            # Check if trick is complete (4 plays)
            if len(trick_plays) == 4:
                winner = determine_trick_winner(trick_plays, decl_id)
                trick_winners.append(winner)

                # Calculate trick points: 1 for trick + count points
                trick_points = 1
                for _, d_id in trick_plays:
                    trick_points += tables.DOMINO_COUNT_POINTS[d_id]

                # Award to winning team
                winning_team = 0 if winner in [0, 2] else 1
                team_scores[winning_team] += trick_points

                trick_plays = []
                current_player = winner
            else:
                current_player = (current_player + 1) % 4

        # Record score after this move
        score_history.append([team_scores[0], team_scores[1]])

    trump_name = DECL_NAMES.get(decl_id, f'unknown-{decl_id}')

    return {
        "game_id": f"seed_{seed}_decl_{decl_id}",
        "seed": seed,
        "trump_id": decl_id,
        "trump_name": trump_name,
        "players": players,
        "domino_order": domino_order,
        "active_player": active_player,
        "domino_played": domino_played.tolist(),
        "pdf_data": pdf_data,  # Sparse: only active decisions
        "score_history": score_history,  # [[team0, team1], ...] after each move
        "trick_winners": trick_winners,  # Winner of each trick (0-3)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export E[Q] PDF data from exp-100games-2M format")
    parser.add_argument("input", type=Path, help="Input .pt file or directory")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSONL file")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of games")
    args = parser.parse_args()

    # Find input files
    if args.input.is_file():
        input_files = [args.input]
    else:
        input_files = sorted(args.input.glob("*.pt"))
        if args.limit:
            input_files = input_files[:args.limit]

    if not input_files:
        print(f"No .pt files found in {args.input}")
        return

    # Default output path
    if args.output is None:
        args.output = Path("forge/analysis/results/data/eq_pdf_v3.jsonl")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(input_files)} game(s)...")

    with open(args.output, 'w') as f:
        for i, input_file in enumerate(input_files):
            print(f"  Game {i+1}/{len(input_files)}: {input_file.name}")
            data = torch.load(input_file, weights_only=False)
            result = process_game(data)
            f.write(json.dumps(result) + '\n')

    print(f"✓ Saved {len(input_files)} games to {args.output}")


if __name__ == "__main__":
    main()
