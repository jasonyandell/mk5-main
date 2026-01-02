#!/usr/bin/env python3
"""Investigation mode: debug why near-certain hands lose in simulation.

Runs simulations, captures full game state, and replays losing deals
trick-by-trick to identify where the model misplayed.

Usage:
    python -m forge.bidding.investigate --hand "6-6,6-5,6-4,6-2,6-1,6-0,2-2" --trump sixes --below 42 --samples 500
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from forge.oracle.declarations import DECL_ID_TO_NAME, DECL_NAME_TO_ID
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW

from .evaluate import parse_hand, format_hand
from .inference import PolicyModel
from .simulator import (
    BatchedGameState,
    deal_random_hands,
    _get_table,
)


@dataclass
class TrickRecord:
    """Record of a single trick."""
    leader: int
    plays: List[tuple[int, int, int]]  # (player, local_idx, global_dom_id)
    winner: int
    points: int


@dataclass
class GameRecord:
    """Full record of a game for investigation."""
    game_idx: int
    hands: List[List[int]]  # 4 players x 7 dominoes (global IDs)
    tricks: List[TrickRecord]
    final_score: int  # Team 0's points


def format_domino(dom_id: int) -> str:
    """Format domino ID as high-low string."""
    return f"{DOMINO_HIGH[dom_id]}-{DOMINO_LOW[dom_id]}"


def simulate_with_history(
    model: PolicyModel,
    bidder_hand: List[int],
    decl_id: int,
    n_games: int,
    seed: int | None = None,
    greedy: bool = True,
) -> tuple[Tensor, List[GameRecord]]:
    """Simulate games and capture full history.

    Args:
        model: The policy model for action selection
        bidder_hand: 7 global domino IDs for player 0 (the bidder)
        decl_id: Declaration/trump type (0-9)
        n_games: Number of games to simulate
        seed: Random seed for reproducibility
        greedy: If True, use greedy actions; else sample from policy

    Returns:
        (points, records):
            points: (n_games,) team 0's final points
            records: List of GameRecord with full game history
    """
    device = model.device

    # Set up RNG
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # Deal hands
    hands = deal_random_hands(bidder_hand, n_games, device, rng)

    # Store hands in CPU for later
    hands_cpu = hands.cpu().tolist()

    # Initialize game state
    state = BatchedGameState(hands, decl_id, device)

    # Warmup model
    model.warmup(n_games)

    # Initialize history tracking
    # For each game: list of (trick_idx, position_in_trick, player, local_idx)
    play_history: List[List[tuple[int, int, int, int]]] = [[] for _ in range(n_games)]

    game_idx_t = torch.arange(n_games, device=device)

    # Run all 28 steps
    for step in range(28):
        trick_idx = step // 4
        pos_in_trick = step % 4

        # Get current state
        active_mask = ~state.is_game_over()
        current = state.current_player()

        # Build tokens
        tokens, mask, current_player = state.build_tokens()

        # Get legal mask
        legal_mask = state.get_legal_mask()

        # Get actions (greedy or sampled)
        if greedy:
            actions = model.greedy_actions(tokens, mask, current_player, legal_mask)
        else:
            actions = model.sample_actions(tokens, mask, current_player, legal_mask)

        # Record plays before stepping (only for active games)
        active_games = game_idx_t[active_mask].cpu().tolist()
        current_players = current[active_mask].cpu().tolist()
        actions_taken = actions[active_mask].cpu().tolist()

        for i, g in enumerate(active_games):
            play_history[g].append((trick_idx, pos_in_trick, current_players[i], actions_taken[i]))

        # Execute actions
        state.step(actions, active_mask)

    # Build GameRecords from history
    records = []

    # Get final points
    final_points = state.team_points[:, 0].cpu()

    # Get trick winner info - we need to reconstruct this from play history
    # This is tricky because we don't track winners during simulation
    # Let's do a post-hoc reconstruction

    count_points_t = _get_table("COUNT_POINTS", device)
    led_suit_t = _get_table("LED_SUIT", device)
    trick_rank_t = _get_table("TRICK_RANK", device)

    for g in range(n_games):
        game_hands = hands_cpu[g]  # (4, 7)
        tricks: List[TrickRecord] = []

        # Group plays by trick
        plays_by_trick: List[List[tuple[int, int, int]]] = [[] for _ in range(7)]
        for trick_idx, pos, player, local_idx in play_history[g]:
            global_dom_id = game_hands[player][local_idx]
            plays_by_trick[trick_idx].append((player, local_idx, global_dom_id))

        # Reconstruct each trick
        current_leader = 0  # Player 0 leads first trick

        for trick_idx, trick_plays in enumerate(plays_by_trick):
            if not trick_plays:
                break  # Game ended early (shouldn't happen in normal play)

            # Calculate trick winner
            lead_dom = trick_plays[0][2]  # global dom id of lead
            led_suit = led_suit_t[lead_dom, decl_id].item()

            best_rank = -1
            winner = current_leader
            total_points = 1  # Base trick point

            for player, local_idx, dom_id in trick_plays:
                rank = trick_rank_t[dom_id, led_suit, decl_id].item()
                if rank > best_rank:
                    best_rank = rank
                    winner = player
                total_points += count_points_t[dom_id].item()

            tricks.append(TrickRecord(
                leader=current_leader,
                plays=trick_plays,
                winner=winner,
                points=total_points,
            ))

            current_leader = winner

        records.append(GameRecord(
            game_idx=g,
            hands=[game_hands[p] for p in range(4)],
            tricks=tricks,
            final_score=final_points[g].item(),
        ))

    return final_points, records


def format_game_replay(record: GameRecord, decl_id: int, bidder_hand: List[int]) -> str:
    """Format a game for investigation output."""
    lines = []
    trump_name = DECL_ID_TO_NAME[decl_id]

    lines.append(f"=== Game {record.game_idx} ===")
    lines.append(f"Final score: {record.final_score} (Team 0)")
    lines.append(f"Trump: {trump_name}")
    lines.append("")

    # Show all hands
    lines.append("Hands:")
    for p in range(4):
        team = "Team 0" if p % 2 == 0 else "Team 1"
        prefix = "* " if p == 0 else "  "
        hand_str = ", ".join(format_domino(d) for d in record.hands[p])
        lines.append(f"{prefix}P{p} ({team}): {hand_str}")
    lines.append("")

    # Track team points
    team0_pts = 0
    team1_pts = 0

    # Show trick-by-trick replay
    lines.append("Trick-by-trick replay:")
    for t, trick in enumerate(record.tricks):
        trick_line_parts = []

        # Show leader
        trick_line_parts.append(f"T{t+1}: ")

        # Show each play
        play_strs = []
        for player, local_idx, dom_id in trick.plays:
            dom_str = format_domino(dom_id)
            play_strs.append(f"P{player}:{dom_str}")

        trick_line_parts.append(" -> ".join(play_strs))

        # Show winner and points
        winner_team = trick.winner % 2
        if winner_team == 0:
            team0_pts += trick.points
        else:
            team1_pts += trick.points

        winner_marker = "*" if winner_team == 0 else ""
        trick_line_parts.append(f"  => P{trick.winner} wins {trick.points}pts{winner_marker}")
        trick_line_parts.append(f" [T0:{team0_pts} T1:{team1_pts}]")

        lines.append("".join(trick_line_parts))

    # Highlight potential issues
    lines.append("")
    if record.final_score < 42:
        lines.append(f"ISSUE: Team 0 scored only {record.final_score}, needed 42")

        # Find tricks won by opponents
        lost_tricks = [t for t in record.tricks if t.winner % 2 == 1]
        if lost_tricks:
            lines.append(f"Lost {len(lost_tricks)} trick(s) to Team 1")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigate losing deals in bidding simulation"
    )
    parser.add_argument(
        "--hand",
        required=True,
        help='7 dominoes as "high-low" pairs, comma-separated (e.g., "6-6,6-5,6-4,6-2,6-1,6-0,2-2")',
    )
    parser.add_argument(
        "--trump",
        required=True,
        help="Trump declaration (e.g., sixes, doubles, notrump)",
    )
    parser.add_argument(
        "--below",
        type=int,
        default=42,
        help="Show games where Team 0 scored below this threshold (default: 42)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of simulations to run (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (uses default if not specified)",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=5,
        help="Maximum number of losing games to show in detail (default: 5)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample from policy distribution instead of greedy (introduces stochasticity)",
    )

    args = parser.parse_args()

    # Parse hand
    try:
        hand = parse_hand(args.hand)
    except ValueError as e:
        print(f"Error: Invalid hand: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse trump
    trump_key = args.trump.strip().lower()
    if trump_key not in DECL_NAME_TO_ID:
        valid = ", ".join(sorted(set(DECL_NAME_TO_ID.keys())))
        print(f"Error: Unknown trump '{args.trump}'. Valid: {valid}", file=sys.stderr)
        sys.exit(1)
    decl_id = DECL_NAME_TO_ID[trump_key]

    print(f"Hand: {format_hand(hand)}")
    print(f"Trump: {DECL_ID_TO_NAME[decl_id]}")
    print(f"Threshold: {args.below}")
    print(f"Samples: {args.samples}")
    print(f"Seed: {args.seed}")
    print(f"Mode: {'sampling' if args.sample else 'greedy'}")
    print()

    # Load model
    print("Loading model...")
    model = PolicyModel(checkpoint_path=args.checkpoint, compile_model=False)
    print(f"Model loaded on {model.device}")
    print()

    # Run simulation
    print("Running simulation with history capture...")
    points, records = simulate_with_history(
        model=model,
        bidder_hand=hand,
        decl_id=decl_id,
        n_games=args.samples,
        seed=args.seed,
        greedy=not args.sample,
    )

    # Analyze results
    points_list = points.tolist()
    below_threshold = [r for r in records if r.final_score < args.below]

    print(f"Results: {len(below_threshold)}/{args.samples} games below {args.below} points")
    print(f"Score distribution: min={min(points_list)}, max={max(points_list)}, mean={sum(points_list)/len(points_list):.1f}")
    print()

    if not below_threshold:
        print("No losing games found!")
        return

    # Sort by score (worst first)
    below_threshold.sort(key=lambda r: r.final_score)

    # Show detailed replays
    n_show = min(args.max_show, len(below_threshold))
    print(f"Showing {n_show} lowest-scoring games:")
    print()

    for record in below_threshold[:n_show]:
        print(format_game_replay(record, decl_id, hand))
        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
