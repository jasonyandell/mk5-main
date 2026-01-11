#!/usr/bin/env python3
"""E[Q] Training Data Debug Viewer.

Interactive console viewer for human evaluation of training data.
Navigate with arrow keys to inspect decisions.

Usage:
    python -m forge.eq.viewer forge/data/eq_dataset.pt
    python -m forge.eq.viewer forge/data/eq_dataset.pt --game 42
    python -m forge.eq.viewer forge/data/eq_dataset.pt --example 1234
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor

from forge.eq.transcript_tokenize import (
    FEAT_DECL,
    FEAT_HIGH,
    FEAT_IS_IN_HAND,
    FEAT_LOW,
    FEAT_PLAYER,
    FEAT_TOKEN_TYPE,
    TOKEN_TYPE_HAND,
    TOKEN_TYPE_PLAY,
)
# Declaration names for display
DECLARATION_NAMES = {
    0: "Blanks",
    1: "Ones",
    2: "Twos",
    3: "Threes",
    4: "Fours",
    5: "Fives",
    6: "Sixes",
    7: "Doubles",
    8: "Doubles (suit)",
    9: "No Trump",
}


def domino_str(high: int, low: int) -> str:
    """Format domino as [H:L]."""
    return f"[{high}:{low}]"


def decode_transcript(tokens: Tensor, length: int) -> dict:
    """Decode transcript tokens back to human-readable form.

    Args:
        tokens: (seq_len, n_features) tensor
        length: Actual sequence length (rest is padding)

    Returns:
        Dict with keys: decl_id, hand, plays
    """
    tokens = tokens[:length]

    # First token is declaration
    decl_id = tokens[0, FEAT_DECL].item()

    # Extract hand tokens (TOKEN_TYPE_HAND)
    hand = []
    for i in range(length):
        if tokens[i, FEAT_TOKEN_TYPE] == TOKEN_TYPE_HAND:
            high = tokens[i, FEAT_HIGH].item()
            low = tokens[i, FEAT_LOW].item()
            hand.append((high, low))

    # Extract play tokens (TOKEN_TYPE_PLAY)
    plays = []
    for i in range(length):
        if tokens[i, FEAT_TOKEN_TYPE] == TOKEN_TYPE_PLAY:
            high = tokens[i, FEAT_HIGH].item()
            low = tokens[i, FEAT_LOW].item()
            player = tokens[i, FEAT_PLAYER].item()  # Relative player ID
            plays.append((player, high, low))

    return {"decl_id": decl_id, "hand": hand, "plays": plays}


def plays_to_tricks(plays: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    """Group plays into tricks of 4."""
    tricks = []
    for i in range(0, len(plays), 4):
        trick = plays[i : i + 4]
        if trick:
            tricks.append(trick)
    return tricks


def player_name(rel_player: int) -> str:
    """Convert relative player ID to name."""
    names = ["ME", "L", "P", "R"]  # Me, Left, Partner, Right
    return names[rel_player]


def render_decision(
    idx: int,
    total: int,
    tokens: Tensor,
    length: int,
    e_logits: Tensor,
    legal_mask: Tensor,
    action_taken: int,
    game_idx: int,
    decision_idx: int,
) -> str:
    """Render a single decision to a string."""
    lines = []

    # Decode transcript
    decoded = decode_transcript(tokens, length)
    decl_id = decoded["decl_id"]
    hand = decoded["hand"]
    plays = decoded["plays"]
    tricks = plays_to_tricks(plays)

    # Header
    trick_num = len(tricks) + (1 if len(plays) % 4 != 0 else 0)
    pos_in_trick = len(plays) % 4
    lines.append(f"Example {idx + 1}/{total}  |  Game {game_idx}  |  Decision {decision_idx}/28")
    lines.append("─" * 60)

    # Declaration
    decl_name = DECLARATION_NAMES.get(decl_id, f"Decl {decl_id}")
    lines.append(f"Declaration: {decl_name} (id={decl_id})")
    lines.append("")

    # Hand
    hand_strs = []
    for i, (h, l) in enumerate(hand):
        dom = domino_str(h, l)
        if i == action_taken:
            dom = f">{dom}<"  # Highlight selected
        if legal_mask[i]:
            hand_strs.append(dom)
        else:
            hand_strs.append(f"({h}:{l})")  # Illegal shown in parens
    lines.append(f"My Hand: {' '.join(hand_strs)}  ({len(hand)} remaining)")
    lines.append("")

    # Completed tricks
    if len(tricks) > 1 or (len(tricks) == 1 and len(tricks[0]) == 4):
        lines.append("Trick History:")
        for t_idx, trick in enumerate(tricks):
            if len(trick) == 4:  # Complete trick
                trick_str = " → ".join(
                    f"{player_name(p)} {domino_str(h, l)}" for p, h, l in trick
                )
                lines.append(f"  T{t_idx + 1}: {trick_str}")
        lines.append("")

    # Current trick (incomplete)
    current_trick = plays[-(len(plays) % 4) :] if len(plays) % 4 != 0 else []
    if current_trick:
        trick_str = " → ".join(
            f"{player_name(p)} {domino_str(h, l)}" for p, h, l in current_trick
        )
        lines.append(f"Current Trick: {trick_str} → ME: ?")
    else:
        lines.append("Leading trick...")
    lines.append("")

    # E[Q] logits as bar chart
    probs = torch.softmax(e_logits, dim=0)
    lines.append("E[Q] (softmax):")

    # Sort by probability for display, but only show hand slots
    sorted_indices = torch.argsort(probs, descending=True)
    max_prob = probs[:len(hand)].max().item() if hand else 0.01

    for i in sorted_indices:
        i = i.item()
        if i >= len(hand):
            continue  # Skip empty slots (played dominoes)

        h, l = hand[i]
        dom = domino_str(h, l)

        prob = probs[i].item()
        bar_len = int(20 * prob / max_prob) if max_prob > 0 else 0
        bar = "█" * bar_len

        marker = " ← SELECTED" if i == action_taken else ""
        legal = "" if legal_mask[i] else " (illegal)"
        lines.append(f"  {dom:>7}  {prob:.2f} {bar}{marker}{legal}")

    lines.append("")
    lines.append("[←] Prev  [→] Next  [j] Jump  [q] Quit")

    return "\n".join(lines)


def run_viewer(dataset_path: str, start_example: int = 0):
    """Run interactive viewer."""
    # Load dataset
    print(f"Loading {dataset_path}...", end=" ", flush=True)
    data = torch.load(dataset_path, weights_only=False)
    print("done")

    tokens = data["transcript_tokens"]
    lengths = data["transcript_lengths"]
    e_logits = data["e_logits"]
    legal_mask = data["legal_mask"]
    action_taken = data["action_taken"]
    game_idx = data["game_idx"]
    decision_idx = data["decision_idx"]

    total = len(tokens)
    current = start_example

    try:
        import curses

        def main(stdscr):
            nonlocal current
            curses.curs_set(0)  # Hide cursor

            while True:
                stdscr.clear()

                # Render current decision
                text = render_decision(
                    current,
                    total,
                    tokens[current],
                    lengths[current].item(),
                    e_logits[current],
                    legal_mask[current],
                    action_taken[current].item(),
                    game_idx[current].item(),
                    decision_idx[current].item(),
                )

                # Display
                for i, line in enumerate(text.split("\n")):
                    try:
                        stdscr.addstr(i, 0, line)
                    except curses.error:
                        pass  # Line too long or off screen

                stdscr.refresh()

                # Handle input
                key = stdscr.getch()
                if key == ord("q"):
                    break
                elif key == curses.KEY_LEFT and current > 0:
                    current -= 1
                elif key == curses.KEY_RIGHT and current < total - 1:
                    current += 1
                elif key == ord("j"):
                    # Jump to example
                    curses.echo()
                    stdscr.addstr(20, 0, "Jump to example: ")
                    stdscr.refresh()
                    try:
                        num_str = stdscr.getstr(20, 18, 10).decode()
                        num = int(num_str)
                        if 0 <= num < total:
                            current = num
                    except (ValueError, curses.error):
                        pass
                    curses.noecho()

        curses.wrapper(main)

    except ImportError:
        # Fallback without curses
        print("curses not available, using simple mode")
        while True:
            print("\n" * 2)
            print(
                render_decision(
                    current,
                    total,
                    tokens[current],
                    lengths[current].item(),
                    e_logits[current],
                    legal_mask[current],
                    action_taken[current].item(),
                    game_idx[current].item(),
                    decision_idx[current].item(),
                )
            )

            cmd = input("\nCommand (n=next, p=prev, j=jump, q=quit): ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "n" and current < total - 1:
                current += 1
            elif cmd == "p" and current > 0:
                current -= 1
            elif cmd == "j":
                try:
                    num = int(input("Jump to: "))
                    if 0 <= num < total:
                        current = num
                except ValueError:
                    pass


def main():
    parser = argparse.ArgumentParser(description="E[Q] Training Data Viewer")
    parser.add_argument("dataset", type=str, help="Path to dataset .pt file")
    parser.add_argument("--example", type=int, default=0, help="Start at example N")
    parser.add_argument("--game", type=int, help="Start at game N (decision 0)")

    args = parser.parse_args()

    if not Path(args.dataset).exists():
        print(f"Error: {args.dataset} not found")
        sys.exit(1)

    start = args.example
    if args.game is not None:
        start = args.game * 28  # 28 decisions per game

    run_viewer(args.dataset, start_example=start)


if __name__ == "__main__":
    main()
