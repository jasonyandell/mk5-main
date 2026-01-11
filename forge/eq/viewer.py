#!/usr/bin/env python3
"""E[Q] Training Data Debug Viewer.

Interactive console viewer with fast browsing and on-demand debug mode.

Modes:
  - Default: Fast navigation through examples with E[Q] display
  - Debug (d): Oracle comparison, world breakdown (loads oracle on first use)
  - World (w): Inspect individual sampled worlds

Usage:
    python -m forge.eq.viewer forge/data/eq_dataset.pt
    python -m forge.eq.viewer forge/data/eq_dataset.pt --game 42
    python -m forge.eq.viewer forge/data/eq_dataset.pt --example 1234
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from forge.eq.transcript_tokenize import (
    FEAT_DECL,
    FEAT_HIGH,
    FEAT_LOW,
    FEAT_PLAYER,
    FEAT_TOKEN_TYPE,
    TOKEN_TYPE_HAND,
    TOKEN_TYPE_PLAY,
)
from forge.oracle.tables import (
    can_follow,
    led_suit_for_lead_domino,
    trick_rank,
)

if TYPE_CHECKING:
    from forge.eq.oracle import Stage1Oracle

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

# Suit names (0-6 = pips, 7 = called suit)
SUIT_NAMES = {0: "0s", 1: "1s", 2: "2s", 3: "3s", 4: "4s", 5: "5s", 6: "6s", 7: "trump"}


def domino_str(high: int, low: int) -> str:
    """Format domino as [H:L]."""
    return f"[{high}:{low}]"


def domino_id_to_pips(domino_id: int) -> tuple[int, int]:
    """Convert domino ID (0-27) to (high, low) pips."""
    # Standard ordering: 0-0, 0-1, 0-2, ..., 1-1, 1-2, ..., 6-6
    high = 0
    idx = 0
    while idx + (7 - high) <= domino_id:
        idx += 7 - high
        high += 1
    low = high + (domino_id - idx)
    return high, low


def pips_to_domino_id(high: int, low: int) -> int:
    """Convert (high, low) pips to domino ID."""
    if high > low:
        high, low = low, high
    # Count dominoes before this high value
    idx = sum(7 - h for h in range(high))
    return idx + (low - high)


def player_name(rel_player: int) -> str:
    """Convert relative player ID to name."""
    names = ["ME", "L", "P", "R"]  # Me, Left, Partner, Right
    return names[rel_player]


def player_name_long(rel_player: int) -> str:
    """Convert relative player ID to long name."""
    names = ["ME", "LEFT (L)", "PARTNER (P)", "RIGHT (R)"]
    return names[rel_player]


def decode_transcript(tokens: Tensor, length: int) -> dict:
    """Decode transcript tokens back to human-readable form.

    Returns:
        Dict with keys: decl_id, hand, plays
        - hand: list of (high, low) tuples
        - plays: list of (rel_player, high, low) tuples
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


def determine_trick_winner(
    trick: list[tuple[int, int, int]], decl_id: int
) -> tuple[int, tuple[int, int]]:
    """Determine winner of a complete trick.

    Args:
        trick: List of (rel_player, high, low) tuples
        decl_id: Declaration ID

    Returns:
        (winner_rel_player, (winning_high, winning_low))
    """
    if len(trick) != 4:
        raise ValueError("Trick must have 4 plays")

    lead_player, lead_high, lead_low = trick[0]
    lead_domino_id = pips_to_domino_id(lead_high, lead_low)
    led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)

    best_idx = 0
    best_rank = trick_rank(lead_domino_id, led_suit, decl_id)

    for i in range(1, 4):
        _, h, l = trick[i]
        domino_id = pips_to_domino_id(h, l)
        rank = trick_rank(domino_id, led_suit, decl_id)
        if rank > best_rank:
            best_idx = i
            best_rank = rank

    winner = trick[best_idx]
    return winner[0], (winner[1], winner[2])


def find_current_winner(
    current_trick: list[tuple[int, int, int]], decl_id: int
) -> tuple[int, tuple[int, int]] | None:
    """Find who is currently winning an incomplete trick.

    Returns:
        (winner_rel_player, (high, low)) or None if trick is empty
    """
    if not current_trick:
        return None

    lead_player, lead_high, lead_low = current_trick[0]
    lead_domino_id = pips_to_domino_id(lead_high, lead_low)
    led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)

    best_idx = 0
    best_rank = trick_rank(lead_domino_id, led_suit, decl_id)

    for i in range(1, len(current_trick)):
        _, h, l = current_trick[i]
        domino_id = pips_to_domino_id(h, l)
        rank = trick_rank(domino_id, led_suit, decl_id)
        if rank > best_rank:
            best_idx = i
            best_rank = rank

    winner = current_trick[best_idx]
    return winner[0], (winner[1], winner[2])


def infer_voids_from_plays(
    plays: list[tuple[int, int, int]], decl_id: int
) -> dict[int, set[int]]:
    """Infer voids from play history.

    Returns:
        {rel_player: {void_suits}}
    """
    voids: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}

    tricks = plays_to_tricks(plays)
    for trick in tricks:
        if len(trick) < 2:
            continue

        lead_player, lead_high, lead_low = trick[0]
        lead_domino_id = pips_to_domino_id(lead_high, lead_low)
        led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)

        for rel_player, h, l in trick[1:]:
            domino_id = pips_to_domino_id(h, l)
            if not can_follow(domino_id, led_suit, decl_id):
                voids[rel_player].add(led_suit)

    return voids


# =============================================================================
# View state management
# =============================================================================


@dataclass
class ViewState:
    """Current viewer state."""

    current_idx: int = 0
    mode: str = "default"  # "default", "debug", "world"
    debug_data: dict | None = None  # Cached debug computation
    world_idx: int = 0  # Which flagged world we're viewing
    flagged_worlds: list[int] | None = None  # Indices of worlds preferring different action


# =============================================================================
# Default mode rendering
# =============================================================================


def render_default_mode(
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
    """Render default (fast) view."""
    lines = []

    # Decode transcript
    decoded = decode_transcript(tokens, length)
    decl_id = decoded["decl_id"]
    hand = decoded["hand"]
    plays = decoded["plays"]
    tricks = plays_to_tricks(plays)

    # Calculate trick number
    complete_tricks = sum(1 for t in tricks if len(t) == 4)
    trick_num = complete_tricks + 1

    # Header
    lines.append(f"Example {idx + 1}/{total}  |  Game {game_idx}  |  Decision {decision_idx + 1}/28")
    lines.append("═" * 80)
    lines.append("")

    # Declaration and trick info
    decl_name = DECLARATION_NAMES.get(decl_id, f"Decl {decl_id}")
    lines.append(f"Declaration: {decl_name}".ljust(50) + f"Trick {trick_num} of 7")
    lines.append("─" * 80)

    # Trick history (completed tricks only)
    completed_tricks = [t for t in tricks if len(t) == 4]
    if completed_tricks:
        lines.append("Trick History:")
        for t_idx, trick in enumerate(completed_tricks):
            winner_player, winner_dom = determine_trick_winner(trick, decl_id)
            trick_str = " → ".join(
                f"{player_name(p)} {domino_str(h, l)}" for p, h, l in trick
            )
            lines.append(f"  T{t_idx + 1}: {trick_str}  |  Won: {player_name(winner_player)}")
        lines.append("")

    # Current trick (incomplete)
    current_trick = plays[-(len(plays) % 4) :] if len(plays) % 4 != 0 else []
    if current_trick:
        trick_str = " → ".join(
            f"{player_name(p)} {domino_str(h, l)}" for p, h, l in current_trick
        )
        winner = find_current_winner(current_trick, decl_id)
        if winner:
            winner_player, (wh, wl) = winner
            winning_str = f"Winning: {player_name(winner_player)} {domino_str(wh, wl)}"
        else:
            winning_str = ""
        lines.append(f"Current Trick: {trick_str} → ME: ?".ljust(50) + f"|  {winning_str}")
    else:
        lines.append("Leading trick...")
    lines.append("─" * 80)

    # Hand display with selected action highlighted
    hand_strs = []
    for i, (h, l) in enumerate(hand):
        dom = domino_str(h, l)
        if i == action_taken:
            dom = f">{dom}<"  # Highlight selected
        hand_strs.append(dom)
    lines.append(f"My Hand: {' '.join(hand_strs)}   ({len(hand)} remaining)")

    # Voids
    voids = infer_voids_from_plays(plays, decl_id)
    void_strs = []
    for player in range(4):
        if voids[player]:
            suits = ",".join(SUIT_NAMES[s] for s in sorted(voids[player]))
            void_strs.append(f"{player_name(player)}→{suits}")
    if void_strs:
        lines.append(f"Voids: {' | '.join(void_strs)}")

    lines.append("─" * 80)

    # E[Q] display (legal actions only)
    legal_indices = [i for i in range(len(hand)) if legal_mask[i]]
    if legal_indices:
        legal_logits = e_logits[legal_indices]
        legal_probs = torch.softmax(legal_logits, dim=0)
        max_prob = legal_probs.max().item()

        # Sort by probability
        sorted_legal = sorted(
            zip(legal_indices, legal_probs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        lines.append("E[Q] (softmax over legal):")
        for i, prob in sorted_legal:
            h, l = hand[i]
            dom = domino_str(h, l)
            bar_len = int(20 * prob / max_prob) if max_prob > 0 else 0
            bar = "█" * bar_len
            marker = " ← SELECTED" if i == action_taken else ""
            lines.append(f"    {dom:>7}  {prob:.2f} {bar}{marker}")

    lines.append("─" * 80)
    lines.append("[←/→] Nav  [j] Jump  [d] Debug details  [q] Quit")

    return "\n".join(lines)


# =============================================================================
# Debug mode
# =============================================================================


def compute_debug_data(
    oracle: Stage1Oracle,
    tokens: Tensor,
    length: int,
    e_logits: Tensor,
    legal_mask: Tensor,
    action_taken: int,
    n_samples: int = 100,
) -> dict:
    """Compute debug data: oracle comparison and world breakdown.

    This is expensive - only computed when entering debug mode.
    """
    from forge.eq.game import GameState
    from forge.eq.sampling import sample_consistent_worlds
    from forge.eq.voids import infer_voids

    # Decode transcript to reconstruct game state
    decoded = decode_transcript(tokens, length)
    decl_id = decoded["decl_id"]
    hand = decoded["hand"]  # List of (high, low) tuples
    plays = decoded["plays"]  # List of (rel_player, high, low)

    # Convert hand to domino IDs
    my_hand_ids = [pips_to_domino_id(h, l) for h, l in hand]

    # Reconstruct played dominoes
    played_ids = set()
    play_history = []  # For void inference
    for rel_player, h, l in plays:
        domino_id = pips_to_domino_id(h, l)
        played_ids.add(domino_id)

        # Determine lead domino for this play
        trick_pos = len(play_history) % 4
        if trick_pos == 0:
            lead_domino_id = domino_id
        else:
            # Find lead domino of current trick
            trick_start = len(play_history) - trick_pos
            lead_domino_id = play_history[trick_start][1]

        play_history.append((rel_player, domino_id, lead_domino_id))

    # Infer voids
    voids = infer_voids(play_history, decl_id)

    # Hand sizes (7 - plays per player)
    plays_per_player = [0, 0, 0, 0]
    for p, _, _ in plays:
        plays_per_player[p] += 1
    hand_sizes = [7 - plays_per_player[p] for p in range(4)]

    # Sample consistent worlds
    rng = np.random.default_rng(42)  # Deterministic for reproducibility
    try:
        worlds = sample_consistent_worlds(
            my_player=0,  # ME is always player 0 in relative coords
            my_hand=my_hand_ids,
            played=played_ids,
            hand_sizes=hand_sizes,
            voids=voids,
            decl_id=decl_id,
            n_samples=n_samples,
            rng=rng,
        )
    except RuntimeError as e:
        return {"error": str(e), "worlds": [], "world_logits": []}

    # Query oracle for each world
    # We need to build proper initial deals for oracle
    # For simplicity, use worlds as initial hands (pretending no plays yet)
    # This is a simplification - real implementation would need full game reconstruction

    # Build game state info for oracle query
    # Current trick from plays
    trick_pos = len(plays) % 4
    current_trick_plays = plays[-trick_pos:] if trick_pos > 0 else []

    # For oracle, we need (player, local_idx) tuples
    # This is tricky since we don't have the original deal
    # For now, use a simplified approach: query with world as initial deal

    world_logits = []
    world_prefs = []  # Which action each world prefers

    for world in worlds:
        # Build remaining bitmask (all dominoes in hand are remaining for this query)
        remaining = np.zeros((1, 4), dtype=np.int32)
        for player_idx in range(4):
            for local_idx in range(len(world[player_idx])):
                remaining[0, player_idx] |= 1 << local_idx

        game_state_info = {
            "decl_id": decl_id,
            "leader": 0,  # Simplified
            "trick_plays": [],  # Simplified - no current trick context
            "remaining": remaining,
        }

        try:
            logits = oracle.query_batch([world], game_state_info, current_player=0)
            logits = logits[0]  # (7,)
            world_logits.append(logits.cpu())

            # Which legal action does this world prefer?
            masked = logits.clone()
            for i in range(len(my_hand_ids)):
                if i >= len(legal_mask) or not legal_mask[i]:
                    masked[i] = float("-inf")
            pref = masked.argmax().item()
            world_prefs.append(pref)
        except Exception as e:
            world_logits.append(None)
            world_prefs.append(-1)

    # Average oracle logits
    valid_logits = [wl for wl in world_logits if wl is not None]
    if valid_logits:
        oracle_avg = torch.stack(valid_logits).mean(dim=0)
    else:
        oracle_avg = torch.zeros(7)

    # Compute agreement stats
    e_probs = torch.softmax(e_logits[: len(my_hand_ids)], dim=0)
    oracle_probs = torch.softmax(oracle_avg, dim=0)

    # Rank oracle's preferences
    oracle_ranking = torch.argsort(oracle_avg, descending=True).tolist()

    # Find flagged worlds (prefer different action than E[Q] selected)
    flagged = [i for i, pref in enumerate(world_prefs) if pref != action_taken and pref >= 0]

    # Count worlds preferring each action
    pref_counts = {}
    for pref in world_prefs:
        if pref >= 0:
            pref_counts[pref] = pref_counts.get(pref, 0) + 1

    return {
        "worlds": worlds,
        "world_logits": world_logits,
        "world_prefs": world_prefs,
        "oracle_avg": oracle_avg,
        "oracle_ranking": oracle_ranking,
        "flagged_worlds": flagged,
        "pref_counts": pref_counts,
        "hand": hand,
        "my_hand_ids": my_hand_ids,
        "decl_id": decl_id,
    }


def render_debug_mode(
    idx: int,
    total: int,
    tokens: Tensor,
    length: int,
    e_logits: Tensor,
    legal_mask: Tensor,
    action_taken: int,
    game_idx: int,
    decision_idx: int,
    debug_data: dict,
) -> str:
    """Render debug mode with oracle comparison."""
    lines = []

    decoded = decode_transcript(tokens, length)
    decl_id = decoded["decl_id"]
    hand = decoded["hand"]
    plays = decoded["plays"]
    tricks = plays_to_tricks(plays)

    complete_tricks = sum(1 for t in tricks if len(t) == 4)
    trick_num = complete_tricks + 1

    # Header with DEBUG MODE indicator
    lines.append(
        f"Example {idx + 1}/{total}  |  Game {game_idx}  |  Decision {decision_idx + 1}/28".ljust(60)
        + "[DEBUG MODE]"
    )
    lines.append("═" * 80)

    decl_name = DECLARATION_NAMES.get(decl_id, f"Decl {decl_id}")
    lines.append(f"Declaration: {decl_name}".ljust(50) + f"Trick {trick_num} of 7")
    lines.append("─" * 80)

    # Check for error
    if "error" in debug_data:
        lines.append(f"ERROR: {debug_data['error']}")
        lines.append("─" * 80)
        lines.append("[←/→] Nav  [j] Jump  [d] Exit debug  [q] Quit")
        return "\n".join(lines)

    # 4-player hand layout (compass style)
    worlds = debug_data.get("worlds", [])
    if worlds:
        # Use first world for opponent hands
        world = worlds[0]
        lines.append("                        PARTNER (P)")
        partner_hand = " ".join(domino_str(*domino_id_to_pips(d)) for d in sorted(world[2]))
        lines.append(f"                     {partner_hand}")
        lines.append("")

        left_hand = [domino_str(*domino_id_to_pips(d)) for d in sorted(world[1])]
        right_hand = [domino_str(*domino_id_to_pips(d)) for d in sorted(world[3])]

        # Side by side
        left_str = " ".join(left_hand[:3])
        right_str = " ".join(right_hand[:3])
        lines.append(f"     LEFT (L)                                    RIGHT (R)")
        lines.append(f"     {left_str:<30}              {right_str}")
        if len(left_hand) > 3 or len(right_hand) > 3:
            left_str2 = " ".join(left_hand[3:])
            right_str2 = " ".join(right_hand[3:])
            lines.append(f"     {left_str2:<30}              {right_str2}")

        lines.append("")
        lines.append("                          ME (Team 0)")
        hand_strs = []
        for i, (h, l) in enumerate(hand):
            dom = domino_str(h, l)
            if i == action_taken:
                dom = f">{dom}<"
            hand_strs.append(dom)
        lines.append(f"                  {' '.join(hand_strs)}")
        lines.append("")

    lines.append("─" * 80)

    # Current trick and sampling info
    current_trick = plays[-(len(plays) % 4) :] if len(plays) % 4 != 0 else []
    if current_trick:
        trick_str = " → ".join(f"{player_name(p)} {domino_str(h, l)}" for p, h, l in current_trick)
        winner = find_current_winner(current_trick, decl_id)
        if winner:
            winner_player, (wh, wl) = winner
            lines.append(f"Current Trick: {trick_str} → ME: ?      |  Winning: {player_name(winner_player)} {domino_str(wh, wl)}")
        else:
            lines.append(f"Current Trick: {trick_str} → ME: ?")
    else:
        lines.append("Leading trick...")

    # Voids and sampling info
    voids = infer_voids_from_plays(plays, decl_id)
    void_strs = []
    for player in range(4):
        if voids[player]:
            suits = ",".join(SUIT_NAMES[s] for s in sorted(voids[player]))
            void_strs.append(f"{player_name(player)}→{suits}")
    voids_str = " | ".join(void_strs) if void_strs else "(none)"
    n_worlds = len(worlds)
    lines.append(f"Voids: {voids_str}  |  Sampling: {n_worlds}/{n_worlds} worlds")

    lines.append("─" * 80)

    # E[Q] vs Oracle comparison table
    oracle_avg = debug_data.get("oracle_avg", torch.zeros(7))
    oracle_ranking = debug_data.get("oracle_ranking", list(range(7)))

    # Header
    lines.append("                    E[Q]      Oracle     Δ       Rank")
    lines.append(f"                   (N={n_worlds})   (actual)")

    # Get legal indices
    legal_indices = [i for i in range(len(hand)) if i < len(legal_mask) and legal_mask[i]]

    # Compute probabilities
    e_probs = torch.softmax(e_logits[: len(hand)], dim=0)
    oracle_probs = torch.softmax(oracle_avg[: len(hand)], dim=0)

    # Sort by E[Q] probability (descending)
    sorted_by_eq = sorted(
        [(i, e_probs[i].item(), oracle_probs[i].item()) for i in legal_indices],
        key=lambda x: x[1],
        reverse=True,
    )

    for i, eq_prob, oracle_prob in sorted_by_eq:
        h, l = hand[i]
        dom = domino_str(h, l)
        delta = eq_prob - oracle_prob
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"

        # Find oracle rank
        try:
            oracle_rank = oracle_ranking.index(i) + 1
        except ValueError:
            oracle_rank = "?"

        marker = " ← SELECTED" if i == action_taken else ""
        lines.append(f"    {dom:>7}       {eq_prob:.2f}       {oracle_prob:.2f}    {delta_str:>6}       {oracle_rank}{marker}")

    lines.append("")

    # Agreement status
    e_best = max(legal_indices, key=lambda i: e_logits[i].item())
    oracle_best = max(legal_indices, key=lambda i: oracle_avg[i].item())
    agreement = "YES" if e_best == oracle_best else "NO"

    # Find oracle rank of selected action
    try:
        selected_oracle_rank = oracle_ranking.index(action_taken) + 1
    except ValueError:
        selected_oracle_rank = "?"

    # Q-gap
    if len(legal_indices) >= 2:
        sorted_oracle = sorted([oracle_avg[i].item() for i in legal_indices], reverse=True)
        q_gap = sorted_oracle[0] - sorted_oracle[1]
    else:
        q_gap = 0

    lines.append(
        f"Agreement: {agreement} (picked oracle rank {selected_oracle_rank})    Oracle Q-gap: {q_gap:.2f}"
    )

    lines.append("─" * 80)

    # World breakdown
    flagged = debug_data.get("flagged_worlds", [])
    world_prefs = debug_data.get("world_prefs", [])
    world_logits = debug_data.get("world_logits", [])
    pref_counts = debug_data.get("pref_counts", {})

    # Count worlds preferring selected action
    selected_count = pref_counts.get(action_taken, 0)
    total_valid = sum(pref_counts.values())

    # Show sample of worlds
    lines.append(f"World Breakdown (5 of {n_worlds}):".ljust(50) + f"Worlds preferring {domino_str(*hand[action_taken])}:")
    sample_worlds = list(range(min(5, n_worlds)))

    for wi in sample_worlds:
        if wi >= len(world_logits) or world_logits[wi] is None:
            continue

        wl = world_logits[wi]
        # Top 3 actions for this world
        top3 = torch.argsort(wl[: len(hand)], descending=True)[:3].tolist()
        probs = torch.softmax(wl[: len(hand)], dim=0)

        top_strs = []
        for ti in top3:
            if ti < len(hand):
                h, l = hand[ti]
                p = probs[ti].item()
                top_strs.append(f"{h}:{l}={p:.2f}")

        pref = world_prefs[wi] if wi < len(world_prefs) else -1
        best_str = domino_str(*hand[pref]) if 0 <= pref < len(hand) else "?"
        flag = " ⚠️" if wi in flagged else ""
        lines.append(f"  W{wi + 1}: {' '.join(top_strs)} → {best_str}{flag}")

    # Summary
    flag_pct = len(flagged) / total_valid * 100 if total_valid > 0 else 0
    warning = " ⚠️" if flag_pct > 30 else ""
    lines.append(f"".ljust(50) + f"  {selected_count}/{total_valid}{warning}")

    lines.append("─" * 80)
    lines.append("[←/→] Nav  [j] Jump  [w] Inspect flagged world  [d] Exit debug  [q] Quit")

    return "\n".join(lines)


# =============================================================================
# World inspection mode
# =============================================================================


def render_world_mode(
    idx: int,
    total: int,
    tokens: Tensor,
    length: int,
    e_logits: Tensor,
    legal_mask: Tensor,
    action_taken: int,
    game_idx: int,
    decision_idx: int,
    debug_data: dict,
    world_idx: int,
) -> str:
    """Render world inspection mode."""
    lines = []

    decoded = decode_transcript(tokens, length)
    decl_id = decoded["decl_id"]
    hand = decoded["hand"]

    flagged = debug_data.get("flagged_worlds", [])
    worlds = debug_data.get("worlds", [])
    world_logits = debug_data.get("world_logits", [])
    world_prefs = debug_data.get("world_prefs", [])

    if not flagged:
        lines.append("No flagged worlds to inspect.")
        lines.append("")
        lines.append("[d] Back to debug  [q] Quit")
        return "\n".join(lines)

    # Get the actual world index from flagged list
    if world_idx >= len(flagged):
        world_idx = 0
    actual_wi = flagged[world_idx]

    pref = world_prefs[actual_wi] if actual_wi < len(world_prefs) else -1
    pref_str = f"prefers {domino_str(*hand[pref])}" if 0 <= pref < len(hand) else "unknown preference"

    # Header
    lines.append(
        f"Example {idx + 1}/{total}  |  Game {game_idx}  |  Decision {decision_idx + 1}/28".ljust(50)
        + f"[WORLD {world_idx + 1}/{len(flagged)} INSPECTION]"
    )
    lines.append("═" * 80)

    lines.append(f"Sampled World {actual_wi + 1} of {len(worlds)} (flagged: {pref_str})")
    lines.append("─" * 80)

    # Get this world's hands
    if actual_wi < len(worlds):
        world = worlds[actual_wi]

        # Compare with first world (reference) to show differences
        ref_world = worlds[0] if worlds else None

        # Partner
        lines.append("                        PARTNER (P)")
        partner_hand = []
        for d in sorted(world[2]):
            h, l = domino_id_to_pips(d)
            dom = domino_str(h, l)
            if ref_world and d not in ref_world[2]:
                dom = f"{dom}^"  # Mark difference
            partner_hand.append(dom)
        lines.append(f"                     {' '.join(partner_hand)}")

        # Check for differences
        if ref_world and set(world[2]) != set(ref_world[2]):
            lines.append("                                  ^ differs from actual")

        lines.append("")

        # Left and Right
        left_hand = []
        for d in sorted(world[1]):
            h, l = domino_id_to_pips(d)
            dom = domino_str(h, l)
            if ref_world and d not in ref_world[1]:
                dom = f"{dom}^"
            left_hand.append(dom)

        right_hand = []
        for d in sorted(world[3]):
            h, l = domino_id_to_pips(d)
            dom = domino_str(h, l)
            if ref_world and d not in ref_world[3]:
                dom = f"{dom}^"
            right_hand.append(dom)

        lines.append("     LEFT (L)                                    RIGHT (R)")
        lines.append(f"     {' '.join(left_hand[:3]):<30}              {' '.join(right_hand[:3])}")
        if len(left_hand) > 3 or len(right_hand) > 3:
            lines.append(f"     {' '.join(left_hand[3:]):<30}              {' '.join(right_hand[3:])}")

        # Difference notes
        diff_notes = []
        if ref_world and set(world[1]) != set(ref_world[1]):
            diff_notes.append("LEFT differs")
        if ref_world and set(world[3]) != set(ref_world[3]):
            diff_notes.append("RIGHT differs")
        if diff_notes:
            lines.append(f"         ^ {', '.join(diff_notes)}")

        lines.append("")
        lines.append("                          ME (Team 0)")
        hand_strs = []
        for i, (h, l) in enumerate(hand):
            dom = domino_str(h, l)
            if i == action_taken:
                dom = f">{dom}<"
            hand_strs.append(dom)
        lines.append(f"                  {' '.join(hand_strs)}")

    lines.append("")
    lines.append("─" * 80)

    # Oracle logits for this world
    if actual_wi < len(world_logits) and world_logits[actual_wi] is not None:
        wl = world_logits[actual_wi]
        probs = torch.softmax(wl[: len(hand)], dim=0)
        max_prob = probs.max().item()

        lines.append("Oracle logits for this world:")

        legal_indices = [i for i in range(len(hand)) if i < len(legal_mask) and legal_mask[i]]
        sorted_by_prob = sorted(
            [(i, probs[i].item()) for i in legal_indices],
            key=lambda x: x[1],
            reverse=True,
        )

        for i, prob in sorted_by_prob:
            h, l = hand[i]
            dom = domino_str(h, l)
            bar_len = int(36 * prob / max_prob) if max_prob > 0 else 0
            bar = "█" * bar_len
            marker = " ← best" if i == pref else ""
            lines.append(f"    {dom:>7}   {prob:.2f} {bar}{marker}")

        # Why does oracle prefer this?
        if pref != action_taken:
            lines.append("")
            lines.append(f"Why does oracle prefer {domino_str(*hand[pref])} here?")

    lines.append("─" * 80)

    # Void consistency check
    voids = infer_voids_from_plays(decoded["plays"], decl_id)
    lines.append("Void consistency check:")
    all_valid = True
    for player in [1, 2, 3]:  # LEFT, PARTNER, RIGHT
        player_voids = voids.get(player, set())
        if player_voids and actual_wi < len(worlds):
            player_hand = worlds[actual_wi][player]
            violations = []
            for suit in player_voids:
                # Check if any domino in hand could follow this suit
                for d in player_hand:
                    if can_follow(d, suit, decl_id):
                        violations.append(SUIT_NAMES[suit])
                        break
            if violations:
                lines.append(f"  ✗ {player_name(player)} has {', '.join(violations)} but should be void")
                all_valid = False
            else:
                lines.append(f"  ✓ {player_name(player)} not void in any revealed suit")
        else:
            lines.append(f"  ✓ {player_name(player)} not void in any revealed suit")

    lines.append("─" * 80)
    lines.append("[←/→] Prev/Next flagged world  [d] Back to debug  [q] Quit")

    return "\n".join(lines)


# =============================================================================
# Main viewer
# =============================================================================


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
    state = ViewState(current_idx=start_example)

    # Oracle loaded on demand
    oracle: Stage1Oracle | None = None

    try:
        import curses

        def main(stdscr):
            nonlocal oracle
            curses.curs_set(0)  # Hide cursor

            while True:
                stdscr.clear()

                # Get current example data
                idx = state.current_idx
                cur_tokens = tokens[idx]
                cur_length = lengths[idx].item()
                cur_e_logits = e_logits[idx]
                cur_legal_mask = legal_mask[idx]
                cur_action = action_taken[idx].item()
                cur_game = game_idx[idx].item()
                cur_decision = decision_idx[idx].item()

                # Render based on mode
                if state.mode == "default":
                    text = render_default_mode(
                        idx, total, cur_tokens, cur_length, cur_e_logits,
                        cur_legal_mask, cur_action, cur_game, cur_decision,
                    )
                elif state.mode == "debug":
                    if state.debug_data is None:
                        # Show loading message
                        stdscr.addstr(0, 0, "Loading oracle and computing debug data...")
                        stdscr.refresh()

                        # Load oracle if needed
                        if oracle is None:
                            from forge.eq.oracle import Stage1Oracle
                            checkpoint = "forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt"
                            oracle = Stage1Oracle(checkpoint, device="cuda")

                        # Compute debug data
                        state.debug_data = compute_debug_data(
                            oracle, cur_tokens, cur_length, cur_e_logits,
                            cur_legal_mask, cur_action,
                        )
                        state.flagged_worlds = state.debug_data.get("flagged_worlds", [])
                        state.world_idx = 0

                    text = render_debug_mode(
                        idx, total, cur_tokens, cur_length, cur_e_logits,
                        cur_legal_mask, cur_action, cur_game, cur_decision,
                        state.debug_data,
                    )
                elif state.mode == "world":
                    text = render_world_mode(
                        idx, total, cur_tokens, cur_length, cur_e_logits,
                        cur_legal_mask, cur_action, cur_game, cur_decision,
                        state.debug_data, state.world_idx,
                    )
                else:
                    text = "Unknown mode"

                # Display
                for i, line in enumerate(text.split("\n")):
                    try:
                        stdscr.addstr(i, 0, line[:curses.COLS - 1])
                    except curses.error:
                        pass

                stdscr.refresh()

                # Handle input
                key = stdscr.getch()

                if key == ord("q"):
                    break

                elif key == curses.KEY_LEFT:
                    if state.mode == "world" and state.flagged_worlds:
                        # Navigate flagged worlds
                        state.world_idx = (state.world_idx - 1) % len(state.flagged_worlds)
                    elif state.current_idx > 0:
                        state.current_idx -= 1
                        state.debug_data = None  # Clear cache

                elif key == curses.KEY_RIGHT:
                    if state.mode == "world" and state.flagged_worlds:
                        state.world_idx = (state.world_idx + 1) % len(state.flagged_worlds)
                    elif state.current_idx < total - 1:
                        state.current_idx += 1
                        state.debug_data = None

                elif key == ord("j"):
                    # Jump to example
                    curses.echo()
                    max_y, _ = stdscr.getmaxyx()
                    stdscr.addstr(max_y - 2, 0, "Jump to example: ")
                    stdscr.refresh()
                    try:
                        num_str = stdscr.getstr(max_y - 2, 18, 10).decode()
                        num = int(num_str)
                        if 0 <= num < total:
                            state.current_idx = num
                            state.debug_data = None
                            state.mode = "default"
                    except (ValueError, curses.error):
                        pass
                    curses.noecho()

                elif key == ord("d"):
                    # Toggle debug mode
                    if state.mode == "default":
                        state.mode = "debug"
                    elif state.mode in ("debug", "world"):
                        state.mode = "default"

                elif key == ord("w"):
                    # Enter world inspection (only from debug mode)
                    if state.mode == "debug" and state.flagged_worlds:
                        state.mode = "world"
                        state.world_idx = 0

        curses.wrapper(main)

    except ImportError:
        # Fallback without curses
        print("curses not available, using simple mode (default view only)")
        while True:
            print("\n" * 2)
            idx = state.current_idx
            print(
                render_default_mode(
                    idx, total, tokens[idx], lengths[idx].item(),
                    e_logits[idx], legal_mask[idx], action_taken[idx].item(),
                    game_idx[idx].item(), decision_idx[idx].item(),
                )
            )

            cmd = input("\nCommand (n=next, p=prev, j=jump, q=quit): ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "n" and state.current_idx < total - 1:
                state.current_idx += 1
            elif cmd == "p" and state.current_idx > 0:
                state.current_idx -= 1
            elif cmd == "j":
                try:
                    num = int(input("Jump to: "))
                    if 0 <= num < total:
                        state.current_idx = num
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
        # Find first example for this game
        data = torch.load(args.dataset, weights_only=False)
        game_indices = data["game_idx"]
        for i, gi in enumerate(game_indices):
            if gi.item() == args.game:
                start = i
                break

    run_viewer(args.dataset, start_example=start)


if __name__ == "__main__":
    main()
