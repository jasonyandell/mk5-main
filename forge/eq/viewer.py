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
    """Convert domino ID (0-27) to (high, low) pips.

    Uses triangular encoding to match oracle tables:
    ID = hi*(hi+1)/2 + lo where hi >= lo
    """
    # Find hi such that hi*(hi+1)/2 <= domino_id < (hi+1)*(hi+2)/2
    hi = 0
    while (hi + 1) * (hi + 2) // 2 <= domino_id:
        hi += 1
    lo = domino_id - hi * (hi + 1) // 2
    return hi, lo


def pips_to_domino_id(high: int, low: int) -> int:
    """Convert (high, low) pips to domino ID.

    Uses triangular encoding to match oracle tables:
    ID = hi*(hi+1)/2 + lo where hi >= lo
    """
    hi = max(high, low)
    lo = min(high, low)
    return hi * (hi + 1) // 2 + lo


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
    mode: str = "default"  # "default", "debug", "world", "diagnostics"
    debug_data: dict | None = None  # Cached debug computation
    world_idx: int = 0  # Which world we're viewing (0 to n_worlds-1)
    n_worlds: int = 0  # Total number of sampled worlds
    show_uniform: bool = False  # Toggle weighted vs uniform in debug mode (t42-64uj.7)


# Exploration mode names (t42-64uj.7)
EXPLORATION_MODE_NAMES = {0: "greedy", 1: "boltzmann", 2: "epsilon", 3: "blunder"}


# =============================================================================
# Default mode rendering
# =============================================================================


def render_default_mode(
    idx: int,
    total: int,
    tokens: Tensor,
    length: int,
    e_q_mean: Tensor,
    legal_mask: Tensor,
    action_taken: int,
    game_idx: int,
    decision_idx: int,
    e_q_var: Tensor | None = None,
    u_mean: float = 0.0,
    u_max: float = 0.0,
    # Posterior diagnostics (t42-64uj.7)
    ess: float | None = None,
    max_w: float | None = None,
    # Exploration stats (t42-64uj.7)
    exploration_mode: int | None = None,
    q_gap: float | None = None,
) -> str:
    """Render default (fast) view with optional uncertainty display.

    Args:
        e_q_mean: E[Q] values in POINTS (NOT logits - do not softmax)
        e_q_var: Var[Q] in points² (optional)
        ess: Effective sample size (optional, v2+ datasets)
        max_w: Max posterior weight (optional, v2+ datasets)
        exploration_mode: How this action was chosen (optional)
        q_gap: Regret vs greedy action (optional)
    """
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

    # E[Q] display (legal actions only) - show raw Q-values (expected points)
    legal_indices = [i for i in range(len(hand)) if legal_mask[i]]
    has_uncertainty = e_q_var is not None and e_q_var.sum().item() > 0
    if legal_indices:
        legal_q = e_q_mean[legal_indices]
        max_q = legal_q.max().item()
        min_q = legal_q.min().item()
        q_range = max_q - min_q if max_q != min_q else 1.0

        # Sort by Q-value (descending)
        if has_uncertainty and e_q_var is not None:
            sorted_legal: list[tuple[int, float, float]] = sorted(
                zip(legal_indices, [e_q_mean[i].item() for i in legal_indices],
                    [torch.sqrt(e_q_var[i]).item() if e_q_var[i] > 0 else 0.0 for i in legal_indices]),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            sorted_legal = sorted(
                zip(legal_indices, [e_q_mean[i].item() for i in legal_indices], [0.0] * len(legal_indices)),
                key=lambda x: x[1],
                reverse=True,
            )

        header = "E[Q] ± σ (expected points ± uncertainty):" if has_uncertainty else "E[Q] (expected points, higher = better):"
        lines.append(header)
        for i, q_val, std_val in sorted_legal:
            h, l = hand[i]
            dom = domino_str(h, l)
            # Bar shows relative Q within legal actions
            bar_len = int(16 * (q_val - min_q) / q_range) if q_range > 0 else 8
            bar = "█" * bar_len
            sign = "+" if q_val >= 0 else ""
            marker = " ← SELECTED" if i == action_taken else ""
            if has_uncertainty and std_val > 0:
                lines.append(f"    {dom:>7}  {sign}{q_val:5.1f} ± {std_val:4.1f} pts {bar}{marker}")
            else:
                lines.append(f"    {dom:>7}  {sign}{q_val:5.1f} pts       {bar}{marker}")

    # State-level uncertainty (t42-64uj.6)
    if has_uncertainty and (u_mean > 0 or u_max > 0):
        lines.append("")
        lines.append(f"State uncertainty: U_mean={u_mean:.2f}, U_max={u_max:.2f}")

    # Posterior health + exploration (t42-64uj.7)
    has_posterior = ess is not None or max_w is not None
    has_exploration = exploration_mode is not None
    if has_posterior or has_exploration:
        lines.append("")
        parts = []
        if ess is not None:
            # Color-code ESS: green if healthy (>50), yellow if low (10-50), red if critical (<10)
            ess_str = f"ESS={ess:.1f}"
            if ess < 10:
                ess_str += " ⚠️"  # Critical
            parts.append(ess_str)
        if max_w is not None:
            parts.append(f"max_w={max_w:.3f}")
        if exploration_mode is not None:
            mode_name = EXPLORATION_MODE_NAMES.get(exploration_mode, f"mode{exploration_mode}")
            parts.append(f"action={mode_name}")
        if q_gap is not None and q_gap > 0:
            parts.append(f"q_gap={q_gap:.1f}pts")
        lines.append("Posterior: " + " | ".join(parts))

    lines.append("─" * 80)
    lines.append("[←/→] Nav  [j] Jump  [d] Debug  [p] Params  [q] Quit")

    return "\n".join(lines)


# =============================================================================
# Diagnostics mode (t42-64uj.7)
# =============================================================================


def render_diagnostics_mode(
    idx: int,
    total: int,
    tokens: Tensor,
    length: int,
    e_q_mean: Tensor,
    legal_mask: Tensor,
    action_taken: int,
    game_idx: int,
    decision_idx: int,
    # Uncertainty fields
    e_q_var: Tensor | None = None,
    u_mean: float = 0.0,
    u_max: float = 0.0,
    # Posterior diagnostics
    ess: float | None = None,
    max_w: float | None = None,
    # Exploration stats
    exploration_mode: int | None = None,
    q_gap: float | None = None,
    # Player/outcome
    player: int | None = None,
    actual_outcome: float | None = None,
    # Dataset metadata
    metadata: dict | None = None,
) -> str:
    """Render diagnostics panel showing posterior params and generation details.

    This mode shows all the metadata that v2 datasets provide for debugging
    silent failures in E[Q] computation.
    """
    lines = []

    # Decode basic info
    decoded = decode_transcript(tokens, length)
    decl_id = decoded["decl_id"]
    hand = decoded["hand"]
    plays = decoded["plays"]
    tricks = plays_to_tricks(plays)

    complete_tricks = sum(1 for t in tricks if len(t) == 4)
    trick_num = complete_tricks + 1

    # Header
    lines.append(
        f"Example {idx + 1}/{total}  |  Game {game_idx}  |  Decision {decision_idx + 1}/28".ljust(55)
        + "[DIAGNOSTICS]"
    )
    lines.append("═" * 80)

    decl_name = DECLARATION_NAMES.get(decl_id, f"Decl {decl_id}")
    lines.append(f"Declaration: {decl_name}".ljust(40) + f"Trick {trick_num} of 7")

    # Hand (compact)
    hand_strs = []
    for i, (h, l) in enumerate(hand):
        dom = domino_str(h, l)
        if i == action_taken:
            dom = f">{dom}<"
        hand_strs.append(dom)
    lines.append(f"Hand: {' '.join(hand_strs)}")
    lines.append("─" * 80)

    # === Per-Decision Diagnostics ===
    lines.append("PER-DECISION DIAGNOSTICS")
    lines.append("")

    # Player info
    if player is not None:
        player_names_abs = ["P0 (Team0)", "P1 (Team1)", "P2 (Team0)", "P3 (Team1)"]
        lines.append(f"  Decision maker: {player_names_abs[player]}")
    if actual_outcome is not None:
        lines.append(f"  Actual outcome: {actual_outcome:+.1f} pts (my_team - opp from here)")

    lines.append("")

    # Posterior health
    lines.append("  Posterior Health:")
    if ess is not None:
        health = "healthy" if ess > 50 else ("low" if ess > 10 else "CRITICAL")
        lines.append(f"    ESS = {ess:.1f}  ({health})")
    else:
        lines.append("    ESS = (not available)")
    if max_w is not None:
        # max_w > 0.5 means one world dominates
        warning = "  ⚠️ one world dominates" if max_w > 0.5 else ""
        lines.append(f"    max_w = {max_w:.4f}{warning}")
    else:
        lines.append("    max_w = (not available)")

    lines.append("")

    # Exploration stats
    lines.append("  Exploration Stats:")
    if exploration_mode is not None:
        mode_name = EXPLORATION_MODE_NAMES.get(exploration_mode, f"unknown({exploration_mode})")
        lines.append(f"    Selection mode: {mode_name}")
    else:
        lines.append("    Selection mode: (not available)")
    if q_gap is not None:
        lines.append(f"    Q-gap (regret): {q_gap:.2f} pts")
    else:
        lines.append("    Q-gap: (not available)")

    lines.append("")

    # Uncertainty
    lines.append("  Uncertainty:")
    if e_q_var is not None:
        legal_indices = [i for i in range(len(hand)) if i < len(legal_mask) and legal_mask[i]]
        if legal_indices:
            legal_var = e_q_var[legal_indices]
            var_min = legal_var.min().item()
            var_max = legal_var.max().item()
            lines.append(f"    Var[Q] range: [{var_min:.2f}, {var_max:.2f}] pts²")
        lines.append(f"    U_mean = {u_mean:.2f} pts, U_max = {u_max:.2f} pts")
    else:
        lines.append("    (not available)")

    lines.append("─" * 80)

    # === Dataset-Level Params ===
    lines.append("DATASET CONFIGURATION (from metadata)")
    lines.append("")

    if metadata:
        version = metadata.get("version", "unknown")
        lines.append(f"  Schema version: {version}")

        # Posterior config
        post_cfg = metadata.get("posterior", {})
        if post_cfg:
            lines.append("")
            lines.append("  Posterior params:")
            lines.append(f"    enabled: {post_cfg.get('enabled', False)}")
            lines.append(f"    τ (tau): {post_cfg.get('tau', '?')}")
            lines.append(f"    β (beta): {post_cfg.get('beta', '?')}")
            lines.append(f"    K (window): {post_cfg.get('window_k', '?')}")
            lines.append(f"    Δ (delta): {post_cfg.get('delta', '?')}")
            lines.append(f"    adaptive_k: {post_cfg.get('adaptive_k_enabled', False)}")
            lines.append(f"    rejuvenation: {post_cfg.get('rejuvenation_enabled', False)}")

        # Exploration config
        expl_cfg = metadata.get("exploration", {})
        if expl_cfg:
            lines.append("")
            lines.append("  Exploration params:")
            lines.append(f"    enabled: {expl_cfg.get('enabled', False)}")
            lines.append(f"    temperature: {expl_cfg.get('temperature', '?')}")
            lines.append(f"    use_boltzmann: {expl_cfg.get('use_boltzmann', False)}")
            lines.append(f"    epsilon: {expl_cfg.get('epsilon', '?')}")
            lines.append(f"    blunder_rate: {expl_cfg.get('blunder_rate', '?')}")
            lines.append(f"    blunder_max_regret: {expl_cfg.get('blunder_max_regret', '?')}")

        # Summary stats
        summary = metadata.get("summary", {})
        if summary:
            lines.append("")
            lines.append("  Dataset summary:")
            q_range = summary.get("q_range", [None, None])
            if q_range[0] is not None:
                lines.append(f"    Q range: [{q_range[0]:.1f}, {q_range[1]:.1f}] pts")
            ess_dist = summary.get("ess_distribution", {})
            if ess_dist:
                lines.append(
                    f"    ESS: min={ess_dist.get('min', '?'):.1f}, "
                    f"p50={ess_dist.get('p50', '?'):.1f}, "
                    f"mean={ess_dist.get('mean', '?'):.1f}"
                )
            expl_stats = summary.get("exploration_stats", {})
            if expl_stats:
                lines.append(f"    Greedy rate: {expl_stats.get('greedy_rate', 0) * 100:.1f}%")
                lines.append(f"    Mean q_gap: {expl_stats.get('mean_q_gap', 0):.2f} pts")
    else:
        lines.append("  (no metadata available - pre-v2 dataset?)")

    lines.append("─" * 80)
    lines.append("[←/→] Nav  [j] Jump  [p] Exit params  [d] Debug  [q] Quit")

    return "\n".join(lines)


# =============================================================================
# Debug mode
# =============================================================================


def compute_debug_data(
    oracle: Stage1Oracle,
    tokens: Tensor,
    length: int,
    e_q_mean: Tensor,
    legal_mask: Tensor,
    action_taken: int,
    n_samples: int = 20,
) -> dict:
    """Compute debug data: oracle comparison and world breakdown.

    This is expensive - only computed when entering debug mode.

    Key insight: sample_consistent_worlds returns REMAINING hands.
    To query oracle, we need INITIAL 7-domino hands.
    Reconstruct: initial[p] = remaining[p] + played_by[p]

    IMPORTANT: Each world gets its own trick_plays computed from that world's
    initial hands. Oracle outputs are projected from local_idx to domino_id
    before averaging, matching the fix in generate.py.

    Args:
        e_q_mean: E[Q] values in POINTS (NOT logits)
    """
    from forge.eq.sampling import sample_consistent_worlds
    from forge.eq.voids import infer_voids

    # Decode transcript to reconstruct game state
    decoded = decode_transcript(tokens, length)
    decl_id = decoded["decl_id"]
    hand = decoded["hand"]  # List of (high, low) tuples - MY remaining hand
    plays = decoded["plays"]  # List of (rel_player, high, low)

    # Convert hand to domino IDs
    my_hand_ids = [pips_to_domino_id(h, l) for h, l in hand]

    # Track played dominoes AND which player played each
    played_ids = set()
    played_by: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}  # player -> [domino_ids]
    play_history = []  # For void inference: (player, domino, lead_domino)

    for rel_player, h, l in plays:
        domino_id = pips_to_domino_id(h, l)
        played_ids.add(domino_id)
        played_by[rel_player].append(domino_id)

        # Determine lead domino for this play
        trick_pos = len(play_history) % 4
        if trick_pos == 0:
            lead_domino_id = domino_id
        else:
            trick_start = len(play_history) - trick_pos
            lead_domino_id = play_history[trick_start][1]

        play_history.append((rel_player, domino_id, lead_domino_id))

    # Infer voids from play history
    voids = infer_voids(play_history, decl_id)

    # Hand sizes (7 - plays per player)
    hand_sizes = [7 - len(played_by[p]) for p in range(4)]

    # Sample consistent REMAINING hands
    rng = np.random.default_rng(42)
    try:
        sampled_remaining = sample_consistent_worlds(
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
        return {"error": str(e), "worlds": [], "world_q_values": [], "initial_hands": []}

    # For each sampled world, reconstruct initial hands and query oracle
    world_q_values = []  # Raw oracle Q-values (7,) per world (initial-hand local slots)
    world_q_values_projected = []  # Projected to remaining-hand order (len(hand),)
    world_prefs = []
    initial_hands_list = []  # Store reconstructed initial hands for display

    # Current trick info for oracle query
    trick_pos = len(plays) % 4
    current_trick_plays_raw = plays[-trick_pos:] if trick_pos > 0 else []

    # Determine current trick leader
    # Leader is whoever started the current trick (or last trick winner if starting new)
    if trick_pos == 0:
        # Starting a new trick - need to find who won last trick
        # For simplicity, assume leader is player 0 at start, track through history
        leader = 0
        for t_start in range(0, len(plays) - trick_pos, 4):
            trick = plays[t_start : t_start + 4]
            if len(trick) == 4:
                lead_h, lead_l = trick[0][1], trick[0][2]
                lead_domino = pips_to_domino_id(lead_h, lead_l)
                led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
                best_idx = 0
                best_rank = trick_rank(lead_domino, led_suit, decl_id)
                for i in range(1, 4):
                    d = pips_to_domino_id(trick[i][1], trick[i][2])
                    r = trick_rank(d, led_suit, decl_id)
                    if r > best_rank:
                        best_idx = i
                        best_rank = r
                leader = trick[best_idx][0]
    else:
        # Mid-trick - leader is first player in current trick
        leader = current_trick_plays_raw[0][0]

    # Accumulate E[Q] by domino ID across worlds
    e_q_by_domino: dict[int, list[float]] = {d: [] for d in my_hand_ids}

    for remaining_world in sampled_remaining:
        # Reconstruct initial 7-domino hands: initial = remaining + played_by
        initial_hands = []
        for p in range(4):
            initial = list(remaining_world[p]) + played_by[p]
            # Sort for consistent ordering (oracle expects sorted initial hands)
            initial.sort()
            initial_hands.append(initial)

        initial_hands_list.append(initial_hands)

        # Build domino -> local_idx lookup for THIS world's initial hand (player 0)
        domino_to_local: dict[int, int] = {}
        for local_idx, domino in enumerate(initial_hands[0]):
            domino_to_local[domino] = local_idx

        # Build remaining bitmask: bit i set if initial_hands[p][i] not yet played
        remaining = np.zeros((1, 4), dtype=np.int32)
        for p in range(4):
            for local_idx, domino in enumerate(initial_hands[p]):
                if domino not in played_ids:
                    remaining[0, p] |= (1 << local_idx)

        # Build trick_plays using DOMINO_ID (world-invariant public info)
        # This matches the batched format in generate.py
        trick_plays = []
        for rel_player, h, l in current_trick_plays_raw:
            domino = pips_to_domino_id(h, l)
            trick_plays.append((rel_player, domino))

        game_state_info = {
            "decl_id": decl_id,
            "leader": leader,
            "trick_plays": trick_plays,  # Uses domino_id, not local_idx
            "remaining": remaining,
        }

        try:
            q_values = oracle.query_batch([initial_hands], game_state_info, current_player=0)
            q_values = q_values[0]  # (7,) - indexed by initial-hand local slots
            world_q_values.append(q_values.cpu())

            # Project from local_idx to domino_id, accumulate for averaging
            for local_idx, domino_id in enumerate(initial_hands[0]):
                if domino_id in e_q_by_domino:
                    e_q_by_domino[domino_id].append(q_values[local_idx].item())

            # Project to remaining-hand order for this world's preference
            projected = torch.zeros(len(my_hand_ids))
            for i, domino_id in enumerate(my_hand_ids):
                local_idx = domino_to_local.get(domino_id)
                if local_idx is not None:
                    projected[i] = q_values[local_idx]
                else:
                    projected[i] = float("-inf")
            world_q_values_projected.append(projected)

            # Which legal action does this world prefer? (in remaining-hand order)
            masked = projected.clone()
            for i in range(len(my_hand_ids)):
                if i >= len(legal_mask) or not legal_mask[i]:
                    masked[i] = float("-inf")
            pref = masked.argmax().item()
            world_prefs.append(pref)
        except Exception:
            world_q_values.append(None)
            world_q_values_projected.append(None)
            world_prefs.append(-1)

    # Compute E[Q] in remaining-hand order (same as generate.py)
    oracle_avg_by_hand = torch.zeros(len(my_hand_ids))
    for i, domino_id in enumerate(my_hand_ids):
        if e_q_by_domino[domino_id]:
            oracle_avg_by_hand[i] = sum(e_q_by_domino[domino_id]) / len(e_q_by_domino[domino_id])
        else:
            oracle_avg_by_hand[i] = float("-inf")

    # Pad to 7 for display compatibility
    oracle_avg = torch.full((7,), float("-inf"))
    oracle_avg[:len(my_hand_ids)] = oracle_avg_by_hand

    # Rank oracle's preferences (in remaining-hand order)
    oracle_ranking = torch.argsort(oracle_avg_by_hand, descending=True).tolist()

    # Find flagged worlds (prefer different action than E[Q] selected)
    flagged = [i for i, pref in enumerate(world_prefs) if pref != action_taken and pref >= 0]

    # Count worlds preferring each action
    pref_counts: dict[int, int] = {}
    for pref in world_prefs:
        if pref >= 0:
            pref_counts[pref] = pref_counts.get(pref, 0) + 1

    return {
        "worlds": sampled_remaining,  # Remaining hands
        "initial_hands": initial_hands_list,  # Reconstructed initial 7-domino hands
        "played_by": played_by,  # Which dominoes each player has played
        "world_q_values": world_q_values,  # Raw oracle outputs (initial-hand order)
        "world_q_values_projected": world_q_values_projected,  # Projected to remaining-hand order
        "world_prefs": world_prefs,  # Preferences in remaining-hand order
        "oracle_avg": oracle_avg,  # E[Q] padded to 7
        "oracle_avg_by_hand": oracle_avg_by_hand,  # E[Q] in remaining-hand order
        "oracle_ranking": oracle_ranking,  # Ranking in remaining-hand order
        "flagged_worlds": flagged,
        "pref_counts": pref_counts,
        "hand": hand,
        "my_hand_ids": my_hand_ids,
        "decl_id": decl_id,
        "voids": voids,
    }


def render_debug_mode(
    idx: int,
    total: int,
    tokens: Tensor,
    length: int,
    e_q_mean: Tensor,
    legal_mask: Tensor,
    action_taken: int,
    game_idx: int,
    decision_idx: int,
    debug_data: dict,
    show_uniform: bool = False,
) -> str:
    """Render debug mode with oracle comparison.

    Args:
        e_q_mean: E[Q] values in POINTS (NOT logits) - the weighted dataset values
        debug_data: Contains 'oracle_avg' which is uniform E[Q] from fresh sampling
        show_uniform: If True, show weighted vs uniform comparison (t42-64uj.7)
    """
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

    # E[Q] vs Oracle comparison table (raw Q-values in points)
    # oracle_avg is UNIFORM E[Q] computed via fresh sampling in debug mode
    # e_q_mean is WEIGHTED E[Q] from the dataset
    oracle_avg = debug_data.get("oracle_avg", torch.zeros(7))
    oracle_ranking = debug_data.get("oracle_ranking", list(range(7)))

    # Get legal indices
    legal_indices = [i for i in range(len(hand)) if i < len(legal_mask) and legal_mask[i]]

    # Header depends on comparison mode (t42-64uj.7)
    if show_uniform:
        # Weighted vs Uniform comparison
        lines.append("             Weighted    Uniform      Δ(W-U)   Rank(U)")
        lines.append(f"             (dataset)  (N={n_worlds})    (pts)")

        sorted_by_eq = sorted(
            [(i, e_q_mean[i].item(), oracle_avg[i].item()) for i in legal_indices],
            key=lambda x: x[1],
            reverse=True,
        )

        for i, weighted_val, uniform_val in sorted_by_eq:
            h, l = hand[i]
            dom = domino_str(h, l)
            delta = weighted_val - uniform_val
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

            try:
                uniform_rank = oracle_ranking.index(i) + 1
            except ValueError:
                uniform_rank = "?"

            w_str = f"+{weighted_val:.1f}" if weighted_val >= 0 else f"{weighted_val:.1f}"
            u_str = f"+{uniform_val:.1f}" if uniform_val >= 0 else f"{uniform_val:.1f}"

            marker = " ← SELECTED" if i == action_taken else ""
            lines.append(f"    {dom:>7}    {w_str:>6}     {u_str:>6}    {delta_str:>6}       {uniform_rank}{marker}")

        # Show if weighted and uniform agree
        w_best = max(legal_indices, key=lambda i: e_q_mean[i].item())
        u_best = max(legal_indices, key=lambda i: oracle_avg[i].item())
        agreement = "YES ✓" if w_best == u_best else "NO ⚠️"
        lines.append("")
        lines.append(f"Weighted/Uniform agree on best action: {agreement}")
    else:
        # Original: E[Q] vs fresh Oracle comparison
        lines.append("                    E[Q]      Oracle      Δ       Rank")
        lines.append(f"                   (N={n_worlds})    (pts)     (pts)")

        sorted_by_eq = sorted(
            [(i, e_q_mean[i].item(), oracle_avg[i].item()) for i in legal_indices],
            key=lambda x: x[1],
            reverse=True,
        )

        for i, eq_val, oracle_val in sorted_by_eq:
            h, l = hand[i]
            dom = domino_str(h, l)
            delta = eq_val - oracle_val
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"

            try:
                oracle_rank = oracle_ranking.index(i) + 1
            except ValueError:
                oracle_rank = "?"

            eq_str = f"+{eq_val:.1f}" if eq_val >= 0 else f"{eq_val:.1f}"
            oracle_str = f"+{oracle_val:.1f}" if oracle_val >= 0 else f"{oracle_val:.1f}"

            marker = " ← SELECTED" if i == action_taken else ""
            lines.append(f"    {dom:>7}     {eq_str:>6}     {oracle_str:>6}   {delta_str:>6}       {oracle_rank}{marker}")

    lines.append("")

    # Agreement status
    e_best = max(legal_indices, key=lambda i: e_q_mean[i].item())
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
        f"Agreement: {agreement} (picked oracle rank {selected_oracle_rank})    Oracle Q-gap: {q_gap:.1f} pts"
    )

    lines.append("─" * 80)

    # World breakdown
    flagged = debug_data.get("flagged_worlds", [])
    world_prefs = debug_data.get("world_prefs", [])
    world_q_values = debug_data.get("world_q_values", [])
    pref_counts = debug_data.get("pref_counts", {})

    # Count worlds preferring selected action
    selected_count = pref_counts.get(action_taken, 0)
    total_valid = sum(pref_counts.values())

    # Show sample of worlds (first 10)
    display_count = min(10, n_worlds)
    lines.append(f"World Breakdown ({display_count} of {n_worlds}):".ljust(50) + f"Worlds preferring {domino_str(*hand[action_taken])}:")
    sample_worlds = list(range(display_count))

    for wi in sample_worlds:
        if wi >= len(world_q_values) or world_q_values[wi] is None:
            continue

        wl = world_q_values[wi]
        # Top 3 actions for this world (show raw Q-values)
        top3 = torch.argsort(wl[: len(hand)], descending=True)[:3].tolist()

        top_strs = []
        for ti in top3:
            if ti < len(hand):
                h, l = hand[ti]
                q = wl[ti].item()
                sign = "+" if q >= 0 else ""
                top_strs.append(f"{h}:{l}={sign}{q:.0f}")

        pref = world_prefs[wi] if wi < len(world_prefs) else -1
        best_str = domino_str(*hand[pref]) if 0 <= pref < len(hand) else "?"
        flag = " ⚠️" if wi in flagged else ""
        lines.append(f"  W{wi + 1}: {' '.join(top_strs)} → {best_str}{flag}")

    # Summary
    flag_pct = len(flagged) / total_valid * 100 if total_valid > 0 else 0
    warning = " ⚠️" if flag_pct > 30 else ""
    lines.append(f"".ljust(50) + f"  {selected_count}/{total_valid}{warning}")

    lines.append("─" * 80)
    uniform_toggle = "[u] Weighted" if show_uniform else "[u] W/U cmp"
    lines.append(f"[←/→] Nav  [j] Jump  [w] Worlds  {uniform_toggle}  [d] Exit  [q] Quit")

    return "\n".join(lines)


# =============================================================================
# World inspection mode
# =============================================================================


def render_world_mode(
    idx: int,
    total: int,
    tokens: Tensor,
    length: int,
    e_q_mean: Tensor,
    legal_mask: Tensor,
    action_taken: int,
    game_idx: int,
    decision_idx: int,
    debug_data: dict,
    world_idx: int,
) -> str:
    """Render world inspection mode - browse all sampled worlds.

    Args:
        e_q_mean: E[Q] values in POINTS (NOT logits)
    """
    lines = []

    decoded = decode_transcript(tokens, length)
    decl_id = decoded["decl_id"]
    hand = decoded["hand"]

    flagged = debug_data.get("flagged_worlds", [])
    worlds = debug_data.get("worlds", [])
    world_q_values = debug_data.get("world_q_values", [])
    world_prefs = debug_data.get("world_prefs", [])

    if not worlds:
        lines.append("No worlds to inspect.")
        lines.append("")
        lines.append("[d] Back to debug  [q] Quit")
        return "\n".join(lines)

    # Navigate through ALL worlds, not just flagged
    if world_idx >= len(worlds):
        world_idx = 0
    actual_wi = world_idx

    pref = world_prefs[actual_wi] if actual_wi < len(world_prefs) else -1
    is_flagged = actual_wi in flagged
    flag_marker = " ⚠️ DISAGREES" if is_flagged else ""
    pref_str = f"prefers {domino_str(*hand[pref])}{flag_marker}" if 0 <= pref < len(hand) else "unknown preference"

    # Header
    lines.append(
        f"Example {idx + 1}/{total}  |  Game {game_idx}  |  Decision {decision_idx + 1}/28".ljust(50)
        + f"[WORLD {world_idx + 1}/{len(worlds)} INSPECTION]"
    )
    lines.append("═" * 80)

    lines.append(f"World {actual_wi + 1} of {len(worlds)} ({pref_str})")
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

    # Oracle Q-values for this world (use projected values in remaining-hand order)
    world_q_values_projected = debug_data.get("world_q_values_projected", [])
    if actual_wi < len(world_q_values_projected) and world_q_values_projected[actual_wi] is not None:
        wl = world_q_values_projected[actual_wi]  # Already in remaining-hand order
        legal_indices = [i for i in range(len(hand)) if i < len(legal_mask) and legal_mask[i]]
        legal_q = wl[legal_indices]
        max_q = legal_q.max().item()
        min_q = legal_q.min().item()
        q_range = max_q - min_q if max_q != min_q else 1.0

        lines.append("Oracle Q-values for this world (expected points):")

        sorted_by_q = sorted(
            [(i, wl[i].item()) for i in legal_indices],
            key=lambda x: x[1],
            reverse=True,
        )

        for i, q_val in sorted_by_q:
            h, l = hand[i]
            dom = domino_str(h, l)
            bar_len = int(36 * (q_val - min_q) / q_range) if q_range > 0 else 18
            bar = "█" * bar_len
            sign = "+" if q_val >= 0 else ""
            marker = " ← best" if i == pref else ""
            lines.append(f"    {dom:>7}   {sign}{q_val:5.1f} pts {bar}{marker}")

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
    lines.append(f"[←/→] Prev/Next world  [d] Back to debug  [q] Quit   ({len(flagged)} flagged)")

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
    e_q_mean = data["e_q_mean"]  # E[Q] in points (NOT logits)
    legal_mask = data["legal_mask"]
    action_taken = data["action_taken"]
    game_idx = data["game_idx"]
    decision_idx = data["decision_idx"]
    # Uncertainty fields (t42-64uj.6)
    e_q_var = data.get("e_q_var")  # Var[Q] in points²
    u_mean = data.get("u_mean")
    u_max = data.get("u_max")
    has_uncertainty = e_q_var is not None
    # Posterior diagnostics (t42-64uj.3 / t42-64uj.7)
    ess = data.get("ess")  # Effective sample size per decision
    max_w = data.get("max_w")  # Max posterior weight per decision
    has_posterior = ess is not None
    # Exploration stats (t42-64uj.5 / t42-64uj.7)
    exploration_mode = data.get("exploration_mode")  # 0=greedy, 1=boltzmann, 2=epsilon, 3=blunder
    q_gap = data.get("q_gap")  # Regret in points
    has_exploration = exploration_mode is not None
    # Player and outcome (t42-26dl / t42-64uj.7)
    player = data.get("player")  # Who made this decision (0-3)
    actual_outcome = data.get("actual_outcome")  # Actual margin from here to end
    # Metadata (t42-64uj.7)
    metadata = data.get("metadata", {})

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
                cur_e_q_mean = e_q_mean[idx]
                cur_legal_mask = legal_mask[idx]
                cur_action = action_taken[idx].item()
                cur_game = game_idx[idx].item()
                cur_decision = decision_idx[idx].item()
                # Uncertainty data (t42-64uj.6)
                cur_e_q_var = e_q_var[idx] if has_uncertainty else None
                cur_u_mean = u_mean[idx].item() if has_uncertainty else 0.0
                cur_u_max = u_max[idx].item() if has_uncertainty else 0.0
                # Posterior diagnostics (t42-64uj.7)
                cur_ess = ess[idx].item() if has_posterior else None
                cur_max_w = max_w[idx].item() if has_posterior else None
                # Exploration stats (t42-64uj.7)
                cur_exploration_mode = exploration_mode[idx].item() if has_exploration else None
                cur_q_gap = q_gap[idx].item() if has_exploration else None
                # Player and outcome (t42-64uj.7)
                cur_player = player[idx].item() if player is not None else None
                cur_actual_outcome = actual_outcome[idx].item() if actual_outcome is not None else None

                # Render based on mode
                if state.mode == "default":
                    text = render_default_mode(
                        idx, total, cur_tokens, cur_length, cur_e_q_mean,
                        cur_legal_mask, cur_action, cur_game, cur_decision,
                        e_q_var=cur_e_q_var, u_mean=cur_u_mean, u_max=cur_u_max,
                        ess=cur_ess, max_w=cur_max_w,
                        exploration_mode=cur_exploration_mode, q_gap=cur_q_gap,
                    )
                elif state.mode == "diagnostics":
                    text = render_diagnostics_mode(
                        idx, total, cur_tokens, cur_length, cur_e_q_mean,
                        cur_legal_mask, cur_action, cur_game, cur_decision,
                        e_q_var=cur_e_q_var, u_mean=cur_u_mean, u_max=cur_u_max,
                        ess=cur_ess, max_w=cur_max_w,
                        exploration_mode=cur_exploration_mode, q_gap=cur_q_gap,
                        player=cur_player, actual_outcome=cur_actual_outcome,
                        metadata=metadata,
                    )
                elif state.mode == "debug":
                    if state.debug_data is None:
                        # Show loading message
                        stdscr.addstr(0, 0, "Loading oracle and computing debug data...")
                        stdscr.refresh()

                        # Load oracle if needed
                        if oracle is None:
                            from forge.eq.oracle import Stage1Oracle
                            checkpoint = "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
                            oracle = Stage1Oracle(checkpoint, device="cuda")

                        # Compute debug data
                        state.debug_data = compute_debug_data(
                            oracle, cur_tokens, cur_length, cur_e_q_mean,
                            cur_legal_mask, cur_action,
                        )
                        state.n_worlds = len(state.debug_data.get("worlds", []))
                        state.world_idx = 0

                    text = render_debug_mode(
                        idx, total, cur_tokens, cur_length, cur_e_q_mean,
                        cur_legal_mask, cur_action, cur_game, cur_decision,
                        state.debug_data,
                        show_uniform=state.show_uniform,
                    )
                elif state.mode == "world":
                    # debug_data is guaranteed to exist when in world mode
                    assert state.debug_data is not None
                    text = render_world_mode(
                        idx, total, cur_tokens, cur_length, cur_e_q_mean,
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
                    if state.mode == "world" and state.n_worlds > 0:
                        # Navigate all worlds
                        state.world_idx = (state.world_idx - 1) % state.n_worlds
                    elif state.current_idx > 0:
                        state.current_idx -= 1
                        state.debug_data = None  # Clear cache

                elif key == curses.KEY_RIGHT:
                    if state.mode == "world" and state.n_worlds > 0:
                        state.world_idx = (state.world_idx + 1) % state.n_worlds
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
                    elif state.mode in ("debug", "world", "diagnostics"):
                        state.mode = "default"

                elif key == ord("p"):
                    # Toggle diagnostics panel (t42-64uj.7)
                    if state.mode == "default":
                        state.mode = "diagnostics"
                    elif state.mode == "diagnostics":
                        state.mode = "default"

                elif key == ord("u"):
                    # Toggle weighted vs uniform display in debug mode (t42-64uj.7)
                    if state.mode == "debug":
                        state.show_uniform = not state.show_uniform

                elif key == ord("w"):
                    # Enter world inspection (only from debug mode)
                    if state.mode == "debug" and state.n_worlds > 0:
                        state.mode = "world"
                        state.world_idx = 0

        curses.wrapper(main)

    except ImportError:
        # Fallback without curses
        print("curses not available, using simple mode (default view only)")
        while True:
            print("\n" * 2)
            idx = state.current_idx
            cur_e_q_var = e_q_var[idx] if has_uncertainty else None
            cur_u_mean = u_mean[idx].item() if has_uncertainty else 0.0
            cur_u_max = u_max[idx].item() if has_uncertainty else 0.0
            # Posterior diagnostics (t42-64uj.7)
            cur_ess = ess[idx].item() if has_posterior else None
            cur_max_w = max_w[idx].item() if has_posterior else None
            # Exploration stats (t42-64uj.7)
            cur_exploration_mode = exploration_mode[idx].item() if has_exploration else None
            cur_q_gap = q_gap[idx].item() if has_exploration else None
            print(
                render_default_mode(
                    idx, total, tokens[idx], lengths[idx].item(),
                    e_q_mean[idx], legal_mask[idx], action_taken[idx].item(),
                    game_idx[idx].item(), decision_idx[idx].item(),
                    e_q_var=cur_e_q_var, u_mean=cur_u_mean, u_max=cur_u_max,
                    ess=cur_ess, max_w=cur_max_w,
                    exploration_mode=cur_exploration_mode, q_gap=cur_q_gap,
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
