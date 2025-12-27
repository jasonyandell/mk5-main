"""
GPU state expansion for Texas 42 solver.

The heart of the solver. Given N packed states, produce (N, 7) tensor of child states.
Each move corresponds to playing the domino at that local index (0-6).

Memory-optimized version: processes one move at a time to avoid huge intermediate tensors.
"""

import torch
from state import pack_state, unpack_remaining, unpack_score, unpack_leader, unpack_trick_len, unpack_plays
from context import SeedContext


def expand_gpu(states: torch.Tensor, ctx: SeedContext) -> torch.Tensor:
    """
    Expand packed states to all possible children.

    Memory-optimized: processes one move at a time instead of creating
    a huge (N, 4, 7) tensor for all moves simultaneously.

    Args:
        states: (N,) int64 packed states
        ctx: SeedContext with precomputed tables

    Returns:
        (N, 7) int64 children, -1 for illegal moves
    """
    N = states.shape[0]
    device = states.device

    # Move context tables to same device
    L = ctx.L.to(device)
    LOCAL_FOLLOW = ctx.LOCAL_FOLLOW.to(device)
    TRICK_WINNER = ctx.TRICK_WINNER.to(device)
    TRICK_POINTS = ctx.TRICK_POINTS.to(device)

    # Unpack all state fields
    remaining = unpack_remaining(states)  # (N, 4)
    score = unpack_score(states)  # (N,)
    leader = unpack_leader(states)  # (N,)
    trick_len = unpack_trick_len(states)  # (N,)
    p0, p1, p2 = unpack_plays(states)  # each (N,)

    # Compute whose turn: player = (leader + trick_len) % 4
    player = (leader + trick_len) % 4  # (N,)

    # Get player's hand using gather
    player_hand = remaining.gather(1, player.unsqueeze(1)).squeeze(1)  # (N,)

    # Determine legal moves
    is_leading = (trick_len == 0)  # (N,)

    # Compute follow mask for non-leading positions
    follower_offset = trick_len
    p0_safe = p0.clamp(max=6)
    follow_idx = leader * 28 + p0_safe * 4 + follower_offset  # (N,)
    follow_mask = LOCAL_FOLLOW[follow_idx.long()].to(torch.int64)  # (N,)

    # Can we follow?
    can_follow_mask = player_hand & follow_mask
    can_follow = can_follow_mask > 0

    # Legal moves: if leading, full hand; if following and can follow, intersection;
    # if following and can't follow (slough), full hand
    legal_mask = torch.where(
        is_leading,
        player_hand,
        torch.where(can_follow, can_follow_mask, player_hand)
    )

    # Is this the fourth play (completing trick)?
    is_completing = (trick_len == 3)

    # Precompute trick lookup base index (for completing tricks)
    p1_safe = p1.clamp(max=6)
    p2_safe = p2.clamp(max=6)
    base_trick_idx = leader * 2401 + p0_safe * 343 + p1_safe * 49 + p2_safe * 7

    # Initialize children tensor
    children = torch.full((N, 7), -1, dtype=torch.int64, device=device)

    # Process each move separately to avoid huge intermediate tensors
    for m in range(7):
        move_bit = 1 << m
        is_legal_m = (legal_mask & move_bit) != 0

        if not is_legal_m.any():
            continue

        # Compute new remaining for this move
        # Remove bit m from current player's hand
        new_player_hand = player_hand ^ move_bit  # Toggle bit m off

        # Update remaining array for current player
        # remaining[i, player[i]] = new_player_hand[i]
        new_remaining = remaining.scatter(1, player.unsqueeze(1), new_player_hand.unsqueeze(1))

        # Compute child state components based on whether trick completes
        # --- Trick completion path ---
        trick_idx = base_trick_idx + m
        winner_offset = TRICK_WINNER[trick_idx.long()].to(torch.int64)
        points = TRICK_POINTS[trick_idx.long()].to(torch.int64)

        new_leader_complete = (leader + winner_offset) % 4
        winning_team = new_leader_complete % 2
        score_add = torch.where(winning_team == 0, points, torch.zeros_like(points))
        new_score_complete = score + score_add

        # Reset trick state after completion
        new_trick_len_complete = torch.zeros_like(trick_len)
        new_p0_complete = torch.full_like(p0, 7)
        new_p1_complete = torch.full_like(p1, 7)
        new_p2_complete = torch.full_like(p2, 7)

        # --- Trick continuation path ---
        new_leader_continue = leader
        new_score_continue = score
        new_trick_len_continue = trick_len + 1

        # Update plays based on trick_len
        new_p0_continue = torch.where(trick_len == 0, torch.full_like(p0, m), p0)
        new_p1_continue = torch.where(trick_len == 1, torch.full_like(p1, m), p1)
        new_p2_continue = torch.where(trick_len == 2, torch.full_like(p2, m), p2)

        # Select based on whether trick completes
        new_leader = torch.where(is_completing, new_leader_complete, new_leader_continue)
        new_score = torch.where(is_completing, new_score_complete, new_score_continue)
        new_trick_len = torch.where(is_completing, new_trick_len_complete, new_trick_len_continue)
        new_p0 = torch.where(is_completing, new_p0_complete, new_p0_continue)
        new_p1 = torch.where(is_completing, new_p1_complete, new_p1_continue)
        new_p2 = torch.where(is_completing, new_p2_complete, new_p2_continue)

        # Pack child state
        child = pack_state(new_remaining, new_score, new_leader, new_trick_len, new_p0, new_p1, new_p2)

        # Store in children tensor (only for legal moves)
        children[:, m] = torch.where(is_legal_m, child, torch.tensor(-1, dtype=torch.int64, device=device))

    return children
