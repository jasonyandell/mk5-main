from __future__ import annotations

import torch

from .context import SeedContext
from .state import LEADER_MASK, TRICK_FIELDS_MASK


def expand_gpu(states: torch.Tensor, ctx: SeedContext) -> torch.Tensor:
    """Expand packed states into children. Returns (N,7) int64, -1 for illegal.

    Fully vectorized over both states and moves - single fused kernel launch.
    """
    if states.dtype != torch.int64:
        raise ValueError(f"states must be int64, got {states.dtype}")
    if states.device.type != ctx.device.type:
        raise ValueError(f"states on {states.device} but ctx.device is {ctx.device}")

    device = states.device

    # Extract fields - all shape (N,)
    leader = (states >> 28) & 0x3
    trick_len = (states >> 30) & 0x3
    p0 = (states >> 32) & 0x7
    p1 = (states >> 35) & 0x7
    p2 = (states >> 38) & 0x7

    player = (leader + trick_len) & 0x3
    shift = player * 7  # (N,)
    hand = torch.bitwise_right_shift(states, shift) & 0x7F

    # Compute legal moves bitmask (N,)
    leading = trick_len == 0
    safe_p0 = p0.clamp(0, 6)
    follow_idx = leader * 28 + safe_p0 * 4 + trick_len
    follow_mask = ctx.LOCAL_FOLLOW[follow_idx]
    can_follow = hand & follow_mask
    must_slough = (can_follow == 0) & (~leading)
    legal = torch.where(leading | must_slough, hand, can_follow)  # (N,)

    # === VECTORIZED OVER ALL 7 MOVES ===

    # Moves tensor: (7,)
    moves = torch.arange(7, device=device, dtype=torch.int64)
    move_bits = 1 << moves  # (7,)

    # Legal mask: (N, 7) - check if each move bit is set
    is_legal = (legal.unsqueeze(1) & move_bits) != 0  # (N, 7)

    # Move mask: (N, 7) - the bit to toggle for each (state, move) pair
    # move_bits needs to be shifted by player's shift amount
    move_mask = torch.bitwise_left_shift(move_bits.unsqueeze(0), shift.unsqueeze(1))  # (N, 7)

    # Base state with move bit toggled: (N, 7)
    base = states.unsqueeze(1) ^ move_mask  # (N, 7)

    # === MID-TRICK UPDATE (trick_len != 3) ===
    completes = (trick_len == 3).unsqueeze(1)  # (N, 1)
    inc = (~completes).to(torch.int64)  # (N, 1)
    new_trick_len = trick_len.unsqueeze(1) + inc  # (N, 1)

    # Update p0/p1/p2 based on trick position - broadcasts (N, 1) with (7,) → (N, 7)
    trick_len_exp = trick_len.unsqueeze(1)  # (N, 1)
    p0_i64 = p0.unsqueeze(1).to(torch.int64)  # (N, 1)
    p1_i64 = p1.unsqueeze(1).to(torch.int64)  # (N, 1)
    p2_i64 = p2.unsqueeze(1).to(torch.int64)  # (N, 1)

    new_p0 = torch.where(trick_len_exp == 0, moves, p0_i64)  # (N, 7)
    new_p1 = torch.where(trick_len_exp == 1, moves, p1_i64)  # (N, 7)
    new_p2 = torch.where(trick_len_exp == 2, moves, p2_i64)  # (N, 7)

    child_mid = (
        (base & ~TRICK_FIELDS_MASK)
        | (new_trick_len << 30)
        | (new_p0 << 32)
        | (new_p1 << 35)
        | (new_p2 << 38)
    )

    # === TRICK COMPLETION UPDATE (trick_len == 3) ===
    safe_p0_trick = p0.clamp(0, 6)
    safe_p1_trick = p1.clamp(0, 6)
    safe_p2_trick = p2.clamp(0, 6)

    # trick_idx: (N, 7) - index into TRICK_WINNER lookup table
    trick_base = (leader * 2401 + safe_p0_trick * 343 + safe_p1_trick * 49 + safe_p2_trick * 7).unsqueeze(1)
    trick_idx = (trick_base + moves).to(torch.int64)  # (N, 7)

    winner_offset = ctx.TRICK_WINNER[trick_idx].to(torch.int64)  # (N, 7)
    winner = (leader.unsqueeze(1) + winner_offset) & 0x3  # (N, 7)

    # Reset trick fields: leader=winner, trick_len=0, p0=p1=p2=7 (sentinel)
    child_done = (base & ~LEADER_MASK) | (winner << 28)
    child_done = (child_done & ~TRICK_FIELDS_MASK) | (7 << 32) | (7 << 35) | (7 << 38)

    # === COMBINE: select mid-trick or completion based on trick_len ===
    child = torch.where(completes, child_done, child_mid)  # (N, 7)

    # Mask illegal moves to -1
    children = torch.where(is_legal, child, -1)  # (N, 7)

    return children
