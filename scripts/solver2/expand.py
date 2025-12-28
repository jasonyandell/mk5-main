from __future__ import annotations

import torch

from .context import SeedContext
from .state import LEADER_MASK, TRICK_FIELDS_MASK


def expand_gpu(states: torch.Tensor, ctx: SeedContext) -> torch.Tensor:
    """Expand packed states into children. Returns (N,7) int64, -1 for illegal."""
    if states.dtype != torch.int64:
        raise ValueError(f"states must be int64, got {states.dtype}")
    if states.device != ctx.device:
        raise ValueError(f"states on {states.device} but ctx.device is {ctx.device}")

    leader = (states >> 28) & 0x3
    trick_len = (states >> 30) & 0x3
    p0 = (states >> 32) & 0x7
    p1 = (states >> 35) & 0x7
    p2 = (states >> 38) & 0x7

    player = (leader + trick_len) & 0x3
    shift = player * 7
    hand = torch.bitwise_right_shift(states, shift) & 0x7F

    leading = trick_len == 0
    safe_p0 = p0.clamp(0, 6)
    follow_idx = leader * 28 + safe_p0 * 4 + trick_len
    follow_mask = ctx.LOCAL_FOLLOW[follow_idx]
    can_follow = hand & follow_mask
    must_slough = (can_follow == 0) & (~leading)
    legal = torch.where(leading | must_slough, hand, can_follow)

    children = torch.full((states.shape[0], 7), -1, dtype=torch.int64, device=states.device)
    completes = trick_len == 3

    for move in range(7):
        move_bit = 1 << move
        is_legal = (legal & move_bit) != 0
        if not bool(is_legal.any()):
            continue

        move_mask = torch.bitwise_left_shift(torch.tensor(move_bit, dtype=torch.int64, device=states.device), shift)
        base = states ^ move_mask

        # Mid-trick update.
        inc = (trick_len != 3).to(torch.int64)
        new_trick_len = trick_len + inc
        new_p0 = torch.where(trick_len == 0, move, p0).to(torch.int64)
        new_p1 = torch.where(trick_len == 1, move, p1).to(torch.int64)
        new_p2 = torch.where(trick_len == 2, move, p2).to(torch.int64)
        child_mid = (base & ~TRICK_FIELDS_MASK) | (new_trick_len << 30) | (new_p0 << 32) | (new_p1 << 35) | (new_p2 << 38)

        # Trick completion update.
        safe_p0_for_trick = p0.clamp(0, 6)
        safe_p1_for_trick = p1.clamp(0, 6)
        safe_p2_for_trick = p2.clamp(0, 6)
        trick_idx = (leader * 2401 + safe_p0_for_trick * 343 + safe_p1_for_trick * 49 + safe_p2_for_trick * 7 + move).to(
            torch.int64
        )
        winner_offset = ctx.TRICK_WINNER[trick_idx].to(torch.int64)
        winner = (leader + winner_offset) & 0x3

        child_done = base
        child_done = (child_done & ~LEADER_MASK) | (winner.to(torch.int64) << 28)
        child_done = (child_done & ~TRICK_FIELDS_MASK) | (0 << 30) | (7 << 32) | (7 << 35) | (7 << 38)

        child = torch.where(completes, child_done, child_mid)
        children[:, move] = torch.where(is_legal, child, -1)

    return children
