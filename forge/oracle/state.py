from __future__ import annotations

import torch


LEADER_MASK = 0x3 << 28
TRICKLEN_MASK = 0x3 << 30
P0_MASK = 0x7 << 32
P1_MASK = 0x7 << 35
P2_MASK = 0x7 << 38
TRICK_FIELDS_MASK = TRICKLEN_MASK | P0_MASK | P1_MASK | P2_MASK

_POPCOUNT_CACHE: dict[torch.device, torch.Tensor] = {}


def popcount_table(device: torch.device) -> torch.Tensor:
    t = _POPCOUNT_CACHE.get(device)
    if t is None:
        t = torch.tensor([bin(i).count("1") for i in range(128)], dtype=torch.int8, device=device)
        _POPCOUNT_CACHE[device] = t
    return t


def pack_state(
    remaining: torch.Tensor,
    leader: torch.Tensor,
    trick_len: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
) -> torch.Tensor:
    return (
        (remaining[:, 0].to(torch.int64))
        | (remaining[:, 1].to(torch.int64) << 7)
        | (remaining[:, 2].to(torch.int64) << 14)
        | (remaining[:, 3].to(torch.int64) << 21)
        | (leader.to(torch.int64) << 28)
        | (trick_len.to(torch.int64) << 30)
        | (p0.to(torch.int64) << 32)
        | (p1.to(torch.int64) << 35)
        | (p2.to(torch.int64) << 38)
    )


def unpack_remaining(states: torch.Tensor) -> torch.Tensor:
    r0 = (states >> 0) & 0x7F
    r1 = (states >> 7) & 0x7F
    r2 = (states >> 14) & 0x7F
    r3 = (states >> 21) & 0x7F
    return torch.stack([r0, r1, r2, r3], dim=1)


def compute_level(states: torch.Tensor) -> torch.Tensor:
    pc = popcount_table(states.device)
    r0 = (states >> 0) & 0x7F
    r1 = (states >> 7) & 0x7F
    r2 = (states >> 14) & 0x7F
    r3 = (states >> 21) & 0x7F
    # Cast to int16 for downstream compatibility (max level is 28)
    return (pc[r0] + pc[r1] + pc[r2] + pc[r3]).to(torch.int16)


def compute_team(states: torch.Tensor) -> torch.Tensor:
    leader = (states >> 28) & 0x3
    trick_len = (states >> 30) & 0x3
    player = (leader + trick_len) & 0x3
    return (player & 0x1) == 0


def compute_terminal_value(states: torch.Tensor) -> torch.Tensor:
    # Terminal states have no more points to gain - return 0
    return torch.zeros(states.shape[0], dtype=torch.int8, device=states.device)
