from __future__ import annotations

from dataclasses import dataclass

import torch

from .declarations import N_DECLS
from .rng import deal_from_seed
from .tables import can_follow, led_suit_for_lead_domino, resolve_trick


@dataclass(frozen=True)
class SeedContext:
    seed: int
    decl_id: int
    device: torch.device

    # Local index -> global domino id
    L: torch.Tensor  # (4, 7) int8

    # LOCAL_FOLLOW[leader * 28 + lead_local_idx * 4 + follower_offset] -> 7-bit mask (int64)
    LOCAL_FOLLOW: torch.Tensor  # (112,) int64

    # TRICK_*[leader * 2401 + p0*343 + p1*49 + p2*7 + p3]
    TRICK_WINNER: torch.Tensor  # (9604,) int8
    TRICK_POINTS: torch.Tensor  # (9604,) int8
    TRICK_REWARD: torch.Tensor  # (9604,) int8 - signed: +points if team0 wins, -points if team1

    def initial_state(self) -> torch.Tensor:
        # remaining masks all 7 bits set; leader=0; trick_len=0; plays unset (7)
        remaining = 0x7F | (0x7F << 7) | (0x7F << 14) | (0x7F << 21)
        state = remaining | (0 << 28) | (0 << 30) | (7 << 32) | (7 << 35) | (7 << 38)
        return torch.tensor(state, dtype=torch.int64, device=self.device)


def build_context(seed: int, decl_id: int, device: torch.device) -> SeedContext:
    if not (0 <= decl_id < N_DECLS):
        raise ValueError(f"decl_id out of range: {decl_id}")

    hands = deal_from_seed(seed)
    L_cpu = torch.tensor(hands, dtype=torch.int8)

    local_follow = torch.zeros((4 * 7 * 4,), dtype=torch.int64)
    for leader in range(4):
        for lead_local in range(7):
            lead_domino_id = int(L_cpu[leader, lead_local])
            led_suit = led_suit_for_lead_domino(lead_domino_id, decl_id)
            for follower_offset in range(4):
                if follower_offset == 0:
                    continue
                player = (leader + follower_offset) % 4
                mask = 0
                for local_idx in range(7):
                    d = int(L_cpu[player, local_idx])
                    if can_follow(d, led_suit, decl_id):
                        mask |= 1 << local_idx
                local_follow[leader * 28 + lead_local * 4 + follower_offset] = mask

    trick_winner = torch.empty((4 * 7 * 7 * 7 * 7,), dtype=torch.int8)
    trick_points = torch.empty((4 * 7 * 7 * 7 * 7,), dtype=torch.int8)
    for leader in range(4):
        p1_seat = (leader + 1) % 4
        p2_seat = (leader + 2) % 4
        p3_seat = (leader + 3) % 4
        for p0 in range(7):
            for p1 in range(7):
                for p2 in range(7):
                    for p3 in range(7):
                        d0 = int(L_cpu[leader, p0])
                        d1 = int(L_cpu[p1_seat, p1])
                        d2 = int(L_cpu[p2_seat, p2])
                        d3 = int(L_cpu[p3_seat, p3])
                        outcome = resolve_trick(d0, (d0, d1, d2, d3), decl_id)
                        idx = leader * 2401 + p0 * 343 + p1 * 49 + p2 * 7 + p3
                        trick_winner[idx] = outcome.winner_offset
                        trick_points[idx] = outcome.points

    # Precompute signed rewards: +points if team0 wins, -points if team1 wins
    # Sign depends on (leader + winner_offset) % 2
    trick_reward = torch.empty((4 * 7 * 7 * 7 * 7,), dtype=torch.int8)
    for idx in range(4 * 2401):
        leader = idx // 2401
        winner_offset = int(trick_winner[idx])
        winner = (leader + winner_offset) % 4
        team0_wins = (winner % 2) == 0
        trick_reward[idx] = trick_points[idx] if team0_wins else -trick_points[idx]

    return SeedContext(
        seed=seed,
        decl_id=decl_id,
        device=device,
        L=L_cpu.to(device=device, non_blocking=True),
        LOCAL_FOLLOW=local_follow.to(device=device, non_blocking=True),
        TRICK_WINNER=trick_winner.to(device=device, non_blocking=True),
        TRICK_POINTS=trick_points.to(device=device, non_blocking=True),
        TRICK_REWARD=trick_reward.to(device=device, non_blocking=True),
    )

