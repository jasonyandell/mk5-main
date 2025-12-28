"""
State packing/unpacking for the GPU solver.

State representation (47 bits in int64):

Bits 0-27: Remaining hands (4 x 7-bit masks)
  [0:7]   remaining[0] - Player 0's local indices still in hand
  [7:14]  remaining[1] - Player 1's
  [14:21] remaining[2] - Player 2's
  [21:28] remaining[3] - Player 3's

Bits 28-35: Game progress
  [28:34] score       - Team 0's points (0-42, 6 bits)
  [34:36] leader      - Current trick leader (0-3, 2 bits)

Bits 36-47: Current trick
  [36:38] trick_len   - Plays so far (0-3, 2 bits)
  [38:41] play0       - Leader's local index (0-6, or 7 if N/A)
  [41:44] play1       - Second player's local index
  [44:47] play2       - Third player's local index
"""

import torch

# Popcount lookup table for 7-bit values (128 entries)
POPCOUNT = torch.tensor([bin(i).count("1") for i in range(128)], dtype=torch.int64)


def pack_state(
    remaining: torch.Tensor,
    score: torch.Tensor,
    leader: torch.Tensor,
    trick_len: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
) -> torch.Tensor:
    """
    Pack game state into int64 tensor.

    Args:
        remaining: (N, 4) tensor of 7-bit masks for each player's hand
        score: (N,) tensor of Team 0's points (0-42)
        leader: (N,) tensor of current trick leader (0-3)
        trick_len: (N,) tensor of plays so far (0-3)
        p0: (N,) tensor of leader's local index (0-6, or 7 if N/A)
        p1: (N,) tensor of second player's local index
        p2: (N,) tensor of third player's local index

    Returns:
        (N,) int64 packed states
    """
    return (
        (remaining[:, 0].to(torch.int64))
        | (remaining[:, 1].to(torch.int64) << 7)
        | (remaining[:, 2].to(torch.int64) << 14)
        | (remaining[:, 3].to(torch.int64) << 21)
        | (score.to(torch.int64) << 28)
        | (leader.to(torch.int64) << 34)
        | (trick_len.to(torch.int64) << 36)
        | (p0.to(torch.int64) << 38)
        | (p1.to(torch.int64) << 41)
        | (p2.to(torch.int64) << 44)
    )


def unpack_remaining(states: torch.Tensor) -> torch.Tensor:
    """
    Unpack remaining hands from packed states.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N, 4) tensor of 7-bit masks
    """
    r0 = (states >> 0) & 0x7F
    r1 = (states >> 7) & 0x7F
    r2 = (states >> 14) & 0x7F
    r3 = (states >> 21) & 0x7F
    return torch.stack([r0, r1, r2, r3], dim=1)


def unpack_score(states: torch.Tensor) -> torch.Tensor:
    """
    Unpack score from packed states.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) tensor of scores (0-42)
    """
    return (states >> 28) & 0x3F


def unpack_leader(states: torch.Tensor) -> torch.Tensor:
    """
    Unpack leader from packed states.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) tensor of leaders (0-3)
    """
    return (states >> 34) & 0x3


def unpack_trick_len(states: torch.Tensor) -> torch.Tensor:
    """
    Unpack trick length from packed states.

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) tensor of trick lengths (0-3)
    """
    return (states >> 36) & 0x3


def unpack_plays(states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpack plays from packed states.

    Args:
        states: (N,) int64 packed states

    Returns:
        (p0, p1, p2) each (N,) tensor of local indices (0-6, or 7 if N/A)
    """
    p0 = (states >> 38) & 0x7
    p1 = (states >> 41) & 0x7
    p2 = (states >> 44) & 0x7
    return p0, p1, p2


def compute_level(states: torch.Tensor) -> torch.Tensor:
    """
    Compute game level (total dominoes remaining across all hands).

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) tensor of levels (0-28)
    """
    r0 = (states >> 0) & 0x7F
    r1 = (states >> 7) & 0x7F
    r2 = (states >> 14) & 0x7F
    r3 = (states >> 21) & 0x7F

    # Use POPCOUNT on same device as states
    popcount = POPCOUNT.to(states.device)
    return popcount[r0] + popcount[r1] + popcount[r2] + popcount[r3]


def compute_team(states: torch.Tensor) -> torch.Tensor:
    """
    Determine if current player is on team 0.

    Current player = (leader + trick_len) % 4
    Team 0 = players 0 and 2

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) boolean tensor (True if current player on team 0)
    """
    leader = (states >> 34) & 0x3
    trick_len = (states >> 36) & 0x3
    player = (leader + trick_len) % 4
    return (player % 2) == 0


def compute_terminal_value(states: torch.Tensor) -> torch.Tensor:
    """
    Compute terminal value (team 0's advantage).

    Terminal value = 2 * score - 42

    Args:
        states: (N,) int64 packed states

    Returns:
        (N,) int8 tensor of terminal values (-42 to +42)
    """
    score = (states >> 28) & 0x3F
    return (2 * score - 42).to(torch.int8)
