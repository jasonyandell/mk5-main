"""Feature extraction from oracle GameRecordGPU → flat tensors.

v0 schema: decl_id one-hot, player_pos one-hot, decision_idx scalar,
and five 28-dim domino masks (mine, played-total, played-by-me,
played-by-left-opp, played-by-partner, played-by-right-opp). Simple and
tokenizer-free — good enough to tire-kick the belief signal.

Every piece of info here is reconstructible from (game.hands, game.decl_id,
[d.player, d.action_taken for d in decisions[:k]]).
"""

from __future__ import annotations

import torch
from torch import Tensor

N_DOMINOES = 28
N_DECLARATIONS = 10
N_PLAYERS = 4

# Feature layout (v0): decl + player + decision_idx + 6 × 28 domino masks
# (my_hand_now, played_total, played_by_{me,left_opp,partner,right_opp})
FEATURE_DIM = N_DECLARATIONS + N_PLAYERS + 1 + 6 * N_DOMINOES  # 10 + 4 + 1 + 168 = 183


def _hand_to_mask(hand: list[int]) -> Tensor:
    """Convert a hand (list of ints with possible -1s) to a [28] multi-hot mask."""
    mask = torch.zeros(N_DOMINOES, dtype=torch.float32)
    for d in hand:
        if d is None:
            continue
        d = int(d)
        if 0 <= d < N_DOMINOES:
            mask[d] = 1.0
    return mask


def _hand_list(game_hand_row: list[int]) -> list[int]:
    """Normalize a row from game.hands — it may be a list of ints or a nested list."""
    # GameRecordGPU.hands is list[list[int]] (4 players × 7 dominoes). Each row is
    # a flat list of 7 ints. Domino value -1 indicates "not in initial hand" (won't
    # happen for initial deals but the convention is there).
    return [int(x) for x in game_hand_row]


def extract_decision_features(
    game_hands: list[list[int]],
    decl_id: int,
    prior_plays: list[tuple[int, int]],  # list of (player, domino) for decisions 0..k-1
    current_player: int,
) -> Tensor:
    """Build the flat [FEATURE_DIM] feature vector for one decision.

    Args:
        game_hands: [4][7] initial deal
        decl_id: declaration id (0..9)
        prior_plays: list of (player, domino_id) for plays before this decision
        current_player: player making the decision (0..3)

    Returns:
        [FEATURE_DIM] float tensor
    """
    # decl_id one-hot
    decl_oh = torch.zeros(N_DECLARATIONS, dtype=torch.float32)
    if 0 <= decl_id < N_DECLARATIONS:
        decl_oh[decl_id] = 1.0

    # player_pos one-hot
    player_oh = torch.zeros(N_PLAYERS, dtype=torch.float32)
    player_oh[current_player] = 1.0

    # decision_idx normalized to [0, 1]
    decision_idx_norm = torch.tensor([len(prior_plays) / 28.0], dtype=torch.float32)

    # Per-seat played masks (absolute seats, not relative)
    played_by_seat = [torch.zeros(N_DOMINOES, dtype=torch.float32) for _ in range(N_PLAYERS)]
    for (p, d) in prior_plays:
        played_by_seat[int(p)][int(d)] = 1.0
    played_total = sum(played_by_seat)  # 28-dim

    # My hand now = initial hand minus what I've played
    my_initial = _hand_to_mask(_hand_list(game_hands[current_player]))
    my_hand_now = my_initial - played_by_seat[current_player]
    my_hand_now.clamp_(min=0.0, max=1.0)

    # Relative seats: self=0, left_opp=+1, partner=+2, right_opp=+3 (mod 4)
    left_opp = (current_player + 1) % 4
    partner = (current_player + 2) % 4
    right_opp = (current_player + 3) % 4

    features = torch.cat([
        decl_oh,
        player_oh,
        decision_idx_norm,
        my_hand_now,
        played_total,
        played_by_seat[current_player],      # played by me
        played_by_seat[left_opp],
        played_by_seat[partner],
        played_by_seat[right_opp],
    ])
    assert features.numel() == FEATURE_DIM, f"expected {FEATURE_DIM}, got {features.numel()}"
    return features


def extract_belief_target(
    game_hands: list[list[int]],
    prior_plays: list[tuple[int, int]],
    current_player: int,
) -> tuple[Tensor, Tensor]:
    """Build the belief target tensor.

    For each of the 28 dominoes, compute which relative seat (1=left_opp,
    2=partner, 3=right_opp) currently holds it, if any. Targets are in
    {0, 1, 2} (mapping left_opp→0, partner→1, right_opp→2 for the head).

    Returns:
        target: [28] long tensor in {0, 1, 2}; value undefined where mask=False
        mask:   [28] bool tensor — True where domino is "unseen":
                not originally in current_player's hand, and not yet played.
    """
    target = torch.zeros(N_DOMINOES, dtype=torch.long)
    mask = torch.zeros(N_DOMINOES, dtype=torch.bool)

    # Played dominoes (public): exclude from mask
    played_set = {int(d) for _, d in prior_plays}

    # My initial hand (known to me): exclude
    my_initial = {int(d) for d in _hand_list(game_hands[current_player]) if int(d) >= 0}

    # Walk each domino
    for d in range(N_DOMINOES):
        if d in played_set or d in my_initial:
            continue
        # Find which seat originally held d
        for s in range(N_PLAYERS):
            if s == current_player:
                continue
            if d in {int(x) for x in _hand_list(game_hands[s])}:
                rel = (s - current_player) % 4
                # rel is in {1, 2, 3}. Map to {0, 1, 2}.
                target[d] = rel - 1
                mask[d] = True
                break

    return target, mask


def reconstruct_prior_plays(
    game_hands: list[list[int]],
    decisions: list,
    up_to_decision_idx: int,
) -> list[tuple[int, int]]:
    """Walk the decision list up to (but not including) index k, returning
    the list of (player, domino_id) plays that happened before.

    Uses the action_taken slot index to look up the domino in the player's
    initial hand.
    """
    plays: list[tuple[int, int]] = []
    for j in range(up_to_decision_idx):
        d = decisions[j]
        player = int(d.player)
        slot = int(d.action_taken)
        # Slot index into initial hand — dominoes aren't renumbered.
        hand = _hand_list(game_hands[player])
        if 0 <= slot < len(hand) and int(hand[slot]) >= 0:
            plays.append((player, int(hand[slot])))
    return plays
