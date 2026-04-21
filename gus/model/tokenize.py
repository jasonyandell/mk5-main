"""Tokenize a (game, decision_idx) into a sequence of plays for a transformer.

Sequence layout (fixed length L = 33):
  pos 0:     [CLS]             — special pooling token
  pos 1:     [DECL=k]          — declaration token (k in 0..9)
  pos 2..8:  [MINE=d]          — each domino still in my hand (pad if fewer)
  pos 9..32: [PLAY=d]          — 6 tricks × 4 plays = 24 play slots
                                  (pad if trick/play hasn't happened yet)

Each position carries 5 channels:
  token_id:     0..27  = domino_id
                28     = PAD
                29     = CLS
                30..39 = decl_k (decl 0 → 30)
  type_id:      0=CLS, 1=DECL, 2=MINE, 3=PLAY
  trick_id:     0..6 = trick index (PLAY only), 7 = N/A
  trick_pos_id: 0..3 = position within trick (PLAY only), 4 = N/A
  player_rel:   0=me, 1=left_opp, 2=partner, 3=right_opp (PLAY only), 4 = N/A

Also returns an attention_mask [L] (1 = real, 0 = PAD).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .features import reconstruct_prior_plays, _hand_list

# Vocabulary sizes
N_DOMINOES = 28
PAD_TOKEN = 28
CLS_TOKEN = 29
DECL_OFFSET = 30  # decl_k → token DECL_OFFSET + k
TOKEN_VOCAB_SIZE = DECL_OFFSET + 10  # 40

N_TYPES = 4           # CLS, DECL, MINE, PLAY
N_TRICK_SLOTS = 8     # 0..6 + NA
N_TRICK_POS = 5       # 0..3 + NA
N_PLAYER_REL = 5      # 0..3 + NA

# Sequence positions
SEQ_LEN = 1 + 1 + 7 + 6 * 4   # 33

TYPE_CLS = 0
TYPE_DECL = 1
TYPE_MINE = 2
TYPE_PLAY = 3

TRICK_NA = 7
POS_NA = 4
PLAYER_REL_NA = 4


def _pad_slot() -> tuple[int, int, int, int, int]:
    """A fully-padded position: (token, type, trick, pos, player_rel)."""
    return (PAD_TOKEN, TYPE_PLAY, TRICK_NA, POS_NA, PLAYER_REL_NA)


def tokenize_decision(
    game_hands: list[list[int]],
    decl_id: int,
    decisions: list,
    decision_idx: int,
) -> tuple[Tensor, Tensor]:
    """Tokenize the public state leading up to decision `decision_idx`.

    Returns:
        tokens: [SEQ_LEN, 5] long tensor — five channels per position.
        attention_mask: [SEQ_LEN] bool — True for real tokens, False for PAD.
    """
    # Reconstruct prior plays
    prior_plays = reconstruct_prior_plays(game_hands, decisions, decision_idx)
    current_player = int(decisions[decision_idx].player)

    # Track who's played what (for MINE slot computation)
    played_by_me: set[int] = set()
    for (p, d) in prior_plays:
        if int(p) == current_player:
            played_by_me.add(int(d))

    # My current hand = initial hand minus what I've played
    my_initial = [int(x) for x in _hand_list(game_hands[current_player]) if int(x) >= 0]
    my_current = [d for d in my_initial if d not in played_by_me]

    # Build slots
    slots: list[tuple[int, int, int, int, int]] = []

    # pos 0: CLS
    slots.append((CLS_TOKEN, TYPE_CLS, TRICK_NA, POS_NA, PLAYER_REL_NA))

    # pos 1: DECL
    decl_token = DECL_OFFSET + int(decl_id)
    slots.append((decl_token, TYPE_DECL, TRICK_NA, POS_NA, PLAYER_REL_NA))

    # pos 2..8: MINE (7 slots; pad the rest)
    for i in range(7):
        if i < len(my_current):
            slots.append((my_current[i], TYPE_MINE, TRICK_NA, POS_NA, 0))  # 0 = me
        else:
            slots.append((PAD_TOKEN, TYPE_MINE, TRICK_NA, POS_NA, 0))

    # pos 9..32: PLAY (6 tricks × 4 plays = 24 slots)
    # We map prior_plays (a flat chronological list) into their (trick, pos) slots.
    # Tricks are groups of 4 consecutive plays.
    for t in range(6):
        for i in range(4):
            flat_idx = t * 4 + i
            if flat_idx < len(prior_plays):
                (p_abs, dom) = prior_plays[flat_idx]
                rel = (int(p_abs) - current_player) % 4  # 0=me, 1=left_opp, 2=partner, 3=right_opp
                slots.append((int(dom), TYPE_PLAY, t, i, rel))
            else:
                # Not yet played
                slots.append(_pad_slot())

    assert len(slots) == SEQ_LEN

    tokens = torch.tensor(slots, dtype=torch.long)  # [SEQ_LEN, 5]

    # Attention mask: real tokens are CLS, DECL, MINE with real dominos, PLAY with real plays.
    # Everything with token_id == PAD is masked out.
    attention_mask = tokens[:, 0] != PAD_TOKEN  # [SEQ_LEN] bool

    return tokens, attention_mask
