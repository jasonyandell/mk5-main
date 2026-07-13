"""Engine-computed void inference from trick history.

Given the list of plays seen so far, determine which (seat, suit) pairs
have been observed void: a player is void in suit S if they failed to
play a domino containing S when S was led.

Returns a [N_PLAYERS_REL-1, N_SUITS] boolean mask — relative to the
current player (so index 0 = left_opp, 1 = partner, 2 = right_opp).

This is the signal a transformer has to learn attentionally from raw
play tokens alone. Injecting it as explicit input features should close
most of the belief gap from 37% toward a true 60%+.

Suit convention follows forge/oracle: 8 "natural" suits total —
0..6 = pip suits (0 = blanks, 6 = sixes), 7 = doubles (for doubles-trump).

Declaration tells us what's TRUMP. In 42:
  - "pip" declarations 0..6: the declared pip number is trump; a domino
    containing that pip is a trump rather than whatever its off-suit is.
  - declaration 7: doubles are trump (all 7 doubles become trump).
  - "follow-me" / "doubles-as-suit" variants... we ignore specials for v1
    and treat declaration as a simple pip trump (0..6) or doubles (7).
"""

from __future__ import annotations

import torch
from torch import Tensor

N_SUITS = 8
N_PLAYERS = 4


def domino_pips(d: int) -> tuple[int, int]:
    """Return (high, low) pips for domino id 0..27 in canonical order.

    Matches DOMINO_NAMES in scratch/joint_worlds_tire_kick.py:
      [(a, b) for a in range(7) for b in range(a + 1)]
    So e.g. domino 0 = (0, 0), domino 1 = (1, 0), domino 2 = (1, 1), ...
    """
    idx = 0
    for a in range(7):
        for b in range(a + 1):
            if idx == d:
                return a, b
            idx += 1
    raise ValueError(f"invalid domino id: {d}")


_DOMINO_PIPS_TABLE: list[tuple[int, int]] | None = None


def _pips_table() -> list[tuple[int, int]]:
    global _DOMINO_PIPS_TABLE
    if _DOMINO_PIPS_TABLE is None:
        _DOMINO_PIPS_TABLE = [domino_pips(d) for d in range(28)]
    return _DOMINO_PIPS_TABLE


def is_trump(d: int, decl_id: int) -> bool:
    """True if domino d is trump under declaration decl_id."""
    a, b = _pips_table()[d]
    if decl_id == 7:
        # doubles are trump
        return a == b
    # pip trump: any domino containing the declared pip (and pip-trump 0..6)
    if 0 <= decl_id <= 6:
        return a == decl_id or b == decl_id
    return False


def led_suit(d: int, decl_id: int) -> int:
    """Suit the led domino establishes as the trick's suit.

    If d is trump (or a double under doubles-trump), the led suit IS trump
    (suit index = decl_id for pip-trump, or 7 for doubles-trump).
    Otherwise the led suit is the HIGHER pip of d (standard 42 rule: the
    larger end is the natural suit of a non-trump domino).
    """
    a, b = _pips_table()[d]
    if is_trump(d, decl_id):
        # Suit of trump is just the declaration — pip or doubles.
        return decl_id
    return max(a, b)


def can_follow(d: int, led: int, decl_id: int) -> bool:
    """True if domino d can follow led suit `led` under declaration decl_id."""
    # If the led suit is trump, only trump dominoes can follow.
    if led == decl_id:
        return is_trump(d, decl_id)
    # Otherwise: follow with (a) any domino containing the led pip and not trump,
    # or (b) actually, standard 42: must follow with a domino containing the led
    # pip IF POSSIBLE (including playing trump off-suit is allowed only when void).
    # For void detection we check: did they HAVE a non-trump-following option?
    # Simpler: domino d follows led if led-pip is in d AND d is NOT trump.
    a, b = _pips_table()[d]
    if led == 7:
        # doubles-as-led only under doubles-trump, already handled above.
        return False
    if (a == led or b == led) and not is_trump(d, decl_id):
        return True
    return False


def infer_voids(
    prior_plays: list[tuple[int, int]],
    decl_id: int,
    current_player: int,
) -> Tensor:
    """Return [3, N_SUITS] bool void mask (for opponents relative to current_player).

    Opponent indices (axis 0):
      0 = left_opp = (current_player + 1) % 4
      1 = partner  = (current_player + 2) % 4
      2 = right_opp= (current_player + 3) % 4

    Axis 1 (suits): 0..6 = pips 0..6, 7 = doubles suit.

    Algorithm: scan plays in trick-order (4 at a time). For each trick:
      - leader = first player who played (the trick's actor[0])
      - led = led_suit(leader's domino, decl_id)
      - For each non-leader in the trick, if their domino didn't follow
        the led suit, mark them void in that suit.
    """
    voids_abs = torch.zeros(N_PLAYERS, N_SUITS, dtype=torch.bool)

    # Walk plays in groups of 4 (complete tricks). If the last trick is
    # partial, process whatever plays we have.
    n_plays = len(prior_plays)
    for t_start in range(0, n_plays, 4):
        trick = prior_plays[t_start:t_start + 4]
        if not trick:
            break
        leader_player, leader_dom = trick[0]
        led = led_suit(int(leader_dom), decl_id)
        for (p, d) in trick[1:]:
            if not can_follow(int(d), led, decl_id):
                voids_abs[int(p), led] = True

    # Project to relative seats (excluding current_player)
    voids_rel = torch.zeros(3, N_SUITS, dtype=torch.bool)
    for rel in range(1, 4):  # 1=left, 2=partner, 3=right
        abs_p = (current_player + rel) % 4
        voids_rel[rel - 1] = voids_abs[abs_p]

    return voids_rel


def voids_feature_vector(
    prior_plays: list[tuple[int, int]],
    decl_id: int,
    current_player: int,
) -> Tensor:
    """Flatten to [24]-dim float feature (3 opponents × 8 suits)."""
    v = infer_voids(prior_plays, decl_id, current_player)  # [3, 8] bool
    return v.float().reshape(-1)  # [24]
