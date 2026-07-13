"""Engine-computed auction features for the belief head (rung #24).

The play-evidence belief head sits at its Bayes ceiling (~39% ≈ chance) because
play alone is a weak signal early in a hand (rung #25's measured null). The
auction carries the missing information: a seat that *won the bid at 35 declaring
fours* almost certainly holds fours-trump strength, which sharpens the posterior
over where its dominoes are — before a single trick is played.

This module turns a completed auction into a fixed-size float feature, RELATIVE
to the current player, mirroring `gus.model.voids.voids_feature_vector`. The
`BidsEncoder` (gus.model.student) projects it into d_model and adds it to the
pooled state embedding, exactly as `VoidsEncoder` injects void evidence — so the
auction is explicit input the heads can read, with no change to the tokenizer
(and therefore no break to existing adapters).

Seat convention matches the belief head: relative seat r is `(current_player + r) % 4`,
so r=0 is me, r=1 left_opp, r=2 partner, r=3 right_opp.

Bid convention (shared by forge.zeb.types.BidState and arena HandRecord):
  <= 0 : pass / no positive bid    30..42 : a points bid    >= 84 : a marks bid
"""

from __future__ import annotations

import torch
from torch import Tensor

N_PLAYERS = 4
N_DECLS = 10  # declarations 0..9 (0-6 pip trumps, 7 doubles, 8 doubles-as-suit, 9 notrump)
N_AUCTION_FEATURES = 4 * N_PLAYERS + 2 + N_DECLS  # 4/seat + 2 global + decl one-hot = 28

# Bid value normalization: 30 (minimum points bid) → 0.0, 42 (max points) → 1.0.
_BID_LO = 30.0
_BID_HI = 42.0
_MARKS_BID = 84.0  # 2-mark bid and above is treated as "marks territory"


def _bid_norm(bid: int) -> float:
    """Map a raw bid to [0, 1]: pass→0, 30→0, 42→1, marks(>=84)→1."""
    if bid is None or bid < _BID_LO:
        return 0.0
    return min(1.0, (float(bid) - _BID_LO) / (_BID_HI - _BID_LO))


def auction_feature_vector(
    bids: tuple[int, ...] | list[int] | None,
    bidder: int | None,
    bid_value: int | None,
    decl_id: int | None,
    current_player: int,
) -> Tensor:
    """Flatten a completed auction to a [28]-dim float feature, current-player POV.

    Layout (per relative seat r = (current_player + r) % 4, r in 0..3):
      [4*r + 0] bid_norm   — normalized bid level of that seat (0 if it passed)
      [4*r + 1] passed     — 1.0 if that seat made no positive bid
      [4*r + 2] made_bid   — 1.0 if that seat bid >= 30
      [4*r + 3] is_winner  — 1.0 if that seat won the auction (== bidder)
    Global tail:
      [16] win_bid_norm    — normalized winning bid value
      [17] is_marks        — 1.0 if the contract is a marks bid (>= 84)
      [18..27] decl one-hot — the declared trump (0-6 pips, 7 doubles, 9 notrump)

    The decl one-hot is what makes "the winner declared fours ⇒ the winner holds
    fours" learnable: paired with the per-seat is_winner flag, the encoder can tie
    the declared suit to the seat that won it. (decl is also a play token, but the
    auction feature is where it joins the winner identity.)

    `bids=None` (no auction recorded, e.g. the seed-imposed corpus) yields an
    all-zero vector — the auction encoder then contributes nothing and the model
    degrades to the play+voids belief.
    """
    feat = torch.zeros(N_AUCTION_FEATURES, dtype=torch.float32)
    if bids is None:
        return feat

    for r in range(N_PLAYERS):
        abs_seat = (current_player + r) % N_PLAYERS
        bid = int(bids[abs_seat]) if abs_seat < len(bids) else 0
        feat[4 * r + 0] = _bid_norm(bid)
        feat[4 * r + 1] = 1.0 if bid <= 0 else 0.0
        feat[4 * r + 2] = 1.0 if bid >= _BID_LO else 0.0
        feat[4 * r + 3] = 1.0 if (bidder is not None and abs_seat == int(bidder)) else 0.0

    feat[16] = _bid_norm(bid_value if bid_value is not None else 0)
    feat[17] = 1.0 if (bid_value is not None and bid_value >= _MARKS_BID) else 0.0
    if decl_id is not None and 0 <= int(decl_id) < N_DECLS:
        feat[18 + int(decl_id)] = 1.0
    return feat
