"""Fast CPU tests for the auction decoder v0 (Lane A).

Covers the leakage discipline (a later seat's bid cannot enter an earlier seat's
features), feature-block shapes, and the class-map roundtrip.
"""
from __future__ import annotations

import torch

from champion.bid_decoder import (
    HIGH_DIM,
    POS_DIM,
    PREFIX_DIM,
    bidding_order,
    build_class_map,
    decision_features,
    feature_dim,
    hand_decisions,
)
from champion.bid_net import FEATURE_DIM as HAND_DIM


def _snap(dealer: int, bids: list[int]) -> dict:
    # Four disjoint 7-domino hands (ids 0..27) so featurize_hand is well-defined.
    hands = [list(range(s * 7, s * 7 + 7)) for s in range(4)]
    return {"dealer": dealer, "bids": bids, "hands": hands,
            "seed": 1, "hand_idx": 0, "bidder": 0, "decl_id": 0}


def test_feature_dim_matches_blocks():
    assert feature_dim(include_hand=False, include_pop=False) == POS_DIM + PREFIX_DIM + HIGH_DIM
    assert feature_dim(include_hand=True, include_pop=False) == HAND_DIM + POS_DIM + PREFIX_DIM + HIGH_DIM
    assert feature_dim(include_hand=True, include_pop=True) == HAND_DIM + POS_DIM + PREFIX_DIM + HIGH_DIM + 3


def test_decision_feature_shape():
    snap = _snap(dealer=0, bids=[33, 30, 31, 32])
    for pos in range(4):
        seat = bidding_order(0)[pos]
        x = decision_features(snap["hands"][seat], snap["bids"], 0, pos, "margin:wp",
                              include_hand=True, include_pop=True)
        assert x.shape == (feature_dim(include_hand=True, include_pop=True),)


def test_later_seat_bid_does_not_leak():
    """Changing a LATER seat's bid must not change an earlier actor's features."""
    dealer = 0
    order = bidding_order(dealer)  # (1, 2, 3, 0)
    # Actor at bidding position 1 sees only position-0's bid.
    pos = 1
    seat = order[pos]
    base_bids = [0, 0, 0, 0]
    base_bids[order[0]] = 30  # the one bid the actor heard
    x0 = decision_features(_snap(dealer, base_bids)["hands"][seat], base_bids, dealer, pos,
                           None, include_hand=True, include_pop=False)
    # Mutate every LATER seat (positions 1..3, i.e. the actor and after).
    mutated = list(base_bids)
    for later_pos in range(pos, 4):
        mutated[order[later_pos]] = 41
    x1 = decision_features(_snap(dealer, mutated)["hands"][seat], mutated, dealer, pos,
                           None, include_hand=True, include_pop=False)
    assert torch.equal(x0, x1), "later-seat bids leaked into an earlier actor's features"


def test_earlier_seat_bid_does_change_features():
    """A bid the actor DID hear must move the features (guards against masking too much)."""
    dealer = 0
    order = bidding_order(dealer)
    pos = 2
    seat = order[pos]
    a = [0, 0, 0, 0]; a[order[0]] = 30
    b = [0, 0, 0, 0]; b[order[0]] = 41
    xa = decision_features(_snap(dealer, a)["hands"][seat], a, dealer, pos, None,
                           include_hand=True, include_pop=False)
    xb = decision_features(_snap(dealer, b)["hands"][seat], b, dealer, pos, None,
                           include_hand=True, include_pop=False)
    assert not torch.equal(xa, xb), "an earlier heard bid did not affect features"


def test_first_actor_has_empty_prefix():
    """Position 0 speaks first: its prefix block and running-high are all zero."""
    dealer = 2
    order = bidding_order(dealer)
    seat = order[0]
    bids = [0, 0, 0, 0]; bids[order[1]] = 35; bids[order[2]] = 36  # later bids
    x = decision_features(_snap(dealer, bids)["hands"][seat], bids, dealer, 0, None,
                          include_hand=False, include_pop=False)
    # layout (no hand): [pos one-hot 4][prefix 9][high 1]
    prefix_and_high = x[POS_DIM:]
    assert torch.count_nonzero(prefix_and_high) == 0


def test_class_map_roundtrip():
    hands = [_snap(0, [0, 30, 0, 31]), _snap(1, [42, 0, 0, 0]), _snap(2, [0, 0, 84, 0])]
    cmap = build_class_map(hands)
    assert cmap[0] == 0  # pass is always class 0
    # every observed action maps to a distinct in-range class, and inverse recovers it.
    inv = {v: k for k, v in cmap.items()}
    assert len(inv) == len(cmap)
    for h in hands:
        for _pos, action in hand_decisions(h):
            assert inv[cmap[action]] == action
    # ascending bid order after pass.
    non_pass = [a for a in cmap if a != 0]
    assert sorted(non_pass) == [a for a in sorted(cmap, key=cmap.get) if a != 0]


def test_hand_decisions_follow_bidding_order():
    snap = _snap(dealer=1, bids=[10, 20, 30, 40])  # arbitrary distinct values
    order = bidding_order(1)  # (2, 3, 0, 1)
    dec = hand_decisions(snap)
    assert [a for _p, a in dec] == [snap["bids"][s] for s in order]
