"""ValueBidder (jud v0, #32) — CPU, fast.

The load-bearing test is featurization-consistency: the ValueBidder's
decision-time view of an auction (BidContext, with -1 = not-yet-bid) must
featurize BYTE-IDENTICALLY to the corpus view of the same completed auction
(`featurize_snapshot`). If it does not, the head is served out of distribution
and every price is wrong. The rest pins the min-positive-utility convention,
pass-when-hopeless, declare caching + cold path, and the CLI registry wiring.
"""
from __future__ import annotations

import json
import random

import torch

from arena.auction import PASS, BidContext, legal_bids
from arena.bidders import HeuristicBidder
from arena.engine import ArenaConfig
from arena.match import run_match, snapshot_rows
from arena.play import RandomPlay
from champion.margin_net import featurize, featurize_snapshot
from champion.value_bidder import ValueBidder, load_margin_net
from forge.bidding.schema import EVAL_DECLS

RNG = random.Random(0)


# --------------------------------------------------------------------- #
#  Mock head: a fixed pmake table, so bid logic is tested in isolation   #
# --------------------------------------------------------------------- #

class FakeMargin:
    """Stands in for MarginNet: returns a fixed {decl: {threshold: P}} table."""

    def __init__(self, table: dict[int, dict[int, float]]):
        self._table = table

    def eval(self):
        return self

    def pmake_table(self, hand, bids, bidder, dealer):
        return {d: dict(row) for d, row in self._table.items()}


def _flat_table(decls_to_p: dict[int, float]) -> dict[int, dict[int, float]]:
    """A monotone-flat table: every threshold 30..42 gets the decl's P."""
    return {d: {t: p for t in range(30, 43)} for d, p in decls_to_p.items()}


def _ctx(hand=(0, 1, 2, 3, 4, 5, 6), *, seat=1, dealer=0, marks=(0, 0)) -> BidContext:
    return BidContext(
        hand=hand, seat=seat, dealer=dealer, bids=(-1, -1, -1, -1),
        high_bid=0, high_seat=-1, legal=legal_bids(0), marks=marks, marks_to_win=7,
    )


# --------------------------------------------------------------------- #
#  1. Featurization consistency: BidContext view == corpus view          #
# --------------------------------------------------------------------- #

def _decision_time_bids(snap: dict) -> tuple[int, ...]:
    """Reconstruct the bidder's decision-time bid vector from a completed
    snapshot: seats BEFORE the bidder in auction order keep their bid, the
    bidder's own seat and every later seat are -1 (not yet bid)."""
    dealer, bidder = int(snap["dealer"]), int(snap["bidder"])
    order = [(dealer + s) % 4 for s in (1, 2, 3, 0)]
    bidder_pos = order.index(bidder)
    bids = [-1, -1, -1, -1]
    for pos, seat in enumerate(order):
        if pos < bidder_pos:
            bids[seat] = int(snap["bids"][seat])
    return tuple(bids)


def test_query_bids_maps_pass_and_not_yet_to_zero():
    ctx = BidContext(
        hand=(0,), seat=2, dealer=0, bids=(30, -1, -1, 0),
        high_bid=30, high_seat=0, legal=(31,),
    )
    assert ValueBidder._query_bids(ctx) == (30, 0, 0, 0)


def test_featurization_matches_corpus_view_handmade():
    # dealer 0 → order seats 1,2,3,0. Bidder = seat 2 wins at 34; seat 1 bid 30,
    # seats 3 & 0 passed. At seat 2's decision time it heard only seat 1.
    snap = {
        "hands": [[7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20],
                  [0, 1, 2, 3, 4, 5, 6], [21, 22, 23, 24, 25, 26, 27]],
        "bids": [0, 30, 34, 0], "bidder": 2, "dealer": 0, "decl_id": 5,
        "bidder_team_pts": 30, "seed": 1, "hand_idx": 0,
    }
    ctx = BidContext(
        hand=tuple(snap["hands"][2]), seat=2, dealer=0,
        bids=_decision_time_bids(snap), high_bid=30, high_seat=1, legal=legal_bids(30),
    )
    qbids = ValueBidder._query_bids(ctx)
    x_serve = featurize(tuple(snap["hands"][2]), qbids, 2, 0, snap["decl_id"])
    assert torch.equal(x_serve, featurize_snapshot(snap))


def test_featurization_matches_corpus_view_over_real_match(tmp_path):
    # Every real snapshot: the decision-time reconstruction must featurize
    # byte-identically to the corpus view, for the winning declaration.
    result = run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
        n_games=4, cfg=ArenaConfig(marks_to_win=2, base_seed=11),
    )
    snaps = snapshot_rows(result)
    assert snaps, "match should produce contracted hands"
    for snap in snaps:
        bidder = int(snap["bidder"])
        qbids = tuple(b if b > 0 else 0 for b in _decision_time_bids(snap))
        x_serve = featurize(
            tuple(snap["hands"][bidder]), qbids, bidder,
            int(snap["dealer"]), int(snap["decl_id"]),
        )
        assert torch.equal(x_serve, featurize_snapshot(snap)), snap


# --------------------------------------------------------------------- #
#  2. Min-positive-utility convention (mock tables)                      #
# --------------------------------------------------------------------- #

def test_bids_minimum_positive_utility_not_maximum():
    # A monster hand: every contract, including 84, has positive utility. The
    # min-positive convention (rung #21) must take the CHEAPEST — bid 30 — where
    # a #31 maximizer would climb toward 84. This is the whole point of the
    # convention (bid magnitude measured dead).
    table = _flat_table({5: 0.95, 3: 0.80})
    bidder = ValueBidder(FakeMargin(table))
    assert bidder.bid(_ctx(), RNG) == 30


def test_declares_argmax_exceedance_declaration():
    # decl 4 dominates → the chosen bid is declared under decl 4, cached for declare.
    table = _flat_table({4: 0.90, 2: 0.60, 9: 0.55})
    bidder = ValueBidder(FakeMargin(table))
    ctx = _ctx()
    value = bidder.bid(ctx, RNG)
    assert value == 30
    assert bidder.declare(ctx.hand, value, RNG) == 4


def test_passes_when_hopeless():
    # Every declaration well below make: all point bids and 84 have negative
    # utility, so the walk falls through to PASS — no heuristic prefilter needed.
    table = _flat_table({5: 0.10, 3: 0.08})
    bidder = ValueBidder(FakeMargin(table))
    assert bidder.bid(_ctx(), RNG) == PASS


def test_boundary_p_half_is_not_positive_utility():
    # P == 0.5 gives exactly zero marks-to-7 utility at an even score; strictly
    # `> margin` (0) is required, so a coin-flip contract passes.
    table = _flat_table({5: 0.5})
    bidder = ValueBidder(FakeMargin(table))
    assert bidder.bid(_ctx(), RNG) == PASS


# --------------------------------------------------------------------- #
#  3. declare(): cached from bid time, plus the forced-open cold path    #
# --------------------------------------------------------------------- #

def test_declare_uses_cached_decl_from_bid_time():
    table = _flat_table({6: 0.92, 1: 0.70})
    bidder = ValueBidder(FakeMargin(table))
    ctx = _ctx()
    value = bidder.bid(ctx, RNG)
    # The cached declaration is served verbatim (no re-argmax that could drift).
    assert bidder.declare(ctx.hand, value, RNG) == bidder._best_decl(value, table)


def test_declare_cold_path_returns_legal_declaration():
    # A forced-open value the bidder never priced (arena force_shaker at redeal
    # cap): declare must still return a valid declaration, via the empty-auction
    # cold path — never crash.
    table = _flat_table({3: 0.40, 7: 0.55})
    bidder = ValueBidder(FakeMargin(table))
    decl = bidder.declare((0, 1, 2, 3, 4, 5, 6), 30, RNG)
    assert decl in EVAL_DECLS


# --------------------------------------------------------------------- #
#  4. Real head + a full CPU match (registry wiring, end-to-end)         #
# --------------------------------------------------------------------- #

def test_parse_bidder_registers_margin():
    from arena.cli import parse_bidder
    from champion.utility import MarksToSeven

    bidder = parse_bidder("margin:wp", device="cpu", gus_adapter=None)
    assert isinstance(bidder, ValueBidder)
    assert isinstance(bidder.utility, MarksToSeven)
    # Bare `margin` also defaults to the value-native marks-to-7 utility.
    assert isinstance(
        parse_bidder("margin", device="cpu", gus_adapter=None).utility, MarksToSeven
    )


def test_real_head_bid_then_declare_are_consistent():
    model = load_margin_net(device="cpu")
    bidder = ValueBidder(model)
    ctx = _ctx(hand=(0, 7, 14, 21, 2, 9, 16))  # arbitrary 7-domino hand
    value = bidder.bid(ctx, RNG)
    if value != PASS:
        decl = bidder.declare(ctx.hand, value, RNG)
        assert decl in EVAL_DECLS


def test_value_bidder_plays_a_full_cpu_match():
    # The real head drives a 2-game match to completion against random play —
    # bid, declare, forced opens, scoring — proving the bidder is arena-legal.
    # (The lens:ev play integration is exercised by the Phase-2 GPU A/B.)
    model = load_margin_net(device="cpu")
    result = run_match(
        bid_a=ValueBidder(model), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=3), play_b=RandomPlay(seed=4),
        n_games=2, cfg=ArenaConfig(marks_to_win=2, base_seed=5),
    )
    assert result.n_games == 2
    assert all(g.hands for g in result.games)
