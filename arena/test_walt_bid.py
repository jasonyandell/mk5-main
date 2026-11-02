# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Walt's arena bidder: Zeb's bidder (forecast + chooseBid) over Walt's planner."""
import random

import pytest

from arena.auction import PASS, BidContext, legal_bids
from arena.cli import parse_bidder, parse_play
from arena.engine import ArenaConfig
from arena.match import run_match
from arena.walt_bid import parse_walt_bidder
from arena.zeb_bid import TARGETS, ZebBidder, best_at, choose_bid
from scenario.planner import Rung

HAND = (0, 1, 6, 8, 9, 15, 20)


def _ctx(high_bid=0, high_seat=-1, bids=(-1, -1, -1, -1), seat=1, dealer=0):
    return BidContext(hand=HAND, seat=seat, dealer=dealer, bids=bids, high_bid=high_bid,
                      high_seat=high_seat, legal=legal_bids(high_bid))


def test_spec():
    b = parse_walt_bidder("k1,n32,m4,c0.7")
    assert isinstance(b, ZebBidder)
    assert (b.cutoff, b.mode, b.over, b.cal, b.label) == (0.7, "hand", False, False, "waltbid:k1,n32,m4,c0.7")
    assert b.walt.config.rungs == (Rung(4, 4), Rung(32, None, 16))
    assert (b.walt.config.utility, b.walt.config.field) == ("make", None)


def test_default_spec_is_k1_n160_m8():
    b = parse_walt_bidder("")
    assert b.cutoff == 0.8
    assert b.walt.config.rungs == (Rung(8, 4), Rung(160, None, 16))
    # an option overrides its default token only
    assert parse_walt_bidder("n32").walt.config.rungs == (Rung(8, 4), Rung(32, None, 16))


@pytest.mark.parametrize("token", ["u=points", "DTNT=random", "py"])
def test_refuses_what_the_forecast_cannot_read(token):
    with pytest.raises(ValueError, match=token):
        parse_walt_bidder(f"k0,n16,{token}")


def test_forecast_is_seven_pip_declarations_weakly_decreasing():
    fc = parse_walt_bidder("k1,n16,m4").forecast(HAND)
    assert [d for d, _ in fc] == list(range(7))
    for _, pm in fc:
        assert len(pm) == len(TARGETS) == 13
        assert all(0.0 <= p <= 1.0 for p in pm)
        assert all(a >= b for a, b in zip(pm, pm[1:]))


def test_forecast_is_deterministic():
    assert parse_walt_bidder("k1,n16,m4").forecast(HAND) == parse_walt_bidder("k1,n16,m4").forecast(HAND)


def test_bid_and_declare():
    b = parse_walt_bidder("k1,n16,m4")
    rng = random.Random(0)
    # seat 1's partner (seat 3) holds the high bid: pass
    assert b.bid(_ctx(high_bid=30, high_seat=3, bids=(0, -1, 0, 30)), rng) == PASS
    for cutoff in (0.3, 0.8):
        b.cutoff = cutoff
        call = b.bid(_ctx(), rng)
        fc = b.forecast(HAND)
        hit = choose_bid(fc, 30, cutoff)
        assert call == (PASS if hit is None else hit[0])
        assert call == PASS or call in range(30, 43)
        decl = b.declare(HAND, max(call, 30), rng)
        assert decl in range(7) and decl == best_at(fc, max(call, 30))[0]


def test_arena_smoke():
    """`waltbid:k0,n16+walt:k0,n16` against `heuristic+walt:k0,n16`, two games to one mark."""
    play = dict(model=None, n_samples=1, device="cpu", seed=0)
    result = run_match(
        bid_a=parse_bidder("waltbid:k0,n16", device="cpu", gus_adapter=None),
        bid_b=parse_bidder("heuristic", device="cpu", gus_adapter=None),
        play_a=parse_play("walt:k0,n16", **play), play_b=parse_play("walt:k0,n16", **play),
        n_games=2, cfg=ArenaConfig(marks_to_win=1, base_seed=7),
    )
    assert len(result.games) == 2
    for g in result.games:
        assert g.marks[g.winner_team] >= 1
        # Walt's seats (team A) pass or bid a point bid; never 84
        assert all(h.bids[s] == PASS or 30 <= h.bids[s] <= 42
                   for h in g.hands for s in range(4) if s % 2 == g.a_team)
