# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The seed-varied Zeb seat (#190): one planner seed per deal, the same in
every arm and both halves; one seed is the plain player."""
from __future__ import annotations

from forge.oracle.rng import deal_from_seed
from forge.zeb.game import apply_action, is_terminal
from forge.zeb.types import BidState, GamePhase, ZebGameState

from arena.seed_varied import SeedVaried, deal_key
from arena.walt_play import WaltPlay


def _player(seed: int = 0) -> WaltPlay:
    return WaltPlay(level=0, n_outer=16, seed=seed)    # the random field: quick, and any player will do


def _state(seed: int, bidder: int = 0, rotate: int = 0) -> ZebGameState:
    hands = [tuple(h) for h in deal_from_seed(seed)]
    hands = hands[rotate:] + hands[:rotate]
    return ZebGameState(
        hands=tuple(hands), dealer=(bidder + 3) % 4, phase=GamePhase.PLAYING,
        bid_state=BidState(bids=tuple(30 if s == bidder else 0 for s in range(4)), high_bidder=bidder, high_bid=30),
        decl_id=4, bidder=bidder, played=frozenset(), play_history=(), current_trick=(),
        trick_leader=bidder, team_points=(0, 0),
    )


def test_one_seed_is_the_plain_player_byte_for_byte():
    plain, varied = _player(seed=3), SeedVaried(_player(), pool=1, offset=3)
    state = _state(7)
    while not is_terminal(state):
        a, b = plain.choose([state], [30]), varied.choose([state], [30])
        assert a == b and plain.last_values == varied.last_values
        state = apply_action(state, a[0])


def test_the_deal_picks_the_seed_and_a_rotation_is_the_same_deal():
    varied = SeedVaried(_player(), pool=16)
    seeds = {varied.seed_of(_state(s)) for s in range(200)}
    assert len(seeds) == 16, "200 deals reach every seed of the pool"
    for s in range(20):
        assert deal_key(_state(s)) == deal_key(_state(s, rotate=1)) == deal_key(_state(s, rotate=2))
    # The pick holds at every decision of the hand.
    state, seed = _state(11), varied.seed_of(_state(11))
    while not is_terminal(state):
        assert varied.seed_of(state) == seed
        state = apply_action(state, varied.choose([state], [30])[0])


def test_a_batch_plays_each_state_at_its_own_seed():
    varied = SeedVaried(_player(), pool=4, offset=10)
    states = [_state(s) for s in range(8)]
    batch = varied.choose(states, [30] * 8)
    for s, slot in zip(states, batch):
        assert slot == _player(seed=varied.seed_of(s)).choose([s], [30])[0]
    assert "seeds=10..13" in repr(varied)


def test_a_wrapper_that_calls_the_planner_gets_the_deals_seed():
    # RegularsPlay and BookPlay drive `.walt` directly; the pooled planner
    # sends each state to its deal's seed there too.
    varied = SeedVaried(_player(), pool=4)
    states = [_state(s) for s in range(6)]
    pooled = varied.walt.decide_batch(states, [30] * 6)
    for s, (tile, q) in zip(states, pooled):
        assert (tile, q) == _player(seed=varied.seed_of(s)).walt.decide_batch([s], [30])[0]
