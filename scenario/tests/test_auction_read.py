# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: MIT

"""The auction in a reading's likelihood (#149, `Situation.auction`): per
read seat, a table over the slot of the hand the world says it was dealt
(`ceiling_slot`), multiplied into the seat's factor. The core's slot is the
arena's `best_trump` ceiling; referee = core to the bit on every draw path;
a table of ones is no evidence; the draw follows the table; and the auction
is refused without a reading."""

import random

import pytest
import scenario_core

from arena.hand_metrics import best_trump
from arena.regulars import BidderBelief
from scenario.compiled import CompiledWalt
from scenario.planner import AUCTION_SLOTS, READ_ENUM, Config, Reading, Rung, Walt, ceiling_slot, situation_from_state
from scenario.record import remaining_hand
from scenario.rules import tiles_of
from scenario.tests.test_compiled import _same_analysis
from scenario.tests.test_field import NET
from scenario.tests.test_posterior import LATE, SAMPLED
from scenario.tests.test_reading import EARLY, MARGIN, RANDOM, _fiber, affordable

CONFIGS = [
    Config((Rung(16, None, 6),), "make", seed=3, field=NET, draw="posterior", draw_budget=20_000,
           reading=Reading("uniform", MARGIN, 0.28)),
    Config((Rung(12, None, 4),), "points", seed=5, field=NET, draw="posterior", draw_budget=20_000,
           reading=Reading("consistent", MARGIN, 0.1)),
    Config((Rung(16, 6, 6),), "make", seed=7, field=NET, draw="posterior", draw_budget=40,
           reading=Reading("uniform", 0, 0.05)),                          # the budget runs out
]


def _auction(state, rng: random.Random):
    """A regular's tables on the viewer's two opponents: one passed at 30, one bid it,
    under a belief that has seen a few hands; the partner unread."""
    viewer = (state.trick_leader + len(state.current_trick)) % 4
    out = [None] * 4
    for seat in range(4):
        if seat % 2 != viewer % 2:
            b = BidderBelief()
            for _ in range(3):
                b.learn(rng.choice((0, 30)), 30, rng.randrange(AUCTION_SLOTS))
            out[seat] = b.table(rng.choice((0, 30)), 30)
    return tuple(out)


def test_the_cores_slot_is_the_arenas_ceiling():
    rng = random.Random(149)
    for _ in range(20_000):
        hand = sum(1 << t for t in rng.sample(range(28), 7))
        best = best_trump(tuple(tiles_of(hand)), 3)
        want = 0 if best is None else 1 + best.bid_ceiling
        assert ceiling_slot(hand) == scenario_core.ceiling_slot(hand) == want


def test_compiled_matches_the_planner_with_the_auction():
    """Referee = core to the bit with auction tables on every draw path (the
    exact fiber, the enumerated draw, rejection, and the budget running out),
    proposals and fallbacks included; and the auction moves the plan."""
    rng = random.Random(7)
    paths = {"exact": 0, "enumerated": 0, "rejection": 0}
    fell = moved = 0
    for cfg in CONFIGS:
        ref, fast = Walt(cfg), CompiledWalt(cfg)
        for state in affordable(EARLY + SAMPLED[::2] + LATE[::2] + RANDOM[::2]):
            auction = _auction(state, rng)
            a = ref.analyze(state, 30, auction=auction)
            _same_analysis(a, fast.analyze(state, 30, auction=auction))
            assert (ref.nodes, ref.proposals, ref.fallbacks) == (fast.nodes, fast.proposals, fast.fallbacks)
            d = fast.decide_batch([state], [30], [auction])[0]
            assert d[0] == a.tile and d[1] == a.q
            size = _fiber(state)
            path = "exact" if size <= cfg.rungs[-1].n_worlds else "enumerated" if size <= READ_ENUM else "rejection"
            paths[path] += not ref.fallbacks
            fell += ref.fallbacks
            moved += a.q != fast.analyze(state, 30).q
    assert min(paths.values()) >= 3 and fell >= 1 and moved >= 8


def test_a_table_of_ones_is_no_evidence():
    ones = tuple(1.0 for _ in range(AUCTION_SLOTS))
    for cfg in CONFIGS[:2]:
        fast = CompiledWalt(cfg)
        for state in SAMPLED[:3] + LATE[:3]:
            plain = fast.analyze(state, 30)
            _same_analysis(plain, fast.analyze(state, 30, auction=(ones, ones, ones, ones)))
        for state in affordable(SAMPLED)[:2] + LATE[:2]:
            _same_analysis(fast.analyze(state, 30), Walt(cfg).analyze(state, 30, auction=(ones, None, ones, None)))


def test_the_draw_follows_the_table():
    """A table that rules out every slot but one leaves only worlds that deal
    that seat a hand in that slot, wherever such worlds exist."""
    cfg = CONFIGS[0]
    checked = 0
    for state in SAMPLED + LATE:
        sit = situation_from_state(state, 30)
        seat = (sit.viewer + 1) % 4
        slot = ceiling_slot(remaining_hand(state, seat) | sit.played_by(seat))   # the real deal's: never empty
        table = tuple(1.0 if s == slot else 0.0 for s in range(AUCTION_SLOTS))
        auction = tuple(table if s == seat else None for s in range(4))
        a = CompiledWalt(cfg).analyze(state, 30, auction=auction)
        if not a.worlds:
            continue
        kept = [w for w, x in zip(a.worlds, a.weights) if x > 0]
        assert kept and all(ceiling_slot(w[seat] | sit.played_by(seat)) == slot for w in kept)
        checked += 1
    assert checked >= 5


def test_the_auction_is_validated():
    state = SAMPLED[0]
    ones = tuple(1.0 for _ in range(AUCTION_SLOTS))
    plain = Config((Rung(8, None, 4),), field=NET)
    with pytest.raises(ValueError):
        Walt(plain).decide(state, 30, auction=(ones, None, None, None))      # no reading to draw with
    with pytest.raises(ValueError):
        CompiledWalt(plain).decide(state, 30, auction=(ones, None, None, None))
    with pytest.raises(ValueError):
        Walt(CONFIGS[0]).decide(state, 30, auction=(ones[:-1], None, None, None))
    with pytest.raises(ValueError):
        CompiledWalt(CONFIGS[0]).decide(state, 30, auction=((2.0,) * AUCTION_SLOTS, None, None, None))
    with pytest.raises(ValueError):
        CompiledWalt(CONFIGS[0]).decide(state, 30, auction=(ones, None, None))
    with pytest.raises(ValueError):
        scenario_core.ceiling_slot(1 << 28)
