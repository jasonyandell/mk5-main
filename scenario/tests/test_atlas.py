"""The atlas. The mold: the compiled count of a deal's positions per layer
is the referee's, on seeded random deals. The pour: Walt's self-play line
replays legally to its recorded ending, every kept splat reproduces the
planner's action values, forced positions label nothing, and the dataset
survives the disk exactly. The readers: advantages, success sets and
holding predictors on a hand-built position."""
import dataclasses
import json
import math

import numpy as np
import pytest

from arena.walt_play import _declared_trump, parse_walt_spec
from scenario.atlas import (
    Position, _COLUMNS, _main, advantages, calibration, deal, fiber, holding_predictors,
    in_plan_units, load, makes, mold, paired_advantages, play_waves, pour, save, success_set,
)
from scenario.compiled import CompiledWalt
from scenario.planner import Config, Rung
from scenario.record import empty_record
from scenario.rules import ALL_TILES, bit, rules_for, tiles_of

SPEC = "k1,n8,m2"
BID = 30


def test_the_mold_of_seed_one_is_the_prototypes():
    """The counts scratch/atlas/atlas1.py logged for its seed-1 deal."""
    hands = deal(1)
    assert _declared_trump(hands[0]) == 6
    assert mold(hands, 6, 0, max_layer=10) == [1, 7, 9, 13, 67, 402, 936, 2258, 5081, 25405, 50083]


@pytest.mark.parametrize("seed, leader", [(1, 0), (7, 2), (23, 3)])
def test_compiled_mold_matches_the_referee(seed, leader):
    hands = deal(seed)
    decl = _declared_trump(hands[leader])
    fast = CompiledWalt(Config((Rung(1, None),)))
    assert fast.mold(hands, decl, leader, max_layer=10) == mold(hands, decl, leader, max_layer=10)


def test_mold_refuses_a_misdeal():
    hands = deal(1)
    with pytest.raises(ValueError):
        mold([hands[0], hands[0], hands[2], hands[3]], 6, 0)
    with pytest.raises(ValueError):
        CompiledWalt(Config((Rung(1, None),))).mold([hands[0], hands[0], hands[2], hands[3]], 6, 0)


# -- deals ---------------------------------------------------------------------------


def test_a_fiber_holds_one_hand_and_deals_the_rest():
    for seat in (0, 2):
        deals = fiber(1, 6, seat)
        assert len(set(deals)) == 6
        for d in deals:
            assert d[seat] == deal(1)[seat]
            assert d[0] | d[1] | d[2] | d[3] == ALL_TILES and all(h.bit_count() == 7 for h in d)


# -- the pour ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def poured():
    return pour([deal(s) for s in (1, 2, 3)], 0, BID, _declared_trump, parse_walt_spec(SPEC).walt)


def _replay(p, d):
    """Deal d's plays through Record.advance, each checked legal; the
    records before each play."""
    hands = [int(h) for h in p.hands[d]]
    rules = rules_for(int(p.decl[d]))
    rec, recs = empty_record(p.bidder), []
    for seat, tile in p.plays[d]:
        assert seat == rec.seat
        assert (rules.legal(hands[seat] & ~rec.played, rec.led) >> int(tile)) & 1
        recs.append(rec)
        rec = rec.advance(int(tile), rules)
    return recs, rec


def test_the_plays_replay_legally_to_the_recorded_ending(poured):
    for d in range(len(poured.hands)):
        _, end = _replay(poured, d)
        assert end.finished
        assert end.banked == tuple(int(b) for b in poured.banked[d])
        assert int(poured.decl[d]) == _declared_trump(int(poured.hands[d, 0]))


def test_forced_positions_are_played_but_not_kept(poured):
    for d in range(len(poured.hands)):
        recs, _ = _replay(poured, d)
        hand = [int(h) for h in poured.hands[d]]
        rules = rules_for(int(poured.decl[d]))
        planned = {k for k, r in enumerate(recs) if rules.legal(hand[r.seat] & ~r.played, r.led).bit_count() > 1}
        assert set(poured.n_plays[poured.deal == d].tolist()) == planned
        assert len(planned) < 28


def test_every_splat_reproduces_its_action_values(poured):
    """Leaves are margins under every utility; `q` is their mean in the plan's
    units. The default spec plans under `make`, so this is the make case."""
    assert poured.utility == "make"
    assert np.abs(poured.outcomes).max() <= 42 and len(np.unique(poured.outcomes)) > 2
    for pos in poured.positions():
        w = pos.weights
        assert pos.outcomes.shape == (len(pos.actions), len(w))
        for a, q in zip(pos.actions, pos.q):
            u = in_plan_units(pos, pos.outcomes[pos.actions.index(a)])
            assert math.isclose(float(w @ u) / w.sum(), q, abs_tol=1e-9)
        legal = rules_for(pos.decl_id).legal(pos.own, pos.record.led)
        assert pos.actions == tuple(tiles_of(legal))


def test_the_chosen_tile_is_the_planners_and_the_worlds_are_the_viewers(poured):
    for pos in poured.positions():
        best = (max if pos.sign > 0 else min)(sorted(zip(pos.q, pos.actions), key=lambda x: x[1]),
                                              key=lambda x: x[0])
        assert pos.chosen == best[1]
        assert np.all(pos.worlds[:, pos.viewer] == pos.own)
        union = np.bitwise_or.reduce(pos.worlds, axis=1)
        assert np.all(union | pos.record.played == ALL_TILES)


def test_the_pour_is_deterministic_and_the_plain_line_is_the_same(poured):
    again = pour([deal(s) for s in (1, 2, 3)], 0, BID, _declared_trump, parse_walt_spec(SPEC).walt)
    for c in _COLUMNS:
        assert np.array_equal(getattr(poured, c), getattr(again, c))
    plain = play_waves([deal(s) for s in (1, 2, 3)], 0, BID, list(poured.decl), parse_walt_spec(SPEC).walt)
    assert [h.plays for h in plain] == [tuple(map(tuple, poured.plays[d].tolist())) for d in range(3)]


def test_save_and_load_round_trip_exactly(poured, tmp_path):
    save(poured, tmp_path, spec=SPEC)
    back = load(tmp_path)
    assert (back.bidder, back.bid, back.utility) == (poured.bidder, poured.bid, poured.utility)
    for c in _COLUMNS:
        a, b = getattr(poured, c), getattr(back, c)
        assert a.dtype == b.dtype and np.array_equal(a, b), c


def test_calibration_reads_the_opening_leads(poured):
    c = calibration(poured)
    assert c.n == 3 and -42 <= c.forecast_margin <= 42 and 0 <= c.forecast_make <= 1
    bt = np.asarray(poured.banked, dtype=int)
    assert c.real_make == np.mean(bt[:, 0] >= BID)
    assert c.real_margin == np.mean(bt[:, 0] - bt[:, 1])


def test_the_command_line_pours_and_molds(tmp_path, capsys):
    _main(["pour", "--deals", "2", "--spec", SPEC + ",u=points", "--seed", "5", "--out", str(tmp_path)])
    p = load(tmp_path)
    assert p.utility == "points" and len(p.hands) == 2
    assert [int(h) for h in p.hands[1]] == list(deal(6))
    assert len(json.loads((tmp_path / "manifest.json").read_text())["commit"].removesuffix("-dirty")) == 40
    c = calibration(p)
    assert c.forecast_margin is not None
    _main(["mold", "--seed", "1", "--max-layer", "3"])
    out = capsys.readouterr().out
    assert "make share" in out and "layer  3:             13" in out


# -- reading a position --------------------------------------------------------------


def _position(utility="points"):
    """Seat 1 (a defender) to play after seat 0's lead, three scenarios
    that differ by one tile swapped between seats 2 and 3."""
    hands = deal(1)
    lead = tiles_of(hands[0])[0]
    a, b = tiles_of(hands[1])[:2]
    rem = [h & ~bit(lead) for h in hands]
    x, y = tiles_of(rem[2])[0], tiles_of(rem[3])[0]
    swapped = list(rem)
    swapped[2], swapped[3] = rem[2] ^ bit(x) ^ bit(y), rem[3] ^ bit(y) ^ bit(x)
    outcomes = np.array([[10, -20, 4], [0, 0, 30]], dtype=np.int8)
    weights = np.array([1.0, 1.0, 2.0])
    return Position(
        deal=0, plays=((0, lead),), hands=hands, decl_id=6, bidder=0, bid=BID, utility=utility,
        actions=(a, b), q=(outcomes @ weights) / weights.sum(), outcomes=outcomes,
        worlds=np.array([rem, swapped, swapped], dtype=np.uint32), weights=weights,
        chosen=a, banked=(30, 12),
    ), (a, b, x, y)


def test_advantages_are_the_movers_and_pair_by_world():
    pos, (a, b, _, _) = _position()
    assert pos.viewer == 1 and pos.sign == -1
    assert list(pos.q) == [-0.5, 15.0]
    assert advantages(pos) == {a: 7.75, b: -7.75}
    paired = paired_advantages(pos)
    assert paired.tolist() == [[-5.0, 10.0, 13.0], [5.0, -10.0, -13.0]]
    assert (paired @ pos.weights / pos.weights.sum()).tolist() == [7.75, -7.75]


def test_success_sets_read_any_bid_off_a_points_splat():
    pos, (a, b, _, _) = _position()
    assert success_set(pos, a).tolist() == []
    assert success_set(pos, b).tolist() == [2]
    assert success_set(pos, b, 36).tolist() == [2]
    assert success_set(pos, b, 37).tolist() == []
    assert success_set(pos, a, 15).tolist() == [0, 2]


def test_a_make_splat_reads_any_bid_too():
    """Under `make` the plan is scored by the make indicator but the leaves
    still hold margins, so the other bids are tails of the same splat."""
    pos, (a, b, _, _) = _position("make")
    assert in_plan_units(pos, pos.outcomes).tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert makes(pos, b).tolist() == [False, False, True]
    assert success_set(pos, b, 31).tolist() == [2]
    assert success_set(pos, a, 15).tolist() == [0, 2]


def test_holding_predictors_say_what_must_be_true():
    pos, (_, b, x, y) = _position()
    base, pred = holding_predictors(pos, success_set(pos, b))
    assert base == 0.5
    assert pred[(3, x)] == (2 / 3, 0.75) and pred[(2, x)] == (0.0, 0.25)
    assert pred[(2, y)] == (2 / 3, 0.75) and pred[(3, y)] == (0.0, 0.25)
    z = tiles_of(int(pos.worlds[0, 2]) & ~bit(x))[0]
    assert pred[(2, z)] == (0.5, 1.0)
    assert all(s != 1 for s, _ in pred) and all(t != pos.plays[0][1] for _, t in pred)
    hidden = ALL_TILES & ~pos.record.played & ~pos.own
    assert {t for _, t in pred} == set(tiles_of(hidden))
