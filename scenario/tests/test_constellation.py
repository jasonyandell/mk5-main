"""Constellations: one remaining game, whatever came before, and a table of
Zeb's plan values that is exact where its field reads only constellations."""
import random

import pytest

from scenario.constellation import Ladder, automorphisms, form, substitute, substitutions
from scenario.record import empty_record
from scenario.rules import rules_for, tiles_of


def _late(n_deals: int, seed: int, min_plays: int = 20):
    rng = random.Random(seed)
    for _ in range(n_deals):
        decl, bidder = rng.randrange(7), rng.randrange(4)
        rules = rules_for(decl)
        d = list(range(28))
        rng.shuffle(d)
        hands = [sum(1 << t for t in d[7 * i:7 * i + 7]) for i in range(4)]
        rec = empty_record(bidder)
        stop = rng.randrange(min_plays, 27)
        while rec.plays_so_far() < stop:
            s = rec.seat
            t = rng.choice(tiles_of(rules.legal(hands[s], rec.led)))
            hands[s] &= ~(1 << t)
            rec = rec.advance(t, rules)
        if len(tiles_of(rules.legal(hands[rec.seat], rec.led))) > 1:
            yield rules, rec, tuple(hands), bidder


def _pairs(n_deals: int, seed: int):
    for rules, rec, hands, bidder in _late(n_deals, seed):
        for d, e, rec2, hands2 in substitutions(rules, rec, hands, 30, bidder % 2)[:2]:
            yield rules, rec, hands, rec2, hands2, d, e, bidder


def _carried(q: dict, iso: dict) -> dict:
    return {iso[t]: v for t, v in q.items()}


def _iso(rules, rec, hands, rec2, hands2, bidder) -> dict:
    """The tile bijection two equal constellations' labelings define."""
    s = rec.seat
    return dict(zip(form(rules, s, hands[s], rec, 30, bidder % 2)[1], form(rules, s, hands2[s], rec2, 30, bidder % 2)[1]))


def test_a_substitution_changes_the_history_and_keeps_the_constellation():
    pairs = list(_pairs(60, 1))
    assert len(pairs) >= 20
    for rules, rec, hands, rec2, hands2, d, e, bidder in pairs:
        s = rec.seat
        assert rec2.played != rec.played and hands2 != hands
        a, oa = form(rules, s, hands[s], rec, 30, bidder % 2)
        b, ob = form(rules, s, hands2[s], rec2, 30, bidder % 2)
        assert a == b
        # The labelings are the isomorphism: it moves the dead tile into the
        # live one's standing, and maybe others along (a new top trump
        # pushes the rest down), and fixes every tile whose standing holds.
        iso = dict(zip(oa, ob))
        assert iso.get(d, e) != d and e in ob and d not in ob


def test_the_world_form_is_blind_to_voids_and_the_seat_form_is_not():
    for rules, rec, hands, bidder in _late(40, 2):
        s = rec.seat
        if not any(rec.forbid):
            continue
        cleared = rec.__class__(rec.played, rec.leader, rec.trick, rec.banked, (0, 0, 0, 0))
        assert form(rules, s, hands[s], rec, 30, 0, hands)[0] == form(rules, s, hands[s], cleared, 30, 0, hands)[0]
        live_void = any(f & ~rec.played for f in rec.forbid)
        if live_void:
            assert form(rules, s, hands[s], rec, 30, 0)[0] != form(rules, s, hands[s], cleared, 30, 0)[0]


def test_the_structural_ladder_plans_alike_on_every_substitution():
    """Equal constellations, different histories: the plan's values agree
    move for move at every level, and the table answers the second position
    with exactly what planning it gives."""
    ladder = Ladder("first")
    n = 0
    for rules, rec, hands, rec2, hands2, d, e, bidder in _pairs(80, 3):
        s, bt = rec.seat, bidder % 2
        iso = _iso(rules, rec, hands, rec2, hands2, bidder)
        for level in (1, 2):
            lv = ladder.level(level)
            direct = lv.plan(rules, s, hands2[s], rec2, 30, bt)
            assert _carried(lv.plan(rules, s, hands[s], rec, 30, bt), iso) == direct
            lv.decide(rules, s, hands[s], rec, 30, bt)
            tile, looked = lv.decide(rules, s, hands2[s], rec2, 30, bt)
            assert looked == direct and tile in direct
        n += 1
    assert n >= 20


def test_a_ladder_level_is_one_choice_per_information_state():
    """The level's tile at a seat is read from that seat's constellation
    alone: two positions with one constellation get the corresponding tile."""
    ladder = Ladder("first")
    lv = ladder.level(2)
    for rules, rec, hands, rec2, hands2, d, e, bidder in _pairs(60, 4):
        s, bt = rec.seat, bidder % 2
        if automorphisms(rules, s, hands[s], rec, 30, bt) > 1:
            continue   # symmetric: the corresponding tile is one of an orbit
        iso = _iso(rules, rec, hands, rec2, hands2, bidder)
        a = lv.move(rules, s, hands[s], rec, 30, bt)
        b = lv.move(rules, s, hands2[s], rec2, 30, bt)
        assert b == iso[a]


def test_the_net_ladder_keys_on_the_exact_information_state():
    from scenario.field import FieldNet
    net = FieldNet.from_bytes(open("zeb/net/margin_d250k_h128x64.fnet", "rb").read())
    lv = Ladder(net).level(1)
    rules, rec, hands, bidder = next(iter(_late(5, 5)))
    s = rec.seat
    key, order = lv.key(rules, s, hands[s], rec, 30, bidder % 2)
    assert order is None and rec.key() in key


def test_substitute_is_an_involution():
    for rules, rec, hands, bidder in _late(10, 6):
        live = tiles_of(((1 << 28) - 1) & ~rec.played)[0]
        dead = tiles_of(rec.played & ~sum(1 << t for t in rec.trick))[0]
        rec2, hands2 = substitute(rec, hands, live, dead)
        assert substitute(rec2, hands2, live, dead) == (rec, hands)


def test_automorphisms_are_counted():
    counts = [automorphisms(rules, rec.seat, hands[rec.seat], rec, 30, bidder % 2)
              for rules, rec, hands, bidder in _late(200, 7)]
    assert min(counts) >= 1


def test_the_ladder_refuses_a_fiber_it_cannot_enumerate():
    rules = rules_for(0)
    rec = empty_record(0)
    hands = [sum(1 << t for t in range(7 * i, 7 * i + 7)) for i in range(4)]
    with pytest.raises(ValueError):
        Ladder("first").level(1).plan(rules, 0, hands[0], rec, 30, 0)
