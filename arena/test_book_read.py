# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The book-aware reader (arena/book_read.py, #147): the stack's verdict as the
book player decides it, the book's likelihood per world, and its batch form."""
import random

import pytest

pytest.importorskip("scenario_core")

from forge.zeb.game import apply_action, is_terminal, legal_actions

from arena.book_play import ask
from arena.book_read import BookBelief, book_weights, parse_bookread_spec, verdict
from arena.test_book_play import playing, seat_of
from scenario.fiber import Fiber
from scenario.planner import situation_from_state
from scenario.record import remaining_hand
from scenario.rules import bit, mask_of


def _hands(seed: int, bid: int = 30):
    """Every decision of a hand the pure stack plays at all four seats."""
    state = playing(seed, seed % 7, bidder=seed % 4, bid=bid)
    while not is_terminal(state):
        yield state
        v = ask(state, bid)
        hand = state.hands[seat_of(state)]
        legal = sorted(hand[i] for i in legal_actions(state))
        tile = v.allowed[0] if v.policy is not None else legal[0]
        state = apply_action(state, list(hand).index(tile))


def _world(state) -> tuple[int, int, int, int]:
    return tuple(mask_of(t for t in state.hands[s] if t not in state.played) for s in range(4))


def test_the_verdict_is_the_book_players_deciding_set():
    asked = 0
    for seed in range(10):
        for state in _hands(seed):
            seat = seat_of(state)
            v = ask(state, 30)
            own = mask_of(t for t in state.hands[seat] if t not in state.played)
            plays = tuple((int(s), int(t)) for s, t in state.play_history)
            got = verdict(state.decl_id, seat, own, state.bidder, 30, plays)
            assert got == (None if v.policy is None else mask_of(v.allowed))
            asked += 1
    assert asked == 280


@pytest.mark.parametrize("likelihood", ["consistent", "uniform"])
def test_book_players_are_never_excluded_by_the_reading(likelihood):
    # Seats that play by the book leave the real deal a positive weight.
    for seed in range(6):
        for state in _hands(seed):
            sit = situation_from_state(state, 30)
            w, row = book_weights(sit, [_world(state)], 30, likelihood)
            assert w == [1.0] and not row["fell_back"]


def test_the_weights_are_the_books_verdicts_world_by_world():
    # On a small fiber, against a brute-force product over each read play.
    rng = random.Random(7)
    checked = 0
    for seed in range(4):
        states = list(_hands(seed))
        for state in (s for s in states if 14 <= len(s.play_history) <= 20):
            sit = situation_from_state(state, 30)
            rec, viewer = sit.record, sit.viewer
            fiber = Fiber(viewer, remaining_hand(state, viewer), rec.played,
                          tuple(rec.hand_size(s) for s in range(4)), rec.forbid)
            worlds = rng.sample(fiber.worlds(), min(fiber.size, 40))
            for likelihood in ("consistent", "uniform"):
                w, _ = book_weights(sit, worlds, 30, likelihood)
                raw = []
                for world in worlds:
                    x, before = 1.0, [0, 0, 0, 0]
                    for i, (seat, tile) in enumerate(sit.plays):
                        if seat in ((viewer + 1) % 4, (viewer + 3) % 4):
                            hand = (world[seat] | sit.played_by(seat)) & ~before[seat]
                            led = sit.plays[i - i % 4][1] if i % 4 else None
                            if _legal(state.decl_id, hand, led).bit_count() > 1:
                                a = verdict(sit.decl_id, seat, hand, sit.bidder, 30, sit.plays[:i])
                                if a is not None:
                                    x = 0.0 if not a >> tile & 1 else x / (a.bit_count() if likelihood == "uniform" else 1)
                        before[seat] |= bit(tile)
                    raw.append(x)
                total = sum(raw)
                assert w == pytest.approx([1.0] * len(worlds) if total == 0 else [x / total for x in raw], rel=1e-12)
                checked += 1
    assert checked > 10


def _legal(decl: int, hand: int, led: int | None) -> int:
    from scenario.rules import rules_for
    return rules_for(decl).legal(hand, led)


def test_the_batch_form_is_the_one_at_a_time_form():
    sits, worlds = [], []
    rng = random.Random(3)
    for seed in range(3):
        for state in list(_hands(seed))[5::6]:
            sit = situation_from_state(state, 30)
            rec = sit.record
            fiber = Fiber(sit.viewer, remaining_hand(state, sit.viewer), rec.played,
                          tuple(rec.hand_size(s) for s in range(4)), rec.forbid)
            sits.append(sit)
            worlds.append([fiber.draw(rng) for _ in range(16)])
    one, pooled = BookBelief("consistent", workers=1), BookBelief("consistent", workers=2)
    assert pooled.many(sits, worlds) == [one(s, w) for s, w in zip(sits, worlds)]
    assert pooled.log == one.log


def test_the_spec_is_plain_zeb_with_the_books_belief():
    r = parse_bookread_spec("uniform/k0,n8")
    assert r.belief.likelihood == "uniform" and r.zeb.walt.belief is r.belief
    with pytest.raises(ValueError):
        parse_bookread_spec("strict/k0,n8")


# ------------------------------------------------------------ the partner read (#150)

def test_the_partner_read_never_excludes_a_book_partner():
    # The book player's partner plays by the stack: the real deal keeps its weight.
    for seed in range(6):
        for state in _hands(seed):
            sit = situation_from_state(state, 30)
            for likelihood in ("consistent", "uniform"):
                w, row = book_weights(sit, [_world(state)], 30, likelihood, seats="partner")
                assert w == [1.0] and not row["fell_back"]


def test_the_partner_read_weighs_the_partners_hand_alone():
    # Worlds that agree on the partner's hand weigh the same, whatever the opponents hold.
    rng = random.Random(11)
    checked = 0
    for seed in range(4):
        for state in list(_hands(seed))[8:20:3]:
            sit = situation_from_state(state, 30)
            rec = sit.record
            fiber = Fiber(sit.viewer, remaining_hand(state, sit.viewer), rec.played,
                          tuple(rec.hand_size(s) for s in range(4)), rec.forbid)
            worlds = [fiber.draw(rng) for _ in range(40)]
            w, _ = book_weights(sit, worlds, 30, "consistent", seats="partner")
            p = (sit.viewer + 2) % 4
            by: dict[int, set[float]] = {}
            for world, x in zip(worlds, w):
                by.setdefault(world[p], set()).add(x)
            assert all(len(v) == 1 for v in by.values())
            checked += 1
    assert checked > 8


def _auctions():
    """Every seat's call in heuristic auctions over fresh deals, forced or not."""
    from arena.auction import run_auction
    from arena.bidders import HeuristicBidder
    from forge.oracle.rng import deal_from_seed
    bidder = HeuristicBidder()
    for seed in range(120):
        hands = tuple(tuple(h) for h in deal_from_seed(40_000 + seed))
        for dealer in range(4):
            for forced in (False, True):
                r = run_auction(hands, dealer, (bidder,) * 4, random.Random(0), force_shaker=forced)
                if r is not None:
                    yield hands, dealer, r


def test_the_calls_read_keeps_the_hand_that_made_them():
    from arena.book_read import calls_weight
    n = 0
    for hands, dealer, r in _auctions():
        for seat in range(4):
            assert calls_weight(seat, mask_of(hands[seat]), dealer, r.bids, r.winner, r.high_bid, r.decl_id) == 1.0
            n += 1
    assert n > 1000


def test_the_calls_read_is_evidence():
    # Another seat's hand often would not have made this seat's call or declaration.
    from arena.book_read import calls_weight
    zero = total = 0
    for hands, dealer, r in _auctions():
        for seat in range(4):
            other = hands[(seat + 1) % 4]
            zero += calls_weight(seat, mask_of(other), dealer, r.bids, r.winner, r.high_bid, r.decl_id) == 0.0
            total += 1
    assert 0.2 < zero / total < 0.95


def test_the_book_players_partner_read_is_its_zebs_belief():
    from arena.book_play import parse_book_spec
    book = parse_book_spec("zeb,least,read=consistent+calls/k0,n8")
    assert book.partner is not None and book.zeb.walt.belief is book.partner
    assert (book.partner.likelihood, book.partner.seats, book.partner.calls) == ("consistent", "partner", True)
    assert parse_book_spec("zeb,zeb,read=uniform/k0,n8").partner.calls is False
    assert parse_book_spec("zeb,least/k0,n8").partner is None
    with pytest.raises(ValueError):
        parse_book_spec("zeb,least,read=consistent+bids/k0,n8")
    with pytest.raises(ValueError):
        parse_book_spec("zeb,least,read=consistent/k0,n8,d=posterior")  # only plain Zeb's draw


def test_the_shadow_logs_whether_the_read_moved_the_tile():
    from arena.book_play import BookPlay
    from arena.walt_play import parse_walt_spec
    book = BookPlay("zeb", "least", parse_walt_spec("k0,n16"), read="consistent", shadow=True)
    state = playing(3, 2, bidder=1)
    while not is_terminal(state):
        slot = book.choose([state], [30])[0]
        state = apply_action(state, slot)
    assert book.moved and all(r["tile"] == r["plain_tile"] for r in book.moved if r["policy"] is not None)
    assert all(r["tile"] == r["zeb"] and r["plain_tile"] == r["plain_zeb"]
               for r in book.moved if r["policy"] is None)
