# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The book player (arena/book_play.py): the Silas stack, its variants, and
the Silas bidders as arena bid policies."""
import random

import pytest

pytest.importorskip("scenario_core")

from forge.oracle.rng import deal_from_seed
from forge.zeb.game import apply_action, is_terminal, legal_actions
from forge.zeb.types import BidState, GamePhase, ZebGameState

from arena.auction import run_auction
from arena.bidders import HeuristicBidder
from arena.book_play import (
    FOLLOW_ORDER, LEAD_ORDER, POLICIES, BookPlay, SilasBidder, ask, compiled, decide, parse_book_spec,
)
from arena.walt_play import WaltPlay
from scenario.planner import situation_from_state
from scenario.rules import mask_of
from scenario.silas import frame


def playing(seed: int, decl: int, bidder: int = 0, bid: int = 30) -> ZebGameState:
    hands = tuple(tuple(h) for h in deal_from_seed(seed))
    return ZebGameState(
        hands=hands, dealer=(bidder + 3) % 4, phase=GamePhase.PLAYING,
        bid_state=BidState(bids=(bid, 0, 0, 0), high_bidder=bidder, high_bid=bid),
        decl_id=decl, bidder=bidder, played=frozenset(), play_history=(),
        current_trick=(), trick_leader=bidder, team_points=(0, 0),
    )


def seat_of(state: ZebGameState) -> int:
    return (state.trick_leader + len(state.current_trick)) % 4


def legal_tiles(state: ZebGameState) -> list[int]:
    hand = state.hands[seat_of(state)]
    return sorted(hand[i] for i in legal_actions(state))


def test_the_stack_names_the_eighteen_book_rules_once_per_phase():
    assert len(POLICIES) == 18
    assert len(set(LEAD_ORDER)) == len(LEAD_ORDER) and len(set(FOLLOW_ORDER)) == len(FOLLOW_ORDER)
    for name in POLICIES:
        compiled(name, 0)  # every file compiles under Zeb's rules


def test_the_first_firing_policy_in_the_phase_order_decides():
    rng = random.Random(1)
    decided = 0
    for seed in range(12):
        state = playing(seed, seed % 7, bidder=seed % 4)
        while not is_terminal(state):
            v = ask(state, 30)
            order = FOLLOW_ORDER if state.current_trick else LEAD_ORDER
            legal = set(legal_tiles(state))
            assert set(v.fired) <= set(POLICIES)
            if v.policy is None:
                assert not set(v.fired) & set(order)
            else:
                decided += 1
                assert v.policy == next(p for p in order if p in v.fired)
                assert v.allowed and set(v.allowed) <= legal
            state = apply_action(state, rng.choice(legal_actions(state)))
    assert decided > 100


def test_a_no_count_rule_lets_the_later_throws_choose():
    # #146: the pounce's and the donation's no-count rules forbid count and
    # choose nothing; the later firing policies of the phase choose within them.
    fired = {
        "pounce": ("no-count-on-a-lost-trick", (1, 3, 9, 12)),
        "dead-double": ("throw-a-dead-double", (0,)),          # count is not in it: no overlap
        "shed-for-a-void": ("throw-a-lone-tile", (9, 12)),
        "follow-low": ("the-lowest-trump", (12, 20)),
    }
    assert decide(fired, FOLLOW_ORDER) == ("pounce", "no-count-on-a-lost-trick", (12,))
    fired["shed-for-a-void"] = ("throw-what-you-need-not-keep", (5,))
    assert decide(fired, FOLLOW_ORDER) == ("pounce", "no-count-on-a-lost-trick", (12,))
    # Alone, it allows every no-count tile.
    alone = {"donate-when-safe": ("no-count-on-a-trick-that-can-be-lost", (2, 4))}
    assert decide(alone, FOLLOW_ORDER) == ("donate-when-safe", "no-count-on-a-trick-that-can-be-lost", (2, 4))
    # Any other rule decides with its own set.
    other = {"pounce": ("play-a-ten-now", (10,)), "shed-for-a-void": ("throw-a-lone-tile", (9,))}
    assert decide(other, FOLLOW_ORDER) == ("pounce", "play-a-ten-now", (10,))
    assert decide({}, FOLLOW_ORDER) == (None, None, ())


def test_the_bidders_first_lead_is_the_books():
    # Seat 0 declares its longest pip; the first lead is first-lead's.
    for seed in range(20):
        hands = deal_from_seed(seed)
        decl = max(range(7), key=lambda p: sum(1 for t in hands[0] if p in _pips(t)))
        v = ask(playing(seed, decl), 30)
        assert v.policy == "first-lead"


def _pips(t: int) -> tuple[int, int]:
    from forge.oracle.tables import DOMINOES
    return DOMINOES[t]


class _StubZeb:
    """Zeb's interface (`.walt.decide_batch`) with a fixed preference: the
    highest tile id is best for the mover. q is the bidding side's make
    probability, as Zeb's is: rising with the tile id for the bidding side,
    falling for the setters."""

    def __init__(self):
        self.walt = self
        self.bids = []

    def decide_batch(self, states, bids):
        self.bids.extend(bids)
        out = []
        for s in states:
            legal = legal_tiles(s)
            bidding_side = seat_of(s) % 2 == s.bidder % 2
            out.append((max(legal), {t: (t if bidding_side else 100 - t) / 100 for t in legal}))
        return out


def _play_out(policy: BookPlay, seed: int, bid: int = 30) -> ZebGameState:
    state = playing(seed, seed % 7, bidder=seed % 4, bid=bid)
    while not is_terminal(state):
        slot = policy.choose([state], [bid])[0]
        assert slot in legal_actions(state)
        state = apply_action(state, slot)
    return state


def test_the_pure_stack_is_silas_own_decision():
    book = BookPlay("lowest", "least", log=True)
    for seed in range(6):
        _play_out(book, seed)
    assert len(book.log) == 6 * 28
    for row in book.log:
        if row["policy"] is None:
            assert row["tile"] == row["legal"][0]
            continue
        # Silas decides the least tile its deciding rule allows.
        assert row["tile"] == min(row["allowed"])
        state = _state_of(row)
        sit = situation_from_state(state, row["bid"])
        own = mask_of(t for t in state.hands[row["seat"]] if t not in state.played)
        assert compiled(row["policy"], row["decl"]).choose(frame(sit, own)) == row["tile"]


def _state_of(row: dict) -> ZebGameState:
    state = playing(0, row["decl"], row["bidder"], row["bid"])
    state = ZebGameState(**{**state.__dict__, "hands": row["hands"], "dealer": (row["bidder"] + 3) % 4})
    for seat, tile in row["plays"]:
        state = apply_action(state, list(state.hands[seat]).index(tile))
    return state


def test_zeb_plays_where_no_rule_fires_and_inside_the_set_with_zeb_ties():
    for ties in ("least", "zeb"):
        book = BookPlay("zeb", ties, _StubZeb(), log=True)
        for seed in range(4):
            _play_out(book, seed)
        for row in book.log:
            if row["policy"] is None:
                assert row["tile"] == row["zeb"] == max(row["legal"])
            elif ties == "least":
                assert row["tile"] == min(row["allowed"])
            else:
                # Zeb's best inside the set, for the bidding side and the setters alike.
                assert row["tile"] == max(row["allowed"])


def test_union_lets_any_firing_rule_of_the_phase_decide():
    book = BookPlay("zeb", "union", _StubZeb(), log=True)
    for seed in range(4):
        _play_out(book, seed)
    decided = 0
    for row in book.log:
        if row["policy"] is None:
            assert row["tile"] == row["zeb"]
            continue
        decided += 1
        order = LEAD_ORDER if len(row["plays"]) % 4 == 0 else FOLLOW_ORDER
        phase = [p for p in order if p in row["fired"]]
        union = sorted({t for p in phase for t in row["fired"][p][1]})
        assert row["allowed"] == union and row["tile"] == max(union)  # the stub's best is the highest
        # Attributed to the first policy of the phase whose set holds the tile.
        assert row["policy"] == next(p for p in phase if row["tile"] in row["fired"][p][1])
    assert decided > 50


class _FlatZeb(_StubZeb):
    """Zeb indifferent everywhere: every legal tile at one q, Zeb's tile the lowest id."""

    def decide_batch(self, states, bids):
        return [(min(legal_tiles(s)), {t: 0.5 for t in legal_tiles(s)}) for s in states]


def test_within_eps_the_book_breaks_zebs_near_ties_and_nothing_else():
    # A strict preference (the stub): no book tile ties Zeb's, so w0 is Zeb at every decision;
    # a band wider than every gap is variant (c).
    exact, wide, c = (BookPlay("zeb", "zeb", _StubZeb(), log=True, within=w) for w in (0.0, 1.0, None))
    for seed in range(4):
        for book in (exact, wide, c):
            _play_out(book, seed)
    assert all(row["tile"] == row["zeb"] for row in exact.log)
    assert [row["tile"] for row in wide.log] == [row["tile"] for row in c.log]
    # Indifference everywhere: at w0 every decision a rule makes is the book's least tile.
    flat = BookPlay("zeb", "zeb", _FlatZeb(), log=True, within=0.0)
    for seed in range(4):
        _play_out(flat, seed)
    for row in flat.log:
        assert row["tile"] == (min(row["allowed"]) if row["policy"] else row["zeb"])
    assert any(row["tile"] != row["zeb"] for row in flat.log)


def test_zeb_is_asked_the_contracts_points_for_a_marks_bid():
    zeb = _StubZeb()
    _play_out(BookPlay("zeb", "least", zeb), 3, bid=84)
    assert set(zeb.bids) == {42}


def test_walt_is_asked_the_contracts_points_for_a_marks_bid():
    play = WaltPlay(level=0, n_outer=8)
    seen = []
    decide_batch = play.walt.decide_batch
    play.walt.decide_batch = lambda states, bids: seen.extend(bids) or decide_batch(states, bids)
    play.choose([playing(5, 2, bid=84)], [84])
    assert seen == [42]


def test_the_spec_names_fallback_ties_and_zeb():
    book = parse_book_spec("lowest,least")
    assert (book.fallback, book.ties, book.zeb) == ("lowest", "least", None)
    book = parse_book_spec("zeb,zeb/k0,n8")
    assert (book.fallback, book.ties) == ("zeb", "zeb") and book.zeb is not None
    with pytest.raises(ValueError):
        parse_book_spec("zeb,least")  # a Zeb fallback with no Zeb
    with pytest.raises(ValueError):
        parse_book_spec("highest,least/k0,n8")
    book = parse_book_spec("zeb,zeb,w0.02/k0,n8")
    assert (book.fallback, book.ties, book.within) == ("zeb", "zeb", 0.02)
    assert parse_book_spec("zeb,union/k0,n8").ties == "union"
    with pytest.raises(ValueError):
        parse_book_spec("zeb,least,w0/k0,n8")  # the book breaks Zeb's ties: zeb,zeb only
    with pytest.raises(ValueError):
        parse_book_spec("zeb,zeb,x1/k0,n8")


def test_silas_bid_only_enough_is_the_arenas_heuristic_bidder():
    silas, arena = SilasBidder("bid-only-enough"), HeuristicBidder()
    for seed in range(150):
        hands = tuple(tuple(h) for h in deal_from_seed(10_000 + seed))
        for dealer in range(4):
            for forced in (False, True):
                a = run_auction(hands, dealer, (silas,) * 4, random.Random(0), force_shaker=forced)
                b = run_auction(hands, dealer, (arena,) * 4, random.Random(0), force_shaker=forced)
                assert a == b, (seed, dealer, forced)


def test_silas_risk_budget_bids_legally_and_declares_a_pip():
    bidder = SilasBidder("risk-budget")
    seen = set()
    for seed in range(300):
        hands = tuple(tuple(h) for h in deal_from_seed(20_000 + seed))
        result = run_auction(hands, seed % 4, (bidder,) * 4, random.Random(0), force_shaker=True)
        assert result is not None and 0 <= result.decl_id <= 6
        seen.add(result.high_bid)
    assert 30 in seen and max(seen) > 30



def test_risk_budget_reads_no_score_in_either_game():
    """The book's bidding is the same call in the tournament's game and in its own."""
    bidder = SilasBidder("risk-budget")
    for seed in range(60):
        hands = tuple(tuple(h) for h in deal_from_seed(30_000 + seed))
        marks = run_auction(hands, seed % 4, (bidder,) * 4, random.Random(0), force_shaker=True,
                            marks=(6, 5))
        points = run_auction(hands, seed % 4, (bidder,) * 4, random.Random(0), force_shaker=True,
                             points=(240, 230), points_to_win=250)
        assert marks == points
