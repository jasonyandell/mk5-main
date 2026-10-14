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
    FOLLOW_ORDER, LEAD_ORDER, POLICIES, BookPlay, SilasBidder, ask, compiled, parse_book_spec,
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
    highest tile id is best, and q ranks every legal tile by its id."""

    def __init__(self):
        self.walt = self
        self.bids = []

    def decide_batch(self, states, bids):
        self.bids.extend(bids)
        out = []
        for s in states:
            legal = legal_tiles(s)
            out.append((max(legal), {t: t / 100 for t in legal}))
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
                assert row["tile"] == max(row["allowed"])  # the stub's q ranks by id


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
