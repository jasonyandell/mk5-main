# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Doubles trump and no trump at the table (#155).

The substrate first: the arena engine plays whole hands under either
contract, every follow legal and every trick won as the rules say
(wiki/topics/rules-of-42.md, written out here independently of
`forge.oracle.tables`); Zeb's plan decides under them with the random field,
the compiled core equal to the Python referee. Deployed Zeb's field net plays
pip trumps only, so it refuses them; `MenuWalt` seats it at a menu table, its
twin on the random field for decls 7-9 and the net itself, byte for byte,
under a pip trump. A table where a bidder may declare what a seat cannot play
is refused before play.
"""
import math

import pytest

from forge.oracle.declarations import DOUBLES_TRUMP, NOTRUMP
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import DOMINO_COUNT_POINTS, DOMINOES
from forge.zeb.game import apply_action, is_terminal, legal_actions
from forge.zeb.types import BidState, GamePhase, ZebGameState

from arena.auction import PASS
from arena.bidders import HeuristicBidder, MenuBidder
from arena.book_play import BookPlay
from arena.engine import ArenaConfig, run_half
from arena.play import RandomPlay, check_menu, plays
from arena.walt_play import MenuWalt, WaltPlay, parse_walt_spec
from scenario.compiled import CompiledWalt
from scenario.planner import Config, Rung, Walt

CH9 = (DOUBLES_TRUMP, NOTRUMP)


# -- The rules, from the wiki, not from forge.oracle.tables.

def _double(t: int) -> bool:
    return DOMINOES[t][0] == DOMINOES[t][1]


def _trump(t: int, decl: int) -> bool:
    if decl == NOTRUMP:
        return False
    if decl == DOUBLES_TRUMP:
        return _double(t)
    return decl in DOMINOES[t]


def _led(t: int, decl: int):
    """The suit a lead calls: trump if it is one, else its high pip."""
    return "trump" if _trump(t, decl) else max(DOMINOES[t])


def _follows(t: int, led, decl: int) -> bool:
    if led == "trump":
        return _trump(t, decl)
    return led in DOMINOES[t] and not _trump(t, decl)


def _rank(t: int, decl: int) -> int:
    """Within its suit: under doubles trump the doubles by pip; elsewhere a
    double tops its suit and the rest go by pip total."""
    if decl == DOUBLES_TRUMP and _double(t):
        return DOMINOES[t][0]
    return 100 if _double(t) else sum(DOMINOES[t])


def _winner(trick: list[int], decl: int) -> int:
    trumps = [i for i, t in enumerate(trick) if _trump(t, decl)]
    led = _led(trick[0], decl)
    pool = trumps or [i for i, t in enumerate(trick) if _follows(t, led, decl)]
    return max(pool, key=lambda i: _rank(trick[i], decl))


def _check_hand(hands, plays_, decl: int, leader: int) -> tuple[int, int]:
    """Replay 28 (seat, tile) plays against the wiki's rules: turn order,
    every follow legal, each trick to its winner. Returns team points."""
    held = [set(h) for h in hands]
    points = [0, 0]
    assert len(plays_) == 28
    for k in range(7):
        trick = plays_[4 * k: 4 * k + 4]
        seats = [s for s, _ in trick]
        tiles = [t for _, t in trick]
        assert seats == [(leader + i) % 4 for i in range(4)]
        led = _led(tiles[0], decl)
        for seat, tile in trick:
            assert tile in held[seat]
            if any(_follows(t, led, decl) for t in held[seat]):
                assert _follows(tile, led, decl), (decl, k, seat, tile)
            held[seat].discard(tile)
        leader = seats[_winner(tiles, decl)]
        points[leader % 2] += 1 + sum(DOMINO_COUNT_POINTS[t] for t in tiles)
    return points[0], points[1]


def test_the_wiki_rules_on_known_tricks():
    """The reference itself, on tricks whose winners the rules name."""
    tile = {f"{h}-{l}": i for i, (h, l) in enumerate(DOMINOES)}
    t = lambda *names: [tile[n] for n in names]  # noqa: E731
    # Doubles trump: a double led calls doubles; the highest double wins.
    assert _winner(t("1-1", "6-6", "6-5", "0-0"), DOUBLES_TRUMP) == 1
    # Doubles trump: 6-5 is the top five; a trumping double beats it.
    assert _winner(t("5-3", "6-5", "5-4", "2-1"), DOUBLES_TRUMP) == 1
    assert _winner(t("5-3", "6-5", "2-2", "5-4"), DOUBLES_TRUMP) == 2
    # No trump: the double tops its suit; an off-suit tile never wins.
    assert _winner(t("6-1", "6-6", "5-5", "6-5"), NOTRUMP) == 1
    assert _winner(t("5-2", "6-6", "4-1", "3-0"), NOTRUMP) == 0


# -- The engine.

class _Declares:
    """Opens 30 and declares `decl`: every hand a Ch. 9 contract."""

    def __init__(self, decl: int):
        self.decl = decl
        self.declarations = (decl,)

    def bid(self, ctx, rng):
        return 30 if ctx.high_bid == 0 else PASS

    def declare(self, hand, bid, rng):
        return self.decl


@pytest.mark.parametrize("decl", CH9)
def test_the_engine_plays_whole_hands_by_the_rules(decl):
    records = run_half(
        n_games=6, a_team=0, cfg=ArenaConfig(marks_to_win=3, base_seed=155),
        bid_a=_Declares(decl), bid_b=_Declares(decl),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
    )
    n = 0
    for g in records:
        for h in g.hands:
            assert h.decl_id == decl
            assert _check_hand(h.hands, h.plays, decl, h.bidder) == h.team_points
            n += 1
    assert n >= 12


# -- Zeb's plan.

def _state(seed: int, decl: int, bidder: int = 0) -> ZebGameState:
    return ZebGameState(
        hands=tuple(tuple(h) for h in deal_from_seed(seed)), dealer=(bidder + 3) % 4,
        phase=GamePhase.PLAYING,
        bid_state=BidState(bids=tuple(30 if s == bidder else 0 for s in range(4)), high_bidder=bidder, high_bid=30),
        decl_id=decl, bidder=bidder, played=frozenset(), play_history=(), current_trick=(),
        trick_leader=bidder, team_points=(0, 0),
    )


def _play_out(policy, state: ZebGameState) -> ZebGameState:
    while not is_terminal(state):
        (slot,) = policy.choose([state], [30])
        assert slot in legal_actions(state)
        state = apply_action(state, slot)
    return state


def _same(a, b):
    assert a[0] == b[0]
    assert a[1].keys() == b[1].keys()
    for t in a[1]:
        assert a[1][t] == b[1][t] or (math.isnan(a[1][t]) and math.isnan(b[1][t]))


@pytest.mark.parametrize("decl", CH9)
@pytest.mark.parametrize("rungs", [(Rung(16, None),), (Rung(2, 3), Rung(8, None))])
def test_the_core_is_the_referee_through_a_whole_hand(decl, rungs):
    """Every decision of a whole hand: the compiled core's tile and action
    values are the Python referee's, and the hand obeys the rules."""
    cfg = Config(rungs, "make", seed=5)
    ref, fast = Walt(cfg), CompiledWalt(cfg)
    state = _state(31 + decl, decl)
    while not is_terminal(state):
        if len(legal_actions(state)) > 1:
            _same(ref.decide(state, 30), fast.decide(state, 30))
        tile, _ = fast.decide(state, 30)
        seat = (state.trick_leader + len(state.current_trick)) % 4
        state = apply_action(state, list(state.hands[seat]).index(tile))
    assert _check_hand(state.hands, state.play_history, decl, 0) == state.team_points


def _deployed(worlds: int = 16):
    from scenario.small_n import player
    return player(worlds)


@pytest.mark.parametrize("decl", CH9)
def test_deployed_zeb_refuses_doubles_and_no_trump(decl):
    """The finding of #155: the field net plays pip trumps only."""
    with pytest.raises(ValueError, match="pip-trump declarations only"):
        _deployed().choose([_state(7, decl)], [30])


@pytest.mark.parametrize("decl", CH9)
def test_the_menu_seat_plays_whole_ch9_hands(decl):
    seat = MenuWalt(_deployed())
    end = _play_out(seat, _state(7, decl))
    assert _check_hand(end.hands, end.play_history, decl, 0) == end.team_points
    assert "DTNT=random" in repr(seat)


def test_the_menu_seat_is_the_net_under_a_pip_trump():
    """Byte for byte: the same tiles and action values at every decision."""
    net, seat = _deployed(), MenuWalt(_deployed())
    state = _state(7, 4)
    while not is_terminal(state):
        a, b = net.choose([state], [30]), seat.choose([state], [30])
        assert a == b and net.last_values == seat.last_values
        state = apply_action(state, a[0])


def test_a_mixed_batch_goes_to_each_player_in_order():
    seat = MenuWalt(_deployed())
    states = [_state(7, 4), _state(7, DOUBLES_TRUMP), _state(8, 2), _state(8, NOTRUMP)]
    slots = seat.choose(states, [30] * 4)
    for s, slot in zip(states, slots):
        assert slot in legal_actions(s)
    assert slots[0] == _deployed().choose([states[0]], [30])[0]


def test_the_menu_spec_tag(tmp_path):
    from scenario.field import FieldNet
    from scenario.small_n import NET
    net = tmp_path / "zeb.npz"
    FieldNet.from_bytes(NET.read_bytes()).save(net)
    seat = parse_walt_spec(f"k0,n16,F={net},f=16,d=posterior,t=margin,DTNT=random")
    assert isinstance(seat, MenuWalt) and seat.random.walt.config.field is None
    assert seat.random.walt.config.rungs[0].n_worlds == 16 and seat.random.walt.config.tie == "margin"
    with pytest.raises(ValueError, match="DTNT=random"):
        parse_walt_spec("k0,n16,DTNT=random")  # no net to wrap


# -- The table guard.

def test_a_menu_table_refuses_a_bare_net_seat():
    with pytest.raises(ValueError, match="#155"):
        run_half(n_games=1, a_team=0, cfg=ArenaConfig(marks_to_win=1, base_seed=1),
                 bid_a=MenuBidder(), bid_b=HeuristicBidder(),
                 play_a=_deployed(), play_b=RandomPlay(seed=0))
    # Wrapped, the net is refused too: a book player whose Zeb is the bare net.
    with pytest.raises(ValueError, match="#155"):
        check_menu([MenuBidder()], [BookPlay("zeb", "zeb", _deployed())])


def test_a_menu_table_accepts_the_menu_seat():
    check_menu([MenuBidder(), HeuristicBidder()],
               [MenuWalt(_deployed()), BookPlay("zeb", "zeb", MenuWalt(_deployed())), RandomPlay()])
    # A pip-only table seats the bare net as it always has.
    check_menu([HeuristicBidder()], [_deployed()])
    assert plays(WaltPlay(level=0, n_outer=4), NOTRUMP)
    assert not plays(_deployed(), DOUBLES_TRUMP) and plays(_deployed(), 6)


def test_a_book_player_on_the_menu_seat_plays_a_doubles_hand():
    end = _play_out(BookPlay("zeb", "zeb", MenuWalt(_deployed())), _state(9, DOUBLES_TRUMP))
    assert _check_hand(end.hands, end.play_history, DOUBLES_TRUMP, 0) == end.team_points


@pytest.mark.parametrize("base_seed,decl", [(36, NOTRUMP), (52, DOUBLES_TRUMP)])
def test_a_menu_match_plays_a_ch9_contract(base_seed, decl):
    """MenuBidder against itself, both teams on the menu seat: a one-mark
    game whose hand the auction declares at doubles trump or no trump."""
    (g,) = run_half(
        n_games=1, a_team=0, cfg=ArenaConfig(marks_to_win=1, base_seed=base_seed),
        bid_a=MenuBidder(), bid_b=MenuBidder(),
        play_a=MenuWalt(_deployed()), play_b=MenuWalt(_deployed()),
    )
    (h,) = g.hands
    assert h.decl_id == decl
    assert _check_hand(h.hands, h.plays, h.decl_id, h.bidder) == h.team_points
