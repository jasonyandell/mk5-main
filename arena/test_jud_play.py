"""JudPlay (jud v1) — CPU, fast.

The load-bearing tests are the two correctness properties of value-native
play: the DEFENDER SIGN (the net prices the declaring team's points; a
defender must minimize them) and LEGAL-MOVE MASKING (only legal children are
ever priced, so the choice is legal by construction — even when the head
would prefer an illegal move). A stub head with a readable preference makes
both observable; the rest covers batching across games, a full engine match,
and the CLI registry wiring for both jud consumers.
"""
from __future__ import annotations

import random

import torch

from arena.bidders import HeuristicBidder
from arena.engine import ArenaConfig
from arena.jud_play import JudPlay
from arena.match import run_match
from champion.jud_net import (
    AUCTION_DIM,
    FEATURE_DIM,
    HAND_DIM,
    JudNet,
    N_POINTS,
    PER_DOMINO,
)
from forge.zeb.game import legal_actions
from forge.zeb.types import BidState, GamePhase, ZebGameState

_PLAY_OFF = HAND_DIM + AUCTION_DIM


# --------------------------------------------------------------------- #
#  Stub head with a readable preference                                   #
# --------------------------------------------------------------------- #

class PreferDomino:
    """Stands in for JudNet: E[pts] is ~42 iff domino ``d`` has been played in
    the queried info-state, else ~21 (uniform logits). JudPlay prices
    POST-move states, so an offense mover holding ``d`` should play it and a
    defender should avoid it."""

    def __init__(self, d: int):
        self.d = d

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        block = x[:, _PLAY_OFF + PER_DOMINO * self.d:][:, :4]
        played = block.sum(dim=1)  # 1.0 iff d appears in the play history
        logits = torch.zeros(x.shape[0], N_POINTS)
        logits[:, 42] = 50.0 * played
        return logits


def _state(hands, *, bidder, decl_id, plays=(), trick_leader=None) -> ZebGameState:
    """A mid-hand PLAYING state built the way arena.engine builds them."""
    return ZebGameState(
        hands=tuple(tuple(h) for h in hands),
        dealer=0,
        phase=GamePhase.PLAYING,
        bid_state=BidState(
            bids=tuple(30 if s == bidder else 0 for s in range(4)),
            high_bidder=bidder,
            high_bid=30,
        ),
        decl_id=decl_id,
        bidder=bidder,
        played=frozenset(d for _, d in plays),
        play_history=tuple(plays),
        current_trick=tuple(d for _, d in plays[-(len(plays) % 4):] if len(plays) % 4),
        trick_leader=bidder if trick_leader is None else trick_leader,
        team_points=(0, 0),
    )


_HANDS = ((0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12, 13),
          (14, 15, 16, 17, 18, 19, 20), (21, 22, 23, 24, 25, 26, 27))


# --------------------------------------------------------------------- #
#  1. Defender sign: offense plays the high-EV move, defense avoids it    #
# --------------------------------------------------------------------- #

def test_offense_maximizes_declaring_team_ev():
    # Bidder = seat 0 leads trick 1; all 7 slots legal. The stub loves
    # domino 3 (slot 3 of seat 0's hand) — offense must play it.
    state = _state(_HANDS, bidder=0, decl_id=6)
    play = JudPlay(PreferDomino(3))
    assert play.choose([state], [30]) == [3]


def test_defender_minimizes_declaring_team_ev():
    # Bidder = seat 3 led 6-0 (id 21) under sixes; seat 0 holds no six, so all
    # 7 slots are legal. Seat 0 defends: it must NOT hand the stub its beloved
    # domino 3, even though offense (above) would.
    state = _state(_HANDS, bidder=3, decl_id=6, plays=((3, 21),))
    assert 3 in legal_actions(state) and len(legal_actions(state)) == 7
    play = JudPlay(PreferDomino(3))
    assert play.choose([state], [30]) != [3]


def test_sign_flip_routes_per_game_within_one_batch():
    # Seat 0 moves in both games of one batched call, holding no domino of the
    # led suits (all 7 slots legal both times). Defending (bidder = seat 3) it
    # avoids the stub's favorite; on the declaring team (partner = seat 2,
    # mover after two plays) it plays that same favorite.
    defend = _state(_HANDS, bidder=3, decl_id=6, plays=((3, 21),))
    attack = _state(
        _HANDS, bidder=2, decl_id=6,
        plays=((2, 20), (3, 22)), trick_leader=2,
    )  # mover = leader 2 + 2 plays = seat 0, the declarer's partner: offense
    play = JudPlay(PreferDomino(3))
    choices = play.choose([defend, attack], [30, 30])
    assert choices[1] == 3      # partner offense feeds the declaring team's EV
    assert choices[0] != 3      # defender starves it


# --------------------------------------------------------------------- #
#  2. Legal-move masking: an illegal favorite is never chosen             #
# --------------------------------------------------------------------- #

def test_illegal_favorite_is_never_priced_or_chosen():
    # Seat 1 leads 3-3 (id 9) under sixes trump; seat 2 holds exactly two
    # threes — 5-3 (18, slot 0) and 3-2 (8, slot 1) — and must follow. The
    # stub's favorite 4-4 (id 14, slot 2) is ILLEGAL here; the choice must be
    # one of the followers.
    hands = ((0, 1, 2, 3, 4, 5, 6), (7, 20, 9, 10, 11, 12, 13),
             (18, 8, 14, 15, 16, 17, 19), (21, 22, 23, 24, 25, 26, 27))
    state = _state(hands, bidder=1, decl_id=6, plays=((1, 9),))
    assert set(legal_actions(state)) == {0, 1}
    play = JudPlay(PreferDomino(14))
    choice = play.choose([state], [30])[0]
    assert choice in (0, 1)


def test_choices_are_always_legal_with_a_real_head():
    # An untrained JudNet across a real match: every decision the policy makes
    # is accepted by the engine (apply_action validates legality), and the
    # match runs to completion. This is the masking property end-to-end.
    torch.manual_seed(0)
    result = run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=JudPlay(JudNet()), play_b=JudPlay(JudNet()),
        n_games=2, cfg=ArenaConfig(marks_to_win=2, base_seed=17),
    )
    assert result.n_games == 2
    assert all(g.hands for g in result.games)


# --------------------------------------------------------------------- #
#  3. Batching: one call, many games, per-game routing                    #
# --------------------------------------------------------------------- #

def test_batched_choose_matches_single_state_calls():
    torch.manual_seed(1)
    model = JudNet()
    play = JudPlay(model)
    states = [
        _state(_HANDS, bidder=0, decl_id=6),
        _state(_HANDS, bidder=3, decl_id=2, plays=((3, 21),)),
        _state(_HANDS, bidder=1, decl_id=4, trick_leader=1, plays=((1, 9),)),
    ]
    batched = play.choose(states, [30, 34, 30])
    singles = [play.choose([s], [30])[0] for s in states]
    assert batched == singles


# --------------------------------------------------------------------- #
#  4. Registry wiring: judplay + jud bidder specs                         #
# --------------------------------------------------------------------- #

def _save_head(tmp_path):
    path = tmp_path / "jud_test.pt"
    torch.save(
        {"model_state": JudNet().state_dict(), "feature_dim": FEATURE_DIM}, path,
    )
    return path


def test_parse_play_registers_judplay(tmp_path):
    from arena.cli import parse_play

    path = _save_head(tmp_path)
    play = parse_play(
        f"judplay:model={path}", model=None, n_samples=1, device="cpu", seed=0,
    )
    assert isinstance(play, JudPlay)


def test_parse_bidder_registers_jud(tmp_path):
    from arena.cli import parse_bidder
    from champion.jud_net import JudNet as JN
    from champion.utility import MarksToSeven
    from champion.value_bidder import ValueBidder

    path = _save_head(tmp_path)
    bidder = parse_bidder(f"jud:wp,model={path}", device="cpu", gus_adapter=None)
    assert isinstance(bidder, ValueBidder)
    assert isinstance(bidder.model, JN)
    assert isinstance(bidder.utility, MarksToSeven)


def test_jud_team_plays_a_full_cpu_match(tmp_path):
    # jud bidder + jud play from the registry drive a whole match: the two
    # consumers of the one organ, end to end against the heuristic baseline.
    from arena.cli import parse_bidder, parse_play

    path = _save_head(tmp_path)
    result = run_match(
        bid_a=parse_bidder(f"jud:model={path}", device="cpu", gus_adapter=None),
        bid_b=HeuristicBidder(),
        play_a=parse_play(f"judplay:model={path}", model=None, n_samples=1,
                          device="cpu", seed=0),
        play_b=JudPlay(JudNet()),
        n_games=2, cfg=ArenaConfig(marks_to_win=2, base_seed=23),
    )
    assert result.n_games == 2
    assert all(g.hands for g in result.games)
