"""JudSearch (jud v1, JS1) — CPU, fast.

The load-bearing properties, mirrored from `test_jud_play` but pushed through
the trick rollout: the DEFENDER SIGN survives the search (the leaf prices the
declaring team's points; a defender must minimize them across worlds) and
LEGAL-MOVE MASKING (only legal root children are ever rolled out). Two
properties are search-specific: when the mover is LAST in the trick the leaf
is world-independent, so any N gives the same choice; and given the SAME
worlds the choice is a pure function of the state, so the batched lockstep
equals one-state-at-a-time calls. (A strict end-to-end batched==sequential
cannot hold: the MRV sampler consumes the global torch RNG, so batch
composition changes the worlds — the same reason --fast-batching is
"statistically equivalent, not byte-identical".)
"""
from __future__ import annotations

import torch

from arena.bidders import HeuristicBidder
from arena.engine import ArenaConfig
from arena.jud_play import JudPlay
from arena.jud_search import JudSearch
from arena.match import run_match
from arena.test_jud_play import _HANDS, _save_head, _state, PreferDomino
from champion.jud_net import JudNet
from forge.zeb.game import current_player, legal_actions


def _true_worlds(state, n_worlds: int) -> torch.Tensor:
    """[n_worlds, 3, 7] worlds that are the actual hidden hands — trivially
    consistent, and fixed, so tests of the deterministic core need no RNG."""
    mover = current_player(state)
    rows = []
    for i in range(3):
        q = (mover + 1 + i) % 4
        remaining = [d for d in state.hands[q] if d not in state.played]
        rows.append(remaining + [-1] * (7 - len(remaining)))
    return torch.tensor(rows).unsqueeze(0).expand(n_worlds, 3, 7).clone()


# --------------------------------------------------------------------- #
#  1. Defender sign, through a trick rollout                              #
# --------------------------------------------------------------------- #

def test_offense_maximizes_declaring_team_ev_through_rollout():
    # Bidder = seat 0 leads trick 1; all 7 slots legal. The stub loves domino 3
    # (slot 3 of seat 0's hand); no other seat can ever play it (it is in the
    # mover's hand, so it is in no sampled world). Every leaf after slot 3
    # scores ~42, every other leaf ~21 — offense must play it.
    state = _state(_HANDS, bidder=0, decl_id=6)
    play = JudSearch(PreferDomino(3), n_worlds=3)
    assert play.choose([state], [30]) == [3]


def test_defender_minimizes_declaring_team_ev_through_rollout():
    # Bidder = seat 3 led 6-0 (id 21) under sixes; seat 0 holds no six, so all
    # 7 slots are legal. Seat 0 defends: it must NOT hand the stub its beloved
    # domino 3, even though offense (above) would.
    state = _state(_HANDS, bidder=3, decl_id=6, plays=((3, 21),))
    assert 3 in legal_actions(state) and len(legal_actions(state)) == 7
    play = JudSearch(PreferDomino(3), n_worlds=3)
    assert play.choose([state], [30]) != [3]


def test_sign_flip_routes_per_game_within_one_batch():
    # Seat 0 moves in both games of one batched call (same scenario as the
    # JudPlay test): defending it starves the stub's favorite, on the
    # declaring team it feeds it — the rollout must not mix the signs up.
    defend = _state(_HANDS, bidder=3, decl_id=6, plays=((3, 21),))
    attack = _state(
        _HANDS, bidder=2, decl_id=6,
        plays=((2, 20), (3, 22)), trick_leader=2,
    )  # mover = leader 2 + 2 plays = seat 0, the declarer's partner: offense
    play = JudSearch(PreferDomino(3), n_worlds=3)
    choices = play.choose([defend, attack], [30, 30])
    assert choices[1] == 3
    assert choices[0] != 3


# --------------------------------------------------------------------- #
#  2. Legal-move masking: an illegal favorite is never rolled out         #
# --------------------------------------------------------------------- #

def test_illegal_favorite_is_never_rolled_out_or_chosen():
    # Seat 1 leads 3-3 (id 9) under sixes trump; seat 2 holds exactly two
    # threes — 5-3 (18, slot 0) and 3-2 (8, slot 1) — and must follow. The
    # stub's favorite 4-4 (id 14, slot 2) is ILLEGAL here.
    hands = ((0, 1, 2, 3, 4, 5, 6), (7, 20, 9, 10, 11, 12, 13),
             (18, 8, 14, 15, 16, 17, 19), (21, 22, 23, 24, 25, 26, 27))
    state = _state(hands, bidder=1, decl_id=6, plays=((1, 9),))
    assert set(legal_actions(state)) == {0, 1}
    play = JudSearch(PreferDomino(14), n_worlds=3)
    assert play.choose([state], [30])[0] in (0, 1)


def test_choices_are_always_legal_with_a_real_head():
    # An untrained JudNet across a real match: every root choice and every
    # simulated in-world reply is accepted by the engine (apply_action
    # validates legality), and the match runs to completion.
    torch.manual_seed(0)
    result = run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=JudSearch(JudNet(), n_worlds=2), play_b=JudPlay(JudNet()),
        n_games=2, cfg=ArenaConfig(marks_to_win=2, base_seed=17),
    )
    assert result.n_games == 2
    assert all(g.hands for g in result.games)


# --------------------------------------------------------------------- #
#  3. Mover last in the trick: no hidden replies, so N cannot matter      #
# --------------------------------------------------------------------- #

def test_mover_last_is_world_count_invariant():
    # Trick 1: seat 3 led 6-0 (21), seats 0 and 1 followed; mover = seat 2 is
    # LAST — the trick resolves on its own move, no hidden seat replies, and
    # the leaf reads only the mover's real hand + public history. Any N must
    # therefore give the identical choice (and JudSearch still differs from
    # JudPlay by design: it prices the POST-RESOLUTION state, not the
    # post-move state).
    state = _state(
        _HANDS, bidder=3, decl_id=6,
        plays=((3, 21), (0, 0), (1, 7)), trick_leader=3,
    )
    assert current_player(state) == 2 and len(state.current_trick) == 3
    assert len(legal_actions(state)) == 7  # seat 2 holds no six
    torch.manual_seed(2)
    model = JudNet()
    choices = {
        JudSearch(model, n_worlds=n).choose([state], [30])[0]
        for n in (1, 10)
    }
    assert len(choices) == 1


# --------------------------------------------------------------------- #
#  4. Batched == sequential on the deterministic core (worlds given)      #
# --------------------------------------------------------------------- #

def test_batched_equals_sequential_given_the_same_worlds():
    torch.manual_seed(1)
    model = JudNet()
    play = JudSearch(model, n_worlds=3)
    states = [
        _state(_HANDS, bidder=0, decl_id=6),
        _state(_HANDS, bidder=3, decl_id=2, plays=((3, 21),)),
        _state(_HANDS, bidder=1, decl_id=4, trick_leader=1, plays=((1, 9),)),
    ]
    worlds = torch.stack([_true_worlds(s, 3) for s in states])  # [3, 3, 3, 7]
    batched = play._choose_given_worlds(states, worlds)
    singles = [
        play._choose_given_worlds([s], worlds[i:i + 1])[0]
        for i, s in enumerate(states)
    ]
    assert batched == singles


def test_forced_follow_short_circuits_without_search():
    # Seat 3 leads 6-6 (id 27) under blanks trump; seat 0's only six is 6-5
    # (26, slot 0) — its blanks are trump, its other tiles off-suit. One legal
    # slot: the choice is forced, no rollout happens at all.
    hands = ((26, 0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11, 12),
             (13, 14, 15, 16, 17, 18, 19), (20, 21, 22, 23, 24, 25, 27))
    state = _state(hands, bidder=3, decl_id=0, plays=((3, 27),))
    assert legal_actions(state) == (0,)
    play = JudSearch(JudNet(), n_worlds=2)
    assert play.choose([state], [30]) == [0]


# --------------------------------------------------------------------- #
#  5. Registry wiring + a full CPU match through it                       #
# --------------------------------------------------------------------- #

def test_parse_play_registers_judsearch(tmp_path):
    from arena.cli import parse_play

    path = _save_head(tmp_path)
    play = parse_play(
        f"judsearch:n2,model={path}", model=None, n_samples=1, device="cpu", seed=0,
    )
    assert isinstance(play, JudSearch) and play.n_worlds == 2
    default = parse_play(
        f"judsearch:model={path}", model=None, n_samples=1, device="cpu", seed=0,
    )
    assert isinstance(default, JudSearch) and default.n_worlds == 10


def test_judsearch_team_plays_a_full_cpu_match(tmp_path):
    # jud bidder + judsearch play from the registry drive a whole match: the
    # search consumer of the one organ, end to end vs the heuristic baseline.
    from arena.cli import parse_bidder, parse_play

    path = _save_head(tmp_path)
    torch.manual_seed(3)
    result = run_match(
        bid_a=parse_bidder(f"jud:model={path}", device="cpu", gus_adapter=None),
        bid_b=HeuristicBidder(),
        play_a=parse_play(f"judsearch:n2,model={path}", model=None, n_samples=1,
                          device="cpu", seed=0),
        play_b=JudPlay(JudNet()),
        n_games=2, cfg=ArenaConfig(marks_to_win=2, base_seed=23),
    )
    assert result.n_games == 2
    assert all(g.hands for g in result.games)
