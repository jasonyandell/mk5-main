# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The seat that loses the count (#172): `arena/count_lapse.py`."""
from __future__ import annotations

import random

from arena.count_lapse import CountLapse, lost_count, replayed
from forge.zeb.game import apply_action, is_terminal, legal_actions, new_game

ALWAYS = 1 << 53


def decisions(n_games: int = 30, seed: int = 7):
    """Every decision of `n_games` random games, in order."""
    rng = random.Random(seed)
    for g in range(n_games):
        s = new_game(seed * 1000 + g)
        while not is_terminal(s):
            yield s
            s = apply_action(s, rng.choice(legal_actions(s)))


def live(state, seat):
    return {d for d in state.hands[seat] if d not in state.played}


class Recorder:
    """The least legal slot, and every state it was shown."""

    def __init__(self):
        self.seen = []

    def choose(self, states, bid_values, marks=None, marks_to_win=7):
        self.seen.extend(states)
        return [legal_actions(s)[0] for s in states]


def test_the_true_history_replays_to_itself():
    for s in decisions():
        assert replayed(s) == s


def test_p0_is_the_inner_policy_shown_the_true_state():
    lapse, inner = CountLapse(Recorder(), 0.0), Recorder()
    for s in decisions(10):
        assert lapse.choose([s], [30]) == inner.choose([s], [30])
        assert lapse.inner.seen[-1] is s
    assert lapse.forgotten == 0


def test_the_view_is_a_game_that_could_have_been_played():
    swapped = 0
    for s in decisions():
        view, forgotten, stood = lost_count(s, ALWAYS)
        seat = (s.trick_leader + len(s.current_trick)) % 4
        assert replayed(view) == view
        assert view.hands[seat] == s.hands[seat] and legal_actions(view) == legal_actions(s)
        assert view.current_trick == s.current_trick and view.trick_leader == s.trick_leader
        assert sorted(d for h in view.hands for d in h) == list(range(28))
        assert [len(live(view, k)) for k in range(4)] == [len(live(s, k)) for k in range(4)]
        assert set(stood) <= set(forgotten)
        for t in stood:            # a forgotten tile with a stand-in is out, in the view
            assert t not in view.played and t in s.played
        swapped += len(stood)
    assert swapped > 100


def test_only_other_seats_completed_plays_are_forgotten():
    for s in decisions(10):
        seat = (s.trick_leader + len(s.current_trick)) % 4
        _, forgotten, _ = lost_count(s, ALWAYS)
        done = s.play_history[:len(s.play_history) - len(s.current_trick)]
        assert set(forgotten) == {d for k, d in done if k != seat}


def test_a_lost_count_stays_lost_for_the_hand():
    coin = (1 << 53) // 4
    for g in range(20):
        rng, s, held = random.Random(g), new_game(5000 + g), {}
        while not is_terminal(s):
            seat = (s.trick_leader + len(s.current_trick)) % 4
            _, forgotten, _ = lost_count(s, coin)
            assert held.get(seat, set()) <= set(forgotten)
            held[seat] = set(forgotten)
            s = apply_action(s, rng.choice(legal_actions(s)))


def test_the_coin_forgets_at_its_rate():
    coin = (1 << 53) // 4
    seen = lost = 0
    for s in decisions(60):
        if len(s.play_history) == 27:      # every other seat's play of six tricks, once a deal
            seat = (s.trick_leader + len(s.current_trick)) % 4
            seen += sum(1 for k, _ in s.play_history[:24] if k != seat)
            lost += len(lost_count(s, coin)[1])
    assert abs(lost / seen - 0.25) < 0.05
