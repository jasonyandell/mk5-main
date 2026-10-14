"""The soft-reading study's bookkeeping (`scenario.reads_soft`): the net
seats play the model the reader assumes, deterministically by state; the
net table routes the team's lower seat to X; and the summary pairs arms by
game pair. The soft reading itself is tested in `test_reading.py`."""

import json
from collections import Counter

from forge.zeb.game import current_player, legal_actions

from scenario.planner import Config, Reading, Rung, Walt, situation_from_state
from scenario.reads_soft import EPS, NetSeat, Table, summarize
from scenario.reads_plays import MARGIN
from scenario.tests.test_field import NET, _pip_states
from scenario.tests.test_reads_plays import _row


def _seat(tmp_path, eps=EPS):
    path = tmp_path / "net.npz"
    NET.save(path)
    return NetSeat(str(path), eps)


def test_a_net_seat_plays_legally_and_the_same_at_the_same_state(tmp_path):
    seat = _seat(tmp_path)
    for state in _pip_states(12, seed=5, min_plays=3):
        t = seat.tile(state, 30)
        hand = state.hands[current_player(state)]
        assert t in {hand[i] for i in legal_actions(state)}
        assert seat.tile(state, 30) == t


def test_a_net_seat_plays_the_soft_uniform_model(tmp_path):
    """Over many deals at one position shape, the share of plays outside the
    net's set is ε·(n − |A|)/n on average, as the soft uniform likelihood
    says; at ε = 0 every play is in the set."""
    sharp, soft = _seat(tmp_path, 0.0), _seat(tmp_path, 0.5)
    states = _pip_states(150, seed=11, min_plays=1)
    out, expect = Counter(), 0.0
    for state in states:
        sit = situation_from_state(state, 30)
        from scenario.record import remaining_hand
        from scenario.rules import rules_for
        rules, seat = rules_for(state.decl_id), current_player(state)
        a = NET.allowed(rules, seat, remaining_hand(state, seat), sit.record, 30, state.bidder % 2, MARGIN)
        n = len(legal_actions(state))
        assert (a >> sharp.tile(state, 30)) & 1
        out[(a >> soft.tile(state, 30)) & 1 == 0] += 1
        expect += 0.5 * (n - a.bit_count()) / n
    assert abs(out[True] - expect) < 3 * expect ** 0.5 + 2


def test_the_soft_likelihood_is_the_net_seats_probability(tmp_path):
    """A net seat's play has, under the soft uniform reading, the probability
    the seat assigns it: ε/n + (1 − ε)·[t ∈ A]/|A|, which sums to 1 over the
    legal tiles."""
    r = Reading("uniform", MARGIN, EPS)
    for a, n in ((0b101, 4), (0b1, 3), (0b111, 3)):
        assert abs(sum(r.weight(t, a, n) for t in range(n)) - 1.0) < 1e-15
    w = Walt(Config((Rung(8, None),), field=NET, draw="posterior", reading=r))
    assert w.config.reading.eps == EPS


class _Fixed:
    def __init__(self, slot):
        self.slot, self.seen = slot, []

    def choose(self, states, bids, marks=None, marks_to_win=7):
        self.seen += [current_player(s) for s in states]
        return [self.slot] * len(states)


def test_the_table_seats_x_at_the_teams_lower_seat(tmp_path):
    seat = _seat(tmp_path)
    x = _Fixed(-1)
    states = _pip_states(40, seed=23)
    slots = Table(x, seat).choose(states, [30] * len(states))
    assert set(x.seen) <= {0, 1} and x.seen
    for s, slot in zip(states, slots):
        assert (slot == -1) == (current_player(s) < 2)


def _run(tmp_path, name, table, arm, seed, games, rows=()):
    d = tmp_path / name
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"table": table, "arm": arm, "seed": seed}))
    with open(d / "games.jsonl", "w") as f:
        for idx, a_team, am, bm, ap, bp in games:
            f.write(json.dumps({"game": idx, "a_team": a_team, "a_marks": am, "b_marks": bm,
                                "a_points": ap, "b_points": bp,
                                "hands": [{"bidder_team": 0, "made": True, "a_team": a_team,
                                           "points": [30, 12]}]}) + "\n")
    with open(d / "decisions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({**r, "distinct": 7}) + "\n")
    return str(d)


def test_the_summary_differences_arms_by_game_pair(tmp_path):
    """The net table's effect is the reader arm's pair margin less the plain
    arm's on the same (seed, game); a pair only one arm holds is left out."""
    plain = _run(tmp_path, "p", "net", "plain", 3,
                 [(0, 0, 7, 2, 250, 100), (0, 1, 7, 4, 240, 200), (1, 0, 3, 7, 90, 250), (1, 1, 7, 7, 250, 250)])
    unif = _run(tmp_path, "u", "net", "uniform", 3,
                [(0, 0, 7, 0, 250, 50), (0, 1, 7, 4, 240, 200), (1, 0, 7, 3, 250, 90), (1, 1, 7, 7, 250, 250),
                 (2, 0, 7, 0, 250, 0), (2, 1, 7, 0, 250, 0)],
                [_row(1, 1, 1, consistent=1), _row(2, 3, 2, consistent=2, fell=1)])
    s = summarize([plain, unif])
    e = s["effects"]["net: uniform - plain (primary)"]
    # pair 0: (7 + 3) − (5 + 3) = 2 → 1 per game; pair 1: (4 + 0) − (−4 + 0) = 8 → 4 per game
    assert e["marks_per_game"]["game_pairs"] == 2 and e["marks_per_game"]["mean"] == 2.5
    assert e["points_per_game"]["mean"] == ((240 - 190) + (160 + 160)) / 2 / 2
    assert s["arms"]["net/uniform"]["fallback_by_trick"][2] == {"n": 2, "rate": 0.5}
    assert "net: consistent - plain" not in s["effects"]
