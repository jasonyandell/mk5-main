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


def test_the_replication_reads_the_primary_the_aa_arm_and_the_game_share(tmp_path):
    """The replication's primary is the consistent reader's mark margin per
    game, a success when its lower bound clears 0; the game share is the
    reader's games won of a pair, over 2; a run at another ε is a sweep arm,
    not the primary."""
    from scenario.reads_soft import replicate

    def run(name, arm, seed, games, eps=EPS, to=None):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"table": "zeb", "arm": arm, "seed": seed, "eps": eps,
                                                 "points_to_win": to}))
        with open(d / "games.jsonl", "w") as f:
            for idx, a_team, am, bm, won in games:
                f.write(json.dumps({"game": idx, "a_team": a_team, "a_marks": am, "b_marks": bm,
                                    "a_points": 10 * am, "b_points": 10 * bm, "a_won": won, "hands": []}) + "\n")
        return str(d)

    win, lose = (0, 0, 7, 3, True), (0, 1, 2, 7, False)
    con = run("c", "consistent", 101, [win, (0, 1, 7, 5, True), (1, 0, 7, 1, True), (1, 1, 7, 6, True)])
    aa = run("a", "plain", 101, [win, lose, (1, 0, 3, 7, False), (1, 1, 7, 3, True)])
    swp = run("s", "consistent", 101, [win, lose], eps=0.1)
    g250 = run("g", "consistent", 201, [win, lose], to=250)
    r = replicate([con, aa, swp, g250])
    p = r["registered"]["primary: consistent vs plain, marks/game"]
    # pair 0: (4 + 2) / 2 = 3; pair 1: (6 + 1) / 2 = 3.5
    assert p["game_pairs"] == 2 and p["mean"] == 3.25 and p["success"]
    assert r["registered"]["consistent vs plain, settled points/game"]["mean"] == 32.5
    # pair 0: (4 − 5) / 2; pair 1: (−4 + 4) / 2
    assert r["registered"]["A/A: plain at seed 1 vs plain, marks/game"]["mean"] == -0.25
    assert r["registered"]["consistent vs plain, share of games to 250 won"]["mean"] == 0.5
    assert r["arms"]["consistent eps=0.28 to=7 marks"]["game_share"]["mean"] == 1.0
    assert r["exploratory"]["sweep: consistent eps=0.1 - eps=0.28, marks/game"]["game_pairs"] == 1
    assert "uniform vs plain, marks/game" not in r["registered"]


def test_the_second_replication_reads_the_uniform_primary_and_its_game_share(tmp_path):
    """With a uniform primary the registered keys follow the reader: its
    marks (a success when the lower bound clears 0), its settled points, its
    share of games to 250, and uniform − A/A as exploratory; the seeds are
    disjoint from every earlier run of the study."""
    from scenario.reads_soft import REP2_250_SEEDS, REP2_GAMES, REP2_PILOT, REP2_SEEDS, REP_250_SEEDS, \
        REP_PILOT, REP_SEEDS, replicate

    def run(name, arm, seed, games, to=None):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"table": "zeb", "arm": arm, "seed": seed, "eps": EPS,
                                                 "points_to_win": to}))
        with open(d / "games.jsonl", "w") as f:
            for idx, a_team, am, bm, won in games:
                f.write(json.dumps({"game": idx, "a_team": a_team, "a_marks": am, "b_marks": bm,
                                    "a_points": 10 * am, "b_points": 10 * bm, "a_won": won, "hands": []}) + "\n")
        return str(d)

    win, lose = (0, 0, 7, 3, True), (0, 1, 2, 7, False)
    uni = run("u", "uniform", 301, [win, (0, 1, 7, 5, True), (1, 0, 7, 1, True), (1, 1, 1, 7, False)])
    aa = run("a", "plain", 301, [win, lose, (1, 0, 3, 7, False), (1, 1, 7, 3, True)])
    g250 = run("g", "uniform", 401, [win, win], to=250)
    r = replicate([uni, aa, g250], "uniform")
    p = r["registered"]["primary: uniform vs plain, marks/game"]
    # pair 0: (4 + 2) / 2 = 3; pair 1: (6 − 6) / 2 = 0
    assert p["game_pairs"] == 2 and p["mean"] == 1.5
    assert r["registered"]["uniform vs plain, settled points/game"]["mean"] == 15.0
    assert r["registered"]["uniform vs plain, share of games to 250 won"]["mean"] == 1.0
    # pair 0: (6 − (−1)) / 2; pair 1: (0 − 0) / 2
    assert r["exploratory"]["uniform - A/A, marks/game"]["mean"] == 1.75
    assert not any(k.startswith("consistent") for k in r["registered"])
    earlier = set(range(1, 8)) | set(range(900, 905)) | set(REP_SEEDS) | set(REP_250_SEEDS) | set(REP_PILOT)
    fresh = set(REP2_SEEDS) | set(REP2_250_SEEDS) | set(REP2_PILOT)
    assert not earlier & fresh and not set(REP2_SEEDS) & set(REP2_250_SEEDS)
    assert len(REP2_SEEDS) * REP2_GAMES >= 300
