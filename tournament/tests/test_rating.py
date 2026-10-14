"""The rating fit: recovers known strengths, ignores record order, and its
intervals cover at the stated rate and respond to the duplicate pairing."""
from __future__ import annotations

import math
import random

import numpy as np

from tournament.rating import ELO_PER_NAT, Game, fit
from tournament.schedule import gains, next_pair


def simulate(theta: dict[str, float], pairs_per_matchup: int, rng: random.Random, luck: float = 0.0):
    """Duplicate pairs: each pair shares a 'deal' effect d that favors seat team 0;
    A sits on team 0 in game 1 and team 1 in game 2, so luck cancels across the pair."""
    names = sorted(theta)
    games = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for k in range(pairs_per_matchup):
                d = rng.gauss(0.0, luck)
                for team in (0, 1):
                    x = theta[a] - theta[b] + (d if team == 0 else -d)
                    won = rng.random() < 1.0 / (1.0 + math.exp(-x))
                    games.append(Game(a, b, won, f"{a}-{b}-{k}"))
    return games


def test_recovers_known_strengths():
    truth = {"w": 0.0, "x": 0.5, "y": -0.4, "z": 1.0}
    f = fit(simulate(truth, 400, random.Random(1)), truth, "w")
    for r in f.ratings:
        want = 1500 + ELO_PER_NAT * truth[r.player]
        # within three of its own standard errors (coverage itself is tested below)
        assert abs(r.elo - want) <= 3 * (r.hi - r.lo) / (2 * 1.96) + 1e-9, (r, want)
        assert abs(r.elo - want) < 60


def test_anchor_is_1500_with_no_width_and_order_does_not_matter():
    truth = {"a": 0.0, "b": 0.3, "c": -0.2}
    games = simulate(truth, 30, random.Random(2))
    f1 = fit(games, truth, "a")
    shuffled = games[:]
    random.Random(3).shuffle(shuffled)
    f2 = fit(shuffled, truth, "a")
    assert np.allclose(f1.theta, f2.theta) and np.allclose(f1.cov, f2.cov)
    anchor = next(r for r in f1.ratings if r.player == "a")
    assert anchor.elo == 1500 and anchor.lo == anchor.hi == 1500


def test_an_unbeaten_player_is_finite_and_an_unplayed_one_sits_at_the_prior():
    games = [Game("a", "b", True, f"c{k}") for k in range(5)]
    f = fit(games, ["a", "b", "new"], "b")
    by = {r.player: r for r in f.ratings}
    assert math.isfinite(by["a"].elo) and by["a"].elo > 1500
    assert by["new"].games == 0 and by["new"].lo < 1500 < by["new"].hi


def test_intervals_cover_at_about_95_percent():
    truth = {"a": 0.0, "b": 0.4, "c": -0.3}
    rng = random.Random(4)
    hits = trials = 0
    for _ in range(200):
        f = fit(simulate(truth, 40, rng, luck=1.0), truth, "a")
        for r in f.ratings:
            if r.player != "a":
                trials += 1
                hits += r.lo <= 1500 + ELO_PER_NAT * truth[r.player] <= r.hi
    assert 0.90 <= hits / trials <= 0.99, hits / trials


def test_the_sandwich_sees_the_pairing():
    """A pair whose two games split 1-1 carries less doubt than two independent
    games; a pair that always agrees carries more. Same point estimate either way."""
    split = [g for k in range(30) for g in (Game("a", "b", True, f"p{k}"), Game("a", "b", False, f"p{k}"))]
    agree = [g for k in range(30) for g in (Game("a", "b", k % 2 == 0, f"p{k}"),) * 2]
    alone = [Game(g.a, g.b, g.a_won, f"{g.cluster}/{n}") for n, g in enumerate(split)]
    var = {name: fit(gs, ["a", "b"], "a").diff_var("a", "b") for name, gs in
           (("split", split), ("agree", agree), ("alone", alone))}
    assert var["split"] < var["alone"] < var["agree"]
    assert np.allclose(fit(split, ["a", "b"], "a").theta, fit(alone, ["a", "b"], "a").theta)


def test_schedule_prefers_the_newcomer_and_ignores_the_anchor():
    truth = {"a": 0.0, "b": 0.2, "c": -0.1}
    games = simulate(truth, 50, random.Random(5))
    players = [*truth, "new"]
    choice = next_pair(fit(games, players, "a"), 2, players)
    assert "new" in (choice.a, choice.b)
    ranked = {anchor: [(c.a, c.b) for c in gains(fit(games, players, anchor), 2, players)]
              for anchor in players}
    assert len({tuple(v) for v in ranked.values()}) == 1


def test_schedule_prefers_the_less_measured_pair():
    # a-b has 200 games, a-c 20, both even: the next faceoff goes to a-c's doubt.
    games = ([Game("a", "b", k % 2 == 0, f"ab{k}") for k in range(200)]
             + [Game("a", "c", k % 2 == 0, f"ac{k}") for k in range(20)]
             + [Game("b", "c", k % 2 == 0, f"bc{k}") for k in range(200)])
    ranked = gains(fit(games, ["a", "b", "c"], "a"), 2, ["a", "b", "c"])
    assert (ranked[0].a, ranked[0].b) == ("a", "c")
