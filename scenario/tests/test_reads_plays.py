"""The reads-plays study's bookkeeping (`scenario.reads_plays`): the lump
masses read off a weighted sample, the pairing of games into deal clusters,
the marks per game, restricted choice's paired difference, and the decision
change rate beside its chance baseline. The reading itself is tested in
`test_reading.py`."""

import json

import numpy as np

from scenario.reads_plays import _masses, clustered, summarize, tile_id, tile_name


def test_tiles_round_trip():
    assert [tile_id(tile_name(t)) for t in range(28)] == list(range(28))
    assert tile_name(tile_id("5-5")) == "5-5" and tile_id("4-6") == tile_id("6-4")


def test_masses_read_the_weighted_worlds_from_the_movers_chair():
    """Two worlds, weights 3:1: the mover (seat 1) sees tile 0 with his left
    (seat 2) in the first and his partner (seat 3) in the second."""
    worlds = [(0, 0b10, 0b1, 0b100), (0, 0b10, 0b100, 0b1)]
    m = _masses(worlds, [3.0, 1.0], mover=1)
    assert m.shape == (3, 28)
    assert np.allclose(m[:, 0], [0.75, 0.25, 0.0])        # left, partner, right
    assert np.allclose(m[:, 2], [0.25, 0.75, 0.0])
    assert np.allclose(m[2], 0.0)                          # the right (seat 0) holds nothing here


def test_clustered_interval_brackets_the_mean():
    v = np.array([1.0, -1.0, 2.0, 0.0, 1.0, 3.0])
    mean, lo, hi = clustered(v)
    assert mean == v.mean() and lo < mean < hi
    assert clustered(np.zeros(5)) == (0.0, 0.0, 0.0)


def _row(tile, plain, reseed, consistent=-1, plays=5, fiber=5000, fell=0):
    z = [[0.0] * 28 for _ in range(3)]
    return {"hands": [[0] * 7] * 4, "decl": 6, "bidder": 0, "bid": 30, "plays": [[0, 0]] * plays,
            "mover": plays % 4, "legal": 3, "tile": tile, "plain": plain, "consistent": consistent,
            "reseed": reseed, "fiber": fiber, "proposals": fiber, "fell_back": fell, "real_like": 1.0,
            "mass_read": z, "mass_plain": z}


def _run(tmp_path, name, likelihood, seed, games, rows):
    d = tmp_path / name
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"likelihood": likelihood, "seed": seed}))
    with open(d / "games.jsonl", "w") as f:
        for g in games:
            f.write(json.dumps(g) + "\n")
    with open(d / "decisions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(d)


def _game(idx, a_team, a, b):
    return {"game": idx, "a_team": a_team, "a_marks": a, "b_marks": b,
            "hands": [{"bidder_team": 0, "made": True, "a_team": a_team, "points": [30, 12]}]}


def test_summary_pairs_games_by_deal_and_differences_the_likelihoods(tmp_path):
    """Two game pairs per arm: the pair is the cluster, marks per game is the
    pair's sum over its two games, halved; restricted choice's value is the
    uniform arm's pair margin less the consistent arm's, pair by pair."""
    cons = _run(tmp_path, "c", "consistent", 7,
                [_game(0, 0, 7, 3), _game(0, 1, 5, 7), _game(1, 0, 7, 6), _game(1, 1, 7, 2)],
                [_row(1, 1, 1), _row(2, 3, 2), _row(4, 4, 5)])
    unif = _run(tmp_path, "u", "uniform", 7,
                [_game(0, 0, 7, 1), _game(0, 1, 7, 5), _game(1, 0, 4, 7), _game(1, 1, 7, 6)],
                [_row(1, 2, 1, consistent=1), _row(3, 3, 3, consistent=4, plays=17, fiber=90)])
    s = summarize([cons, unif])
    # consistent: pair 0 = (7−3) + (5−7) = 2, pair 1 = 1 + 5 = 6 → per game 1 and 3
    assert s["consistent"]["marks_per_game"]["mean"] == 2.0
    # uniform: pair 0 = 6 + 2 = 8, pair 1 = −3 + 1 = −2 → per game 4 and −1
    assert s["uniform"]["marks_per_game"]["mean"] == 1.5
    assert s["restricted_choice"]["marks_per_game"]["mean"] == -0.5 and s["restricted_choice"]["game_pairs"] == 2
    c = s["consistent"]
    assert c["decisions"] == {"n": 3, "changed": 1, "rate": 1 / 3}
    assert c["reseed_baseline"] == {"n": 3, "changed": 2, "rate": 2 / 3}   # plain against plain at another seed
    u = s["uniform"]
    assert u["uniform_vs_consistent"]["changed"] == 1 and u["by_path"]["exact"]["n"] == 1
    assert u["by_trick"][5]["n"] == 1 and u["by_trick"][2]["changed"] == 1
