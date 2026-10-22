# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""The Walt endgame probe (#244): the course is Walt's own model, whole, and its exact
solve is what Walt computes when it enumerates."""
from dataclasses import replace

import pytest

from lem import endgame
from scenario import atlas
from scenario.planner import Config, Rung
from scenario.record import empty_record
from scenario.rules import rules_for, tiles_of


def endgame_state(seed: int, n_plays: int):
    """Deal `seed`, seat 0 bidding 30 in suit `seed % 7`, every seat playing its lowest
    legal tile until `n_plays` tiles are down."""
    hands, decl = atlas.deal(seed), seed % 7
    rules, rec, plays = rules_for(decl), empty_record(0), ()
    while len(plays) < n_plays:
        seat = rec.seat
        tile = tiles_of(rules.legal(hands[seat] & ~rec.played, rec.led))[0]
        plays, rec = plays + ((seat, tile),), rec.advance(tile, rules)
    return atlas._state(tuple(tuple(tiles_of(h)) for h in hands), 0, 30, decl, plays, rec)


@pytest.fixture(scope="module")
def zeb():
    return endgame.walt_config("zeb")


@pytest.mark.parametrize("seed,n_plays", [(1, 16), (2, 17), (3, 19), (4, 20), (7, 22)])
def test_the_exact_twin_is_the_course_solved(zeb, seed, n_plays):
    row, c = endgame.probe(zeb, endgame_state(seed, n_plays), 30)
    assert len(row["menu"]) > 1 and row["twin_agrees"]
    assert sorted(int(t) for t in row["exact"]) == row["menu"] == sorted(int(t) for t in row["q"])
    assert set(row["best"]) <= set(row["menu"]) and row["best"]
    if not row["sampled"]:   # enumerated: Walt is the twin, so its tile is in the set
        assert row["tile"] in row["best"]


def test_every_observation_partitions_the_worlds_behind_a_play(zeb):
    row, c = endgame.probe(zeb, endgame_state(1, 16), 30)
    choices = c["choices"]
    assert int(choices[c["root"]]["mass"]) == row["fiber"]
    for ch in choices:
        for t in ch["menu"]:
            behind = [int(choices[b["next"]]["mass"]) for b in ch["blocks"] if b["play"] == t]
            assert behind == [] or sum(behind) == int(ch["mass"])
        assert all(0 <= int(p) <= int(ch["mass"]) for p in ch["payoff"])


def test_the_probe_reads_only_a_modelled_walt(zeb):
    with pytest.raises(ValueError, match="dice"):
        endgame.course(endgame.Walt(Config((Rung(512, None),), "make")), endgame_state(1, 16), 30)
    with pytest.raises(ValueError, match="uniform"):
        endgame.course(endgame.Walt(replace(zeb, draw="posterior")), endgame_state(1, 16), 30)
