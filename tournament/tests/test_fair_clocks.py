# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""FAIR CLOCKS (CLAUDE.md, #181): every scheduled player thinks on a deterministic budget.

Offline: the records are the real ones; nothing is seated or played.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tournament import seats
from tournament import worker as W
from tournament.participants import (ANCHOR, FAIR_WALT, ROSTER, WALT, WITHDRAWN, Participant,
                                     deterministic_budget, schedulable)
from tournament.rating import Game, fit

HOST = Path(seats.HOST)


def test_every_scheduled_roster_participant_declares_a_deterministic_budget():
    scheduled = [p for p in ROSTER if schedulable(p.to_record())]
    assert FAIR_WALT in scheduled and len(scheduled) >= 2
    for p in scheduled:
        assert p.config.get("clock") != "real", p.id
        assert deterministic_budget(p.to_record()), p.id
    # a wall-clock participant on the roster can only be a withdrawn one
    for p in ROSTER:
        if p.config.get("clock") == "real":
            assert p.id in WITHDRAWN and not schedulable(p.to_record())


def test_a_real_clock_is_never_scheduled_or_seated():
    clocked = Participant("clocked", "arena", {"spec": "bid30+random"}, {"clock": "real"})
    other = Participant("other", "arena", {"spec": "heuristic+random"})
    records = {p.id: p.to_record() for p in (clocked, other, FAIR_WALT)}
    assert W.runnable(records, {"node"}) == ["other", FAIR_WALT.id]  # never "clocked"
    assert not deterministic_budget(clocked.to_record())
    with pytest.raises(ValueError, match="FAIR CLOCKS"):
        seats.seat(clocked, "g")
    with pytest.raises(ValueError, match="FAIR CLOCKS"):
        seats.seat(WALT, "g")


def test_a_walt_seat_must_say_which_clock_and_its_step():
    for config in ({}, {"clock": "virtual"}, {"clock": "virtual", "clock_step_us": 0}):
        assert not deterministic_budget(Participant("w", "plunge-walt", {}, config).to_record())
    assert deterministic_budget(Participant("w", "plunge-walt", {}, {"clock": "virtual", "clock_step_us": 0.5}).to_record())


def test_the_host_refuses_a_walt_without_a_clock(tmp_path):
    for config in ("{}", '{"clock":"virtual"}'):
        run = subprocess.run(["node", str(HOST), "walt", str(tmp_path), config], input="", capture_output=True, text=True)
        assert run.returncode != 0 and "clock" in run.stderr


def test_the_fair_walt_is_production_walt_bytes_on_a_virtual_clock():
    assert FAIR_WALT.id == WALT.id + "-fair" and FAIR_WALT.kind == WALT.kind == "plunge-walt"
    assert FAIR_WALT.pin == WALT.pin  # the same commit and the same sha256 of the wasm, book and manifest
    assert FAIR_WALT.config["clock"] == "virtual" and WALT.config["clock"] == "real"
    assert {k: v for k, v in FAIR_WALT.config.items() if not k.startswith("clock")} == \
        {k: v for k, v in WALT.config.items() if not k.startswith("clock")}


def test_production_walt_is_withdrawn_its_records_kept_and_the_fair_walt_anchors():
    assert ANCHOR == FAIR_WALT.id and WALT.id in WITHDRAWN and not schedulable(WALT.to_record())
    runnable = W.runnable({p.id: p.to_record() for p in ROSTER}, {"node"})
    assert FAIR_WALT.id in runnable and WALT.id not in runnable
    # its faceoffs still rate: a withdrawn player's games are in the fit, only scheduling stops
    rec = {"id": "f1", "a": WALT.id, "b": "x", "participants": {WALT.id: WALT.to_record()},
           "games": [{"pair": 0, "a_won": True}]}
    assert len(W.games_of([rec])) == 1


def test_reanchoring_only_shifts_the_scale():
    # Bradley-Terry keeps differences: anchoring on the fair Walt instead of production moves
    # every point rating by one constant, so every gap between two players is unchanged.
    names = ["old", "fair", "zeb", "demon"]
    beats = [("old", "zeb", 7, 3), ("fair", "zeb", 6, 4), ("old", "demon", 4, 6), ("fair", "demon", 5, 5),
             ("old", "fair", 5, 5), ("zeb", "demon", 7, 3)]
    games = [Game(a, b, i < w, f"{a}{b}/{i}") for a, b, w, n in beats for i in range(n)]
    on_old, on_fair = fit(games, names, "old"), fit(games, names, "fair")
    elo = lambda f: {r.player: r for r in f.ratings}
    shift = elo(on_old)["fair"].elo - 1500
    for n in names:
        assert elo(on_fair)[n].elo + shift == pytest.approx(elo(on_old)[n].elo, abs=1e-6)
        if n not in ("old", "fair"):
            assert elo(on_fair)[n].hi - elo(on_fair)[n].lo != 0
    assert elo(on_fair)["fair"].elo == 1500
