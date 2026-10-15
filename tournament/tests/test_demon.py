# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

# STRATEGY-FUSION FENCE.
# The Demon: a fenced tournament opponent; measuring stick only.
# These tests seat it at the arena's table (wiki/topics/the-demon.md).
"""The Demon's seat: legal, deterministic whole games, and a pin that is checked.

The binary's own tests (tournament/demon/tests) hold its solve to a brute-force
minimax; here it plays real faceoff games through the tournament's seat.
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest

from tournament import faceoff as F
from tournament import seats as S
from tournament.demon_fenced_seat import BINARY, DemonSeat, build, decision_seed
from tournament.participants import DEMON_SOURCE, STRAIGHT_DECLARATIONS, Participant, demon

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "scratch" / "demon-target"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture(scope="module")
def working_tree_demon(tmp_path_factory) -> Path:
    """The binary built from this checkout's working tree (what the Rust tests test)."""
    subprocess.run(["cargo", "build", "--release", "--locked", "--quiet", "--manifest-path",
                    str(ROOT / "tournament/demon/Cargo.toml"), "--target-dir", str(TARGET)], check=True)
    d = tmp_path_factory.mktemp("demon")
    (d / BINARY).symlink_to(TARGET / "release" / BINARY)
    return d


def _game(spec: F.FaceoffSpec, participants: dict, progress: Path) -> dict:
    return F.play_game(spec.__dict__, 0, 0, str(progress), participants)


@pytest.mark.parametrize("bidder,weight", [("heuristic", 1), ("heuristic", 32),
                                          ("pimc-declare:2", 1), ("pimc:2:0.75", 1)])
def test_the_demon_plays_whole_games_legally_and_the_same_game_twice(working_tree_demon, tmp_path, monkeypatch, bidder, weight):
    monkeypatch.setattr(S, "materialize", lambda p: working_tree_demon)
    head = _git("rev-parse", "HEAD")
    d = demon(head, {p: _git("rev-parse", f"HEAD:{p}") for p in DEMON_SOURCE}, worlds=8,
              bidder=bidder, declaration_weight=weight)
    foe = Participant("heur-rand", "arena", {"spec": "heuristic+random"})
    participants = {p.id: p.to_record() for p in (d, foe)}
    spec = F.FaceoffSpec("marks7", d.id, foe.id, 0x5EED, 1, "20261014T000000Z")
    first = _game(spec, participants, tmp_path / "one.json")
    second = _game(spec, participants, tmp_path / "two.json")
    assert first["done"] and max(first["marks"]) >= 7
    # The arena referee checks every play's legality; a whole game is the proof.
    assert first["hands"] == second["hands"], "same faceoff, same seeds: the same game"
    assert all(h["decl_id"] in STRAIGHT_DECLARATIONS for h in first["hands"])


def test_decision_seeds_depend_on_the_place_in_the_faceoff_only():
    assert decision_seed("g", 1, 0, 0) == decision_seed("g", 1, 0, 0)
    assert len({decision_seed("g", h, s, n) for h in (1, 2) for s in range(4) for n in (0, 4)}) == 16


def test_native_declaration_model_matches_arena_bidder(working_tree_demon):
    from arena.bidders import HeuristicBidder

    rng = random.Random(666888)
    hands = [tuple(sorted(rng.sample(range(28), 7))) for _ in range(2000)]
    requests = [json.dumps({"op": "declaration_model", "hand_mask": sum(1 << t for t in hand)})
                for hand in hands]
    result = subprocess.run([str(working_tree_demon / BINARY)], input="\n".join(requests) + "\n",
                            capture_output=True, text=True, check=True)
    got = [json.loads(line)["decl"] for line in result.stdout.splitlines()]
    bidder = HeuristicBidder()
    assert got == [bidder.declare(hand, 30, rng) for hand in hands]


def test_new_models_get_distinct_participant_ids():
    base = demon("a" * 40, {}, worlds=100)
    read = demon("a" * 40, {}, worlds=100, declaration_weight=32)
    bid = demon("a" * 40, {}, worlds=100, bidder="pimc:16:0.875")
    assert len({p.id for p in (base, read, bid)}) == 3
    assert "declaration_weight" not in base.config
    assert read.config["declaration_weight"] == 32
    assert bid.config["declarations"] == list(STRAIGHT_DECLARATIONS)
    assert "-straight-" in bid.id
    with pytest.raises(ValueError):
        demon("a" * 40, {}, worlds=100, declaration_weight=0)


def test_forecast_cannot_choose_nello_physics_even_when_it_scores_best():
    seat = object.__new__(DemonSeat)
    seat.hand_no = 1
    seat.game = "straight"
    seat.bid_spec = "pimc:16:0.875"
    seat.declarations = STRAIGHT_DECLARATIONS
    seat._forecasts = {}
    seat._ask = lambda r: ({"makes": [0, 0, 0, 0, 0, 0, 12, 13, 16, 14], "worlds": 16}
                          if r["op"] == "forecast" else {"decl": 8})
    assert seat._forecast(tuple(range(7)), 0, 30)[0] == 9


def test_the_pinned_build_checks_its_trees(tmp_path):
    head = _git("rev-parse", "HEAD")
    trees = {p: _git("rev-parse", f"HEAD:{p}") for p in DEMON_SOURCE}
    wrong = dict(trees, **{"tournament/demon": _git("rev-parse", "HEAD:tournament/tests")})
    with pytest.raises(RuntimeError, match="the pin says"):
        build(demon(head, wrong, worlds=8), tmp_path / "wrong", TARGET)
    out = tmp_path / "right"
    build(demon(head, trees, worlds=8), out, ROOT / "scratch" / "demon-pinned-target")
    reply = subprocess.run([str(out / BINARY), "8", "1"], input='{"op": "hello"}\n',
                           capture_output=True, text=True, check=True).stdout
    assert json.loads(reply) == {"demon": "0.1.0", "worlds": 8, "margin_tiebreak": True}
