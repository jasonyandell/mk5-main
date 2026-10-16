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


@pytest.mark.parametrize("bidder,weight,bounded", [("heuristic", 1, False), ("heuristic", 32, False),
                                                  ("pimc-declare:2", 1, False), ("pimc:2:0.75", 1, False),
                                                  ("pimc-over:2:0.75:0.5", 1, False),
                                                  ("pimc-jump:4:0.75:0.5", 1, True),
                                                  ("pimc-bait:4:0.75:0.5", 1, True)])
def test_the_demon_plays_whole_games_legally_and_the_same_game_twice(working_tree_demon, tmp_path, monkeypatch, bidder, weight, bounded):
    monkeypatch.setattr(S, "materialize", lambda p: working_tree_demon)
    head = _git("rev-parse", "HEAD")
    d = demon(head, {p: _git("rev-parse", f"HEAD:{p}") for p in DEMON_SOURCE}, worlds=8,
              bidder=bidder, declaration_weight=weight, bounded_forecast=bounded)
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


@pytest.mark.parametrize("op", ["forecast", "forecast_best", "forecast_support"])
def test_native_forecast_requires_explicit_valid_declarations(working_tree_demon, op):
    base = {"op": op, "hand_mask": sum(1 << t for t in range(21, 28)),
            "seat": 0, "bid": 42, "worlds": 1, "seed": 7, "required": 1}
    requests = [{**base, "declarations": ds} for ds in
                ([6], [], [6, 6], [10], [256], [True], [-1], [1.5])] + [base]
    result = subprocess.run([str(working_tree_demon / BINARY)],
                            input="".join(json.dumps(r) + "\n" for r in requests),
                            capture_output=True, text=True, check=True)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    if op == "forecast_support":
        assert replies[0]["supported"] is True and replies[0]["worlds"] == 1
    else:
        assert replies[0]["makes"] == [None] * 6 + [1] + [None] * 3
    assert all("error" in r for r in replies[1:])


def test_new_models_get_distinct_participant_ids():
    base = demon("a" * 40, {}, worlds=100)
    read = demon("a" * 40, {}, worlds=100, declaration_weight=32)
    bid = demon("a" * 40, {}, worlds=100, bidder="pimc:16:0.875")
    over = demon("a" * 40, {}, worlds=100, bidder="pimc-over:16:0.875:0.5")
    bounded = demon("a" * 40, {}, worlds=100, bidder="pimc-over:16:0.875:0.5", bounded_forecast=True)
    jump = demon("a" * 40, {}, worlds=100, bidder="pimc-jump:16:0.875:0.5", bounded_forecast=True)
    support = demon("a" * 40, {}, worlds=100, bidder="pimc-over:16:0.875:0.5",
                    bounded_forecast=True, support_forecast=True)
    bait = demon("a" * 40, {}, worlds=100, bidder="pimc-bait:16:0.875:0.5", bounded_forecast=True)
    assert len({p.id for p in (base, read, bid, over, bounded, jump, support, bait)}) == 8
    assert "support_forecast" not in bounded.config
    assert support.config["support_forecast"] is True
    with pytest.raises(ValueError):
        demon("a" * 40, {}, worlds=100, bidder="pimc-over:16:0.875:0.5", support_forecast=True)
    assert "declaration_weight" not in base.config
    assert read.config["declaration_weight"] == 32
    assert bid.config["declarations"] == list(STRAIGHT_DECLARATIONS)
    assert "-straight-" in bid.id
    assert "bounded_forecast" not in over.config
    assert bounded.config["bounded_forecast"] is True
    with pytest.raises(ValueError):
        demon("a" * 40, {}, worlds=100, declaration_weight=0)
    with pytest.raises(ValueError):
        demon("a" * 40, {}, worlds=100, bounded_forecast=True)


def test_forecast_cannot_choose_nello_physics_even_when_it_scores_best():
    seat = object.__new__(DemonSeat)
    seat.support_forecast = False
    seat.hand_no = 1
    seat.game = "straight"
    seat.bid_spec = "pimc:16:0.875"
    seat.bounded_forecast = False
    seat.declarations = STRAIGHT_DECLARATIONS
    seat._forecasts = {}
    seat._ask = lambda r: ({"makes": [0, 0, 0, 0, 0, 0, 12, 13, 16, 14], "worlds": 16}
                          if r["op"] == "forecast" else {"decl": 8})
    assert seat._forecast(tuple(range(7)), 0, 30)[0] == 9


@pytest.mark.parametrize("bidder", ["pimc-over:8:0.875:0.5", "pimc-jump:8:0.875:0.5", "pimc-bait:8:0.875:0.5"])
def test_bounds_preserve_complete_bidding_games(working_tree_demon, tmp_path, monkeypatch, bidder):
    from dataclasses import replace

    monkeypatch.setattr(S, "materialize", lambda p: working_tree_demon)
    head = _git("rev-parse", "HEAD")
    trees = {p: _git("rev-parse", f"HEAD:{p}") for p in DEMON_SOURCE}
    foe = Participant("heur-rand", "arena", {"spec": "heuristic+random"})
    games = []
    for bounded, support in ((False, False), (True, False), (True, True)):
        d = replace(demon(head, trees, worlds=8, bidder=bidder,
                          bounded_forecast=bounded, support_forecast=support), id="demon-parity")
        spec = F.FaceoffSpec("marks7", d.id, foe.id, 0xB0A1D, 1, "20261016T000000Z")
        games.append(_game(spec, {p.id: p.to_record() for p in (d, foe)}, tmp_path / f"{bounded}-{support}.json"))
    assert all(g["hands"] == games[0]["hands"] and g["marks"] == games[0]["marks"] for g in games[1:])


def test_jump_bids_the_highest_supported_legal_point_and_passes_over_partner():
    from arena.auction import BidContext, legal_bids

    seat = object.__new__(DemonSeat)
    seat.support_forecast = False
    seat._bid_seats = {}
    seat.bid_spec = "pimc-jump:64:0.875:0.5"
    requests = []

    def forecast(hand, bidder, target):
        requests.append(target)
        count = 56 if target <= 37 else 32 if target <= 40 else 31
        decl = target % 7  # The best declaration may change with the target.
        makes = [None] * 10
        makes[decl] = count
        return decl, {"makes": makes, "worlds": 64}

    seat._forecast = forecast
    for seat_id in range(4):
        for holder, high, expected in [(-1, 0, 37), ((seat_id + 1) % 4, 30, 40),
                                       ((seat_id + 1) % 4, 39, 40), ((seat_id + 1) % 4, 40, 0),
                                       ((seat_id + 1) % 4, 42, 0), ((seat_id + 1) % 4, 84, 0),
                                       ((seat_id + 2) % 4, 30, 0)]:
            requests.clear()
            ctx = BidContext(tuple(range(7)), seat_id, 0, (-1,) * 4, high, holder, legal_bids(high))
            assert seat.bid(ctx, random.Random(0)) == expected
            assert len(requests) <= 5
            if high >= 42 or holder == (seat_id + 2) % 4:
                assert requests == []


@pytest.mark.parametrize("seat_id", range(4))
@pytest.mark.parametrize("bidder", ["pimc-over:64:0.875:0.5", "pimc-bait:64:0.875:0.5"])
def test_overcalling_changes_only_opponent_held_point_auctions(seat_id, bidder):
    from arena.auction import BidContext, legal_bids

    seat = object.__new__(DemonSeat)
    seat.support_forecast = False
    seat._bid_seats = {}
    seat.bid_spec = bidder
    requests = []
    count = 32

    def forecast(hand, bidder, target):
        requests.append((hand, bidder, target))
        return 6, {"makes": [0] * 6 + [count] + [0, None, 0], "worlds": 64}

    seat._forecast = forecast
    hand = tuple(range(7))

    def call(holder, high):
        return seat.bid(BidContext(hand, seat_id, 0, tuple(-1 if s == seat_id else 0 for s in range(4)), high, holder,
                                  legal_bids(high)), random.Random(0))

    assert call(-1, 0) == 0  # 50% is not enough to open.
    count = 56
    assert call(-1, 0) == 30  # The original 87.5% opening boundary.
    count = 32
    assert call((seat_id + 1) % 4, 35) == 36
    assert requests[-1] == (hand, seat_id, 36)
    count = 31
    assert call((seat_id + 3) % 4, 35) == 0
    count = 64
    n = len(requests)
    assert call((seat_id + 2) % 4, 35) == 0  # Never overcall partner.
    assert call((seat_id + 1) % 4, 42) == 0  # No point raise remains.
    assert call((seat_id + 1) % 4, 84) == 0
    assert len(requests) == n
    seat.bid_spec = "pimc:64:0.875"
    count = 32
    assert call((seat_id + 1) % 4, 35) == 0  # Existing pins keep their rule.


@pytest.mark.parametrize("spec", ["pimc-over:64:0.875", "pimc-over:0:0.875:0.5",
                                 "pimc-over:64:nan:0.5", "pimc-over:64:0.875:1.1",
                                 "pimc-jump:64:0.875", "pimc-jump:4097:0.875:0.5",
                                 "pimc-bait:64:0.875", "pimc-bait:64:0.875:nan"])
def test_invalid_overcall_parameters_fail_before_starting_a_seat(tmp_path, spec):
    p = Participant("bad", "demon", {}, {"bidder": spec})
    with pytest.raises(ValueError, match="pimc-over"):
        DemonSeat(p, tmp_path, "invalid")


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


@pytest.mark.parametrize("seat_id", range(4))
def test_bait_opens_only_in_the_first_two_positions_without_a_standing_bid(seat_id):
    from arena.auction import BidContext, bidding_order, legal_bids

    seat = object.__new__(DemonSeat)
    seat.support_forecast = False
    seat._bid_seats = {}
    seat.bid_spec = "pimc-bait:64:0.875:0.875"
    requests = []
    def forecast(hand, bidder, target):
        requests.append(target)
        return 6, {"makes": [None] * 6 + [0] + [None] * 3, "worlds": 64}
    seat._forecast = forecast
    for position in range(4):
        dealer = (seat_id - position - 1) % 4
        prior = bidding_order(dealer)[:position]
        bids = tuple(0 if s in prior else -1 for s in range(4))
        ctx = BidContext(tuple(range(7)), seat_id, dealer, bids, 0, -1, legal_bids(0))
        requests.clear()
        assert seat.bid(ctx, random.Random(0)) == (30 if position < 2 else 0)
        assert requests == ([] if position < 2 else [30])


def test_required_counts_preserve_float_cutoff_boundaries():
    import math
    from tournament.demon_fenced_seat import required_makes

    assert math.ceil(0.07 * 100) == 8
    assert required_makes(0.07, 100) == 7
    for n in (1, 8, 100, 4096):
        for k in (0, n // 3, n // 2, n):
            c = k / n
            for cutoff in (math.nextafter(c, 0.0), c, math.nextafter(c, 1.0)):
                expected = next(i for i in range(n + 1) if i / n >= cutoff)
                assert required_makes(cutoff, n) == expected
    for cutoff, n in ((float("nan"), 64), (float("inf"), 64), (-0.1, 64), (1.1, 64), (0.5, 0)):
        with pytest.raises(ValueError):
            required_makes(cutoff, n)
