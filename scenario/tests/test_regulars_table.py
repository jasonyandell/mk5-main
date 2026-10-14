# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""#149's table: the arms, the registered jobs, and the interaction estimator."""

from arena.bidders import HeuristicBidder, VariedBidder
from arena.regulars import RegularsPlay
from scenario.regulars import ARMS, GAMES, POINT_SEEDS, SEEDS, contrast, interaction, jobs, seats


def test_the_arms_seat_what_the_registration_says():
    for arm in ARMS:
        bid_a, play_a, bid_b, play_b = seats(arm)
        target, opp = arm.removeprefix("P").split(".")
        assert type(bid_a) is (VariedBidder if target == "Zv" else HeuristicBidder)
        assert type(bid_b) is HeuristicBidder and bid_b.caution == 0
        assert isinstance(play_b, RegularsPlay) and play_b.memory == (opp == "R")
        a, b = play_a.walt.config, play_b.walt.walt.config
        assert a.seed == (1 if target == "Za" else 0) and b.seed == 0
        assert (a.rungs, a.reading, a.tie, a.draw) == (b.rungs, b.reading, b.tie, b.draw)


def test_the_registered_jobs():
    j = jobs()
    assert len(j) == 6 * len(SEEDS) + 6 * len(POINT_SEEDS) and len(set(j)) == len(j)
    assert GAMES * len(SEEDS) == 384 and GAMES * len(POINT_SEEDS) == 192
    assert [a for a, _ in j[:6]] == [a for a in ARMS if not a.startswith("P")]


def test_the_interaction_is_the_difference_of_differences_per_pair():
    arms = {"Zv.R": {(1, 0): 3.0, (1, 1): 1.0, (2, 0): 0.0},
            "Z.R": {(1, 0): 1.0, (1, 1): 1.0, (2, 0): 0.5},
            "Zv.R0": {(1, 0): 0.0, (1, 1): 2.0},
            "Z.R0": {(1, 0): 0.0, (1, 1): 1.0}}
    c = contrast(arms, interaction("Zv"))
    assert c["game_pairs"] == 2 and c["mean"] == ((3 - 1 - 0 + 0) + (1 - 1 - 2 + 1)) / 2
    assert c["lo"] <= c["mean"] <= c["hi"]
    assert contrast(arms, interaction("Za")) == {}
