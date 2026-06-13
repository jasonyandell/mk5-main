"""Marks-to-7 utility tests: the race WP table and both bid utilities."""
import pytest

from champion.utility import MarkEV, MarksToSeven, race_wp


def test_race_wp_boundaries():
    assert race_wp(0, 5) == 1.0
    assert race_wp(5, 0) == 0.0
    assert race_wp(0, 0) == 1.0  # we hit the post first by convention


def test_race_wp_symmetry_and_diagonal():
    for a in range(1, 8):
        assert race_wp(a, a) == pytest.approx(0.5)
        for b in range(1, 8):
            assert race_wp(a, b) == pytest.approx(1.0 - race_wp(b, a))


def test_race_wp_known_values():
    # We need 7, they need 1: we win only by taking 7 straight coin flips.
    assert race_wp(7, 1) == pytest.approx(2.0 ** -7)
    assert race_wp(1, 7) == pytest.approx(1.0 - 2.0 ** -7)


def test_race_wp_monotone_in_needs():
    for a in range(1, 7):
        for b in range(1, 7):
            assert race_wp(a, b) > race_wp(a + 1, b)
            assert race_wp(a, b) < race_wp(a, b + 1)


def test_mark_ev_swing():
    ev = MarkEV()
    args = dict(team=0, marks=(0, 0), marks_to_win=7)
    assert ev.value(0.5, 30, **args) == pytest.approx(0.0)
    assert ev.value(0.75, 30, **args) == pytest.approx(0.5)
    assert ev.value(0.75, 84, **args) == pytest.approx(1.0)  # two marks at stake
    assert ev.value(0.25, 35, **args) == pytest.approx(-0.5)


def test_marks_to_seven_one_mark_sign_is_score_free():
    """Pascal identity: 1-mark contracts flip sign at p = 1/2 at EVERY score."""
    wp = MarksToSeven()
    for marks in [(0, 0), (6, 6), (6, 0), (0, 6), (3, 5)]:
        for team in (0, 1):
            args = dict(team=team, marks=marks, marks_to_win=7)
            assert wp.value(0.51, 30, **args) > 0
            assert wp.value(0.49, 30, **args) < 0


def test_marks_to_seven_six_six_is_pure_coin():
    """At 6-6 a 1-mark contract decides the game: U = p - 1/2 exactly."""
    wp = MarksToSeven()
    args = dict(team=0, marks=(6, 6), marks_to_win=7)
    for p in (0.2, 0.5, 0.9):
        assert wp.value(p, 30, **args) == pytest.approx(p - 0.5)


def test_marks_to_seven_two_mark_thresholds_move_with_score():
    """84 needs p > 3/4 ahead 6-0, p > 1/2 even, p > 1/4 behind 0-6."""
    wp = MarksToSeven()

    def u(p, marks):
        return wp.value(p, 84, team=0, marks=marks, marks_to_win=7)

    assert u(0.74, (6, 0)) < 0 < u(0.76, (6, 0))
    assert u(0.49, (0, 0)) < 0 < u(0.51, (0, 0))
    assert u(0.24, (0, 6)) < 0 < u(0.26, (0, 6))


def test_marks_to_seven_desperation_84_beats_safe_30():
    """Behind 0-6, a p=0.3 two-mark gamble is right while p=0.45 at 30 is not
    — the walk past negative point bids onto 84 is load-bearing."""
    wp = MarksToSeven()
    args = dict(team=0, marks=(0, 6), marks_to_win=7)
    assert wp.value(0.45, 30, **args) < 0
    assert wp.value(0.30, 84, **args) > 0
