"""Tests for void suit inference."""

import pytest
from forge.eq.voids import infer_voids
from forge.oracle.declarations import NOTRUMP


def test_empty_plays_returns_empty_voids():
    """Empty plays should return empty voids for all players."""
    voids = infer_voids([], decl_id=0)
    assert voids == {0: set(), 1: set(), 2: set(), 3: set()}


def test_single_play_that_follows_suit_no_void():
    """A play that follows suit should not mark a void."""
    # Player 0 plays domino 0 (0-0) leading with domino 0 (0-0) in blanks trump
    # Domino 0 is (0, 0) - the blank double
    # When leading with 0-0 in blanks trump, led_suit is 7 (called suit)
    # Playing 0-0 can follow suit 7 in blanks trump
    plays = [(0, 0, 0)]
    voids = infer_voids(plays, decl_id=0)
    assert voids[0] == set()


def test_single_play_that_fails_to_follow_marks_void():
    """A play that can't follow the led suit should mark a void."""
    # Player 0 leads with domino 7 (3-1) in blanks trump (decl_id=0)
    # Domino 7 is (3, 1) - not in blanks, so led_suit = 3 (the high pip)
    # Player 1 plays domino 0 (0-0) which is in blanks trump (called suit)
    # Domino 0 cannot follow suit 3, so player 1 is void in suit 3
    plays = [(1, 0, 7)]
    voids = infer_voids(plays, decl_id=0)
    assert 3 in voids[1]


def test_multiple_plays_accumulate_voids():
    """Multiple failed follows should accumulate voids."""
    # Setup: decl_id=9 (notrump) to avoid trump complications
    # Domino 27 is (6-6), domino 20 is (5-5), domino 0 is (0-0), domino 1 is (1-0)
    # Player 0 leads with domino 27 (6-6), led_suit = 6
    # Player 1 plays domino 0 (0-0), which cannot follow suit 6 → void in 6
    # Player 0 leads with domino 20 (5-5), led_suit = 5
    # Player 1 plays domino 1 (1-0), which cannot follow suit 5 → void in 5
    plays = [
        (1, 0, 27),  # Player 1 fails to follow suit 6
        (1, 1, 20),  # Player 1 fails to follow suit 5
    ]
    voids = infer_voids(plays, decl_id=NOTRUMP)
    assert voids[1] == {5, 6}
    assert voids[0] == set()
    assert voids[2] == set()
    assert voids[3] == set()
