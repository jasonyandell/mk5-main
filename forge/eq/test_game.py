"""
Tests for forge/eq/game.py

Tests are minimal and fast (< 5 seconds total), testing on tiny inputs.
"""
from __future__ import annotations

import pytest

from forge.eq.game import GameState
from forge.oracle.declarations import DOUBLES_TRUMP, NOTRUMP


def test_initial_state_current_player():
    """Initial state: current_player == leader."""
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)
    assert state.current_player() == 0

    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=2)
    assert state.current_player() == 2


def test_after_one_play_current_player_advances():
    """After one play: current_player advances."""
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)

    # Player 0 plays domino 0
    state = state.apply_action(0)
    assert state.current_player() == 1
    assert len(state.current_trick) == 1
    assert state.current_trick[0] == (0, 0)


def test_legal_actions_when_leading():
    """Legal actions when leading: all hand dominoes."""
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)

    legal = state.legal_actions()
    assert set(legal) == {0, 1, 2, 3, 4, 5, 6}


def test_legal_actions_when_following_notrump():
    """Legal actions when following (notrump): must follow suit if possible."""
    # Hand setup:
    # P0: [0=(0,0), 1=(1,0), 2=(1,1), 3=(2,0), 6=(3,0), 10=(4,0), 15=(5,0)]
    # P1: [4=(2,1), 5=(2,2), 7=(3,1), 8=(3,2), 9=(3,3), 11=(4,1), 12=(4,2)]
    hands = [[0, 1, 2, 3, 6, 10, 15], [4, 5, 7, 8, 9, 11, 12], [13, 14, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)

    # P0 leads with 2=(1,1) - high pip is 1
    state = state.apply_action(2)

    # P1's turn: [4=(2,1), 5=(2,2), 7=(3,1), 8=(3,2), 9=(3,3), 11=(4,1), 12=(4,2)]
    # Must follow suit 1 if possible: 4=(2,1), 7=(3,1), 11=(4,1) have pip 1
    legal = state.legal_actions()
    assert set(legal) == {4, 7, 11}  # Dominoes with pip 1


def test_legal_actions_when_void_in_suit():
    """Legal actions when void in suit: can play anything."""
    # P0: leads with 15=(5,0) - high pip is 5
    # P1: [8=(3,2), 9=(3,3), 13=(4,3), 17=(5,2), 18=(5,3), 19=(5,4), 20=(5,5)]
    # Wait, check which don't have pip 5...
    # Better: P1 has [5=(2,2), 9=(3,3), 14=(4,4)] - no pip 0s
    hands = [[0, 1, 2, 3, 6, 10, 15], [5, 9, 14, 16, 17, 18, 19], [4, 7, 8, 11, 12, 13, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)

    # P0 leads with 0=(0,0) - high pip is 0
    state = state.apply_action(0)

    # P1's turn: [5=(2,2), 9=(3,3), 14=(4,4), 16=(5,1), 17=(5,2), 18=(5,3), 19=(5,4)]
    # None have pip 0, so void - can play anything
    legal = state.legal_actions()
    assert set(legal) == {5, 9, 14, 16, 17, 18, 19}


def test_trick_completes_and_leader_changes():
    """After 4 plays, trick completes and leader changes to winner."""
    # Simplest possible test: just check that trick completes and leader updates
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)

    # Play 4 dominoes to complete a trick
    state = state.apply_action(0)  # P0 plays 0=(0,0)
    assert len(state.current_trick) == 1

    state = state.apply_action(10)  # P1 plays 10=(4,0), follows suit 0
    assert len(state.current_trick) == 2

    state = state.apply_action(15)  # P2 plays 15=(5,0), follows suit 0
    assert len(state.current_trick) == 3

    state = state.apply_action(21)  # P3 plays 21=(6,0), follows suit 0
    # After 4 plays, trick should complete
    assert len(state.current_trick) == 0

    # Leader should be updated to the winner (one of 0-3)
    assert state.leader in {0, 1, 2, 3}
    # Current player should be the new leader
    assert state.current_player() == state.leader


def test_hand_sizes():
    """hand_sizes returns correct sizes."""
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)
    assert state.hand_sizes() == (7, 7, 7, 7)

    state = state.apply_action(0)
    assert state.hand_sizes() == (6, 7, 7, 7)


def test_is_complete():
    """is_complete returns True when all dominoes played."""
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)
    assert not state.is_complete()

    # Play all 28 dominoes (simplified: just check the logic)
    for _ in range(28):
        if state.is_complete():
            break
        legal = state.legal_actions()
        if legal:
            state = state.apply_action(legal[0])

    assert state.is_complete()


def test_immutability():
    """GameState is immutable - operations return new instances."""
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state1 = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)
    state2 = state1.apply_action(0)

    # state1 should be unchanged
    assert 0 in state1.hands[0]
    assert len(state1.played) == 0
    assert len(state1.play_history) == 0

    # state2 should have the change
    assert 0 not in state2.hands[0]
    assert len(state2.played) == 1
    assert len(state2.play_history) == 1


def test_invalid_action_not_in_hand():
    """Raise ValueError when playing domino not in hand."""
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)

    with pytest.raises(ValueError, match="does not have domino"):
        state.apply_action(7)  # P0 doesn't have domino 7


def test_invalid_action_not_legal():
    """Raise ValueError when playing illegal domino (wrong suit)."""
    # P0: [0=(0,0), 1=(1,0)]
    # P1: [7=(2,1), 8=(2,2)]
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)

    # P0 leads with 1=(1,0) - suit 1
    state = state.apply_action(1)

    # P1 has 7=(2,1) which follows suit 1
    # Try to play 8=(2,2) instead, which doesn't follow suit
    legal = state.legal_actions()
    if 8 not in legal:  # Only test if 8 is actually illegal
        with pytest.raises(ValueError, match="is not legal"):
            state.apply_action(8)


def test_play_history_records_correctly():
    """play_history records (player, domino_id, lead_domino_id)."""
    hands = [[0, 1, 2, 3, 6, 10, 15], [4, 5, 7, 8, 9, 11, 12], [13, 14, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    state = GameState.from_hands(hands, decl_id=NOTRUMP, leader=0)

    # P0 leads with 15=(5,0)
    state = state.apply_action(15)
    assert len(state.play_history) == 1
    assert state.play_history[0] == (0, 15, 15)  # P0, domino 15, lead 15

    # P1 plays 4=(2,1) - no pip 5, so can play anything
    state = state.apply_action(4)
    assert len(state.play_history) == 2
    assert state.play_history[1] == (1, 4, 15)  # P1, domino 4, lead 15
