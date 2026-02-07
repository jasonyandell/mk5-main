"""Tests for consistent world sampling."""

import pytest
from forge.eq.sampling import sample_consistent_worlds, hand_violates_voids
from forge.oracle.declarations import NOTRUMP


def test_hand_violates_voids_empty_voids():
    """Hand with no void constraints should never violate."""
    hand = [0, 1, 2]
    voids = set()
    assert not hand_violates_voids(hand, voids, decl_id=0)


def test_hand_violates_voids_pip_suit():
    """Hand containing a domino in a void pip suit should violate."""
    # Domino 27 is (6-6), void in suit 6
    hand = [27]
    voids = {6}
    assert hand_violates_voids(hand, voids, decl_id=NOTRUMP)


def test_hand_violates_voids_called_suit():
    """Hand containing a domino in void called suit should violate."""
    # Domino 0 is (0-0), in blanks trump (decl_id=0)
    hand = [0]
    voids = {7}  # Void in called suit
    assert hand_violates_voids(hand, voids, decl_id=0)


def test_hand_violates_voids_trump_excludes_pip():
    """Domino in trump should not violate pip void."""
    # Domino 0 is (0-0), in blanks trump (decl_id=0)
    # Even though it has pip 0, it's in trump so doesn't count for pip suit 0
    hand = [0]
    voids = {0}  # Void in pip 0
    assert not hand_violates_voids(hand, voids, decl_id=0)


def test_sample_no_voids_small():
    """Sample with no void constraints should always succeed quickly."""
    my_player = 0
    my_hand = [0, 1]
    # Must account for all 28 dominoes: my_hand(2) + played(20) + opponents(6) = 28
    played = set(range(2, 22))  # 20 played dominoes
    hand_sizes = [2, 2, 2, 2]  # 2 cards each
    voids = {0: set(), 1: set(), 2: set(), 3: set()}
    decl_id = NOTRUMP

    worlds = sample_consistent_worlds(
        my_player, my_hand, played, hand_sizes, voids, decl_id, n_samples=3
    )

    assert len(worlds) == 3
    for world in worlds:
        # Check my hand is preserved
        assert world[0] == my_hand
        # Check all hands have correct size
        for player, hand in enumerate(world):
            assert len(hand) == hand_sizes[player]
        # Check no overlap
        all_dominoes = set()
        for hand in world:
            all_dominoes.update(hand)
        assert len(all_dominoes) == sum(hand_sizes)
        # Check no played dominoes
        assert all_dominoes.isdisjoint(played)


def test_sample_with_simple_void():
    """Sample respecting a simple void constraint."""
    my_player = 0
    my_hand = [0, 1]  # Dominoes (0,0) and (1,0)
    # Pool will be {4-20} (no pip 6)
    # Must account for all 28: my_hand(2) + played(9) + opponents(17) = 28
    played = set(range(21, 28))  # All dominoes with pip 6 are played (21-27 = 7 dominoes)
    played.update([2, 3])  # Plus 2 more to get total to 9 played
    hand_sizes = [2, 6, 6, 5]  # Total 19 cards in hands (2 mine, 17 opponents)
    # Player 1 is void in suit 6 - since all pip-6 dominoes are played, this is easy to satisfy
    voids = {0: set(), 1: {6}, 2: set(), 3: set()}
    decl_id = NOTRUMP

    worlds = sample_consistent_worlds(
        my_player, my_hand, played, hand_sizes, voids, decl_id, n_samples=2
    )

    assert len(worlds) == 2
    for world in worlds:
        # Player 1 should not have any domino containing pip 6
        player_1_hand = world[1]
        # Check it doesn't violate the void
        assert not hand_violates_voids(player_1_hand, voids[1], decl_id)
        # Since all pip-6 dominoes are played, this should be trivially true
        for domino_id in player_1_hand:
            assert domino_id not in range(21, 28)


def test_sample_impossible_constraint():
    """Impossible constraint should raise RuntimeError."""
    my_player = 0
    my_hand = list(range(7))  # 7 dominoes
    played = set()
    hand_sizes = [7, 7, 7, 7]
    # Player 1 must have 7 cards but is void in all suits - impossible
    voids = {0: set(), 1: {0, 1, 2, 3, 4, 5, 6, 7}, 2: set(), 3: set()}
    decl_id = NOTRUMP

    with pytest.raises(RuntimeError, match="No valid hand distribution exists"):
        sample_consistent_worlds(
            my_player,
            my_hand,
            played,
            hand_sizes,
            voids,
            decl_id,
            n_samples=1,
            max_attempts_per_sample=10,
        )


def test_sample_preserves_my_hand():
    """My hand should always appear at my_player position."""
    my_player = 2  # Not position 0
    my_hand = [10, 11, 12]
    # Must account for all 28: my_hand(3) + played(19) + opponents(6) = 28
    played = set(list(range(0, 10)) + list(range(13, 22)))  # 19 played
    hand_sizes = [2, 2, 3, 2]
    voids = {0: set(), 1: set(), 2: set(), 3: set()}
    decl_id = NOTRUMP

    worlds = sample_consistent_worlds(
        my_player, my_hand, played, hand_sizes, voids, decl_id, n_samples=2
    )

    for world in worlds:
        assert world[2] == my_hand


def test_sample_validates_hand_size_mismatch():
    """Should raise ValueError if my_hand size doesn't match hand_sizes."""
    my_player = 0
    my_hand = [0, 1]  # 2 cards
    played = set()
    hand_sizes = [3, 2, 2, 2]  # Says I should have 3
    voids = {0: set(), 1: set(), 2: set(), 3: set()}
    decl_id = NOTRUMP

    with pytest.raises(ValueError, match="my_hand length"):
        sample_consistent_worlds(
            my_player, my_hand, played, hand_sizes, voids, decl_id, n_samples=1
        )


def test_sample_validates_pool_size():
    """Should raise ValueError if pool doesn't match total needed."""
    my_player = 0
    my_hand = [0, 1]
    played = {2, 3, 4, 5, 6}  # 5 played
    hand_sizes = [2, 2, 2, 2]  # Total 8 cards
    # Pool should be 28 - 5 - 2 = 21, but opponents need 6 cards
    voids = {0: set(), 1: set(), 2: set(), 3: set()}
    decl_id = NOTRUMP

    with pytest.raises(ValueError, match="Pool size"):
        sample_consistent_worlds(
            my_player, my_hand, played, hand_sizes, voids, decl_id, n_samples=1
        )
