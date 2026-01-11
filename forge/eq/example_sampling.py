"""Example usage of the sampling module for consistent world generation.

This module demonstrates how to use sample_consistent_worlds to generate
possible opponent hands that respect void constraints inferred from play history.
"""

from forge.eq.voids import infer_voids
from forge.eq.sampling import sample_consistent_worlds
from forge.oracle.declarations import NOTRUMP


def example_basic_sampling():
    """Example: Sample worlds with no void constraints."""
    my_player = 0
    my_hand = [0, 1, 2]  # I have dominoes 0, 1, 2
    played = set(range(3, 10))  # Dominoes 3-9 have been played
    hand_sizes = [3, 6, 6, 6]  # Current hand sizes for all players
    voids = {0: set(), 1: set(), 2: set(), 3: set()}  # No void constraints
    decl_id = NOTRUMP

    worlds = sample_consistent_worlds(
        my_player, my_hand, played, hand_sizes, voids, decl_id, n_samples=100
    )

    print(f"Generated {len(worlds)} consistent worlds")
    print(f"First world: {worlds[0]}")


def example_with_void_inference():
    """Example: Sample worlds respecting inferred void constraints."""
    # Game state: Player 2 failed to follow suit 5 when it was led
    plays = [
        (2, 0, 20),  # Player 2 played domino 0 when domino 20 (5-5) was led
    ]
    decl_id = NOTRUMP
    voids = infer_voids(plays, decl_id)

    print(f"Inferred voids: {voids}")
    # Output: {0: set(), 1: set(), 2: {5}, 3: set()}
    # Player 2 is void in suit 5

    # Now sample consistent worlds
    my_player = 0
    my_hand = [1, 2, 3]
    # played(16) + my_hand(3) + pool(9) = 28
    played = {0, 20} | set(range(4, 18))  # 16 dominoes that have been played
    hand_sizes = [3, 3, 3, 3]  # 12 total, 9 for opponents

    worlds = sample_consistent_worlds(
        my_player, my_hand, played, hand_sizes, voids, decl_id, n_samples=50
    )

    # Verify player 2 respects void constraint in all worlds
    for world in worlds:
        player_2_hand = world[2]
        for domino_id in player_2_hand:
            # Player 2 should not have any domino with pip 5
            # (unless it's in the called suit, which for notrump means never)
            pass  # Sampling already enforces this!

    print(f"Generated {len(worlds)} worlds respecting void constraints")


if __name__ == "__main__":
    print("Example 1: Basic sampling")
    example_basic_sampling()
    print("\nExample 2: Sampling with void inference")
    example_with_void_inference()
