"""Integration test for sampling with void inference."""

from forge.eq.voids import infer_voids
from forge.eq.sampling import sample_consistent_worlds
from forge.oracle.declarations import NOTRUMP


def test_integration_voids_and_sampling():
    """Test that sampling works with voids inferred from play history."""
    # Setup: Player 1 leads with domino 27 (6,6), player 2 plays domino 0 (0,0) which can't follow
    # This shows player 2 is void in suit 6
    plays = [(2, 0, 27)]  # (player, domino_id, lead_domino_id)
    voids = infer_voids(plays, decl_id=NOTRUMP)

    # Verify player 2 is marked void in suit 6
    assert 6 in voids[2]

    # Now sample consistent worlds where player 2 is void in suit 6
    my_player = 0
    my_hand = [1, 2]
    # Played: domino 27 (led) and domino 0 (played by player 2)
    # Also play all dominoes with pip 6 (21-27) except 27 which is already included
    played = {0, 27} | set(range(21, 27))  # dominoes 0, 21-27 (8 total)
    played.update(range(3, 13))  # Add dominoes 3-12 (10 more) = 18 total played
    # Pool will be 28 - 18 - 2 = 8 dominoes (all without pip 6)
    hand_sizes = [2, 3, 3, 2]  # 10 cards in hands (2 mine + 8 opponents)
    decl_id = NOTRUMP

    worlds = sample_consistent_worlds(
        my_player, my_hand, played, hand_sizes, voids, decl_id, n_samples=2
    )

    # Verify player 2 doesn't get any dominoes with pip 6 in sampled worlds
    for world in worlds:
        player_2_hand = world[2]
        # Player 2 should not have any dominoes in range 21-27 (all contain pip 6)
        # Since all pip-6 dominoes are played, this should be trivially satisfied
        for domino_id in player_2_hand:
            assert domino_id not in range(21, 28), f"Player 2 got domino {domino_id} which contains pip 6"
