"""
Example usage of transcript_tokenize for Stage 2 (E[Q] training).

This shows how to convert a game state to tokens for transformer input.
"""

from forge.eq import tokenize_transcript
from forge.oracle.declarations import DECL_ID_TO_NAME


def example_game_in_progress():
    """Example: Game in progress from player 1's perspective."""
    print("=" * 60)
    print("EXAMPLE: Game in progress from player 1's perspective")
    print("=" * 60)

    # Player 1's remaining hand (3 dominoes)
    my_hand = [5, 7, 12]  # (2,0), (2,2), (4,2)

    # Plays so far (absolute player IDs)
    plays = [
        (0, 20),  # Player 0 led with (5,5) - 10 points!
        (1, 15),  # Player 1 (me) played (5,0) - 5 points
    ]

    # Declaration and perspective
    decl_id = 5  # Fives
    current_player = 1  # I am player 1

    print(f"\nDeclaration: {DECL_ID_TO_NAME[decl_id]}")
    print(f"Current player: {current_player}")
    print(f"My hand: {my_hand}")
    print(f"Plays so far: {plays}")

    # Tokenize
    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    print(f"\nOutput shape: {tokens.shape}")
    print(f"Output dtype: {tokens.dtype}")
    print(f"\nSequence length: {tokens.shape[0]} tokens")
    print(f"  = 1 declaration + {len(my_hand)} hand + {len(plays)} plays")

    # Show token breakdown
    print("\nToken breakdown:")
    print(f"  [0]: Declaration token")
    print(f"  [1-{len(my_hand)}]: Hand tokens")
    print(f"  [{len(my_hand) + 1}-{tokens.shape[0] - 1}]: Play tokens")

    # Show first few tokens
    print("\nFirst few tokens (features):")
    for i in range(min(5, tokens.shape[0])):
        print(f"  Token {i}: {tokens[i].tolist()}")

    return tokens


def example_start_of_hand():
    """Example: Start of hand, no plays yet."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Start of hand, no plays yet")
    print("=" * 60)

    # Full hand at start
    my_hand = [0, 1, 2, 3, 4, 5, 6]  # All 7 dominoes

    # No plays yet
    plays = []

    # Declaration and perspective
    decl_id = 7  # Doubles trump
    current_player = 0

    print(f"\nDeclaration: {DECL_ID_TO_NAME[decl_id]}")
    print(f"Current player: {current_player}")
    print(f"My hand: {my_hand} (full hand)")
    print(f"Plays so far: {plays} (none yet)")

    # Tokenize
    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    print(f"\nOutput shape: {tokens.shape}")
    print(f"Sequence: 1 decl + 7 hand + 0 plays = {tokens.shape[0]} tokens")

    return tokens


def example_relative_players():
    """Example: Demonstrate relative player ID encoding."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Relative player ID encoding")
    print("=" * 60)

    # Minimal hand
    my_hand = [0]

    # Four plays from different absolute players
    plays = [
        (2, 1),  # Absolute player 2
        (3, 2),  # Absolute player 3
        (0, 3),  # Absolute player 0
        (1, 4),  # Absolute player 1
    ]

    decl_id = 0
    current_player = 2  # We are player 2

    print(f"\nCurrent player: {current_player}")
    print(f"Plays (absolute player, domino):")
    for abs_player, domino in plays:
        rel_player = (abs_player - current_player) % 4
        role = ["me", "left", "partner", "right"][rel_player]
        print(f"  Player {abs_player} -> Relative {rel_player} ({role})")

    # Tokenize
    tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)

    print(f"\nRelative player IDs in tokens (feature 4):")
    for i, (abs_player, _) in enumerate(plays):
        token_idx = 2 + i  # Skip decl and hand tokens
        rel_player = tokens[token_idx, 4].item()
        expected = (abs_player - current_player) % 4
        role = ["me", "left", "partner", "right"][rel_player]
        print(f"  Token {token_idx}: player {rel_player} ({role}) - " f"absolute {abs_player}")
        assert rel_player == expected, "Mismatch!"

    print("\n✓ All relative player IDs correct!")

    return tokens


if __name__ == "__main__":
    example_game_in_progress()
    example_start_of_hand()
    example_relative_players()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
