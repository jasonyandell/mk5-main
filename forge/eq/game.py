"""
GameState: immutable game state for E[Q] training.

Represents a partial or complete game of Texas 42 with:
- 4 players, 7 dominoes each, 7 tricks total
- Immutable state transitions via apply_action()
- Legal action determination based on follow-suit rules
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from forge.oracle.tables import (
    can_follow,
    led_suit_for_lead_domino,
    resolve_trick,
)


@dataclass(frozen=True)
class GameState:
    """Immutable game state for a single hand of Texas 42."""

    hands: tuple[tuple[int, ...], ...]  # 4 hands, each tuple of domino IDs
    played: frozenset[int]              # Global domino IDs that have been played
    play_history: tuple[tuple[int, int, int], ...]  # (player, domino_id, lead_domino_id)
    current_trick: tuple[tuple[int, int], ...]      # (player, domino_id) for current trick
    leader: int                         # Current trick leader (0-3)
    decl_id: int                        # Declaration ID

    def current_player(self) -> int:
        """Return whose turn it is (0-3)."""
        return (self.leader + len(self.current_trick)) % 4

    def legal_actions(self) -> tuple[int, ...]:
        """Return tuple of domino IDs current player can legally play."""
        player = self.current_player()
        hand = self.hands[player]

        # If leading, any domino in hand is legal
        if len(self.current_trick) == 0:
            return hand

        # Following: must follow suit if possible
        lead_domino_id = self.current_trick[0][1]
        led_suit = led_suit_for_lead_domino(lead_domino_id, self.decl_id)

        # Check which dominoes can follow
        followers = tuple(d for d in hand if can_follow(d, led_suit, self.decl_id))

        # If can follow, must follow; otherwise can play anything
        return followers if followers else hand

    def apply_action(self, domino_id: int) -> GameState:
        """Return new GameState after playing domino_id. Immutable."""
        player = self.current_player()

        # Verify domino is in player's hand
        if domino_id not in self.hands[player]:
            raise ValueError(f"Player {player} does not have domino {domino_id}")

        # Verify domino is legal
        legal = self.legal_actions()
        if domino_id not in legal:
            raise ValueError(f"Domino {domino_id} is not legal for player {player}")

        # Remove domino from hand
        new_hands = tuple(
            tuple(d for d in hand if d != domino_id) if i == player else hand
            for i, hand in enumerate(self.hands)
        )

        # Add to played set
        new_played = self.played | {domino_id}

        # Determine lead domino for this trick
        if len(self.current_trick) == 0:
            lead_domino_id = domino_id
        else:
            lead_domino_id = self.current_trick[0][1]

        # Add to play history
        new_play_history = self.play_history + ((player, domino_id, lead_domino_id),)

        # Add to current trick
        new_current_trick = self.current_trick + ((player, domino_id),)

        # Check if trick is complete
        if len(new_current_trick) == 4:
            # Resolve trick to find winner
            domino_ids = tuple(d for _, d in new_current_trick)
            outcome = resolve_trick(lead_domino_id, domino_ids, self.decl_id)

            # New leader is the winner
            new_leader = (self.leader + outcome.winner_offset) % 4

            # Clear current trick for next trick
            new_current_trick = ()
        else:
            # Trick continues
            new_leader = self.leader

        return GameState(
            hands=new_hands,
            played=new_played,
            play_history=new_play_history,
            current_trick=new_current_trick,
            leader=new_leader,
            decl_id=self.decl_id,
        )

    def hand_sizes(self) -> tuple[int, ...]:
        """Return tuple of hand sizes for each player."""
        return tuple(len(hand) for hand in self.hands)

    def is_complete(self) -> bool:
        """Return True if all dominoes have been played."""
        return len(self.played) == 28

    @classmethod
    def from_hands(
        cls,
        hands: list[list[int]],
        decl_id: int,
        leader: int = 0
    ) -> GameState:
        """Create initial game state from dealt hands."""
        if len(hands) != 4:
            raise ValueError(f"Expected 4 hands, got {len(hands)}")

        for i, hand in enumerate(hands):
            if len(hand) != 7:
                raise ValueError(f"Hand {i} has {len(hand)} dominoes, expected 7")

        # Convert to immutable tuples
        hands_tuple = tuple(tuple(hand) for hand in hands)

        return GameState(
            hands=hands_tuple,
            played=frozenset(),
            play_history=(),
            current_trick=(),
            leader=leader,
            decl_id=decl_id,
        )
