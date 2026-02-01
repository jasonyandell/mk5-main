"""Zeb game state wrapper with immutable transitions."""
from __future__ import annotations

import random
from typing import Tuple

from forge.oracle.declarations import N_DECLS
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    can_follow,
    led_suit_for_lead_domino,
    resolve_trick,
)

from .types import BidState, GamePhase, ZebGameState


def new_game(seed: int, dealer: int, skip_bidding: bool = True) -> ZebGameState:
    """Create a new game state.

    Args:
        seed: RNG seed for dealing and random choices
        dealer: Dealer seat (0-3)
        skip_bidding: If True, randomly assign bid winner and declaration

    Returns:
        Initial ZebGameState ready for play
    """
    if not 0 <= dealer <= 3:
        raise ValueError(f"dealer must be 0-3, got {dealer}")

    hands_list = deal_from_seed(seed)
    hands = tuple(tuple(h) for h in hands_list)

    if skip_bidding:
        rng = random.Random(seed)
        # Random bidder (0-3) and bid amount (30-42)
        bidder = rng.randint(0, 3)
        high_bid = rng.randint(30, 42)
        # Random declaration (0-9)
        decl_id = rng.randint(0, N_DECLS - 1)

        # Bidder leads first trick
        trick_leader = bidder

        bid_state = BidState(
            bids=(high_bid, 0, 0, 0),  # Placeholder - bidder won
            high_bidder=bidder,
            high_bid=high_bid,
        )

        return ZebGameState(
            hands=hands,
            dealer=dealer,
            phase=GamePhase.PLAYING,
            bid_state=bid_state,
            decl_id=decl_id,
            bidder=bidder,
            played=frozenset(),
            play_history=(),
            current_trick=(),
            trick_leader=trick_leader,
            team_points=(0, 0),
        )
    else:
        # Full bidding mode (not implemented in v1)
        first_bidder = (dealer + 1) % 4
        bid_state = BidState(
            bids=(-1, -1, -1, -1),
            high_bidder=-1,
            high_bid=0,
        )

        return ZebGameState(
            hands=hands,
            dealer=dealer,
            phase=GamePhase.BIDDING,
            bid_state=bid_state,
            decl_id=-1,
            bidder=-1,
            played=frozenset(),
            play_history=(),
            current_trick=(),
            trick_leader=-1,
            team_points=(0, 0),
        )


def _current_player(state: ZebGameState) -> int:
    """Return the current player to act."""
    if state.phase == GamePhase.PLAYING:
        # Current player is trick_leader + number of cards in current trick
        return (state.trick_leader + len(state.current_trick)) % 4
    raise ValueError(f"Cannot get current player in phase {state.phase}")


def _get_remaining_hand(state: ZebGameState, player: int) -> Tuple[int, ...]:
    """Return the dominoes still in a player's hand (not yet played)."""
    return tuple(d for d in state.hands[player] if d not in state.played)


def legal_actions(state: ZebGameState) -> Tuple[int, ...]:
    """Return legal slot indices (0-6) for the current player.

    Returns slot indices into the original hand, not domino IDs.
    Only slots that still contain unplayed dominoes are valid.
    """
    if state.phase != GamePhase.PLAYING:
        raise ValueError(f"Cannot get legal actions in phase {state.phase}")

    player = _current_player(state)
    hand = state.hands[player]

    # Find valid slot indices (slots with unplayed dominoes)
    valid_slots = [i for i, d in enumerate(hand) if d not in state.played]

    if not valid_slots:
        return ()

    # If leading trick, all remaining dominoes are legal
    if len(state.current_trick) == 0:
        return tuple(valid_slots)

    # Following - check which dominoes can follow the led suit
    lead_domino = state.current_trick[0]
    led_suit = led_suit_for_lead_domino(lead_domino, state.decl_id)

    followers = []
    for slot in valid_slots:
        domino_id = hand[slot]
        if can_follow(domino_id, led_suit, state.decl_id):
            followers.append(slot)

    # If no dominoes can follow, all remaining are legal
    if not followers:
        return tuple(valid_slots)

    return tuple(followers)


def apply_action(state: ZebGameState, action: int) -> ZebGameState:
    """Apply an action (slot index) and return the new state.

    Args:
        state: Current game state
        action: Slot index (0-6) into the current player's hand

    Returns:
        New immutable game state after the action
    """
    if state.phase != GamePhase.PLAYING:
        raise ValueError(f"Cannot apply action in phase {state.phase}")

    player = _current_player(state)
    hand = state.hands[player]

    if not 0 <= action < 7:
        raise ValueError(f"action must be 0-6, got {action}")

    domino_id = hand[action]

    if domino_id in state.played:
        raise ValueError(f"Domino at slot {action} already played")

    # Verify action is legal
    legal = legal_actions(state)
    if action not in legal:
        raise ValueError(f"Slot {action} is not a legal action. Legal: {legal}")

    # Add domino to current trick
    new_trick = state.current_trick + (domino_id,)
    new_played = state.played | {domino_id}
    new_history = state.play_history + ((player, domino_id),)

    # Check if trick is complete
    if len(new_trick) == 4:
        # Resolve trick
        outcome = resolve_trick(
            new_trick[0],
            (new_trick[0], new_trick[1], new_trick[2], new_trick[3]),
            state.decl_id,
        )
        winner = (state.trick_leader + outcome.winner_offset) % 4
        points = outcome.points

        # Update team points
        team = winner % 2
        new_points = list(state.team_points)
        new_points[team] += points
        new_team_points = (new_points[0], new_points[1])

        # Check if game is over (all 7 tricks played)
        if len(new_played) == 28:
            return ZebGameState(
                hands=state.hands,
                dealer=state.dealer,
                phase=GamePhase.TERMINAL,
                bid_state=state.bid_state,
                decl_id=state.decl_id,
                bidder=state.bidder,
                played=new_played,
                play_history=new_history,
                current_trick=(),
                trick_leader=winner,
                team_points=new_team_points,
            )

        # Start new trick with winner as leader
        return ZebGameState(
            hands=state.hands,
            dealer=state.dealer,
            phase=GamePhase.PLAYING,
            bid_state=state.bid_state,
            decl_id=state.decl_id,
            bidder=state.bidder,
            played=new_played,
            play_history=new_history,
            current_trick=(),
            trick_leader=winner,
            team_points=new_team_points,
        )

    # Trick not complete yet
    return ZebGameState(
        hands=state.hands,
        dealer=state.dealer,
        phase=GamePhase.PLAYING,
        bid_state=state.bid_state,
        decl_id=state.decl_id,
        bidder=state.bidder,
        played=new_played,
        play_history=new_history,
        current_trick=new_trick,
        trick_leader=state.trick_leader,
        team_points=state.team_points,
    )


def is_terminal(state: ZebGameState) -> bool:
    """Check if the game has ended."""
    return state.phase == GamePhase.TERMINAL


def get_outcome(state: ZebGameState, player: int) -> float:
    """Get the outcome for a player (-1 to 1).

    Args:
        state: Terminal game state
        player: Player seat (0-3)

    Returns:
        1.0 if player's team won (reached bid)
        -1.0 if player's team lost
        Scaled value based on margin for intermediate cases
    """
    if not is_terminal(state):
        raise ValueError("Cannot get outcome for non-terminal state")

    team = player % 2
    bidder_team = state.bidder % 2

    team0_points, team1_points = state.team_points
    my_points = team0_points if team == 0 else team1_points
    opp_points = team1_points if team == 0 else team0_points

    bid = state.bid_state.high_bid

    if team == bidder_team:
        # We bid - did we make it?
        if my_points >= bid:
            # Made the bid - scale by margin (more points = better)
            margin = (my_points - bid) / 42.0  # Normalize by max possible
            return 0.5 + 0.5 * min(margin + 0.5, 1.0)
        else:
            # Set (didn't make bid) - scale by how close we got
            deficit = (bid - my_points) / 42.0
            return -0.5 - 0.5 * min(deficit, 1.0)
    else:
        # Opponents bid - did we set them?
        if opp_points >= bid:
            # They made it - we lose, scale by their margin
            margin = (opp_points - bid) / 42.0
            return -0.5 - 0.5 * min(margin + 0.5, 1.0)
        else:
            # We set them - scale by their deficit
            deficit = (bid - opp_points) / 42.0
            return 0.5 + 0.5 * min(deficit, 1.0)


def current_player(state: ZebGameState) -> int:
    """Return the current player to act (public API)."""
    return _current_player(state)
