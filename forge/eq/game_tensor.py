"""
GPU-vectorized game state for E[Q] training.

Processes N games in parallel on GPU, matching CPU GameState semantics exactly.
Pre-computes lookup tables for efficient batch operations.
"""
from __future__ import annotations

import torch
from forge.oracle.tables import (
    DOMINO_HIGH,
    DOMINO_LOW,
    DOMINO_IS_DOUBLE,
    DOMINO_COUNT_POINTS,
)
from forge.oracle.declarations import (
    PIP_TRUMP_IDS,
    DOUBLES_TRUMP,
    DOUBLES_SUIT,
    NOTRUMP,
    N_DECLS,
    has_trump_power,
)
from forge.eq.sampling_gpu import CAN_FOLLOW


# Pre-computed lookup tensors (module-level constants)

def _build_led_suit_table() -> torch.Tensor:
    """Build lookup tensor for led suit determination.

    Returns:
        Tensor of shape (28, 10) where LED_SUIT_TABLE[lead_domino_id, decl_id]
        gives the led suit (0-6 for pip suits, 7 for called suit).
    """
    led_suit_table = torch.zeros(28, N_DECLS, dtype=torch.int8)

    for lead_domino_id in range(28):
        high = DOMINO_HIGH[lead_domino_id]
        is_double = DOMINO_IS_DOUBLE[lead_domino_id]

        for decl_id in range(N_DECLS):
            # Determine if lead domino is in called suit
            if decl_id in PIP_TRUMP_IDS:
                in_called = (decl_id == high) or (decl_id == DOMINO_LOW[lead_domino_id])
            elif decl_id in (DOUBLES_TRUMP, DOUBLES_SUIT):
                in_called = is_double
            elif decl_id == NOTRUMP:
                in_called = False
            else:
                raise ValueError(f"Unknown decl_id: {decl_id}")

            # Determine led suit
            if decl_id == NOTRUMP:
                led_suit = high
            elif in_called:
                led_suit = 7  # Called suit
            else:
                led_suit = high

            led_suit_table[lead_domino_id, decl_id] = led_suit

    return led_suit_table


def _build_trick_rank_table() -> torch.Tensor:
    """Build lookup tensor for trick ranking.

    Returns:
        Tensor of shape (28, 8, 10) where TRICK_RANK_TABLE[domino_id, led_suit, decl_id]
        gives the 6-bit ranking key for trick resolution (higher wins).
    """
    trick_rank_table = torch.zeros(28, 8, N_DECLS, dtype=torch.int8)

    for domino_id in range(28):
        high = DOMINO_HIGH[domino_id]
        low = DOMINO_LOW[domino_id]
        is_double = DOMINO_IS_DOUBLE[domino_id]
        domino_sum = high + low

        for decl_id in range(N_DECLS):
            # Determine if domino is in called suit
            if decl_id in PIP_TRUMP_IDS:
                in_called = (decl_id == high) or (decl_id == low)
            elif decl_id in (DOUBLES_TRUMP, DOUBLES_SUIT):
                in_called = is_double
            elif decl_id == NOTRUMP:
                in_called = False
            else:
                raise ValueError(f"Unknown decl_id: {decl_id}")

            # Compute rank in pip suit
            rank_in_pip_suit = 14 if is_double else domino_sum

            # Compute rank in called suit (if applicable)
            if decl_id in PIP_TRUMP_IDS:
                rank_in_called_suit = rank_in_pip_suit
            elif decl_id == DOUBLES_TRUMP:
                rank_in_called_suit = high
            else:
                rank_in_called_suit = 0  # Not used for DOUBLES_SUIT/NOTRUMP

            # Compute rank in doubles suit
            rank_in_doubles_suit = high

            # For each possible led suit
            for led_suit in range(8):
                # Check if domino can follow
                if led_suit == 7:  # Called suit
                    can_follow = in_called
                else:  # Pip suit
                    has_pip = (led_suit == high) or (led_suit == low)
                    can_follow = has_pip and not in_called

                # Compute trick rank (6-bit key: tier << 4 | rank)
                if has_trump_power(decl_id) and in_called:
                    # Tier 2: Called suit with trump power
                    tier = 2
                    rank = rank_in_called_suit
                    trick_rank = (tier << 4) + rank
                elif can_follow:
                    # Tier 1: Follows led suit
                    tier = 1
                    rank = rank_in_doubles_suit if led_suit == 7 else rank_in_pip_suit
                    trick_rank = (tier << 4) + rank
                else:
                    # Tier 0: Does not follow
                    trick_rank = 0

                trick_rank_table[domino_id, led_suit, decl_id] = trick_rank

    return trick_rank_table


# Module-level constants
LED_SUIT_TABLE = _build_led_suit_table()
TRICK_RANK_TABLE = _build_trick_rank_table()

# Convert CPU constants to tensors
DOMINO_HIGH_T = torch.tensor(DOMINO_HIGH, dtype=torch.int8)
DOMINO_LOW_T = torch.tensor(DOMINO_LOW, dtype=torch.int8)
DOMINO_IS_DOUBLE_T = torch.tensor(DOMINO_IS_DOUBLE, dtype=torch.bool)
DOMINO_COUNT_POINTS_T = torch.tensor(DOMINO_COUNT_POINTS, dtype=torch.int8)


class GameStateTensor:
    """Vectorized game state for N concurrent games.

    State representation:
    - hands: (n_games, 4, 7) int8 - Domino IDs, -1 for played slots
    - played_mask: (n_games, 28) bool - True if domino has been played
    - history: (n_games, 28, 3) int8 - (player, domino_id, lead_domino_id), -1 for unused
    - trick_plays: (n_games, 4) int8 - Current trick dominoes, -1 for empty
    - leader: (n_games,) int8 - Current trick leader (0-3)
    - decl_ids: (n_games,) int8 - Declaration ID for each game
    - bidder: (n_games,) int8 - Player who won the bid (0-3), determines offense team
    - device: Device where tensors reside
    """

    def __init__(
        self,
        hands: torch.Tensor,
        played_mask: torch.Tensor,
        history: torch.Tensor,
        trick_plays: torch.Tensor,
        leader: torch.Tensor,
        decl_ids: torch.Tensor,
        device: str | torch.device,
        bidder: torch.Tensor | None = None,
    ):
        """Internal constructor. Use from_deals() to create instances."""
        self.hands = hands
        self.played_mask = played_mask
        self.history = history
        self.trick_plays = trick_plays
        self.leader = leader
        self.decl_ids = decl_ids
        self.device = device
        self.n_games = hands.shape[0]
        # Default bidder to 0 (P0 is bidder) if not provided
        if bidder is None:
            self.bidder = torch.zeros(self.n_games, dtype=torch.int8, device=device)
        else:
            self.bidder = bidder

    @classmethod
    def from_deals(
        cls,
        hands: list[list[list[int]]],
        decl_ids: list[int],
        device: str | torch.device = 'cuda',
        bidders: list[int] | None = None,
    ) -> GameStateTensor:
        """Initialize N games from dealt hands.

        Args:
            hands: List of N games, each with 4 hands of 7 domino IDs
            decl_ids: List of N declaration IDs
            device: Device to place tensors on
            bidders: Optional list of N bidder player IDs (0-3). Defaults to 0 for all games.

        Returns:
            GameStateTensor with N games initialized
        """
        n_games = len(hands)

        # Validate input
        if len(decl_ids) != n_games:
            raise ValueError(f"Expected {n_games} decl_ids, got {len(decl_ids)}")

        if bidders is not None and len(bidders) != n_games:
            raise ValueError(f"Expected {n_games} bidders, got {len(bidders)}")

        for game_idx, game_hands in enumerate(hands):
            if len(game_hands) != 4:
                raise ValueError(f"Game {game_idx}: expected 4 hands, got {len(game_hands)}")
            for player_idx, hand in enumerate(game_hands):
                if len(hand) != 7:
                    raise ValueError(
                        f"Game {game_idx}, player {player_idx}: expected 7 dominoes, got {len(hand)}"
                    )

        # Convert to tensors
        hands_t = torch.tensor(hands, dtype=torch.int8, device=device)  # (n_games, 4, 7)
        decl_ids_t = torch.tensor(decl_ids, dtype=torch.int8, device=device)  # (n_games,)

        # Initialize state tensors
        played_mask = torch.zeros(n_games, 28, dtype=torch.bool, device=device)
        history = torch.full((n_games, 28, 3), -1, dtype=torch.int8, device=device)
        trick_plays = torch.full((n_games, 4), -1, dtype=torch.int8, device=device)
        leader = torch.zeros(n_games, dtype=torch.int8, device=device)

        # Bidder tensor (default 0)
        if bidders is not None:
            bidder_t = torch.tensor(bidders, dtype=torch.int8, device=device)
        else:
            bidder_t = torch.zeros(n_games, dtype=torch.int8, device=device)

        return cls(
            hands=hands_t,
            played_mask=played_mask,
            history=history,
            trick_plays=trick_plays,
            leader=leader,
            decl_ids=decl_ids_t,
            device=device,
            bidder=bidder_t,
        )

    @property
    def current_player(self) -> torch.Tensor:
        """Returns (n_games,) tensor of current players (0-3)."""
        # Count non-(-1) entries in trick_plays
        trick_len = (self.trick_plays >= 0).sum(dim=1)  # (n_games,)
        return ((self.leader.long() + trick_len) % 4).to(torch.int8)

    def legal_actions(self) -> torch.Tensor:
        """Returns (n_games, 7) boolean mask of legal actions.

        For each game, returns which slot indices in current player's hand are legal.
        Slot i is legal if hands[game, current_player, i] >= 0 and follows suit rules.
        """
        current_player = self.current_player.long()  # (n_games,) as int64 for indexing

        # Get current player's hand for each game
        # hands: (n_games, 4, 7), current_player: (n_games,)
        # Need to index hands[i, current_player[i], :] for each i
        batch_indices = torch.arange(self.n_games, device=self.device)
        current_hands = self.hands[batch_indices, current_player]  # (n_games, 7)

        # Mask of valid hand slots (not yet played)
        valid_slots = current_hands >= 0  # (n_games, 7)

        # Check if leading (trick_len == 0)
        trick_len = (self.trick_plays >= 0).sum(dim=1)  # (n_games,)
        is_leading = trick_len == 0  # (n_games,)

        # For leading: all valid slots are legal
        # For following: must check follow-suit rules

        # Get lead domino for each game (trick_plays[:, 0])
        lead_domino = self.trick_plays[:, 0]  # (n_games,), -1 for leading games

        # Look up led suit for each game
        # LED_SUIT_TABLE: (28, 10)
        # Need to handle -1 lead_domino (when leading)
        # Use 0 as placeholder for leading games (will be masked out)
        lead_domino_safe = torch.where(is_leading, torch.zeros_like(lead_domino), lead_domino).long()
        led_suit_table = LED_SUIT_TABLE.to(self.device)
        led_suit = led_suit_table[lead_domino_safe, self.decl_ids.long()]  # (n_games,)

        # Check which hand slots can follow
        # CAN_FOLLOW: (28, 8, 10) - [domino_id, led_suit, decl_id]
        # current_hands: (n_games, 7) - domino IDs, -1 for played

        # For each game and slot, check if can follow
        # Use 0 as placeholder for -1 dominoes (will be masked by valid_slots)
        current_hands_safe = torch.where(current_hands >= 0, current_hands, torch.zeros_like(current_hands)).long()

        # Look up CAN_FOLLOW for each domino
        # CAN_FOLLOW: (28, 8, 10)
        # We need to index: CAN_FOLLOW[domino, led_suit, decl_id] for each (game, slot)
        # Flatten to make indexing easier, then reshape
        n_games = self.n_games
        flat_dominoes = current_hands_safe.reshape(-1)  # (n_games * 7,)
        flat_led_suits = led_suit.long().unsqueeze(1).expand(-1, 7).reshape(-1)  # (n_games * 7,)
        flat_decl_ids = self.decl_ids.long().unsqueeze(1).expand(-1, 7).reshape(-1)  # (n_games * 7,)

        can_follow_table = CAN_FOLLOW.to(self.device)
        flat_can_follow = can_follow_table[flat_dominoes, flat_led_suits, flat_decl_ids]  # (n_games * 7,)
        can_follow_mask = flat_can_follow.reshape(n_games, 7)  # (n_games, 7)

        # Check if any domino in hand can follow
        has_follower = (can_follow_mask & valid_slots).any(dim=1)  # (n_games,)

        # Determine legal slots
        # If leading: all valid slots
        # If following and has follower: only followers
        # If following and no follower: all valid slots (void)
        legal = torch.where(
            is_leading.unsqueeze(1),  # (n_games, 1)
            valid_slots,
            torch.where(
                has_follower.unsqueeze(1),  # (n_games, 1)
                can_follow_mask & valid_slots,
                valid_slots
            )
        )

        return legal

    def apply_actions(self, actions: torch.Tensor) -> GameStateTensor:
        """Vectorized state transition. Returns new GameStateTensor.

        Args:
            actions: (n_games,) tensor of slot indices to play (0-6)

        Returns:
            New GameStateTensor after applying actions immutably
        """
        if actions.shape[0] != self.n_games:
            raise ValueError(f"Expected {self.n_games} actions, got {actions.shape[0]}")

        current_player = self.current_player.long()  # (n_games,) as int64 for indexing
        batch_indices = torch.arange(self.n_games, device=self.device)

        # Get domino IDs being played
        # hands: (n_games, 4, 7), current_player: (n_games,), actions: (n_games,)
        domino_ids = self.hands[batch_indices, current_player, actions.long()]  # (n_games,)

        # Clone all state tensors for immutable update
        new_hands = self.hands.clone()
        new_played_mask = self.played_mask.clone()
        new_history = self.history.clone()
        new_trick_plays = self.trick_plays.clone()
        new_leader = self.leader.clone()

        # Remove dominoes from hands (set to -1)
        new_hands[batch_indices, current_player, actions.long()] = -1

        # Add to played mask
        new_played_mask[batch_indices, domino_ids.long()] = True

        # Determine lead domino for history
        trick_len = (self.trick_plays >= 0).sum(dim=1)  # (n_games,)
        is_leading = trick_len == 0

        # Lead domino is either current domino (if leading) or first in trick
        lead_domino_id = torch.where(
            is_leading,
            domino_ids,
            self.trick_plays[:, 0]
        )

        # Add to history at position = played_mask.sum()
        history_pos = self.played_mask.sum(dim=1)  # (n_games,)
        new_history[batch_indices, history_pos.long(), 0] = current_player.to(torch.int8)
        new_history[batch_indices, history_pos.long(), 1] = domino_ids
        new_history[batch_indices, history_pos.long(), 2] = lead_domino_id

        # Add to trick_plays
        new_trick_plays[batch_indices, trick_len.long()] = domino_ids

        # Check if trick is complete (new trick_len == 4)
        new_trick_len = trick_len + 1
        trick_complete = new_trick_len == 4

        # Resolve completed tricks
        if trick_complete.any():
            # Get games with completed tricks
            complete_indices = trick_complete.nonzero(as_tuple=True)[0]

            # Get trick dominoes for completed games
            trick_dominoes = new_trick_plays[complete_indices]  # (n_complete, 4)

            # Get led suit for each completed trick
            lead_dominoes = trick_dominoes[:, 0].long()
            decl_ids_complete = self.decl_ids[complete_indices].long()
            led_suit_table = LED_SUIT_TABLE.to(self.device)
            led_suits = led_suit_table[lead_dominoes, decl_ids_complete]  # (n_complete,)

            # Look up trick ranks for all 4 dominoes
            # TRICK_RANK_TABLE: (28, 8, 10)
            # trick_dominoes: (n_complete, 4)
            trick_rank_table = TRICK_RANK_TABLE.to(self.device)
            ranks = trick_rank_table[
                trick_dominoes.long(),  # (n_complete, 4)
                led_suits.long().unsqueeze(1).expand(-1, 4),  # (n_complete, 4)
                decl_ids_complete.unsqueeze(1).expand(-1, 4)  # (n_complete, 4)
            ]  # (n_complete, 4)

            # Find winner (argmax of ranks)
            winner_offset = ranks.argmax(dim=1)  # (n_complete,)

            # Update leader for completed tricks
            new_leader[complete_indices] = ((self.leader[complete_indices].long() + winner_offset) % 4).to(torch.int8)

            # Clear trick_plays for completed tricks
            new_trick_plays[complete_indices] = -1

        return GameStateTensor(
            hands=new_hands,
            played_mask=new_played_mask,
            history=new_history,
            trick_plays=new_trick_plays,
            leader=new_leader,
            decl_ids=self.decl_ids,  # Immutable, can reuse
            device=self.device,
            bidder=self.bidder,  # Immutable, can reuse
        )

    def active_games(self) -> torch.Tensor:
        """Returns (n_games,) boolean mask of incomplete games.

        A game is active if any player has dominoes remaining.
        """
        # Count dominoes in each hand
        hand_counts = (self.hands >= 0).sum(dim=2)  # (n_games, 4)
        # Game is active if any player has dominoes
        return hand_counts.sum(dim=1) > 0

    def hand_sizes(self) -> torch.Tensor:
        """Returns (n_games, 4) tensor of domino counts per player."""
        return (self.hands >= 0).sum(dim=2)
