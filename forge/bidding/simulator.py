"""Batched game simulation for bidding evaluation.

Runs many games in parallel using vectorized PyTorch operations.
Each "step" is a single forward pass for all games.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from forge.oracle.declarations import N_DECLS, NOTRUMP, PIP_TRUMP_IDS, DOUBLES_TRUMP, DOUBLES_SUIT
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
    DOMINO_SUM,
    is_in_called_suit,
    trick_rank,
)

from .inference import PolicyModel


# Token type constants (from tokenize.py)
TOKEN_TYPE_CONTEXT = 0
TOKEN_TYPE_PLAYER0 = 1
TOKEN_TYPE_TRICK_P0 = 5

# Count value mapping
COUNT_VALUE_MAP = {0: 0, 5: 1, 10: 2}


# =============================================================================
# Tensor Lookup Tables (computed once at module load)
# =============================================================================

def _build_tensor_tables() -> dict[str, Tensor]:
    """Build all lookup tables as tensors for vectorized operations."""
    tables = {}

    # Basic domino properties: shape (28,)
    tables["DOMINO_HIGH"] = torch.tensor(DOMINO_HIGH, dtype=torch.long)
    tables["DOMINO_LOW"] = torch.tensor(DOMINO_LOW, dtype=torch.long)
    tables["DOMINO_IS_DOUBLE"] = torch.tensor(DOMINO_IS_DOUBLE, dtype=torch.bool)
    tables["DOMINO_SUM"] = torch.tensor(DOMINO_SUM, dtype=torch.long)

    # Count points with value mapping: shape (28,)
    count_vals = torch.zeros(28, dtype=torch.long)
    for d in range(28):
        count_vals[d] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[d]]
    tables["COUNT_VALUE"] = count_vals

    # Raw count points for scoring: shape (28,)
    tables["COUNT_POINTS"] = torch.tensor(DOMINO_COUNT_POINTS, dtype=torch.long)

    # is_in_called_suit: shape (28, N_DECLS)
    in_called = torch.zeros(28, N_DECLS, dtype=torch.bool)
    for d in range(28):
        for decl in range(N_DECLS):
            in_called[d, decl] = is_in_called_suit(d, decl)
    tables["IS_IN_CALLED_SUIT"] = in_called

    # led_suit for lead domino: shape (28, N_DECLS)
    # Returns 0-6 for pip suit, 7 for called suit (trump)
    led_suit = torch.zeros(28, N_DECLS, dtype=torch.long)
    for d in range(28):
        for decl in range(N_DECLS):
            if decl == NOTRUMP:
                led_suit[d, decl] = DOMINO_HIGH[d]
            elif is_in_called_suit(d, decl):
                led_suit[d, decl] = 7  # Called suit
            else:
                led_suit[d, decl] = DOMINO_HIGH[d]
    tables["LED_SUIT"] = led_suit

    # can_follow: shape (28, 8, N_DECLS) - domino x led_suit x decl
    # led_suit 0-6 = pip suits, 7 = called suit
    can_follow = torch.zeros(28, 8, N_DECLS, dtype=torch.bool)
    for d in range(28):
        high, low = DOMINO_HIGH[d], DOMINO_LOW[d]
        is_double = DOMINO_IS_DOUBLE[d]
        for led in range(8):
            for decl in range(N_DECLS):
                if led == 7:
                    # Must follow called suit
                    can_follow[d, led, decl] = is_in_called_suit(d, decl)
                else:
                    # Must follow pip suit (and not be trump)
                    has_pip = (high == led or low == led)
                    is_trump = is_in_called_suit(d, decl)
                    can_follow[d, led, decl] = has_pip and not is_trump
    tables["CAN_FOLLOW"] = can_follow

    # trick_rank: shape (28, 8, N_DECLS) - domino x led_suit x decl
    # Higher value wins the trick
    trick_rank_t = torch.zeros(28, 8, N_DECLS, dtype=torch.long)
    for d in range(28):
        for led in range(8):
            for decl in range(N_DECLS):
                trick_rank_t[d, led, decl] = trick_rank(d, led, decl)
    tables["TRICK_RANK"] = trick_rank_t

    # trump_rank for tokenization: shape (28, N_DECLS)
    # Rank 0-7 within trumps (0=highest), 7 for non-trumps
    trump_rank = torch.full((28, N_DECLS), 7, dtype=torch.long)
    for decl in range(N_DECLS):
        if decl == NOTRUMP:
            continue  # All 7s
        # Find all trumps and rank them
        trumps = []
        for d in range(28):
            if is_in_called_suit(d, decl):
                tau = trick_rank(d, 7, decl)
                trumps.append((d, tau))
        trumps.sort(key=lambda x: -x[1])
        for rank, (d, _) in enumerate(trumps):
            trump_rank[d, decl] = rank
    tables["TRUMP_RANK"] = trump_rank

    return tables


# Build tables once at module load
_TABLES = _build_tensor_tables()

# Export individual tables for convenience
DOMINO_HIGH_T = _TABLES["DOMINO_HIGH"]
DOMINO_LOW_T = _TABLES["DOMINO_LOW"]
DOMINO_IS_DOUBLE_T = _TABLES["DOMINO_IS_DOUBLE"]
DOMINO_SUM_T = _TABLES["DOMINO_SUM"]
COUNT_VALUE_T = _TABLES["COUNT_VALUE"]
COUNT_POINTS_T = _TABLES["COUNT_POINTS"]
IS_IN_CALLED_SUIT_T = _TABLES["IS_IN_CALLED_SUIT"]
LED_SUIT_T = _TABLES["LED_SUIT"]
CAN_FOLLOW_T = _TABLES["CAN_FOLLOW"]
TRICK_RANK_T = _TABLES["TRICK_RANK"]
TRUMP_RANK_T = _TABLES["TRUMP_RANK"]


def _get_table(name: str, device: torch.device) -> Tensor:
    """Get a table tensor on the specified device (cached)."""
    key = (name, device)
    if not hasattr(_get_table, "_cache"):
        _get_table._cache = {}
    if key not in _get_table._cache:
        _get_table._cache[key] = _TABLES[name].to(device)
    return _get_table._cache[key]


@dataclass
class SimulationConfig:
    """Configuration for game simulation."""

    device: torch.device
    n_games: int
    decl_id: int


class BatchedGameState:
    """Vectorized game state for parallel simulation.

    Tracks:
    - hands: (n_games, 4, 7) global domino IDs for each player
    - remaining: (n_games, 4, 7) bool mask of remaining dominoes
    - team_points: (n_games, 2) points accumulated by each team
    - leader: (n_games,) current trick leader
    - trick_plays: (n_games, 4) local indices played in current trick (-1 = not played)
    - trick_len: (n_games,) number of plays in current trick
    """

    def __init__(
        self,
        hands: Tensor,  # (n_games, 4, 7) global domino IDs
        decl_id: int,
        device: torch.device,
    ):
        """Initialize game state.

        Args:
            hands: (n_games, 4, 7) global domino IDs
            decl_id: Declaration/trump type
            device: Torch device
        """
        self.n_games = hands.shape[0]
        self.device = device
        self.decl_id = decl_id

        self.hands = hands.to(device)  # (n_games, 4, 7)
        self.remaining = torch.ones(self.n_games, 4, 7, dtype=torch.bool, device=device)
        self.team_points = torch.zeros(self.n_games, 2, dtype=torch.int32, device=device)

        # Trick state
        self.leader = torch.zeros(self.n_games, dtype=torch.long, device=device)
        self.trick_plays = torch.full((self.n_games, 4), -1, dtype=torch.long, device=device)
        self.trick_len = torch.zeros(self.n_games, dtype=torch.long, device=device)
        self.tricks_played = torch.zeros(self.n_games, dtype=torch.long, device=device)

        # Precompute static domino features for tokenization
        self._precompute_domino_features()

    def _precompute_domino_features(self) -> None:
        """Precompute static features for all dominoes in all hands (vectorized)."""
        # Get lookup tables on device
        high_t = _get_table("DOMINO_HIGH", self.device)
        low_t = _get_table("DOMINO_LOW", self.device)
        is_double_t = _get_table("DOMINO_IS_DOUBLE", self.device)
        count_val_t = _get_table("COUNT_VALUE", self.device)
        trump_rank_t = _get_table("TRUMP_RANK", self.device)

        # Use hands as indices into lookup tables
        # hands: (n_games, 4, 7) -> features: (n_games, 4, 7, 5)
        dom_ids = self.hands  # (n_games, 4, 7)

        self.domino_features = torch.stack([
            high_t[dom_ids],                      # high pip
            low_t[dom_ids],                       # low pip
            is_double_t[dom_ids].long(),          # is_double
            count_val_t[dom_ids],                 # count value (0/1/2)
            trump_rank_t[dom_ids, self.decl_id],  # trump rank (0-7)
        ], dim=-1)  # (n_games, 4, 7, 5)

    def current_player(self) -> Tensor:
        """Get current player for each game. Shape: (n_games,)"""
        return (self.leader + self.trick_len) % 4

    def is_game_over(self) -> Tensor:
        """Check if games are over. Shape: (n_games,)"""
        return self.tricks_played >= 7

    def get_legal_mask(self) -> Tensor:
        """Get legal action mask for current player (vectorized). Shape: (n_games, 7)"""
        current = self.current_player()  # (n_games,)
        game_idx = torch.arange(self.n_games, device=self.device)

        # Get remaining dominoes for current player using advanced indexing
        # remaining: (n_games, 4, 7) -> player_remaining: (n_games, 7)
        player_remaining = self.remaining[game_idx, current]  # (n_games, 7)

        # If leading (trick_len == 0), all remaining are legal
        is_leading = self.trick_len == 0  # (n_games,)

        # Early exit if all games are leading
        if is_leading.all():
            return player_remaining

        # Get lookup tables
        led_suit_t = _get_table("LED_SUIT", self.device)
        can_follow_t = _get_table("CAN_FOLLOW", self.device)

        # Get lead domino for each game
        lead_local = self.trick_plays[:, 0]  # (n_games,)
        # Clamp to valid range for indexing (will be masked out for leading games)
        lead_local_safe = lead_local.clamp(min=0)
        lead_dom_id = self.hands[game_idx, self.leader, lead_local_safe]  # (n_games,)

        # Get led suit using lookup table
        led_suit = led_suit_t[lead_dom_id, self.decl_id]  # (n_games,)

        # Get current player's hand
        player_hands = self.hands[game_idx, current]  # (n_games, 7)

        # Check which dominoes can follow using lookup table
        # can_follow_t: (28, 8, N_DECLS)
        # We need can_follow_t[player_hands, led_suit, decl_id] -> (n_games, 7)
        can_follow = can_follow_t[
            player_hands,                          # (n_games, 7)
            led_suit.unsqueeze(1).expand(-1, 7),   # (n_games, 7)
            self.decl_id                           # scalar
        ]  # (n_games, 7)

        # Dominoes that can follow AND are remaining
        follow_mask = can_follow & player_remaining  # (n_games, 7)

        # Check if player can follow any suit
        can_follow_any = follow_mask.any(dim=1, keepdim=True)  # (n_games, 1)

        # If leading OR can't follow any, use player_remaining; else use follow_mask
        # Use where: legal = is_leading ? player_remaining : (can_follow_any ? follow_mask : player_remaining)
        legal = torch.where(
            is_leading.unsqueeze(1) | ~can_follow_any,
            player_remaining,
            follow_mask
        )

        return legal

    def step(self, actions: Tensor, active_mask: Tensor | None = None) -> None:
        """Execute actions and update state (vectorized).

        Args:
            actions: (n_games,) local indices of dominoes to play
            active_mask: (n_games,) optional mask for active games (defaults to ~is_game_over)
        """
        if active_mask is None:
            active_mask = ~self.is_game_over()

        current = self.current_player()  # (n_games,)
        trick_pos = self.trick_len.clone()  # (n_games,)
        game_idx = torch.arange(self.n_games, device=self.device)

        # Mark dominoes as played using advanced indexing (only for active games)
        # We need to set remaining[g, current[g], actions[g]] = False for active g
        # Use scatter_ with a temporary tensor
        active_games = game_idx[active_mask]
        if active_games.numel() > 0:
            self.remaining[active_games, current[active_games], actions[active_games]] = False

            # Record in trick using scatter
            # trick_plays[g, trick_pos[g]] = actions[g] for active g
            self.trick_plays[active_games] = self.trick_plays[active_games].scatter(
                1,
                trick_pos[active_games].unsqueeze(1),
                actions[active_games].unsqueeze(1)
            )

        # Increment trick_len only for active games
        self.trick_len += active_mask.long()

        # Check for trick completion
        trick_complete = (self.trick_len >= 4) & active_mask
        if trick_complete.any():
            self._resolve_tricks(trick_complete)

    def _resolve_tricks(self, complete_mask: Tensor) -> None:
        """Resolve completed tricks and update state (vectorized)."""
        if not complete_mask.any():
            return

        game_idx = torch.arange(self.n_games, device=self.device)

        # Get lookup tables
        led_suit_t = _get_table("LED_SUIT", self.device)
        trick_rank_t = _get_table("TRICK_RANK", self.device)
        count_points_t = _get_table("COUNT_POINTS", self.device)

        # Compute players for each position: (leader + pos) % 4
        # positions: (4,), leader: (n_games,) -> players: (n_games, 4)
        positions = torch.arange(4, device=self.device)
        players = (self.leader.unsqueeze(1) + positions) % 4  # (n_games, 4)

        # Get domino IDs for all 4 plays
        # hands: (n_games, 4, 7), trick_plays: (n_games, 4)
        # We need: hands[g, players[g, p], trick_plays[g, p]]
        dom_ids = self.hands[
            game_idx.unsqueeze(1).expand(-1, 4),  # (n_games, 4)
            players,                               # (n_games, 4)
            self.trick_plays.clamp(min=0),        # (n_games, 4) - clamp for safety
        ]  # (n_games, 4)

        # Get led suit from first domino
        lead_dom = dom_ids[:, 0]  # (n_games,)
        led_suit = led_suit_t[lead_dom, self.decl_id]  # (n_games,)

        # Get trick ranks for all 4 dominoes
        # trick_rank_t: (28, 8, N_DECLS)
        ranks = trick_rank_t[
            dom_ids,                               # (n_games, 4)
            led_suit.unsqueeze(1).expand(-1, 4),   # (n_games, 4)
            self.decl_id                           # scalar
        ]  # (n_games, 4)

        # Find winner (argmax of ranks)
        best_pos = ranks.argmax(dim=1)  # (n_games,)
        winner = (self.leader + best_pos) % 4  # (n_games,)
        winner_team = winner % 2  # (n_games,)

        # Score tricks: 1 base + count points
        # count_points_t: (28,)
        points = 1 + count_points_t[dom_ids].sum(dim=1)  # (n_games,)

        # Update team points for completed games
        # Use scatter_add to accumulate points
        completed_games = game_idx[complete_mask]
        if completed_games.numel() > 0:
            # Add points to the winning team
            self.team_points.scatter_add_(
                1,
                winner_team[complete_mask].unsqueeze(1),
                points[complete_mask].unsqueeze(1).to(self.team_points.dtype)
            )

            # Set up next trick for completed games
            self.leader[complete_mask] = winner[complete_mask]
            self.trick_plays[complete_mask] = -1
            self.trick_len[complete_mask] = 0
            self.tricks_played[complete_mask] += 1

    def build_tokens(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Build token tensors for model input (vectorized).

        Returns:
            tokens: (n_games, 32, 12) int64
            mask: (n_games, 32) float32
            current_player: (n_games,) int64
        """
        tokens = torch.zeros(self.n_games, 32, 12, dtype=torch.long, device=self.device)
        mask = torch.zeros(self.n_games, 32, dtype=torch.float32, device=self.device)

        current = self.current_player()  # (n_games,)

        # Context token (position 0)
        normalized_leader = (self.leader - current + 4) % 4  # (n_games,)
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = self.decl_id
        tokens[:, 0, 11] = normalized_leader
        mask[:, 0] = 1

        # Hand tokens (positions 1-28) - vectorized over players and local indices
        # Build normalized_p for all players: (n_games, 4)
        players = torch.arange(4, device=self.device)  # (4,)
        normalized_p = (players - current.unsqueeze(1) + 4) % 4  # (n_games, 4)

        for p in range(4):
            for local_idx in range(7):
                token_idx = 1 + p * 7 + local_idx

                # Static domino features from precomputed
                tokens[:, token_idx, 0] = self.domino_features[:, p, local_idx, 0]  # high
                tokens[:, token_idx, 1] = self.domino_features[:, p, local_idx, 1]  # low
                tokens[:, token_idx, 2] = self.domino_features[:, p, local_idx, 2]  # is_double
                tokens[:, token_idx, 3] = self.domino_features[:, p, local_idx, 3]  # count_val
                tokens[:, token_idx, 4] = self.domino_features[:, p, local_idx, 4]  # trump_rank

                # Dynamic features
                np = normalized_p[:, p]  # (n_games,)
                tokens[:, token_idx, 5] = np
                tokens[:, token_idx, 6] = (np == 0).long()
                tokens[:, token_idx, 7] = (np == 2).long()
                tokens[:, token_idx, 8] = self.remaining[:, p, local_idx].long()
                tokens[:, token_idx, 9] = TOKEN_TYPE_PLAYER0 + p
                tokens[:, token_idx, 10] = self.decl_id
                tokens[:, token_idx, 11] = normalized_leader

                mask[:, token_idx] = 1

        # Trick tokens (positions 29-31) - vectorized
        # Get lookup tables
        high_t = _get_table("DOMINO_HIGH", self.device)
        low_t = _get_table("DOMINO_LOW", self.device)
        is_double_t = _get_table("DOMINO_IS_DOUBLE", self.device)
        count_val_t = _get_table("COUNT_VALUE", self.device)
        trump_rank_t = _get_table("TRUMP_RANK", self.device)

        game_idx = torch.arange(self.n_games, device=self.device)

        for trick_pos in range(3):
            token_idx = 29 + trick_pos

            # Mask for games that have this trick position played
            has_play = self.trick_len > trick_pos  # (n_games,)

            if not has_play.any():
                continue

            # Get local index and player for this position
            local_idx = self.trick_plays[:, trick_pos].clamp(min=0)  # (n_games,)
            play_player = (self.leader + trick_pos) % 4  # (n_games,)

            # Get domino IDs using advanced indexing
            dom_ids = self.hands[game_idx, play_player, local_idx]  # (n_games,)

            # Normalized player position
            normalized_pp = (play_player - current + 4) % 4  # (n_games,)

            # Fill all games (will be masked by 'mask' tensor for invalid ones)
            tokens[:, token_idx, 0] = high_t[dom_ids]
            tokens[:, token_idx, 1] = low_t[dom_ids]
            tokens[:, token_idx, 2] = is_double_t[dom_ids].long()
            tokens[:, token_idx, 3] = count_val_t[dom_ids]
            tokens[:, token_idx, 4] = trump_rank_t[dom_ids, self.decl_id]
            tokens[:, token_idx, 5] = normalized_pp
            tokens[:, token_idx, 6] = (normalized_pp == 0).long()
            tokens[:, token_idx, 7] = (normalized_pp == 2).long()
            tokens[:, token_idx, 8] = 0  # Not remaining (played)
            tokens[:, token_idx, 9] = TOKEN_TYPE_TRICK_P0 + trick_pos
            tokens[:, token_idx, 10] = self.decl_id
            tokens[:, token_idx, 11] = normalized_leader

            # Only set mask for games that have this play
            mask[:, token_idx] = has_play.float()

        return tokens, mask, current


def deal_random_hands(
    bidder_hand: List[int],
    n_games: int,
    device: torch.device,
    rng: torch.Generator | None = None,
) -> Tensor:
    """Deal random opponent hands given the bidder's hand (vectorized).

    Args:
        bidder_hand: 7 global domino IDs for player 0 (the bidder)
        n_games: Number of games to simulate
        device: Torch device
        rng: Optional random generator for reproducibility

    Returns:
        hands: (n_games, 4, 7) global domino IDs
    """
    # Remaining dominoes (all except bidder's)
    all_doms = set(range(28))
    bidder_set = set(bidder_hand)
    remaining = sorted(all_doms - bidder_set)

    if len(remaining) != 21:
        raise ValueError(f"Expected 21 remaining dominoes, got {len(remaining)}")

    # Convert to tensor
    remaining_t = torch.tensor(remaining, dtype=torch.long)  # (21,)

    # Generate all random permutations at once using argsort trick
    # Random values: (n_games, 21), then argsort to get permutations
    random_vals = torch.rand(n_games, 21, generator=rng)
    perms = random_vals.argsort(dim=1)  # (n_games, 21)

    # Apply permutations to remaining dominoes
    shuffled = remaining_t[perms]  # (n_games, 21)

    # Sort each player's 7 dominoes
    # Player 1: shuffled[:, 0:7], Player 2: shuffled[:, 7:14], Player 3: shuffled[:, 14:21]
    p1_sorted, _ = shuffled[:, 0:7].sort(dim=1)
    p2_sorted, _ = shuffled[:, 7:14].sort(dim=1)
    p3_sorted, _ = shuffled[:, 14:21].sort(dim=1)

    # Build hands tensor
    hands = torch.zeros(n_games, 4, 7, dtype=torch.long, device=device)
    hands[:, 0, :] = torch.tensor(sorted(bidder_hand), dtype=torch.long, device=device)
    hands[:, 1, :] = p1_sorted.to(device)
    hands[:, 2, :] = p2_sorted.to(device)
    hands[:, 3, :] = p3_sorted.to(device)

    return hands


def simulate_games(
    model: PolicyModel,
    bidder_hand: List[int],
    decl_id: int,
    n_games: int,
    seed: int | None = None,
    greedy: bool = False,
) -> Tensor:
    """Simulate multiple games and return team 0's points.

    Args:
        model: The policy model for action selection
        bidder_hand: 7 global domino IDs for player 0 (the bidder)
        decl_id: Declaration/trump type (0-9)
        n_games: Number of games to simulate
        seed: Random seed for reproducibility
        greedy: If True, use greedy actions; else sample from policy

    Returns:
        points: (n_games,) team 0's final points
    """
    device = model.device

    # Set up RNG
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    # Deal hands
    hands = deal_random_hands(bidder_hand, n_games, device, rng)

    # Initialize game state
    state = BatchedGameState(hands, decl_id, device)

    # Warmup model with this batch size
    model.warmup(n_games)

    # Run all 28 steps (7 tricks × 4 plays) without GPU sync
    # Finished games are masked out in step()
    for _ in range(28):
        # Compute active mask without syncing (stays on GPU)
        active_mask = ~state.is_game_over()

        # Build tokens
        tokens, mask, current_player = state.build_tokens()

        # Get legal mask
        legal_mask = state.get_legal_mask()

        # Get actions (will be ignored for finished games via active_mask)
        if greedy:
            actions = model.greedy_actions(tokens, mask, current_player, legal_mask)
        else:
            actions = model.sample_actions(tokens, mask, current_player, legal_mask)

        # Execute actions (uses active_mask internally)
        state.step(actions, active_mask)

    # Return team 0's points
    return state.team_points[:, 0]
