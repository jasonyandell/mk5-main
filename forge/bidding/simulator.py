"""Batched game simulation for bidding evaluation.

Runs many games in parallel using vectorized PyTorch operations.
Each "step" is a single forward pass for all games.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from forge.oracle.declarations import N_DECLS, NOTRUMP
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_IS_DOUBLE,
    DOMINO_LOW,
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


def _build_trump_rank_table() -> dict[Tuple[int, int], int]:
    """Precompute trump rank for all (domino_id, decl_id) pairs."""
    table = {}
    for decl_id in range(N_DECLS):
        if decl_id == NOTRUMP:
            for dom_id in range(28):
                table[(dom_id, decl_id)] = 7
            continue

        # Find all trumps and rank them
        trumps = []
        for d in range(28):
            if is_in_called_suit(d, decl_id):
                tau = trick_rank(d, 7, decl_id)
                trumps.append((d, tau))
        trumps.sort(key=lambda x: -x[1])

        for dom_id in range(28):
            if not is_in_called_suit(dom_id, decl_id):
                table[(dom_id, decl_id)] = 7
            else:
                for rank, (d, _) in enumerate(trumps):
                    if d == dom_id:
                        table[(dom_id, decl_id)] = rank
                        break

    return table


TRUMP_RANK_TABLE = _build_trump_rank_table()


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
        """Precompute static features for all dominoes in all hands."""
        # Shape: (n_games, 4, 7, 5) for high, low, is_double, count_val, trump_rank
        self.domino_features = torch.zeros(
            self.n_games, 4, 7, 5, dtype=torch.long, device=self.device
        )

        # Build features using CPU then transfer (faster for this one-time setup)
        features_cpu = self.domino_features.cpu().numpy()
        hands_cpu = self.hands.cpu().numpy()

        for g in range(self.n_games):
            for p in range(4):
                for local_idx in range(7):
                    dom_id = int(hands_cpu[g, p, local_idx])
                    features_cpu[g, p, local_idx, 0] = DOMINO_HIGH[dom_id]
                    features_cpu[g, p, local_idx, 1] = DOMINO_LOW[dom_id]
                    features_cpu[g, p, local_idx, 2] = 1 if DOMINO_IS_DOUBLE[dom_id] else 0
                    features_cpu[g, p, local_idx, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[dom_id]]
                    features_cpu[g, p, local_idx, 4] = TRUMP_RANK_TABLE[(dom_id, self.decl_id)]

        self.domino_features = torch.from_numpy(features_cpu).to(self.device)

    def current_player(self) -> Tensor:
        """Get current player for each game. Shape: (n_games,)"""
        return (self.leader + self.trick_len) % 4

    def is_game_over(self) -> Tensor:
        """Check if games are over. Shape: (n_games,)"""
        return self.tricks_played >= 7

    def get_legal_mask(self) -> Tensor:
        """Get legal action mask for current player. Shape: (n_games, 7)"""
        current = self.current_player()  # (n_games,)

        # Get remaining dominoes for current player
        # Gather remaining masks for current player
        player_remaining = torch.zeros(self.n_games, 7, dtype=torch.bool, device=self.device)
        for g in range(self.n_games):
            player_remaining[g] = self.remaining[g, current[g]]

        # If leading (trick_len == 0), all remaining are legal
        is_leading = self.trick_len == 0

        if is_leading.all():
            return player_remaining

        # For following, need to check suit rules
        legal = player_remaining.clone()

        # Get lead domino's suit for each game
        for g in range(self.n_games):
            if is_leading[g]:
                continue

            leader_idx = int(self.leader[g])
            lead_local = int(self.trick_plays[g, 0])
            if lead_local < 0:
                continue

            lead_dom_id = int(self.hands[g, leader_idx, lead_local])
            led_suit = self._led_suit(lead_dom_id)

            # Check which of player's remaining dominoes can follow
            p = int(current[g])
            can_follow_any = False
            follow_mask = torch.zeros(7, dtype=torch.bool, device=self.device)

            for local_idx in range(7):
                if not player_remaining[g, local_idx]:
                    continue
                dom_id = int(self.hands[g, p, local_idx])
                if self._can_follow(dom_id, led_suit):
                    follow_mask[local_idx] = True
                    can_follow_any = True

            # If can follow suit, must follow suit
            if can_follow_any:
                legal[g] = follow_mask
            # Otherwise all remaining are legal (can't follow)

        return legal

    def _led_suit(self, lead_dom_id: int) -> int:
        """Get led suit (0-6 for pip suit, 7 for trump suit)."""
        if self.decl_id == NOTRUMP:
            return DOMINO_HIGH[lead_dom_id]
        if is_in_called_suit(lead_dom_id, self.decl_id):
            return 7
        return DOMINO_HIGH[lead_dom_id]

    def _can_follow(self, dom_id: int, led_suit: int) -> bool:
        """Check if domino can follow the led suit."""
        if led_suit == 7:
            return is_in_called_suit(dom_id, self.decl_id)
        return (
            (DOMINO_HIGH[dom_id] == led_suit or DOMINO_LOW[dom_id] == led_suit)
            and not is_in_called_suit(dom_id, self.decl_id)
        )

    def step(self, actions: Tensor) -> None:
        """Execute actions and update state.

        Args:
            actions: (n_games,) local indices of dominoes to play
        """
        current = self.current_player()
        trick_pos = self.trick_len.clone()

        # Record play in trick
        for g in range(self.n_games):
            if self.is_game_over()[g]:
                continue
            p = int(current[g])
            pos = int(trick_pos[g])
            local_idx = int(actions[g])

            # Mark domino as played
            self.remaining[g, p, local_idx] = False

            # Record in trick
            self.trick_plays[g, pos] = local_idx

        self.trick_len += 1

        # Check for trick completion
        trick_complete = self.trick_len >= 4
        if trick_complete.any():
            self._resolve_tricks(trick_complete)

    def _resolve_tricks(self, complete_mask: Tensor) -> None:
        """Resolve completed tricks and update state."""
        for g in range(self.n_games):
            if not complete_mask[g]:
                continue

            leader = int(self.leader[g])

            # Get the 4 played dominoes
            dom_ids = []
            for pos in range(4):
                p = (leader + pos) % 4
                local_idx = int(self.trick_plays[g, pos])
                dom_ids.append(int(self.hands[g, p, local_idx]))

            # Find winner using trick_rank
            lead_dom = dom_ids[0]
            led_suit = self._led_suit(lead_dom)

            best_pos = 0
            best_rank = trick_rank(dom_ids[0], led_suit, self.decl_id)
            for pos in range(1, 4):
                r = trick_rank(dom_ids[pos], led_suit, self.decl_id)
                if r > best_rank:
                    best_pos = pos
                    best_rank = r

            winner = (leader + best_pos) % 4
            winner_team = winner % 2

            # Score trick (1 base + count values)
            points = 1
            for dom_id in dom_ids:
                points += DOMINO_COUNT_POINTS[dom_id]

            self.team_points[g, winner_team] += points

            # Set up next trick
            self.leader[g] = winner
            self.trick_plays[g] = -1
            self.trick_len[g] = 0
            self.tricks_played[g] += 1

    def build_tokens(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Build token tensors for model input.

        Returns:
            tokens: (n_games, 32, 12) int64
            mask: (n_games, 32) float32
            current_player: (n_games,) int64
        """
        tokens = torch.zeros(self.n_games, 32, 12, dtype=torch.long, device=self.device)
        mask = torch.zeros(self.n_games, 32, dtype=torch.float32, device=self.device)

        current = self.current_player()

        # Context token (position 0)
        # Normalized leader = (leader - current + 4) % 4
        normalized_leader = (self.leader - current + 4) % 4
        tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
        tokens[:, 0, 10] = self.decl_id
        tokens[:, 0, 11] = normalized_leader
        mask[:, 0] = 1

        # Hand tokens (positions 1-28)
        for p in range(4):
            # normalized_p = (p - current + 4) % 4, shape (n_games,)
            normalized_p = (p - current + 4) % 4

            for local_idx in range(7):
                token_idx = 1 + p * 7 + local_idx

                # Static domino features
                tokens[:, token_idx, 0] = self.domino_features[:, p, local_idx, 0]  # high
                tokens[:, token_idx, 1] = self.domino_features[:, p, local_idx, 1]  # low
                tokens[:, token_idx, 2] = self.domino_features[:, p, local_idx, 2]  # is_double
                tokens[:, token_idx, 3] = self.domino_features[:, p, local_idx, 3]  # count_val
                tokens[:, token_idx, 4] = self.domino_features[:, p, local_idx, 4]  # trump_rank

                # Dynamic features - normalized_p is a tensor of shape (n_games,)
                tokens[:, token_idx, 5] = normalized_p
                tokens[:, token_idx, 6] = (normalized_p == 0).long()
                tokens[:, token_idx, 7] = (normalized_p == 2).long()
                tokens[:, token_idx, 8] = self.remaining[:, p, local_idx].long()
                tokens[:, token_idx, 9] = TOKEN_TYPE_PLAYER0 + p
                tokens[:, token_idx, 10] = self.decl_id
                tokens[:, token_idx, 11] = normalized_leader

                mask[:, token_idx] = 1

        # Trick tokens (positions 29-31)
        for trick_pos in range(3):
            token_idx = 29 + trick_pos

            for g in range(self.n_games):
                if self.trick_len[g] <= trick_pos:
                    continue

                local_idx = int(self.trick_plays[g, trick_pos])
                if local_idx < 0:
                    continue

                play_player = (int(self.leader[g]) + trick_pos) % 4
                dom_id = int(self.hands[g, play_player, local_idx])
                cur = int(current[g])
                normalized_pp = (play_player - cur + 4) % 4

                tokens[g, token_idx, 0] = DOMINO_HIGH[dom_id]
                tokens[g, token_idx, 1] = DOMINO_LOW[dom_id]
                tokens[g, token_idx, 2] = 1 if DOMINO_IS_DOUBLE[dom_id] else 0
                tokens[g, token_idx, 3] = COUNT_VALUE_MAP[DOMINO_COUNT_POINTS[dom_id]]
                tokens[g, token_idx, 4] = TRUMP_RANK_TABLE[(dom_id, self.decl_id)]
                tokens[g, token_idx, 5] = normalized_pp
                tokens[g, token_idx, 6] = 1 if normalized_pp == 0 else 0
                tokens[g, token_idx, 7] = 1 if normalized_pp == 2 else 0
                tokens[g, token_idx, 8] = 0  # Not remaining (played)
                tokens[g, token_idx, 9] = TOKEN_TYPE_TRICK_P0 + trick_pos
                tokens[g, token_idx, 10] = self.decl_id
                tokens[g, token_idx, 11] = int(normalized_leader[g])

                mask[g, token_idx] = 1

        return tokens, mask, current


def deal_random_hands(
    bidder_hand: List[int],
    n_games: int,
    device: torch.device,
    rng: torch.Generator | None = None,
) -> Tensor:
    """Deal random opponent hands given the bidder's hand.

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
    remaining = list(all_doms - bidder_set)

    if len(remaining) != 21:
        raise ValueError(f"Expected 21 remaining dominoes, got {len(remaining)}")

    hands = torch.zeros(n_games, 4, 7, dtype=torch.long, device=device)

    # Player 0 always gets the bidder's hand
    hands[:, 0, :] = torch.tensor(sorted(bidder_hand), dtype=torch.long, device=device)

    # Randomly deal remaining 21 dominoes to players 1, 2, 3
    for g in range(n_games):
        # Shuffle remaining
        perm = torch.randperm(21, generator=rng, device="cpu")
        shuffled = [remaining[i] for i in perm.tolist()]

        for p in range(1, 4):
            start = (p - 1) * 7
            player_doms = sorted(shuffled[start : start + 7])
            hands[g, p, :] = torch.tensor(player_doms, dtype=torch.long, device=device)

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

    # Run games until completion (max 28 steps = 7 tricks × 4 plays)
    for _ in range(28):
        if state.is_game_over().all():
            break

        # Build tokens
        tokens, mask, current_player = state.build_tokens()

        # Get legal mask
        legal_mask = state.get_legal_mask()

        # Get actions
        if greedy:
            actions = model.greedy_actions(tokens, mask, current_player, legal_mask)
        else:
            actions = model.sample_actions(tokens, mask, current_player, legal_mask)

        # Execute actions
        state.step(actions)

    # Return team 0's points
    return state.team_points[:, 0]
