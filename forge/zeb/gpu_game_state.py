"""GPU-native GameState for batched MCTS.

Stores N parallel game states on GPU as tensors. All operations are vectorized
with no Python loops in the hot path.

Key design decisions:
- Slot indices (0-6) for actions, not domino IDs
- All tensors on same device (GPU)
- Batch dimension first: (N, ...)
- -1 represents "empty" or "played" slots
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from forge.zeb.cuda_only import require_cuda

if TYPE_CHECKING:
    from forge.eq.game import GameState


# Declaration constants (matching forge/oracle/declarations.py)
PIP_TRUMP_IDS = tuple(range(7))  # 0-6 are pip trumps
DOUBLES_TRUMP = 7
DOUBLES_SUIT = 8
NOTRUMP = 9


def _build_domino_tables(device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build lookup tables for domino properties.

    Returns:
        domino_high: (28,) high pip for each domino
        domino_low: (28,) low pip for each domino
        domino_is_double: (28,) bool, true if double
        domino_sum: (28,) sum of pips
    """
    # Generate domino IDs in canonical order: (0,0), (1,0), (1,1), (2,0), ...
    high_list = []
    low_list = []
    for high in range(7):
        for low in range(high + 1):
            high_list.append(high)
            low_list.append(low)

    domino_high = torch.tensor(high_list, dtype=torch.int32, device=device)
    domino_low = torch.tensor(low_list, dtype=torch.int32, device=device)
    domino_is_double = domino_high == domino_low
    domino_sum = domino_high + domino_low

    return domino_high, domino_low, domino_is_double, domino_sum


def _build_count_points_table(device: torch.device) -> Tensor:
    """Build lookup table for domino point values.

    Returns:
        (28,) int32 tensor of point values (0, 5, or 10)
    """
    points = []
    for high in range(7):
        for low in range(high + 1):
            if (high, low) in ((5, 5), (6, 4)):
                points.append(10)
            elif (high, low) in ((5, 0), (4, 1), (3, 2)):
                points.append(5)
            else:
                points.append(0)
    return torch.tensor(points, dtype=torch.int32, device=device)


@dataclass
class DominoTables:
    """Cached lookup tables for domino properties."""

    high: Tensor      # (28,) int32 - high pip
    low: Tensor       # (28,) int32 - low pip
    is_double: Tensor # (28,) bool
    sum: Tensor       # (28,) int32
    points: Tensor    # (28,) int32 - count points (0, 5, 10)

    @classmethod
    def create(cls, device: torch.device) -> DominoTables:
        high, low, is_double, domino_sum = _build_domino_tables(device)
        points = _build_count_points_table(device)
        return cls(
            high=high,
            low=low,
            is_double=is_double,
            sum=domino_sum,
            points=points,
        )


# Global cache of domino tables per device
_DOMINO_TABLES: dict[torch.device, DominoTables] = {}


def get_domino_tables(device: torch.device) -> DominoTables:
    """Get or create cached domino lookup tables for a device."""
    if device not in _DOMINO_TABLES:
        _DOMINO_TABLES[device] = DominoTables.create(device)
    return _DOMINO_TABLES[device]


def domino_contains_pip_gpu(
    domino_ids: Tensor,  # (...) int32 domino IDs, -1 for invalid
    pip: Tensor,         # (...) int32 pip value
    tables: DominoTables,
) -> Tensor:
    """Check if dominoes contain a specific pip value.

    Args:
        domino_ids: Domino IDs, -1 for invalid/played
        pip: Pip value to check for
        tables: Domino lookup tables

    Returns:
        Bool tensor, True where domino contains pip
    """
    # Handle -1 (invalid) by clamping to valid range for lookup
    safe_ids = domino_ids.clamp(min=0)
    high = tables.high[safe_ids]
    low = tables.low[safe_ids]

    # Check if pip matches high or low, but not if domino is invalid
    valid = domino_ids >= 0
    contains = (high == pip) | (low == pip)
    return valid & contains


def is_in_called_suit_gpu(
    domino_ids: Tensor,  # (...) int32 domino IDs
    decl_ids: Tensor,    # (...) int32 declaration IDs
    tables: DominoTables,
) -> Tensor:
    """Check if dominoes are in the called suit.

    For pip trumps (0-6): domino contains that pip
    For doubles trump/suit: domino is a double
    For notrump: always False

    Returns:
        Bool tensor
    """
    safe_ids = domino_ids.clamp(min=0)
    is_double = tables.is_double[safe_ids]

    # Pip trump: contains the trump pip
    is_pip_trump = decl_ids < 7
    contains_trump_pip = domino_contains_pip_gpu(domino_ids, decl_ids, tables)

    # Doubles trump/suit: is a double
    is_doubles_decl = (decl_ids == DOUBLES_TRUMP) | (decl_ids == DOUBLES_SUIT)

    # Notrump: never in called suit
    is_notrump = decl_ids == NOTRUMP

    result = torch.zeros_like(domino_ids, dtype=torch.bool)
    result = torch.where(is_pip_trump, contains_trump_pip, result)
    result = torch.where(is_doubles_decl, is_double, result)
    result = torch.where(is_notrump, torch.zeros_like(result), result)

    # Invalid dominoes are never in called suit
    valid = domino_ids >= 0
    return result & valid


def led_suit_for_lead_domino_gpu(
    lead_domino_ids: Tensor,  # (...) int32 lead domino IDs
    decl_ids: Tensor,         # (...) int32 declaration IDs
    tables: DominoTables,
) -> Tensor:
    """Compute led suit from lead domino.

    Returns suit in {0..6, 7=called suit}.
    - If notrump: high pip of lead domino
    - If lead is in called suit: 7
    - Otherwise: high pip of lead domino
    """
    safe_ids = lead_domino_ids.clamp(min=0)
    high_pip = tables.high[safe_ids]

    in_called = is_in_called_suit_gpu(lead_domino_ids, decl_ids, tables)

    # If in called suit: led suit is 7, otherwise high pip
    led_suit = torch.where(in_called, torch.full_like(high_pip, 7), high_pip)

    # Notrump always uses high pip
    is_notrump = decl_ids == NOTRUMP
    led_suit = torch.where(is_notrump, high_pip, led_suit)

    return led_suit


def can_follow_gpu(
    domino_ids: Tensor,   # (...) int32 domino IDs
    led_suits: Tensor,    # (...) int32 led suit (0-7)
    decl_ids: Tensor,     # (...) int32 declaration IDs
    tables: DominoTables,
) -> Tensor:
    """Check if dominoes can follow the led suit.

    If led_suit is 7 (called suit): must be in called suit
    Otherwise: must contain that pip AND not be in called suit
    """
    # Led suit is called suit (7)
    is_called_lead = led_suits == 7
    in_called = is_in_called_suit_gpu(domino_ids, decl_ids, tables)

    # Led suit is a pip (0-6)
    contains_pip = domino_contains_pip_gpu(domino_ids, led_suits, tables)

    # Can follow pip suit: contains pip AND not in called suit
    can_follow_pip = contains_pip & ~in_called

    # Select based on led suit type
    result = torch.where(is_called_lead, in_called, can_follow_pip)

    # Invalid dominoes cannot follow
    valid = domino_ids >= 0
    return result & valid


def has_trump_power_gpu(decl_ids: Tensor) -> Tensor:
    """Check if declaration has trump power.

    Pip trumps (0-6) and doubles trump (7) have trump power.
    Doubles suit (8) and notrump (9) do not.
    """
    return (decl_ids < 7) | (decl_ids == DOUBLES_TRUMP)


def _rank_in_pip_suit_gpu(domino_ids: Tensor, tables: DominoTables) -> Tensor:
    """Compute rank within a pip suit.

    Doubles rank 14, others rank by sum of pips.
    """
    safe_ids = domino_ids.clamp(min=0)
    is_double = tables.is_double[safe_ids]
    pip_sum = tables.sum[safe_ids]

    return torch.where(is_double, torch.full_like(pip_sum, 14), pip_sum)


def _rank_in_called_suit_gpu(
    domino_ids: Tensor,
    decl_ids: Tensor,
    tables: DominoTables,
) -> Tensor:
    """Compute rank within the called suit.

    For pip trumps: use pip suit ranking (double=14, else sum)
    For doubles trump: use high pip
    """
    safe_ids = domino_ids.clamp(min=0)
    pip_rank = _rank_in_pip_suit_gpu(domino_ids, tables)
    double_rank = tables.high[safe_ids]

    is_doubles_trump = decl_ids == DOUBLES_TRUMP
    return torch.where(is_doubles_trump, double_rank, pip_rank)


def trick_rank_gpu(
    domino_ids: Tensor,  # (...) int32
    led_suits: Tensor,   # (...) int32
    decl_ids: Tensor,    # (...) int32
    tables: DominoTables,
) -> Tensor:
    """Compute trick rank for dominoes. Higher rank wins.

    Returns 6-bit key: (tier << 4) + rank
    - Tier 2: In called suit with trump power
    - Tier 1: Can follow led suit
    - Tier 0: Cannot follow
    """
    safe_ids = domino_ids.clamp(min=0)

    # Check conditions
    in_called = is_in_called_suit_gpu(domino_ids, decl_ids, tables)
    trump_power = has_trump_power_gpu(decl_ids)
    can_follow = can_follow_gpu(domino_ids, led_suits, decl_ids, tables)

    # Compute ranks for different tiers
    called_rank = _rank_in_called_suit_gpu(domino_ids, decl_ids, tables)

    # For led suit rank: use pip ranking unless led suit is 7 (called suit)
    is_called_lead = led_suits == 7
    pip_rank = _rank_in_pip_suit_gpu(domino_ids, tables)
    high_pip_rank = tables.high[safe_ids]  # For doubles suit
    led_rank = torch.where(is_called_lead, high_pip_rank, pip_rank)

    # Build result
    result = torch.zeros_like(domino_ids, dtype=torch.int32)

    # Tier 1: can follow (but not trump with power)
    tier1 = can_follow & ~(in_called & trump_power)
    result = torch.where(tier1, (1 << 4) + led_rank, result)

    # Tier 2: in called suit with trump power (overrides tier 1)
    tier2 = in_called & trump_power
    result = torch.where(tier2, (2 << 4) + called_rank, result)

    # Invalid dominoes get rank 0
    valid = domino_ids >= 0
    return torch.where(valid, result, torch.zeros_like(result))


@dataclass
class GPUGameState:
    """Batched game state on GPU - N games in parallel.

    All tensors have batch dimension first.
    Uses slot indices (0-6) for hands, not domino IDs directly in actions.
    """

    hands: Tensor           # (N, 4, 7) int32 - domino IDs, -1 for played
    played_mask: Tensor     # (N, 28) bool - which dominoes have been played
    play_history: Tensor    # (N, 28, 3) int32 - (player, domino, lead_domino) per play
    n_plays: Tensor         # (N,) int32 - number of plays made
    current_trick: Tensor   # (N, 4, 2) int32 - (player, domino) for current trick
    trick_len: Tensor       # (N,) int32 - number of plays in current trick
    leader: Tensor          # (N,) int32 - current trick leader
    decl_id: Tensor         # (N,) int32 - declaration ID
    scores: Tensor          # (N, 2) int32 - team scores (points won)

    @property
    def device(self) -> torch.device:
        return self.hands.device

    @property
    def batch_size(self) -> int:
        return self.hands.shape[0]

    def clone(self) -> GPUGameState:
        """Create a deep copy of the state."""
        return GPUGameState(
            hands=self.hands.clone(),
            played_mask=self.played_mask.clone(),
            play_history=self.play_history.clone(),
            n_plays=self.n_plays.clone(),
            current_trick=self.current_trick.clone(),
            trick_len=self.trick_len.clone(),
            leader=self.leader.clone(),
            decl_id=self.decl_id.clone(),
            scores=self.scores.clone(),
        )


def current_player_gpu(states: GPUGameState) -> Tensor:
    """Compute current player for each game.

    Returns:
        (N,) int32 - current player (0-3)
    """
    return (states.leader + states.trick_len) % 4


def legal_actions_gpu(states: GPUGameState) -> Tensor:
    """Compute legal action mask for each game.

    Actions are slot indices (0-6), not domino IDs.

    Returns:
        (N, 7) bool - True for legal slots
    """
    n = states.batch_size
    device = states.device
    tables = get_domino_tables(device)

    # Get current player for each game
    players = current_player_gpu(states)  # (N,)

    # Get current player's hand: (N, 7)
    # Index into hands using advanced indexing
    batch_idx = torch.arange(n, device=device)
    player_hands = states.hands[batch_idx, players]  # (N, 7)

    # A slot is in-hand if its value is not -1
    in_hand = player_hands >= 0  # (N, 7) bool

    # If leading (trick_len == 0), all in-hand dominoes are legal
    is_leading = states.trick_len == 0  # (N,)

    # For following, need to check suit
    # Get lead domino from current trick (first play)
    lead_domino = states.current_trick[:, 0, 1]  # (N,) - domino ID of first play

    # Compute led suit
    led_suit = led_suit_for_lead_domino_gpu(lead_domino, states.decl_id, tables)  # (N,)

    # For each slot, check if it can follow
    # player_hands: (N, 7) domino IDs
    # led_suit: (N,) -> broadcast to (N, 7)
    # decl_id: (N,) -> broadcast to (N, 7)
    led_suit_exp = led_suit.unsqueeze(1).expand(-1, 7)  # (N, 7)
    decl_exp = states.decl_id.unsqueeze(1).expand(-1, 7)  # (N, 7)

    can_follow_mask = can_follow_gpu(player_hands, led_suit_exp, decl_exp, tables)  # (N, 7)
    can_follow_mask = can_follow_mask & in_hand  # Only if also in hand

    # Check if player has any followers
    has_followers = can_follow_mask.any(dim=1, keepdim=True)  # (N, 1)

    # If must follow: only followers are legal
    # If no followers: all in-hand are legal
    follow_legal = torch.where(has_followers, can_follow_mask, in_hand)

    # Final result: leading uses in_hand, following uses follow_legal
    is_leading_exp = is_leading.unsqueeze(1)  # (N, 1)
    legal = torch.where(is_leading_exp, in_hand, follow_legal)

    return legal


def _score_trick_gpu(
    domino_ids: Tensor,  # (N, 4) int32 - dominoes in trick
    tables: DominoTables,
) -> Tensor:
    """Compute points for completed tricks.

    Returns:
        (N,) int32 - points (1 base + count points)
    """
    safe_ids = domino_ids.clamp(min=0)
    points = tables.points[safe_ids]  # (N, 4)
    total = points.sum(dim=1) + 1  # Base 1 point per trick
    return total


def apply_action_gpu(states: GPUGameState, actions: Tensor) -> GPUGameState:
    """Apply actions to game states.

    Args:
        states: Current game states
        actions: (N,) int32 - slot indices (0-6), not domino IDs

    Returns:
        New GPUGameState after applying actions
    """
    n = states.batch_size
    device = states.device
    tables = get_domino_tables(device)

    batch_idx = torch.arange(n, device=device, dtype=torch.int64)
    players = current_player_gpu(states).to(torch.int64)  # (N,) - need int64 for indexing
    actions_i64 = actions.to(torch.int64)  # For indexing

    # Get domino ID from slot
    domino_ids = states.hands[batch_idx, players, actions_i64]  # (N,)
    domino_ids_i64 = domino_ids.to(torch.int64)

    # Clone tensors for modification
    new_hands = states.hands.clone()
    new_played_mask = states.played_mask.clone()
    new_play_history = states.play_history.clone()
    new_n_plays = states.n_plays.clone()
    new_current_trick = states.current_trick.clone()
    new_trick_len = states.trick_len.clone()
    new_leader = states.leader.clone()
    new_scores = states.scores.clone()

    # Remove domino from hand (set to -1)
    new_hands[batch_idx, players, actions_i64] = -1

    # Mark domino as played
    new_played_mask[batch_idx, domino_ids_i64] = True

    # Determine lead domino for this play
    is_leading = states.trick_len == 0
    lead_domino = torch.where(
        is_leading,
        domino_ids,
        states.current_trick[:, 0, 1]
    )

    # Add to play history
    play_idx = states.n_plays.to(torch.int64)  # (N,)
    new_play_history[batch_idx, play_idx, 0] = players.to(torch.int32)
    new_play_history[batch_idx, play_idx, 1] = domino_ids.to(torch.int32)
    new_play_history[batch_idx, play_idx, 2] = lead_domino.to(torch.int32)
    new_n_plays = new_n_plays + 1

    # Add to current trick
    trick_pos = states.trick_len.to(torch.int64)  # (N,)
    new_current_trick[batch_idx, trick_pos, 0] = players.to(torch.int32)
    new_current_trick[batch_idx, trick_pos, 1] = domino_ids.to(torch.int32)
    new_trick_len = new_trick_len + 1

    # Check if trick is complete (4 plays)
    trick_complete = new_trick_len == 4

    # Resolve completed tricks (masked; avoids GPU->CPU sync from tensor.any()).
    # Get dominoes in tricks
    trick_dominoes = new_current_trick[:, :, 1]  # (N, 4)

    # Compute ranks for all dominoes
    lead_domino_complete = new_current_trick[:, 0, 1]  # (N,)
    led_suit = led_suit_for_lead_domino_gpu(lead_domino_complete, states.decl_id, tables)

    # Expand for broadcast: (N, 4)
    led_suit_exp = led_suit.unsqueeze(1).expand(-1, 4)
    decl_exp = states.decl_id.unsqueeze(1).expand(-1, 4)

    ranks = trick_rank_gpu(trick_dominoes, led_suit_exp, decl_exp, tables)  # (N, 4)

    # Find winner (highest rank)
    winner_offset = ranks.argmax(dim=1).to(torch.int32)  # (N,)

    # New leader is (old leader + winner_offset) % 4
    winner_player = ((states.leader + winner_offset) % 4).to(torch.int32)

    # Compute points for this trick
    points = _score_trick_gpu(trick_dominoes, tables)

    # Update scores: team is player % 2
    winner_team = winner_player % 2
    team0_points = torch.where(winner_team == 0, points, torch.zeros_like(points))
    team1_points = torch.where(winner_team == 1, points, torch.zeros_like(points))

    # Apply updates only for complete tricks
    new_leader = torch.where(trick_complete, winner_player, new_leader)
    new_trick_len = torch.where(trick_complete, torch.zeros_like(new_trick_len), new_trick_len)
    new_scores[:, 0] = new_scores[:, 0] + torch.where(trick_complete, team0_points, torch.zeros_like(team0_points))
    new_scores[:, 1] = new_scores[:, 1] + torch.where(trick_complete, team1_points, torch.zeros_like(team1_points))

    # Clear current trick for completed games
    clear_mask = trick_complete.unsqueeze(1).unsqueeze(2)  # (N, 1, 1)
    new_current_trick = torch.where(clear_mask, torch.zeros_like(new_current_trick), new_current_trick)

    return GPUGameState(
        hands=new_hands,
        played_mask=new_played_mask,
        play_history=new_play_history,
        n_plays=new_n_plays,
        current_trick=new_current_trick,
        trick_len=new_trick_len,
        leader=new_leader,
        decl_id=states.decl_id,  # Declaration doesn't change
        scores=new_scores,
    )


def is_terminal_gpu(states: GPUGameState) -> Tensor:
    """Check if games are complete (all 28 dominoes played).

    Returns:
        (N,) bool - True for completed games
    """
    return states.n_plays == 28


def deal_random_gpu(
    n: int,
    device: torch.device,
    decl_ids: Tensor | int | None = None,
    leaders: Tensor | int | None = None,
) -> GPUGameState:
    """Generate N random deals on GPU.

    Args:
        n: Number of games to create
        device: Target device
        decl_ids: Declaration IDs (N,) or scalar. If None, random 0-9.
        leaders: Starting leaders (N,) or scalar. If None, all 0.

    Returns:
        GPUGameState with N fresh deals
    """
    device = require_cuda(device, where="deal_random_gpu")

    # Generate random permutations of 28 dominoes for each game
    # torch doesn't have direct batch permutation, so we use argsort of random values
    random_vals = torch.rand(n, 28, device=device)
    shuffled = random_vals.argsort(dim=1)  # (N, 28) permuted indices

    # Reshape into 4 hands of 7
    hands = shuffled.reshape(n, 4, 7).to(torch.int32)

    # Initialize other state tensors
    played_mask = torch.zeros(n, 28, dtype=torch.bool, device=device)
    play_history = torch.zeros(n, 28, 3, dtype=torch.int32, device=device)
    n_plays = torch.zeros(n, dtype=torch.int32, device=device)
    current_trick = torch.zeros(n, 4, 2, dtype=torch.int32, device=device)
    trick_len = torch.zeros(n, dtype=torch.int32, device=device)

    # Handle leaders
    if leaders is None:
        leader = torch.zeros(n, dtype=torch.int32, device=device)
    elif isinstance(leaders, int):
        leader = torch.full((n,), leaders, dtype=torch.int32, device=device)
    else:
        leader = leaders.to(torch.int32)

    # Handle declaration IDs
    if decl_ids is None:
        decl_id = torch.randint(0, 10, (n,), dtype=torch.int32, device=device)
    elif isinstance(decl_ids, int):
        decl_id = torch.full((n,), decl_ids, dtype=torch.int32, device=device)
    else:
        decl_id = decl_ids.to(torch.int32)

    scores = torch.zeros(n, 2, dtype=torch.int32, device=device)

    return GPUGameState(
        hands=hands,
        played_mask=played_mask,
        play_history=play_history,
        n_plays=n_plays,
        current_trick=current_trick,
        trick_len=trick_len,
        leader=leader,
        decl_id=decl_id,
        scores=scores,
    )


def from_python_states(
    states: list[GameState],
    device: torch.device,
    original_hands: list[list[list[int]]] | None = None,
) -> GPUGameState:
    """Convert Python GameState objects to GPUGameState.

    This is for bootstrapping/testing - in production, games should
    be created and simulated entirely on GPU.

    Args:
        states: List of Python GameState objects
        device: Target device
        original_hands: Optional original hands for each game (N, 4, 7).
                        If provided, maintains slot positions with -1 for played.
                        If None, packs remaining dominoes to the left.

    Returns:
        GPUGameState with equivalent state
    """
    raise RuntimeError(
        "from_python_states() is disabled: GPU-native pipeline must not depend on CPU GameState "
        "bootstrapping. Use deal_random_gpu() / GPU-native self-play instead."
    )


def to_python_state(gpu_state: GPUGameState, idx: int) -> GameState:
    """Convert a single game from GPUGameState to Python GameState.

    For testing and debugging.
    """
    from forge.eq.game import GameState

    # Extract hands, filtering out -1
    hands_raw = gpu_state.hands[idx].cpu().tolist()
    hands = tuple(
        tuple(d for d in hand if d >= 0)
        for hand in hands_raw
    )

    # Extract played set
    played_mask = gpu_state.played_mask[idx].cpu().tolist()
    played = frozenset(d for d in range(28) if played_mask[d])

    # Extract play history
    n_plays = gpu_state.n_plays[idx].item()
    play_history_raw = gpu_state.play_history[idx].cpu().tolist()
    play_history = tuple(
        (p[0], p[1], p[2])
        for p in play_history_raw[:n_plays]
    )

    # Extract current trick
    trick_len = gpu_state.trick_len[idx].item()
    current_trick_raw = gpu_state.current_trick[idx].cpu().tolist()
    current_trick = tuple(
        (t[0], t[1])
        for t in current_trick_raw[:trick_len]
    )

    leader = gpu_state.leader[idx].item()
    decl_id = gpu_state.decl_id[idx].item()

    return GameState(
        hands=hands,
        played=played,
        play_history=play_history,
        current_trick=current_trick,
        leader=leader,
        decl_id=decl_id,
    )
