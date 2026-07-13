"""World sampling functions for GPU E[Q] pipeline."""

from __future__ import annotations

import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.sampling_gpu import WorldSampler
from forge.eq.sampling_mrv_gpu import WorldSamplerMRV

# Per-device caches of small constant tensors. Rebuilding these per call is a
# host->device transfer + kernel launch each on GPU backends.
_ARANGE28: dict[str, Tensor] = {}
_VOID_TABLES: dict[str, tuple[Tensor, Tensor]] = {}


def _arange28(device) -> Tensor:
    key = str(device)
    t = _ARANGE28.get(key)
    if t is None:
        t = torch.arange(28, device=device)
        _ARANGE28[key] = t
    return t


def sample_worlds_batched(
    states: GameStateTensor,
    sampler: WorldSampler | WorldSamplerMRV,
    n_samples: int,
    max_pool_size: int | None = None,
) -> Tensor:
    """Sample consistent worlds for all games.

    Vectorized implementation - no Python loops or .item() calls.

    Args:
        states: GameStateTensor with n_games
        sampler: Pre-allocated WorldSampler or WorldSamplerMRV
        n_samples: Number of samples per game
        max_pool_size: Largest pool (unseen dominoes) across the batch, if the
            caller already knows it. The legacy-named MRV sampler validates
            this hint against the tensor-derived size.

    Returns:
        [n_games, n_samples, 3, 7] opponent hands

    Note:
        ``WorldSamplerMRV`` now uses exact suffix-completion counts internally.
        It samples valid assignments uniformly without rejection.
    """
    n_games = states.n_games
    device = states.device
    current_players = states.current_player.long()  # [N]

    # === Step 1: Vectorize pool computation ===
    # For each game, pool = all_dominoes - played - my_hand
    all_dominoes = _arange28(device)  # [28]

    # Get current player's hand for each game using gather
    # states.hands: [N, 4, 7]
    # current_players: [N] -> expand to [N, 1, 7] for gathering
    player_idx_for_hands = current_players.view(n_games, 1, 1).expand(n_games, 1, 7)
    my_hands = torch.gather(states.hands, dim=1, index=player_idx_for_hands).squeeze(1)  # [N, 7]

    # Build mask for my dominoes across all games
    # my_hands: [N, 7], values are domino IDs or -1
    my_hand_mask = my_hands >= 0  # [N, 7]

    # Create pool masks for all games: [N, 28]
    # pool_mask[g, d] = True if domino d is in the pool for game g
    pool_masks = ~states.played_mask  # [N, 28] - start with unplayed

    # Remove my dominoes from pool. scatter_add on an int counter is safe under
    # duplicate indices (padding slots all target index 0 with value 0), and
    # avoids the boolean-mask advanced indexing that forces a GPU->CPU sync.
    my_dominoes_safe = torch.where(my_hand_mask, my_hands.long(), torch.zeros_like(my_hands, dtype=torch.long))
    mine_counts = torch.zeros(n_games, 28, dtype=torch.int8, device=device)
    mine_counts.scatter_add_(1, my_dominoes_safe, my_hand_mask.to(torch.int8))
    pool_masks = pool_masks & (mine_counts == 0)

    # Convert pool_masks to pool lists with padding: ascending domino IDs first,
    # then -1 padding — byte-identical to the historical per-game loop, so the
    # legacy WorldSampler rejection path (order-sensitive) sees the same input.
    # Sort trick: valid slots keep their domino ID, empty slots become 99, an
    # ascending sort packs the IDs to the front, then 99 -> -1.
    pool_vals = torch.where(pool_masks, all_dominoes, torch.full_like(all_dominoes, 99))  # [N, 28]
    pool_sorted, _ = torch.sort(pool_vals, dim=1)
    pools = torch.where(pool_sorted == 99, torch.full_like(pool_sorted, -1), pool_sorted)[:, :21].to(torch.int32)

    # === Step 2: Vectorize hand sizes computation ===
    # hand_counts: [N, 4] - number of dominoes per player per game
    hand_counts = (states.hands >= 0).sum(dim=2)  # [N, 4]

    # For each game, get opponent hand sizes (3 opponents)
    # Opponent i = (current_player + i + 1) % 4 for i in [0, 1, 2]
    offsets = torch.arange(1, 4, device=device).unsqueeze(0)  # [1, 3]
    opponent_indices = (current_players.unsqueeze(1) + offsets) % 4  # [N, 3]

    # Gather opponent hand sizes: [N, 3]
    # hand_counts: [N, 4], opponent_indices: [N, 3]
    hand_sizes_t = torch.gather(hand_counts, dim=1, index=opponent_indices.long())  # [N, 3]

    # === Step 3: Vectorize voids inference ===
    voids_t = infer_voids_batched(states)  # [N, 3, 8]

    # Get decl_ids
    decl_ids_t = states.decl_ids

    # Sample worlds - GPU only, no fallback
    if isinstance(sampler, WorldSamplerMRV):
        worlds = sampler.sample(
            pools, hand_sizes_t, voids_t, decl_ids_t, n_samples,
            max_pool_size=max_pool_size,
        )
    else:
        worlds = sampler.sample(pools, hand_sizes_t, voids_t, decl_ids_t, n_samples)

    return worlds


def infer_voids_batched(states: GameStateTensor) -> Tensor:
    """Infer void suits from play history for all games (fully vectorized).

    Uses precomputed lookup tables to eliminate Python loops and CPU roundtrips.

    Args:
        states: GameStateTensor with N games

    Returns:
        [N, 3, 8] boolean tensor where voids[g, opp_idx, suit] = True if opponent is void in suit
        Opponent indices are relative to current_player:
        - 0 = (current_player + 1) % 4
        - 1 = (current_player + 2) % 4
        - 2 = (current_player + 3) % 4
    """
    n_games = states.n_games
    device = states.device

    # Get lookup tables on correct device (cached; .to() re-copies every call
    # when the module-level tables live on CPU)
    key = str(device)
    tables = _VOID_TABLES.get(key)
    if tables is None:
        from forge.eq.game_tensor import LED_SUIT_TABLE
        from forge.eq.sampling_gpu import CAN_FOLLOW

        tables = (LED_SUIT_TABLE.to(device), CAN_FOLLOW.to(device))
        _VOID_TABLES[key] = tables
    led_suit_table, can_follow_table = tables  # [28, 10], [28, 8, 10]

    # history: [N, 28, 3] with columns (player, domino_id, lead_domino_id)
    history = states.history  # Keep on GPU
    decl_ids = states.decl_ids  # [N]
    current_players = states.current_player  # [N]

    # Find valid history entries (player >= 0)
    valid_mask = history[:, :, 0] >= 0  # [N, 28]

    # Extract columns
    players = history[:, :, 0].long()  # [N, 28]
    domino_ids = history[:, :, 1].long()  # [N, 28]
    lead_domino_ids = history[:, :, 2].long()  # [N, 28]

    # Clamp to valid range for indexing (invalid entries will be masked out)
    domino_ids_safe = domino_ids.clamp(0, 27)
    lead_domino_ids_safe = lead_domino_ids.clamp(0, 27)

    # Expand decl_ids for indexing: [N] -> [N, 28]
    decl_ids_expanded = decl_ids.unsqueeze(1).expand(-1, 28).long()

    # Look up led suits: led_suit_table[lead_domino_id, decl_id]
    # Use advanced indexing: [N, 28]
    led_suits = led_suit_table[lead_domino_ids_safe, decl_ids_expanded]  # [N, 28]

    # Look up can_follow: can_follow_table[domino_id, led_suit, decl_id]
    can_follow_result = can_follow_table[domino_ids_safe, led_suits.long(), decl_ids_expanded]  # [N, 28]

    # A void is revealed when can_follow == False
    void_revealed = ~can_follow_result & valid_mask  # [N, 28]

    # Compute relative opponent indices: opp_idx = (player - current_player - 1) % 4
    # current_players: [N] -> [N, 1] for broadcasting
    relative_opp = (players - current_players.unsqueeze(1) - 1) % 4  # [N, 28]

    # Filter: only opponents (opp_idx < 3) and not current player
    is_opponent = (players != current_players.unsqueeze(1)) & (relative_opp < 3)  # [N, 28]

    # Final mask for void revelations from opponents
    void_mask = void_revealed & is_opponent & valid_mask  # [N, 28]

    # Scatter voids into result tensor: for each (game, history_entry) where
    # void_mask is True, set voids[game, relative_opp, led_suit] = True.
    #
    # Maskless scatter_add (no .any()/.nonzero() GPU->CPU sync): masked-out
    # entries contribute 0 at a clamped index, which is harmless.
    flat_idx = relative_opp.clamp(0, 2) * 8 + led_suits.long()  # [N, 28] in [0, 24)
    counts = torch.zeros(n_games, 24, dtype=torch.int32, device=device)
    counts.scatter_add_(1, flat_idx, void_mask.to(torch.int32))
    voids = (counts > 0).view(n_games, 3, 8)

    return voids
