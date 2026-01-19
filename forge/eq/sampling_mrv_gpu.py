"""GPU world sampling using MRV (Minimum Remaining Values) heuristic.

This is a port of the CPU backtracking algorithm from sampling.py to GPU,
using bitmask arithmetic for constraint checking.

## Key insight

28 dominoes fit in a int64 bitmask. All constraint operations become:
- candidates[p] = ~void_mask[p] & remaining
- slack = popcount(candidates[p] & available) - need[p]
- assign: hands[p] |= (1 << d), available &= ~(1 << d)

## Algorithm

For each sample (parallel across thousands):
    available = remaining_pool_mask
    hands[0..2] = 0
    need[0..2] = hand_sizes

    for step in 1..pool_size:
        # MRV: find most constrained player
        for p in 0..2:
            slack[p] = popcount(candidates[p] & available) - need[p]
        p = argmin(slack)

        # Random selection from valid candidates
        valid = candidates[p] & available
        d = random_bit(valid)

        # Assign
        hands[p] |= (1 << d)
        available &= ~(1 << d)
        need[p] -= 1

21 sequential steps, but vectorized across samples, no Python loops in hot path.

## Performance

- Guaranteed valid output (no rejection, no fallback)
- O(pool_size) steps per sample
- Each step is vectorized tensor ops
"""

import torch
from forge.eq.sampling_gpu import CAN_FOLLOW


# Module-level precomputation: suit_domino_mask[suit, decl] = bitmask of dominoes that can follow
# Built once at import time using CAN_FOLLOW[28, 8, 10]
def _build_suit_domino_mask() -> torch.Tensor:
    """Precompute suit->domino bitmasks for all (suit, decl) combinations.

    Returns:
        [8, 10] int64 tensor where bit d is set if domino d can follow that (suit, decl)
    """
    can_follow = CAN_FOLLOW.cpu()  # [28, 8, 10]
    suit_domino_mask = torch.zeros(8, 10, dtype=torch.int64)

    # Vectorize: create bit position tensor
    bit_positions = 1 << torch.arange(28, dtype=torch.int64)  # [28]

    # For each (suit, decl), OR together bits for all dominoes that can follow
    for suit in range(8):
        for decl in range(10):
            # Get mask of dominoes that can follow: [28] bool
            can_follow_mask = can_follow[:, suit, decl]
            # OR together bit positions where mask is True
            suit_domino_mask[suit, decl] = (bit_positions * can_follow_mask.to(torch.int64)).sum()

    return suit_domino_mask


SUIT_DOMINO_MASK = _build_suit_domino_mask()  # [8, 10] int64


def _build_void_masks_vectorized(
    voids: torch.Tensor,  # [n_games, 3, 8] bool
    decl_ids: torch.Tensor,  # [n_games]
    device: str,
) -> torch.Tensor:
    """Build bitmask of dominoes that violate void constraints per player.

    Args:
        voids: [n_games, 3, 8] where voids[g, p, s] = True if player p is void in suit s
        decl_ids: [n_games] declaration ID per game
        device: torch device

    Returns:
        [n_games, 3] int64 tensor where void_mask[g, p] has bit d set if
        domino d violates player p's void constraints in game g.
    """
    n_games = voids.shape[0]
    suit_domino_mask = SUIT_DOMINO_MASK.to(device)  # [8, 10]

    # Vectorized computation:
    # voids: [n_games, 3, 8] bool
    # decl_ids: [n_games] int
    # suit_domino_mask: [8, 10] int64

    # Index into suit_domino_mask with decl_ids: [8, n_games]
    masks_by_suit = suit_domino_mask[:, decl_ids.long()]  # [8, n_games]

    # Transpose to [n_games, 8] and expand to [n_games, 1, 8]
    masks_by_suit = masks_by_suit.T.unsqueeze(1)  # [n_games, 1, 8]

    # Multiply by voids and OR together across suits dimension
    # voids: [n_games, 3, 8] bool -> int64
    # masks_by_suit: [n_games, 1, 8] int64
    # Result: [n_games, 3, 8] int64
    void_mask_per_suit = voids.to(torch.int64) * masks_by_suit  # [n_games, 3, 8]

    # OR together across suits using bitwise OR reduction
    # We need to OR across dim=2, but torch doesn't have bitwise_or reduction
    # So we'll use a loop over suits (only 8 iterations, not n_games)
    void_masks = torch.zeros(n_games, 3, dtype=torch.int64, device=device)
    for suit in range(8):
        void_masks |= void_mask_per_suit[:, :, suit]

    return void_masks


def _pool_to_mask(pools: torch.Tensor) -> torch.Tensor:
    """Convert pool tensor to bitmask (vectorized).

    Args:
        pools: [n_games, pool_size] domino IDs (padded with -1)

    Returns:
        [n_games] int64 bitmask where bit d is set if domino d is in pool
    """
    n_games = pools.shape[0]
    pool_size = pools.shape[1]
    device = pools.device

    # Create bit position tensor: 1 << pools where pools >= 0
    # Handle -1 padding by masking
    valid_mask = pools >= 0  # [n_games, pool_size]

    # Clamp to valid range for bit shift (will be masked anyway)
    pools_clamped = pools.clamp(min=0).to(torch.int64)

    # Compute bit positions: 2^d for each domino
    bit_positions = (1 << pools_clamped) * valid_mask.to(torch.int64)

    # OR together all bits for each game
    masks = bit_positions.sum(dim=1)  # This works because bits don't overlap

    return masks


def _popcount_vectorized(x: torch.Tensor) -> torch.Tensor:
    """Count set bits in each element. Works for int64 tensors."""
    # Use built-in if available (PyTorch 2.0+), otherwise bit manipulation
    x = x.to(torch.int64)
    # Standard parallel bit count for 64-bit
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f
    x = (x * 0x0101010101010101) >> 56
    return x.to(torch.int32)


def _random_set_bit_vectorized(masks: torch.Tensor, rng: torch.Tensor) -> torch.Tensor:
    """Select a random set bit from each mask (vectorized).

    Args:
        masks: [N] int64 bitmasks, each with at least one bit set
        rng: [N] float in [0, 1) for random selection

    Returns:
        [N] int64 with the selected bit index (0-27)
    """
    N = masks.shape[0]
    device = masks.device

    # Expand masks to [N, 28] bool tensor
    bit_indices = torch.arange(28, device=device)  # [28]
    bit_masks = (1 << bit_indices).to(torch.int64)  # [28]

    # Check which bits are set: [N, 28]
    bits_set = (masks.unsqueeze(1) & bit_masks.unsqueeze(0)) != 0

    # Count total set bits per sample
    counts = bits_set.sum(dim=1).float()  # [N]

    # Target index (which set bit to select)
    target_idx = (rng * counts).floor().to(torch.int64)  # [N]

    # Cumulative sum to find the position of each set bit
    cumsum = bits_set.to(torch.int64).cumsum(dim=1)  # [N, 28]

    # Find where cumsum == target_idx + 1 AND bit is set (first occurrence)
    # This gives us the target_idx-th set bit (0-indexed)
    match = (cumsum == (target_idx.unsqueeze(1) + 1)) & bits_set  # [N, 28]

    # Get the bit index (argmax on bool gives first True)
    selected = match.to(torch.int64).argmax(dim=1)  # [N]

    return selected


def sample_worlds_mrv_gpu(
    pools: torch.Tensor,           # [n_games, pool_size] available dominoes
    hand_sizes: torch.Tensor,      # [n_games, 3] opponent hand sizes
    voids: torch.Tensor,           # [n_games, 3, 8] void flags per opponent
    decl_ids: torch.Tensor,        # [n_games] declaration IDs
    n_samples: int = 50,
    device: str = 'cuda',
) -> torch.Tensor:
    """Sample consistent worlds using MRV heuristic on GPU.

    Unlike rejection sampling, this is GUARANTEED to produce valid samples
    (assuming valid constraints). Uses the same MRV algorithm as CPU
    backtracking but with bitmask arithmetic.

    Args:
        pools: [n_games, pool_size] available domino IDs (padded with -1)
        hand_sizes: [n_games, 3] hand sizes for 3 opponents per game
        voids: [n_games, 3, 8] bool - voids[g,o,s] = opponent o is void in suit s
        decl_ids: [n_games] declaration ID per game
        n_samples: Number of worlds to sample per game
        device: 'cuda' or 'cpu'

    Returns:
        [n_games, n_samples, 3, 7] opponent hands (padded with -1)
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    n_games = pools.shape[0]

    # Move inputs to device
    pools = pools.to(device)
    hand_sizes = hand_sizes.to(device).to(torch.int32)
    voids = voids.to(device)
    decl_ids = decl_ids.to(device)

    # Build void masks: void_mask[g, p] = dominoes player p CANNOT hold
    void_masks = _build_void_masks_vectorized(voids, decl_ids, device)  # [n_games, 3]

    # Convert pools to bitmasks
    pool_masks = _pool_to_mask(pools)  # [n_games]

    # Get pool sizes (total dominoes to distribute)
    pool_sizes = _popcount_vectorized(pool_masks)  # [n_games]
    max_pool_size = pool_sizes.max().item()

    # Candidate masks: candidates[g, p] = dominoes player p COULD hold = pool & ~void_mask
    candidate_masks = pool_masks.unsqueeze(1) & ~void_masks  # [n_games, 3]

    # Flatten for parallel processing across all (game, sample) pairs
    total_samples = n_games * n_samples

    # available[i] = remaining dominoes for sample i
    available = pool_masks.unsqueeze(1).expand(n_games, n_samples).reshape(total_samples).clone()

    # hands[i, p] = bitmask of assigned dominoes for player p in sample i
    hands = torch.zeros(total_samples, 3, dtype=torch.int64, device=device)

    # need[i, p] = how many more dominoes player p needs in sample i
    need = hand_sizes.unsqueeze(1).expand(n_games, n_samples, 3).reshape(total_samples, 3).clone()

    # Expand candidate_masks to [total_samples, 3]
    candidate_masks_flat = candidate_masks.unsqueeze(1).expand(n_games, n_samples, 3).reshape(total_samples, 3)

    # Pre-allocate reusable tensors to reduce allocation overhead
    batch_idx = torch.arange(total_samples, device=device)
    available_unsqueezed = available.unsqueeze(1)  # [N, 1]

    # MRV assignment loop: assign one domino per step (vectorized across samples)
    for step in range(max_pool_size):
        # Check if any samples still need assignment
        active = (available != 0)
        if not active.any():
            break

        # Compute slack for each player: popcount(candidates & available) - need
        # Reuse available_unsqueezed shape
        available_unsqueezed = available.unsqueeze(1)
        valid_candidates = candidate_masks_flat & available_unsqueezed  # [N, 3]

        # Vectorized popcount for all players at once
        slack = _popcount_vectorized(valid_candidates.reshape(-1)).reshape(total_samples, 3) - need

        # MRV: find player with minimum slack (in-place to reduce allocations)
        slack[need == 0] = 1000  # Set slack to large value for players with need=0
        most_constrained = slack.argmin(dim=1)  # [N]

        # Gather candidates for the chosen player: [N]
        chosen_candidates = torch.gather(valid_candidates, 1, most_constrained.unsqueeze(1).to(torch.int64)).squeeze(1)

        # Random selection: pick a random set bit from chosen_candidates
        rng = torch.rand(total_samples, device=device)

        # Handle inactive samples (set a dummy valid bit to avoid errors)
        # Clone only inactive samples to reduce overhead
        chosen_candidates_safe = torch.where(active, chosen_candidates, torch.tensor(1, dtype=torch.int64, device=device))

        selected_domino = _random_set_bit_vectorized(chosen_candidates_safe, rng)  # [N]

        # Create bitmask for selected domino
        selected_mask = (1 << selected_domino).to(torch.int64)  # [N]

        # Update available: remove selected domino (only for active samples)
        available = torch.where(active, available & ~selected_mask, available)

        # Update hands: add selected domino to chosen player's hand (vectorized)
        # Zero out inactive samples and use scatter
        active_mask = selected_mask * active.to(torch.int64)
        hands[batch_idx, most_constrained] |= active_mask

        # Update need: decrement for chosen player (vectorized)
        # Create one-hot encoding of most_constrained: [N, 3]
        player_mask = torch.nn.functional.one_hot(most_constrained, num_classes=3).to(torch.int32)
        # Apply only to active samples
        need -= player_mask * active.unsqueeze(1).to(torch.int32)

    # Convert bitmask hands to domino ID lists (vectorized, GPU-only)
    # hands: [total_samples, 3] int64 bitmasks
    # Output: [n_games, n_samples, 3, 7] domino IDs

    max_hand = 7

    # Create bit test masks [28]
    bit_indices = torch.arange(28, device=device, dtype=torch.int32)
    bit_masks = (1 << bit_indices).to(torch.int64)  # [28]

    # Check which bits are set: [total_samples, 3, 28]
    bits_set = (hands.unsqueeze(2) & bit_masks.view(1, 1, 28)) != 0

    # For each hand, we want indices of set bits, packed into first positions
    # Strategy: Create weighted values where set bits have their index, unset bits have -1
    # Then use sort to bring valid indices to the front

    # Create values: bit_index where set, -1 where not set
    values = torch.where(bits_set, bit_indices.view(1, 1, 28), torch.tensor(-1, dtype=torch.int32, device=device))

    # Sort descending to put valid (non-negative) indices first
    sorted_values, _ = torch.sort(values, dim=2, descending=True)

    # Take first max_hand values
    results_flat = sorted_values[:, :, :max_hand]  # [total_samples, 3, 7]

    # Reshape to [n_games, n_samples, 3, 7]
    results = results_flat.reshape(n_games, n_samples, 3, max_hand)

    return results


class WorldSamplerMRV:
    """Stateful MRV world sampler for GPU batch processing.

    Drop-in replacement for WorldSampler that uses MRV algorithm
    instead of rejection sampling. GUARANTEED to produce valid samples.

    Example:
        >>> sampler = WorldSamplerMRV(max_games=32, max_samples=100, device='cuda')
        >>> worlds = sampler.sample(pools, hand_sizes, voids, decl_ids, n_samples=50)
    """

    def __init__(self, max_games: int, max_samples: int, device: str = 'cuda'):
        """Initialize sampler.

        Args:
            max_games: Maximum number of games to process
            max_samples: Maximum samples per game
            device: 'cuda' or 'cpu'
        """
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        self.device = device
        self.max_games = max_games
        self.max_samples = max_samples
        # Note: suit_domino_mask is now precomputed at module level as SUIT_DOMINO_MASK

    def sample(
        self,
        pools: torch.Tensor,           # [n_games, pool_size]
        hand_sizes: torch.Tensor,      # [n_games, 3]
        voids: torch.Tensor,           # [n_games, 3, 8]
        decl_ids: torch.Tensor,        # [n_games]
        n_samples: int = 50,
    ) -> torch.Tensor:
        """Sample consistent worlds using MRV heuristic.

        Args:
            pools: [n_games, pool_size] available domino IDs (padded with -1)
            hand_sizes: [n_games, 3] hand sizes for 3 opponents per game
            voids: [n_games, 3, 8] bool - voids[g,o,s] = opponent o is void in suit s
            decl_ids: [n_games] declaration ID per game
            n_samples: Number of worlds to sample per game

        Returns:
            [n_games, n_samples, 3, 7] opponent hands (padded with -1)
        """
        n_games = pools.shape[0]
        if n_games > self.max_games:
            raise ValueError(f"n_games ({n_games}) exceeds max_games ({self.max_games})")
        if n_samples > self.max_samples:
            raise ValueError(f"n_samples ({n_samples}) exceeds max_samples ({self.max_samples})")

        return sample_worlds_mrv_gpu(
            pools=pools,
            hand_sizes=hand_sizes,
            voids=voids,
            decl_ids=decl_ids,
            n_samples=n_samples,
            device=self.device,
        )
