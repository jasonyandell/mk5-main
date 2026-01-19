"""GPU-accelerated world sampling using parallel first-fit algorithm.

Instead of CPU backtracking, generates many random permutations in parallel on GPU
and filters for valid ones.

## Performance Characteristics

Current benchmarks show CPU backtracking is actually FASTER for typical use cases:
- CPU: ~5ms for 100 samples with realistic voids
- GPU: ~80ms per call (overhead from tensor creation and transfer)

The GPU overhead comes from:
1. Creating tensors for each call
2. Moving CAN_FOLLOW to GPU device
3. CPU<->GPU memory transfers
4. Kernel launch overhead

## When GPU Sampling Makes Sense

The GPU implementation is valuable for:
1. **Batched oracle queries across games** - when we batch world sampling for
   multiple games simultaneously, we can amortize the overhead
2. **Very large sample counts** (1000+) - the parallelism scales better
3. **Future optimization** - pre-allocating tensors and reusing buffers could
   dramatically reduce overhead

## Current Recommendation

Use CPU sampling (`sampling.py`) for the current E[Q] pipeline. Consider GPU
sampling when implementing cross-game batching (t42-5kvo phase 1).
"""

import numpy as np
import torch
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW, DOMINO_IS_DOUBLE
from forge.oracle.declarations import PIP_TRUMP_IDS, DOUBLES_TRUMP, DOUBLES_SUIT, NOTRUMP, N_DECLS


# Pre-compute constraint tensors at module load
# CAN_FOLLOW[domino_id, led_suit, decl_id] = True if domino can follow the led suit
# led_suit in {0..6, 7=called suit}
def _build_can_follow_tensor() -> torch.Tensor:
    """Build lookup tensor for void constraint checking.

    Returns:
        Tensor of shape (28, 8, 10) where CAN_FOLLOW[domino, suit, decl] is True
        if the domino can follow that suit under that declaration.
    """
    can_follow = torch.zeros(28, 8, N_DECLS, dtype=torch.bool)

    for domino_id in range(28):
        high = DOMINO_HIGH[domino_id]
        low = DOMINO_LOW[domino_id]
        is_double = DOMINO_IS_DOUBLE[domino_id]

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

            # For each possible led suit
            for led_suit in range(8):
                if led_suit == 7:  # Called suit led
                    # Can follow if in called suit
                    can_follow[domino_id, led_suit, decl_id] = in_called
                else:  # Pip suit led (0-6)
                    # Can follow if: (1) contains that pip AND (2) not in called suit
                    has_pip = (led_suit == high) or (led_suit == low)
                    can_follow[domino_id, led_suit, decl_id] = has_pip and not in_called

    return can_follow


CAN_FOLLOW = _build_can_follow_tensor()


def check_void_constraints_gpu(
    hands: torch.Tensor,  # (M, 3, max_hand_size) - padded with -1
    hand_sizes: list[int],  # [h0, h1, h2]
    voids: dict[int, set[int]],  # {player: {void_suits}}
    decl_id: int,
    device: str,
) -> torch.Tensor:  # (M,) bool
    """Check void constraints for all M permutations in parallel.

    Args:
        hands: Tensor of shape (M, 3, max_hand_size) with domino IDs (padded with -1)
        hand_sizes: List of actual hand sizes for each of 3 opponents
        voids: Dict mapping opponent index (0, 1, 2) to set of void suits
        decl_id: Declaration ID
        device: Device to run on

    Returns:
        Boolean tensor of shape (M,) where True means valid (no violations)
    """
    can_follow = CAN_FOLLOW.to(device)
    return _check_void_constraints_gpu_internal(
        hands, hand_sizes, voids, decl_id, device, can_follow
    )


def sample_worlds_gpu(
    pool: set[int],
    hand_sizes: list[int],  # [h0, h1, h2] for 3 opponents
    voids: dict[int, set[int]],  # {opp_idx: {void_suits}}
    decl_id: int,
    n_samples: int,
    oversample_factor: float = 10.0,
    device: str = 'cuda',
    max_retries: int = 3,
    _cached_pool_tensor: torch.Tensor | None = None,
    _cached_can_follow: torch.Tensor | None = None,
) -> list[list[int]]:
    """Sample opponent hands using GPU parallel first-fit algorithm.

    Generates M >> N random permutations in parallel, validates constraints,
    and returns first N valid samples. Falls back to CPU if not enough valid
    samples found after retries.

    Args:
        pool: Set of available domino IDs (not my hand, not played)
        hand_sizes: List of [h0, h1, h2] - hand sizes for 3 opponents
        voids: Dict mapping opponent index (0-2) to set of void suits (0-6 or 7)
        decl_id: Declaration ID for constraint checking
        n_samples: Number of valid worlds to return
        oversample_factor: Generate this many times n_samples permutations
        device: 'cuda' or 'cpu'
        max_retries: How many times to retry if not enough valid samples
        _cached_pool_tensor: Internal - pre-computed pool tensor (for optimization)
        _cached_can_follow: Internal - pre-computed CAN_FOLLOW on device

    Returns:
        List of N opponent hand assignments, each is [hand0, hand1, hand2]
        where each hand is a list of domino IDs.

    Raises:
        RuntimeError: If unable to find enough valid samples after retries
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    pool_list = sorted(pool)  # Deterministic ordering
    pool_size = len(pool_list)

    # Validate input
    total_needed = sum(hand_sizes)
    if pool_size != total_needed:
        raise ValueError(
            f"Pool size ({pool_size}) doesn't match total needed ({total_needed})"
        )

    # Pre-allocate pool tensor once (reuse if cached)
    if _cached_pool_tensor is None:
        pool_t = torch.tensor(pool_list, dtype=torch.int32, device=device)
    else:
        pool_t = _cached_pool_tensor

    # Pre-move CAN_FOLLOW to device once
    if _cached_can_follow is None:
        can_follow = CAN_FOLLOW.to(device)
    else:
        can_follow = _cached_can_follow

    max_hand_size = max(hand_sizes)

    # Try with increasing oversample factors
    for retry in range(max_retries):
        current_oversample = oversample_factor * (1.5 ** retry)
        M = int(n_samples * current_oversample)

        # 1. Generate M random permutations
        rand = torch.rand(M, pool_size, device=device)
        perm_indices = torch.argsort(rand, dim=1)
        permuted = pool_t[perm_indices]  # (M, pool_size)

        # 2. Split into opponent hands (padded to max hand size)
        hands = torch.full((M, 3, max_hand_size), -1, dtype=torch.int32, device=device)

        start_idx = 0
        for opp_idx, hand_size in enumerate(hand_sizes):
            hands[:, opp_idx, :hand_size] = permuted[:, start_idx:start_idx + hand_size]
            start_idx += hand_size

        # 3. Check void constraints (vectorized) - pass pre-computed can_follow
        valid = _check_void_constraints_gpu_internal(
            hands, hand_sizes, voids, decl_id, device, can_follow
        )

        # 4. Get valid samples
        valid_indices = valid.nonzero(as_tuple=True)[0]
        n_valid = len(valid_indices)

        if n_valid >= n_samples:
            # Success! Extract first n_samples valid worlds
            selected_indices = valid_indices[:n_samples]
            valid_hands = hands[selected_indices].cpu().numpy()

            # Convert to list format
            worlds = []
            for i in range(n_samples):
                world = []
                for opp_idx in range(3):
                    hand_size = hand_sizes[opp_idx]
                    hand = valid_hands[i, opp_idx, :hand_size].tolist()
                    world.append(hand)
                worlds.append(world)

            return worlds

    # Failed to find enough valid samples - fall back to CPU
    from forge.eq.sampling import sample_consistent_worlds

    # Need to map opponent indices (0, 1, 2) to player indices
    # We don't know my_player here, so we need to handle this carefully
    # For now, raise an error and require caller to handle fallback
    raise RuntimeError(
        f"GPU sampling failed to find {n_samples} valid worlds after {max_retries} retries. "
        f"Found only {n_valid}/{M} valid samples. "
        f"This may indicate overly constrained voids. Consider using CPU fallback."
    )


def _check_void_constraints_gpu_internal(
    hands: torch.Tensor,
    hand_sizes: list[int],
    voids: dict[int, set[int]],
    decl_id: int,
    device: str,
    can_follow: torch.Tensor,
) -> torch.Tensor:
    """Internal version that accepts pre-computed can_follow tensor."""
    M = hands.shape[0]
    valid = torch.ones(M, dtype=torch.bool, device=device)

    for opp_idx in range(3):
        void_suits = voids.get(opp_idx, set())
        if not void_suits:
            continue

        hand_size = hand_sizes[opp_idx]
        hand = hands[:, opp_idx, :hand_size]

        for void_suit in void_suits:
            can_follow_mask = can_follow[hand, void_suit, decl_id]
            violates = can_follow_mask.any(dim=1)
            valid &= ~violates

    return valid


def sample_worlds_gpu_with_fallback(
    my_player: int,
    my_hand: list[int],
    played: set[int],
    hand_sizes: list[int],  # [h0, h1, h2, h3]
    voids: dict[int, set[int]],  # {player: {void_suits}}
    decl_id: int,
    n_samples: int,
    device: str = 'cuda',
    rng: np.random.Generator | None = None,
) -> list[list[list[int]]]:
    """Sample worlds with automatic CPU fallback.

    Tries GPU sampling first, falls back to CPU backtracking if needed.

    Args:
        my_player: Which player I am (0-3)
        my_hand: My known dominoes
        played: Set of dominoes already played
        hand_sizes: How many dominoes each player currently has
        voids: Void constraints from infer_voids()
        decl_id: Declaration for checking suit membership
        n_samples: How many consistent worlds to generate
        device: Device for GPU sampling ('cuda' or 'cpu')
        rng: NumPy random generator for CPU fallback

    Returns:
        List of N worlds, each world is [hand0, hand1, hand2, hand3]
        where each hand is a list of domino IDs.
    """
    # Create pool
    all_dominoes = set(range(28))
    pool = all_dominoes - played - set(my_hand)

    # Map player indices to opponent indices (0, 1, 2)
    opponents = [p for p in range(4) if p != my_player]
    opponent_hand_sizes = [hand_sizes[p] for p in opponents]
    opponent_voids = {i: voids.get(opponents[i], set()) for i in range(3)}

    try:
        # Try GPU sampling
        opponent_worlds = sample_worlds_gpu(
            pool=pool,
            hand_sizes=opponent_hand_sizes,
            voids=opponent_voids,
            decl_id=decl_id,
            n_samples=n_samples,
            device=device,
        )

        # Convert back to full 4-player worlds
        worlds = []
        for opp_world in opponent_worlds:
            world = [None] * 4
            world[my_player] = my_hand.copy()
            for i, opp_idx in enumerate(opponents):
                world[opp_idx] = opp_world[i]
            worlds.append(world)

        return worlds

    except RuntimeError:
        # Fall back to CPU backtracking
        from forge.eq.sampling import sample_consistent_worlds

        return sample_consistent_worlds(
            my_player=my_player,
            my_hand=my_hand,
            played=played,
            hand_sizes=hand_sizes,
            voids=voids,
            decl_id=decl_id,
            n_samples=n_samples,
            rng=rng,
        )
