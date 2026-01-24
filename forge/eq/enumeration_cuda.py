"""GPU-native world enumeration via Numba CUDA kernels.

Eliminates Python loop overhead for H100-scale enumeration (1000+ games) by
running enumeration entirely on GPU using Numba CUDA.

## Thread Mapping
```
blockIdx.x  = game index (0 to n_games-1)
threadIdx.x = world index (0 to max_worlds_per_game-1)
```
H100: 1000 games × 1000 worlds = 1M threads (fully saturates GPU)

## Algorithm: Early-Exit Combinadic Enumeration

Each thread computes one potential world by:
1. Decompose world index into combination indices (c0, c1, c2)
2. Unrank c0 to get combo0 for opp0, check validity (early exit if invalid)
3. Unrank c1 from remaining pool for opp1, check validity
4. Unrank c2 from remaining pool for opp2, check validity
5. If all valid, write output using atomic counter

## Combinadic Unranking (Co-Lexicographic Order)

Convert a rank r to a k-combination from n elements:
- Find the unique combination S where rank = sum(C(S[i], i+1))
- Uses pre-computed binomial coefficient table for O(k) per unrank
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

# Try importing Numba CUDA - graceful fallback if unavailable
try:
    from numba import cuda
    import numpy as np

    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    cuda = None
    CUDA_AVAILABLE = False

if TYPE_CHECKING:
    import numpy as np


# =============================================================================
# Binomial Coefficient Table
# =============================================================================

# Maximum values for enumeration
MAX_N = 22  # Maximum pool size (28 - 7 hand + padding)
MAX_K = 8   # Maximum slot size (7 + 1)


def _build_binom_table() -> "np.ndarray":
    """Build binomial coefficient table C(n, k) for CUDA kernel.

    Returns:
        [MAX_N, MAX_K] int32 numpy array where binom[n, k] = C(n, k)
    """
    import numpy as np

    binom = np.zeros((MAX_N, MAX_K), dtype=np.int32)
    for n in range(MAX_N):
        for k in range(min(n + 1, MAX_K)):
            binom[n, k] = math.comb(n, k)
    return binom


# Module-level table (built lazily on first CUDA use)
_BINOM_TABLE_HOST: "np.ndarray | None" = None
_BINOM_TABLE_DEVICE = None


def _get_binom_table_device():
    """Get binomial table on CUDA device (lazy initialization)."""
    global _BINOM_TABLE_HOST, _BINOM_TABLE_DEVICE

    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available")

    if _BINOM_TABLE_HOST is None:
        _BINOM_TABLE_HOST = _build_binom_table()

    if _BINOM_TABLE_DEVICE is None:
        _BINOM_TABLE_DEVICE = cuda.to_device(_BINOM_TABLE_HOST)

    return _BINOM_TABLE_DEVICE


# =============================================================================
# CUDA Device Functions
# =============================================================================

if CUDA_AVAILABLE:
    @cuda.jit(device=True)
    def _unrank_colex(rank: int, k: int, n: int, binom_table, result):
        """Unrank a combination in co-lexicographic order.

        Converts rank to k-combination from n elements. The combination is
        written to `result` array (must have length >= k).

        Co-lex ordering: combinations are ordered by their elements from
        smallest to largest, then by position. For C(5,2):
        rank 0 = {0,1}, rank 1 = {0,2}, rank 2 = {1,2}, rank 3 = {0,3}, ...

        Args:
            rank: The rank (0-indexed) of the combination
            k: Size of combination
            n: Size of universe (unused but kept for API consistency)
            binom_table: [MAX_N, MAX_K] binomial coefficients
            result: Output array to write combination (size >= k)
        """
        m = rank
        for i in range(k - 1, -1, -1):
            # Find largest ell where C(ell, i+1) <= m
            ell = i
            while ell + 1 < MAX_N and binom_table[ell + 1, i + 1] <= m:
                ell += 1
            result[i] = ell
            m -= binom_table[ell, i + 1]

    @cuda.jit(device=True)
    def _count_less_than(value: int, excluded, n_excluded: int) -> int:
        """Count how many values in excluded are less than value."""
        count = 0
        for i in range(n_excluded):
            if excluded[i] < value:
                count += 1
        return count

    @cuda.jit(device=True)
    def _map_to_remaining(combo_idx: int, excluded, n_excluded: int) -> int:
        """Map a combination index to actual pool index, skipping excluded.

        If excluded = [2, 5] and combo_idx = 3, we want the 4th element
        from [0,1,3,4,6,7,...]. This maps 3 -> 4 (skipping 2).

        Args:
            combo_idx: Index in the remaining pool (after exclusions)
            excluded: Array of excluded indices (sorted)
            n_excluded: Number of excluded indices

        Returns:
            Actual pool index
        """
        actual = combo_idx
        for _ in range(n_excluded + 1):  # Iterate enough times
            # Count how many excluded values are <= actual
            offset = 0
            for i in range(n_excluded):
                if excluded[i] <= actual:
                    offset += 1
            new_actual = combo_idx + offset
            if new_actual == actual:
                break
            actual = new_actual
        return actual

    # =============================================================================
    # Main CUDA Kernel
    # =============================================================================

    @cuda.jit
    def enumerate_worlds_kernel(
        pools,           # [n_games, max_pool] int8 - domino IDs
        pool_sizes,      # [n_games] int32 - actual pool size
        slots,           # [n_games, 3] int32 - slots needed per opponent
        can_assign,      # [n_games, 28, 3] bool - can domino be assigned to opp?
        known,           # [n_games, 3, 7] int8 - known dominoes
        known_counts,    # [n_games, 3] int32 - count of known per opp
        binom_table,     # [MAX_N, MAX_K] int32
        output,          # [n_games, max_worlds, 3, 7] int32 - output hands
        counts,          # [n_games] int32 - atomic output counts
        max_worlds,      # int - max worlds per game
    ):
        """CUDA kernel for world enumeration with early-exit void filtering.

        Thread mapping:
            blockIdx.x = game index
            threadIdx.x = world index within game

        Each thread attempts to construct one world. If void constraints
        make the world invalid, the thread exits early without writing.
        """
        g = cuda.blockIdx.x       # game index
        w = cuda.threadIdx.x      # world index

        # Load game parameters
        pool_size = pool_sizes[g]
        s0 = slots[g, 0]
        s1 = slots[g, 1]
        s2 = slots[g, 2]

        # Compute total worlds for this game: C(pool, s0) * C(pool-s0, s1) * C(pool-s0-s1, s2)
        c0_count = binom_table[pool_size, s0] if pool_size < MAX_N and s0 < MAX_K else 0
        c1_count = binom_table[pool_size - s0, s1] if pool_size - s0 < MAX_N and s1 < MAX_K else 0
        c2_count = binom_table[pool_size - s0 - s1, s2] if pool_size - s0 - s1 < MAX_N and s2 < MAX_K else 0

        total_worlds = c0_count * c1_count * c2_count
        if total_worlds == 0:
            total_worlds = 1  # At least 1 world if all slots are 0

        # Early exit if this thread's world index is out of range
        if w >= total_worlds or w >= max_worlds:
            return

        # Compute strides for decomposing world index
        stride1 = c2_count if c2_count > 0 else 1
        stride0 = c1_count * stride1 if c1_count > 0 else stride1

        # Decompose world index into combination indices
        c0_idx = w // stride0
        c1_idx = (w % stride0) // stride1
        c2_idx = w % stride1

        # Local arrays for combination results (max 7 elements each)
        combo0 = cuda.local.array(7, dtype=cuda.int32)
        combo1 = cuda.local.array(7, dtype=cuda.int32)
        combo2 = cuda.local.array(7, dtype=cuda.int32)

        # Track excluded indices for subsequent unranking
        excluded01 = cuda.local.array(14, dtype=cuda.int32)  # combo0 + combo1

        # ---------- Unrank combo0 for opponent 0 ----------
        if s0 > 0:
            _unrank_colex(c0_idx, s0, pool_size, binom_table, combo0)

            # Check validity: can each domino be assigned to opponent 0?
            for i in range(s0):
                pool_idx = combo0[i]
                if pool_idx >= pool_size:
                    return  # Invalid index
                domino_id = pools[g, pool_idx]
                if domino_id < 0 or domino_id >= 28:
                    return  # Invalid domino
                if not can_assign[g, domino_id, 0]:
                    return  # Void constraint violation - early exit

        # ---------- Unrank combo1 for opponent 1 ----------
        if s1 > 0:
            # First unrank in the reduced space [0, pool_size - s0)
            _unrank_colex(c1_idx, s1, pool_size - s0, binom_table, combo1)

            # Map indices to actual pool, skipping combo0
            for i in range(s1):
                raw_idx = combo1[i]
                # Map to actual pool index, skipping indices in combo0
                actual_idx = _map_to_remaining(raw_idx, combo0, s0)
                combo1[i] = actual_idx

                if actual_idx >= pool_size:
                    return  # Invalid index
                domino_id = pools[g, actual_idx]
                if domino_id < 0 or domino_id >= 28:
                    return
                if not can_assign[g, domino_id, 1]:
                    return  # Void constraint violation - early exit

        # Build excluded01 = combo0 + combo1
        n_excluded = 0
        for i in range(s0):
            excluded01[n_excluded] = combo0[i]
            n_excluded += 1
        for i in range(s1):
            excluded01[n_excluded] = combo1[i]
            n_excluded += 1

        # Simple insertion sort for excluded01 (small array)
        for i in range(1, n_excluded):
            key = excluded01[i]
            j = i - 1
            while j >= 0 and excluded01[j] > key:
                excluded01[j + 1] = excluded01[j]
                j -= 1
            excluded01[j + 1] = key

        # ---------- Unrank combo2 for opponent 2 ----------
        if s2 > 0:
            _unrank_colex(c2_idx, s2, pool_size - s0 - s1, binom_table, combo2)

            for i in range(s2):
                raw_idx = combo2[i]
                actual_idx = _map_to_remaining(raw_idx, excluded01, n_excluded)
                combo2[i] = actual_idx

                if actual_idx >= pool_size:
                    return
                domino_id = pools[g, actual_idx]
                if domino_id < 0 or domino_id >= 28:
                    return
                if not can_assign[g, domino_id, 2]:
                    return  # Void constraint violation - early exit

        # ---------- All valid - write output ----------
        output_idx = cuda.atomic.add(counts, g, 1)
        if output_idx >= max_worlds:
            return  # Exceeded max_worlds

        # Initialize output row to -1
        for opp in range(3):
            for i in range(7):
                output[g, output_idx, opp, i] = -1

        # Write known dominoes first
        for opp in range(3):
            kc = known_counts[g, opp]
            for i in range(kc):
                if i < 7:
                    output[g, output_idx, opp, i] = known[g, opp, i]

        # Write unknown dominoes (from combos)
        # Opponent 0
        kc0 = known_counts[g, 0]
        for i in range(s0):
            if kc0 + i < 7:
                pool_idx = combo0[i]
                output[g, output_idx, 0, kc0 + i] = pools[g, pool_idx]

        # Opponent 1
        kc1 = known_counts[g, 1]
        for i in range(s1):
            if kc1 + i < 7:
                pool_idx = combo1[i]
                output[g, output_idx, 1, kc1 + i] = pools[g, pool_idx]

        # Opponent 2
        kc2 = known_counts[g, 2]
        for i in range(s2):
            if kc2 + i < 7:
                pool_idx = combo2[i]
                output[g, output_idx, 2, kc2 + i] = pools[g, pool_idx]


# =============================================================================
# Host-Side Functions
# =============================================================================

def _build_can_assign_mask(
    voids: Tensor,  # [n_games, 3, 8] bool - void suits per opponent
    decl_ids: Tensor,  # [n_games]
    device: str,
) -> Tensor:
    """Build validity mask for domino-opponent assignments.

    A domino cannot be assigned to an opponent if:
    - Opponent is void in a suit S, AND
    - The domino can follow suit S (given the declaration)

    Args:
        voids: [n_games, 3, 8] bool - True if opponent is void in suit
        decl_ids: [n_games] declaration IDs
        device: torch device

    Returns:
        [n_games, 28, 3] bool - True if domino can be assigned to opponent
    """
    from forge.eq.sampling_mrv_gpu import SUIT_DOMINO_MASK

    n_games = voids.shape[0]
    suit_domino_mask = SUIT_DOMINO_MASK.to(device)  # [8, 10] int64 bitmasks

    # Build per-game, per-opponent void violation mask
    # void_violates[g, opp, d] = True if d violates opp's void constraints

    # For each game, get the relevant suit_domino_mask column
    # suit_domino_mask[:, decl_id] gives [8] bitmasks for each suit
    decl_indices = decl_ids.long()  # [n_games]
    masks_per_suit = suit_domino_mask[:, decl_indices]  # [8, n_games]

    # Expand for broadcasting: [8, n_games, 1] for OR with voids [n_games, 3, 8]
    masks_per_suit = masks_per_suit.permute(1, 0).unsqueeze(1)  # [n_games, 1, 8]

    # For each (game, opp), compute bitmask of violating dominoes
    # voids[g, opp, s] = True means opponent opp in game g is void in suit s
    # If void in s, any domino that can follow s is a violation

    # Expand voids for element-wise multiply: [n_games, 3, 8]
    # masks_per_suit: [n_games, 1, 8] - bitmask of dominoes that can follow each suit

    # Compute violation bitmask per (game, opp)
    violation_bitmasks = torch.zeros(n_games, 3, dtype=torch.int64, device=device)
    for s in range(8):
        # voids[:, :, s]: [n_games, 3] bool
        # masks_per_suit[:, 0, s]: [n_games] int64 bitmask
        void_in_s = voids[:, :, s].to(torch.int64)  # [n_games, 3]
        mask_s = masks_per_suit[:, 0, s].unsqueeze(1)  # [n_games, 1]
        violation_bitmasks |= void_in_s * mask_s

    # Convert bitmask to [n_games, 28, 3] bool
    # can_assign[g, d, opp] = not (bit d set in violation_bitmasks[g, opp])
    bit_positions = (1 << torch.arange(28, device=device, dtype=torch.int64)).view(1, 28, 1)
    violation_expanded = violation_bitmasks.unsqueeze(1)  # [n_games, 1, 3]
    violates = (violation_expanded & bit_positions) != 0  # [n_games, 28, 3]
    can_assign = ~violates

    return can_assign


def enumerate_worlds_cuda(
    pools: Tensor,  # [n_games, max_pool] int8 - domino IDs
    pool_sizes: Tensor,  # [n_games] int32
    known: Tensor,  # [n_games, 3, 7] int8 - known dominoes
    known_counts: Tensor,  # [n_games, 3] int32
    slot_sizes: Tensor,  # [n_games, 3] int32
    voids: Tensor,  # [n_games, 3, 8] bool
    decl_ids: Tensor,  # [n_games]
    max_worlds: int = 100_000,
) -> tuple[Tensor, Tensor]:
    """Enumerate valid worlds using CUDA kernel.

    This is the main entry point for GPU-native enumeration. It launches
    one CUDA block per game, with threads enumerating potential worlds
    in parallel.

    Args:
        pools: [n_games, max_pool] available domino IDs, -1 padded
        pool_sizes: [n_games] actual pool size per game
        known: [n_games, 3, 7] known dominoes per opponent, -1 padded
        known_counts: [n_games, 3] count of known dominoes
        slot_sizes: [n_games, 3] unknown slots needed per opponent
        voids: [n_games, 3, 8] bool - void suits
        decl_ids: [n_games] declaration IDs
        max_worlds: Maximum worlds to enumerate per game

    Returns:
        Tuple of (worlds, counts):
            - worlds: [n_games, max_worlds, 3, 7] opponent hands, -1 padded
            - counts: [n_games] actual world count per game

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available for enumeration_cuda")

    import numpy as np

    n_games = pools.shape[0]
    device = pools.device

    # Build can_assign mask (vectorized on GPU)
    can_assign = _build_can_assign_mask(voids, decl_ids, device)  # [n_games, 28, 3]

    # Get binomial table on device
    binom_table = _get_binom_table_device()

    # Allocate output tensors
    worlds = torch.full(
        (n_games, max_worlds, 3, 7), -1,
        dtype=torch.int32, device=device
    )
    counts = torch.zeros(n_games, dtype=torch.int32, device=device)

    # Convert tensors to Numba CUDA arrays
    # Numba needs contiguous arrays
    pools_np = pools.cpu().numpy().astype(np.int8)
    pool_sizes_np = pool_sizes.cpu().numpy().astype(np.int32)
    slot_sizes_np = slot_sizes.cpu().numpy().astype(np.int32)
    can_assign_np = can_assign.cpu().numpy()
    known_np = known.cpu().numpy().astype(np.int8)
    known_counts_np = known_counts.cpu().numpy().astype(np.int32)

    d_pools = cuda.to_device(pools_np)
    d_pool_sizes = cuda.to_device(pool_sizes_np)
    d_slots = cuda.to_device(slot_sizes_np)
    d_can_assign = cuda.to_device(can_assign_np)
    d_known = cuda.to_device(known_np)
    d_known_counts = cuda.to_device(known_counts_np)

    # Output arrays (will copy back to torch)
    d_output = cuda.to_device(np.full((n_games, max_worlds, 3, 7), -1, dtype=np.int32))
    d_counts = cuda.to_device(np.zeros(n_games, dtype=np.int32))

    # Compute max threads needed per block
    # Each block handles one game, threads handle worlds
    # Use min(max_worlds, 1024) threads per block (CUDA limit is 1024)
    threads_per_block = min(max_worlds, 1024)

    # Launch kernel: n_games blocks, threads_per_block threads each
    enumerate_worlds_kernel[n_games, threads_per_block](
        d_pools, d_pool_sizes, d_slots, d_can_assign,
        d_known, d_known_counts, binom_table,
        d_output, d_counts, max_worlds
    )

    # Copy results back to torch tensors
    output_np = d_output.copy_to_host()
    counts_np = d_counts.copy_to_host()

    worlds = torch.from_numpy(output_np).to(device)
    counts = torch.from_numpy(counts_np).to(device)

    return worlds, counts
