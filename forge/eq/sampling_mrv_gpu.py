"""Uniform, void-consistent world sampling on torch devices.

The public names in this module retain ``MRV`` for caller compatibility.  The
historical implementation was a greedy, vectorized approximation of the CPU
MRV backtracker.  It was neither a backtracker nor a uniform sampler: some
local choices had unequal numbers of legal completions, and a dead end caused
an all-false ``argmax`` to inject domino 0 into the result.

The implementation now counts exact suffix completions over the three
remaining hand capacities.  At each tile it chooses a seat proportional to the
number of legal completions behind that choice.  The resulting policy is
uniform over complete, void-consistent assignments without rejection, even
when valid deals are vanishingly rare under random partitioning.  All dynamic
programming and sampling stays on the requested torch device; there is no CPU
fallback.
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

# Per-device caches of small constant tensors (host->device copies and scalar
# tensor construction inside the assignment loop are one dispatch each).
_DEVICE_CONSTS: dict[str, dict[str, torch.Tensor]] = {}

SAMPLER_ALGORITHM = "uniform-completion-dp-v1"


def _consts(device) -> dict[str, torch.Tensor]:
    key = str(device)
    c = _DEVICE_CONSTS.get(key)
    if c is None:
        c = {
            "suit_domino_mask": SUIT_DOMINO_MASK.to(device),
        }
        _DEVICE_CONSTS[key] = c
    return c


def _build_void_masks_vectorized(
    voids: torch.Tensor,  # [n_games, 3, 8] bool
    decl_ids: torch.Tensor,  # [n_games]
    device: str | torch.device,
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
    suit_domino_mask = _consts(device)["suit_domino_mask"]  # [8, 10]

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
    # Clamp both ends so this helper is safe during preflight validation; the
    # caller still rejects IDs outside -1..27 before sampling.
    pools_clamped = pools.clamp(min=0, max=27).to(torch.int64)

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


def _uniform_below(
    bounds: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one integer uniformly from ``[0, bound)`` per tensor element.

    Completion counts are larger than float32's exact integer range, and even
    float64 ``floor(U * bound)`` is not literally uniform unless ``bound``
    divides 2^53. Draws therefore use rejection from the power-of-two range
    ``[0, 2^62)``. Two candidates are drawn at once to avoid a hot-path host
    synchronization; the second return value marks the astronomically rare
    event that both candidates fell in the rejected tail. Successful draws are
    exactly uniform. All bounds are positive and at most 399,072,960, so the
    per-candidate rejection probability is below 8.7e-11.
    """
    random_range = 1 << 62
    draw_shape = (*bounds.shape, 2)
    if device.type == "mps":
        high = torch.randint(
            0, 1 << 31, draw_shape, dtype=torch.int64, device=device
        )
        low = torch.randint(
            0, 1 << 31, draw_shape, dtype=torch.int64, device=device
        )
        draws = (high << 31) | low
    else:
        draws = torch.randint(
            0, random_range, draw_shape, dtype=torch.int64, device=device
        )

    limit = random_range - torch.remainder(random_range, bounds)
    accepted = draws < limit.unsqueeze(-1)
    first_accepted = accepted.to(torch.int8).argmax(dim=-1, keepdim=True)
    selected = torch.gather(draws, -1, first_accepted).squeeze(-1)
    return torch.remainder(selected, bounds), ~accepted.any(dim=-1)


def _build_suffix_completion_counts(
    pools: torch.Tensor,
    pool_sizes: torch.Tensor,
    void_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact suffix counts for every capacity triple.

    Returns ``(tile_ids, active_tiles, allowed, suffix)`` where
    ``suffix[g, t, r0, r1, r2]`` counts assignments of tiles ``t..end`` that
    exactly fill the three remaining capacities. Counts fit in int64: the
    maximum reachable root is ``21! / (7!^3) = 399,072,960``.
    """

    n_games, pool_width = pools.shape
    device = pools.device
    pool_valid = pools >= 0
    packed_pools = torch.where(
        pool_valid, pools, torch.full_like(pools, 99)
    ).sort(dim=1).values
    positions = torch.arange(pool_width, device=device)
    active_tiles = positions.unsqueeze(0) < pool_sizes.unsqueeze(1)
    tile_ids = packed_pools.clamp(min=0, max=27)
    tile_bits = 1 << tile_ids
    allowed = (
        (tile_bits.unsqueeze(2) & void_masks.unsqueeze(1)) == 0
    ) & active_tiles.unsqueeze(2)

    suffix = torch.zeros(
        n_games,
        pool_width + 1,
        8,
        8,
        8,
        dtype=torch.int64,
        device=device,
    )
    suffix[:, pool_width, 0, 0, 0] = 1
    for position in range(pool_width - 1, -1, -1):
        next_counts = suffix[:, position + 1]
        assign_0 = torch.zeros_like(next_counts)
        assign_1 = torch.zeros_like(next_counts)
        assign_2 = torch.zeros_like(next_counts)
        assign_0[:, 1:, :, :] = next_counts[:, :-1, :, :]
        assign_1[:, :, 1:, :] = next_counts[:, :, :-1, :]
        assign_2[:, :, :, 1:] = next_counts[:, :, :, :-1]
        completion_counts = (
            assign_0 * allowed[:, position, 0].view(n_games, 1, 1, 1)
            + assign_1 * allowed[:, position, 1].view(n_games, 1, 1, 1)
            + assign_2 * allowed[:, position, 2].view(n_games, 1, 1, 1)
        )
        suffix[:, position] = torch.where(
            active_tiles[:, position].view(n_games, 1, 1, 1),
            completion_counts,
            next_counts,
        )

    return tile_ids, active_tiles, allowed, suffix


def sample_worlds_mrv_gpu(
    pools: torch.Tensor,           # [n_games, pool_size] available dominoes
    hand_sizes: torch.Tensor,      # [n_games, 3] opponent hand sizes
    voids: torch.Tensor,           # [n_games, 3, 8] void flags per opponent
    decl_ids: torch.Tensor,        # [n_games] declaration IDs
    n_samples: int = 50,
    device: str | torch.device = 'cuda',
    max_pool_size: int | None = None,
) -> torch.Tensor:
    """Sample uniformly from worlds consistent with public void constraints.

    ``MRV`` remains in the function name for API compatibility.  A suffix
    dynamic program counts legal completions for every remaining-capacity
    triple.  Each tile's seat is then sampled in proportion to the exact count
    behind that choice, producing a uniform complete assignment directly.

    Args:
        pools: [n_games, pool_size] available domino IDs (padded with -1)
        hand_sizes: [n_games, 3] hand sizes for 3 opponents per game
        voids: [n_games, 3, 8] bool - voids[g,o,s] = opponent o is void in suit s
        decl_ids: [n_games] declaration ID per game
        n_samples: Number of worlds to sample per game
        device: Torch device. CUDA unavailability is an error; explicit CPU is
            supported for focused tests only.
        max_pool_size: Optional caller-computed largest pool.  Retained for API
            compatibility and checked against the inputs when supplied.
    Returns:
        [n_games, n_samples, 3, 7] opponent hands, descending within each
        hand and padded with -1.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested_device} requested but CUDA is unavailable; "
            "the production sampler has no CPU fallback"
        )
    if requested_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"MPS device {requested_device} requested but MPS is unavailable")

    if pools.ndim != 2:
        raise ValueError(f"pools must have shape [games, pool_width], got {tuple(pools.shape)}")
    n_games, pool_width = pools.shape
    if n_games <= 0:
        raise ValueError("at least one game is required")
    if hand_sizes.shape != (n_games, 3):
        raise ValueError(
            f"hand_sizes must have shape {(n_games, 3)}, got {tuple(hand_sizes.shape)}"
        )
    if voids.shape != (n_games, 3, 8):
        raise ValueError(
            f"voids must have shape {(n_games, 3, 8)}, got {tuple(voids.shape)}"
        )
    if decl_ids.shape != (n_games,):
        raise ValueError(
            f"decl_ids must have shape {(n_games,)}, got {tuple(decl_ids.shape)}"
        )
    if pool_width > 28:
        raise ValueError(f"pool width cannot exceed 28, got {pool_width}")

    # All tensor work stays on the explicitly requested device. CPU remains
    # useful for deterministic focused tests only.
    pools = pools.to(device=requested_device, dtype=torch.int64)
    hand_sizes = hand_sizes.to(device=requested_device, dtype=torch.int64)
    voids = voids.to(device=requested_device, dtype=torch.bool)
    decl_ids = decl_ids.to(device=requested_device, dtype=torch.int64)

    invalid_pool_ids = ((pools < -1) | (pools > 27)).any(dim=1)
    pool_valid = pools >= 0
    pool_sizes = pool_valid.sum(dim=1, dtype=torch.int64)
    pool_masks = _pool_to_mask(pools)
    unique_pool_sizes = _popcount_vectorized(pool_masks).to(torch.int64)
    hand_totals = hand_sizes.sum(dim=1)

    invalid_shape = (
        (unique_pool_sizes != pool_sizes)
        | (hand_totals != pool_sizes)
        | (hand_sizes < 0).any(dim=1)
        | (hand_sizes > 7).any(dim=1)
        | (decl_ids < 0)
        | (decl_ids >= SUIT_DOMINO_MASK.shape[1])
    )
    hint_mismatch = (
        pool_sizes.max() != max_pool_size
        if max_pool_size is not None
        else torch.tensor(False, device=requested_device)
    )

    void_masks = _build_void_masks_vectorized(
        voids, decl_ids.clamp(0, SUIT_DOMINO_MASK.shape[1] - 1), requested_device
    )  # [games, 3]
    candidate_masks = pool_masks.unsqueeze(1) & ~void_masks

    tile_ids, active_tiles, allowed, suffix = _build_suffix_completion_counts(
        pools, pool_sizes, void_masks
    )

    game_indices = torch.arange(n_games, device=requested_device)
    safe_hand_sizes = hand_sizes.clamp(min=0, max=7)
    root_counts = suffix[
        game_indices,
        0,
        safe_hand_sizes[:, 0],
        safe_hand_sizes[:, 1],
        safe_hand_sizes[:, 2],
    ]
    infeasible = root_counts == 0

    # One device synchronization covers the full preflight.  Detailed tensor
    # transfers happen only on the exceptional path.
    preflight = torch.stack(
        [
            invalid_pool_ids.any(),
            invalid_shape.any(),
            infeasible.any(),
            hint_mismatch,
        ]
    ).detach().cpu().tolist()
    if any(preflight):
        if preflight[0]:
            bad = pools[(pools < -1) | (pools > 27)].detach().cpu().tolist()
            raise ValueError(f"pool domino IDs must be -1 or 0..27, got {bad[:8]}")
        if preflight[1]:
            bad_games = invalid_shape.nonzero(as_tuple=True)[0].detach().cpu().tolist()
            raise ValueError(
                "invalid sampler inputs for game indices "
                f"{bad_games}: pool_sizes={pool_sizes[bad_games].detach().cpu().tolist()}, "
                f"unique_pool_sizes={unique_pool_sizes[bad_games].detach().cpu().tolist()}, "
                f"hand_sizes={hand_sizes[bad_games].detach().cpu().tolist()}, "
                f"decl_ids={decl_ids[bad_games].detach().cpu().tolist()}"
            )
        if preflight[3]:
            actual_max_pool_size = int(pool_sizes.max().item())
            raise ValueError(
                f"max_pool_size hint {max_pool_size} does not match actual "
                f"maximum {actual_max_pool_size}"
            )

        bad_games = infeasible.nonzero(as_tuple=True)[0].detach().cpu().tolist()
        candidate_counts = _popcount_vectorized(candidate_masks).detach().cpu()
        raise ValueError(
            "no void-consistent hand assignment exists for game indices "
            f"{bad_games}: hand_sizes={hand_sizes[bad_games].detach().cpu().tolist()}, "
            f"candidate_counts={candidate_counts[bad_games].tolist()}, "
            f"root_completion_counts={root_counts[bad_games].detach().cpu().tolist()}"
        )

    if pool_width == 0:
        return torch.full(
            (n_games, n_samples, 3, 7),
            -1,
            dtype=torch.int32,
            device=requested_device,
        )

    # Sample each tile's owner using the exact suffix completion counts. The
    # product of conditional probabilities telescopes to 1/root_count for
    # every complete assignment.
    remaining = hand_sizes.unsqueeze(1).expand(
        n_games, n_samples, 3
    ).clone()
    hand_masks = torch.zeros(
        n_games, n_samples, 3, dtype=torch.int64, device=requested_device
    )
    game_grid = game_indices.unsqueeze(1).expand(n_games, n_samples)
    seat_ids = torch.arange(3, device=requested_device).view(1, 1, 3)
    dead_end = torch.zeros(
        n_games, n_samples, dtype=torch.bool, device=requested_device
    )
    random_exhausted = torch.zeros_like(dead_end)

    for position in range(pool_width):
        active = active_tiles[:, position].unsqueeze(1)
        next_suffix = suffix[:, position + 1]
        branch_counts: list[torch.Tensor] = []
        remaining_0 = remaining[:, :, 0]
        remaining_1 = remaining[:, :, 1]
        remaining_2 = remaining[:, :, 2]
        for player in range(3):
            eligible = (
                active
                & allowed[:, position, player].unsqueeze(1)
                & (remaining[:, :, player] > 0)
            )
            counts = next_suffix[
                game_grid,
                (remaining_0 - int(player == 0)).clamp(min=0, max=7),
                (remaining_1 - int(player == 1)).clamp(min=0, max=7),
                (remaining_2 - int(player == 2)).clamp(min=0, max=7),
            ]
            branch_counts.append(counts * eligible.to(torch.int64))

        count_0, count_1, count_2 = branch_counts
        total_counts = count_0 + count_1 + count_2
        dead_end |= active & (total_counts == 0)
        safe_totals = total_counts.clamp_min(1)
        target, draw_exhausted = _uniform_below(safe_totals, requested_device)
        random_exhausted |= active & draw_exhausted
        selected = (
            (target >= count_0).to(torch.int64)
            + (target >= count_0 + count_1).to(torch.int64)
        )

        selected_seat = seat_ids == selected.unsqueeze(2)
        selected_seat &= active.unsqueeze(2)
        domino_bit = (1 << tile_ids[:, position]).view(n_games, 1, 1)
        hand_masks |= selected_seat.to(torch.int64) * domino_bit
        remaining.scatter_add_(
            2,
            selected.unsqueeze(2),
            -active.expand(n_games, n_samples).to(torch.int64).unsqueeze(2),
        )

    invalid_sample = dead_end.any() | random_exhausted.any() | (remaining != 0).any()
    if bool(invalid_sample.item()):
        causes = []
        if bool(dead_end.any().item()):
            causes.append("zero completion count")
        if bool(random_exhausted.any().item()):
            causes.append("two rejected 62-bit draws")
        if bool((remaining != 0).any().item()):
            causes.append("unfilled hand capacity")
        raise RuntimeError(
            f"{SAMPLER_ALGORITHM} internal sampling failure: {', '.join(causes)}; "
            "no partial world was returned"
        )

    bit_indices = torch.arange(28, device=requested_device, dtype=torch.int64)
    bit_values = 1 << bit_indices
    bits_set = (hand_masks.unsqueeze(3) & bit_values.view(1, 1, 1, 28)) != 0
    values = torch.where(
        bits_set,
        bit_indices.to(torch.int32).view(1, 1, 1, 28),
        torch.tensor(-1, dtype=torch.int32, device=requested_device),
    )
    return values.sort(dim=3, descending=True).values[:, :, :, :7]


class WorldSamplerMRV:
    """Stateful uniform world sampler for torch batch processing.

    The legacy class name is retained as a drop-in API for existing callers.
    Every successful call returns a uniform sample of valid partitions. Exact
    completion counting makes sampling independent of rejection acceptance.

    Example:
        >>> sampler = WorldSamplerMRV(max_games=32, max_samples=100, device='cuda')
        >>> worlds = sampler.sample(pools, hand_sizes, voids, decl_ids, n_samples=50)
    """

    algorithm = SAMPLER_ALGORITHM

    def __init__(
        self,
        max_games: int,
        max_samples: int,
        device: str | torch.device = 'cuda',
    ):
        """Initialize sampler.

        Args:
            max_games: Maximum number of games to process
            max_samples: Maximum samples per game
            device: Torch device. CPU must be requested explicitly.
        """
        if max_games <= 0:
            raise ValueError(f"max_games must be positive, got {max_games}")
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {requested_device} requested but CUDA is unavailable; "
                "the production sampler has no CPU fallback"
            )
        if requested_device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                f"MPS device {requested_device} requested but MPS is unavailable"
            )

        # Preserve the historical public attribute type for callers that use
        # it in manifests or string comparisons.
        self.device = str(requested_device)
        self.max_games = max_games
        self.max_samples = max_samples

    def sample(
        self,
        pools: torch.Tensor,           # [n_games, pool_size]
        hand_sizes: torch.Tensor,      # [n_games, 3]
        voids: torch.Tensor,           # [n_games, 3, 8]
        decl_ids: torch.Tensor,        # [n_games]
        n_samples: int = 50,
        max_pool_size: int | None = None,
    ) -> torch.Tensor:
        """Sample uniform, void-consistent worlds or raise diagnostics.

        Args:
            pools: [n_games, pool_size] available domino IDs (padded with -1)
            hand_sizes: [n_games, 3] hand sizes for 3 opponents per game
            voids: [n_games, 3, 8] bool - voids[g,o,s] = opponent o is void in suit s
            decl_ids: [n_games] declaration ID per game
            n_samples: Number of worlds to sample per game
            max_pool_size: Optional caller-computed largest pool, checked
                against the inputs.

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
            max_pool_size=max_pool_size,
        )
