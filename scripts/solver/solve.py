"""
GPU solver for Texas 42.

Three-phase solver:
1. enumerate_gpu: BFS from initial state to find all reachable states
2. build_child_index: Map each move to child's position in sorted array
3. solve_gpu: Backward induction to compute minimax values
"""

import torch
from state import pack_state, compute_level, compute_team, compute_terminal_value
from context import SeedContext, build_context
from expand import expand_gpu


def pack_initial_state(ctx: SeedContext) -> int:
    """
    Create the initial state with all players having full hands.

    Returns:
        int64 packed initial state
    """
    remaining = torch.full((1, 4), 0b1111111, dtype=torch.int64)  # All 7 local indices
    score = torch.zeros(1, dtype=torch.int64)
    # Player 1 leads first (player left of dealer, matching TS behavior)
    leader = torch.ones(1, dtype=torch.int64)
    trick_len = torch.zeros(1, dtype=torch.int64)
    p0 = torch.full((1,), 7, dtype=torch.int64)  # 7 = no play yet
    p1 = torch.full((1,), 7, dtype=torch.int64)
    p2 = torch.full((1,), 7, dtype=torch.int64)
    return pack_state(remaining, score, leader, trick_len, p0, p1, p2)[0].item()


def enumerate_gpu(ctx: SeedContext, device: torch.device = None) -> torch.Tensor:
    """
    BFS from initial state, return sorted tensor of all reachable states.

    Memory-efficient implementation: accumulates states incrementally
    and frees intermediate tensors as we go.

    Args:
        ctx: SeedContext with precomputed tables
        device: Target device (defaults to CPU)

    Returns:
        Sorted (M,) int64 tensor of all reachable states
    """
    import gc

    if device is None:
        device = torch.device('cpu')

    initial = pack_initial_state(ctx)
    frontier = torch.tensor([initial], dtype=torch.int64, device=device)

    # Accumulate all states in a single growing tensor
    all_states = frontier.clone()

    # BFS: level goes from 28 down to 1
    for level in range(28, 0, -1):
        if frontier.numel() == 0:
            break

        # Expand in chunks if frontier is very large to limit peak memory
        chunk_size = 100000
        if frontier.numel() > chunk_size:
            all_children = []
            for i in range(0, frontier.numel(), chunk_size):
                chunk = frontier[i:i + chunk_size]
                children = expand_gpu(chunk, ctx)
                children = children.flatten()
                children = children[children >= 0]
                if children.numel() > 0:
                    all_children.append(children)
                del chunk
            if all_children:
                children = torch.cat(all_children)
                del all_children
            else:
                children = torch.tensor([], dtype=torch.int64, device=device)
        else:
            children = expand_gpu(frontier, ctx)
            children = children.flatten()
            children = children[children >= 0]

        # Dedup children
        children = torch.unique(children)

        if children.numel() > 0:
            # Merge with existing states
            all_states = torch.cat([all_states, children])
            # Dedup periodically to keep memory bounded
            if all_states.numel() > 5000000:  # 5M threshold
                all_states = torch.unique(all_states)

        # Move to next level
        del frontier
        frontier = children
        gc.collect()

    # Final sort and dedup
    all_states = torch.unique(all_states)
    return torch.sort(all_states).values


def build_child_index(all_states: torch.Tensor, ctx: SeedContext) -> torch.Tensor:
    """
    Build (N, 7) index mapping each move to child's position.

    Args:
        all_states: Sorted (N,) int64 tensor of all states
        ctx: SeedContext with precomputed tables

    Returns:
        (N, 7) int64 tensor where result[i, m] is the index of state i's
        child after move m, or -1 if move is illegal

    Raises:
        ValueError: If any legal child is not found in all_states
    """
    N = all_states.shape[0]
    device = all_states.device

    children = expand_gpu(all_states, ctx)  # (N, 7)

    # Use searchsorted to find where each child would be
    # clamp to valid range for indexing
    child_idx = torch.searchsorted(all_states, children.clamp(min=0))

    # CRITICAL: Verify found indices actually match
    # This catches silent corruption from missing states
    valid_idx = child_idx.clamp(0, N - 1)
    found_states = all_states[valid_idx]
    child_exists = (children >= 0) & (found_states == children)

    # Check for mismatches (legal moves that don't map to enumerated states)
    legal_but_missing = (~child_exists) & (children >= 0)
    if legal_but_missing.any():
        mismatch = legal_but_missing.nonzero()
        first_mismatch = mismatch[0]
        state_idx, move_idx = first_mismatch[0].item(), first_mismatch[1].item()
        missing_child = children[state_idx, move_idx].item()
        parent_state = all_states[state_idx].item()
        raise ValueError(
            f"Child not in enumeration: parent={parent_state}, move={move_idx}, "
            f"child={missing_child}"
        )

    return torch.where(children >= 0, child_idx, torch.tensor(-1, device=device, dtype=torch.int64))


def solve_gpu(
    all_states: torch.Tensor,
    child_idx: torch.Tensor,
    ctx: SeedContext
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward induction to compute minimax values.

    Args:
        all_states: Sorted (N,) int64 tensor of all states
        child_idx: (N, 7) int64 child indices from build_child_index
        ctx: SeedContext (unused but kept for consistency)

    Returns:
        (V, move_values) where:
        - V: (N,) int8 minimax value for each state
        - move_values: (N, 7) int8 value of each move (-128 for illegal)
    """
    N = all_states.shape[0]
    device = all_states.device

    V = torch.zeros(N, dtype=torch.int8, device=device)
    move_values = torch.full((N, 7), -128, dtype=torch.int8, device=device)

    level_of = compute_level(all_states)  # (N,)
    is_team0 = compute_team(all_states)  # (N,) bool

    # Terminal states (level 0) - all dominoes played
    terminal = (level_of == 0)
    V[terminal] = compute_terminal_value(all_states[terminal])

    # Backward pass from level 1 to 28
    for level in range(1, 29):
        mask = (level_of == level)
        if not mask.any():
            continue

        idx = mask.nonzero(as_tuple=True)[0]  # indices of states at this level
        K = idx.shape[0]

        # Get child indices for these states
        cidx = child_idx[idx]  # (K, 7)
        legal = (cidx >= 0)  # (K, 7)

        # Clamp for safe indexing (illegal moves will be masked anyway)
        cidx_safe = cidx.clamp(min=0)

        # Get child values
        cv = V[cidx_safe]  # (K, 7)

        # Store move values for legal moves
        move_values[idx] = torch.where(
            legal,
            cv.to(torch.int8),
            torch.tensor(-128, dtype=torch.int8, device=device)
        )

        # Compute minimax: max for team 0, min for team 1
        # Use extreme values for illegal moves so they don't affect min/max
        is_team0_k = is_team0[idx]  # (K,)

        # For max (team 0): illegal moves should be -128 (won't be chosen)
        # For min (team 1): illegal moves should be +127 (won't be chosen)
        cv_for_max = torch.where(legal, cv.to(torch.int16), torch.tensor(-128, dtype=torch.int16, device=device))
        cv_for_min = torch.where(legal, cv.to(torch.int16), torch.tensor(127, dtype=torch.int16, device=device))

        max_val = cv_for_max.max(dim=1).values  # (K,)
        min_val = cv_for_min.min(dim=1).values  # (K,)

        V[idx] = torch.where(
            is_team0_k,
            max_val.to(torch.int8),
            min_val.to(torch.int8)
        )

    return V, move_values


def solve_seed(seed: int, decl_id: int, device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Solve one seed completely.

    Args:
        seed: RNG seed for dealing
        decl_id: Declaration ID (0-6 for pip trump)
        device: Target device (defaults to CPU)

    Returns:
        (all_states, V, move_values, root_value) where:
        - all_states: (M,) int64 sorted tensor of all reachable states
        - V: (M,) int8 minimax value for each state
        - move_values: (M, 7) int8 value of each move
        - root_value: int minimax value of the initial state
    """
    if device is None:
        device = torch.device('cpu')

    ctx = build_context(seed, decl_id)

    # Phase 1: Enumerate all reachable states
    all_states = enumerate_gpu(ctx, device)

    # Phase 2: Build child index
    child_idx = build_child_index(all_states, ctx)

    # Phase 3: Solve via backward induction
    V, move_values = solve_gpu(all_states, child_idx, ctx)

    # Find root value (initial state is NOT at index 0 after sorting!)
    initial_state = pack_initial_state(ctx)
    root_idx = torch.searchsorted(all_states, initial_state).item()
    root_value = int(V[root_idx])

    return all_states, V, move_values, root_value
