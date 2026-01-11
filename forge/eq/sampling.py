"""Sample consistent opponent hands using backtracking with MRV heuristic.

This algorithm is ported from src/game/ai/hand-sampler.ts.

Unlike rejection sampling, backtracking is GUARANTEED to find a valid solution
if one exists (and one always exists - the real game state is one).
"""

import numpy as np
from forge.oracle.tables import is_in_called_suit, domino_contains_pip


def get_candidate_dominoes(
    pool: set[int], void_suits: set[int], decl_id: int
) -> set[int]:
    """Get dominoes from pool that don't violate void constraints.

    A domino violates void constraints if it could follow any void suit:
    - For pip suits (0-6): domino contains that pip AND is not in called suit
    - For called suit (7): domino is in called suit

    Args:
        pool: Set of domino IDs available for assignment
        void_suits: Set of suits (0-6 or 7 for called suit) this player is void in
        decl_id: Declaration ID for checking called suit membership

    Returns:
        Set of domino IDs that could belong to this player
    """
    if not void_suits:
        return pool.copy()

    candidates = set()
    for domino_id in pool:
        is_valid = True
        for suit in void_suits:
            if suit == 7:  # Called suit
                if is_in_called_suit(domino_id, decl_id):
                    is_valid = False
                    break
            else:  # Pip suit (0-6)
                if domino_contains_pip(domino_id, suit) and not is_in_called_suit(
                    domino_id, decl_id
                ):
                    is_valid = False
                    break
        if is_valid:
            candidates.add(domino_id)
    return candidates


def _backtrack(
    opponents: list[int],
    remaining: dict[int, int],
    available: set[int],
    candidates_per_opponent: dict[int, set[int]],
    hands: dict[int, list[int]],
    rng: np.random.Generator,
) -> bool:
    """Backtracking search with MRV (Minimum Remaining Values) heuristic.

    Always assigns to the player with minimum slack (fewest excess candidates)
    first. This aggressive pruning makes the algorithm efficient in practice.

    Args:
        opponents: List of opponent player indices
        remaining: How many more dominoes each player needs
        available: Dominoes still available to assign
        candidates_per_opponent: Which dominoes each player COULD hold
        hands: Current partial assignment (modified in place)
        rng: Random generator for shuffling candidates

    Returns:
        True if a valid assignment was found, False otherwise
    """
    # Find player with minimum slack (MRV heuristic)
    min_slack = float("inf")
    most_constrained = None

    for opp in opponents:
        need = remaining[opp]
        if need == 0:
            continue

        # Count how many candidates are still available
        available_count = sum(1 for d in candidates_per_opponent[opp] if d in available)
        slack = available_count - need

        # Early pruning: if any player can't be satisfied, backtrack immediately
        if slack < 0:
            return False

        if slack < min_slack:
            min_slack = slack
            most_constrained = opp

    # All players satisfied
    if most_constrained is None:
        return True

    # Get shuffled candidates for randomness in solution
    available_candidates = [
        d for d in candidates_per_opponent[most_constrained] if d in available
    ]
    rng.shuffle(available_candidates)

    # Try each candidate with backtracking
    for candidate_id in available_candidates:
        # Choose
        available.remove(candidate_id)
        hands[most_constrained].append(candidate_id)
        remaining[most_constrained] -= 1

        # Recurse
        if _backtrack(
            opponents, remaining, available, candidates_per_opponent, hands, rng
        ):
            return True

        # Backtrack
        available.add(candidate_id)
        hands[most_constrained].pop()
        remaining[most_constrained] += 1

    return False  # No valid assignment found in this branch


def sample_consistent_worlds(
    my_player: int,
    my_hand: list[int],
    played: set[int],
    hand_sizes: list[int],
    voids: dict[int, set[int]],
    decl_id: int,
    n_samples: int,
    max_attempts_per_sample: int = 100,  # Kept for API compatibility, not used
    rng: np.random.Generator | None = None,
) -> list[list[list[int]]]:
    """Sample opponent hands consistent with void constraints using backtracking.

    Uses backtracking with MRV heuristic - GUARANTEED to find a valid solution
    if one exists. The real game state is always a valid solution, so this
    should never fail unless there's a bug in void inference.

    Args:
        my_player: Which player I am (my hand is fixed)
        my_hand: My known dominoes
        played: Set of dominoes already played
        hand_sizes: How many dominoes each player currently has
        voids: Void constraints from infer_voids()
        decl_id: Declaration for checking suit membership
        n_samples: How many consistent worlds to generate
        max_attempts_per_sample: (Unused - kept for API compatibility)
        rng: NumPy random generator (optional, creates default if None)

    Returns:
        List of N worlds, each world is [hand0, hand1, hand2, hand3]
        where each hand is a list of domino IDs.
        my_hand is always at position my_player.

    Raises:
        RuntimeError: If unable to find valid assignment (indicates bug in voids)
    """
    # Initialize RNG
    if rng is None:
        rng = np.random.default_rng()

    # Create pool of available dominoes
    all_dominoes = set(range(28))
    pool = all_dominoes - played - set(my_hand)

    # Validate input
    if len(my_hand) != hand_sizes[my_player]:
        raise ValueError(
            f"my_hand length ({len(my_hand)}) doesn't match "
            f"hand_sizes[{my_player}] ({hand_sizes[my_player]})"
        )

    total_needed = sum(hand_sizes) - hand_sizes[my_player]
    if len(pool) != total_needed:
        raise ValueError(
            f"Pool size ({len(pool)}) doesn't match "
            f"total opponent cards needed ({total_needed})"
        )

    # Get opponent player indices
    opponents = [p for p in range(4) if p != my_player]

    # Precompute candidate sets for each opponent (respecting void constraints)
    candidates_per_opponent = {}
    for opp in opponents:
        candidates_per_opponent[opp] = get_candidate_dominoes(
            pool, voids.get(opp, set()), decl_id
        )

    # Generate N samples
    worlds = []
    for _ in range(n_samples):
        # Initialize for this sample
        available = pool.copy()
        remaining = {opp: hand_sizes[opp] for opp in opponents}
        hands_result: dict[int, list[int]] = {opp: [] for opp in opponents}

        # Run backtracking search
        success = _backtrack(
            opponents,
            remaining,
            available,
            candidates_per_opponent,
            hands_result,
            rng,
        )

        if not success:
            raise RuntimeError(
                f"No valid hand distribution exists. "
                f"This indicates a bug in constraint/void tracking.\n"
                f"Pool size: {len(pool)}, Voids: {voids}"
            )

        # Build complete world with all 4 players
        world = [None] * 4
        world[my_player] = my_hand.copy()
        for opp in opponents:
            world[opp] = hands_result[opp]
        worlds.append(world)

    return worlds
