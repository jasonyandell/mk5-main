"""
Texas 42 Solver - Verification

Verify solver correctness through internal consistency checks.
For cross-validation with TypeScript, use the unit tests.
"""

from typing import Dict, List
import random

from .context import build_context, SeedContext
from .state import initial_state, pack_state, get_legal_mask, apply_move, State
from .tables import DOMINO_PIPS


def verify_terminal_values(ctx: SeedContext, V: Dict[int, int], MoveValues: Dict[int, List[int]], verbose: bool = False) -> bool:
    """Verify that terminal state values match the scoring formula."""
    errors = 0

    for packed, value in V.items():
        moves = MoveValues.get(packed, [])
        # Terminal states have all moves = -128
        if all(m == -128 for m in moves):
            # This should be a terminal state
            # Value should be 2 * team0_points - 42
            # Extract team0_points from packed state (bits 28-33)
            team0_points = (packed >> 28) & 0x3F
            expected = 2 * team0_points - 42
            if value != expected:
                if verbose:
                    print(f"Terminal value mismatch: packed={hex(packed)}, value={value}, expected={expected}")
                errors += 1

    if verbose:
        print(f"Terminal value check: {errors} errors")
    return errors == 0


def verify_minimax_consistency(ctx: SeedContext, V: Dict[int, int], MoveValues: Dict[int, List[int]],
                                sample_size: int = 100, verbose: bool = False) -> bool:
    """Verify minimax consistency: V[s] should equal best child value."""
    errors = 0
    checked = 0

    # Sample random non-terminal states
    non_terminal = [(p, v) for p, v in V.items() if not all(m == -128 for m in MoveValues.get(p, []))]
    if len(non_terminal) > sample_size:
        non_terminal = random.sample(non_terminal, sample_size)

    for packed, value in non_terminal:
        moves = MoveValues[packed]
        legal_values = [m for m in moves if m != -128]

        if not legal_values:
            continue

        checked += 1

        # Determine whose turn (extract current_player from packed)
        current_player = (packed >> 36) & 0x3
        team0_turn = (current_player % 2 == 0)

        if team0_turn:
            expected = max(legal_values)
        else:
            expected = min(legal_values)

        if value != expected:
            if verbose:
                print(f"Minimax mismatch: packed={hex(packed)}, value={value}, "
                      f"expected={expected}, moves={moves}, player={current_player}")
            errors += 1

    if verbose:
        print(f"Minimax consistency check: {errors}/{checked} errors")
    return errors == 0


def verify_deal_consistency(seed: int, decl_id: int, verbose: bool = False) -> bool:
    """Verify that dealing produces valid hands."""
    from .deal import deal_dominoes_with_seed, hands_to_bitmasks

    hands = deal_dominoes_with_seed(seed)
    masks = hands_to_bitmasks(hands)

    # Check all 28 dominoes dealt exactly once
    combined = masks[0] | masks[1] | masks[2] | masks[3]
    if combined != (1 << 28) - 1:
        if verbose:
            print(f"Not all 28 dominoes dealt: {bin(combined)}")
        return False

    # Check no overlaps
    for i in range(4):
        for j in range(i + 1, 4):
            if masks[i] & masks[j]:
                if verbose:
                    print(f"Overlap between player {i} and {j}")
                return False

    if verbose:
        print("Deal consistency: OK")
    return True


def run_verification(seed: int, decl_id: int, V: Dict[int, int], MoveValues: Dict[int, List[int]],
                     verbose: bool = True) -> bool:
    """Run all verification checks."""
    ctx = build_context(seed, decl_id)

    print(f"Running verification for seed={seed}, decl={decl_id}")
    print(f"  Total states: {len(V):,}")

    all_ok = True

    # Check 1: Deal consistency
    if not verify_deal_consistency(seed, decl_id, verbose):
        all_ok = False

    # Check 2: Terminal values
    if not verify_terminal_values(ctx, V, MoveValues, verbose):
        all_ok = False

    # Check 3: Minimax consistency
    if not verify_minimax_consistency(ctx, V, MoveValues, sample_size=1000, verbose=verbose):
        all_ok = False

    if all_ok:
        print("All verification checks passed!")
    else:
        print("VERIFICATION FAILED")

    return all_ok


if __name__ == "__main__":
    # Run quick verification on depth-limited solve
    import sys
    sys.setrecursionlimit(10000)

    seed = 12345
    decl_id = 3
    ctx = build_context(seed, decl_id)

    print("Running depth-limited solve for verification...")
    V = {}
    MoveValues = {}

    def dp(state: State, depth: int) -> int:
        packed = pack_state(state)
        if packed in V:
            return V[packed]

        if depth >= 12 or state.is_terminal():
            V[packed] = state.value()
            MoveValues[packed] = [-128] * 7
            return V[packed]

        legal = get_legal_mask(state, ctx)
        team0 = state.current_player % 2 == 0
        move_vals = [-128] * 7
        best = -43 if team0 else 43

        for i in range(7):
            if (legal >> i) & 1:
                child = apply_move(state, i, ctx)
                val = dp(child, depth + 1)
                move_vals[i] = val
                if team0:
                    best = max(best, val)
                else:
                    best = min(best, val)

        V[packed] = best
        MoveValues[packed] = move_vals
        return best

    root = initial_state(ctx)
    result = dp(root, 0)
    print(f"Depth-limited result: {result:+d}, {len(V):,} states")
    print()

    run_verification(seed, decl_id, V, MoveValues)
