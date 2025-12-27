"""
Texas 42 Solver - CPU Backward Induction

Core solver that computes optimal values and ALL move values for every reachable state.
Uses recursive memoization (top-down DP) for simplicity and correctness.

Output:
- V: dict mapping packed_state -> optimal value (-42 to +42)
- MoveValues: dict mapping packed_state -> list of 7 move values (-128 for illegal)
"""

from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import time
import sys

from .context import SeedContext, build_context
from .state import State, pack_state, unpack_state, get_legal_mask, apply_move, initial_state


# Increase recursion limit for deep game trees
sys.setrecursionlimit(50000)


@dataclass
class SolveResult:
    """Result of solving a seed + declaration."""
    seed: int
    decl_id: int
    root_value: int
    V: Dict[int, int]  # packed_state -> optimal value
    MoveValues: Dict[int, List[int]]  # packed_state -> [7 move values]
    stats: dict


def solve_cpu(ctx: SeedContext, first_leader: int = 0, verbose: bool = False) -> SolveResult:
    """
    Solve using CPU recursive memoization.

    Returns SolveResult with optimal values and all move values for every state.
    """
    start_time = time.time()

    if verbose:
        print(f"Solving seed={ctx.seed}, decl={ctx.decl_id}...")

    V: Dict[int, int] = {}
    MoveValues: Dict[int, List[int]] = {}
    nodes_visited = [0]  # Use list for closure mutation
    last_report = [time.time()]

    def dp(state: State) -> int:
        """Recursive minimax with memoization."""
        packed = pack_state(state)

        if packed in V:
            return V[packed]

        nodes_visited[0] += 1

        # Progress report
        if verbose and nodes_visited[0] % 100000 == 0:
            elapsed = time.time() - start_time
            rate = nodes_visited[0] / elapsed
            print(f"  {nodes_visited[0]:,} nodes visited ({rate:,.0f}/s)")

        # Terminal state
        if state.is_terminal():
            val = state.value()
            V[packed] = val
            MoveValues[packed] = [-128] * 7
            return val

        # Get legal moves
        legal = get_legal_mask(state, ctx)
        team0_turn = (state.current_player % 2 == 0)

        move_vals = [-128] * 7
        best = -43 if team0_turn else 43

        for local_idx in range(7):
            if not ((legal >> local_idx) & 1):
                continue

            child = apply_move(state, local_idx, ctx)
            child_val = dp(child)
            move_vals[local_idx] = child_val

            if team0_turn:
                best = max(best, child_val)
            else:
                best = min(best, child_val)

        V[packed] = best
        MoveValues[packed] = move_vals
        return best

    # Solve from initial state
    root = initial_state(ctx, first_leader)
    root_value = dp(root)

    total_time = time.time() - start_time

    stats = {
        'total_states': len(V),
        'nodes_visited': nodes_visited[0],
        'total_time': total_time,
    }

    if verbose:
        print(f"Solve complete in {total_time:.2f}s")
        print(f"States: {len(V):,}, Nodes visited: {nodes_visited[0]:,}")
        print(f"Root value: {root_value:+d} (team 0 {'wins' if root_value > 0 else 'loses' if root_value < 0 else 'ties'} by {abs(root_value)} points)")

    return SolveResult(
        seed=ctx.seed,
        decl_id=ctx.decl_id,
        root_value=root_value,
        V=V,
        MoveValues=MoveValues,
        stats=stats
    )


def solve_seed(seed: int, decl_id: int, first_leader: int = 0, verbose: bool = False) -> SolveResult:
    """Convenience function to solve a seed from scratch."""
    ctx = build_context(seed, decl_id, verbose=verbose)
    return solve_cpu(ctx, first_leader, verbose=verbose)


def solve_seed_iterative(seed: int, decl_id: int, first_leader: int = 0, verbose: bool = False) -> SolveResult:
    """
    Solve using iterative DFS (no recursion limit issues).
    """
    from collections import deque

    start_time = time.time()
    ctx = build_context(seed, decl_id, verbose=verbose)

    if verbose:
        print(f"Solving seed={seed}, decl={decl_id} (iterative)...", flush=True)

    V: Dict[int, int] = {}
    MoveValues: Dict[int, List[int]] = {}

    # Post-order DFS using explicit stack
    # Stack entries: (state, phase, move_idx, move_vals, children_vals)
    # phase 0: expand children, phase 1: process result
    root = initial_state(ctx, first_leader)
    stack = [(root, 0, 0, [-128] * 7, [])]
    nodes = 0
    last_report = time.time()

    while stack:
        state, phase, move_idx, move_vals, children_vals = stack.pop()
        packed = pack_state(state)

        if packed in V:
            continue

        nodes += 1
        if verbose and time.time() - last_report > 10:
            elapsed = time.time() - start_time
            print(f"  {nodes:,} nodes, {len(V):,} solved, {elapsed:.0f}s", flush=True)
            last_report = time.time()

        if state.is_terminal():
            V[packed] = state.value()
            MoveValues[packed] = [-128] * 7
            continue

        legal = get_legal_mask(state, ctx)

        if phase == 0:
            # Phase 0: Queue children for processing
            children_to_process = []
            for i in range(7):
                if (legal >> i) & 1:
                    child = apply_move(state, i, ctx)
                    child_packed = pack_state(child)
                    if child_packed not in V:
                        children_to_process.append((i, child))

            if children_to_process:
                # Push ourselves back for phase 1
                stack.append((state, 1, 0, move_vals, []))
                # Push children
                for i, child in reversed(children_to_process):
                    stack.append((child, 0, 0, [-128] * 7, []))
            else:
                # All children already solved, compute our value
                team0 = state.current_player % 2 == 0
                best = -43 if team0 else 43
                for i in range(7):
                    if (legal >> i) & 1:
                        child = apply_move(state, i, ctx)
                        child_packed = pack_state(child)
                        if child_packed in V:
                            val = V[child_packed]
                            move_vals[i] = val
                            if team0:
                                best = max(best, val)
                            else:
                                best = min(best, val)
                V[packed] = best
                MoveValues[packed] = move_vals
        else:
            # Phase 1: All children processed, compute value
            team0 = state.current_player % 2 == 0
            best = -43 if team0 else 43
            for i in range(7):
                if (legal >> i) & 1:
                    child = apply_move(state, i, ctx)
                    child_packed = pack_state(child)
                    if child_packed in V:
                        val = V[child_packed]
                        move_vals[i] = val
                        if team0:
                            best = max(best, val)
                        else:
                            best = min(best, val)
            V[packed] = best
            MoveValues[packed] = move_vals

    total_time = time.time() - start_time
    root_packed = pack_state(root)
    root_value = V.get(root_packed, 0)

    if verbose:
        print(f"Solve complete in {total_time:.1f}s", flush=True)
        print(f"States: {len(V):,}", flush=True)
        print(f"Root value: {root_value:+d}", flush=True)

    return SolveResult(
        seed=seed,
        decl_id=decl_id,
        root_value=root_value,
        V=V,
        MoveValues=MoveValues,
        stats={'total_states': len(V), 'nodes_visited': nodes, 'total_time': total_time}
    )


def print_move_values(result: SolveResult, ctx: SeedContext, state: State = None):
    """Print move values for a state (default: initial state)."""
    from .tables import DOMINO_PIPS

    if state is None:
        state = initial_state(ctx)

    packed = pack_state(state)
    optimal = result.V[packed]
    move_vals = result.MoveValues[packed]
    legal = get_legal_mask(state, ctx)

    print(f"State: P{state.current_player}'s turn, trick_len={state.trick_len}")
    print(f"Optimal value: {optimal:+d}")
    print("Move values:")

    for i in range(7):
        if (legal >> i) & 1:
            gid = int(ctx.L[state.current_player, i])
            lo, hi = DOMINO_PIPS[gid]
            mv = move_vals[i]
            regret = abs(optimal - mv) if state.current_player % 2 == 0 else abs(mv - optimal)
            marker = "*" if mv == optimal else " "
            print(f"  {marker} {hi}-{lo}: value={mv:+3d}, regret={regret:2d}")
        else:
            print(f"    [local {i}: not in hand]")


if __name__ == "__main__":
    # Test solver on a sample seed
    seed = 12345
    decl_id = 3  # Threes trump

    print(f"=== Solving seed={seed}, decl={decl_id} ===\n")

    result = solve_seed(seed, decl_id, verbose=True)

    print(f"\n=== Results ===")
    print(f"Total states: {result.stats['total_states']:,}")
    print(f"Root value: {result.root_value:+d}")

    # Show move values at root
    ctx = build_context(seed, decl_id)
    print(f"\nMove values at initial state:")
    print_move_values(result, ctx)
