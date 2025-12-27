#!/usr/bin/env python3
"""
Texas 42 Solver - CLI Entry Point

Usage:
    python -m scripts.solver.main --seed 12345 --decl 3
    python -m scripts.solver.main --seed 12345 --decl 3 --output scratch/seed_12345.json.gz
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Texas 42 Perfect Play Solver")
    parser.add_argument("--seed", type=int, required=True, help="Deal seed")
    parser.add_argument("--decl", type=int, required=True, help="Declaration ID (0-9)")
    parser.add_argument("--leader", type=int, default=0, help="First leader (0-3)")
    parser.add_argument("--output", "-o", type=str, help="Output file (default: scratch/seed_SEED_decl_DECL.json.gz)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Quick test (depth-limited)")

    args = parser.parse_args()

    # Imports after arg parsing for faster --help
    from .context import build_context
    from .solve import solve_cpu, SolveResult
    from .state import initial_state, pack_state, get_legal_mask, apply_move, State
    from .output import save_json, estimate_size

    print("=" * 60)
    print("Texas 42 Solver")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Declaration: {args.decl} ({['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes', 'doubles-trump', 'doubles-suit', 'no-trump'][args.decl]})")
    print(f"First leader: Player {args.leader}")
    print()

    # Build context
    print("Building context...", flush=True)
    ctx = build_context(args.seed, args.decl, verbose=args.verbose)
    print(f"  Trick outcomes precomputed: {ctx.TRICK_WINNER.size:,}")
    print()

    if args.quick:
        # Quick depth-limited test
        print("Running depth-limited test (8 moves)...", flush=True)
        V = {}
        count = [0]
        start = time.time()

        def dp_limited(state: State, depth: int) -> int:
            packed = pack_state(state)
            if packed in V:
                return V[packed]
            count[0] += 1
            if depth >= 8 or state.is_terminal():
                V[packed] = state.value()
                return V[packed]
            legal = get_legal_mask(state, ctx)
            team0 = state.current_player % 2 == 0
            best = -43 if team0 else 43
            for i in range(7):
                if (legal >> i) & 1:
                    child = apply_move(state, i, ctx)
                    val = dp_limited(child, depth + 1)
                    if team0:
                        best = max(best, val)
                    else:
                        best = min(best, val)
            V[packed] = best
            return best

        root = initial_state(ctx, args.leader)
        result_value = dp_limited(root, 0)
        elapsed = time.time() - start

        print(f"Result: {result_value:+d}")
        print(f"States: {len(V):,} in {elapsed:.2f}s")
        print()
        print("Quick test complete. Use --full for complete solve (takes 10+ minutes).")
        return

    # Full solve
    print("Starting full solve (this may take 10-30 minutes)...", flush=True)
    print("Progress updates every 10 seconds.", flush=True)
    print()

    start_time = time.time()

    # Use recursive solver with progress
    sys.setrecursionlimit(100000)

    V = {}
    MoveValues = {}
    count = [0]
    last_report = [time.time()]

    def dp(state: State) -> int:
        packed = pack_state(state)
        if packed in V:
            return V[packed]

        count[0] += 1
        now = time.time()
        if now - last_report[0] > 10:
            elapsed = now - start_time
            rate = count[0] / elapsed
            eta_states = 60_000_000  # rough estimate
            eta_remaining = (eta_states - count[0]) / rate if rate > 0 else 0
            print(f"  {count[0]/1e6:.1f}M nodes, {len(V)/1e6:.1f}M solved, "
                  f"{elapsed/60:.1f}min elapsed, ~{eta_remaining/60:.0f}min remaining", flush=True)
            last_report[0] = now

        if state.is_terminal():
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
                val = dp(child)
                move_vals[i] = val
                if team0:
                    best = max(best, val)
                else:
                    best = min(best, val)

        V[packed] = best
        MoveValues[packed] = move_vals
        return best

    root = initial_state(ctx, args.leader)
    try:
        root_value = dp(root)
    except RecursionError:
        print(f"ERROR: RecursionError at {count[0]:,} nodes")
        print("Try running with: python -u -X recursionlimit=200000 ...")
        sys.exit(1)

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("SOLVE COMPLETE")
    print("=" * 60)
    print(f"Root value: {root_value:+d}")
    print(f"  Team 0 {'wins' if root_value > 0 else 'loses' if root_value < 0 else 'ties'} by {abs(root_value)} points")
    print(f"Total states: {len(V):,}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print()

    # Create result
    result = SolveResult(
        seed=args.seed,
        decl_id=args.decl,
        root_value=root_value,
        V=V,
        MoveValues=MoveValues,
        stats={"total_states": len(V), "nodes_visited": count[0], "total_time": elapsed}
    )

    # Estimate size
    sizes = estimate_size(result)
    print(f"Estimated output size: {sizes['estimated_gz_mb']:.1f} MB compressed")

    # Save output
    if args.output:
        output_path = args.output
    else:
        output_path = f"scratch/seed_{args.seed}_decl_{args.decl}.json.gz"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}...", flush=True)
    save_json(result, output_path)
    print(f"Saved!")


if __name__ == "__main__":
    main()
