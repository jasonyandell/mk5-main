#!/usr/bin/env python3
"""
Cross-validation script - compares Python solver results with TypeScript minimax.

Usage:
    python scripts/solver/validate.py --seeds 10           # Validate 10 random seeds
    python scripts/solver/validate.py --seed-range 0 100   # Validate specific range
    python scripts/solver/validate.py --output-dir ./output # Validate existing output files
"""

import argparse
import json
import subprocess
import sys
import random
from pathlib import Path

import torch


def run_ts_minimax(seed: int, decl_id: int, project_dir: Path) -> dict:
    """
    Run TypeScript minimax evaluation for a seed.

    Returns parsed JSON result.
    """
    result = subprocess.run(
        ['npx', 'tsx', 'scripts/minimax-eval.ts', str(seed), str(decl_id)],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"TS minimax failed: {result.stderr}")

    return json.loads(result.stdout)


def solve_python(seed: int, decl_id: int) -> int:
    """
    Solve with Python and return root value.
    """
    from solve import solve_seed

    _, V, _, root_value = solve_seed(seed, decl_id)
    return root_value


def validate_seed(seed: int, decl_id: int, project_dir: Path, verbose: bool = True) -> bool:
    """
    Validate a single seed against TypeScript minimax.

    Returns True if values match.
    """
    if verbose:
        print(f"Validating seed={seed} decl={decl_id}...", end=" ", flush=True)

    # Get TypeScript result
    ts_result = run_ts_minimax(seed, decl_id, project_dir)
    ts_value = ts_result['pythonValue']

    # Get Python result
    py_value = solve_python(seed, decl_id)

    if py_value == ts_value:
        if verbose:
            print(f"PASS (value={py_value:+d})")
        return True
    else:
        if verbose:
            print(f"FAIL: Python={py_value:+d} TypeScript={ts_value:+d}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Cross-validate Python solver against TypeScript minimax',
    )

    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument('--seeds', type=int,
                           help='Number of random seeds to validate')
    seed_group.add_argument('--seed-range', type=int, nargs=2, metavar=('START', 'END'),
                           help='Range of seeds to validate')
    seed_group.add_argument('--seed', type=int,
                           help='Single seed to validate')

    parser.add_argument('--decl', type=int, default=0, choices=range(7),
                       help='Declaration ID (default: 0)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only show failures and summary')

    args = parser.parse_args()

    # Find project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent.parent

    # Determine seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds is not None:
        seeds = random.sample(range(1, 1000000), args.seeds)
    else:
        seeds = list(range(args.seed_range[0], args.seed_range[1]))

    print(f"Validating {len(seeds)} seeds against TypeScript minimax...")
    print(f"Declaration: {args.decl}")
    print()

    # Run validation
    passed = 0
    failed = 0

    for seed in seeds:
        try:
            if validate_seed(seed, args.decl, project_dir, verbose=not args.quiet):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"ERROR seed={seed}: {e}")
            failed += 1

    # Summary
    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL VALIDATIONS PASSED!")
        sys.exit(0)


if __name__ == '__main__':
    main()
