"""CLI entry point for the unified eval system.

Usage:
    python -m forge.zeb.eval random vs heuristic --n-games 1000
    python -m forge.zeb.eval "eq:n=100" vs "zeb:source=hf" --n-games 500 --json
    python -m forge.zeb.eval --matrix random heuristic zeb "eq:n=50" --n-games 200
"""

from __future__ import annotations

import argparse
import sys

from .engine import MatchConfig, run_match
from .players import parse_player_spec
from .results import format_result, format_matrix


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python -m forge.zeb.eval',
        description='Unified evaluation CLI for Crystal Forge players',
    )
    parser.add_argument('--n-games', type=int, default=1000, help='Total games per matchup')
    parser.add_argument('--device', default='cuda', help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Max games per GPU batch (0=all at once)')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress progress output')

    # Mutually exclusive: positional "A vs B" or --matrix
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--matrix', nargs='+', metavar='PLAYER',
                       help='Run all-pairs matrix evaluation')
    group.add_argument('matchup', nargs='*', default=[],
                       help='PLAYER_A vs PLAYER_B')

    args = parser.parse_args(argv)

    # Validate matchup format
    if args.matchup:
        if len(args.matchup) != 3 or args.matchup[1].lower() != 'vs':
            parser.error("Matchup must be: PLAYER_A vs PLAYER_B")

    return args


def _run_single(args: argparse.Namespace) -> None:
    spec_a = parse_player_spec(args.matchup[0])
    spec_b = parse_player_spec(args.matchup[2])

    config = MatchConfig(
        spec_a=spec_a,
        spec_b=spec_b,
        n_games=args.n_games,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        quiet=args.quiet,
    )

    result = run_match(config)

    if not args.quiet:
        print()
    print(format_result(result, json_mode=args.json))


def _run_matrix(args: argparse.Namespace) -> None:
    names = args.matrix
    specs = [parse_player_spec(name) for name in names]
    n = len(specs)

    # Shared model cache across all matchups
    model_cache: dict = {}

    results: list[list[MatchResult | None]] = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            if not args.quiet:
                print(f'\n--- {specs[i].display_name} vs {specs[j].display_name} ---')

            config = MatchConfig(
                spec_a=specs[i],
                spec_b=specs[j],
                n_games=args.n_games,
                device=args.device,
                seed=args.seed,
                batch_size=args.batch_size,
                quiet=args.quiet,
                model_cache=model_cache,
            )

            results[i][j] = run_match(config)

    display_names = [s.display_name for s in specs]
    if not args.quiet:
        print()
    print(format_matrix(results, display_names, json_mode=args.json))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.matrix:
        _run_matrix(args)
    else:
        _run_single(args)


if __name__ == '__main__':
    main()
