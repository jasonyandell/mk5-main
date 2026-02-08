"""Evaluate E[Q] player vs random or vs Zeb.

Usage:
    python -u -m forge.zeb.eval_eq                          # E[Q] vs random
    python -u -m forge.zeb.eval_eq --vs-zeb                 # E[Q] vs Zeb (latest from HF)
    python -u -m forge.zeb.eval_eq --vs-zeb path/to/ckpt.pt # E[Q] vs Zeb (local checkpoint)
    python -u -m forge.zeb.eval_eq --n-games 500 --n-samples 50
"""

from __future__ import annotations

import argparse
import time
import warnings

import torch

from .eval.loading import DEFAULT_ORACLE, load_oracle as _load_oracle, load_zeb as _load_zeb


def main():
    parser = argparse.ArgumentParser(description='Evaluate E[Q] player')
    parser.add_argument(
        '--checkpoint', default=DEFAULT_ORACLE,
        help='Stage 1 oracle checkpoint for E[Q]',
    )
    parser.add_argument(
        '--vs-zeb', default=None, nargs='?', const='hf', metavar='PATH',
        help='Zeb model: "hf" (default) for latest from HuggingFace, '
             'or local .pt path. Omit entirely for E[Q] vs random.',
    )
    parser.add_argument(
        '--weights-name', default=None,
        help='HF weights namespace (e.g. "large" for large.pt). Only used with --vs-zeb hf.',
    )
    parser.add_argument('--n-games', type=int, default=1000, help='Total games')
    parser.add_argument('--n-samples', type=int, default=100, help='Worlds per E[Q] decision')
    parser.add_argument('--batch-size', type=int, default=0, help='Max games per GPU batch (0=all at once)')
    parser.add_argument('--device', default='cuda', help='Device')
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*nested tensors.*prototype.*")

    print(f'Loading oracle: {args.checkpoint}')
    oracle = _load_oracle(args.checkpoint, args.device)

    if args.vs_zeb:
        # E[Q] vs Zeb
        print(f'Loading Zeb: {args.vs_zeb}')
        zeb = _load_zeb(args.vs_zeb, args.device, weights_name=args.weights_name)

        print(f'\nE[Q] vs Zeb Evaluation')
        print(f'  E[Q] samples per decision: {args.n_samples}')
        print(f'  Total games: {args.n_games}')
        print()

        t0 = time.time()
        from .eq_player import evaluate_eq_vs_zeb
        results = evaluate_eq_vs_zeb(
            oracle, zeb,
            n_games=args.n_games,
            n_samples=args.n_samples,
            device=args.device,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0

        r0 = results['as_team0']
        r1 = results['as_team1']

        print()
        print(f'E[Q] as Team 0 vs Zeb:  {r0["eq_wins"]}/{r0["n_games"]} '
              f'({r0["eq_win_rate"]:.1%})  margin {r0["avg_margin"]:+.1f}')
        print(f'E[Q] as Team 1 vs Zeb:  {r1["eq_wins"]}/{r1["n_games"]} '
              f'({r1["eq_win_rate"]:.1%})  margin {r1["avg_margin"]:+.1f}')
        print()
        print(f'Overall E[Q] win rate vs Zeb: {results["eq_win_rate"]:.1%} '
              f'({results["eq_wins"]}/{results["total_games"]})')

    else:
        # E[Q] vs random
        print(f'\nE[Q] vs Random Evaluation')
        print(f'  Samples per decision: {args.n_samples}')
        print(f'  Total games: {args.n_games}')
        print()

        t0 = time.time()
        from .eq_player import evaluate_eq_vs_random
        results = evaluate_eq_vs_random(
            oracle,
            n_games=args.n_games,
            n_samples=args.n_samples,
            device=args.device,
        )
        elapsed = time.time() - t0

        r0 = results['as_team0']
        r1 = results['as_team1']

        print()
        print(f'E[Q] as Team 0 vs Random:  {r0["eq_wins"]}/{r0["n_games"]} '
              f'({r0["eq_win_rate"]:.1%})  margin {r0["avg_margin"]:+.1f}')
        print(f'E[Q] as Team 1 vs Random:  {r1["eq_wins"]}/{r1["n_games"]} '
              f'({r1["eq_win_rate"]:.1%})  margin {r1["avg_margin"]:+.1f}')
        print()
        print(f'Overall E[Q] win rate: {results["eq_win_rate"]:.1%} '
              f'({results["eq_wins"]}/{results["total_games"]})')
        print(f'(Zeb reference: ~69-70% vs random)')

    print(f'\nElapsed: {elapsed:.1f}s ({results["total_games"]/elapsed:.1f} games/s)')


if __name__ == '__main__':
    main()
