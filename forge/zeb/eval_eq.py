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

DEFAULT_ORACLE = 'forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt'


def _load_oracle(checkpoint_path: str, device: str):
    """Load Stage 1 oracle, bypassing Lightning RNG state issues."""
    from forge.ml.module import DominoLightningModule

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint['hyper_parameters']
    model = DominoLightningModule(
        embed_dim=hparams.get('embed_dim', 64),
        n_heads=hparams.get('n_heads', 4),
        n_layers=hparams.get('n_layers', 2),
        ff_dim=hparams.get('ff_dim', 128),
        dropout=hparams.get('dropout', 0.1),
        lr=hparams.get('lr', 1e-3),
    )
    state_dict = checkpoint['state_dict']
    if any(k.startswith('model._orig_mod.') for k in state_dict.keys()):
        state_dict = {
            k.replace('._orig_mod.', '.'): v for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def _load_zeb(source: str, device: str, **kwargs):
    """Load ZebModel from a local checkpoint or HuggingFace Hub.

    Args:
        source: Local .pt path, or 'hf' / HF repo ID (e.g. 'jasonyandell/zeb-42').
    """
    from .hf import DEFAULT_REPO, load_zeb_from_hf

    # HuggingFace path: 'hf', or contains '/' but isn't a local file
    if source == 'hf' or (not source.endswith('.pt') and '/' in source):
        repo_id = DEFAULT_REPO if source == 'hf' else source
        weights_name = kwargs.get('weights_name') or 'model.pt'
        label = f'{repo_id} ({weights_name})' if weights_name != 'model.pt' else repo_id
        print(f'  Loading Zeb from HF: {label}')
        return load_zeb_from_hf(repo_id, device=device, weights_name=weights_name)

    # Local checkpoint
    from .model import ZebModel

    ckpt = torch.load(source, map_location='cpu', weights_only=False)

    if 'model_config' in ckpt:
        model_config = ckpt['model_config']
    elif 'config' in ckpt and 'model_config' in ckpt['config']:
        model_config = ckpt['config']['model_config']
    elif 'config' in ckpt:
        config = ckpt['config']
        model_config = {k: v for k, v in config.items()
                       if k in ('embed_dim', 'n_heads', 'n_layers', 'ff_dim', 'dropout', 'max_tokens')}
    else:
        raise ValueError("Checkpoint missing model config")

    model = ZebModel(**model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)

    epoch = ckpt.get('epoch', '?')
    print(f"  Zeb epoch: {epoch}")
    return model


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
