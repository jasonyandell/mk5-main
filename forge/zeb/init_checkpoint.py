"""Create a fresh model checkpoint for bootstrapping a new training run.

Usage:
    python -m forge.zeb.init_checkpoint --size large -o forge/zeb/checkpoints/large-init.pt
    python -m forge.zeb.init_checkpoint --size medium -o forge/zeb/checkpoints/medium-init.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from forge.zeb.model import ZebModel, get_model_config


def create_checkpoint(size: str, output_path: Path) -> None:
    config = get_model_config(size)
    model = ZebModel(**config)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': config,
        'epoch': 0,
        'total_games': 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Created {size} checkpoint: {output_path}")
    print(f"  {n_params:,} parameters")
    print(f"  config: {config}")


def main():
    parser = argparse.ArgumentParser(description='Create a fresh model checkpoint')
    parser.add_argument('--size', type=str, default='large',
                        choices=['small', 'medium', 'large'],
                        help='Model size (default: large)')
    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output checkpoint path')
    args = parser.parse_args()
    create_checkpoint(args.size, args.output)


if __name__ == '__main__':
    main()
