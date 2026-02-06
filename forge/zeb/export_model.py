"""Export a model-only snapshot from a training checkpoint.

Strips optimizer state, replay buffer, and other training artifacts,
keeping only what's needed to load and run the model.

Usage:
    python -m forge.zeb.export_model \
        forge/zeb/checkpoints/selfplay-epoch2839.pt \
        forge/zeb/models/zeb-e2839.pt
"""

import argparse
from pathlib import Path

import torch


def export_model(checkpoint_path: Path, output_path: Path) -> dict:
    """Extract model weights + config from a training checkpoint.

    Returns metadata about the exported snapshot.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract model config (handle multiple checkpoint formats)
    if 'model_config' in ckpt:
        model_config = ckpt['model_config']
    elif 'config' in ckpt and 'model_config' in ckpt['config']:
        model_config = ckpt['config']['model_config']
    else:
        raise ValueError("Checkpoint missing model config")

    snapshot = {
        'model_state_dict': ckpt['model_state_dict'],
        'model_config': model_config,
        'epoch': ckpt.get('epoch', 0),
        'total_games': ckpt.get('total_games', 0),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, output_path)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description='Export model-only snapshot')
    parser.add_argument('checkpoint', type=Path, help='Training checkpoint')
    parser.add_argument('output', type=Path, help='Output path for slim snapshot')
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    snapshot = export_model(args.checkpoint, args.output)

    orig_kb = args.checkpoint.stat().st_size / 1024
    new_kb = args.output.stat().st_size / 1024
    print(f"Exported epoch {snapshot['epoch']} ({snapshot['total_games']} games)")
    print(f"  {orig_kb:.0f} KB -> {new_kb:.0f} KB ({new_kb/orig_kb*100:.1f}%)")


if __name__ == '__main__':
    main()
