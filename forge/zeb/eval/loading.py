"""Model loading utilities for evaluation.

Centralizes oracle and Zeb model loading, shared by eval_eq.py and the eval CLI.
"""

from __future__ import annotations

import torch

DEFAULT_ORACLE = 'forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt'


def load_oracle(checkpoint_path: str, device: str):
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


def load_zeb(source: str, device: str, **kwargs):
    """Load ZebModel from a local checkpoint or HuggingFace Hub.

    Args:
        source: Local .pt path, or 'hf' / HF repo ID (e.g. 'jasonyandell/zeb-42').
    """
    from ..hf import DEFAULT_REPO, load_zeb_from_hf

    # HuggingFace path: 'hf', or contains '/' but isn't a local file
    if source == 'hf' or (not source.endswith('.pt') and '/' in source):
        repo_id = DEFAULT_REPO if source == 'hf' else source
        weights_name = kwargs.get('weights_name') or 'model.pt'
        label = f'{repo_id} ({weights_name})' if weights_name != 'model.pt' else repo_id
        print(f'  Loading Zeb from HF: {label}')
        return load_zeb_from_hf(repo_id, device=device, weights_name=weights_name)

    # Local checkpoint
    from ..model import ZebModel

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
