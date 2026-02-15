"""Model loading utilities for evaluation.

Centralizes oracle and Zeb model loading, shared by eval_eq.py and the eval CLI.
"""

from __future__ import annotations

import torch

# NOTE: This is the *large* oracle (3.3M params, qgap=0.071). The MCTS pipeline
# in oracle_value.py uses a different default: the non-large 3.3M oracle
# (qgap=0.074, qmae=0.96). They are distinct models — keep this explicit.
DEFAULT_ORACLE = 'forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt'


def load_oracle(checkpoint_path: str, device: str):
    """Load Stage 1 oracle from a Lightning checkpoint using pure PyTorch.

    Extracts DominoTransformer weights directly, avoiding any Lightning dependency.
    """
    from forge.ml.transformer import DominoTransformer

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint['hyper_parameters']
    model = DominoTransformer(
        embed_dim=hparams.get('embed_dim', 64),
        n_heads=hparams.get('n_heads', 4),
        n_layers=hparams.get('n_layers', 2),
        ff_dim=hparams.get('ff_dim', 128),
        dropout=hparams.get('dropout', 0.1),
    )
    state_dict = checkpoint['state_dict']
    # Strip Lightning wrapper prefix (model.X -> X) and torch.compile prefix
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace('._orig_mod.', '.')
        if k.startswith('model.'):
            k = k[len('model.'):]
        cleaned[k] = v
    model.load_state_dict(cleaned)
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
    from .. import load_model

    model, ckpt = load_model(source, device=device, eval_mode=True)

    epoch = ckpt.get('epoch', '?')
    print(f"  Zeb epoch: {epoch}")
    return model
