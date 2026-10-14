"""A tiny, deterministic student for tests that exercise structure, not strength.

The trained adapters (14 MB each) are generated data and live outside git. A test
that checks legality, determinism, information-state purity or finiteness holds
for any weights, so it builds this student instead: the same classes and the same
adapter file format, a few thousand parameters, a fixed seed.
"""
from __future__ import annotations

from pathlib import Path

import torch

_ARGS = dict(d_model=16, n_heads=2, n_layers=1, ff_dim=32, d_world=8, q_hidden=16)


def save_tiny_student(path: str | Path, *, auction: bool, seed: int = 0) -> Path:
    """Write a randomly initialised voids (or voids+auction) student adapter to `path`.

    The file round-trips through `gus.model.load.load_student` like a trained one.
    """
    from gus.model.student import StudentTransformerFullVoids, StudentTransformerFullVoidsAuction

    args = dict(_ARGS, voids_hidden=8)
    if auction:
        args.update(auction=True, bids_hidden=8)
    cls = StudentTransformerFullVoidsAuction if auction else StudentTransformerFullVoids
    kwargs = {k: v for k, v in args.items() if k != "auction"}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = cls(dropout=0.0, **kwargs)
    path = Path(path)
    torch.save({"args": args, "model_state": model.state_dict()}, path)
    return path
