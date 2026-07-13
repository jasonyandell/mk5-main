"""Single source of truth for reconstructing a student from a saved adapter.

Every consumer (bidding evaluator, arena, eval/*, drama atlas) used to carry
its own copy of "rebuild the model class from ckpt['args'] then load_state_dict".
Those copies drifted: only the bidding evaluator grew the auction branch, so the
sibling copies would crash at load_state_dict on an auction adapter (#24/#30).

This module is that branch in one place. It detects the student variant from the
saved args and returns the constructed, eval-mode model alongside `is_voids`
(which callers use to decide whether to feed the voids feature vector).
"""
from __future__ import annotations

from pathlib import Path

import torch

from gus.model.student import (
    StudentTransformerFull,
    StudentTransformerFullVoids,
    StudentTransformerFullVoidsAuction,
)


def load_student(adapter_path: str | Path, device: str):
    """Reconstruct a student from a saved adapter and load its weights.

    Returns (model, is_voids). `is_voids` stays True for the auction model (it
    consumes voids too); callers detect the auction path via
    `hasattr(model, "bids_encoder")`.
    """
    ckpt = torch.load(adapter_path, weights_only=False, map_location=device)
    args = ckpt["args"]
    # The auction student (#24) is a superset of the voids student — it also
    # carries voids — so detect it first via the explicit --auction flag
    # (NOT "bids_hidden" in args: that key has a default, so it is present on
    # every voids run too).
    is_auction = bool(args.get("auction", False))
    is_voids = "voids_hidden" in args
    if is_auction:
        cls = StudentTransformerFullVoidsAuction
    elif is_voids:
        cls = StudentTransformerFullVoids
    else:
        cls = StudentTransformerFull
    kwargs = dict(
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        ff_dim=args.get("ff_dim", 256),
        dropout=0.0,
        d_world=args.get("d_world", 64),
        q_hidden=args.get("q_hidden", 256),
    )
    if is_voids:
        kwargs["voids_hidden"] = args.get("voids_hidden", 64)
    if is_auction:
        kwargs["bids_hidden"] = args.get("bids_hidden", 64)
    model = cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # Stash the source args so consumers that want them (e.g. a log line of
    # d_model/n_layers) need not re-read the checkpoint.
    model.adapter_args = args
    return model, is_voids
