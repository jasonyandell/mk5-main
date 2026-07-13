"""Shared adapter loader (#30): one args->model reconstruction for every consumer.

The bug being guarded against: sibling loaders (eval/*, drama atlas) used to
reconstruct the model from saved args themselves with NO auction branch, so they
crashed at load_state_dict on an auction adapter. `gus.model.load.load_student`
is now that branch in one place — these tests round-trip all three student
variants through it, and confirm `load_gus` still delegates to it unchanged.
"""
import torch

from gus.bidding.evaluate import load_gus
from gus.model.load import load_student
from gus.model.student import (
    StudentTransformerFull,
    StudentTransformerFullVoids,
    StudentTransformerFullVoidsAuction,
)

_SMALL = dict(d_model=32, n_heads=2, n_layers=1, ff_dim=64, d_world=16, q_hidden=32)


def _save(tmp_path, model, args):
    p = tmp_path / "adapter.pt"
    torch.save({"model_state": model.state_dict(), "args": args, "epoch": 1}, p)
    return str(p)


def test_loads_plain(tmp_path):
    model = StudentTransformerFull(**_SMALL)
    args = {**_SMALL}  # no voids_hidden, no auction
    loaded, is_voids = load_student(_save(tmp_path, model, args), device="cpu")
    assert isinstance(loaded, StudentTransformerFull)
    assert not isinstance(loaded, StudentTransformerFullVoids)
    assert is_voids is False


def test_loads_voids(tmp_path):
    model = StudentTransformerFullVoids(voids_hidden=16, **_SMALL)
    # A voids run carries bids_hidden in args (it has a default) but auction=False:
    # detection must key on the auction flag, not the bids_hidden key.
    args = {**_SMALL, "voids_hidden": 16, "bids_hidden": 64, "auction": False}
    loaded, is_voids = load_student(_save(tmp_path, model, args), device="cpu")
    assert isinstance(loaded, StudentTransformerFullVoids)
    assert not isinstance(loaded, StudentTransformerFullVoidsAuction)
    assert not hasattr(loaded, "bids_encoder")
    assert is_voids is True


def test_loads_auction(tmp_path):
    """The #30 bug: this is the branch the sibling loaders lacked — an auction
    adapter must reconstruct as the auction student and load its weights."""
    model = StudentTransformerFullVoidsAuction(voids_hidden=16, bids_hidden=16, **_SMALL)
    args = {**_SMALL, "voids_hidden": 16, "bids_hidden": 16, "auction": True}
    loaded, is_voids = load_student(_save(tmp_path, model, args), device="cpu")
    assert isinstance(loaded, StudentTransformerFullVoidsAuction)
    assert hasattr(loaded, "bids_encoder")
    assert is_voids is True  # the auction model consumes voids too


def test_stashes_adapter_args(tmp_path):
    model = StudentTransformerFullVoids(voids_hidden=16, **_SMALL)
    args = {**_SMALL, "voids_hidden": 16}
    loaded, _ = load_student(_save(tmp_path, model, args), device="cpu")
    assert loaded.adapter_args["d_model"] == _SMALL["d_model"]
    assert loaded.adapter_args["n_layers"] == _SMALL["n_layers"]


def test_load_gus_delegates(tmp_path):
    """load_gus must keep its (model, is_voids) contract while delegating to the
    shared loader (other code + a concurrent session build on this signature)."""
    model = StudentTransformerFullVoidsAuction(voids_hidden=16, bids_hidden=16, **_SMALL)
    args = {**_SMALL, "voids_hidden": 16, "bids_hidden": 16, "auction": True}
    loaded, is_voids = load_gus(_save(tmp_path, model, args), device="cpu")
    assert isinstance(loaded, StudentTransformerFullVoidsAuction)
    assert is_voids is True
