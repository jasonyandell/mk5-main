"""Auction-conditioned student (#24): forward shapes + adapter auto-detection."""
import torch

from gus.bidding.evaluate import load_gus
from gus.model.auction import N_AUCTION_FEATURES
from gus.model.student import (
    StudentTransformerFullVoids,
    StudentTransformerFullVoidsAuction,
)
from gus.model.tokenize import SEQ_LEN

_SMALL = dict(d_model=32, n_heads=2, n_layers=1, ff_dim=64, d_world=16, q_hidden=32)


def _batch(b=3):
    tokens = torch.zeros(b, SEQ_LEN, 5, dtype=torch.long)
    attn = torch.ones(b, SEQ_LEN, dtype=torch.bool)
    world = torch.zeros(b, 28, 3)
    voids = torch.zeros(b, 24)
    bids = torch.zeros(b, N_AUCTION_FEATURES)
    return tokens, attn, world, voids, bids


def test_auction_forward_shapes():
    model = StudentTransformerFullVoidsAuction(voids_hidden=16, bids_hidden=16, **_SMALL)
    tokens, attn, world, voids, bids = _batch(3)
    out = model(tokens, attn, world, voids, bids)
    assert out["belief_logits"].shape == (3, 28, 3)
    assert out["v"].shape == (3,)
    assert out["pi_me_logits"].shape == (3, 7)
    assert out["q"].shape == (3, 7)


def test_auction_uses_the_bids_input():
    """Different auction features must change the belief logits — proves the
    BidsEncoder is actually wired into the pooled state embedding."""
    torch.manual_seed(0)
    model = StudentTransformerFullVoidsAuction(voids_hidden=16, bids_hidden=16, **_SMALL)
    model.eval()
    tokens, attn, world, voids, _ = _batch(1)
    z = model(tokens, attn, world, voids, torch.zeros(1, N_AUCTION_FEATURES))
    nz = model(tokens, attn, world, voids, torch.ones(1, N_AUCTION_FEATURES))
    assert not torch.allclose(z["belief_logits"], nz["belief_logits"])


def _save(tmp_path, model, args):
    p = tmp_path / "adapter.pt"
    torch.save({"model_state": model.state_dict(), "args": args, "epoch": 1}, p)
    return str(p)


def test_load_gus_detects_auction(tmp_path):
    model = StudentTransformerFullVoidsAuction(voids_hidden=16, bids_hidden=16, **_SMALL)
    args = {**_SMALL, "voids_hidden": 16, "bids_hidden": 16, "auction": True}
    loaded, is_voids = load_gus(_save(tmp_path, model, args), device="cpu")
    assert isinstance(loaded, StudentTransformerFullVoidsAuction)
    assert hasattr(loaded, "bids_encoder")
    assert is_voids is True  # the auction model consumes voids too


def test_load_gus_voids_not_misdetected_as_auction(tmp_path):
    """A voids run carries bids_hidden in its args (it has a default), so
    detection must key on the auction flag, not on the bids_hidden key."""
    model = StudentTransformerFullVoids(voids_hidden=16, **_SMALL)
    args = {**_SMALL, "voids_hidden": 16, "bids_hidden": 64, "auction": False}
    loaded, is_voids = load_gus(_save(tmp_path, model, args), device="cpu")
    assert isinstance(loaded, StudentTransformerFullVoids)
    assert not hasattr(loaded, "bids_encoder")
    assert is_voids is True
