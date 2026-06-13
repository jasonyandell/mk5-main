"""Auction belief query path (#24): belief_logits_for_states feeds the live
bid_state to the auction student, and the voids path is unchanged."""
import torch

from champion.belief import belief_logits_for_states
from forge.zeb.game import new_game
from gus.model.student import (
    StudentTransformerFullVoids,
    StudentTransformerFullVoidsAuction,
)

_SMALL = dict(d_model=32, n_heads=2, n_layers=1, ff_dim=64, d_world=16, q_hidden=32)


def test_auction_model_queries_over_real_state():
    model = StudentTransformerFullVoidsAuction(voids_hidden=16, bids_hidden=16, **_SMALL)
    model.eval()
    states = [new_game(seed) for seed in (1, 2, 3)]
    logP = belief_logits_for_states(model, is_voids=True, states=states, device="cpu")
    assert logP.shape == (3, 28, 3)
    assert torch.isfinite(logP).all()
    # log-softmax over the 3 seats sums to 0 in prob space (≈1.0).
    assert torch.allclose(logP.exp().sum(dim=-1), torch.ones(3, 28), atol=1e-5)


def test_voids_model_path_unchanged():
    model = StudentTransformerFullVoids(voids_hidden=16, **_SMALL)
    model.eval()
    states = [new_game(seed) for seed in (4, 5)]
    logP = belief_logits_for_states(model, is_voids=True, states=states, device="cpu")
    assert logP.shape == (2, 28, 3)
    assert torch.isfinite(logP).all()
