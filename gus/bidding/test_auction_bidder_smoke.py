"""Regression: the gus bidder simulation must not crash on an auction model.

load_gus returns is_voids=True for the auction student (it consumes voids too),
so the bidder's forward dispatch has to detect the auction head (bids_encoder)
and feed the all-zero pre-auction bids vector — otherwise the 5-arg forward is
called with 4 args and raises TypeError (caught by the #24/#26 review)."""
import torch

from gus.bidding.simulate import simulate_all_gus_batch
from gus.model.student import StudentTransformerFullVoidsAuction


def test_auction_model_runs_through_the_bidder():
    torch.manual_seed(0)
    model = StudentTransformerFullVoidsAuction(
        d_model=32, n_heads=2, n_layers=1, ff_dim=64,
        d_world=16, q_hidden=32, voids_hidden=16, bids_hidden=16,
    )
    model.eval()
    hand = [0, 2, 5, 9, 14, 20, 27]  # 7 distinct domino ids
    out = simulate_all_gus_batch(
        model, is_voids=True, bidder_hand=hand, trump_ids=[3],
        n_per_trump=1, device="cpu", seed=1,
    )
    assert set(out.keys()) == {3}
    assert out[3].shape == (1,)
