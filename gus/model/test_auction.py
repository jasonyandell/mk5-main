"""Auction feature vector for the belief head (rung #24), pure tensor core."""
import torch

from gus.model.auction import N_AUCTION_FEATURES, auction_feature_vector


def test_none_bids_is_all_zero():
    """No recorded auction (the seed-imposed corpus) → zero feature, so the
    BidsEncoder contributes nothing and the model degrades to play+voids."""
    feat = auction_feature_vector(None, None, None, decl_id=4, current_player=0)
    assert feat.shape == (N_AUCTION_FEATURES,)
    assert torch.count_nonzero(feat) == 0


def test_winner_and_levels_from_my_pov():
    # Seat 2 won at 35 declaring fours (decl 4); seat 0 bid 31; seats 1,3 passed.
    bids = (31, 0, 35, 0)
    feat = auction_feature_vector(bids, bidder=2, bid_value=35, decl_id=4, current_player=0)
    # relative seat 0 == me == abs 0 (bid 31)
    assert feat[0] == torch.tensor((31 - 30) / 12).float()  # bid_norm
    assert feat[1] == 0.0   # not passed
    assert feat[2] == 1.0   # made a bid
    assert feat[3] == 0.0   # not winner
    # relative seat 2 == partner == abs 2 (winner at 35)
    assert feat[4 * 2 + 1] == 0.0                                   # not passed
    assert feat[4 * 2 + 2] == 1.0                                   # made a bid
    assert feat[4 * 2 + 3] == 1.0                                   # winner
    assert abs(float(feat[4 * 2 + 0]) - (35 - 30) / 12) < 1e-6      # bid_norm
    # relative seat 1 == abs 1 (passed)
    assert feat[4 * 1 + 1] == 1.0   # passed
    assert feat[4 * 1 + 2] == 0.0   # did not bid
    # globals
    assert abs(float(feat[16]) - (35 - 30) / 12) < 1e-6   # winning bid norm
    assert feat[17] == 0.0                                # not a marks bid
    # decl one-hot: fours == decl 4 lights up slot 18+4, nothing else in the tail
    assert feat[18 + 4] == 1.0
    assert float(feat[18:].sum()) == 1.0


def test_pov_rotation_moves_the_winner_slot():
    """The winner flag must land on the relative seat matching current_player."""
    bids = (31, 0, 35, 0)
    f0 = auction_feature_vector(bids, bidder=2, bid_value=35, decl_id=4, current_player=0)
    f2 = auction_feature_vector(bids, bidder=2, bid_value=35, decl_id=4, current_player=2)
    # POV 0: winner (abs 2) is at relative seat 2.
    assert f0[4 * 2 + 3] == 1.0
    # POV 2: winner (abs 2) is "me" at relative seat 0.
    assert f2[4 * 0 + 3] == 1.0


def test_marks_bid_flag():
    feat = auction_feature_vector((0, 84, 0, 0), bidder=1, bid_value=84, decl_id=7, current_player=0)
    assert feat[17] == 1.0                # is_marks
    assert float(feat[16]) == 1.0         # marks clamp to 1.0
    # the marks bidder's normalized bid also clamps to 1.0
    assert float(feat[4 * 1 + 0]) == 1.0
    assert feat[18 + 7] == 1.0            # doubles-trump decl one-hot


def test_notrump_decl_one_hot():
    # notrump is decl 9 (the top of the 10-wide one-hot tail).
    feat = auction_feature_vector((30, 0, 0, 0), bidder=0, bid_value=30, decl_id=9, current_player=0)
    assert feat[18 + 9] == 1.0
    assert float(feat[18:].sum()) == 1.0
