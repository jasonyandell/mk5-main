"""Capacity control (#24): shuffle_bids sources each game's auction feature from
a different game (cyclic derangement), leaving the belief target untouched — so
the auction student keeps its BidsEncoder capacity but loses the real auction↔deal
correlation. Used to prove +2.42pp is auction *information*, not added params."""
import torch

from forge.eq.generate.types import DecisionRecordGPU, GameRecordGPU
from gus.model.dataset_seq_world import JointWorldFullDataset


def _game(bids, bidder, bid_value, decl_id):
    hands = [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13],
             [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]]
    # A valid world for P0's opening decision: unseen = {7..27} split across the
    # three opponent seats {left_opp=P1, partner=P2, right_opp=P3}. Must be a real
    # partition — JointWorldFullDataset now filters malformed worlds by default,
    # and a degenerate all-zero placeholder would be dropped entirely.
    valid_world = [[7, 8, 9, 10, 11, 12, 13],
                   [14, 15, 16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25, 26, 27]]
    dec = DecisionRecordGPU(
        player=0, e_q=torch.zeros(7), action_taken=0,
        legal_mask=torch.ones(7, dtype=torch.bool),
        world_hands=torch.tensor([valid_world, valid_world], dtype=torch.long),
        q_per_world=torch.zeros(2, 7),
    )
    g = GameRecordGPU(decisions=[dec], hands=hands, decl_id=decl_id)
    g.bids, g.bidder, g.bid_value = bids, bidder, bid_value
    return g


def test_shuffle_bids_sources_next_game_keeps_target(tmp_path):
    games = [
        _game((30, 0, 0, 0), 0, 30, 1),
        _game((0, 0, 32, 0), 2, 32, 4),
        _game((0, 33, 0, 0), 1, 33, 6),
    ]
    p = tmp_path / "corpus.pt"
    torch.save({"results": games, "seeds": []}, str(p))

    real = JointWorldFullDataset(str(p), seed=1)
    shuf = JointWorldFullDataset(str(p), seed=1, shuffle_bids=True)

    assert shuf.bid_perm == [1, 2, 0]                       # cyclic shift by 1
    # game 0's shuffled auction comes from game 1 (decorrelated from its own deal)
    assert not torch.equal(real[0]["bids"], shuf[0]["bids"])
    assert torch.equal(shuf[0]["bids"], real[1]["bids"])
    # everything else is untouched — only the auction source moved
    assert torch.equal(real[0]["belief_target"], shuf[0]["belief_target"])
    assert torch.equal(real[0]["voids"], shuf[0]["voids"])
    assert torch.equal(real[0]["world_assignment"], shuf[0]["world_assignment"])
