"""#26 bridge: GameRecordGPU auction provenance + snapshot->corpus bridge.

Two fast, oracle-free tests:
  * a GameRecordGPU stamped with bids/bidder round-trips through torch.save/load;
  * the bridge (generate_eq_from_snapshots) produces a corpus that
    JointWorldFullDataset loads, carrying the real auction on each game.

An optional, tiny MPS smoke for the real oracle path runs only when MPS is
present (no skip otherwise — the body is gated on hardware, not pytest.skip).
"""
from __future__ import annotations

import torch

from forge.eq.generate.types import DecisionRecordGPU, GameRecordGPU
from forge.oracle.rng import deal_from_seed


def _joint_world_decision(hands, player: int = 0, m: int = 2) -> DecisionRecordGPU:
    """A minimal fixed-sampling decision carrying the joint-world tensors that
    JointWorldFullDataset requires (world_hands [M,3,7], q_per_world [M,7]).

    The stored worlds must be VALID for `player`'s opening decision — the dataset
    filters malformed worlds by default — so each world is the truthful partition
    of the unseen tiles across the opponent seats {left_opp, partner, right_opp}.
    """
    opp_rows = [[int(x) for x in hands[(player + r + 1) % 4]] for r in range(3)]
    world_hands = torch.tensor([opp_rows] * m, dtype=torch.long)  # [m, 3, 7]
    return DecisionRecordGPU(
        player=player,
        e_q=torch.zeros(7),
        action_taken=0,
        legal_mask=torch.tensor([True] + [False] * 6),
        world_hands=world_hands,
        q_per_world=torch.zeros(m, 7),
    )


def test_game_record_auction_provenance_roundtrip(tmp_path):
    hands = deal_from_seed(123)
    rec = GameRecordGPU(
        decisions=[_joint_world_decision(hands)],
        hands=hands,
        decl_id=3,
        bid_value=31,
        bids=(0, 31, 0, 30),
        bidder=1,
    )
    path = tmp_path / "rec.pt"
    torch.save({"results": [rec], "seeds": []}, str(path))
    loaded = torch.load(str(path), weights_only=False)["results"][0]
    assert loaded.bids == (0, 31, 0, 30)
    assert loaded.bidder == 1
    assert loaded.bid_value == 31
    assert loaded.decl_id == 3


def test_bridge_produces_loadable_corpus_with_auction(tmp_path, monkeypatch):
    """Stub the oracle eq generation; assert the bridge output loads via
    JointWorldFullDataset and carries the real auction on the loaded game."""
    from forge.cli import generate_eq_from_snapshots as bridge

    snapshot = {
        "hands": deal_from_seed(7),
        "decl_id": 2,
        "bids": [30, 0, 0, 0],
        "bidder": 0,
        "bid_value": 30,
    }

    def fake_generate(*, model, hands, decl_ids, n_samples, device,
                      save_joint_worlds, bid_values, bidders, forced_actions=None):
        # One record per game, with the joint-world tensors the dataset needs.
        assert save_joint_worlds is True
        # The bridge must thread the real bid winner through so the declarer leads.
        assert bidders == [0]
        # Non-teacher-forced path: no recorded line is forced.
        assert forced_actions is None
        return [
            GameRecordGPU(
                decisions=[_joint_world_decision(hands[i])],
                hands=hands[i],
                decl_id=decl_ids[i],
                bid_value=bid_values[i],
            )
            for i in range(len(hands))
        ]

    monkeypatch.setattr(bridge, "generate_eq_games_gpu", fake_generate)

    results = bridge.generate_corpus(
        model=object(),
        snapshots=[snapshot],
        n_samples=4,
        device="cpu",
        batch_size=8,
    )
    assert len(results) == 1
    assert results[0].bids == (30, 0, 0, 0)
    assert results[0].bidder == 0
    assert results[0].bid_value == 30

    out = tmp_path / "corpus.pt"
    torch.save({"results": results, "seeds": []}, str(out))

    from gus.model.dataset_seq_world import JointWorldFullDataset

    ds = JointWorldFullDataset(str(out), seed=0)
    assert len(ds) == 1  # one joint-world decision
    loaded_game = ds.games[0]
    assert loaded_game.bids == (30, 0, 0, 0)
    assert loaded_game.bidder == 0
    # The dataset item builds without error (tokenize/belief/voids all run).
    item = ds[0]
    assert item["q_per_world"].shape == (7,)
    assert item["world_assignment"].shape == (28, 3)


def test_bridge_oracle_smoke_mps(tmp_path):
    """Tiny real-oracle smoke (1 game) — runs ONLY when MPS is present and a
    checkpoint exists. No pytest.skip: the body short-circuits on hardware."""
    from pathlib import Path

    if not torch.backends.mps.is_available():
        return
    ckpt = (
        Path(__file__).parent.parent
        / "models"
        / "domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt"
    )
    if not ckpt.exists():
        return

    from forge.cli import generate_eq_from_snapshots as bridge
    from forge.eq.oracle import Stage1Oracle
    from gus.model.dataset_seq_world import JointWorldFullDataset

    snapshot = {
        "hands": deal_from_seed(42),
        "decl_id": 0,
        "bids": [30, 0, 0, 0],
        "bidder": 0,
        "bid_value": 30,
    }
    oracle = Stage1Oracle(str(ckpt), device="mps", compile=False)
    results = bridge.generate_corpus(
        model=oracle.model,
        snapshots=[snapshot],
        n_samples=4,
        device="mps",
        batch_size=1,
    )
    assert len(results) == 1
    assert results[0].bids == (30, 0, 0, 0)
    out = tmp_path / "smoke.pt"
    torch.save({"results": results, "seeds": []}, str(out))
    ds = JointWorldFullDataset(str(out), seed=0)
    assert len(ds) > 0  # play produced joint-world decisions
