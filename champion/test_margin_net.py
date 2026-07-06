"""MarginNet (jud v0, #32) — realized-value head. CPU, seconds.

Covers the leakage defense (canonical auction + level-blindness), the
distribution helpers, the dataset (built from a tiny live arena match), and a
training-smoke gradient check.
"""
from __future__ import annotations

import json

import torch

from arena.engine import ArenaConfig
from arena.bidders import HeuristicBidder
from arena.match import run_match, snapshot_rows
from arena.play import RandomPlay

from champion.margin_net import (
    AUCTION_DIM,
    CANON_BID,
    FEATURE_DIM,
    HAND_DIM,
    MarginDataset,
    MarginNet,
    N_POINTS,
    canonical_auction,
    exceedance,
    featurize,
    featurize_snapshot,
    split_of,
)


# --------------------------------------------------------------------- #
#  1. canonical_auction: later-seat masking + winner constant            #
# --------------------------------------------------------------------- #

def test_canonical_auction_masks_later_seats_and_stamps_winner():
    # Auction order is dealer+1, dealer+2, dealer+3, dealer. Enumerate a few
    # (dealer, bidder) combos and check: earlier bids kept, bidder := CANON_BID,
    # later seats := 0.
    real = [11, 32, 40, 30]  # arbitrary distinct positive bids, seat-ordered

    for dealer in range(4):
        order = [(dealer + s) % 4 for s in (1, 2, 3, 0)]
        for bidder in range(4):
            canon = canonical_auction(real, bidder, dealer)
            bidder_pos = order.index(bidder)
            # Bidder's own entry is the constant, level erased.
            assert canon[bidder] == CANON_BID
            for pos, seat in enumerate(order):
                if pos < bidder_pos:
                    assert canon[seat] == real[seat], (dealer, bidder, seat)
                elif pos > bidder_pos:
                    assert canon[seat] == 0, (dealer, bidder, seat)


def test_canonical_auction_dealer0_bidder3_keeps_early_two():
    # dealer 0 → order seats 1,2,3,0; bidder seat 3 hears seats 1 & 2 only.
    real = [0, 33, 35, 30]  # seat 3 is the (real) winner at 30
    canon = canonical_auction(real, bidder=3, dealer=0)
    assert canon == [0, 33, 35, CANON_BID]  # seat 0 (last, after bidder) masked → 0


def test_canonical_auction_clamps_pass_and_not_bid():
    # -1 (not-yet-bid) and 0 (pass) both canonicalize to 0 on earlier seats.
    canon = canonical_auction([-1, 0, 0, 30], bidder=3, dealer=0)
    assert canon == [0, 0, 0, CANON_BID]


# --------------------------------------------------------------------- #
#  2. Level-blindness: the winning bid value never enters the features   #
# --------------------------------------------------------------------- #

def test_featurization_is_blind_to_winning_bid_level():
    hand = (0, 1, 2, 3, 4, 5, 6)
    # Same snapshot, two different winning levels (30 vs 42) at seat 3.
    x_lo = featurize(hand, [0, 0, 0, 30], bidder=3, dealer=0, decl_id=5)
    x_hi = featurize(hand, [0, 0, 0, 42], bidder=3, dealer=0, decl_id=5)
    assert torch.equal(x_lo, x_hi)
    assert x_lo.shape == (FEATURE_DIM,) == (HAND_DIM + AUCTION_DIM,)


def test_featurize_snapshot_matches_featurize():
    snap = {
        "hands": [[7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6],
                  [14, 15, 16, 17, 18, 19, 20], [21, 22, 23, 24, 25, 26, 27]],
        "bids": [0, 30, 0, 0], "bidder": 1, "dealer": 0, "decl_id": 3,
        "bidder_team_pts": 24, "seed": 1, "hand_idx": 0,
    }
    x = featurize_snapshot(snap)
    x_ref = featurize(tuple(snap["hands"][1]), snap["bids"], 1, 0, 3)
    assert torch.equal(x, x_ref)


# --------------------------------------------------------------------- #
#  3. exceedance: monotone, P(pts ≥ 0) == 1                              #
# --------------------------------------------------------------------- #

def test_exceedance_monotone_and_normalized():
    torch.manual_seed(0)
    logits = torch.randn(5, N_POINTS)
    exc = exceedance(logits)
    assert exc.shape == (5, N_POINTS)
    # P(pts ≥ 0) == 1 for every row.
    assert torch.allclose(exc[:, 0], torch.ones(5), atol=1e-5)
    # Monotone non-increasing across thresholds.
    diffs = exc[:, 1:] - exc[:, :-1]
    assert (diffs <= 1e-6).all()
    # 1-D input path.
    exc1 = exceedance(logits[0])
    assert exc1.shape == (N_POINTS,)
    assert abs(float(exc1[0]) - 1.0) < 1e-5


# --------------------------------------------------------------------- #
#  Tiny live-arena fixture (mirrors arena/test_snapshots.py)             #
# --------------------------------------------------------------------- #

def _tiny_snapshot_file(tmp_path, n_games=2):
    result = run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
        n_games=n_games, cfg=ArenaConfig(marks_to_win=2, base_seed=7),
    )
    snaps = snapshot_rows(result)
    path = tmp_path / "snaps.json"
    path.write_text(json.dumps({"snapshots": snaps, "metadata": {"n": len(snaps)}}))
    return path, snaps


# --------------------------------------------------------------------- #
#  4. Dataset: sample count, dedupe across halves, y in range           #
# --------------------------------------------------------------------- #

def test_dataset_from_tiny_match(tmp_path):
    path, snaps = _tiny_snapshot_file(tmp_path, n_games=2)
    assert snaps, "tiny match should produce contracted hands"

    ds = MarginDataset(path, split="all")
    # Paired halves replay identical seeds with identical (HeuristicBidder)
    # auctions → exact duplicates collapse. So deduped < raw.
    from champion.margin_net import _dedup_key
    n_unique = len({_dedup_key(s) for s in snaps})
    assert len(ds) == n_unique
    assert len(ds) < len(snaps), "dedupe must remove the paired-half duplicates"

    # Every sample: 91-dim x, y a class index in 0..42.
    for x, y in ds:
        assert x.shape == (FEATURE_DIM,)
        assert 0 <= int(y) <= 42

    # The three splits partition the deduped set with no overlap.
    tr = MarginDataset(path, split="train")
    va = MarginDataset(path, split="val")
    te = MarginDataset(path, split="test")
    assert len(tr) + len(va) + len(te) == len(ds)
    all_keys = set(tr.keys) | set(va.keys) | set(te.keys)
    assert len(all_keys) == len(ds)


def test_split_of_is_deterministic_and_partitions():
    counts = {"train": 0, "val": 0, "test": 0}
    for seed in range(2000):
        s = split_of(seed, seed % 5)
        assert s == split_of(seed, seed % 5)  # stable
        counts[s] += 1
    # ~90/5/5; just assert every bucket is populated and train dominates.
    assert counts["train"] > counts["val"] > 0
    assert counts["test"] > 0


# --------------------------------------------------------------------- #
#  5. Training smoke: CE decreases over 30 gradient steps                #
# --------------------------------------------------------------------- #

def test_training_smoke_ce_decreases(tmp_path):
    path, _ = _tiny_snapshot_file(tmp_path, n_games=2)
    ds = MarginDataset(path, split="all")
    xs = torch.stack([x for x, _ in ds])
    ys = torch.stack([y for _, y in ds])

    torch.manual_seed(0)
    model = MarginNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    first = None
    last = None
    for step in range(30):
        logits = model(xs)
        loss = loss_fn(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())
    assert last < first, f"CE did not decrease: {first:.4f} → {last:.4f}"


def test_pmake_table_shape_and_bounds(tmp_path):
    from forge.bidding.schema import EVAL_DECLS

    model = MarginNet()
    table = model.pmake_table(
        hand=(0, 1, 2, 3, 4, 5, 6), bids=[0, 0, 0, 30], bidder=3, dealer=0,
    )
    assert set(table) == set(EVAL_DECLS)
    for decl, row in table.items():
        assert set(row) == set(range(30, 43))
        for t, p in row.items():
            assert 0.0 <= p <= 1.0
        # Exceedance is monotone non-increasing in the threshold.
        vals = [row[t] for t in range(30, 43)]
        assert all(a >= b - 1e-6 for a, b in zip(vals, vals[1:]))
