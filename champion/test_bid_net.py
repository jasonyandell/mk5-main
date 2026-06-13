"""Tests for champion/bid_net.py.

Run with:
    python -m pytest champion/test_bid_net.py -q
"""
from __future__ import annotations

import torch
import pytest

from champion.bid_net import (
    BidNet,
    FEATURE_DIM,
    N_DECLS,
    N_BIDS,
    featurize_hand,
    parse_hand_str,
)


# ---------------------------------------------------------------------- #
#  Test fixtures                                                           #
# ---------------------------------------------------------------------- #

def _sample_hand() -> tuple[int, ...]:
    """Deterministic 7-domino hand."""
    return (0, 5, 10, 15, 20, 25, 27)  # 7 distinct domino IDs


# ---------------------------------------------------------------------- #
#  featurize_hand                                                          #
# ---------------------------------------------------------------------- #

def test_featurize_hand_output_dim() -> None:
    hand = _sample_hand()
    feat = featurize_hand(hand)
    assert feat.shape == (FEATURE_DIM,), f"expected ({FEATURE_DIM},), got {feat.shape}"


def test_featurize_hand_range() -> None:
    """All features should be in a reasonable range (we normalise to ~[0,1])."""
    hand = _sample_hand()
    feat = featurize_hand(hand)
    assert feat.min().item() >= -0.01, "feature below 0"
    assert feat.max().item() <= 1.01, "feature above 1"


def test_featurize_hand_different_hands_differ() -> None:
    h1 = (0, 1, 2, 3, 4, 5, 6)
    h2 = (21, 22, 23, 24, 25, 26, 27)
    f1 = featurize_hand(h1)
    f2 = featurize_hand(h2)
    assert not torch.allclose(f1, f2), "different hands should produce different features"


def test_parse_hand_str_roundtrip() -> None:
    hand_str = "6-4,5-5,4-3,3-2,2-1,1-0,0-0"
    hand = parse_hand_str(hand_str)
    assert len(hand) == 7
    feat = featurize_hand(hand)
    assert feat.shape == (FEATURE_DIM,)


# ---------------------------------------------------------------------- #
#  BidNet forward                                                          #
# ---------------------------------------------------------------------- #

def test_bid_net_forward_shape() -> None:
    model = BidNet()
    B = 4
    x = torch.rand(B, FEATURE_DIM)
    out = model(x)
    assert out.shape == (B, N_DECLS, N_BIDS), f"expected ({B}, {N_DECLS}, {N_BIDS}), got {out.shape}"


def test_bid_net_output_sigmoid_range() -> None:
    model = BidNet()
    x = torch.rand(8, FEATURE_DIM)
    out = model(x)
    assert out.min().item() >= 0.0 - 1e-6, "sigmoid output below 0"
    assert out.max().item() <= 1.0 + 1e-6, "sigmoid output above 1"


def test_bid_net_single_sample() -> None:
    """Single-sample forward (batch=1)."""
    model = BidNet()
    hand = _sample_hand()
    x = featurize_hand(hand).unsqueeze(0)  # [1, 63]
    out = model(x)
    assert out.shape == (1, N_DECLS, N_BIDS)
    assert 0.0 <= out.min().item() <= out.max().item() <= 1.0


# ---------------------------------------------------------------------- #
#  Tiny overfit smoke test                                                 #
# ---------------------------------------------------------------------- #

def test_loss_decreases_on_two_sample_overfit() -> None:
    """BidNet should be able to overfit a 2-sample dataset.

    Uses MSE loss so the true minimum is 0 (BCE on random targets has an
    irreducible entropy floor that prevents near-zero convergence).
    We verify: (a) loss strictly decreases; (b) MSE falls below 0.01 on
    just 2 samples, confirming gradient flow and sufficient model capacity.
    """
    torch.manual_seed(0)
    model = BidNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = torch.nn.MSELoss()

    # Two synthetic samples with structured targets (monotone-ish, realistic range)
    x = torch.rand(2, FEATURE_DIM)
    # Targets: decreasing across bids (realistic p_make shape), in [0, 1]
    y_raw = torch.linspace(0.9, 0.1, N_DECLS * N_BIDS).unsqueeze(0).expand(2, -1).clone()
    y_raw[1] = torch.linspace(0.7, 0.05, N_DECLS * N_BIDS)  # different second sample

    # Capture initial loss
    model.eval()
    with torch.no_grad():
        pred0 = model(x).view(2, -1)
        loss0 = loss_fn(pred0, y_raw).item()

    # Train for 500 steps
    model.train()
    for _ in range(500):
        pred = model(x).view(2, -1)
        loss = loss_fn(pred, y_raw)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_final = model(x).view(2, -1)
        loss_final = loss_fn(pred_final, y_raw).item()

    assert loss_final < loss0, (
        f"Loss should decrease on 2-sample overfit: {loss0:.4f} → {loss_final:.4f}"
    )
    # Model should achieve near-zero MSE on just 2 samples
    assert loss_final < 0.01, (
        f"Expected MSE < 0.01 on 2-sample overfit, got {loss_final:.4f}"
    )
