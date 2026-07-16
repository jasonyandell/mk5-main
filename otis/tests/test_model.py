"""otis/tests/test_model.py — OtisNet shape parity, export bit-exactness,
fate-class derivation, and consistency-penalty math."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from champion.margin_net import FEATURE_DIM, MarginNet
from champion.value_bidder import load_margin_net
from otis.model import (
    MODE_IDX,
    CAPTURE_IDX,
    OtisNet,
    consistency_terms,
    fate_class,
)

ROOT = Path(__file__).resolve().parent.parent.parent
FATES_PARQUET = ROOT / "scratch" / "otis-night" / "master_fates.parquet"
# The 8-class fate contract, capture-major then mode-minor. Previously read
# from scratch/otis-night/fate_base_rates.json (generated, did not survive the
# scratch clear); the order is the load-bearing fact, so it lives here.
FATE_CLASSES = [
    f"{cap}|{mode}"
    for cap in ("bidding_team", "opp_of_bidder")
    for mode in ("led", "followed", "trumped_in", "sloughed")
]


# --------------------------------------------------------------------------- #
# (4) control == MarginNet shape parity                                         #
# --------------------------------------------------------------------------- #


def test_control_matches_marginnet_shape():
    otis = OtisNet(in_dim=FEATURE_DIM, treatment=False)
    margin = MarginNet(in_dim=FEATURE_DIM)
    exported = otis.export_margin_state_dict()
    ref = margin.state_dict()
    # exact same parameter keys
    assert set(exported.keys()) == set(ref.keys())
    # exact same per-key shapes
    for k in ref:
        assert exported[k].shape == ref[k].shape, k
    # trunk+pricing forward has MarginNet's [B,43] pricing shape
    x = torch.randn(7, FEATURE_DIM)
    assert otis.pricing_logits(x).shape == margin(x).shape == (7, 43)


def test_treatment_forward_shapes():
    otis = OtisNet(in_dim=FEATURE_DIM, treatment=True)
    x = torch.randn(4, FEATURE_DIM)
    out = otis(x)
    assert out["pricing"].shape == (4, 43)
    assert out["fate"].shape == (4, 5, 8)
    assert out["trick"].shape == (4, 8)


# --------------------------------------------------------------------------- #
# (1) export bit-exactness (both arms)                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("treatment", [False, True])
def test_export_bit_exact(tmp_path, treatment):
    torch.manual_seed(123)
    otis = OtisNet(in_dim=FEATURE_DIM, treatment=treatment)
    otis.eval()
    state = otis.export_margin_state_dict()
    out = tmp_path / "exported.pt"
    torch.save({"model_state": state, "feature_dim": FEATURE_DIM}, out)

    # STRICT reload through the exact consumer path.
    margin = load_margin_net(out, device="cpu")
    assert isinstance(margin, MarginNet)

    x = torch.randn(100, FEATURE_DIM)
    with torch.no_grad():
        a = otis.pricing_logits(x)
        b = margin(x)
    assert torch.equal(a, b), f"max abs diff {(a - b).abs().max().item():e}"


def test_export_bidder_module(tmp_path):
    """End-to-end through otis.export_bidder.export."""
    from otis.export_bidder import export

    torch.manual_seed(7)
    otis = OtisNet(in_dim=FEATURE_DIM, treatment=True)
    ck = tmp_path / "otis.pt"
    torch.save({"model_state": otis.state_dict(), "feature_dim": FEATURE_DIM,
                "arm": "treatment", "treatment": True}, ck)
    res = export(ck, tmp_path / "otis_v0_treatment.pt", n_verify=100)
    assert res["strict_load_ok"] and res["bit_exact"]
    assert res["max_abs_diff"] == 0.0


# --------------------------------------------------------------------------- #
# (2) fate-class derivation on hand-derived rows                                #
# --------------------------------------------------------------------------- #


def test_fate_class_matches_base_rate_ordering():
    """The 8-class encoding must match the base-rate `classes` order —
    otherwise the P2 NLL compares model logits to the wrong base-rate cells.

    The contract is spelled out as a literal (capture-major, mode-minor); it
    previously came from scratch/otis-night/fate_base_rates.json, which was a
    generated file (circular) and did not survive the scratch clear."""
    classes = FATE_CLASSES
    # 3 hand-picked (capture, mode) rows with a hand-computed class index.
    cases = [
        ("bidding_team", "led", 0),        # 0*4 + 0
        ("opp_of_bidder", "trumped_in", 6),  # 1*4 + 2
        ("bidding_team", "sloughed", 3),   # 0*4 + 3
    ]
    for cap, mode, expected in cases:
        assert fate_class(cap, mode) == expected
        assert classes[fate_class(cap, mode)] == f"{cap}|{mode}"


@pytest.mark.skipif(not FATES_PARQUET.exists(), reason="master_fates.parquet absent")
def test_fate_class_on_real_rows():
    """3 real fate rows: the derived class must round-trip its label strings."""
    classes = FATE_CLASSES
    f = pd.read_parquet(FATES_PARQUET, columns=["capture_bidding", "played_mode"])
    for _, r in f.head(3).iterrows():
        cid = fate_class(r["capture_bidding"], r["played_mode"])
        assert cid == CAPTURE_IDX[r["capture_bidding"]] * 4 + MODE_IDX[r["played_mode"]]
        assert classes[cid] == f"{r['capture_bidding']}|{r['played_mode']}"


# --------------------------------------------------------------------------- #
# (3) consistency-penalty math on synthetic pdfs                                #
# --------------------------------------------------------------------------- #


def _peaked(n_rows, n_class, idx, hot=60.0):
    """Logits with ~all softmax mass on class ``idx`` (per row)."""
    z = torch.zeros(n_rows, n_class)
    z[:, idx] = hot
    return z


def test_consistency_all_captured():
    # E[price]=20; all 5 tiles captured by my team (class 0) → Σ value = 35;
    # E[tricks]=5 → target 40; |20-40| = 20.
    pricing = _peaked(3, 43, 20)
    fate = torch.stack([_peaked(3, 8, 0) for _ in range(5)], dim=1)  # [3,5,8], class 0=my|led
    trick = _peaked(3, 8, 5)
    cons = consistency_terms(pricing, fate, trick)
    assert torch.allclose(cons, torch.full((3,), 20.0), atol=1e-3)


def test_consistency_tile0_lost():
    # E[price]=20; 5-5 (value 10) NOT captured (class 4 = opp|led), others captured
    # → Σ value = 25; E[tricks]=5 → target 30; |20-30| = 10.
    pricing = _peaked(2, 43, 20)
    heads = [_peaked(2, 8, 4)] + [_peaked(2, 8, 0) for _ in range(4)]
    fate = torch.stack(heads, dim=1)  # [2,5,8]
    trick = _peaked(2, 8, 5)
    cons = consistency_terms(pricing, fate, trick)
    assert torch.allclose(cons, torch.full((2,), 10.0), atol=1e-3)


def test_consistency_zero_when_matched():
    # E[price]=35 = Σ value(35) + E[tricks](0) → penalty 0.
    pricing = _peaked(2, 43, 35)
    fate = torch.stack([_peaked(2, 8, 0) for _ in range(5)], dim=1)
    trick = _peaked(2, 8, 0)
    cons = consistency_terms(pricing, fate, trick)
    assert torch.allclose(cons, torch.zeros(2), atol=1e-3)
