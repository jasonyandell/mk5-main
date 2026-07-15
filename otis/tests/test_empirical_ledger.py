"""Tests for otis.analysis.empirical_ledger — the W2 measurement primitives.

Uses tiny synthetic hand sets so the independence composition, tail arithmetic,
and base-rate/entropy math are checkable by hand — no reliance on the corpus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from otis.analysis.empirical_ledger import (
    TILE_PIPS,
    correlation_matrix,
    fate_base_rates,
    independence_pmf,
    measure_p3,
    p3_slice,
    tail_ge,
)


def _hand(x55, x64, x50, x41, x32, T, decl_id=0):
    pts = 10 * x55 + 10 * x64 + 5 * x50 + 5 * x41 + 5 * x32 + T
    return {
        "X_5-5": x55, "X_6-4": x64, "X_5-0": x50, "X_4-1": x41, "X_3-2": x32,
        "bidder_tricks": T, "bidder_team_pts": pts, "bid_value": 30,
        "source": "selfplay", "decl_id": decl_id,
    }


def test_independence_pmf_normalizes_and_matches_mean():
    # Two hands: all-captured/7 tricks (42) and none/0 tricks (0).
    caps = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=float)
    T = np.array([7, 0], dtype=float)
    pmf = independence_pmf(caps, T)
    assert abs(pmf.sum() - 1.0) < 1e-9
    # Each marginal p_t = 0.5, E[T] = 3.5 -> E[points] = 0.5*35 + 3.5 = 21.
    exp = sum(i * pmf[i] for i in range(len(pmf)))
    assert abs(exp - 21.0) < 1e-9


def test_tail_ge_and_p3_slice_direction():
    # Perfectly correlated: 50% sweep (42), 50% zero. Empirical P(>=30)=0.5.
    hands = pd.DataFrame([_hand(1, 1, 1, 1, 1, 7) for _ in range(50)]
                         + [_hand(0, 0, 0, 0, 0, 0) for _ in range(50)])
    caps = hands[[f"X_{p}" for p in TILE_PIPS]].to_numpy(float)
    T = hands["bidder_tricks"].to_numpy(float)
    pts = hands["bidder_team_pts"].to_numpy(float)
    res = p3_slice(pts, caps, T, 30)
    assert abs(res["empirical_tail"] - 0.5) < 1e-9
    # Independence (all p=0.5, T uniform on {0,7}) spreads mass across 0..42, so
    # its P(>=30) < 0.5 -> independence underprices this high threshold (diff>0).
    assert res["diff_pp"] > 0
    # tail_ge consistency with empirical.
    assert abs(tail_ge(np.eye(43)[42], 30) - 1.0) < 1e-9


def test_correlation_matrix_shape_and_diag():
    hands = pd.DataFrame([_hand(1, 1, 1, 1, 1, 7), _hand(0, 0, 0, 0, 0, 0),
                          _hand(1, 0, 1, 0, 1, 4), _hand(0, 1, 0, 1, 0, 3)])
    corr = correlation_matrix(hands)
    assert corr.shape == (6, 6)
    assert np.allclose(np.diag(corr), 1.0)


def test_fate_base_rates_probs_and_entropy():
    # 4 fate rows for one tile, two classes each 50% -> entropy ln 2.
    rows = []
    for cap, mode in [("bidding_team", "led"), ("bidding_team", "led"),
                      ("opp_of_bidder", "sloughed"), ("opp_of_bidder", "sloughed")]:
        rows.append({"tile": "5-5", "capture_bidding": cap, "played_mode": mode,
                     "decl_id": 0, "decl_name": "blanks"})
    # Fill the other four tiles so the per-tile loop has data (one class each).
    for t in ("6-4", "5-0", "4-1", "3-2"):
        for _ in range(4):
            rows.append({"tile": t, "capture_bidding": "bidding_team",
                         "played_mode": "led", "decl_id": 0, "decl_name": "blanks"})
    br = fate_base_rates(pd.DataFrame(rows))
    o = br["overall"]["5-5"]
    assert o["n"] == 4
    assert abs(o["probs"]["bidding_team|led"] - 0.5) < 1e-9
    assert abs(o["entropy_nats"] - np.log(2)) < 1e-9
    # A pure single-class tile has zero entropy.
    assert abs(br["overall"]["6-4"]["entropy_nats"]) < 1e-9


def test_measure_p3_smoke():
    hands = pd.DataFrame([_hand(1, 1, 1, 1, 1, 7) for _ in range(600)]
                         + [_hand(0, 0, 0, 0, 0, 0) for _ in range(600)])
    out = measure_p3(hands, min_n=500)
    assert "bid_30" in out["slices"]
    assert out["slices"]["bid_30"]["n"] == 1200
    assert len(out["correlation_matrix"]) == 6
    assert "crossover_threshold" in out["tail_curve"]
