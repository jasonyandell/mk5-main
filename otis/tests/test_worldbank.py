"""Unit tests for the world-bank clustering + bimodality core on synthetic
worlds. These exercise the pure functions with no corpus / model dependency."""

from __future__ import annotations

import numpy as np

from otis.analysis.worldbank import (
    BimodalResult,
    Cluster,
    bimodality_split,
    cluster_worlds,
    belief_weights,
    world_seat_matrix,
    id_to_pips,
    pips_to_id,
    THREE_TWO,
    OTHER_COUNT_TILES,
)

import torch


# --- domino id sanity --------------------------------------------------------

def test_domino_ids():
    assert pips_to_id(3, 2) == 8 == THREE_TWO
    assert pips_to_id(5, 5) == 20
    assert pips_to_id(6, 4) == 25
    assert pips_to_id(5, 0) == 15
    assert pips_to_id(4, 1) == 11
    for d in range(28):
        hi, lo = id_to_pips(d)
        assert pips_to_id(hi, lo) == d
    assert set(OTHER_COUNT_TILES) == {20, 25, 15, 11}


# --- clustering --------------------------------------------------------------

def test_cluster_worlds_basic_grouping():
    # Three distinct contexts, equal mass, distinct q levels.
    codes = [(0, 0, 0, 0)] * 4 + [(1, 1, 1, 1)] * 4 + [(2, 2, 2, 2)] * 2
    q = np.array([10, 10, 10, 10, 30, 30, 30, 30, 50, 50], dtype=float)
    w = np.full(10, 0.1)
    clusters = cluster_worlds(codes, q, w, w, merge_threshold=0.0)
    assert len(clusters) == 3
    # sorted ascending by mean
    means = [c.mean_belief for c in clusters]
    assert means == sorted(means)
    assert abs(clusters[0].mean_belief - 10) < 1e-9
    assert abs(clusters[-1].mean_belief - 50) < 1e-9
    assert abs(sum(c.mass_belief for c in clusters) - 1.0) < 1e-9


def test_cluster_worlds_merges_small_into_other():
    # One dominant cluster (90% mass) + two tiny ones (5% each) -> merged.
    codes = [(0, 0, 0, 0)] * 18 + [(1, 1, 1, 1)] * 1 + [(2, 2, 2, 2)] * 1
    q = np.array([10.0] * 18 + [99.0] + [-99.0])
    w = np.full(20, 0.05)  # each world 5%
    clusters = cluster_worlds(codes, q, w, w, merge_threshold=0.06)
    keys = {c.key for c in clusters}
    assert ("other",) in keys
    # dominant cluster kept, the two 5% ones folded into 'other'
    assert len(clusters) == 2
    other = [c for c in clusters if c.key == ("other",)][0]
    assert other.n_worlds == 2
    assert abs(other.mass_belief - 0.10) < 1e-9


def test_cluster_belief_vs_uniform_weight_differ():
    codes = [(0,)] * 2 + [(1,)] * 2
    q = np.array([0.0, 0.0, 100.0, 100.0])
    w_belief = np.array([0.4, 0.4, 0.1, 0.1])   # weights the low cluster
    w_uniform = np.full(4, 0.25)
    clusters = cluster_worlds(codes, q, w_belief, w_uniform, merge_threshold=0.0)
    by_key = {c.key: c for c in clusters}
    assert abs(by_key[(0,)].mass_belief - 0.8) < 1e-9
    assert abs(by_key[(0,)].mass_uniform - 0.5) < 1e-9


# --- bimodality --------------------------------------------------------------

def test_bimodality_clear_two_modes():
    # Two balanced clusters 40 points apart -> bimodal.
    means = np.array([5.0, 45.0])
    masses = np.array([0.5, 0.5])
    r = bimodality_split(means, masses)
    assert isinstance(r, BimodalResult)
    assert r.is_bimodal
    assert abs(r.gap - 40.0) < 1e-9
    assert r.low_mass >= 0.2 and r.high_mass >= 0.2


def test_bimodality_gap_too_small():
    # Balanced but only 4 points apart -> fails the >=10 gap.
    means = np.array([20.0, 24.0])
    masses = np.array([0.5, 0.5])
    r = bimodality_split(means, masses)
    assert not r.is_bimodal


def test_bimodality_mass_too_small():
    # Big gap but the high mode is only 5% mass -> fails the >=20% mass rule.
    means = np.array([10.0, 60.0])
    masses = np.array([0.95, 0.05])
    r = bimodality_split(means, masses)
    assert not r.is_bimodal


def test_bimodality_single_cluster_not_bimodal():
    r = bimodality_split(np.array([12.0]), np.array([1.0]))
    assert not r.is_bimodal
    assert r.n_clusters == 1


def test_bimodality_three_clusters_finds_best_split():
    # low pair near 0, high single near 40, all >=20% mass.
    means = np.array([0.0, 3.0, 40.0])
    masses = np.array([0.3, 0.3, 0.4])
    r = bimodality_split(means, masses)
    assert r.is_bimodal
    # best split puts the two low clusters together vs the high one
    assert r.split_k == 2
    assert r.high_mass >= 0.2
    # low group mean ~1.5, high ~40
    assert abs(r.high_mean - 40.0) < 1e-9
    assert r.gap >= 10.0


def test_bimodality_prefers_mass_valid_over_larger_unbalanced_gap():
    # A tiny extreme cluster would give a huge gap but fails mass; the balanced
    # split must be chosen instead.
    means = np.array([0.0, 20.0, 200.0])
    masses = np.array([0.45, 0.45, 0.10])
    r = bimodality_split(means, masses)
    # mass-valid splits: k=1 (low .45 / high .55) and NOT k=2 (high .10 <.20).
    # So chosen split is k=1, gap = mean(high group {20,200 weighted}) - 0.
    assert r.split_k == 1
    assert r.low_mass >= 0.2 and r.high_mass >= 0.2


# --- belief weighting --------------------------------------------------------

def test_world_seat_matrix_and_weights():
    # 2 worlds, 3 seats x 7 dominoes. Put domino ids 0..20 across seats,
    # pad the rest with -1. Hidden = {0, 7, 14} (one per seat in world 0).
    M = 2
    wh = torch.full((M, 3, 7), -1, dtype=torch.long)
    # world 0: seat0 holds 0..6, seat1 holds 7..13, seat2 holds 14..20
    for seat in range(3):
        wh[0, seat] = torch.arange(seat * 7, seat * 7 + 7)
    # world 1: rotate — seat0 holds 14..20, seat1 holds 0..6, seat2 holds 7..13
    wh[1, 0] = torch.arange(14, 21)
    wh[1, 1] = torch.arange(0, 7)
    wh[1, 2] = torch.arange(7, 14)

    seat_of = world_seat_matrix(wh)
    assert seat_of[0, 0] == 0 and seat_of[0, 7] == 1 and seat_of[0, 14] == 2
    assert seat_of[1, 0] == 1 and seat_of[1, 7] == 2 and seat_of[1, 14] == 0
    # unassigned domino 27 -> -1 in both
    assert seat_of[0, 27] == -1

    # Belief logits that strongly prefer world 0's layout for the hidden set.
    logits = torch.zeros(28, 3)
    logits[0, 0] = 10.0   # domino 0 -> seat 0
    logits[7, 1] = 10.0   # domino 7 -> seat 1
    logits[14, 2] = 10.0  # domino 14 -> seat 2
    w, ess = belief_weights(logits, seat_of, hidden_ids=[0, 7, 14])
    assert w.shape == (2,)
    assert abs(w.sum() - 1.0) < 1e-6
    assert w[0] > 0.99  # world 0 matches the peaked belief
    assert 1.0 <= ess <= 2.0


def test_belief_weights_uniform_when_flat():
    M = 3
    wh = torch.full((M, 3, 7), -1, dtype=torch.long)
    for m in range(M):
        for seat in range(3):
            wh[m, seat] = torch.arange(seat * 7, seat * 7 + 7)
    seat_of = world_seat_matrix(wh)
    logits = torch.zeros(28, 3)  # flat belief
    w, ess = belief_weights(logits, seat_of, hidden_ids=[0, 7, 14])
    assert np.allclose(w, 1.0 / M, atol=1e-6)
    assert abs(ess - M) < 1e-6
