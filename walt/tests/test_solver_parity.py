"""Solver parity gate: the wavefront solver vs recorded golden fixtures.

A subset of the committed H4 fixtures (walt/tests/fixtures_h4.jsonl) solved
end-to-end — enumerate physics worlds, exact-B(σ) filter, uniform weights,
payoff='points' — and checked against walt/tests/fixtures_h4_expected.jsonl:
best_move IDENTICAL, value within 1e-9 relative, and the σ-filtered world
count unchanged. Node/query counters must also match exactly: the recursion
tree is a deterministic function of σ, so any drift there means a σ-decision
diverged even if the value happens to agree.

The subset spans all three strata (natural/mid/big by world count) while
keeping runtime CI-friendly (<60s). The full 46-fixture sweep with timing
lives in scratch/bench_fixtures.py — a script, not a test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from walt.contracts import EndgameRoot
from walt.field import FieldOracle, sigma_consistent
from walt.grade import _make_moves_filter
from walt.solver import solve
from walt.worlds import enumerate_worlds

FIX = Path(__file__).parent / "fixtures_h4.jsonl"
EXP = Path(__file__).parent / "fixtures_h4_expected.jsonl"
REL_TOL = 1e-9

# Spans the strata: natural (<500 worlds), mid (500-5000), big (>=5000);
# 777009/777013 lock the all-my-decision-wave shape (me leads every trick).
SEEDS = (777003, 777004, 777005, 777010, 777016, 777017, 777022,
         777006, 777009, 777011, 777013, 777018, 777027)


def _root_of(rd: dict) -> EndgameRoot:
    return EndgameRoot(
        decl_id=rd["decl_id"], bidder=rd["bidder"], bid_value=rd["bid_value"],
        bids=tuple(rd["bids"]), dealer=rd["dealer"], me=rd["me"],
        my_hand=tuple(rd["my_hand"]),
        play_history=tuple((s, d) for s, d in rd["play_history"]),
        trick_leader=rd["trick_leader"],
        current_trick=tuple(rd["current_trick"]),
        team_points=tuple(rd["team_points"]),
    )


@pytest.fixture(scope="module")
def oracle() -> FieldOracle:
    torch.set_num_threads(1)
    return FieldOracle(device="cpu")


@pytest.mark.parametrize(
    "seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_fixture_parity(seed: int, oracle: FieldOracle) -> None:
    rec = next(r for r in map(json.loads, FIX.open()) if r["seed"] == seed)
    exp = next(r for r in map(json.loads, EXP.open()) if r["seed"] == seed)

    root = _root_of(rec["root"])
    worlds = enumerate_worlds(root)
    keep = sigma_consistent(root, worlds, oracle, _make_moves_filter(root))
    filtered = worlds[keep]
    if len(filtered) > 0:
        worlds = filtered
    assert len(worlds) == exp["n_worlds"], "σ-filtered world set drifted"

    weights = np.full(len(worlds), 1.0 / len(worlds), dtype=np.float64)
    res = solve(root, worlds, weights, oracle, payoff="points")
    oracle.evict_if_huge()

    assert res.best_move == exp["best_move"]
    assert abs(res.value - exp["value"]) <= REL_TOL * max(1.0, abs(exp["value"]))
    assert res.n_nodes == exp["n_nodes"]
    assert res.n_field_queries == exp["n_field_queries"]
