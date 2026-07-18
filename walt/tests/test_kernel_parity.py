"""K1 parity gate: net-free BR (compiled jud σ) vs walt.solver.solve.

Same golden-fixture subset as test_solver_parity (spans the world-count
strata + the all-my-decision-wave shapes): for each root, walt's wavefront
solve is the fresh reference, then `compile_sigma` + `br_solve` must
reproduce best_move identically, value within 1e-9 relative, and EVERY
per-move root value within 1e-9 — on the cached fast path AND on the
generic-engine rewalk (which must agree with the fast path exactly).
The full 46-fixture sweep with timing is walt/kernel/bench_kernel.py.

Run: PYTHONPATH=. .venv python -m pytest walt/tests/test_kernel_parity.py
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
from walt.kernel import br_solve, build_subgame, compile_sigma, payoff_points
from walt.solver import solve
from walt.worlds import enumerate_worlds

FIX = Path(__file__).parent / "fixtures_h4.jsonl"
REL_TOL = 1e-9

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


@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_kernel_k1_parity(seed: int, oracle: FieldOracle) -> None:
    rec = next(r for r in map(json.loads, FIX.open()) if r["seed"] == seed)
    root = _root_of(rec["root"])
    worlds = enumerate_worlds(root)
    keep = sigma_consistent(root, worlds, oracle, _make_moves_filter(root))
    filtered = worlds[keep]
    if len(filtered) > 0:
        worlds = filtered
    weights = np.full(len(worlds), 1.0 / len(worlds), dtype=np.float64)

    ref = solve(root, worlds, weights, oracle, payoff="points")
    table = compile_sigma(root, worlds, oracle)
    oracle.evict_if_huge()
    sub = build_subgame(root, worlds, weights)
    pay = payoff_points()

    br = br_solve(sub, table, pay)
    assert br.best_move == ref.best_move
    assert abs(br.value - ref.value) <= REL_TOL * max(1.0, abs(ref.value))
    assert set(br.root_values) == set(ref.root_values)
    for mv, v in ref.root_values.items():
        assert abs(br.root_values[mv] - v) <= REL_TOL * max(1.0, abs(v)), \
            f"root value drift at move {mv}"
    # the compiled walk must be the same tree walt walked
    assert table.n_nodes == ref.n_nodes

    # generic-engine rewalk through the table: same tree, same numbers
    br2 = br_solve(sub, table, pay, rewalk=True)
    assert br2.best_move == br.best_move
    assert abs(br2.value - br.value) <= 1e-12
    assert br2.n_nodes == table.n_nodes
    assert br2.root_values == pytest.approx(br.root_values, abs=1e-12)

    # the BR strategy exists at the root and proposes the best move
    assert br.strategy[()] == br.best_move
