"""hoyt/tests/test_refsweep.py — cascade ladder logic + wall-budget cap.

Two halves, both toy-fast (no nets, no big roots):

- Pure ladder logic (parse/dispatch/claim/resume/merge) — the functions
  refsweep's driver runs, tested without solving anything.
- cfr_solve wall_budget_s semantics on toys, impl=hoyt.reference: a capped
  stop is a QUANTIFIED verdict (capped=True, gap present and exactly equal
  to an uncapped run stopped at the same iteration — the solver is
  deterministic, so equality is exact, not approximate); default-off is
  behaviorally identical; the loop engine (frozen parity mirror) refuses
  the knob.
"""
from __future__ import annotations

import json

import pytest

from hoyt import reference as ref
from hoyt import toys as T
from hoyt.cfr import cfr_solve
from hoyt.refsweep import (
    DEFAULT_RUNGS,
    Rung,
    best_row,
    claim,
    dispatch_order,
    load_ledger,
    next_rung,
    parse_rungs,
)

PAY = T.payoff_points()


# --------------------------------------------------------------------------- #
#  ladder logic (pure)                                                        #
# --------------------------------------------------------------------------- #

def test_parse_rungs_roundtrip():
    spec = ",".join(r.spec() for r in DEFAULT_RUNGS)
    assert parse_rungs(spec) == DEFAULT_RUNGS


def test_parse_rungs_rejects_malformed():
    with pytest.raises(ValueError):
        parse_rungs("80:0.05:90")          # missing slots
    with pytest.raises(ValueError):
        parse_rungs("")


def test_dispatch_order_longest_first_by_size():
    sized = [(1000 + i, (i * 37) % 400) for i in range(17)]
    q = dispatch_order(sized)
    assert sorted(q) == sorted(s for s, _ in sized)
    sizes = dict(sized)
    assert [sizes[s] for s in q] == sorted(
        (z for _, z in sized), reverse=True)
    assert dispatch_order(sized) == q                  # deterministic
    assert dispatch_order([]) == []


def test_dispatch_order_ties_break_by_seed():
    assert dispatch_order([(5, 7), (3, 7), (4, 9)]) == [4, 3, 5]


def test_claim_is_exclusive(tmp_path):
    p = tmp_path / "555000"
    assert claim(p) is True
    assert claim(p) is False                # second claimant loses
    assert claim(tmp_path / "555001") is True


def test_next_rung_ladder():
    n = 3
    assert next_rung({}, n) == 0
    assert next_rung({0: {"verdict": "converged"}}, n) is None
    assert next_rung({0: {"verdict": "gap_capped"}}, n) == 1
    assert next_rung({0: {"verdict": "slot_capped"}}, n) == 1
    assert next_rung({0: {"verdict": "error"}}, n) == 1
    assert next_rung({0: {"verdict": "gap_capped"},
                      1: {"verdict": "slot_capped"}}, n) == 2
    assert next_rung({0: {"verdict": "gap_capped"},
                      1: {"verdict": "converged"}}, n) is None
    # ladder exhausted: capped at the last rung stays capped
    assert next_rung({0: {"verdict": "gap_capped"},
                      1: {"verdict": "gap_capped"},
                      2: {"verdict": "gap_capped"}}, n) is None


def test_best_row_precedence():
    conv = {"verdict": "converged", "final_gap": 0.04, "rung": 1}
    gcap_a = {"verdict": "gap_capped", "final_gap": 0.30, "rung": 0}
    gcap_b = {"verdict": "gap_capped", "final_gap": 0.09, "rung": 1}
    scap = {"verdict": "slot_capped", "rung": 0}
    scap2 = {"verdict": "slot_capped", "rung": 2}
    err = {"verdict": "error", "rung": 2}
    assert best_row([gcap_a, conv, scap]) is conv
    assert best_row([gcap_a, gcap_b]) is gcap_b        # smaller gap wins
    assert best_row([scap, err]) is scap
    assert best_row([scap, scap2]) is scap2            # deepest attempt
    assert best_row([err]) is err


def test_load_ledger_union_first_wins(tmp_path):
    a = tmp_path / "rung0_shard0.jsonl"
    b = tmp_path / "rung1_shard0.jsonl"
    a.write_text(
        json.dumps({"seed": 5, "rung": 0, "verdict": "gap_capped"}) + "\n"
        + json.dumps({"seed": 5, "rung": 0, "verdict": "error"}) + "\n"
        + '{"seed": 9, "rung"'    # torn tail of a killed shard: skipped
    )
    b.write_text(json.dumps({"seed": 5, "rung": 1,
                             "verdict": "converged"}) + "\n")
    led = load_ledger(tmp_path)
    assert set(led) == {5}
    assert led[5][0]["verdict"] == "gap_capped"        # first write wins
    assert led[5][1]["verdict"] == "converged"
    assert next_rung(led[5], 3) is None


def test_rung_spec_inf_wall():
    r = Rung(10, 0.5, None, 1000)
    assert parse_rungs(r.spec()) == (r,)


# --------------------------------------------------------------------------- #
#  cfr_solve wall_budget_s semantics (toys, impl=reference)                   #
# --------------------------------------------------------------------------- #

def _toy_sub():
    return T.get_toy("t2_decl_w3").build()


def test_wall_budget_rejected_on_loop_engine():
    with pytest.raises(ValueError, match="wall_budget_s"):
        cfr_solve(_toy_sub(), PAY, iters=5, impl=ref, engine="loop",
                  wall_budget_s=1.0)


def test_wall_cap_is_quantified_and_exact():
    sub = _toy_sub()
    # budget 0: the first iteration boundary is already over budget, so the
    # solve measures once (off the br_every cadence) and stops capped.
    capped = cfr_solve(sub, PAY, iters=60, br_every=20, impl=ref,
                       wall_budget_s=0.0)
    assert capped.capped is True
    assert capped.iters_run == 1
    assert len(capped.trace) == 1 and capped.trace[0][0] == 1
    # the capped gap is EXACT: an uncapped run stopped at the same
    # iteration reproduces it bit-for-bit (deterministic solver).
    uncapped = cfr_solve(sub, PAY, iters=1, br_every=1, impl=ref)
    assert uncapped.capped is False
    assert capped.gap == uncapped.gap
    assert capped.value == uncapped.value
    assert capped.trace == uncapped.trace


def test_wall_budget_non_binding_is_identical():
    sub = _toy_sub()
    kw = dict(iters=40, br_every=10, impl=ref)
    base = cfr_solve(sub, PAY, **kw)
    budgeted = cfr_solve(sub, PAY, wall_budget_s=1e9, **kw)
    assert base.capped is False and budgeted.capped is False
    assert budgeted.trace == base.trace
    assert budgeted.gap == base.gap
    assert budgeted.value == base.value
    assert budgeted.iters_run == base.iters_run


def test_convergence_beats_wall_cap():
    # target_gap so loose the first measurement converges: even with a
    # blown budget the stop is a convergence, not a cap.
    res = cfr_solve(_toy_sub(), PAY, iters=60, br_every=20, impl=ref,
                    target_gap=1e9, wall_budget_s=0.0)
    assert res.capped is False
    assert res.iters_run == 1
    assert res.gap <= 1e9
