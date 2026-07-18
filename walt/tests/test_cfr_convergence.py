#!/usr/bin/env python
"""walt/tests/test_cfr_convergence.py — CFR+ convergence gates (V1, V2).

Plain executable script (house style): one PASS/FAIL line per gate, exit
nonzero on any FAIL. Run from the repo root:

    /Users/jason/code/mk5-main/.venv/bin/python -u walt/tests/test_cfr_convergence.py

Gates:
  V1a 2p zero-sum      — pin one seat per team to a known SigmaTable; the two
                         free seats form a genuine 2p zero-sum game whose
                         exact mixed value comes from enumerating ALL pure
                         info-set profiles + an LP; CFR's average-profile
                         value must converge to it (2-player theorems apply).
  V1b team common-payoff — pin the whole opposing team (the task-literal
                         2-player-ization): the free TEAM's exact optimum is
                         the max over joint pure profiles; CFR must reach it
                         on the toys (decentralized common-payoff has no
                         general guarantee — a miss here would be a measured
                         coordination trap, not a code bug; none observed).
  V1c LP sanity        — zero_sum_value on synthetic matrices with known
                         mixed values (the registry toys all admit pure
                         equilibria — measured fact — so mixing pressure on
                         the LP path is exercised synthetically).
  V1d make payoff      — V1a repeated under the make/set step table.
  V2  gap trace        — exploitability trace weakly decreasing in the tail,
                         final gap < 0.05 pts within the test budget.
"""
from __future__ import annotations

import sys

import numpy as np

from walt.kernel import reference as ref  # noqa: E402
from walt.kernel import toys as T  # noqa: E402
from walt.kernel.cfr import cfr_solve  # noqa: E402

PAY = T.payoff_points()
_failures: list[str] = []


def _gate(ok: bool, name: str, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def _pin_setup(sub, free):
    others = [s for s in range(4) if s not in free]
    sigma = T.lowest_legal_sigma(sub, others)
    return {s: sigma for s in others}


def _exact_zero_sum(sub, payoff, u, v, pinned):
    isets = T.discover_free_infosets(sub, {u, v}, pinned)
    pu = T.pure_profiles(isets, u)
    pv = T.pure_profiles(isets, v)
    M = T.joint_value_matrix(sub, payoff, u, pu, v, pv, pinned)
    return T.zero_sum_value(M, ref.sign_of(u, sub.bid_team)), len(pu) * len(pv)


def gate_V1a() -> None:
    bad = []
    lines = []
    for name in ("t2_decl_w1", "t2_decl_w3", "t2_def_w3",
                 "t2_decl_w12", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        u, v = sub.me, (sub.me + 1) % 4          # opposite teams
        pinned = _pin_setup(sub, (u, v))
        exact, pairs = _exact_zero_sum(sub, PAY, u, v, pinned)
        res = cfr_solve(sub, PAY, iters=300, br_every=25, target_gap=1e-3,
                        impl=ref, pinned=pinned)
        d = abs(res.value - exact)
        lines.append(f"{name}:|d|={d:.1e}@{res.iters_run}it")
        if d > 1e-2:
            bad.append(f"{name}: cfr={res.value} exact={exact} ({pairs} pairs)")
    _gate(not bad, "V1a CFR value == enumerated+LP 2p zero-sum value",
          "; ".join(bad) if bad else " ".join(lines))


def gate_V1b() -> None:
    bad = []
    lines = []
    for name in ("t2_decl_w1", "t2_decl_w3", "t2_def_w3", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        u, v = sub.me, (sub.me + 2) % 4          # hero + partner (free team)
        pinned = _pin_setup(sub, (u, v))
        isets = T.discover_free_infosets(sub, {u, v}, pinned)
        pu = T.pure_profiles(isets, u)
        pv = T.pure_profiles(isets, v)
        if len(pu) * len(pv) > 20000:
            lines.append(f"{name}:skipped({len(pu)}x{len(pv)})")
            continue
        M = T.joint_value_matrix(sub, PAY, u, pu, v, pv, pinned)
        sgn = ref.sign_of(u, sub.bid_team)       # same team: common payoff
        exact = sgn * np.max(sgn * M)
        res = cfr_solve(sub, PAY, iters=300, br_every=25, target_gap=1e-3,
                        impl=ref, pinned=pinned)
        d = abs(res.value - exact)
        lines.append(f"{name}:|d|={d:.1e}")
        if d > 1e-2:
            bad.append(f"{name}: cfr={res.value} team-opt={exact}")
    _gate(not bad, "V1b CFR reaches the free TEAM's enumerated optimum",
          "; ".join(bad) if bad else " ".join(lines))


def gate_V1c() -> None:
    # the 3x2 asymmetric game discriminates the sign convention: row-maximizer
    # takes the pure safe row (0.6); row-minimizer mixes the top rows (0.5).
    asym = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.6]])
    cases = [
        (np.array([[1.0, -1.0], [-1.0, 1.0]]), 1.0, 0.0),          # matching pennies
        (np.array([[3.0, 0.0], [1.0, 2.0]]), 1.0, 1.5),            # forced mixing
        (asym, 1.0, 0.6),
        (asym, -1.0, 0.5),
        (np.array([[5.0, 5.0], [5.0, 5.0]]), 1.0, 5.0),            # constant
    ]
    bad = []
    for M, sgn, want in cases:
        got = T.zero_sum_value(M, sgn)
        if abs(got - want) > 1e-9:
            bad.append(f"sign={sgn} want={want} got={got}")
    _gate(not bad, "V1c LP matrix-game values on synthetic mixed games",
          "; ".join(bad) if bad else f"{len(cases)} matrices incl. forced mixing")


def gate_V1d() -> None:
    # engine seeds whose 2-trick roots are UNDECIDED at the bid line
    # (banked declaring points < bid <= banked + remaining), so the make
    # step table is not constant on the reachable leaves.
    bad = []
    lines = []
    for seed, n in ((33, 6), (29, 6)):
        root, worlds = T.gen_engine_root(seed, 2)
        idx = np.unique(np.linspace(0, worlds.shape[0] - 1, n).round().astype(int))
        sub = ref.build_subgame(root, worlds[idx],
                                np.arange(1, len(idx) + 1, dtype=np.float64))
        name = f"seed{seed}"
        pay = T.payoff_make(root.bid_value)
        u, v = sub.me, (sub.me + 1) % 4
        pinned = _pin_setup(sub, (u, v))
        exact, _ = _exact_zero_sum(sub, pay, u, v, pinned)
        res = cfr_solve(sub, pay, iters=300, br_every=25, target_gap=1e-4,
                        impl=ref, pinned=pinned)
        d = abs(res.value - exact)
        lines.append(f"{name}:P(make) exact={exact:.4f} |d|={d:.1e}")
        if d > 1e-2:
            bad.append(f"{name}: cfr={res.value} exact={exact}")
    _gate(not bad, "V1d make-payoff convergence (step leaf table)",
          "; ".join(bad) if bad else " ".join(lines))


def gate_V2() -> None:
    bad = []
    lines = []
    for name in ("t2_def_w3", "t2_decl_w12", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        res = cfr_solve(sub, PAY, iters=120, br_every=10, impl=ref)
        gaps = [g for _, g in res.trace]
        tail = gaps[len(gaps) // 2:]
        mono = all(tail[i + 1] <= tail[i] + 1e-6 for i in range(len(tail) - 1))
        lines.append(f"{name}:final={gaps[-1]:.1e}")
        if not mono:
            bad.append(f"{name}: tail not weakly decreasing {tail}")
        if gaps[-1] >= 0.05:
            bad.append(f"{name}: final gap {gaps[-1]} >= 0.05")
    _gate(not bad, "V2 gap trace weakly decreasing tail, final < 0.05",
          "; ".join(bad) if bad else " ".join(lines))


def main() -> int:
    gate_V1a()
    gate_V1b()
    gate_V1c()
    gate_V1d()
    gate_V2()
    if _failures:
        print(f"\n{len(_failures)} gate(s) FAILED: {_failures}")
        return 1
    print("\nall convergence gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
