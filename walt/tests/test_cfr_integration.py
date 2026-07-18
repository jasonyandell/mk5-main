#!/usr/bin/env python
"""walt/tests/test_cfr_integration.py — same cfr_solve on both kernels.

Plain executable script (house style): one PASS/FAIL line per gate, exit
nonzero on any FAIL. Run from the repo root:

    /Users/jason/code/mk5-main/.venv/bin/python -u walt/tests/test_cfr_integration.py

The injectability contract, exercised for real: cfr_solve(subgame, payoff43,
impl=...) must produce the SAME trace whether impl is the pure-python
reference mirror or the fast net-free kernel (walt.kernel). The two br_solve
implementations were built independently in separate lanes; agreement here
cross-checks both through the full CFR loop (profile export -> exact BR gap
pricing) on every measurement.

Gates:
  I1 value/gap identity  — identical average-profile value and identical
                           measured gap trace across impls (1e-12).
  I2 profile transport   — the reference BR against the kernel-exported
                           profile equals the kernel BR against it (the
                           profile-domain convention survives the swap).
"""
from __future__ import annotations

import sys

import numpy as np

import walt.kernel as K  # noqa: E402
from walt.kernel import reference as ref  # noqa: E402
from walt.kernel import toys as T  # noqa: E402
from walt.kernel.cfr import cfr_solve  # noqa: E402

PAY = T.payoff_points()
_failures: list[str] = []


def _gate(ok: bool, name: str, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def gate_I1() -> None:
    bad = []
    lines = []
    for name in ("t2_decl_w3", "t2_def_w3", "t2_decl_w12", "t2_def_w12"):
        toy = T.get_toy(name)
        res_r = cfr_solve(ref.build_subgame(toy.root, toy.worlds, toy.weights),
                          PAY, iters=60, br_every=20, impl=ref)
        res_k = cfr_solve(K.build_subgame(toy.root, toy.worlds, toy.weights),
                          PAY, iters=60, br_every=20, impl=K)
        dv = abs(res_r.value - res_k.value)
        traces_match = len(res_r.trace) == len(res_k.trace) and all(
            i1 == i2 and abs(g1 - g2) <= 1e-12
            for (i1, g1), (i2, g2) in zip(res_r.trace, res_k.trace))
        lines.append(f"{name}:|dv|={dv:.1e}")
        if dv > 1e-12 or not traces_match:
            bad.append(f"{name}: ref {res_r.value}/{res_r.trace} vs "
                       f"kernel {res_k.value}/{res_k.trace}")
    _gate(not bad, "I1 cfr_solve identical on reference and fast kernel",
          "; ".join(bad) if bad else " ".join(lines))


def gate_I2() -> None:
    bad = []
    for name in ("t2_def_w3", "t2_decl_w12"):
        toy = T.get_toy(name)
        sub_r = ref.build_subgame(toy.root, toy.worlds, toy.weights)
        sub_k = K.build_subgame(toy.root, toy.worlds, toy.weights)
        res_k = cfr_solve(sub_k, PAY, iters=40, br_every=20, impl=K)
        # the kernel-exported profile: replayable by BOTH br implementations.
        # reference br needs .dist; adapt the kernel profile's .get.
        kprof = res_k.profile

        class _Adapt:
            def dist(self, seat, hand, node):
                got = kprof.get(seat, hand, node)
                if got is None:
                    raise KeyError((seat, hand, node))
                moves, probs = got
                return tuple(int(m) for m in moves), probs

        for hero in range(4):
            v_r = ref.br_solve(sub_r, _Adapt(), PAY, hero=hero).value
            v_k = K.br_solve(sub_k, kprof, PAY, hero=hero).value
            if abs(v_r - v_k) > 1e-9:
                bad.append(f"{name}/hero{hero}: ref BR {v_r} vs kernel BR {v_k}")
    _gate(not bad, "I2 exported profile prices identically under both BRs",
          "; ".join(bad) if bad else "2 toys x 4 heroes, 1e-9")


def main() -> int:
    gate_I1()
    gate_I2()
    if _failures:
        print(f"\n{len(_failures)} gate(s) FAILED: {_failures}")
        return 1
    print("\nall integration gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
