#!/usr/bin/env python
"""walt/tests/test_cfr_parity.py — vectorized wave engine == verified loop.

Plain executable script (house style): one PASS/FAIL line per gate, exit
nonzero on any FAIL. Run from the repo root:

    /Users/jason/code/mk5-main/.venv/bin/python -u walt/tests/test_cfr_parity.py

The recursive loop engine (engine="loop") is the implementation the V1-V5
battery originally verified against enumeration/LP; the wave engine
(engine="wave", default) is the vectorized rewrite over the kernel's SoA
full-width walk. Same math, different traversal order — parity here pins
the rewrite to the verified reference.

Gates:
  P1 4-seat parity  — identical gap traces, self-play values, and exported
                      average profiles (same key sets, probs <= 1e-9) on
                      unpinned toys.
  P2 pinned parity  — same, under the 2p-izable pin (the wave tree contains
                      pinned zero-probability subtrees the loop tree prunes;
                      reach-gated export must make the domains identical).
"""
from __future__ import annotations

import sys

import numpy as np

from hoyt import reference as ref  # noqa: E402
from hoyt import toys as T  # noqa: E402
from hoyt.cfr import cfr_solve  # noqa: E402

PAY = T.payoff_points()
TOL = 1e-9
_failures: list[str] = []


def _gate(ok: bool, name: str, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def _compare(name, res_l, res_w, bad):
    if len(res_l.trace) != len(res_w.trace):
        bad.append(f"{name}: trace lengths {len(res_l.trace)} vs "
                   f"{len(res_w.trace)}")
        return 0.0
    dmax = 0.0
    for (i1, g1), (i2, g2) in zip(res_l.trace, res_w.trace):
        if i1 != i2:
            bad.append(f"{name}: trace iters {i1} vs {i2}")
            return dmax
        dmax = max(dmax, abs(g1 - g2))
    dmax = max(dmax, abs(res_l.value - res_w.value))
    if dmax > TOL:
        bad.append(f"{name}: trace/value drift {dmax:.2e}")
    el, ew = res_l.profile.entries, res_w.profile.entries
    if set(el) != set(ew):
        only_l = len(set(el) - set(ew))
        only_w = len(set(ew) - set(el))
        bad.append(f"{name}: profile domains differ "
                   f"(loop-only {only_l}, wave-only {only_w})")
        return dmax
    for key, (mv_l, pr_l) in el.items():
        mv_w, pr_w = ew[key]
        if mv_l != mv_w:
            bad.append(f"{name}: moves differ at {key}")
            return dmax
        d = float(np.max(np.abs(pr_l - pr_w))) if len(pr_l) else 0.0
        dmax = max(dmax, d)
        if d > TOL:
            bad.append(f"{name}: probs drift {d:.2e} at {key}")
            return dmax
    return dmax


def gate_P1() -> None:
    bad = []
    lines = []
    for name in ("t2_decl_w3", "t2_def_w3", "t2_decl_w12", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        kw = dict(iters=60, br_every=20, impl=ref)
        res_l = cfr_solve(sub, PAY, engine="loop", **kw)
        res_w = cfr_solve(sub, PAY, engine="wave", **kw)
        d = _compare(name, res_l, res_w, bad)
        lines.append(f"{name}:{d:.1e}")
    _gate(not bad, "P1 wave == loop on 4-seat toys (traces + profiles)",
          "; ".join(bad) if bad else " ".join(lines))


def gate_P2() -> None:
    bad = []
    lines = []
    for name in ("t2_decl_w3", "t2_def_w12", "t2_def_w3"):
        toy = T.get_toy(name)
        sub = toy.build()
        u, v = sub.me, (sub.me + 1) % 4
        others = [s for s in range(4) if s not in (u, v)]
        sigma = T.lowest_legal_sigma(sub, others)
        pinned = {s: sigma for s in others}
        kw = dict(iters=80, br_every=20, impl=ref, pinned=pinned)
        res_l = cfr_solve(sub, PAY, engine="loop", **kw)
        res_w = cfr_solve(sub, PAY, engine="wave", **kw)
        d = _compare(name, res_l, res_w, bad)
        lines.append(f"{name}:{d:.1e}")
    _gate(not bad, "P2 wave == loop under 2p-izable pins",
          "; ".join(bad) if bad else " ".join(lines))


def main() -> int:
    gate_P1()
    gate_P2()
    if _failures:
        print(f"\n{len(_failures)} gate(s) FAILED: {_failures}")
        return 1
    print("\nall parity gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
