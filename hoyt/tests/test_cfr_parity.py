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
  P3 fused parity   — engine="fused" (numba kernels + forced-slot-compressed
                      updates, #82) == engine="loop" on the same toys and
                      pins, fp64. The fused engine is designed bitwise-equal
                      to "wave"; this gate pins it to the verified loop.
  P4 pricing oracle — the wave/fused engines' in-struct gap pricing
                      (_wave_br, perf-log 18m) == recomputing the gap on
                      the exported profile with ref.br_solve, <= 1e-9, on
                      toys and pins. br_solve stays the standing oracle.
  P5 gap_exit       — partial intermediate pricing (perf-log 18o) leaves
                      the stop iteration and the final gap/value/profile
                      exactly unchanged; intermediate partial gaps are
                      certified > target and <= the full max.
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


def gate_P3() -> None:
    bad = []
    lines = []
    for name in ("t2_decl_w3", "t2_def_w3", "t2_decl_w12", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        kw = dict(iters=60, br_every=20, impl=ref)
        res_l = cfr_solve(sub, PAY, engine="loop", **kw)
        res_f = cfr_solve(sub, PAY, engine="fused", **kw)
        d = _compare(name, res_l, res_f, bad)
        lines.append(f"{name}:{d:.1e}")
    for name in ("t2_decl_w3", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        u, v = sub.me, (sub.me + 1) % 4
        others = [s for s in range(4) if s not in (u, v)]
        sigma = T.lowest_legal_sigma(sub, others)
        pinned = {s: sigma for s in others}
        kw = dict(iters=80, br_every=20, impl=ref, pinned=pinned)
        res_l = cfr_solve(sub, PAY, engine="loop", **kw)
        res_f = cfr_solve(sub, PAY, engine="fused", **kw)
        d = _compare(f"{name}+pin", res_l, res_f, bad)
        lines.append(f"{name}+pin:{d:.1e}")
    _gate(not bad, "P3 fused == loop (4-seat toys + 2p-izable pins)",
          "; ".join(bad) if bad else " ".join(lines))


def gate_P4() -> None:
    from hoyt.reference import sign_of
    bad = []
    lines = []
    cases = [(name, None) for name in
             ("t2_decl_w3", "t2_def_w3", "t2_decl_w12", "t2_def_w12")]
    for name in ("t2_decl_w3", "t2_def_w12"):
        cases.append((name, "pin"))
    for name, mode in cases:
        toy = T.get_toy(name)
        sub = toy.build()
        kw = dict(iters=60, br_every=20, impl=ref)
        pinned = {}
        if mode == "pin":
            u, v = sub.me, (sub.me + 1) % 4
            others = [s for s in range(4) if s not in (u, v)]
            sigma = T.lowest_legal_sigma(sub, others)
            pinned = {s: sigma for s in others}
            kw.update(iters=80, pinned=pinned)
        res = cfr_solve(sub, PAY, engine="fused", **kw)
        bid_team = int(sub.root.bidder) % 2
        live = [s for s in range(4) if s not in pinned]
        gap_ref = 0.0
        for s in live:
            bru = ref.br_solve(sub, res.profile, PAY, hero=s,
                               want_strategy=False).value
            gap_ref = max(gap_ref, sign_of(s, bid_team) * (bru - res.value))
        d = abs(res.gap - gap_ref)
        tag = f"{name}{'+pin' if mode else ''}"
        lines.append(f"{tag}:{d:.1e}")
        if d > TOL:
            bad.append(f"{tag}: in-struct gap {res.gap} vs br_solve "
                       f"{gap_ref} (|d|={d:.2e})")
    _gate(not bad, "P4 in-struct pricing == ref.br_solve oracle (<=1e-9)",
          "; ".join(bad) if bad else " ".join(lines))


def gate_P5() -> None:
    # gap_exit invariance (perf-log 18o): partial intermediate pricing must
    # not move the stop iteration or the final result — final gap/value/
    # profile identical (exact equality), trace iterations identical, and
    # every intermediate partial gap <= the full measurement's (it is a
    # max over a prefix of seats) while still certifying > target.
    bad = []
    lines = []
    n_intermediate = 0
    for name in ("t2_decl_w3", "t2_def_w3", "t2_decl_w12", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        # 0.0005 forces 6-9 above-target intermediate measurements on the
        # w12 toys, so the partial-exit path is genuinely exercised
        kw = dict(iters=60, br_every=5, target_gap=0.0005, impl=ref)
        res_f = cfr_solve(sub, PAY, engine="fused", **kw)
        res_x = cfr_solve(sub, PAY, engine="fused", gap_exit=True, **kw)
        n_intermediate += len(res_x.trace) - 1
        if res_x.iters_run != res_f.iters_run:
            bad.append(f"{name}: stop iter {res_x.iters_run} vs "
                       f"{res_f.iters_run}")
            continue
        if res_x.gap != res_f.gap or res_x.value != res_f.value:
            bad.append(f"{name}: final gap/value differ "
                       f"({res_x.gap} vs {res_f.gap})")
        if [i for i, _ in res_x.trace] != [i for i, _ in res_f.trace]:
            bad.append(f"{name}: trace iters differ")
            continue
        for (i, gx), (_, gf) in zip(res_x.trace[:-1], res_f.trace[:-1]):
            if gx > gf + 1e-15 or gx <= kw["target_gap"]:
                bad.append(f"{name}: partial gap {gx} vs full {gf} at {i}")
        if res_x.trace[-1][1] != res_f.trace[-1][1]:
            bad.append(f"{name}: stopping gap differs")
        ex, ef = res_x.profile.entries, res_f.profile.entries
        if set(ex) != set(ef) or any(
                ex[k][0] != ef[k][0]
                or (len(ex[k][1]) and float(np.max(np.abs(
                    np.asarray(ex[k][1]) - np.asarray(ef[k][1])))) != 0.0)
                for k in ex):
            bad.append(f"{name}: profiles differ")
        lines.append(f"{name}:it{res_x.iters_run}")
    if n_intermediate < 4:
        bad.append(f"only {n_intermediate} intermediate measurements — "
                   "the exit path was not exercised")
    _gate(not bad, "P5 gap_exit invariance (stop iter + final result exact)",
          "; ".join(bad) if bad else
          " ".join(lines) + f" ({n_intermediate} intermediates)")


def gate_P13() -> None:
    # threaded fused == sequential fused, exactly (perf-log P13): every
    # parallel fold owns a disjoint output range with unchanged
    # within-range accumulation order, so thread count must not move a bit.
    # threads=3 forces uneven chunking; gap_exit composes with threading.
    bad = []
    lines = []
    for name in ("t2_decl_w3", "t2_def_w3", "t2_decl_w12", "t2_def_w12"):
        toy = T.get_toy(name)
        sub = toy.build()
        kw = dict(iters=60, br_every=20, impl=ref)
        res_s = cfr_solve(sub, PAY, engine="fused", **kw)
        for th in (3, 4):
            res_t = cfr_solve(sub, PAY, engine="fused", threads=th, **kw)
            d = _compare(f"{name}@th{th}", res_s, res_t, bad)
            if d != 0.0:
                bad.append(f"{name}@th{th}: max |d| {d:.2e} != 0")
        kx = dict(iters=60, br_every=5, target_gap=0.0005, impl=ref)
        res_x = cfr_solve(sub, PAY, engine="fused", gap_exit=True, **kx)
        res_xt = cfr_solve(sub, PAY, engine="fused", gap_exit=True,
                           threads=4, **kx)
        dx = _compare(f"{name}+exit@th4", res_x, res_xt, bad)
        if dx != 0.0 or res_x.trace != res_xt.trace:
            bad.append(f"{name}+exit@th4: not exact (|d| {dx:.2e})")
        lines.append(name)
    _gate(not bad, "P13 threaded fused == sequential fused (bitwise)",
          "; ".join(bad) if bad else " ".join(lines))


def main() -> int:
    gate_P1()
    gate_P2()
    gate_P3()
    gate_P4()
    gate_P5()
    gate_P13()
    if _failures:
        print(f"\n{len(_failures)} gate(s) FAILED: {_failures}")
        return 1
    print("\nall parity gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
