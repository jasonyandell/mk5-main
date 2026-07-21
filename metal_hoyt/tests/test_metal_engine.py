"""metal_hoyt gates M1–M4 (DESIGN.md). M5 (speed) lives in bench.py.

Everything here runs on the Metal GPU and hard-fails without one — the
no-CPU-fallback doctrine is itself under test.
"""
import numpy as np
import pytest

import hoyt
from hoyt.cfr import cfr_solve
from hoyt.toys import all_toys, get_toy, payoff_points
from metal_hoyt import cfr_solve_metal
from metal_hoyt.kernels import require_metal


def _sub(toy):
    return hoyt.build_subgame(toy.root, toy.worlds, toy.weights)


def test_metal_required():
    require_metal()          # this machine has one; absence must raise


# --------------------------------------------------------------------------- #
#  M1 — toys: convergence + ε-equilibrium agreement with the CPU engine        #
# --------------------------------------------------------------------------- #

def test_m1_toys_converge_and_agree():
    pay = payoff_points()
    for toy in all_toys():
        cpu = cfr_solve(_sub(toy), pay, iters=200, target_gap=0.01,
                        br_every=10)
        gpu = cfr_solve_metal(_sub(toy), pay, iters=200, target_gap=0.01,
                              br_every=10)
        assert gpu.gap <= 0.01, (toy.name, gpu.gap)
        # two ε-equilibrium values can differ by at most the two gaps
        # against the shared game value; give fp32 a hair on top
        tol = cpu.gap + gpu.gap + 1e-3
        assert abs(cpu.value - gpu.value) <= tol, \
            (toy.name, cpu.value, gpu.value)


def test_m1_pinned_seats_stay_pinned():
    """A pinned seat's profile must ride through the metal solve unchanged
    and be excluded from the gap — mirror of the CPU pinned contract."""
    from hoyt.toys import lowest_legal_sigma
    pay = payoff_points()
    toy = get_toy("t2_decl_w12")
    sub = _sub(toy)
    sigma = lowest_legal_sigma(toy.build(), (1, 3))   # reference-impl domain
    pin = {1: sigma, 3: sigma}
    cpu = cfr_solve(sub, pay, iters=300, target_gap=0.005, br_every=10,
                    pinned=pin)
    gpu = cfr_solve_metal(_sub(toy), pay, iters=300, target_gap=0.005,
                          br_every=10, pinned=pin)
    assert gpu.gap <= 0.005
    tol = cpu.gap + gpu.gap + 1e-3
    assert abs(cpu.value - gpu.value) <= tol


# --------------------------------------------------------------------------- #
#  M2 — kernel parity: MSL vs same-fold-order numpy fp32 mirrors               #
# --------------------------------------------------------------------------- #

def _rand_struct(seed, E=40_000, P=7_000, G=3_000):
    rng = np.random.default_rng(seed)
    ps = np.sort(rng.integers(0, P, E)).astype(np.int32)
    cgid = rng.integers(-1, G, E).astype(np.int32)
    eseat = rng.integers(0, 4, E).astype(np.int8)
    sig = rng.random(G, dtype=np.float32)
    return rng, ps, cgid, eseat, sig


def test_m2_fwd_kernel_bitwise():
    import mlx.core as mx
    from metal_hoyt.kernels import grid1, k_fwd, tg1
    rng, ps, cgid, eseat, sig = _rand_struct(1)
    P = int(ps.max()) + 1
    E = len(ps)
    rmu_p = rng.random(P, dtype=np.float32)
    ru_p = rng.random(P, dtype=np.float32)
    for u in range(4):
        rmu_c, ru_c = k_fwd(
            inputs=[mx.array(ps), mx.array(cgid), mx.array(eseat),
                    mx.array(np.array([u], np.int32)), mx.array(sig),
                    mx.array(rmu_p), mx.array(ru_p),
                    mx.array(np.array([E], np.int32))],
            grid=grid1(E), threadgroup=tg1(E),
            output_shapes=[(E,), (E,)],
            output_dtypes=[mx.float32, mx.float32])
        pr = np.where(cgid >= 0, sig[np.maximum(cgid, 0)],
                      np.float32(1.0)).astype(np.float32)
        mine = eseat == u
        ref_rmu = np.where(mine, rmu_p[ps], (rmu_p[ps] * pr))
        ref_ru = np.where(mine, (ru_p[ps] * pr), ru_p[ps])
        assert np.array_equal(np.array(rmu_c), ref_rmu.astype(np.float32))
        assert np.array_equal(np.array(ru_c), ref_ru.astype(np.float32))


def test_m2_bwd_v_kernel_bitwise():
    import mlx.core as mx
    from metal_hoyt.kernels import grid1, k_bwd_v, tg1
    rng, ps, cgid, eseat, sig = _rand_struct(2)
    P = int(ps.max()) + 1
    E = len(ps)
    poff = np.searchsorted(ps, np.arange(P + 1)).astype(np.int32)
    perm = np.arange(E, dtype=np.int32)
    v_c = rng.random(E, dtype=np.float32)
    def run():
        (v_p,) = k_bwd_v(
            inputs=[mx.array(poff), mx.array(perm), mx.array(cgid),
                    mx.array(eseat), mx.array(np.array([-1], np.int32)),
                    mx.array(sig), mx.array(np.zeros(1, np.float32)),
                    mx.array(np.array([0], np.int32)), mx.array(v_c),
                    mx.array(np.array([P], np.int32))],
            grid=grid1(P), threadgroup=tg1(P),
            output_shapes=[(P,)], output_dtypes=[mx.float32])
        return np.array(v_p)
    got = run()
    # Metal contracts mul+add to FMA, so fold kernels are NOT bitwise vs a
    # separate-mul-add mirror; the claim is ulp-level agreement with the
    # fp64 same-order fold (measured 1.8e-7) plus exact repeatability.
    w64 = np.where(cgid >= 0,
                   sig[np.maximum(cgid, 0)].astype(np.float64) * v_c, v_c)
    ref64 = np.add.reduceat(w64, poff[:-1].astype(np.int64).clip(0, E - 1))
    ref64[poff[1:] == poff[:-1]] = 0.0
    rel = np.abs(got - ref64) / np.maximum(np.abs(ref64), 1e-12)
    assert float(rel.max()) < 1e-6
    assert np.array_equal(got, run())        # deterministic fp32


def test_m2_rm_update_kernel_bitwise():
    import mlx.core as mx
    from metal_hoyt.kernels import grid1, k_rm_update, tg1
    rng = np.random.default_rng(3)
    nI, ns_max = 500, 4
    nleg = rng.integers(2, ns_max + 1, nI)
    iso = np.concatenate(([0], np.cumsum(nleg))).astype(np.int32)
    ns = int(iso[-1])
    sig = rng.random(ns, dtype=np.float32)
    reg = rng.random(ns, dtype=np.float32)
    avg = rng.random(ns, dtype=np.float32)
    cf = (rng.random(ns, dtype=np.float32) - 0.5).astype(np.float32)
    xI = rng.random(nI, dtype=np.float32)
    inv = np.repeat(1.0 / nleg, nleg).astype(np.float32)
    sign, itv = np.float32(-1.0), np.float32(7.0)
    def run():
        return k_rm_update(
            inputs=[mx.array(iso), mx.array(sig), mx.array(reg),
                    mx.array(avg), mx.array(cf), mx.array(xI),
                    mx.array(inv), mx.array(np.array([sign])),
                    mx.array(np.array([itv])),
                    mx.array(np.array([nI], np.int32))],
            grid=grid1(nI), threadgroup=tg1(nI),
            output_shapes=[(ns,), (ns,), (ns,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32])
    sig_o, reg_o, avg_o = run()
    # fp64 same-order mirror; FMA contraction bounds the fp32 deviation
    rs, rr, ra = (np.empty(ns) for _ in range(3))
    s64, c64, a64 = (x.astype(np.float64) for x in (sig, cf, avg))
    for i in range(nI):
        a, b = iso[i], iso[i + 1]
        cfv = float(np.dot(s64[a:b], c64[a:b]))
        txv = float(itv) * float(xI[i])
        for k in range(a, b):
            ra[k] = a64[k] + txv * s64[k]
            rr[k] = max(reg[k] + float(sign) * (c64[k] - cfv), 0.0)
        tot = float(rr[a:b].sum())
        for k in range(a, b):
            rs[k] = rr[k] / tot if tot > 0.0 else inv[k]
    for got, ref in ((sig_o, rs), (reg_o, rr), (avg_o, ra)):
        rel = np.abs(np.array(got) - ref) / np.maximum(np.abs(ref), 1e-6)
        assert float(rel.max()) < 1e-5
    sig_o2, reg_o2, avg_o2 = run()
    assert all(np.array_equal(np.array(x), np.array(y))
               for x, y in ((sig_o, sig_o2), (reg_o, reg_o2),
                            (avg_o, avg_o2)))


def test_m2_br_choose_first_max():
    import mlx.core as mx
    from metal_hoyt.kernels import grid1, k_br_choose, tg1
    iso = np.array([0, 3, 5, 9], dtype=np.int32)
    score = np.array([1.0, 2.0, 2.0,   5.0, 5.0,   -1.0, -3.0, -1.0, 0.0],
                     dtype=np.float32)
    (ch,) = k_br_choose(
        inputs=[mx.array(iso), mx.array(score),
                mx.array(np.array([1.0], np.float32)),
                mx.array(np.array([3], np.int32))],
        grid=grid1(3), threadgroup=tg1(3),
        output_shapes=[(9,)], output_dtypes=[mx.float32])
    got = np.array(ch)
    exp = np.zeros(9, dtype=np.float32)
    exp[[1, 3, 8]] = 1.0        # first max in each segment
    assert np.array_equal(got, exp)
    # sign flip: minimizer picks first MIN
    (ch,) = k_br_choose(
        inputs=[mx.array(iso), mx.array(score),
                mx.array(np.array([-1.0], np.float32)),
                mx.array(np.array([3], np.int32))],
        grid=grid1(3), threadgroup=tg1(3),
        output_shapes=[(9,)], output_dtypes=[mx.float32])
    exp = np.zeros(9, dtype=np.float32)
    exp[[0, 3, 6]] = 1.0
    assert np.array_equal(np.array(ch), exp)


# --------------------------------------------------------------------------- #
#  M3/M4 — evalset roots: certified convergence + honest fp32 deltas           #
# --------------------------------------------------------------------------- #

def _evalset_sub(seed, cap=64):
    import json
    from pathlib import Path

    from hoyt.refsweep import _root_of
    from walt.grade import _world_cap_rng
    from walt.worlds import enumerate_worlds
    path = Path(__file__).resolve().parents[2] / "hoyt/evalset_h4_v1.jsonl"
    recs = {r["seed"]: r for r in map(json.loads, open(path))}
    root = _root_of(recs[seed]["root"])
    w = enumerate_worlds(root)
    if len(w) > cap:
        idx = _world_cap_rng(root, cap).choice(len(w), size=cap,
                                               replace=False)
        w = w[np.sort(idx)]
    return hoyt.build_subgame(root, w, np.full(len(w), 1.0 / len(w)))


@pytest.mark.parametrize("seed", [555000, 555001, 555002])
def test_m3_evalset_certified(seed):
    pay = payoff_points()
    res = cfr_solve_metal(_evalset_sub(seed), pay, iters=80,
                          target_gap=0.05, br_every=10)
    # the returned gap is fp64-certified by construction; the bar is the
    # reference bar
    assert res.gap <= 0.05, (seed, res.gap)
    # M4 honesty: the fp32 steering gap tracks fp64. Measured worst case
    # 0.0085 (555002, cap-64 — near-tied BR argmax flips under fp32); the
    # gate holds a 2.4x ceiling over it, and certification protects the
    # claim regardless — a bigger delta only wastes iterations.
    assert res.timings["fp32_vs_fp64_gap"] <= 0.02, \
        (seed, res.timings["fp32_vs_fp64_gap"])


def test_m3_value_matches_cpu_reference():
    """Certified metal value vs the CPU engine's on one root: both are
    ε-equilibrium self-play values with ε = 0.05, so they agree within
    the sum of gaps (plus fp32 hair)."""
    pay = payoff_points()
    sub1 = _evalset_sub(555001)
    cpu = cfr_solve(sub1, pay, iters=80, target_gap=0.05, br_every=10,
                    gap_exit=True)
    gpu = cfr_solve_metal(_evalset_sub(555001), pay, iters=80,
                          target_gap=0.05, br_every=10)
    assert abs(cpu.value - gpu.value) <= cpu.gap + gpu.gap + 1e-3
