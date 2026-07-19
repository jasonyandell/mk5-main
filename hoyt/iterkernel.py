"""hoyt/iterkernel.py — fused per-wave edge kernels for the CFR iterate.

The engine="fused" lane of hoyt/cfr.py (#82): single-threaded numba loops
that replicate the numpy wave engine's exact accumulation order, so fp64
results are BITWISE identical to engine="wave" (the pinned mirror) — the
gathers and multiplies are elementwise (same rounding), and every
accumulation target receives its contributions in ascending edge order,
which is precisely np.bincount's summation order. numba's default strict
IEEE mode (no fastmath) keeps multiplies and adds separate, so no FMA
contraction can perturb the last ulp.

Why fused: the numpy iterate is memory-bandwidth-bound (perf-log 18g/18k).
Per edge these kernels read int32 indices (half the int64 tax), skip the
sigma gather entirely on FORCED edges (sigma == 1.0 forever, ~70% of edges
at H4 scale — cgid < 0 encodes them), and never materialize the pr/pm/pu
edge temporaries that each cost a DRAM round trip in numpy.

fp64 only, deliberately: an fp32 lane was built and measured dead on the
2026-07-18 anchor (P7 refuted — gap drift 2.6e-3 > the 1e-3 license bar,
AND zero throughput win: the fused loops are gather-latency-bound, not
float-bandwidth-bound). Receipts in wiki perf-log 18l.
"""
from __future__ import annotations

from numba import njit

__all__ = ["fwd_edges", "bwd_edges"]


@njit(cache=True, boundscheck=False)
def fwd_edges(ps, cgid, eseat, u, sig_c, rmu_p, ru_p, rmu_c, ru_c):
    """One wave transition of the forward reach pass for updating seat u.

    Mirrors (bitwise, fp64): pr = sig[GID]; pm = pr.copy(); pm[ue] = 1.0;
    r_mu = r_mu[PS] * pm; pu = ones; pu[ue] = pr[ue]; r_u = r_u[PS] * pu.
    x * 1.0 == x bitwise, so the skipped multiplies are exact.
    """
    for i in range(ps.shape[0]):
        p = ps[i]
        g = cgid[i]
        pr = sig_c[g] if g >= 0 else 1.0
        if eseat[i] == u:
            rmu_c[i] = rmu_p[p]
            ru_c[i] = ru_p[p] * pr
        else:
            rmu_c[i] = rmu_p[p] * pr
            ru_c[i] = ru_p[p]


@njit(cache=True, boundscheck=False)
def bwd_edges(ps, cgid, eseat, u, sig_c, v_c, rmu_p, v_p, cf_c):
    """One wave transition of the backward value pass for updating seat u.

    Mirrors (bitwise, fp64): cf[slot] += bincount(GID[ue], r_mu[PS[ue]]*v[ue])
    and v = bincount(PS, sig[GID]*v) — both accumulate in ascending edge
    order, which this loop reproduces. cf on FORCED slots is provably inert
    in RM+ (single-slot iset: cf - cfv == 0 exactly), so cgid < 0 edges skip
    it; they also skip the sigma gather (sigma == 1.0).
    """
    for i in range(v_p.shape[0]):
        v_p[i] = 0.0
    for i in range(ps.shape[0]):
        p = ps[i]
        g = cgid[i]
        vc = v_c[i]
        if g >= 0:
            if eseat[i] == u:
                cf_c[g] += rmu_p[p] * vc
            v_p[p] += sig_c[g] * vc
        else:
            v_p[p] += vc
