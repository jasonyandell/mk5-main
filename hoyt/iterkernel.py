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

from numba import njit, prange

__all__ = ["fwd_edges", "bwd_edges",
           "fwd_edges_par", "bwd_v_map", "bwd_v_seg", "cf_seg",
           "rm_update_seg"]


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


# --------------------------------------------------------------------------- #
#  parallel lane (P13) — same folds, disjoint outputs per prange iteration    #
# --------------------------------------------------------------------------- #
# The bitwise doctrine survives threading because every prange iteration owns
# a DISJOINT output range and accumulates its contributions in the SAME
# ascending-edge order as the sequential kernels (= np.bincount's fold). The
# grouping that makes outputs disjoint is precomputed structure built once in
# _build_fused (stable argsorts), never a runtime heuristic — so results are
# bitwise identical regardless of thread count or scheduling. Register
# accumulators seeded at +0.0 store the exact sequential fold: a +0.0-seeded
# IEEE sum can never produce -0.0, so acc == (0.0 + t1) + t2 + ... bitwise.
# numba's default strict IEEE mode (no fastmath) holds for parallel=True.


@njit(cache=True, parallel=True, boundscheck=False)
def fwd_edges_par(ps, cgid, eseat, u, sig_c, rmu_p, ru_p, rmu_c, ru_c):
    """fwd_edges, threaded: a pure map — edge i writes only rmu_c[i]/ru_c[i],
    so any chunking is bitwise."""
    for i in prange(ps.shape[0]):
        p = ps[i]
        g = cgid[i]
        pr = sig_c[g] if g >= 0 else 1.0
        if eseat[i] == u:
            rmu_c[i] = rmu_p[p]
            ru_c[i] = ru_p[p] * pr
        else:
            rmu_c[i] = rmu_p[p] * pr
            ru_c[i] = ru_p[p]


@njit(cache=True, parallel=True, boundscheck=False)
def bwd_v_map(ps, cgid, sig_c, v_c, v_p):
    """Backward v for bijection waves (each parent has exactly ONE edge —
    the trick-tail waves): a pure map. 0.0 + t reproduces the sequential
    zero-then-accumulate fold exactly (incl. the sign of zero)."""
    for i in prange(ps.shape[0]):
        g = cgid[i]
        vc = v_c[i]
        v_p[ps[i]] = 0.0 + (sig_c[g] * vc if g >= 0 else vc)


@njit(cache=True, parallel=True, boundscheck=False)
def bwd_v_seg(poff, perm, cgid, sig_c, v_c, v_p):
    """Backward v for general waves: parent p owns edges
    perm[poff[p]:poff[p+1]] (stable argsort of PS — ascending edge order
    within each parent, the bincount fold)."""
    for p in prange(v_p.shape[0]):
        acc = 0.0
        for k in range(poff[p], poff[p + 1]):
            i = perm[k]
            g = cgid[i]
            acc += sig_c[g] * v_c[i] if g >= 0 else v_c[i]
        v_p[p] = acc


@njit(cache=True, parallel=True, boundscheck=False)
def cf_seg(goff, gids, perm, ps, rmu_p, v_c, cf_c):
    """Counterfactual accumulation for the updating seat: group gl owns the
    seat's non-forced edges of one strategy slot, ascending edge order
    (stable argsort of cgid). cf_c is zeroed before the pass and each gid
    lives in exactly one wave, so the store is the full sequential fold."""
    for gl in prange(gids.shape[0]):
        acc = 0.0
        for k in range(goff[gl], goff[gl + 1]):
            i = perm[k]
            acc += rmu_p[ps[i]] * v_c[i]
        cf_c[gids[gl]] = acc


@njit(cache=True, parallel=True, boundscheck=False)
def rm_update_seg(iso, sig_u, reg_u, avg_u, cf_u, xI_u, inv_u, sign_u, it):
    """The CFR+ update block, threaded by info set: iset iu owns the
    contiguous slot run iso[iu]:iso[iu+1] and every fold is iset-local
    (cfv and tot in ascending slot order = the bincount fold; the clamp
    matches np.maximum's +0.0 on ties)."""
    for iu in prange(xI_u.shape[0]):
        a, b = iso[iu], iso[iu + 1]
        cfv = 0.0
        for k in range(a, b):
            cfv += sig_u[k] * cf_u[k]
        txv = it * xI_u[iu]
        for k in range(a, b):
            avg_u[k] += txv * sig_u[k]
            r = reg_u[k] + sign_u * (cf_u[k] - cfv)
            reg_u[k] = r if r > 0.0 else 0.0
        tot = 0.0
        for k in range(a, b):
            tot += reg_u[k]
        if tot > 0.0:
            for k in range(a, b):
                sig_u[k] = reg_u[k] / tot
        else:
            for k in range(a, b):
                sig_u[k] = inv_u[k]
