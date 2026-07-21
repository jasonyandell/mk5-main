"""metal_hoyt/kernels.py — the MSL kernel library (see DESIGN.md).

Six kernels, each the Metal twin of a numba fold in hoyt/iterkernel.py or a
block of hoyt.cfr._wave_br. Every fold accumulates sequentially inside one
thread over a precomputed segment (the P13 groupings), so results are
deterministic fp32 with no atomics. Scalars ride in 1-element input arrays
so each kernel compiles exactly once per process.

Metal has no fp64; the fp32 license and the fp64 certification seam are
DESIGN.md's contract, not this module's concern.
"""
from __future__ import annotations

import mlx.core as mx

__all__ = ["require_metal", "k_fwd", "k_bwd_v", "k_cf_seg", "k_rm_update",
           "k_br_choose", "k_asig_norm", "TG"]

TG = 256                      # threadgroup width for 1-D grids


def require_metal() -> None:
    """Hard-fail without a Metal GPU — hoyt IS the CPU lane (DESIGN.md)."""
    if not mx.metal.is_available():
        raise RuntimeError(
            "metal_hoyt requires a Metal GPU; use hoyt (CPU) instead")


# --------------------------------------------------------------------------- #
#  forward reach — twin of iterkernel.fwd_edges_par                            #
# --------------------------------------------------------------------------- #
k_fwd = mx.fast.metal_kernel(
    name="mh_fwd",
    input_names=["ps", "cgid", "eseat", "u", "sig", "rmu_p", "ru_p", "n"],
    output_names=["rmu_c", "ru_c"],
    source="""
    uint i = thread_position_in_grid.x;
    if (i >= (uint)n[0]) return;
    int p = ps[i];
    int g = cgid[i];
    float pr = g >= 0 ? sig[g] : 1.0f;
    if ((int)eseat[i] == u[0]) {
        rmu_c[i] = rmu_p[p];
        ru_c[i] = ru_p[p] * pr;
    } else {
        rmu_c[i] = rmu_p[p] * pr;
        ru_c[i] = ru_p[p];
    }
""")


# --------------------------------------------------------------------------- #
#  backward values — twin of iterkernel.bwd_v_seg; with u >= 0 and a chosen    #
#  mask it is also the BR backward of hoyt.cfr._wave_br (the deviating        #
#  seat's edges weigh chosen[g - lo] instead of sig[g]; forced edges 1.0)     #
# --------------------------------------------------------------------------- #
k_bwd_v = mx.fast.metal_kernel(
    name="mh_bwd_v",
    input_names=["poff", "perm", "cgid", "eseat", "u", "sig", "chosen",
                 "lo", "v_c", "n"],
    output_names=["v_p"],
    source="""
    uint p = thread_position_in_grid.x;
    if (p >= (uint)n[0]) return;
    int uu = u[0];
    float acc = 0.0f;
    for (int k = poff[p]; k < poff[p + 1]; ++k) {
        int i = perm[k];
        int g = cgid[i];
        float vc = v_c[i];
        float w;
        if (uu >= 0 && (int)eseat[i] == uu) {
            w = g >= 0 ? chosen[g - lo[0]] : 1.0f;
        } else {
            w = g >= 0 ? sig[g] : 1.0f;
        }
        acc += w * vc;
    }
    v_p[p] = acc;
""")


# --------------------------------------------------------------------------- #
#  counterfactual / BR-score accumulation — twin of iterkernel.cf_seg.         #
#  Group gl owns edges perm[goff[gl]:goff[gl+1]] of ONE strategy slot; the    #
#  (j, u) block's slots are contiguous, so output index gl IS slot lo + gl.   #
# --------------------------------------------------------------------------- #
k_cf_seg = mx.fast.metal_kernel(
    name="mh_cf_seg",
    input_names=["goff", "perm", "ps", "rmu_p", "v_c", "n"],
    output_names=["cf"],
    source="""
    uint gl = thread_position_in_grid.x;
    if (gl >= (uint)n[0]) return;
    float acc = 0.0f;
    for (int k = goff[gl]; k < goff[gl + 1]; ++k) {
        int i = perm[k];
        acc += rmu_p[ps[i]] * v_c[i];
    }
    cf[gl] = acc;
""")


# --------------------------------------------------------------------------- #
#  RM+ update — twin of iterkernel.rm_update_seg, functional (new arrays)      #
# --------------------------------------------------------------------------- #
k_rm_update = mx.fast.metal_kernel(
    name="mh_rm_update",
    input_names=["iso", "sig_u", "reg_u", "avg_u", "cf_u", "xI_u", "inv_u",
                 "sign", "itf", "n"],
    output_names=["sig_o", "reg_o", "avg_o"],
    source="""
    uint iu = thread_position_in_grid.x;
    if (iu >= (uint)n[0]) return;
    int a = iso[iu], b = iso[iu + 1];
    float cfv = 0.0f;
    for (int k = a; k < b; ++k) cfv += sig_u[k] * cf_u[k];
    float txv = itf[0] * xI_u[iu];
    float tot = 0.0f;
    for (int k = a; k < b; ++k) {
        avg_o[k] = avg_u[k] + txv * sig_u[k];
        float r = reg_u[k] + sign[0] * (cf_u[k] - cfv);
        r = r > 0.0f ? r : 0.0f;
        reg_o[k] = r;
        tot += r;
    }
    if (tot > 0.0f) {
        for (int k = a; k < b; ++k) sig_o[k] = reg_o[k] / tot;
    } else {
        for (int k = a; k < b; ++k) sig_o[k] = inv_u[k];
    }
""")


# --------------------------------------------------------------------------- #
#  BR argmax — per-iset first-max of the SIGNED score (lowest move wins       #
#  ties: slots ascend by move and the comparison is strict)                   #
# --------------------------------------------------------------------------- #
k_br_choose = mx.fast.metal_kernel(
    name="mh_br_choose",
    input_names=["iso", "score", "sign", "n"],
    output_names=["chosen"],
    source="""
    uint iu = thread_position_in_grid.x;
    if (iu >= (uint)n[0]) return;
    int a = iso[iu], b = iso[iu + 1];
    float best = -INFINITY;
    int arg = a;
    for (int k = a; k < b; ++k) {
        chosen[k] = 0.0f;
        float s = sign[0] * score[k];
        if (s > best) { best = s; arg = k; }
    }
    chosen[arg] = 1.0f;
""")


# --------------------------------------------------------------------------- #
#  normalized average strategy — twin of _avg_sig_fused's per-seat block       #
# --------------------------------------------------------------------------- #
k_asig_norm = mx.fast.metal_kernel(
    name="mh_asig_norm",
    input_names=["iso", "avg_u", "inv_u", "n"],
    output_names=["asig_u"],
    source="""
    uint iu = thread_position_in_grid.x;
    if (iu >= (uint)n[0]) return;
    int a = iso[iu], b = iso[iu + 1];
    float tot = 0.0f;
    for (int k = a; k < b; ++k) tot += avg_u[k];
    if (tot > 0.0f) {
        for (int k = a; k < b; ++k) asig_u[k] = avg_u[k] / tot;
    } else {
        for (int k = a; k < b; ++k) asig_u[k] = inv_u[k];
    }
""")


def grid1(n: int) -> tuple:
    """Exact 1-D launch grid (MLX grids are thread counts, not groups)."""
    return (int(n), 1, 1)


def tg1(n: int) -> tuple:
    return (min(int(n), TG), 1, 1)
