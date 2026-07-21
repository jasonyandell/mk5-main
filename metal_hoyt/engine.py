"""metal_hoyt/engine.py — the GPU CFR+ engine and the fp64 certification seam.

`cfr_solve_metal` mirrors hoyt.cfr.cfr_solve's contract (same CFRResult,
same verdict semantics) with the iterate and intermediate gap pricing on
the Metal GPU in fp32 and every banked number certified on CPU in fp64 by
hoyt's own `_wave_values` / `_wave_br` over the same resident structure.
The fp32 gap only ever decides WHEN to certify; it is never a claim
(DESIGN.md).

Structure comes from hoyt's verified build (`_build_wave`,
`_build_fused(par=True)`) — deliberate reuse of private hoyt internals;
the parity tests pin this seam.
"""
from __future__ import annotations

import time

import numpy as np
import mlx.core as mx

from hoyt.cfr import (
    CFRResult,
    _avg_sig_fused,
    _build_fused,
    _build_wave,
    _wave_br,
    _wave_values,
    sign_of,
)
from metal_hoyt.kernels import (
    grid1,
    k_asig_norm,
    k_br_choose,
    k_bwd_v,
    k_cf_seg,
    k_fwd,
    k_rm_update,
    require_metal,
    tg1,
)
from metal_hoyt.layout import MetalLayout, _f32, _scalar_i32

__all__ = ["MetalEngine", "cfr_solve_metal"]


class MetalEngine:
    """Mutable fp32 CFR state over an immutable MetalLayout."""

    def __init__(self, lay: MetalLayout, live_seats, signs):
        self.lay = lay
        self.live = list(live_seats)
        self.signs = {u: _f32(np.array([signs[u]])) for u in range(4)}
        self.sig_u = [lay.sig0[u] for u in range(4)]
        self.reg_u = [None if lay.sig0[u] is None
                      else mx.zeros(lay.n_slots[u], dtype=mx.float32)
                      for u in range(4)]
        self.avg_u = [None if lay.sig0[u] is None
                      else mx.zeros(lay.n_slots[u], dtype=mx.float32)
                      for u in range(4)]
        # global gid offset scalar per (j, u) block, for the BR backward
        self.lo_scal = {}
        for u in range(4):
            for (j, _, _, slot_lo, _n) in lay.cf_blocks[u]:
                self.lo_scal[(j, u)] = _scalar_i32(
                    int(lay.c_soff[u]) + slot_lo)
        self._sig_global = None

    # ---- state assembly --------------------------------------------------
    def sig_global(self) -> mx.array:
        if self._sig_global is None:
            parts = [s for s in self.sig_u if s is not None]
            if not parts:
                self._sig_global = self.lay.dummy_f   # fully-forced subgame
            elif len(parts) == 1:
                self._sig_global = parts[0]
            else:
                self._sig_global = mx.concatenate(parts)
        return self._sig_global

    # ---- one wave chain of forward reach (rmu per wave; ru optional) -----
    def _fwd_chain(self, u: int, sig: mx.array, want_ru: bool):
        lay = self.lay
        rmu = [lay.w0n]
        ru = [mx.ones(lay.nS[0], dtype=mx.float32)] if want_ru else None
        ru_j = ru[0] if want_ru else lay.w0n     # kernel needs a valid array
        for j in range(lay.L):
            n = lay.nE[j]
            rmu_c, ru_c = k_fwd(
                inputs=[lay.ps[j], lay.cgid[j], lay.eseat[j], lay.u_scal[u],
                        sig, rmu[j], ru_j, lay.nsc(n)],
                grid=grid1(n), threadgroup=tg1(n),
                output_shapes=[(n,), (n,)],
                output_dtypes=[mx.float32, mx.float32])
            rmu.append(rmu_c)
            if want_ru:
                ru.append(ru_c)
                ru_j = ru_c
        return rmu, ru

    def _bwd_step(self, j: int, u_scal, sig, chosen, lo, v):
        lay = self.lay
        nS = lay.nS[j]
        (v_p,) = k_bwd_v(
            inputs=[lay.poff[j], lay.perm[j], lay.cgid[j], lay.eseat[j],
                    u_scal, sig, chosen, lo, v, lay.nsc(nS)],
            grid=grid1(nS), threadgroup=tg1(nS),
            output_shapes=[(nS,)], output_dtypes=[mx.float32])
        return v_p

    # ---- one CFR+ seat pass ---------------------------------------------
    def seat_pass(self, u: int, it: int, itf: mx.array) -> None:
        lay = self.lay
        if lay.n_slots[u] == 0:
            return
        sig = self.sig_global()
        rmu, ru = self._fwd_chain(u, sig, want_ru=True)
        xI_u = mx.concatenate(
            [mx.take(ru[j], reps) for (j, reps) in lay.xi_blocks[u]])
        cf_by_j = {}
        v = lay.payleaf
        blocks = {j: (goff, eperm, n)
                  for (j, goff, eperm, _lo, n) in lay.cf_blocks[u]}
        for j in range(lay.L - 1, -1, -1):
            blk = blocks.get(j)
            if blk is not None:
                goff, eperm, n = blk
                (cf,) = k_cf_seg(
                    inputs=[goff, eperm, lay.ps[j], rmu[j], v, lay.nsc(n)],
                    grid=grid1(n), threadgroup=tg1(n),
                    output_shapes=[(n,)], output_dtypes=[mx.float32])
                cf_by_j[j] = cf
            v = self._bwd_step(j, lay.u_none, sig, lay.dummy_f,
                               lay.zero_i, v)
        cf_u = mx.concatenate([cf_by_j[j] for j in sorted(cf_by_j)])
        nI = lay.n_isets[u]
        ns = lay.n_slots[u]
        sig_o, reg_o, avg_o = k_rm_update(
            inputs=[lay.iso[u], self.sig_u[u], self.reg_u[u], self.avg_u[u],
                    cf_u, xI_u, lay.inv[u], self.signs[u], itf, lay.nsc(nI)],
            grid=grid1(nI), threadgroup=tg1(nI),
            output_shapes=[(ns,), (ns,), (ns,)],
            output_dtypes=[mx.float32, mx.float32, mx.float32])
        self.sig_u[u], self.reg_u[u], self.avg_u[u] = sig_o, reg_o, avg_o
        self._sig_global = None

    def iterate(self, it: int) -> None:
        itf = _f32(np.array([float(it)]))
        for u in self.live:
            self.seat_pass(u, it, itf)
        mx.eval(*[a for a in self.sig_u + self.reg_u + self.avg_u
                  if a is not None])

    # ---- fp32 average strategy ------------------------------------------
    def asig_parts(self):
        lay = self.lay
        parts = []
        for u in range(4):
            if lay.n_slots[u] == 0:
                continue
            if u in self.live:
                nI, ns = lay.n_isets[u], lay.n_slots[u]
                (asig_u,) = k_asig_norm(
                    inputs=[lay.iso[u], self.avg_u[u], lay.inv[u],
                            lay.nsc(nI)],
                    grid=grid1(nI), threadgroup=tg1(nI),
                    output_shapes=[(ns,)], output_dtypes=[mx.float32])
                parts.append(asig_u)
            else:
                parts.append(self.sig_u[u])   # pinned: fixed probs
        return parts

    # ---- fp32 in-struct gap (steering only, never a claim) ---------------
    def gpu_measure(self):
        lay = self.lay
        parts = self.asig_parts()
        asig = mx.concatenate(parts) if parts else lay.dummy_f
        v = lay.payleaf
        for j in range(lay.L - 1, -1, -1):
            v = self._bwd_step(j, lay.u_none, asig, lay.dummy_f,
                               lay.zero_i, v)
        vbar = mx.sum(v * lay.w0n)
        gains = {}
        for u in self.live:
            rmu, _ = self._fwd_chain(u, asig, want_ru=False)
            blocks = {j: (goff, eperm, n)
                      for (j, goff, eperm, _lo, n) in lay.cf_blocks[u]}
            isob = {j: (iso_loc, n_isets)
                    for (j, iso_loc, _ilo, n_isets) in lay.iso_blocks[u]}
            v_br = lay.payleaf
            for j in range(lay.L - 1, -1, -1):
                blk = blocks.get(j)
                if blk is not None:
                    goff, eperm, n = blk
                    (score,) = k_cf_seg(
                        inputs=[goff, eperm, lay.ps[j], rmu[j], v_br,
                                lay.nsc(n)],
                        grid=grid1(n), threadgroup=tg1(n),
                        output_shapes=[(n,)], output_dtypes=[mx.float32])
                    iso_loc, nI = isob[j]
                    (chosen,) = k_br_choose(
                        inputs=[iso_loc, score, self.signs[u], lay.nsc(nI)],
                        grid=grid1(nI), threadgroup=tg1(nI),
                        output_shapes=[(n,)], output_dtypes=[mx.float32])
                    v_br = self._bwd_step(j, lay.u_scal[u], asig, chosen,
                                          self.lo_scal[(j, u)], v_br)
                else:
                    v_br = self._bwd_step(j, lay.u_scal[u], asig,
                                          lay.dummy_f, lay.zero_i, v_br)
            gains[u] = mx.sum(v_br * lay.w0n)
        mx.eval(vbar, *gains.values())
        vb = float(vbar.item())
        signs = {u: float(np.array(self.signs[u])[0]) for u in self.live}
        gaps = {u: signs[u] * (float(gains[u].item()) - vb)
                for u in self.live}
        return vb, (max(gaps.values()) if gaps else 0.0), gaps

    # ---- fp64 average for certification ----------------------------------
    def avg64(self) -> np.ndarray:
        out = np.zeros(self.lay.n_c, dtype=np.float64)
        for u in range(4):
            if self.avg_u[u] is None:
                continue
            sl = slice(int(self.lay.c_soff[u]), int(self.lay.c_soff[u + 1]))
            if u in self.live:
                out[sl] = np.array(self.avg_u[u], copy=False)
        return out


def cfr_solve_metal(subgame, payoff43, iters: int = 200,
                    target_gap: float | None = None, br_every: int = 10,
                    margin: float = 0.01, impl=None,
                    pinned: dict | None = None,
                    slot_budget: int | None = None,
                    wall_budget_s: float | None = None,
                    threads: int | None = None) -> CFRResult:
    """CFR+ on the Metal GPU; certified fp64 claims. Contract mirrors
    hoyt.cfr.cfr_solve (CFRResult fields, verdict semantics) with these
    documented differences:

    - trace rows before the last are fp32 GPU gaps (steering values); the
      returned `gap` and `value` are always fp64, priced by hoyt's exact
      in-struct BR on the certified average profile.
    - no bitwise parity with the CPU engines is claimed (Metal has no
      fp64); the claim class is identical — profile + exactly-priced gap.
    - `margin`: certification is attempted once the fp32 gap reaches
      target_gap − margin; a failed certification resumes iterating, so a
      wrong fp32 gap can only waste iterations, never mis-claim.
    """
    require_metal()
    t0 = time.time()
    if threads:
        # perf-log 19d: cap numba BEFORE the build — the build's parallel
        # kernels otherwise run at numba's default (all cores) and
        # oversubscribe a multi-worker sweep
        import numba
        numba.set_num_threads(threads)
    tm = {"build": 0.0, "upload": 0.0, "iterate": 0.0, "gpu_gap": 0.0,
          "certify": 0.0, "export": 0.0, "fp32_vs_fp64_gap": 0.0}
    if impl is None:
        import hoyt as impl
    payoff43 = np.asarray(payoff43, dtype=np.float64).reshape(-1)
    if payoff43.shape[0] != 43:
        raise ValueError("payoff43 must have 43 entries")
    pinned = {int(s): p for s, p in (pinned or {}).items()}

    ws = _build_wave(subgame.root, subgame.worlds, subgame.weights, pinned,
                     slot_budget, False,
                     bulk_export=hasattr(impl.StochasticProfile, "set_bulk"),
                     kernels=True)
    bid_team = int(subgame.root.bidder) % 2
    signs = {u: sign_of(u, bid_team) for u in range(4)}
    sig = 1.0 / ws.nlegal_slot
    for off, probs in ws.pin_slots:
        sig[off:off + len(probs)] = probs
    live_seats = [u for u in range(4) if u not in pinned]
    fl = _build_fused(ws, sig, par=True)
    tm["build"] = time.time() - t0

    t1 = time.time()
    lay = MetalLayout(ws, fl, payoff43)
    eng = MetalEngine(lay, live_seats, signs)
    tm["upload"] = time.time() - t1

    payleaf64 = payoff43[ws.leaf_pts]
    rmu_scratch = [None] * ws.L

    def certify():
        """fp64 value + gap of the current average profile (the claim)."""
        t2 = time.time()
        asig = _avg_sig_fused(fl, eng.avg64(), live_seats)
        v0 = _wave_values(ws, asig, payleaf64)
        vbar = float(ws.w0 @ v0) / ws.total_w
        gap = 0.0
        for u in live_seats:
            bru = _wave_br(ws, asig, u, payleaf64, signs[u], rmu_scratch)
            gap = max(gap, signs[u] * (bru - vbar))
        tm["certify"] += time.time() - t2
        return asig, vbar, gap

    over = (lambda: time.time() - t0 > wall_budget_s) \
        if wall_budget_s is not None else (lambda: False)
    # steering bar: target − margin, floored at target/2 so a tight target
    # (≤ margin) still certifies on cadence instead of never firing —
    # CFR+ gaps are not monotone, so "iterate to the cap first" can END
    # WORSE than an earlier certifiable iterate
    bar32 = None if target_gap is None \
        else max(target_gap - margin, 0.5 * target_gap)

    trace: list = []
    result = None
    capped = False
    it = 0
    for it in range(1, iters + 1):
        t1 = time.time()
        eng.iterate(it)
        tm["iterate"] += time.time() - t1
        stop = it == iters or over()
        if it % br_every == 0 or stop:
            t1 = time.time()
            _vb32, gap32, _ = eng.gpu_measure()
            tm["gpu_gap"] += time.time() - t1
            trace.append((it, gap32))
            want_cert = stop or (bar32 is not None and gap32 <= bar32)
            if want_cert:
                asig, vbar, gap = certify()
                tm["fp32_vs_fp64_gap"] = abs(gap - gap32)
                trace[-1] = (it, gap)
                result = (asig, vbar, gap)
                if target_gap is not None and gap <= target_gap:
                    break
                if stop:
                    # hoyt semantics: capped names the wall as the binding
                    # constraint; a plain iters-exhausted stop is not capped
                    capped = over()
                    break
    if result is None:
        asig, vbar, gap = certify()
        trace.append((it, gap))
        result = (asig, vbar, gap)

    asig_f, vbar, gap = result
    t1 = time.time()
    prof = impl.StochasticProfile()
    if ws.exp_cols is not None:
        c = ws.exp_cols
        prof.set_bulk(c["seat"], c["hand"], c["h1"], c["h2"], c["moff"],
                      c["mv"], asig_f[c["slot_idx"]])
    else:
        for seat, hand, path, moves, off, n in ws.exp_entries:
            prof.set(seat, hand, path, moves, asig_f[off:off + n])
    tm["export"] = time.time() - t1
    mx.clear_cache()          # release GPU buffer cache between solves —
    #                           a fleet peer's RSS is not ours to hold
    return CFRResult(profile=prof, trace=trace, gap=gap, value=vbar,
                     iters_run=it, capped=capped, timings=tm)
