"""Device-resident float32 CFR+ iteration kernels for Hoyt.

This module is intentionally narrower than :mod:`hoyt.cfr`: it owns the
expensive alternating-update loop and exact-in-structure gap audit, while the
standing CPU lane still owns the walk build and public profile export. Keeping
that seam explicit lets us measure the first Metal vertical slice before
attempting the GPU builder and batched H5/H6 scheduler.

The arrays produced by one update remain MLX arrays throughout all iterations.
Only ``average_numpy`` crosses back to the host, at a requested audit boundary.
The arithmetic is float32 by design.  Metal is calibrated against the fp64
lane; bit replication is not part of its contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover - exercised by fail-fast path
    mx = None
    _MLX_IMPORT_ERROR = exc
else:
    _MLX_IMPORT_ERROR = None

__all__ = ["MetalCFR", "metal_available"]


def metal_available() -> bool:
    """Return whether MLX is present and its default device is a GPU."""
    return mx is not None and "gpu" in str(mx.default_device()).lower()


def _require_metal() -> None:
    if mx is None:
        raise RuntimeError(
            "engine='metal' requires MLX (install burl/requirements-mlx.txt)"
        ) from _MLX_IMPORT_ERROR
    if "gpu" not in str(mx.default_device()).lower():
        raise RuntimeError(
            f"engine='metal' requires the MLX Metal GPU device; got "
            f"{mx.default_device()}"
        )


def _launch(kernel, inputs, n, shapes, dtypes):
    if n <= 0:
        raise ValueError("Metal kernels require at least one output element")
    return kernel(
        inputs=inputs,
        grid=(int(n), 1, 1),
        threadgroup=(min(256, int(n)), 1, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )


_FWD_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= ps_shape[0]) return;
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
"""

_BWD_MAP_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= ps_shape[0]) return;
    int g = cgid[i];
    float z = g >= 0 ? sig[g] * v_c[i] : v_c[i];
    v_p[ps[i]] = z;
"""

_BWD_SEG_SOURCE = r"""
    uint p = thread_position_in_grid.x;
    if (p + 1 >= poff_shape[0]) return;
    float acc = 0.0f;
    for (int k = poff[p]; k < poff[p + 1]; ++k) {
        int i = perm[k];
        int g = cgid[i];
        acc += g >= 0 ? sig[g] * v_c[i] : v_c[i];
    }
    v_p[p] = acc;
"""

_CF_SEG_SOURCE = r"""
    uint gl = thread_position_in_grid.x;
    if (gl >= gids_shape[0]) return;
    float acc = 0.0f;
    for (int k = goff[gl]; k < goff[gl + 1]; ++k) {
        int i = perm[k];
        acc += rmu_p[ps[i]] * v_c[i];
    }
    cf[gl] = acc;
"""

_RM_SOURCE = r"""
    uint iu = thread_position_in_grid.x;
    if (iu >= xi_shape[0]) return;
    int a = iso[iu];
    int b = iso[iu + 1];
    float cfv = 0.0f;
    for (int k = a; k < b; ++k) cfv += sig[k] * cf[k];
    float txv = iteration[0] * xi[iu];
    float total = 0.0f;
    for (int k = a; k < b; ++k) {
        avg_out[k] = avg[k] + txv * sig[k];
        float r = reg[k] + sign[0] * (cf[k] - cfv);
        r = r > 0.0f ? r : 0.0f;
        reg_out[k] = r;
        total += r;
    }
    if (total > 0.0f) {
        for (int k = a; k < b; ++k) sig_out[k] = reg_out[k] / total;
    } else {
        for (int k = a; k < b; ++k) sig_out[k] = inv[k];
    }
"""

_AVG_SOURCE = r"""
    uint iu = thread_position_in_grid.x;
    if (iu + 1 >= iso_shape[0]) return;
    int a = iso[iu];
    int b = iso[iu + 1];
    float total = 0.0f;
    for (int k = a; k < b; ++k) total += avg[k];
    if (total > 0.0f) {
        for (int k = a; k < b; ++k) out[k] = avg[k] / total;
    } else {
        for (int k = a; k < b; ++k) out[k] = inv[k];
    }
"""

_BR_PICK_SOURCE = r"""
    uint iu = thread_position_in_grid.x;
    if (iu + 1 >= iso_shape[0]) return;
    int a = iso[iu];
    int b = iso[iu + 1];
    int best = a;
    float best_value = sign[0] * score[a];
    for (int k = a + 1; k < b; ++k) {
        float z = sign[0] * score[k];
        if (z > best_value) {
            best = k;
            best_value = z;
        }
    }
    for (int k = a; k < b; ++k) pick[k] = k == best ? 1.0f : 0.0f;
"""

_BR_BWD_MAP_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    if (i >= ps_shape[0]) return;
    int g = cgid[i];
    float z;
    if ((int)eseat[i] == u[0]) {
        z = g >= 0 ? pick[g - base[0]] * v_c[i] : v_c[i];
    } else {
        z = g >= 0 ? sig[g] * v_c[i] : v_c[i];
    }
    v_p[ps[i]] = z;
"""

_BR_BWD_SEG_SOURCE = r"""
    uint p = thread_position_in_grid.x;
    if (p + 1 >= poff_shape[0]) return;
    float acc = 0.0f;
    for (int k = poff[p]; k < poff[p + 1]; ++k) {
        int i = perm[k];
        int g = cgid[i];
        float z;
        if ((int)eseat[i] == u[0]) {
            z = g >= 0 ? pick[g - base[0]] * v_c[i] : v_c[i];
        } else {
            z = g >= 0 ? sig[g] * v_c[i] : v_c[i];
        }
        acc += z;
    }
    v_p[p] = acc;
"""


@dataclass(frozen=True)
class _WaveArrays:
    ps: object
    cgid: object
    eseat: object
    poff: object | None
    perm: object | None


class MetalCFR:
    """A float32 alternating CFR+ state whose iterate stays on Metal."""

    def __init__(self, ws, fl, payleaf, sig0, live_seats):
        _require_metal()
        if not fl.par:
            raise ValueError("MetalCFR requires _build_fused(..., par=True)")
        self.ws = ws
        self.fl = fl
        self.live_seats = tuple(int(u) for u in live_seats)

        self._fwd = mx.fast.metal_kernel(
            name="hoyt_fwd", input_names=["ps", "cgid", "eseat", "u",
            "sig", "rmu_p", "ru_p"], output_names=["rmu_c", "ru_c"],
            source=_FWD_SOURCE)
        self._bwd_map = mx.fast.metal_kernel(
            name="hoyt_bwd_map", input_names=["ps", "cgid", "sig", "v_c"],
            output_names=["v_p"], source=_BWD_MAP_SOURCE)
        self._bwd_seg = mx.fast.metal_kernel(
            name="hoyt_bwd_seg", input_names=["poff", "perm", "cgid",
            "sig", "v_c"], output_names=["v_p"], source=_BWD_SEG_SOURCE)
        self._cf_seg = mx.fast.metal_kernel(
            name="hoyt_cf_seg", input_names=["goff", "gids", "perm", "ps",
            "rmu_p", "v_c"], output_names=["cf"], source=_CF_SEG_SOURCE)
        self._rm = mx.fast.metal_kernel(
            name="hoyt_rm", input_names=["iso", "sig", "reg", "avg", "cf",
            "xi", "inv", "sign", "iteration"],
            output_names=["sig_out", "reg_out", "avg_out"], source=_RM_SOURCE)
        self._avg = mx.fast.metal_kernel(
            name="hoyt_avg", input_names=["iso", "avg", "inv"],
            output_names=["out"], source=_AVG_SOURCE)
        self._br_pick = mx.fast.metal_kernel(
            name="hoyt_br_pick", input_names=["iso", "score", "sign"],
            output_names=["pick"], source=_BR_PICK_SOURCE)
        self._br_bwd_map = mx.fast.metal_kernel(
            name="hoyt_br_bwd_map", input_names=["ps", "cgid", "eseat",
            "u", "sig", "v_c", "pick", "base"],
            output_names=["v_p"], source=_BR_BWD_MAP_SOURCE)
        self._br_bwd_seg = mx.fast.metal_kernel(
            name="hoyt_br_bwd_seg", input_names=["poff", "perm", "ps",
            "cgid", "eseat", "u", "sig", "v_c", "pick", "base"],
            output_names=["v_p"], source=_BR_BWD_SEG_SOURCE)

        self.waves = []
        for j in range(ws.L):
            seg = fl.vseg[j]
            poff = perm = None
            if seg is not None:
                poff = mx.array(seg[0].astype(np.int32, copy=False))
                perm = mx.array(seg[1])
            self.waves.append(_WaveArrays(
                ps=mx.array(fl.ps32[j]), cgid=mx.array(fl.cgid32[j]),
                eseat=mx.array(fl.eseat8[j]), poff=poff, perm=perm))

        self.ciju = {}
        for key, (cids, reps) in fl.c_iju.items():
            self.ciju[key] = (np.asarray(cids, dtype=np.int64),
                              mx.array(np.asarray(reps, dtype=np.int32)))
        self.cfju = {}
        for key, (goff, gids, perm) in fl.cf_ju.items():
            labels = fl.c_isl_loc[gids]
            local = labels - labels[0]
            niso = int(local[-1]) + 1
            iso = np.searchsorted(local, np.arange(niso + 1)) \
                .astype(np.int32)
            self.cfju[key] = (
                mx.array(goff.astype(np.int32, copy=False)),
                np.asarray(gids, dtype=np.int64),
                mx.array(np.asarray(gids, dtype=np.int32)),
                mx.array(perm),
                mx.array(iso),
                mx.array([int(gids[0])], dtype=mx.int32),
            )

        self.iso = {}
        self.inv = {}
        self.sig = {}
        self.reg = {}
        self.avg = {}
        for u in range(4):
            a, b = int(fl.c_soff[u]), int(fl.c_soff[u + 1])
            self.iso[u] = mx.array(fl.iso[u].astype(np.int32, copy=False))
            self.inv[u] = mx.array(
                np.asarray(fl.inv_nleg[a:b], dtype=np.float32))
            self.sig[u] = mx.array(np.asarray(sig0[fl.c_flat[a:b]],
                                               dtype=np.float32))
            self.reg[u] = mx.zeros((b - a,), dtype=mx.float32)
            self.avg[u] = mx.zeros((b - a,), dtype=mx.float32)

        self.w0 = mx.array(np.asarray(fl.w0, dtype=np.float32))
        self.payleaf = mx.array(np.asarray(payleaf, dtype=np.float32))
        self._ones = mx.ones((ws.S[0],), dtype=mx.float32)
        self._u = {u: mx.array([u], dtype=mx.int32) for u in range(4)}
        self._sign = {
            -1: mx.array([-1.0], dtype=mx.float32),
            1: mx.array([1.0], dtype=mx.float32),
        }
        self._assert_layout()
        mx.reset_peak_memory()
        mx.eval(self.w0, self.payleaf, list(self.sig.values()),
                list(self.reg.values()), list(self.avg.values()))

    def _assert_layout(self):
        for u in range(4):
            cids = []
            gids = []
            for j in range(self.ws.L):
                got = self.ciju.get((j, u))
                if got is not None:
                    cids.extend(got[0].tolist())
                got = self.cfju.get((j, u))
                if got is not None:
                    gids.extend(got[1].tolist())
            want_i = list(range(int(self.fl.c_isoff[u]),
                                int(self.fl.c_isoff[u + 1])))
            want_g = list(range(int(self.fl.c_soff[u]),
                                int(self.fl.c_soff[u + 1])))
            if cids != want_i or gids != want_g:
                raise ValueError(
                    "Metal layout requires wave-major contiguous seat slices")

    def _global_sig(self):
        return mx.concatenate([self.sig[u] for u in range(4)])

    def update(self, u: int, sign: int, iteration: int) -> None:
        """Queue one seat update; all inputs and outputs remain on device."""
        ni = int(self.fl.c_isoff[u + 1] - self.fl.c_isoff[u])
        if ni == 0:
            return
        sig = self._global_sig()
        rmu = self.w0
        ru = self._ones
        rmu_store = []
        xi_parts = []
        for j, wave in enumerate(self.waves):
            got = self.ciju.get((j, u))
            if got is not None:
                xi_parts.append(ru[got[1]])
            rmu_store.append(rmu)
            n = int(self.ws.S[j + 1])
            rmu, ru = _launch(
                self._fwd,
                [wave.ps, wave.cgid, wave.eseat, self._u[u], sig, rmu, ru],
                n, [(n,), (n,)], [mx.float32, mx.float32])

        v = self.payleaf
        cf_parts = []
        for j in range(self.ws.L - 1, -1, -1):
            wave = self.waves[j]
            got = self.cfju.get((j, u))
            if got is not None:
                goff, gids, gids_dev, perm, _iso, _base = got
                ng = len(gids)
                cf_parts.append((j, _launch(
                    self._cf_seg,
                    [goff, gids_dev, perm, wave.ps, rmu_store[j], v],
                    ng, [(ng,)], [mx.float32])[0]))
            np_ = int(self.ws.S[j])
            if wave.poff is None:
                v, = _launch(self._bwd_map,
                             [wave.ps, wave.cgid, sig, v], np_, [(np_,)],
                             [mx.float32])
            else:
                v, = _launch(self._bwd_seg,
                             [wave.poff, wave.perm, wave.cgid, sig, v], np_,
                             [(np_,)], [mx.float32])

        # The fused layout is wave-major.  Backward traversal discovered the
        # pieces in reverse, so restore wave order before concatenation.
        cf = mx.concatenate([part for _, part in sorted(cf_parts)])
        xi = mx.concatenate(xi_parts)
        ns = int(self.fl.c_soff[u + 1] - self.fl.c_soff[u])
        sig, reg, avg = _launch(
            self._rm,
            [self.iso[u], self.sig[u], self.reg[u], self.avg[u], cf, xi,
             self.inv[u], self._sign[int(sign)],
             mx.array([float(iteration)], dtype=mx.float32)],
            ni, [(ns,), (ns,), (ns,)],
            [mx.float32, mx.float32, mx.float32])
        self.sig[u], self.reg[u], self.avg[u] = sig, reg, avg
        # Materialize each alternating update without transferring it.  This
        # bounds MLX's lazy graph and lets temporary wave buffers be reclaimed.
        # A long async chain can release custom-kernel intermediates before
        # Metal has consumed them (observed as a GPU address fault on w12), so
        # the explicit device synchronization is a correctness boundary for
        # this first lane.
        mx.eval(sig, reg, avg)

    def round(self, signs: dict[int, int], iteration: int) -> None:
        for u in self.live_seats:
            self.update(u, signs[u], iteration)

    def average_numpy(self) -> np.ndarray:
        """Return a full-slot average profile at an explicit audit boundary."""
        compressed = np.asarray(self.average_device(), dtype=np.float32) \
            .astype(np.float64)
        # Public profiles require each distribution to sum to one at fp64
        # tolerance. Float32 division can leave a few ulps of residue; repair
        # that representation boundary without changing the Metal iterate.
        for u in self.live_seats:
            base = int(self.fl.c_soff[u])
            iso = np.asarray(self.fl.iso[u], dtype=np.int64)
            a, b = base, int(self.fl.c_soff[u + 1])
            if b > a:
                totals = np.add.reduceat(compressed[a:b], iso[:-1])
                compressed[a:b] /= np.repeat(totals, np.diff(iso))
        out = self.fl.sig0_full.copy()
        out[self.fl.c_flat] = compressed
        return out

    def average_device(self):
        """Return the compressed average profile without leaving Metal."""
        parts = []
        for u in range(4):
            if u not in self.live_seats:
                parts.append(self.sig[u])
                continue
            ni = int(self.fl.c_isoff[u + 1] - self.fl.c_isoff[u])
            ns = int(self.fl.c_soff[u + 1] - self.fl.c_soff[u])
            if ni == 0:
                parts.append(self.sig[u])
                continue
            out, = _launch(self._avg, [self.iso[u], self.avg[u], self.inv[u]],
                           ni, [(ns,)], [mx.float32])
            parts.append(out)
        return mx.concatenate(parts)

    def _backward_value(self, sig):
        v = self.payleaf
        for j in range(self.ws.L - 1, -1, -1):
            wave = self.waves[j]
            np_ = int(self.ws.S[j])
            if wave.poff is None:
                v, = _launch(self._bwd_map, [wave.ps, wave.cgid, sig, v],
                             np_, [(np_,)], [mx.float32])
            else:
                v, = _launch(self._bwd_seg,
                             [wave.poff, wave.perm, wave.cgid, sig, v],
                             np_, [(np_,)], [mx.float32])
        return mx.sum(self.w0 * v) / np.float32(self.ws.total_w)

    def _best_response(self, sig, u: int, sign: int):
        rmu = self.w0
        ru = self._ones
        rmu_store = []
        for wave in self.waves:
            rmu_store.append(rmu)
            n = int(wave.ps.shape[0])
            rmu, ru = _launch(
                self._fwd,
                [wave.ps, wave.cgid, wave.eseat, self._u[u], sig, rmu, ru],
                n, [(n,), (n,)], [mx.float32, mx.float32])

        v = self.payleaf
        empty_pick = mx.zeros((1,), dtype=mx.float32)
        zero_base = mx.array([0], dtype=mx.int32)
        for j in range(self.ws.L - 1, -1, -1):
            wave = self.waves[j]
            got = self.cfju.get((j, u))
            if got is None:
                pick, base = empty_pick, zero_base
            else:
                goff, gids, gids_dev, perm, iso, base = got
                ng = len(gids)
                score, = _launch(
                    self._cf_seg,
                    [goff, gids_dev, perm, wave.ps, rmu_store[j], v],
                    ng, [(ng,)], [mx.float32])
                ni = int(iso.shape[0]) - 1
                pick, = _launch(self._br_pick,
                                [iso, score, self._sign[int(sign)]], ni,
                                [(ng,)], [mx.float32])
            np_ = int(self.ws.S[j])
            if wave.poff is None:
                v, = _launch(
                    self._br_bwd_map,
                    [wave.ps, wave.cgid, wave.eseat, self._u[u], sig, v,
                     pick, base], np_, [(np_,)], [mx.float32])
            else:
                v, = _launch(
                    self._br_bwd_seg,
                    [wave.poff, wave.perm, wave.ps, wave.cgid, wave.eseat,
                     self._u[u], sig, v, pick, base],
                    np_, [(np_,)], [mx.float32])
        return mx.sum(self.w0 * v) / np.float32(self.ws.total_w)

    def measure(self, signs: dict[int, int]):
        """GPU value and exact-in-structure single-seat BR gap."""
        sig = self.average_device()
        vbar = self._backward_value(sig)
        brs = {u: self._best_response(sig, u, signs[u])
               for u in self.live_seats}
        mx.eval(vbar, brs)
        value = float(vbar.item())
        gap = max((signs[u] * (float(brs[u].item()) - value)
                   for u in self.live_seats), default=0.0)
        return value, gap

    def debug_numpy(self):
        mx.synchronize()
        return {
            "sig_c": np.asarray(self._global_sig(), dtype=np.float32),
            "reg_c": np.asarray(mx.concatenate([self.reg[u]
                                                 for u in range(4)]),
                                dtype=np.float32),
            "avg_c": np.asarray(mx.concatenate([self.avg[u]
                                                 for u in range(4)]),
                                dtype=np.float32),
        }

    @property
    def peak_memory(self) -> int:
        return int(mx.get_peak_memory())

    def synchronize(self) -> None:
        mx.synchronize()
