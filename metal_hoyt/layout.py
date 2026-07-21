"""metal_hoyt/layout.py — GPU-resident structure from hoyt's verified build.

The single source of structure truth is hoyt's CPU build: `_build_wave`
(the full-width walk + info-set index) and `_build_fused(par=True)` (the
P13 groupings — vseg/cf_ju/iso — stable argsorts computed once). This
module packs those arrays into int32/fp32 MLX buffers and ASSERTS the
three contiguity invariants DESIGN.md documents; it never re-derives
structure.

Everything here is per-root and immutable after construction; the engine
holds the mutable fp32 state.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx

__all__ = ["MetalLayout"]


def _i32(a) -> mx.array:
    return mx.array(np.ascontiguousarray(a, dtype=np.int32))


def _f32(a) -> mx.array:
    return mx.array(np.ascontiguousarray(a, dtype=np.float32))


def _scalar_i32(v: int) -> mx.array:
    return mx.array(np.array([v], dtype=np.int32))


class MetalLayout:
    """Immutable GPU structure for one subgame's CFR solve.

    Wave arrays (per transition j, sized by wave j+1's slots):
      ps[j], cgid[j], eseat[j]        — edge topology (int32/int32/int8)
      poff[j], perm[j]                — parent segmentation (bincount fold)
    Per (wave j, seat u) blocks, each a contiguous run of the seat's
    compressed slice (asserted):
      cf_blocks[u]  = [(j, goff, perm, slot_lo, n_slots)]  wave-ascending
      xi_blocks[u]  = [(j, reps)]                          wave-ascending
      iso_blocks[u] = [(j, iso_loc, iset_lo, n_isets)]     wave-ascending
    Seat-level:
      iso[u] — per-iset slot offsets over the seat slice (rm_update /
      asig_norm); inv[u] — uniform fallback per slot; n_slots/n_isets.
    """

    def __init__(self, ws, fl, payoff43):
        self.L = ws.L
        self.n_c = fl.n_c
        self.c_soff = np.asarray(fl.c_soff, dtype=np.int64)
        self.c_isoff = np.asarray(fl.c_isoff, dtype=np.int64)
        self.S = list(ws.S)

        # ---- wave topology + parent segmentation ------------------------
        self.ps, self.cgid, self.eseat = [], [], []
        self.poff, self.perm = [], []
        self.nE, self.nS = [], []
        for j in range(ws.L):
            ps = fl.ps32[j]
            self.nE.append(len(ps))
            self.nS.append(ws.S[j])
            self.ps.append(_i32(ps))
            self.cgid.append(_i32(fl.cgid32[j]))
            self.eseat.append(mx.array(
                np.ascontiguousarray(fl.eseat8[j], dtype=np.int8)))
            seg = fl.vseg[j] if fl.vseg is not None else None
            if seg is None:
                # bijection wave or fl built without par: derive the same
                # stable segmentation the fused lane would
                cnt = np.bincount(ps, minlength=ws.S[j])
                poff = np.zeros(ws.S[j] + 1, dtype=np.int64)
                np.cumsum(cnt, out=poff[1:])
                perm = np.argsort(ps, kind="stable")
            else:
                poff, perm = seg
            self.poff.append(_i32(poff))
            self.perm.append(_i32(perm))

        # ---- per-seat compressed slices ---------------------------------
        self.n_slots = [int(self.c_soff[u + 1] - self.c_soff[u])
                        for u in range(4)]
        self.n_isets = [int(self.c_isoff[u + 1] - self.c_isoff[u])
                        for u in range(4)]
        self.inv = [None] * 4
        self.iso = [None] * 4
        self.sig0 = [None] * 4          # fp32 initial sigma per seat slice
        sig0_c = fl.sig0_full[fl.c_flat]
        for u in range(4):
            sl = slice(int(self.c_soff[u]), int(self.c_soff[u + 1]))
            if self.n_slots[u] == 0:
                continue
            self.inv[u] = _f32(fl.inv_nleg[sl])
            self.sig0[u] = _f32(sig0_c[sl])
            iso_u = fl.iso[u] if fl.iso is not None else None
            if iso_u is None:
                isl = fl.c_isl_loc[sl]
                iso_u = np.searchsorted(isl, np.arange(self.n_isets[u] + 1))
            self.iso[u] = _i32(iso_u)

        # ---- per (j, u) blocks + the contiguity invariants --------------
        self.cf_blocks = [[] for _ in range(4)]
        self.xi_blocks = [[] for _ in range(4)]
        self.iso_blocks = [[] for _ in range(4)]
        iso_np = [None] * 4
        for u in range(4):
            if self.n_slots[u]:
                sl = slice(int(self.c_soff[u]), int(self.c_soff[u + 1]))
                isl = fl.c_isl_loc[sl]
                iso_np[u] = np.searchsorted(
                    isl, np.arange(self.n_isets[u] + 1)).astype(np.int64)
        cover_slots = [[] for _ in range(4)]
        cover_isets = [[] for _ in range(4)]
        for j in range(ws.L):
            for u in range(4):
                cfj = fl.cf_ju.get((j, u)) if fl.cf_ju is not None else None
                cij = fl.c_iju.get((j, u))
                if cfj is None and cij is None:
                    continue
                if cfj is None or cij is None:
                    raise AssertionError(
                        f"(j={j}, u={u}): cf_ju and c_iju must coexist")
                goff, gids, eperm = cfj
                cids, reps = cij
                # invariant 1: the block's strategy slots are contiguous
                if not np.array_equal(gids,
                                      np.arange(gids[0], gids[0] + len(gids))):
                    raise AssertionError(f"(j={j}, u={u}): slots not contiguous")
                # invariant 2: the block's iset ids are contiguous
                if not np.array_equal(cids,
                                      np.arange(cids[0], cids[0] + len(cids))):
                    raise AssertionError(f"(j={j}, u={u}): isets not contiguous")
                slot_lo = int(gids[0] - self.c_soff[u])   # seat-relative
                iset_lo = int(cids[0] - self.c_isoff[u])
                niju = len(cids)
                iso_u = iso_np[u]
                iso_loc = iso_u[iset_lo:iset_lo + niju + 1] - iso_u[iset_lo]
                if int(iso_u[iset_lo]) != slot_lo or \
                        int(iso_loc[-1]) != len(gids):
                    raise AssertionError(
                        f"(j={j}, u={u}): iso block misaligned with slots")
                self.cf_blocks[u].append(
                    (j, _i32(goff), _i32(eperm), slot_lo, len(gids)))
                self.xi_blocks[u].append((j, _i32(reps)))
                self.iso_blocks[u].append((j, _i32(iso_loc), iset_lo, niju))
                cover_slots[u].append((slot_lo, len(gids)))
                cover_isets[u].append((iset_lo, niju))
        # invariant 3: wave-ascending blocks tile each seat's slice exactly
        for u in range(4):
            pos = 0
            for lo, n in cover_slots[u]:
                if lo != pos:
                    raise AssertionError(f"seat {u}: slot tiling gap at {pos}")
                pos += n
            if pos != self.n_slots[u]:
                raise AssertionError(f"seat {u}: slot tiling short at {pos}")
            pos = 0
            for lo, n in cover_isets[u]:
                if lo != pos:
                    raise AssertionError(f"seat {u}: iset tiling gap at {pos}")
                pos += n
            if pos != self.n_isets[u]:
                raise AssertionError(f"seat {u}: iset tiling short at {pos}")

        # ---- leaves and root weights ------------------------------------
        pay = np.asarray(payoff43, dtype=np.float64)
        self.payleaf = _f32(pay[ws.leaf_pts])
        self.w0n = _f32(ws.w0 / ws.total_w)

        # scalar buffers reused across dispatches
        self.u_scal = [_scalar_i32(u) for u in range(4)]
        self.u_none = _scalar_i32(-1)
        self.zero_i = _scalar_i32(0)
        self.n_scal = {}
        for n in set(self.nE) | set(self.nS) | set(self.n_slots) \
                | set(self.n_isets) \
                | {n for u in range(4) for _, _, _, n in self.iso_blocks[u]} \
                | {n for u in range(4) for _, _, _, _, n in self.cf_blocks[u]}:
            self.n_scal[int(n)] = _scalar_i32(int(n))
        self.dummy_f = _f32(np.zeros(1))

    def nsc(self, n: int) -> mx.array:
        s = self.n_scal.get(int(n))
        if s is None:
            s = _scalar_i32(int(n))
            self.n_scal[int(n)] = s
        return s
