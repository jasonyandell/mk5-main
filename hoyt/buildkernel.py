"""hoyt/buildkernel.py — numba kernels for the CFR structural build.

The engine="fused" lane's expand_full_width provider (perf-log 18n): the
numpy mirror materializes a (slots, 28) bool matrix per wave and scans it
with np.nonzero to emit (slot, move) pairs — ~1 s of nonzero plus the
matrix allocation per anchor build. This kernel emits the pairs straight
off the legal-move bitmasks in the SAME ascending (slot, move) order
(np.nonzero's row-major order), so the walk's output is identical by
construction — integer structure, no floats, no tolerance.

The wave/loop lanes stay numba-free (CONTRACTS.md): the numpy provider
path remains the pinned mirror, selected by _FullWidthProvider(kernels=
False), and the standing fused-vs-wave bitwise parity gates therefore
cover this kernel too.
"""
from __future__ import annotations

from numba import njit

__all__ = ["fw_fill"]


@njit(cache=True, boundscheck=False)
def fw_fill(lm, off, si, mv):
    """Scatter each slot's legal-move bitmask into flat (si, mv) arrays,
    ascending (slot, move); off[i] is the running popcount offset of slot
    i (caller computes it from np.bitwise_count + cumsum)."""
    for i in range(lm.shape[0]):
        m = lm[i]
        k = off[i]
        t = 0
        while m:
            if m & 1:
                si[k] = i
                mv[k] = t
                k += 1
            m >>= 1
            t += 1
