"""metal_hoyt — the GPU searcher under hoyt's fp64 referee (DESIGN.md).

Public surface:
    cfr_solve_metal(subgame, payoff43, ...) -> hoyt.cfr.CFRResult
        CFR+ iterate + intermediate gap on the Metal GPU (fp32, raw MSL
        kernels); returned gap/value certified fp64 by hoyt's exact BR.

Requires a Metal GPU — no CPU fallback (hoyt IS the CPU lane).
"""
from metal_hoyt.engine import cfr_solve_metal

__all__ = ["cfr_solve_metal"]
