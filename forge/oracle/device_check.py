"""Startup self-check: int64 sort/unique must be exact on the chosen device (#137).

On Apple MPS (torch 2.11) ``torch.unique`` keeps only the low 32 bits of int64
values and ``torch.sort`` rounds them as float32, so state enumeration silently
collapses. Forge is GPU-only (no CPU fallback), so the remedy is to fail fast.
"""
from __future__ import annotations

import torch

# > 2^32, and not exactly representable in float32 (0x1f84ffffffe -> 0x1f850000000).
_PROBE = (0x1F84FFFFFFE, 0x1F84FFFFFFF, 0x1F850000000, 0x100000001, 0x1, 0x1F84FFFFFFE)
_SORTED = sorted(_PROBE)
_UNIQUE = sorted(set(_PROBE))

_verified: set[tuple[str, str]] = set()


def int64_ops_failure(device: torch.device | str) -> str | None:
    """Return a description of the failure, or None if sort/unique are exact on device."""
    t = torch.tensor(_PROBE, dtype=torch.int64, device=device)
    got_sort = torch.sort(t).values.cpu().tolist()
    if got_sort != _SORTED:
        return f"torch.sort corrupted int64: got {[hex(x) for x in got_sort]}, expected {[hex(x) for x in _SORTED]}"
    got_uniq = torch.unique(t).cpu().tolist()
    if got_uniq != _UNIQUE:
        return f"torch.unique corrupted int64: got {[hex(x) for x in got_uniq]}, expected {[hex(x) for x in _UNIQUE]}"
    return None


def require_exact_int64_ops(device: torch.device | str) -> None:
    """Raise RuntimeError if int64 sort/unique are wrong on ``device``. No fallback."""
    dev = torch.device(device)
    key = (str(dev), torch.__version__)
    if key in _verified:
        return
    failure = int64_ops_failure(dev)
    if failure is not None:
        raise RuntimeError(
            f"forge oracle refuses to run on device '{dev}' with torch {torch.__version__}: "
            f"{failure}. Results would be silently wrong (see issue #137). "
            "Forge is GPU-only; use CUDA or a torch build where int64 sort/unique are verified."
        )
    _verified.add(key)
