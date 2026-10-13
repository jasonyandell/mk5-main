"""Tests for the int64 sort/unique startup self-check (#137).

No skips (repo rule): the GPU test adapts to whatever accelerator is present.
CPU appears only as a control for the check itself, never as a forge path.
"""
from __future__ import annotations

import pytest
import torch

from forge.oracle import device_check
from forge.oracle.context import build_context
from forge.oracle.device_check import int64_ops_failure, require_exact_int64_ops


def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def test_known_good_tensor_passes():
    # Control: CPU int64 ops are exact, so the check must accept them.
    assert int64_ops_failure("cpu") is None
    require_exact_int64_ops("cpu")


def test_available_accelerator_is_accepted_or_refused_loudly():
    dev = _accelerator()
    if dev is None:
        assert torch.cuda.is_available() is False  # nothing to check here
        return
    if int64_ops_failure(dev) is None:
        require_exact_int64_ops(dev)  # runs as before
        build_context(seed=0, decl_id=0, device=torch.device(dev))
        return
    # Broken int64 ops (e.g. MPS on torch 2.11): forge must refuse, not compute.
    with pytest.raises(RuntimeError) as e:
        require_exact_int64_ops(dev)
    assert dev in str(e.value) and torch.__version__ in str(e.value) and "#137" in str(e.value)
    with pytest.raises(RuntimeError, match="#137"):
        build_context(seed=0, decl_id=0, device=torch.device(dev))


def test_broken_ops_raise_with_device_and_version(monkeypatch):
    real_unique = torch.unique
    monkeypatch.setattr(torch, "unique", lambda t, *a, **k: real_unique(t & 0xFFFFFFFF, *a, **k))
    monkeypatch.setattr(device_check, "_verified", set())
    msg = int64_ops_failure("cpu")
    assert msg is not None and "torch.unique" in msg
    with pytest.raises(RuntimeError) as e:
        require_exact_int64_ops("cpu")
    assert "cpu" in str(e.value) and torch.__version__ in str(e.value)
