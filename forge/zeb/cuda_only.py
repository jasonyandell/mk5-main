"""CUDA-only guardrails for GPU-native Zeb code.

The GPU-native MCTS and training pipeline must never silently run on CPU.
Use these helpers at entrypoints to crash fast when CUDA is unavailable or a
CPU device/tensor is passed.
"""

from __future__ import annotations

import torch
from torch import Tensor


def require_cuda(device: torch.device | str | None = None, *, where: str) -> torch.device:
    """Return a CUDA device or raise a clear error."""
    if not torch.cuda.is_available():
        raise RuntimeError(f"{where} requires CUDA, but torch.cuda.is_available() is False.")

    if device is None:
        return torch.device("cuda")

    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(f"{where} requires a CUDA device, got {dev!s}.")
    return dev


def require_cuda_tensor(tensor: Tensor, *, where: str, name: str = "tensor") -> None:
    """Raise if the tensor is not on CUDA."""
    if tensor.device.type != "cuda":
        raise RuntimeError(f"{where} requires {name} to be on CUDA, got {tensor.device!s}.")

