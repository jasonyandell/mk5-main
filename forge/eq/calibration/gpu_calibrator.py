"""GPU chunk size calibration for model forward passes.

Finds optimal batch size for maximum throughput on the current GPU.
Results are cached per GPU model for reuse.

Usage:
    from forge.eq.calibration import get_optimal_chunk, calibrate_chunk_size

    # Get cached or calibrate
    chunk_size = get_optimal_chunk(model)

    # Force recalibration
    calibration = calibrate_chunk_size(model, force=True)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, Tensor

CACHE_PATH = Path(__file__).parent / "gpu_chunks.json"


@dataclass
class GPUCalibration:
    """Calibration results for a specific GPU."""
    gpu_name: str
    optimal_chunk: int
    tokens_per_sec: float
    vram_gb: float
    calibrated_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'GPUCalibration':
        return cls(**d)


def get_optimal_chunk(model: nn.Module, device: str = 'cuda') -> int:
    """Get optimal chunk size, calibrating if needed.

    Args:
        model: The model to calibrate for
        device: Device to calibrate on

    Returns:
        Optimal chunk size for model forward passes
    """
    calibration = _load_cached_calibration(device)
    if calibration is not None:
        return calibration.optimal_chunk

    # Need to calibrate
    calibration = calibrate_chunk_size(model, device)
    return calibration.optimal_chunk


def calibrate_chunk_size(
    model: nn.Module,
    device: str = 'cuda',
    force: bool = False,
) -> GPUCalibration:
    """Calibrate optimal chunk size for this GPU.

    Scans batch sizes to find peak throughput, then caches result.

    Args:
        model: The model to calibrate for
        device: Device to calibrate on
        force: If True, recalibrate even if cached

    Returns:
        GPUCalibration with optimal settings
    """
    if not force:
        cached = _load_cached_calibration(device)
        if cached is not None:
            return cached

    gpu_name = torch.cuda.get_device_properties(0).name
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"Calibrating chunk size for {gpu_name} ({vram_gb:.1f} GB)...", flush=True)

    # Scan batch sizes
    results = _scan_batch_sizes(model, device)

    if not results:
        raise RuntimeError("Calibration failed - no valid batch sizes found")

    # Find peak throughput
    best_batch, best_tps = max(results, key=lambda x: x[1])

    # Apply 75% headroom for other pipeline allocations
    # (world sampling, tokenizer, PDF, state tensors)
    best_batch = int(best_batch * 0.75)

    calibration = GPUCalibration(
        gpu_name=gpu_name,
        optimal_chunk=best_batch,
        tokens_per_sec=best_tps,
        vram_gb=round(vram_gb, 1),
        calibrated_at=datetime.now().isoformat(timespec='seconds'),
    )

    # Save to cache
    _save_calibration(calibration)

    print(f"Optimal chunk: {best_batch} ({best_tps:,.0f} tokens/s)", flush=True)

    return calibration


def _scan_batch_sizes(
    model: nn.Module,
    device: str,
    n_runs: int = 3,
) -> list[tuple[int, float]]:
    """Scan batch sizes and measure throughput.

    Uses early exit when throughput drops significantly.
    """
    # Batch sizes to test (powers of 2 + some intermediates)
    candidates = [1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536]

    results = []
    peak_tps = 0

    for batch_size in candidates:
        try:
            tps = _measure_throughput(model, device, batch_size, n_runs)
            results.append((batch_size, tps))

            if tps > peak_tps:
                peak_tps = tps

            # Early exit if throughput drops significantly (hit the cliff)
            if tps < peak_tps * 0.5:
                break

            torch.cuda.empty_cache()

        except (torch.cuda.OutOfMemoryError, RuntimeError):
            # Hit hard OOM limit
            torch.cuda.empty_cache()
            break

    return results


def _measure_throughput(
    model: nn.Module,
    device: str,
    batch_size: int,
    n_runs: int = 3,
) -> float:
    """Measure tokens/sec at given batch size."""
    # Create dummy inputs matching model signature
    tokens = torch.zeros(batch_size, 32, 12, dtype=torch.int32, device=device)
    masks = torch.ones(batch_size, 32, dtype=torch.int32, device=device)
    players = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Warmup
    with torch.inference_mode():
        _ = model(tokens, masks, players)
    torch.cuda.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.inference_mode():
            _ = model(tokens, masks, players)
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens_per_sec = (batch_size * n_runs) / elapsed
    return tokens_per_sec


def _load_cached_calibration(device: str) -> GPUCalibration | None:
    """Load cached calibration for current GPU."""
    if device != 'cuda' or not torch.cuda.is_available():
        return None

    if not CACHE_PATH.exists():
        return None

    gpu_name = torch.cuda.get_device_properties(0).name

    try:
        cache = json.loads(CACHE_PATH.read_text())
        if gpu_name in cache:
            return GPUCalibration.from_dict(cache[gpu_name])
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return None


def _save_calibration(calibration: GPUCalibration) -> None:
    """Save calibration to cache file."""
    # Load existing cache
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            cache = {}
    else:
        cache = {}

    # Update with new calibration
    cache[calibration.gpu_name] = calibration.to_dict()

    # Write back
    CACHE_PATH.write_text(json.dumps(cache, indent=2) + "\n")
