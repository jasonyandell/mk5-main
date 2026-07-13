"""Stage 0 CUDA throughput bench for uniform-completion-dp-v1.

Shapes: (32,50) audit reference (CPU 4.25 ms), (128,100) and (256,100)
production eval-aux shapes. Reports ms/call, worlds/sec, peak MB.
Run: python -u bench_sampler.py [--device cuda]
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
from forge.eq.sampling_mrv_gpu import sample_worlds_mrv_gpu  # noqa: E402

SHAPES = [(32, 50), (128, 100), (256, 100)]
WARMUP = 3
ITERS = 20


def bench(device: str) -> None:
    dev = torch.device(device)
    print(f"device={device} torch={torch.__version__}", flush=True)
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    for n_games, n_samples in SHAPES:
        # 21 unseen dominoes split 7/7/7 across three hidden seats, no voids.
        pools = torch.arange(21, dtype=torch.int64, device=dev).unsqueeze(0).expand(n_games, 21).contiguous()
        hand_sizes = torch.full((n_games, 3), 7, dtype=torch.int64, device=dev)
        voids = torch.zeros(n_games, 3, 8, dtype=torch.bool, device=dev)
        decl_ids = torch.full((n_games,), 9, dtype=torch.int64, device=dev)
        for _ in range(WARMUP):
            sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples=n_samples, device=dev)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            sample_worlds_mrv_gpu(pools, hand_sizes, voids, decl_ids, n_samples=n_samples, device=dev)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / ITERS
        worlds_per_s = n_games * n_samples / dt
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
        print(
            f"shape=({n_games},{n_samples}) ms_per_call={dt * 1e3:.2f} "
            f"worlds_per_s={worlds_per_s:,.0f} peak_mb={peak_mb:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    bench(args.device)
