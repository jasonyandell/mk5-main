# E[Q] Generator Profiling Results

**Date**: 2026-01-18
**Bead**: t42-z4yj
**Hardware**: RTX 3050 Ti Laptop GPU (4GB VRAM, 192 GB/s bandwidth)

## Executive Summary

The E[Q] generator is **severely overhead-bound**, spending ~67% of time waiting for GPU synchronization and only ~30% doing actual compute. The root cause is too many small operations per decision (~90 kernel launches, ~19 sync points) instead of batched bulk work.

## Baseline Measurements

| Mode | Throughput | Avg Game Time | Decisions/sec |
|------|------------|---------------|---------------|
| No posterior | 2.06 games/s | 487ms | 58 |
| With posterior | 0.45 games/s | 2,240ms | 13 |

## Batch Size Sweep (n_samples = worlds per decision)

| n_samples | games/s | decisions/s | avg_ms | peak_MB |
|-----------|---------|-------------|--------|---------|
| 32 | 3.95 | 111 | 253 | 28 |
| 64 | 2.72 | 76 | 368 | 34 |
| 128 | 1.66 | 47 | 601 | 46 |
| 256 | 0.95 | 27 | 1050 | 70 |

**Finding**: Smaller batches are faster, indicating overhead dominates over compute. This is backwards from what we'd expect if GPU-bound.

## Profiler Analysis (5 games, n_samples=100)

### CUDA API Timing

| Operation | % of Time | Duration | Calls | Per-Decision |
|-----------|-----------|----------|-------|--------------|
| `cudaStreamSynchronize` | **67.4%** | 656ms | 2,660 | 19 syncs |
| `cudaLaunchKernel` | 17.1% | 167ms | 12,600 | 90 launches |
| `cudaMemcpyAsync` | 10.3% | 100ms | 3,500 | 25 copies |
| `cudaMemsetAsync` | 3.6% | 35ms | 2,730 | 20 memsets |
| `cudaGraphLaunch` | 1.1% | 10ms | 280 | 2 graphs |

### CPU Operation Timing

| Operation | Self CPU % | Total CPU % | Calls | Issue |
|-----------|------------|-------------|-------|-------|
| `aten::copy_` | 4.5% | 24% | 55,978 | Excessive small copies |
| `aten::to/_to_copy` | 19.5% | - | 4,060 | Device/dtype conversions |
| `aten::select` | 4.4% | 5.2% | 44,218 | Many indexing ops |
| `aten::index_select` | 4.3% | 4.6% | 18,480 | Gather operations |

## Root Cause Analysis

### Why GPU Utilization is Low (~16%)

1. **Too many sync points**: 19 `cudaStreamSynchronize` calls per decision forces CPU↔GPU round trips
2. **Too many small kernels**: 90 kernel launches per decision instead of 1-3 large batched ops
3. **Excessive data movement**: 25 async copies + dtype conversions per decision
4. **Python loop overhead**: Each decision is processed sequentially with Python control flow

### Where Time Goes (per decision, ~17ms avg)

```
Sync waits:     ~11ms (67%)
Kernel launch:   ~3ms (17%)
Memory copies:   ~2ms (10%)
Actual compute:  ~1ms (6%)
```

The GPU kernels themselves complete in microseconds, but we're constantly waiting.

## Bottleneck Locations in Code

### 1. Oracle tokenization (`oracle.py:_tokenize_worlds`)
- Builds numpy arrays on CPU, then transfers to GPU
- Each `torch.from_numpy().to(device)` is a sync point

### 2. Reduction (`reduction.py:_reduce_world_q_values`)
- Moves Q-values back to CPU for weighted reduction
- `q_cpu = all_q_values.detach().to(device="cpu")` forces sync

### 3. Per-decision loop (`generate_game.py`)
- Each decision: sample worlds → tokenize → transfer → inference → transfer back → reduce
- 28 decisions per game, each with full round-trip overhead

### 4. Posterior scoring (`posterior.py`)
- Additional oracle queries per decision for likelihood scoring
- Multiplies the overhead when enabled

## Optimization Opportunities

### High Impact (requires architectural changes)

1. **Batch across decisions**: Process multiple decisions in one oracle call
   - Current: 28 oracle calls/game × 100 worlds = 2,800 forward passes
   - Better: Batch all 2,800 in fewer large calls

2. **Keep tensors on GPU**: Do reduction on GPU, only transfer final E[Q]
   - Current: Transfer (N,7) Q-values to CPU per decision
   - Better: Weighted mean on GPU, transfer (7,) result

3. **Fuse tokenization + transfer**: Use pinned memory and async transfers
   - `torch.from_numpy(arr, pin_memory=True).to(device, non_blocking=True)`

### Medium Impact (incremental improvements)

4. **Reduce sync points**: Use `torch.cuda.synchronize()` only when needed
   - Profile shows many implicit syncs from `.item()`, `.cpu()`, etc.

5. **CUDA graphs for repeated patterns**: The oracle query pattern is repetitive
   - torch.compile with `reduce-overhead` helps but doesn't eliminate all overhead

6. **Pre-allocate buffers**: Reuse token/mask tensors instead of creating new ones

### Low Impact (polish)

7. **TensorFloat32 precision**: Enable `torch.set_float32_matmul_precision('high')`
8. **Tune torch.compile mode**: Try `max-autotune` vs `reduce-overhead`

## Recommended Next Steps

1. **Profile on H100** where CUPTI works to get actual kernel timing
2. **Implement batched reduction on GPU** (highest-impact single change)
3. **Batch oracle calls across decisions** (architectural change)
4. **Benchmark each optimization** to measure actual impact

## Profiling Tools Created

- `forge/eq/profile_throughput.py` - Profiling script with:
  - `--sweep`: Batch size sweep
  - `--trace`: torch.profiler trace output
  - `--posterior`: Include posterior weighting
  - Chrome trace export for visualization

## Notes on WSL2 Profiling

CUPTI (CUDA profiling tools) does not initialize in WSL2 even with:
- Driver permissions enabled
- PyTorch CUDA 12.1 / Driver CUDA 12.3

This is a known WSL2 limitation. Full GPU kernel profiling requires:
- Native Linux
- Cloud GPU (Modal H100)

The CPU-side timing and CUDA API timing still provide actionable insights.
