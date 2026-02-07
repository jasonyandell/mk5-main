# Phase 4: Async Pipeline with CUDA Streams

**Bead**: t42-tg2r
**Date**: 2026-01-18
**Status**: Complete

## Overview

Phase 4 implements an async pipeline using CUDA streams to overlap CPU and GPU work in the E[Q] generator. The goal is to reduce GPU idle time by overlapping data transfers with computation.

## Problem Statement

From profiling (see `profiling_results.md`):
- **67% of time** spent in `cudaStreamSynchronize` (waiting for GPU)
- **19 sync points per decision** forcing CPU↔GPU round trips
- GPU utilization only ~16%
- Small batches faster than large batches (overhead-bound, not compute-bound)

The bottleneck is synchronization overhead, not compute capacity.

## Solution: CUDA Streams + Non-blocking Transfers

### Key Concepts

1. **CUDA Streams**: Allow async GPU operations to run concurrently
2. **Non-blocking H2D transfers**: `tensor.to(device, non_blocking=True)` returns immediately
3. **Pinned memory**: Page-locked host memory enables fast DMA transfers
4. **Double buffering**: Two sets of GPU buffers to enable pipelining (future optimization)

### Architecture

```
CPU Thread                  CUDA Stream H2D         CUDA Stream Compute
──────────                  ───────────────         ───────────────────

1. Tokenize (numpy)
   ↓
2. Copy to pinned memory ──→ 3. Async H2D transfer
   (CPU-only, no GPU wait)      (DMA, no CPU wait)
                                   ↓
                                   └──→ 4. Wait for H2D
                                        5. Inference
                                           ↓
6. Return (sync on .clone())  ←──────────┘
```

### Implementation Details

#### Pinned Host Memory

```python
self._pinned_tokens = torch.empty(
    (size, MAX_TOKENS, N_FEATURES),
    dtype=torch.int32,
    pin_memory=True  # Enables fast DMA, bypassing CPU cache
)
```

Benefits:
- 2-3x faster H2D transfers vs pageable memory
- GPU can access directly via DMA without OS kernel involvement
- Pre-allocated with headroom to avoid frequent reallocs

#### Double Buffering

```python
self._gpu_buffers = [
    {'tokens': ..., 'masks': ..., 'current_player': ...}
    for _ in range(2)
]
```

Current implementation alternates buffers between calls, preparing for future pipelining where:
- GPU processes batch N on buffer 0
- CPU prepares batch N+1 on buffer 1
- Swap buffers, repeat

Full pipelining would require API changes to expose "submit batch" + "wait batch".

#### Async H2D Transfers

```python
with torch.cuda.stream(self.stream_h2d):
    tokens_gpu.copy_(pinned_tokens, non_blocking=True)
    masks_gpu.copy_(pinned_masks, non_blocking=True)
```

`non_blocking=True` means:
- Returns immediately
- GPU DMA engine runs in background
- CPU continues to next operation without waiting

#### Stream Synchronization

```python
self.stream_compute.wait_stream(self.stream_h2d)
```

Before inference, ensure H2D transfer is complete. This is necessary because inference depends on the transferred data.

Only sync point: `q_values.clone()` implicitly syncs to ensure results are ready.

## Usage

### Enable Async Mode (Default)

```python
oracle = Stage1Oracle(
    checkpoint_path,
    device="cuda",
    use_async=True  # Default on CUDA
)
```

### Disable Async Mode (Fallback)

```python
oracle = Stage1Oracle(
    checkpoint_path,
    device="cuda",
    use_async=False  # Sync mode for debugging
)
```

Async mode automatically disabled on CPU/MPS devices.

## API Compatibility

All existing APIs unchanged:
- `oracle.query_batch(...)` - Works identically
- `oracle.query_batch_multi_state(...)` - Works identically

Internally routes to `_query_batch_async()` or sync path based on `use_async` flag.

## Benchmark

Run benchmark to measure improvement:

```bash
python -m forge.eq.benchmark_async \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
    --n-worlds 100 \
    --n-iterations 50
```

Expected improvements (estimated):
- **5-15% latency reduction** from non-blocking H2D transfers
- **Minor memory overhead** from pinned buffers (~2x buffer size)
- **No accuracy impact** (same computation, just reordered)

Note: Full pipelining (overlapping batch N+1 prep with batch N inference) would require generator refactoring to prepare batches ahead of time.

## Testing

All existing tests pass with async mode:

```bash
python -m pytest forge/eq/test_oracle.py -v
```

New test added:
- `test_async_mode_produces_same_results` - Verifies sync/async consistency

## Memory Usage

Async mode adds:
- **Pinned host memory**: 2x tokenization buffer size (~8 MB for 256 worlds)
- **Double GPU buffers**: 2x GPU buffer size (~4 MB for 256 worlds)
- Total overhead: ~12 MB (negligible on modern GPUs)

Buffers grow dynamically with headroom (2x) to avoid frequent reallocs.

## Limitations & Future Work

### Current Limitations

1. **Single-call pipelining**: Each `query_batch()` is independent
   - Can't overlap batch N+1 prep with batch N inference
   - Would need API changes to submit multiple batches

2. **Sync on return**: `q_values.clone()` forces wait
   - Could return async tensor, let caller decide when to sync
   - Requires API change

3. **Per-decision overhead**: Still 28 oracle calls per game
   - Phase 3 (batched generator) addresses this
   - Combining Phase 3 + Phase 4 = maximum benefit

### Future Optimizations

1. **Batch submission API**:
   ```python
   handle = oracle.submit_batch(worlds, ...)  # Returns immediately
   q_values = oracle.wait_batch(handle)       # Sync only when needed
   ```

2. **Producer-consumer pattern**:
   ```python
   # CPU: Prepare batches in background thread
   # GPU: Process batches from queue
   ```

3. **Keep reduction on GPU**:
   - Currently transfer Q-values to CPU for weighted mean
   - Could do weighted reduction on GPU, transfer only E[Q]
   - See `reduction.py` for opportunity

## Integration with Phase 3

Phase 3 (batched generator) batches oracle calls **across games**.
Phase 4 (async pipeline) batches **H2D transfers**.

Combined effect:
- Phase 3: Fewer oracle calls (batch N games → 1 call)
- Phase 4: Each call has lower overhead (async H2D)
- Together: Maximize GPU utilization

## Debugging

Disable async mode to isolate issues:

```python
oracle = Stage1Oracle(checkpoint, device="cuda", use_async=False)
```

Check CUDA stream synchronization:

```python
torch.cuda.current_stream().synchronize()  # Ensure all ops complete
```

Profile with:

```python
with torch.profiler.profile() as prof:
    oracle.query_batch(...)
print(prof.key_averages().table())
```

## References

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CUDA C Programming Guide - Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

## Checklist

- [x] Implement async H2D transfers with pinned memory
- [x] Add CUDA stream management (H2D + compute)
- [x] Double buffering infrastructure
- [x] Backward compatibility (sync fallback)
- [x] Tests pass (all existing + new async test)
- [x] Benchmark script
- [x] Documentation
- [ ] Performance validation on GPU hardware
- [ ] Integration with Phase 3 batched generator
