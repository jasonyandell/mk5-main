# Concurrency and Parallel Execution

## Execution Methods

### `.remote()` - Single invocation
```python
result = my_function.remote(arg1, arg2)
```

### `.map()` - Parallel over iterable
```python
results = list(my_function.map([1, 2, 3, 4, 5]))
```

### `.starmap()` - Parallel with multiple args
```python
tasks = [(1, "a"), (2, "b"), (3, "c")]
results = list(my_function.starmap(tasks))
```

### `.spawn()` - Async (non-blocking)
```python
handle = my_function.spawn(arg1, arg2)
# ... do other work ...
result = handle.get()  # Block until done
```

## starmap() Pattern

The primary pattern for batch GPU workloads:

```python
@app.function(gpu="A10G", timeout=600)
def process_shard(seed: int, config: dict) -> dict:
    # Process one unit of work
    return {"seed": seed, "status": "done"}

@app.local_entrypoint()
def batch():
    # Build task list
    tasks = [
        (seed, {"param": value})
        for seed in range(1000)
    ]

    # Execute in parallel across GPU pool
    results = list(process_shard.starmap(tasks))

    # Process results
    success = sum(1 for r in results if r["status"] == "done")
    print(f"Completed: {success}/{len(tasks)}")
```

## @modal.concurrent Decorator

Run multiple inputs per container (same GPU):

```python
@app.function(gpu="A10G")
@modal.concurrent(max_inputs=2)  # 2 concurrent tasks per GPU
def process(seed: int) -> dict:
    # Tasks share GPU VRAM - be mindful of memory
    ...
```

### Parameters

```python
@modal.concurrent(
    max_inputs=4,      # Hard limit per container
    target_inputs=2,   # Autoscaling target (optional)
)
```

- `max_inputs`: Maximum concurrent tasks per container
- `target_inputs`: Target for autoscaler (allows bursting to max_inputs)

### When to Use

**Good for:**
- I/O-bound workloads (waiting on network, disk)
- GPU tasks with headroom (not using full VRAM)
- Continuous batching (vLLM-style inference)

**Avoid for:**
- CPU-bound tasks (use more containers instead)
- Tasks using full GPU memory

### Memory Considerations

With 2x concurrency on A10G (24GB VRAM):
- Each task gets ~12GB effective VRAM
- ~2% of large tasks may OOM
- Acceptable tradeoff for 2x throughput

```python
# If OOM is unacceptable, use 1x concurrency
@app.function(gpu="A10G")
# No @modal.concurrent decorator = 1 task per GPU
def process_large(seed: int) -> dict:
    ...
```

## Autoscaling

Modal automatically scales containers based on demand:

```python
@app.function(
    gpu="A10G",
    # Autoscaling hints
    min_containers=0,     # Scale to zero when idle
    max_containers=100,   # Cap at 100 GPUs
    buffer_containers=2,  # Keep 2 warm for latency
)
def process(seed: int):
    ...
```

### Scaling Behavior

1. Initial request → cold start (container spin-up)
2. Subsequent requests → route to existing containers
3. Demand exceeds capacity → scale up
4. Idle timeout → scale down
5. No requests → scale to zero

## Preventing Duplicate Work

When running parallel jobs (or multiple GPU type variants):

```python
@app.function(gpu="A10G", volumes={"/data": volume})
def process(seed: int) -> dict:
    output_path = Path(f"/data/result_{seed:08d}.parquet")
    lock_path = output_path.with_suffix(".lock")

    # Check if already done or in-progress
    if output_path.exists() or lock_path.exists():
        return {"status": "skipped", "seed": seed}

    # Claim this work immediately
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    try:
        # ... GPU work ...
        result = compute(seed)
        write_parquet(output_path, result)
        return {"status": "done", "seed": seed}
    finally:
        # Clean up lock
        lock_path.unlink(missing_ok=True)
```

## Orchestrator Pattern

CPU orchestrator fans out to GPU workers:

```python
@app.function(gpu="A10G", timeout=300)
def gpu_worker(seed: int) -> dict:
    # GPU-intensive work
    return process_on_gpu(seed)

@app.function(timeout=86400)  # 24 hours - CPU only
def orchestrator(start: int, end: int):
    # Build task list
    tasks = [(seed,) for seed in range(start, end)]

    # Fan out to GPU pool
    results = list(gpu_worker.starmap(tasks))

    # Aggregate
    return {
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "done"),
    }

@app.local_entrypoint()
def main():
    result = orchestrator.remote(0, 10000)
    print(result)
```

## Rate Limiting

For external API calls:

```python
import time

@app.function()
@modal.concurrent(max_inputs=10)
async def call_api(item: str) -> dict:
    # Rate limit to 100 requests/second across all containers
    # Use Modal Queue or external rate limiter
    await asyncio.sleep(0.01)  # Simple throttle
    return await fetch_api(item)
```

## Throughput Optimization Summary

| Technique | Speedup | Notes |
|-----------|---------|-------|
| `starmap()` | Nx | N = number of GPUs |
| `@modal.concurrent(max_inputs=2)` | ~2x | Per-GPU multiplier |
| Remove per-shard commits | ~1.7x | Avoid volume.commit() |
| Lock files | Variable | Prevents wasted duplicate work |
| **Combined** | **~3.8x** | Real-world measurement |
