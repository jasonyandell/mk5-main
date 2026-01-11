---
name: modal
description: Modal serverless GPU cloud - activates for GPU workloads, ML inference, batch processing, or cloud compute tasks. Use when deploying functions to cloud GPUs, running parallel starmap jobs, managing Modal volumes, optimizing GPU utilization, or estimating cloud costs. Covers authentication, GPU selection, concurrency patterns, persistent storage, cost estimation, job monitoring, and common pitfalls.
---

# Modal GPU Infrastructure

Serverless cloud platform for running Python on GPUs. Pay per second, scale to thousands of containers, zero infrastructure management.

## Quick Start

```bash
# First time setup
pip install modal
modal token new        # Opens browser for auth

# Run a job
modal run app.py::main

# Check status
modal app list         # Running apps
modal container list   # Individual containers
```

## Core Pattern

```python
import modal

app = modal.App("my-app")
image = modal.Image.debian_slim().pip_install("torch", "numpy")
volume = modal.Volume.from_name("my-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=600,
)
def gpu_task(seed: int) -> dict:
    import torch
    assert torch.cuda.is_available()
    # GPU work here
    return {"seed": seed, "status": "done"}

@app.local_entrypoint()
def main():
    # Parallel execution across GPU pool
    tasks = [(i,) for i in range(100)]
    results = list(gpu_task.starmap(tasks))
```

## GPU Selection

| GPU | $/hour | VRAM | Use Case |
|-----|--------|------|----------|
| T4 | $0.59 | 16GB | Cheapest, light inference |
| L4 | $0.80 | 24GB | Budget inference |
| **A10G** | $1.10 | 24GB | **Best value, always available** |
| L40S | $1.95 | 48GB | Large inference |
| A100-40GB | $2.10 | 40GB | Training |
| A100-80GB | $2.50 | 80GB | Large training |
| H100 | $3.95 | 80GB | Fast training (often queued) |
| H200 | $4.54 | 141GB | Premium speed |
| B200 | $6.25 | 192GB | Fastest (sometimes queued) |

**Availability ranking**: A10 > A100-80GB > H200 > L40S > B200 > H100

See [references/gpu-selection.md](references/gpu-selection.md) for detailed guidance.

## Performance Optimizations

These optimizations achieved **3.8x speedup** in production:

### 1. Skip per-write volume commits (1.7x speedup)

```python
# BAD - commits after every write (~1s overhead each!)
write_file(path, data)
volume.commit()

# GOOD - let Modal auto-commit on container shutdown
write_file(path, data)
# Volume commits periodically and on shutdown automatically
```

### 2. Enable concurrent inputs (2x throughput)

```python
@app.function(gpu="A10G")
@modal.concurrent(max_inputs=2)  # 2 tasks per GPU
def process(seed: int) -> dict:
    # ~2% of large tasks may OOM - acceptable tradeoff
    ...
```

### 3. Lock files prevent duplicate work

```python
def process(seed: int) -> dict:
    output_path = Path(f"/data/result_{seed}.parquet")
    lock_path = output_path.with_suffix(".lock")

    # Skip if done or in-progress
    if output_path.exists() or lock_path.exists():
        return {"status": "skipped"}

    # Claim immediately
    lock_path.touch()

    try:
        # ... GPU work ...
        write_result(output_path, result)
    finally:
        lock_path.unlink(missing_ok=True)
```

See [references/concurrency.md](references/concurrency.md) for parallel patterns.

## Cost Estimation

```
Per GPU-minute:
  T4:   $0.0098     A100-80GB: $0.0417
  L4:   $0.0133     H100:      $0.0658
  A10:  $0.0183     H200:      $0.0757
  L40S: $0.0325     B200:      $0.1042

Formula: GPUs × minutes × rate

Example: 5 H200s × 10 min = 5 × 10 × $0.0757 = $3.79
```

**Billing**: Web only at https://modal.com/settings → Usage and Billing

Free tier: $30/month for new accounts (with payment method).

## Critical Warnings

### Ctrl+C does NOT stop Modal jobs!

Killing your local shell only stops the local process. Cloud jobs keep running and billing.

```bash
# Check what's still running
modal app list

# Stop a specific app
modal app stop ap-XXXXXXXXXXXX

# Nuclear: stop all ephemeral apps
modal app list | grep ephemeral | awk '{print $1}' | xargs -I {} modal app stop {}
```

### Volume commits are expensive

Per-shard `volume.commit()` adds ~1 second overhead each. Let Modal batch commits automatically.

### High-end GPUs burn money fast

10 GPUs × premium tier × 6 minutes = $5-10. Start with A10 for development.

## Monitoring

```bash
# List running apps with task counts
modal app list

# List individual containers
modal container list

# GPU stats from container
modal container exec <container-id> -- nvidia-smi

# Compact GPU stats (utilization, VRAM, temp)
modal container exec <container-id> -- nvidia-smi \
    --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader
```

## Common Patterns

### Parallel batch processing

```python
@app.function(gpu="A10G", timeout=600)
def process_shard(seed: int, config: dict) -> dict:
    # Process one unit of work
    return {"seed": seed, "status": "done"}

@app.local_entrypoint()
def batch_process():
    tasks = [(seed, {"param": value}) for seed in range(1000)]
    results = list(process_shard.starmap(tasks))
    print(f"Completed {len(results)} tasks")
```

### Download from volume

```python
@app.local_entrypoint()
def download(output_dir: str = "./data"):
    import shutil
    volume.reload()
    for f in Path("/data").glob("*.parquet"):
        shutil.copy(f, Path(output_dir) / f.name)
```

### Find missing work

```python
@app.local_entrypoint()
def find_missing():
    volume.reload()
    existing = {int(f.stem.split("_")[1]) for f in Path("/data").glob("*.parquet")}
    expected = set(range(1000))
    missing = expected - existing
    print(f"Missing: {sorted(missing)}")
```

## Reference Files

- [references/gpu-selection.md](references/gpu-selection.md) - GPU types, pricing, availability, memory specs
- [references/volumes.md](references/volumes.md) - Persistent storage patterns, commits, concurrent access
- [references/concurrency.md](references/concurrency.md) - Parallel execution, starmap, concurrent inputs
- [references/troubleshooting.md](references/troubleshooting.md) - Common errors and solutions
