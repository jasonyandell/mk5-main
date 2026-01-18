---
name: modal
description: Modal serverless GPU cloud - activates for GPU workloads, ML inference, batch processing, web endpoints, scheduled jobs, or cloud compute tasks. Use when deploying functions to cloud GPUs, serving inference APIs, running parallel starmap jobs, managing Modal volumes, or estimating cloud costs. Covers authentication, GPU selection, concurrency, volumes, secrets, web endpoints, scheduled jobs, and common pitfalls.
---

# Modal GPU Infrastructure

Serverless cloud platform for running Python on GPUs. Pay per second, scale to thousands of containers, zero infrastructure management.

## Quick Start

```bash
# First time setup
pip install modal
modal token new        # Opens browser for auth

# Development (hot-reload)
modal serve app.py     # Web endpoints with auto-reload

# Run batch job
modal run app.py::main

# Deploy to production
modal deploy app.py

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

## Classes (Stateful Services)

Load model once, reuse across requests:

```python
@app.cls(gpu="A10G", image=image)
class Inference:
    @modal.enter()  # Runs once on container start
    def load_model(self):
        import torch
        self.model = torch.load("/data/model.pt")
        self.model.eval()

    @modal.method()
    def predict(self, x: list[float]) -> list[float]:
        return self.model(x).tolist()

    @modal.exit()  # Cleanup on container shutdown
    def cleanup(self):
        del self.model

# Usage
inference = Inference()
result = inference.predict.remote([1.0, 2.0, 3.0])
```

## Web Endpoints

Deploy inference APIs with automatic HTTPS:

```python
@app.function(gpu="A10G", image=image)
@modal.web_endpoint(method="POST")
def predict(request: dict) -> dict:
    result = model.predict(request["input"])
    return {"prediction": result}

# Returns URL like: https://your-app--predict.modal.run
```

For FastAPI apps:

```python
from fastapi import FastAPI
web_app = FastAPI()

@web_app.post("/predict")
def predict(request: dict):
    return {"result": model(request["input"])}

@app.function(gpu="A10G", image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
```

See [references/web-endpoints.md](references/web-endpoints.md) for deployment patterns.

## Scheduled Jobs

```python
# Run every hour
@app.function(schedule=modal.Period(hours=1))
def hourly_sync():
    sync_data()

# Cron syntax (9am UTC daily)
@app.function(schedule=modal.Cron("0 9 * * *"))
def daily_report():
    generate_report()
```

## Secrets

```python
# Create secret via CLI
# modal secret create my-api-key API_KEY=sk-xxx

@app.function(secrets=[modal.Secret.from_name("my-api-key")])
def call_api():
    import os
    key = os.environ["API_KEY"]  # Injected from secret
    return requests.get(url, headers={"Authorization": f"Bearer {key}"})
```

## Image Building

```python
# Basic pip install
image = modal.Image.debian_slim().pip_install("torch", "numpy")

# System packages + pip
image = (modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .pip_install("torch", "transformers"))

# From requirements.txt
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

# Run shell commands
image = (modal.Image.debian_slim()
    .run_commands("git clone https://github.com/...", "cd repo && pip install -e ."))

# From Dockerfile
image = modal.Image.from_dockerfile("./Dockerfile")

# Include local code
image = modal.Image.debian_slim().add_local_dir("./src", "/app/src")
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

### Training Performance (GPU Saturation)

For ML training, low GPU utilization usually means data loading bottleneck:

```python
@app.cls(
    gpu="T4",
    memory=32768,
    min_containers=9,  # Pre-warm all containers for sweep
)
class Trainer:
    @modal.enter()
    def load_data(self):
        # Load data ONCE into RAM - critical for GPU saturation
        self.train_data = torch.load("/data/train.pt")

    @modal.method()
    def train(self, config: dict) -> dict:
        # Use large batch size + AMP for GPU saturation
        train_loader = DataLoader(
            self.train_data,
            batch_size=8192,  # 16x default, saturates T4
            num_workers=8,
            pin_memory=True,
        )

        scaler = torch.amp.GradScaler('cuda')
        for batch in train_loader:
            with torch.amp.autocast('cuda'):
                loss = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

**Key settings for high GPU utilization:**

| Setting | Low GPU% Fix | Example |
|---------|--------------|---------|
| Batch size | Increase 8-16x | 512 → 4096-8192 |
| AMP | Enable float16 | `torch.amp.autocast('cuda')` |
| num_workers | Match CPU cores | 8 for Modal containers |
| Data location | RAM not disk | Load in `@modal.enter()` |

**Measured improvement**: 49% → 100% GPU utilization with batch_size=8192 + AMP on T4.

### True Parallelism with `.map()` (Critical!)

**Problem**: Using `@app.cls` with multiple instances serializes execution:

```python
# BAD - runs sequentially despite multiple instances!
trainers = [Trainer() for _ in range(9)]
for i, cfg in enumerate(configs):
    futures.append(trainers[i].train.remote(cfg))
results = [f.get() for f in futures]  # Waits sequentially
```

**Solution**: Use `@app.function` with `.map()` for true parallelism:

```python
# GOOD - runs all 9 in parallel!
@app.function(gpu="T4", ...)
def train_config(config: dict) -> dict:
    # Load data, train, return results
    ...

@app.local_entrypoint()
def main():
    configs = [{"run_id": 1, ...}, {"run_id": 2, ...}, ...]
    results = list(train_config.map(configs))  # TRUE PARALLELISM
```

**Why it matters**: With class instances, Modal may route all `.remote()` calls to the same container. With `.map()`, Modal spawns separate containers for each input.

**Tradeoff**: `.map()` doesn't support per-input cancellation. You can't selectively kill run 3 of 9 - it's all or nothing. For early stopping, build it into your function logic.

### Batch Processing Optimizations

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

### Worker preemption loses progress

Modal uses spot-like instances. Workers can be preempted mid-training - you pay for time used but lose progress.

```
Runner interrupted due to worker preemption. Your Function will be restarted with the same input.
```

**Solution**: Add checkpoint recovery:

```python
@app.function(gpu="T4", volumes={"/data": volume})
def train(config: dict):
    resume_path = Path(f"/data/checkpoints/run{config['id']}_resume.pt")
    start_epoch = 0

    # Resume from checkpoint if preempted
    if resume_path.exists():
        ckpt = torch.load(resume_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        train_epoch(...)

        # Save resume checkpoint after each epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }, resume_path)

    # Clean up on successful completion
    resume_path.unlink(missing_ok=True)
```

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
- [references/web-endpoints.md](references/web-endpoints.md) - Web APIs, FastAPI, deployment, scaling
- [references/troubleshooting.md](references/troubleshooting.md) - Common errors and solutions
