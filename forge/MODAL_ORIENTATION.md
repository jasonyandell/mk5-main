# Modal GPU Infrastructure - Orientation

Guide to running GPU workloads on [Modal](https://modal.com/) based on our oracle shard generation experience.

## Quick Start

```bash
# First time setup
pip install modal
modal token new        # Opens browser for auth, saves to ~/.modal.toml

# Run a job
modal run forge/modal_app.py::generate_valtest

# Check status
modal app list         # Shows running apps with task counts
modal container list   # Shows individual GPU containers
```

## GPU Pricing (as of Jan 2025)

| GPU | $/hour | $/second | Speed vs A10 | Cost Efficiency |
|-----|--------|----------|--------------|-----------------|
| **B200** | $6.25 | $0.001736 | ~6x | Premium speed |
| **H200** | $4.54 | $0.001261 | ~5x | Best value for speed |
| **H100** | $3.95 | $0.001097 | ~4x | Good balance |
| A100-80GB | $2.50 | $0.000694 | ~2.5x | Solid workhorse |
| A100-40GB | $2.10 | $0.000583 | ~2x | Budget A100 |
| L40S | $1.95 | $0.000542 | ~1.8x | Good availability |
| **A10** | $1.10 | $0.000306 | 1x (baseline) | Most cost-efficient |
| L4 | $0.80 | $0.000222 | ~0.8x | Budget option |
| T4 | $0.59 | $0.000164 | ~0.5x | Cheapest |

**Key insight**: Faster GPUs have similar or better cost-per-shard because speed gains offset price. H200 at 5x speed for 4x price = better value than A10 for wall-clock time.

### What $5 Gets You

| GPU | Shards | Wall-clock |
|-----|--------|------------|
| A10 | ~10,000 | ~4 hours |
| H200 | ~10,000 | ~1 hour |
| H100 | ~10,000 | ~1.2 hours |

## Checking Usage & Billing

**No CLI command** - billing is web-only:
1. Go to https://modal.com/settings
2. Navigate to "Usage and Billing"
3. Click "Manage payment information" for Stripe invoices

Free tier: $30/month credits for new accounts.

### Estimating Costs

Calculate GPU-seconds from your run:

```bash
# Get GPU count and duration
modal container list  # Count active containers
# Multiply: GPUs × minutes × ($/hour ÷ 60)
```

**Quick cost formulas:**
```
A10:        GPUs × minutes × $0.0183/min
A100-80GB:  GPUs × minutes × $0.0417/min
H100:       GPUs × minutes × $0.0658/min
H200:       GPUs × minutes × $0.0757/min
B200:       GPUs × minutes × $0.1042/min
```

**Example**: 5 H200s for 10 min = 5 × 10 × $0.0757 = **$3.79**

### Lesson Learned

Running 10 high-end GPUs in parallel burns money fast:
- 5 H200 + 3 B200 + 2 A100 for 6 min = ~$5
- Same work on 10 A10s for 20 min = ~$3.67

**For budget runs**: Stick to A10, accept longer wall-clock time.

## CLI Commands Reference

### Check Running Jobs

```bash
# List apps with task counts
modal app list

# Output:
# ┃ App ID                    ┃ State     ┃ Tasks ┃
# │ ap-OHuBfg6yIAyk9tbHFyHhoB │ ephemeral │ 5     │

# List individual GPU containers
modal container list

# Output shows each GPU instance with App ID, start time
```

### Monitor Progress

```bash
# Watch container count
watch -n 5 'modal container list | grep -c "ta-01"'

# Count containers by app
modal container list 2>&1 | grep -c "ap-YOUR_APP_ID"
```

## Performance Optimizations

### 1. Remove Per-Shard Volume Commits (~90% speedup)

```python
# BAD - commits after every shard, huge overhead
write_result(output_path, ...)
shard_volume.commit()  # ~1 second per shard!

# GOOD - let Modal batch commits automatically
write_result(output_path, ...)
# Volume auto-commits periodically and on container shutdown
```

**Impact**: 60 shards/min → 100+ shards/min

### 2. Use Concurrent Inputs (2x throughput)

```python
@app.function(gpu="A10G", ...)
@modal.concurrent(max_inputs=2)  # Run 2 tasks per GPU
def generate_shard(...):
```

**Tradeoff**: ~2% of large seeds (>200M states) may OOM with Arrow memory pool errors. Acceptable if you can retry those individually.

**Impact**: 100 shards/min → 190+ shards/min

### 3. Lock Files to Prevent Duplicate Work

When running multiple jobs in parallel, prevent race conditions:

```python
lock_path = output_path.with_suffix(".lock")

# Check both parquet AND lock file
if output_path.exists() or lock_path.exists():
    return {"status": "skipped"}

# Claim immediately before GPU work
lock_path.touch()

# ... do GPU work ...

write_result(output_path, ...)
lock_path.unlink(missing_ok=True)  # Clean up lock
```

### Combined Performance (A10 baseline)

| Config | Rate | vs Baseline |
|--------|------|-------------|
| A10, 1x, per-shard commit | 60/min | 1x |
| A10, 1x, no commit | 100/min | 1.7x |
| A10, 2x, no commit | 190/min | 3.2x |
| A10, 2x, no commit + locks | 225/min | **3.8x** |

## Our Experience: What Works

### GPU Availability (Best to Worst)

1. **A10** - Always available, good for background jobs
2. **A100-80GB** - Usually available, 2.5x faster
3. **H200** - Often available, excellent speed
4. **B200** - Sometimes queued, newest hardware
5. **H100** - Often queued (high demand)

### Strategy: Fire Multiple GPU Types

When one GPU type is queued, launch jobs on multiple types:

```python
# In modal_app.py - easy to switch
GPU_B200 = "B200"
GPU_H200 = "H200"
GPU_H100 = "H100"
GPU_A100_80GB = "A100-80GB"
GPU_A10G = "A10G"

@app.function(gpu=GPU_H200, ...)  # Change this line to switch
def generate_shard(...):
```

Launch same job with different GPU configs - duplicates auto-skip via file existence check:

```bash
# Fire all at once, first available wins
modal run forge/modal_app.py::generate_range &  # Uses whatever GPU is configured
# Edit modal_app.py to use different GPU
modal run forge/modal_app.py::generate_range &  # Second GPU type
```

### Parallel Execution with starmap()

```python
@app.function(gpu="H200", timeout=300)
def generate_shard(base_seed: int, opp_seed: int, decl_id: int) -> dict:
    # GPU work here
    return {"status": "success", ...}

@app.function(timeout=86400)  # Orchestrator on CPU
def generate_range(start_seed: int, end_seed: int, n_opp_seeds: int):
    tasks = []
    for seed in range(start_seed, end_seed):
        for opp in range(n_opp_seeds):
            tasks.append((seed, opp, seed % 10))

    # Fan out to GPU pool - Modal handles parallelism
    results = list(generate_shard.starmap(tasks))
    return summarize(results)
```

## Common Errors & Solutions

### Killing local shell doesn't stop Modal jobs!

**Critical**: `Ctrl+C` or killing your terminal only stops the *local* process. The Modal jobs keep running in the cloud and burning money!

```bash
# See what's still running
modal app list

# Stop a specific app
modal app stop ap-XXXXXXXXXXXX

# Stop all running apps (nuclear option)
modal app list | grep ephemeral | awk '{print $1}' | xargs -I {} modal app stop {}
```

### "Token missing" on first run
```bash
modal token new  # Authenticate via browser
```

### Job queued (no GPUs available)
- High-end GPUs (H100, B200) often queued
- Switch to A100-80GB or A10 for availability
- Or launch multiple GPU types in parallel

### Container timeout
```python
@app.function(
    gpu="H200",
    timeout=600,  # Increase from default 300s
)
```

### OOM on large state counts
```python
def generate_shard(..., max_states: int = 500_000_000):
    if state_count > max_states:
        return {"status": "skipped", "reason": "too large"}
```

A10 (24GB) can handle ~500M states. For larger seeds, use A100-80GB.

### Arrow Memory Pool OOM (with 2x concurrency)

With `@modal.concurrent(max_inputs=2)`, large seeds can exhaust memory when writing parquet:
```
/arrow/cpp/src/arrow/memory_pool.cc:683: Internal error: cannot create default memory pool
```

**Solutions**:
1. Accept ~2% failure rate, retry large seeds individually
2. Reduce to 1x concurrency (slower but safer)
3. Use larger GPU for problematic seeds

### Volume commits

```python
# Per-shard commit (SLOW - adds ~1s per shard)
shard_volume.commit()

# Better: let Modal auto-commit
# Volume commits periodically and on container shutdown
# Data is safe even without explicit commits
```

**When to use explicit commit**: Only if you need immediate durability and can't tolerate any data loss on crash.

## Volume Management

```python
# Define persistent volume
shard_volume = modal.Volume.from_name("texas-42-shards", create_if_missing=True)

@app.function(volumes={"/shards": shard_volume})
def generate_shard(...):
    # Write to /shards/...
    output_path.write_bytes(data)
    shard_volume.commit()  # Important!
```

```bash
# List volume contents (via a Modal function)
modal run forge/modal_app.py::list_shards --split val
```

## Downloading Results

```bash
# Download from Modal volume to local
modal run forge/modal_app.py::download --split val --output-dir data/shards-marginalized
modal run forge/modal_app.py::download --split test --output-dir data/shards-marginalized
```

## Progress Monitoring Script

We built a logger to track GPU utilization:

```bash
#!/bin/bash
# scratch/log-progress.sh
ts=$(date +%H:%M:%S)
h200=$(grep -c 'Root value:' /tmp/claude/tasks/h200.output 2>/dev/null || echo 0)
containers=$(modal container list 2>&1 | grep -c "ta-01")
echo "$ts,$h200,$containers" >> scratch/modal-progress.csv
```

Run every minute:
```bash
while true; do bash scratch/log-progress.sh; sleep 60; done &
```

## Real Performance Data

### Full 10,000 Shard Run (Jan 2025)

**Final dataset**: 10,000 shards (~1.5-2 TB)
- Train: 9,000 shards (seeds 0-899 × 10 opp_seeds)
- Val: 500 shards (seeds 900-949 × 10 opp_seeds)
- Test: 500 shards (seeds 950-999 × 10 opp_seeds)

**Timeline**:
| Time | Event | Config |
|------|-------|--------|
| 21:00 | Started | H200+B200+A100 mix |
| 21:38 | Budget exceeded, switched | A10 only |
| 22:00 | Added optimizations | A10 + 2x concurrency + no commit |
| 22:22 | Arrow OOM crash | Restarted |
| 22:39 | Added lock files | Final config |
| 23:05 | Train complete | 9,000 shards |
| 01:03 | All complete | 10,000 shards |

**Throughput by config**:
- Mixed high-end GPUs: ~300 shards/min (expensive!)
- A10 baseline: ~60 shards/min
- A10 optimized (2x + no commit + locks): ~225 shards/min

**Cost**: ~$25-35 total (used free tier + small overage)

**GPU Usage Breakdown**:
| GPU | GPU-Minutes | Est. Cost |
|-----|-------------|-----------|
| H200 | 45 | $3.41 |
| B200 | 27 | $2.81 |
| A100 | 18 | $0.75 |
| A10 | ~950 | $17.39 |
| **Total** | **~1,040** | **~$24-25** |

~17 hours of GPU time total, 91% on A10s.

## Architecture Pattern

```
┌─────────────────────────────────────────────────────────┐
│  Local: modal run forge/modal_app.py::generate_valtest  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Modal Cloud: generate_range() on CPU                   │
│  - Builds task list: [(seed, opp, decl), ...]          │
│  - Calls generate_shard.starmap(tasks)                  │
└─────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      ┌──────────┐    ┌──────────┐    ┌──────────┐
      │ H200 #1  │    │ H200 #2  │    │ H200 #N  │
      │ GPU Task │    │ GPU Task │    │ GPU Task │
      └────┬─────┘    └────┬─────┘    └────┬─────┘
           │               │               │
           └───────────────┴───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Modal Volume: /shards/     │
              │  Persistent across runs     │
              └─────────────────────────────┘
```

## Training Sweeps (Parallel Hyperparameter Search)

### Key Pattern: Use `.map()` for True Parallelism

```python
# Define training function (NOT a class method)
@app.function(gpu="T4", memory=32768, timeout=3600, volumes={"/data": volume})
def train_config(config: dict) -> dict:
    data = load_data("/data")  # Each container loads its own
    model = train(data, config)
    return {"config": config, "metrics": model.best_metrics}

@app.local_entrypoint()
def sweep():
    configs = [
        {"run_id": 1, "lr": 1e-4, "batch_size": 8192},
        {"run_id": 2, "lr": 3e-4, "batch_size": 8192},
        # ... 9 configs total
    ]
    results = list(train_config.map(configs))  # 9 parallel T4s!
```

### GPU Saturation Settings (T4)

| Setting | Value | Impact |
|---------|-------|--------|
| batch_size | 8192 | Saturates GPU compute |
| AMP | `torch.amp.autocast('cuda')` | 2x throughput |
| num_workers | 8 | Feeds data fast enough |

**Result**: 49% → 100% GPU utilization

### Common Pitfall: Class Instances Serialize!

```python
# BAD - runs sequentially despite 9 instances
trainers = [Trainer() for _ in range(9)]
futures = [t.train.remote(cfg) for t, cfg in zip(trainers, configs)]

# GOOD - runs in true parallel
results = list(train_config.map(configs))
```

## Links

- [Modal Docs](https://modal.com/docs)
- [GPU Pricing](https://modal.com/pricing)
- [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)
- [Billing](https://modal.com/docs/guide/billing)
