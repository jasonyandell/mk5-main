# Troubleshooting

## Critical Issues

### Jobs keep running after Ctrl+C

**Problem**: Killing local shell only stops local process. Cloud jobs continue running and billing.

**Solution**:
```bash
# Check what's running
modal app list

# Stop specific app
modal app stop ap-XXXXXXXXXXXX

# Stop all ephemeral apps
modal app list | grep ephemeral | awk '{print $1}' | xargs -I {} modal app stop {}
```

**Prevention**: Always check `modal app list` before ending session.

### Billing limit reached

**Error**: `App creation failed: workspace billing cycle spend limit reached`

**Solution**: Add payment method at https://modal.com/settings

**Prevention**: Monitor costs, start with A10 for development.

## Authentication

### Token missing

**Error**: `modal.exception.AuthError: Token not found`

**Solution**:
```bash
modal token new  # Opens browser for auth
```

Token saved to `~/.modal.toml`

### Token expired

**Error**: `modal.exception.AuthError: Token expired`

**Solution**:
```bash
modal token new  # Re-authenticate
```

## GPU Issues

### GPU not available / queued

**Error**: Job sits in queue waiting for GPU

**Causes**: H100 and B200 often have limited availability

**Solutions**:
1. Switch to more available GPU (A10G, A100-80GB, H200)
2. Launch same job on multiple GPU types in parallel
3. Wait for availability

```python
# Option 1: Use more available GPU
@app.function(gpu="A10G")  # Instead of H100

# Option 2: Multi-GPU fallback (if supported)
gpu = modal.gpu.A100(count=1).or_else(modal.gpu.A10G(count=1))
```

### GPU OOM (Out of Memory)

**Error**: `CUDA out of memory` or `RuntimeError: CUDA error`

**Solutions**:
1. Reduce batch size
2. Use larger GPU (A100-80GB, H200)
3. Reduce `@modal.concurrent(max_inputs=N)`
4. Skip large inputs

```python
@app.function(gpu="A10G")
def process(seed: int, max_states: int = 500_000_000) -> dict:
    state_count = estimate_states(seed)
    if state_count > max_states:
        return {"status": "skipped", "reason": "too large"}
    # ... process ...
```

### Arrow Memory Pool OOM

**Error**: `/arrow/cpp/src/arrow/memory_pool.cc:683: Internal error: cannot create default memory pool`

**Cause**: Parquet writing with 2x concurrency on large data

**Solutions**:
1. Accept ~2% failure rate, retry individually
2. Reduce to 1x concurrency
3. Use larger GPU for problematic inputs

## Timeout Issues

### Container timeout

**Error**: `TimeoutError: Function exceeded maximum runtime`

**Solution**:
```python
@app.function(
    gpu="A10G",
    timeout=600,  # Increase from default 300s
)
def long_running():
    ...
```

Maximum timeout: 86400 seconds (24 hours)

### Orchestrator timeout

For batch jobs, give orchestrator much longer timeout:

```python
@app.function(timeout=86400)  # 24 hours for orchestrator
def batch_orchestrator():
    # This fans out to many GPU tasks
    results = list(gpu_task.starmap(tasks))
```

## Volume Issues

### Volume busy on reload

**Error**: `VolumeError: volume busy`

**Cause**: Calling `volume.reload()` with open file handles

**Solution**:
```python
# Close files before reload
f.close()
volume.reload()

# Or use context managers
with open(path) as f:
    data = f.read()
# File closed here
volume.reload()  # Now safe
```

### Data not visible across containers

**Cause**: Missing commit or reload

**Solution**:
```python
# Writer
write_file(path, data)
volume.commit()  # Make visible to others

# Reader
volume.reload()  # Fetch latest
data = read_file(path)
```

### Per-shard commit slowdown

**Symptom**: 60 shards/min instead of 100+

**Cause**: Calling `volume.commit()` after every write

**Solution**: Remove per-write commits, let Modal auto-commit

```python
# Remove this
volume.commit()  # ~1 second per call!
```

## Import Issues

### Import not found in cloud

**Error**: `ModuleNotFoundError: No module named 'xyz'`

**Cause**: Dependency not in Modal Image

**Solution**:
```python
image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy",
    "xyz",  # Add missing package
)
```

### Import works locally but fails on Modal

**Cause**: Local imports executed before container built

**Solution**: Move imports inside function

```python
@app.function(image=image)
def process():
    import torch  # Import inside function
    import numpy as np
    # ...
```

## Debugging

### View logs

```bash
# Stream logs for deployed app
modal app logs my-app

# View container logs
modal container logs <container-id>
```

### Interactive shell

```bash
# Shell into running container
modal container exec <container-id> -- /bin/bash

# Run one-off command
modal container exec <container-id> -- nvidia-smi
```

### Local testing

```python
# Test function locally (no cloud)
result = my_function.local(arg1, arg2)

# Vs remote (runs on Modal)
result = my_function.remote(arg1, arg2)
```

## Performance Issues

### Slow cold starts

**Cause**: Large image, many dependencies

**Solutions**:
1. Use slim base image
2. Cache model weights in Volume
3. Use `keep_warm` containers

```python
@app.function(
    buffer_containers=1,  # Keep 1 warm
)
def inference():
    ...
```

### Low GPU utilization

**Symptoms**: nvidia-smi shows 0% GPU, high idle time

**Causes**:
- I/O bound (waiting on disk/network)
- Between starmap tasks
- CPU-bound preprocessing

**Solutions**:
1. Add `@modal.concurrent(max_inputs=2)` to overlap I/O
2. Batch more work per task
3. Move preprocessing to separate CPU function

### Slow volume writes

**Cause**: Many small files, frequent commits

**Solutions**:
1. Batch writes into larger files
2. Remove per-write commits
3. Use V2 volumes for many files
