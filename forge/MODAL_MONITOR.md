# Modal Job Monitor

## Quick Status

```bash
# What's running?
modal app list

# How many GPUs?
modal container list --json | jq length

# Container IDs (for exec commands)
modal container list --json | jq -r '.[]["Container ID"]'
```

## Training Sweep Monitoring

### Live Progress

```bash
# Watch training output (replace JOB_ID with background task ID)
tail -f /tmp/claude/tasks/JOB_ID.output

# Filter for epoch completions
tail -100 /tmp/claude/tasks/JOB_ID.output | grep "Epoch"

# Watch for all runs
watch -n 10 'tail -50 /tmp/claude/tasks/JOB_ID.output | grep -E "(Epoch|Run |val_acc)"'
```

### GPU Utilization Check

```bash
# Get full container ID first
CONTAINER=$(modal container list --json | jq -r '.[0]["Container ID"]')

# GPU stats (utilization, memory, temp)
modal container exec $CONTAINER -- nvidia-smi \
    --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv

# Expected healthy values for training:
#   GPU%: 80-100% (low = batch size too small)
#   Memory: 20-80% of total
#   Temp: <80°C
```

### wandb Dashboard

```bash
# Training runs logged to wandb
# View at: https://wandb.ai/<username>/<project>

# Common filters:
#   Group by: stage1-sweep (or your group name)
#   Sort by: val/accuracy, val/q_gap
#   Compare: overlay loss curves
```

### Performance Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| GPU% < 50% | Batch size too small | Increase to 4096-8192 |
| GPU% ~0% | Data loading bottleneck | Load data in @modal.enter() |
| Memory > 90% | Batch too large | Reduce batch size or use AMP |
| OOM errors | Model + batch > VRAM | Enable AMP, reduce batch |

### Key Training Metrics

```bash
# Parse metrics from output
grep "Epoch" /tmp/claude/tasks/JOB_ID.output | \
    awk -F'[=,]' '{print $1, "loss="$3, "acc="$5, "qgap="$7}'
```

## Detailed GPU Monitoring

```bash
# Get GPU stats from a single container
modal container exec <container-id> -- nvidia-smi

# Get compact stats (GPU%, VRAM, temp) from one container
modal container exec <container-id> -- nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

# Get stats from ALL containers (run scratch/gpu-stats.sh)
bash scratch/gpu-stats.sh
```

### GPU Stats Script (scratch/gpu-stats.sh)

```bash
#!/bin/bash
# Get GPU stats from all Modal containers

echo "Container | GPU% | VRAM Used | VRAM Total | Temp"
echo "----------|------|-----------|------------|-----"

modal container list --json 2>/dev/null | jq -r '.[]["Container ID"]' | while read cid; do
  stats=$(modal container exec "$cid" -- nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
  short_id="${cid:0:12}"
  echo "$short_id | $stats"
done
```

### Interpreting GPU Stats

| Metric | Healthy Range | Notes |
|--------|---------------|-------|
| GPU% | 50-100% | 0% often means I/O bound or between tasks |
| VRAM | 60-90% of total | With 2x concurrency, expect 16-22GB on A10 |
| Temp | <80°C | A10s run cool, 30-55°C typical |

**Why GPU% shows 0% on some containers:**
- Between starmap tasks (scheduling gap)
- Writing shards to volume (I/O bound)
- In enumeration phase (CPU-bound before GPU solve)

## Progress Logging

The logger script `scratch/log-progress.sh` tracks shard generation across jobs.

### Updating the Logger for New Jobs

When you start new Modal jobs, update the script with the correct output file paths:

```bash
# Edit scratch/log-progress.sh
# Find your job output files in /tmp/claude/tasks/
# Update the grep lines to point to the correct .output files:

train=$(grep -c 'Root value:' /tmp/claude/tasks/<train-job-id>.output 2>/dev/null | head -1 || echo 0)
valtest=$(grep -c 'Root value:' /tmp/claude/tasks/<valtest-job-id>.output 2>/dev/null | head -1 || echo 0)
```

### Current Logger Script

```bash
#!/bin/bash
# scratch/log-progress.sh - Log Modal generation progress

LOG=scratch/modal-progress.csv
ts=$(date +%H:%M:%S)

# Count shards from job output (update path for new jobs!)
# Find the job output file in /tmp/claude/tasks/*.output
shards=$(grep -c 'Root value:' /tmp/claude/tasks/YOUR_JOB_ID.output 2>/dev/null | head -1 || echo 0)
shards=${shards:-0}

# Get container count
containers=$(modal container list 2>&1 | grep -c "ta-01" || echo 0)
containers=${containers:-0}

echo "$ts,shards=$shards,gpus=$containers" >> "$LOG"
echo "$ts | Shards: $shards | GPUs: $containers"
```

**To use**: Update `YOUR_JOB_ID` with the actual background task ID from Claude's output.

### Run Every Minute in Background

```bash
# Start logging
while true; do bash scratch/log-progress.sh; sleep 60; done &

# Stop logging (kill all log-progress.sh processes)
pkill -f "log-progress.sh"
```

## Check Progress

```bash
# View log
cat scratch/modal-progress.csv

# Latest entry
tail -1 scratch/modal-progress.csv

# Watch live
watch -n 60 'bash scratch/log-progress.sh && tail -5 scratch/modal-progress.csv'
```

## Stop All Jobs (Emergency)

```bash
# List running
modal app list | grep ephemeral

# Stop specific app
modal app stop ap-XXXXXXXXXXXX

# Nuclear: stop all ephemeral apps
for app in $(modal app list 2>&1 | grep ephemeral | awk '{print $1}'); do
  modal app stop "$app"
done
```

## Current Jobs

Generation complete! 10,000 shards on Modal volume.

**To run new jobs**:
```bash
# Full generation (9000 train + 1000 val/test)
modal run forge/modal_app.py::generate_range --start-seed 0 --end-seed 900 --n-opp-seeds 10
modal run forge/modal_app.py::generate_valtest

# Single shard (for retrying failed seeds)
modal run forge/modal_app.py::main --base-seed 922 --opp-seed 1

# Find missing shards
modal run forge/modal_app.py::find_missing

# Count shards on volume
modal run forge/modal_app.py::count_shards
```

**Current settings** (in modal_app.py):
- `max_states=500M` - A10 can handle up to ~500M states
- `@modal.concurrent(max_inputs=2)` - 2 tasks per GPU
- Lock files prevent duplicate work across parallel jobs
- No per-shard commit (auto-commits on container shutdown)

## Cost Tracking

```bash
# A10 rate: $0.0183/GPU-minute
# Formula: GPUs × minutes × 0.0183

# Example: 5 GPUs for 60 min = $5.49
echo "5 * 60 * 0.0183" | bc
```

## When It's Done

```bash
# Download shards from Modal volume
modal run forge/modal_app.py::download --output-dir data/shards-marginalized

# Verify counts
ls data/shards-marginalized/train/*.parquet | wc -l
ls data/shards-marginalized/val/*.parquet | wc -l
ls data/shards-marginalized/test/*.parquet | wc -l
```
