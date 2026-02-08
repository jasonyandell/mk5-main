# Vast.ai Worker Runbook

## Quick Reference

```bash
export HF_TOKEN=hf_...              # set before any vast_up

# Autonomous (recommended) — one command, self-healing
./vast_monitor.sh 4                 # maintain 4 workers, Ctrl-C to stop

# Manual workflow
./vast_up.sh 4                      # launch 4 workers
./vast_status.sh                    # see what's running
vastai logs INSTANCE_ID --tail 50   # check worker output
./vast_down.sh                      # destroy everything

# Multi-model fleet (new model alongside existing)
ZEB_WEIGHTS_NAME=large ZEB_FLEET=zeb-large ./vast_up.sh 4
./vast_status.sh zeb-large          # only zeb-large-* instances
./vast_down.sh zeb-large            # tear down just the large fleet
```

---

## 1. Launch Workers

```bash
export HF_TOKEN=hf_...    # fine-grained token with write to zeb-42 + zeb-42-examples
cd forge/zeb/vast
./vast_up.sh 4
```

**Expected output:**
```
Searching for 4 cheapest offers...
Launching 4 workers...
  Worker 0: RTX_3070 @ $0.102/hr (offer 12345678)
    started. {'success': True, 'new_contract': 87654321}
  ...
Fleet launched. Workers will take 3-5 min to start generating.
```

**If "No matching offers found":**
- Market might be empty for your filters. Try a specific GPU:
  ```bash
  ./vast_up.sh 2 RTX_3090
  ```
- Or check what's available manually:
  ```bash
  vastai search offers 'gpu_name in [RTX_3060,RTX_3070,RTX_3080,RTX_3090] rentable=true' -o dph_total --limit 10
  ```

---

## 2. Check Instance Status

```bash
./vast_status.sh
```

**Expected output:**
```
Label                  GPU              Status       $/hr    ID  SSH
------------------------------------------------------------------------------------------
zeb-worker-vast-0      RTX_3070         running     0.102   87654321  ssh -p 12345 root@ssh5.vast.ai
zeb-worker-vast-1      RTX_3060_Ti      running     0.089   87654322  ssh -p 12346 root@ssh3.vast.ai
...
Total: 4 instances, $0.382/hr ($9.17/day)
```

**Status meanings:**
| Status | Meaning | Action |
|--------|---------|--------|
| `loading` | Docker image pulling | Wait 1-3 min |
| `running` | Container is up | Check logs |
| `exited` | Container crashed | Check logs, destroy & recreate |
| `stopped` | Preempted or paused | Destroy & recreate |
| (missing) | Instance gone | Recreate with `./vast_up.sh 1` |

---

## 3. View Logs

### Quick check (most common)
```bash
# Get instance IDs first
# NOTE: pipe vastai --raw to a file, not directly to python (broken pipe bug)
vastai show instances --raw > /tmp/vast.json 2>/dev/null
python3 -c "
import json
for i in json.load(open('/tmp/vast.json')):
    if (i.get('label') or '').startswith('zeb-'):
        print(f\"  {i['id']}  {i.get('label', '?'):<24} {i.get('actual_status', '?')}\")
"

# Then check a specific instance
vastai logs 87654321 --tail 30
```

### What healthy startup looks like
```
=== Setup started at Thu Feb  6 12:00:00 UTC 2026 ===
git installed
repo cloned
deps installed
GPU OK: NVIDIA GeForce RTX 3070
=== Starting worker worker-vast-0 at Thu Feb  6 12:02:30 UTC 2026 ===
Pulling initial weights from jasonyandell/zeb-42...
  Model loaded (step 142), 557,186 params
  Examples repo: jasonyandell/zeb-42-examples
Creating self-play pipeline...
Pipeline created in 8.3s

Worker worker-vast-0 starting (output: jasonyandell/zeb-42-examples)
[worker-vast-0] batch 1: 7168 examples, 1.4 games/s, step=142, total_games=256
[worker-vast-0] batch 2: 7168 examples, 1.5 games/s, step=142, total_games=512
```

### What problems look like

**No logs at all (blank output):**
- Instance still loading. Wait 2-3 min and check again.
- If still blank after 5 min: the onstart script may have failed before producing output.
  ```bash
  # SSH in to investigate
  ssh -p PORT root@HOST
  cat /proc/1/cmdline | tr '\0' ' '   # what's the main process?
  nvidia-smi                            # GPU working?
  ```

**Logs stop after "git installed" or "repo cloned":**
- Git clone or pip install failed. SSH in:
  ```bash
  ssh -p PORT root@HOST
  ls /root/code/          # did clone work?
  pip list | grep hugging  # did install work?
  ```

**"No CUDA" error:**
- Driver mismatch. Destroy this instance and create a new one — it's a bad host.
  ```bash
  vastai destroy instance 87654321
  ./vast_up.sh 1    # get a replacement
  ```

**"CUDA out of memory":**
- GPU doesn't have enough VRAM. Destroy and filter for more memory:
  ```bash
  vastai destroy instance 87654321
  ./vast_up.sh 1 RTX_3080   # 10GB VRAM instead of 8GB
  ```

**Worker running but no examples appearing on HF:**
- Check for HF auth issues:
  ```bash
  vastai logs 87654321 --tail 50   # look for 401/403 errors
  ```
- The HF_TOKEN might be wrong or lack write permissions.

**"Connection refused" on SSH:**
- Instance is still in `loading` state. Wait for `running`.
- Or the instance has no SSH port mapped. Check `./vast_status.sh` for the SSH column.

**HF "412 Precondition Failed" or "httpx.ReadTimeout":**
- Concurrent commit conflict or network timeout during folder upload.
- Workers deployed after commit `f37a0b6` handle this automatically (retry next cycle).
- If running older code, the worker process crashes. Destroy and replace — the
  replacement will clone fresh code with the fix.

---

## 4. SSH Into an Instance

```bash
# Get SSH command from status
./vast_status.sh
# Copy the ssh command from the SSH column, e.g.:
ssh -p 12345 root@ssh5.vast.ai
```

**Once inside, useful commands:**
```bash
nvidia-smi                          # GPU status + memory usage
ps aux | grep worker                # is the worker process alive?
df -h                               # disk space
pip list | grep -E "torch|hugging"  # installed packages
cat /etc/os-release                 # what OS
```

---

## 5. Replace a Dead Worker

```bash
# Destroy the dead one
vastai destroy instance 87654321

# Launch a replacement
./vast_up.sh 1
```

Or use replenish to auto-detect and replace:
```bash
./vast_replenish.sh 4              # ensures 4 default-fleet workers exist
./vast_replenish.sh 4 zeb-large    # ensures 4 zeb-large workers exist
```

---

## 6. Tear Down

```bash
./vast_down.sh                      # all zeb-* instances (confirmation prompt)
./vast_down.sh zeb-large            # only zeb-large-* instances
./vast_down.sh --force              # skip confirmation
./vast_down.sh zeb-large --force    # specific fleet, no confirmation
```

---

## 7. Cost Monitoring

```bash
# Current hourly burn rate
./vast_status.sh   # shows $/hr per instance and total

# Check billing
# Go to: https://cloud.vast.ai/billing/
```

**Budget rules of thumb (Feb 2026 pricing):**
| Workers | Typical GPU | $/hr | $/day | $/week |
|---------|------------|------|-------|--------|
| 4 | RTX 3060 mix | $0.25 | $6.00 | $42 |
| 4 | RTX 3070 mix | $0.30 | $7.20 | $50 |
| 4 | RTX 3060 Ti | $0.22 | $5.28 | $37 |

---

## 8. Claude Code Fleet Management Team

For hands-off fleet management, ask Claude Code to spawn a monitoring team.
This sets up background agents that replace dead workers and track stats.

**Prompt:**
> Launch 4 vast workers and set up a team to monitor them. Keep 4 healthy
> workers running, max $0.08/hr per card. Check every 10 minutes and replace
> duds. Track performance stats by GPU type.

**What gets created:**

| Agent | Role | Duration |
|-------|------|----------|
| fleet-fixer | Destroys bad instances, launches replacements | One-shot, shuts down after |
| fleet-monitor | Checks health every ~10 min, replaces duds | 3 cycles (~30 min) |
| stats-tracker | Collects games/s and $/hr by GPU type | 3 cycles (~30 min) |

**How it works:**
1. Team lead creates a task list with sequential monitoring cycles
2. `fleet-fixer` handles any immediate replacements
3. `fleet-monitor` runs `vast_status.sh` + `vastai logs` each cycle, destroys stuck instances (>10 min loading), launches replacements via `vast_up.sh 1`
4. `stats-tracker` parses log lines for `games/s` and builds efficiency tables

**Key parameters to specify:**
- Number of workers (e.g., 4)
- Max $/hr per card (e.g., $0.08)
- Check interval (e.g., 10 min)
- Number of monitoring cycles (e.g., 3)

**Known issues the team handles automatically:**
- Dud hosts stuck in "loading" or "No such container" >10 min
- SSH port forwarding errors (host-side problem, need a new instance)
- Crashed/exited workers

**GPU performance reference (Feb 2026, 264 observations over 8h run):**

| GPU | Avg games/s | Typical $/hr | Games/$/hr | Notes |
|-----|-------------|--------------|------------|-------|
| RTX 3060 | 3.8-4.0 | $0.052-0.058 | ~214K | Cheapest, most available, best value |
| RTX 3060 Ti | 4.2 | $0.055 | ~275K | Best bang for buck when available |
| RTX 3070 Ti | 5.1 | $0.098 | ~187K | Fastest practical option |
| RTX 3080 | 5.1 | $0.107 | ~171K | Same speed as 3070 Ti, 9% more expensive |
| Expensive 3060 | 3.9 | $0.083 | ~169K | Worst value — avoid paying >$0.06 for a 3060 |

**Session benchmark (Feb 2026):**
8h run, 4 workers, ~500K games generated, ~$2.88 total cost, step 0→11000.

**HuggingFace rate limit (128 commits/hr):**
Workers batch examples locally and upload as a folder every 240 seconds.
With 4 workers that's ~60 commits/hr. With 6 workers, ~90 commits/hr.
Do NOT reduce `--upload-interval` below 120s with 4+ workers.

---

## 9. Multi-Model Fleets

Multiple models can share the same HF repos, isolated by namespace. Each model gets its own fleet label prefix.

### Launch a new model alongside the existing one

```bash
# 1. Create a fresh checkpoint
python -m forge.zeb.init_checkpoint --size large -o forge/zeb/checkpoints/large-init.pt

# 2. Start learner locally
python -u -m forge.zeb.learner.run \
    --repo-id jasonyandell/zeb-42 \
    --examples-repo-id jasonyandell/zeb-42-examples \
    --weights-name large \
    --checkpoint forge/zeb/checkpoints/large-init.pt \
    --lr 1e-4 --batch-size 64 --replay-buffer-size 50000 \
    --training-steps-per-cycle 1000 --wandb

# 3. Launch workers for the new model
ZEB_WEIGHTS_NAME=large ZEB_FLEET=zeb-large ./vast_up.sh 4
```

### Fleet management with prefixes

All shell scripts accept an optional fleet prefix:

```bash
./vast_status.sh                    # all zeb-* instances (both fleets)
./vast_status.sh zeb-large          # only zeb-large-* instances
./vast_down.sh zeb-large            # tear down just the large fleet
./vast_down.sh zeb-large --force    # skip confirmation
./vast_replenish.sh 4 zeb-large     # ensure 4 zeb-large workers
```

### Environment variables for `vast_up.sh`

| Variable | Default | Purpose |
|----------|---------|---------|
| `ZEB_REPO_ID` | `jasonyandell/zeb-42` | HF weights repo |
| `ZEB_EXAMPLES_REPO_ID` | `jasonyandell/zeb-42-examples` | HF examples repo |
| `ZEB_WEIGHTS_NAME` | `zeb-557k-1m` | Weights filename stem |
| `ZEB_FLEET` | `zeb` | Instance label prefix |

### HF repo layout with multiple models

```
jasonyandell/zeb-42/
├── config.json                      # default model config (backward compat)
├── model.pt                         # default weights (if using default namespace)
├── training_state.json              # default step/games
├── zeb-557k-1m.pt                   # existing medium model
├── zeb-557k-1m-config.json
├── zeb-557k-1m-state.json
├── large.pt                         # new large model
├── large-config.json
└── large-state.json

jasonyandell/zeb-42-examples/
├── worker-vast-0_170712_abc.pt      # existing model examples (root)
└── large/
    ├── worker-vast-0_170799_def.pt  # large model examples (subdirectory)
    └── worker-vast-1_170799_ghi.pt
```

---

## 10. Autonomous Monitor

`vast_monitor.sh` is a long-running process that replaces the manual up → status → replenish → down workflow. It maintains a target fleet size, auto-replaces stuck/dead/expensive workers, and tears down cleanly on Ctrl-C.

### Quick start

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
./vast_monitor.sh 4            # maintain 4 workers, Ctrl-C to stop
```

### How it works

1. **Bootstrap check** — verifies weights exist on HF before launching anything
2. **Phase 1 (first 30 min)** — checks every 60s for fast ramp-up
3. **Phase 2 (after 30 min)** — checks every 600s for steady-state
4. **Each check**: queries fleet status, inspects logs, replaces failures, replenishes missing workers
5. **Ctrl-C** — runs `vast_down.sh` to tear down the fleet, then exits

### Configuration

Uses the same `ZEB_*` env vars as `vast_up.sh`, plus:

| Variable | Default | Purpose |
|----------|---------|---------|
| `ZEB_MAX_DPH` | `0.09` | Max cost per worker $/hr — workers above this get replaced |
| `ZEB_GPUS` | `RTX_3060 ... RTX_3090` | Space-separated GPU types to consider |

### Dry run

```bash
./vast_monitor.sh 4 --dry-run   # verify config, bootstrap check, then exit
```

### Worker replacement triggers

| Condition | Timeout | Action |
|-----------|---------|--------|
| Fatal error (Exception, OOM, etc.) | 5 min grace period | Destroy + respawn |
| Stuck in SETUP (git/pip phase) | 10 min | Destroy + respawn |
| Stuck in BOOT (Docker pull) | 10 min | Destroy + respawn |
| No progress (WAIT state) | 15 min | Destroy + respawn |
| Over cost limit (`ZEB_MAX_DPH`) | Immediate | Destroy + respawn |

### Multi-model fleets

```bash
ZEB_WEIGHTS_NAME=large ZEB_FLEET=zeb-large ./vast_monitor.sh 4
```

---

## 11. Daily Operations Checklist

**Morning:**
1. `./vast_status.sh` — all workers running?
2. Check HF examples repo — new files from overnight?
3. Check W&B — is the learner training and loss improving?

**If workers died overnight:**
```bash
./vast_replenish.sh 4
```

**End of day:**
- Nothing to do. Workers run indefinitely. Learner checkpoints locally.

**End of experiment:**
```bash
./vast_down.sh
```
