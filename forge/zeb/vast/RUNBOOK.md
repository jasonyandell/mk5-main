# Vast.ai Worker Runbook

## Quick Reference

```bash
export HF_TOKEN=hf_...              # set before any vast_up

./vast_up.sh 4                      # launch 4 workers
./vast_status.sh                    # see what's running
vastai logs INSTANCE_ID --tail 50   # check worker output
./vast_down.sh                      # destroy everything
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
vastai show instances --raw | python3 -c "
import sys, json
for i in json.load(sys.stdin):
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
./vast_replenish.sh 4    # ensures 4 workers exist, creates any missing
```

---

## 6. Tear Down Everything

```bash
./vast_down.sh
```

Shows instances and asks for confirmation before destroying.
Use `./vast_down.sh --force` to skip the prompt.

---

## 7. Cost Monitoring

```bash
# Current hourly burn rate
./vast_status.sh   # shows $/hr per instance and total

# Check billing
# Go to: https://cloud.vast.ai/billing/
```

**Budget rules of thumb:**
| Workers | Typical GPU | $/hr | $/day | $/week |
|---------|------------|------|-------|--------|
| 2 | RTX 3060 | $0.18 | $4.30 | $30 |
| 4 | RTX 3060 | $0.36 | $8.60 | $60 |
| 4 | RTX 3080 | $0.60 | $14.40 | $101 |

---

## 8. Daily Operations Checklist

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
