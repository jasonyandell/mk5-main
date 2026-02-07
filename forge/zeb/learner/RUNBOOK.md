# Learner Runbook

## Quick Reference

```bash
# Start learner
python -u -m forge.zeb.learner.run \
    --repo-id jasonyandell/zeb-42 \
    --examples-repo-id jasonyandell/zeb-42-examples \
    --checkpoint forge/zeb/models/zeb-557k-1m.pt \
    --run-name zeb-557k-1m

# Check output
tail -20 LEARNER_OUTPUT_FILE

# Check HF state
python -c "
from huggingface_hub import hf_hub_download
import json
path = hf_hub_download('jasonyandell/zeb-42', 'training_state.json', force_download=True)
with open(path) as f: print(json.load(f))
"
```

---

## Architecture

HuggingFace is the **single source of truth** for model weights. There are no local checkpoints.

```
Workers (Vast.ai)                    Learner (local 3050 Ti)
┌──────────────┐                    ┌──────────────────┐
│ Self-play    │──examples.pt──→    │ Ingest examples  │
│ MCTS games   │   (HF examples    │ Train on buffer  │
│              │    repo)           │ Push weights     │
│              │                    │ Log to W&B       │
│ Pull weights │←──weights.pt──     │                  │
└──────────────┘   (HF weights      └──────────────────┘
                    repo)
```

**On restart, the learner:**
1. Loads bootstrap checkpoint (just to get model config)
2. Pulls latest weights from HF → picks up where it left off
3. Rebuilds replay buffer from HF examples repo
4. Creates a new W&B run (same display name, new ID)
5. Continues training

---

## Starting the Learner

```bash
python -u -m forge.zeb.learner.run \
    --repo-id jasonyandell/zeb-42 \
    --examples-repo-id jasonyandell/zeb-42-examples \
    --checkpoint forge/zeb/models/zeb-557k-1m.pt \
    --run-name zeb-557k-1m \
    --lr 1e-4 \
    --batch-size 64 \
    --replay-buffer-size 200000 \
    --training-steps-per-cycle 100 \
    --push-every 25 \
    --eval-every 50 \
    --device cuda
```

### Key Parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--replay-buffer-size` | 200000 | Larger = more diverse training, more VRAM |
| `--push-every` | 25 | Push weights to HF every N cycles (~1 push/min) |
| `--eval-every` | 50 | Eval vs random every N cycles |
| `--training-steps-per-cycle` | 100 | Gradient steps per cycle |
| `--batch-size` | 64 | Training batch size |
| `--run-name` | auto | Used as W&B display name AND HF weights filename (`{name}.pt`) |
| `--min-buffer-size` | 5000 | Won't train until buffer has this many examples |

### What Healthy Startup Looks Like

```
Loading bootstrap checkpoint: forge/zeb/models/zeb-557k-1m.pt
  Model: 556,970 parameters
Pulling weights from HF: zeb-557k-1m.pt (step 9925, 1,464,832 games)
Replay buffer (HF): 172,032/200,000 examples from 24 files
W&B run: https://wandb.ai/jasonyandell-forge42/zeb-mcts/runs/abc123

=== Distributed Learner ===
Examples: jasonyandell/zeb-42-examples
Repo: jasonyandell/zeb-42
Starting from cycle 9925
LR: 0.0001, batch: 64, steps/cycle: 100
Min buffer: 5,000, buffer capacity: 200,000

Ingested 7,168 examples from 1 files [buffer: 179,200]
Cycle 9926: policy_loss=0.3226, value_loss=0.2622 (train=3.40s) [buffer: 179,200, games: 1,465,088]
```

---

## W&B Metrics

### What's Tracked

| Metric | What it tells you |
|--------|-------------------|
| `train/policy_loss` | Cross-entropy between model and MCTS policy targets |
| `train/value_loss` | MSE between predicted and actual game values |
| `train/policy_entropy` | How exploratory the policy is (higher = more spread) |
| `train/policy_top1_accuracy` | Does model's best move match MCTS's best move? |
| `train/policy_kl_divergence` | How far model distribution is from MCTS targets |
| `train/grad_norm` | Overall gradient magnitude |
| `train/grad_norm_value_head` | Value head gradient magnitude (canary for explosions) |
| `train/value_mean` | Where value predictions are centered |
| `train/value_std` | Are predictions collapsing to a single value? |
| `train/value_target_mean` | Mean of MCTS value targets in replay buffer |
| `train/lr` | Current learning rate |
| `eval/vs_random_win_rate` | Win rate against random player (every `--eval-every` cycles) |
| `stats/total_games` | Cumulative games across all workers |
| `stats/replay_buffer_size` | Current buffer fill level |

### W&B Run Management

- Each learner restart creates a **new W&B run** with the same display name
- This is intentional — avoids resume conflicts after run deletion
- To delete old/noisy runs:
  ```python
  import wandb
  api = wandb.Api()
  r = api.run('jasonyandell-forge42/zeb-mcts/RUN_ID')
  r.delete()
  ```
- To list runs:
  ```python
  import wandb
  api = wandb.Api()
  for r in api.runs('jasonyandell-forge42/zeb-mcts'):
      print(f'{r.id}  {r.state}  {r.name}')
  ```

---

## HF Weights File Naming

The weights filename on HF is derived from `--run-name`:
- `--run-name zeb-557k-1m` → HF file is `zeb-557k-1m.pt`
- Workers must use `--weights-name zeb-557k-1m` to match

**Both learner and workers must agree on the weights name.** The worker flag is `--weights-name` (without `.pt`), set in `vast_up.sh` via `WEIGHTS_NAME`.

---

## Error Handling

### HF Timeouts (httpx.ReadTimeout)

All HF API calls have retry with exponential backoff (3 retries, 5s/10s/20s delays). If all retries fail, the learner **logs the error and continues** — it won't crash.

```
  HF transient error (attempt 1/4): ReadTimeout: The read operation timed out
  Retrying in 5s...
```

If retries exhaust:
```
  HF ingest error (skipping): ReadTimeout: The read operation timed out
  HF push error (will retry next cycle): ReadTimeout: The read operation timed out
```

The learner continues training on its existing buffer. The next cycle will try HF again.

### Static Games Count

If the `games:` count stops increasing, it means no new examples are being ingested. Causes:
- Workers are down (check `./forge/zeb/vast/vast_status.sh`)
- HF examples repo has no new files
- HF is unreachable (timeouts)

The learner will keep training on the existing buffer — it won't crash or stall. But training on stale data for hours isn't productive. Restart workers.

### Bootstrap vs Resume

| Scenario | What happens |
|----------|-------------|
| Fresh HF repo (no `training_state.json`) | Pushes bootstrap checkpoint weights as step 0 |
| HF has weights | Pulls HF weights, ignores bootstrap checkpoint |
| HF timeout on startup | **Crashes** — startup pulls are not wrapped in try/except |

If the learner crashes on startup due to HF timeout, just restart it.

---

## Reverting to an Earlier Model

HF repos have git history. To revert:

```python
from huggingface_hub import hf_hub_download, HfApi

# 1. List commit history
api = HfApi()
commits = api.list_repo_commits('jasonyandell/zeb-42')
for c in commits:
    print(f'{c.commit_id[:10]}  {c.created_at}  {c.title}')

# 2. Download weights from a specific commit
revision = 'COMMIT_HASH'
model_path = hf_hub_download('jasonyandell/zeb-42', 'zeb-557k-1m.pt', revision=revision)
state_path = hf_hub_download('jasonyandell/zeb-42', 'training_state.json', revision=revision)

# 3. Re-upload to HEAD
api.upload_file(path_or_fileobj=model_path, path_in_repo='zeb-557k-1m.pt',
                repo_id='jasonyandell/zeb-42', commit_message='revert to earlier model')
api.upload_file(path_or_fileobj=state_path, path_in_repo='training_state.json',
                repo_id='jasonyandell/zeb-42', commit_message='revert training state')
```

Then restart the learner — it will pull the reverted weights.

---

## Training a New Model

To train a different model architecture or from a different checkpoint:

1. Create a new bootstrap checkpoint (or use an existing one)
2. Choose a new `--run-name` (e.g. `zeb-1m-2m`)
3. The HF repo will store the new weights as `zeb-1m-2m.pt` alongside the old ones
4. Update `WEIGHTS_NAME` in `vast_up.sh` for workers
5. Restart workers so they pull the new model

```bash
# Learner
python -u -m forge.zeb.learner.run \
    --repo-id jasonyandell/zeb-42 \
    --examples-repo-id jasonyandell/zeb-42-examples \
    --checkpoint forge/zeb/models/NEW_CHECKPOINT.pt \
    --run-name zeb-1m-2m \
    ...
```

---

## Monitoring

### Manual Spot Check

```bash
# Learner output
tail -20 LEARNER_OUTPUT_FILE

# HF state
python -c "
from huggingface_hub import hf_hub_download
import json
path = hf_hub_download('jasonyandell/zeb-42', 'training_state.json', force_download=True)
with open(path) as f: print(json.load(f))
"

# HF examples count
python -c "
from huggingface_hub import HfApi
api = HfApi()
files = [f for f in api.list_repo_files('jasonyandell/zeb-42-examples') if f.endswith('.pt')]
print(f'{len(files)} example files')
"
```

### Automated Monitor (Claude Code)

Ask Claude Code to monitor the learner:
> Start monitoring the learner. Every 10 minutes, check if new data is being ingested.
> If it stalls, check HF for new example files.

This creates a background monitor that:
- Checks learner output every 10 minutes for new `Ingested` lines
- If stalled, switches to 1-minute polling and checks HF examples repo
- Warns if workers appear dead (no new HF files for 5+ checks)
- Alerts if the learner process dies

---

## Lessons Learned

1. **HF is the single source of truth.** No local checkpoints. Restart = pull from HF.

2. **HF timeouts are frequent.** All API calls retry 3x with backoff. If retries exhaust, the learner skips that operation and continues training. It won't crash.

3. **W&B runs are disposable.** Each restart creates a new run with the same display name. Don't try to resume W&B runs — it causes conflicts if the run was deleted or if wandb's init hangs.

4. **Replay buffer rebuilds from HF examples repo on startup.** The examples repo is pruned to `--keep-example-files` (default 15) files. With ~7k examples per file and 200k buffer, ~28 files fill the buffer. After pruning, startup typically gets ~100-170k examples.

5. **Push frequency tradeoff.** `--push-every 25` means workers get updated weights roughly every minute. Lower values mean more HF API calls (rate limit risk on free tier). Higher values mean more stale worker data and more training lost on crash.

6. **Workers and learner must agree on weights filename.** Learner uses `--run-name` → `{name}.pt`. Workers use `--weights-name` (set via `WEIGHTS_NAME` in `vast_up.sh`).

7. **Vast.ai workers are unreliable.** Hosts die, get preempted, or are duds. Plan for attrition and monitor regularly. The learner doesn't care — it just trains slower when workers die.

8. **The `games:` count in learner output only increases on ingest.** If workers are down, the count stays flat. The learner keeps training on the existing buffer, which is fine short-term but not productive for hours.
