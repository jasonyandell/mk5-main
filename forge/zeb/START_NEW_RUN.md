# Start a New Training Run

Single-page checklist for launching a fresh self-play training run.
For deeper ops detail see [vast/RUNBOOK.md](vast/RUNBOOK.md) and [learner/RUNBOOK.md](learner/RUNBOOK.md).
For the canonical experiment contract (`create` / `learn`), see [EXPERIMENTS.md](EXPERIMENTS.md).

## Prerequisites

- `vastai` CLI installed and configured (`vastai set api-key ...`)
- HF token with **write** access to both repos:
  ```bash
  export HF_TOKEN=$(cat ~/.cache/huggingface/token)
  ```
- HF repos already created on huggingface.co (the learner will init them on first push):
  - Weights: `jasonyandell/zeb-42`
  - Examples: `jasonyandell/zeb-42-examples`

## Run Identity (Set Once Per Experiment)

Set these first so all commands stay consistent:

```bash
export HF_WEIGHTS_REPO=jasonyandell/zeb-42
export HF_EXAMPLES_REPO=jasonyandell/zeb-42-examples

export EXP=zeb-large-belief-eq-v2            # --weights-name namespace
export FLEET=zeb-${EXP}                      # Vast fleet label prefix
export WANDB_PROJECT=zeb-mcts
export WANDB_RUN=${EXP}-stage0
```

Derived naming:
- HF weights files: `${EXP}.pt`, `${EXP}-config.json`, `${EXP}-state.json`
- HF examples directory: `${EXP}/`
- W&B lineage: tied to `${EXP}` via `wandb_run_id` in `${EXP}-state.json`

## Steps

### 1. Create a fresh checkpoint

```bash
python -m forge.zeb.init_checkpoint --size large -o forge/zeb/checkpoints/${EXP}-init.pt
```

Sizes: `small` (64d, 25K params), `medium` (128d, 557K params), `large` (256d, 2.3M params).

Add `--tokenizer v1` to be explicit (v1 is the default).

### 2. Pick a weights name

The `--weights-name` is the namespace on HF. It determines filenames
(`${EXP}.pt`, `${EXP}-config.json`) and the examples subdirectory (`${EXP}/`).
Pick something descriptive:

```
large              # just a bigger model
large-v2           # large model with v2 tokenizer
```

If you omit `--weights-name`, the default namespace is `model` (files: `model.pt`, etc.).

### 2.1 W&B and namespace coupling

Learner W&B identity is coupled to `--weights-name`:

- Learner reads `wandb_run_id` from `<weights-name>-state.json` in HF.
- Learner calls `wandb.init(..., id=wandb_run_id, resume="allow")`.
- Learner writes `wandb_run_id` back to HF state on pushes.

This is intentional for crash-safe resume. Practical implications:

- Re-running the same `--weights-name` resumes the same W&B lineage.
- `--run-name` only affects first creation; it will not force a new run if the namespace already has `wandb_run_id`.
- For a truly new experiment log lineage, use a new `--weights-name`.

Bootstrap rule:

- `--checkpoint` seeds HF only when the namespace is new.
- After first push, restarts pull from HF for that namespace; local `--checkpoint` is no longer authoritative.

### 3. Start the learner (local machine)

```bash
python -u -m forge.zeb.learner.run \
    --repo-id ${HF_WEIGHTS_REPO} \
    --examples-repo-id ${HF_EXAMPLES_REPO} \
    --checkpoint forge/zeb/checkpoints/${EXP}-init.pt \
    --weights-name ${EXP} \
    --lr 1e-4 \
    --batch-size 64 \
    --replay-buffer-size 500000 \
    --training-steps-per-cycle 1000 \
    --keep-example-files 128 \
    --amp --amp-dtype fp16 \
    --local-replay-cache-enabled \
    --local-replay-cache-dir ~/.cache/forge/zeb/replay \
    --local-replay-cache-save-every 25 \
    --wandb --wandb-project ${WANDB_PROJECT} --run-name ${WANDB_RUN}
```

**Wait for** `Bootstrapped HF repo` or `Pulling weights from HF` — this confirms the HF repo is set up and the learner is ready to receive examples.

The learner will print `Buffer N/5000 — waiting for workers...` until workers produce enough data.

### 3.1 New experiment from existing weights (warm start)

If you want to branch from an existing model but keep a new W&B/HF lineage, create a new namespace and bootstrap checkpoint first.

From an existing HF namespace (example: `large-belief`):

```bash
export EXP=zeb-large-belief-eq-v2
python - <<'PY'
import os
import torch
from pathlib import Path
from forge.zeb.hf import pull_weights

repo = os.environ["HF_WEIGHTS_REPO"]
source_weights_name = "large-belief.pt"
exp = os.environ["EXP"]
state_dict, model_config = pull_weights(repo, device="cpu", weights_name=source_weights_name)

out = Path(f"forge/zeb/checkpoints/{exp}-bootstrap.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": state_dict,
        "model_config": model_config,
        "epoch": 0,
        "total_games": 0,
        "source_weights_name": source_weights_name,
    },
    out,
)
print(out)
PY
```

Then start learner on the new namespace:

```bash
python -u -m forge.zeb.learner.run \
    --repo-id ${HF_WEIGHTS_REPO} \
    --examples-repo-id ${HF_EXAMPLES_REPO} \
    --checkpoint forge/zeb/checkpoints/${EXP}-bootstrap.pt \
    --weights-name ${EXP} \
    --lr 1e-4 \
    --batch-size 64 \
    --replay-buffer-size 500000 \
    --training-steps-per-cycle 1000 \
    --keep-example-files 128 \
    --amp --amp-dtype fp16 \
    --local-replay-cache-enabled \
    --local-replay-cache-dir ~/.cache/forge/zeb/replay \
    --local-replay-cache-save-every 25 \
    --wandb --wandb-project ${WANDB_PROJECT} --run-name ${WANDB_RUN}
```

From a local checkpoint file:

```bash
export EXP=zeb-large-belief-eq-v2
python -m forge.zeb.export_model \
    forge/zeb/checkpoints/existing-run-epochNNNN.pt \
    forge/zeb/checkpoints/${EXP}-bootstrap.pt
```

Then use that `${EXP}-bootstrap.pt` with the same learner command pattern above.

### 3.2 Fresh random-init experiment namespace

For non-belief models:

```bash
export EXP=zeb-large-v2
python -m forge.zeb.init_checkpoint --size large --tokenizer v1 \
    -o forge/zeb/checkpoints/${EXP}-init.pt
```

For belief-head models:

```bash
export EXP=zeb-large-belief-eq-v2
python - <<'PY'
import os
import torch
from pathlib import Path
from forge.zeb.model import ZebModel, get_model_config

exp = os.environ["EXP"]
cfg = get_model_config("large", tokenizer="v1", belief_head=True)
model = ZebModel(**cfg)
out = Path(f"forge/zeb/checkpoints/{exp}-init.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_config": cfg,
        "epoch": 0,
        "total_games": 0,
    },
    out,
)
print(out)
PY
```

Then start learner with:

```bash
python -u -m forge.zeb.learner.run \
    --repo-id ${HF_WEIGHTS_REPO} \
    --examples-repo-id ${HF_EXAMPLES_REPO} \
    --checkpoint forge/zeb/checkpoints/${EXP}-init.pt \
    --weights-name ${EXP} \
    --lr 1e-4 \
    --batch-size 64 \
    --replay-buffer-size 500000 \
    --training-steps-per-cycle 1000 \
    --keep-example-files 128 \
    --amp --amp-dtype fp16 \
    --local-replay-cache-enabled \
    --local-replay-cache-dir ~/.cache/forge/zeb/replay \
    --local-replay-cache-save-every 25 \
    --wandb --wandb-project ${WANDB_PROJECT} --run-name ${WANDB_RUN}
```

### 4. Launch Vast.ai workers

```bash
cd forge/zeb/vast
ZEB_WEIGHTS_NAME=${EXP} ZEB_FLEET=${FLEET} ./vast_up.sh 4
```

4 workers is a good starting point (~$0.25/hr total with RTX 3060s).

### 5. Verify everything is running

```bash
# Worker status
./vast_status.sh ${FLEET}

# Worker logs (get IDs from status output)
vastai logs INSTANCE_ID --tail 30

# Learner should start printing training cycles within ~5 min
```

**Healthy worker output:**
```
[worker-vast-0] batch 1: 7168 examples, 3.8 games/s, step=0, total_games=256
```

**Healthy learner output:**
```
Ingested 28,672 examples from 4 files [buffer: 28,672]
Cycle    1: policy_loss=1.9432, value_loss=0.2451 (train=3.21s)
```

### 6. Monitor overnight

- **W&B dashboard**: check policy_loss trending down, value_loss stabilizing
- **`./vast_status.sh ${FLEET}`**: make sure workers are still alive
- Workers auto-sync weights every 10 batches; learner pushes to HF every 25 cycles

### 7. Tear down in the morning

```bash
# Stop workers (learner can keep running)
cd forge/zeb/vast
./vast_down.sh ${FLEET}

# Or stop everything
./vast_down.sh
```

The learner can be stopped with Ctrl-C. All state is on HF — restart from step 3 to resume.

## Running alongside an existing model

No conflict. Each `--weights-name` gets its own namespace on HF:

```
jasonyandell/zeb-42/
├── zeb-557k-1m.pt          # existing model
├── ${EXP}.pt                # new model namespace
└── ...

jasonyandell/zeb-42-examples/
├── worker-vast-0_*.pt       # existing model examples (root)
└── ${EXP}/
    └── worker-vast-0_*.pt   # new model examples (subdirectory)
```

Just start both learners and both fleets. They don't interfere.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Worker stuck at `loading` > 10 min | Bad host, slow Docker pull | `vastai destroy INSTANCE_ID`, launch replacement |
| `412 Precondition Failed` in worker logs | HF concurrent commit conflict | Normal — workers retry automatically |
| Learner stuck at `waiting for workers` | Workers haven't uploaded yet | Check worker logs; first upload takes ~4 min |
| `No matching offers found` | GPU market empty | Try specific GPU: `./vast_up.sh 2 RTX_3090` |
| Worker logs show `OutOfMemoryError` | GPU too small for model | Use GPUs with >= 6GB VRAM for large model |

## Cost estimates

| Workers | GPU mix | Approx cost/hr | Overnight (8hr) |
|---------|---------|-----------------|------------------|
| 4 | RTX 3060 | ~$0.24 | ~$1.90 |
| 4 | RTX 3060 Ti | ~$0.22 | ~$1.76 |
| 6 | Mixed 3060/3070 | ~$0.45 | ~$3.60 |
