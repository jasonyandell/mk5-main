# Start a New Training Run

Single-page checklist for launching a fresh self-play training run.
For deeper ops detail see [vast/RUNBOOK.md](vast/RUNBOOK.md) and [learner/RUNBOOK.md](learner/RUNBOOK.md).

## Prerequisites

- `vastai` CLI installed and configured (`vastai set api-key ...`)
- HF token with **write** access to both repos:
  ```bash
  export HF_TOKEN=$(cat ~/.cache/huggingface/token)
  ```
- HF repos already created on huggingface.co (the learner will init them on first push):
  - Weights: `jasonyandell/zeb-42`
  - Examples: `jasonyandell/zeb-42-examples`

## Steps

### 1. Create a fresh checkpoint

```bash
python -m forge.zeb.init_checkpoint --size large -o forge/zeb/checkpoints/large-init.pt
```

Sizes: `small` (64d, 25K params), `medium` (128d, 557K params), `large` (256d, 2.3M params).

Add `--tokenizer v1` to be explicit (v1 is the default).

### 2. Pick a weights name

The `--weights-name` is the namespace on HF. It determines filenames (`large.pt`, `large-config.json`) and the examples subdirectory (`large/`). Pick something descriptive:

```
large              # just a bigger model
large-v2           # large model with v2 tokenizer
```

If you omit `--weights-name`, the default namespace is `model` (files: `model.pt`, etc.).

### 3. Start the learner (local machine)

```bash
python -u -m forge.zeb.learner.run \
    --repo-id jasonyandell/zeb-42 \
    --examples-repo-id jasonyandell/zeb-42-examples \
    --checkpoint forge/zeb/checkpoints/large-init.pt \
    --weights-name large \
    --lr 1e-4 \
    --batch-size 64 \
    --replay-buffer-size 200000 \
    --training-steps-per-cycle 100 \
    --wandb --run-name large-overnight
```

**Wait for** `Bootstrapped HF repo` or `Pulling weights from HF` — this confirms the HF repo is set up and the learner is ready to receive examples.

The learner will print `Buffer N/5000 — waiting for workers...` until workers produce enough data.

### 4. Launch Vast.ai workers

```bash
cd forge/zeb/vast
ZEB_WEIGHTS_NAME=large ZEB_FLEET=zeb-large ./vast_up.sh 4
```

4 workers is a good starting point (~$0.25/hr total with RTX 3060s).

### 5. Verify everything is running

```bash
# Worker status
./vast_status.sh zeb-large

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
- **`./vast_status.sh zeb-large`**: make sure workers are still alive
- Workers auto-sync weights every 10 batches; learner pushes to HF every 25 cycles

### 7. Tear down in the morning

```bash
# Stop workers (learner can keep running)
cd forge/zeb/vast
./vast_down.sh zeb-large

# Or stop everything
./vast_down.sh
```

The learner can be stopped with Ctrl-C. All state is on HF — restart from step 3 to resume.

## Running alongside an existing model

No conflict. Each `--weights-name` gets its own namespace on HF:

```
jasonyandell/zeb-42/
├── zeb-557k-1m.pt          # existing model
├── large.pt                 # new model
└── ...

jasonyandell/zeb-42-examples/
├── worker-vast-0_*.pt       # existing model examples (root)
└── large/
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
