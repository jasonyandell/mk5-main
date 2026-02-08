# Vast.ai Deployment

Shell scripts for running zeb workers on Vast.ai GPU marketplace.

**Philosophy**: The learner runs locally (free 3050 Ti, no preemption risk).
Workers are stateless and go on cheap interruptible Vast.ai instances.
If one dies, `vast_replenish.sh` replaces it. The human is the control plane.

## Prerequisites

```bash
pip install vastai
vastai set api-key YOUR_API_KEY          # from https://cloud.vast.ai/cli/
export HF_TOKEN=hf_...                   # fine-grained token, write access to zeb-42 repos
```

## Usage

```bash
# Launch 4 workers on cheapest 3000-series GPUs
./vast_up.sh 4

# Check what's running
./vast_status.sh

# Replace dead workers (safe to run repeatedly / from cron)
./vast_replenish.sh 4

# Tear everything down
./vast_down.sh
```

### Multi-model fleets

Multiple models can run concurrently using the same HF repos, isolated by namespace:

```bash
# Launch a second model fleet alongside the existing one
ZEB_WEIGHTS_NAME=large ZEB_FLEET=zeb-large ./vast_up.sh 4

# Monitor / manage just that fleet
./vast_status.sh zeb-large
./vast_replenish.sh 4 zeb-large
./vast_down.sh zeb-large
```

Environment variables for `vast_up.sh`: `ZEB_REPO_ID`, `ZEB_EXAMPLES_REPO_ID`, `ZEB_WEIGHTS_NAME`, `ZEB_FLEET`.

## Workflow

1. Start the learner locally:
   ```bash
   python -u -m forge.zeb.learner.run \
       --repo-id jasonyandell/zeb-42 \
       --examples-repo-id jasonyandell/zeb-42-examples \
       --checkpoint forge/zeb/checkpoints/selfplay-epoch3599.pt \
       --weights-name zeb-557k-1m \
       --lr 1e-4 --batch-size 64 \
       --replay-buffer-size 500000 \
       --training-steps-per-cycle 1000 \
       --push-every 10 --save-every 10 --eval-every 10 \
       --eval-games 2000 --keep-checkpoints 3 \
       --keep-example-files 15 --wandb
   ```

2. Launch workers: `./vast_up.sh 4`

3. Monitor:
   - `./vast_status.sh` — instance status and costs
   - W&B dashboard — training curves
   - HF examples repo — are new files appearing?

4. If workers die: `./vast_replenish.sh 4` (or add to cron for auto-recovery)

5. When done: `./vast_down.sh`

## Debugging a Worker

```bash
# Get SSH command from status
./vast_status.sh

# SSH in
ssh -p PORT root@HOST

# Check logs
tail -f /root/worker.log    # worker output
cat /root/setup.log          # startup/setup log
nvidia-smi                   # GPU status
ps aux | grep worker         # process alive?
```

## Cost Estimates

| GPU | Interruptible $/hr | 4 workers $/day |
|-----|-------------------|-----------------|
| RTX 3060 | ~$0.08 | ~$7.70 |
| RTX 3070 | ~$0.12 | ~$11.50 |
| RTX 3080 | ~$0.15 | ~$14.40 |
| RTX 3090 | ~$0.20 | ~$19.20 |
