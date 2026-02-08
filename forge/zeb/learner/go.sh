#!/usr/bin/env bash
# go.sh — Launch the large model learner
# Usage: ./go.sh
exec python -u -m forge.zeb.learner.run \
    --repo-id jasonyandell/zeb-42 \
    --examples-repo-id jasonyandell/zeb-42-examples \
    --checkpoint forge/zeb/checkpoints/large-init.pt \
    --weights-name large \
    --lr 1e-4 --batch-size 64 \
    --replay-buffer-size 1000000 \
    --training-steps-per-cycle 1000 \
    --push-every 10 --eval-every 10 \
    --eval-games 2000 \
    --keep-example-files 15 --wandb
