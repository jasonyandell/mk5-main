# Zeb Experiments

Canonical workflow for running Zeb training as experiments.

An experiment has two operations only:
- `create`
- `learn`

## Model

An experiment is a namespace (`weights-name`) plus optional metadata.

Namespace drives:
- HF weights files: `<EXP>.pt`, `<EXP>-config.json`, `<EXP>-state.json`
- HF examples directory: `<EXP>/`
- W&B lineage via `wandb_run_id` stored in `<EXP>-state.json`

W&B note:
- Same `<EXP>` resumes the same W&B run lineage.
- New lineage requires a new `<EXP>`.

## Identity Variables

Set these once per experiment:

```bash
export HF_WEIGHTS_REPO=jasonyandell/zeb-42
export HF_EXAMPLES_REPO=jasonyandell/zeb-42-examples

export EXP=zeb-large-belief-eq-v2
export FLEET_SELFPLAY=zeb-${EXP}-selfplay
export FLEET_EVALAUX=zeb-${EXP}-evalaux
export BOOTSTRAP=forge/zeb/checkpoints/${EXP}-init.pt

export WANDB_PROJECT=zeb-mcts
export WANDB_RUN=${EXP}-stage0
```

## Operation: `create`

`create` prepares a bootstrap checkpoint for a new experiment namespace.

Important:
- `--checkpoint` is only used to seed a namespace that does not exist yet.
- Once seeded, restarts pull from HF for that namespace.

### Create from random init (non-belief)

```bash
python -m forge.zeb.init_checkpoint --size large --tokenizer v1 \
  -o forge/zeb/checkpoints/${EXP}-init.pt
```

### Create from random init (belief-head)

```bash
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

### Create from existing HF namespace (warm start)

```bash
export SOURCE_WEIGHTS_NAME=large-belief.pt
python - <<'PY'
import os
import torch
from pathlib import Path
from forge.zeb.hf import pull_weights

repo = os.environ["HF_WEIGHTS_REPO"]
exp = os.environ["EXP"]
source = os.environ["SOURCE_WEIGHTS_NAME"]
state_dict, model_config = pull_weights(repo, device="cpu", weights_name=source)

out = Path(f"forge/zeb/checkpoints/{exp}-bootstrap.pt")
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model_state_dict": state_dict,
        "model_config": model_config,
        "epoch": 0,
        "total_games": 0,
        "source_weights_name": source,
    },
    out,
)
print(out)
PY
```

### Create from local checkpoint file (warm start)

```bash
python -m forge.zeb.export_model \
  forge/zeb/checkpoints/existing-run-epochNNNN.pt \
  forge/zeb/checkpoints/${EXP}-bootstrap.pt
```

## Operation: `learn`

`learn` runs the distributed pipeline for an experiment namespace.

### Learn Stage 0 (self-play only)

```bash
python -u -m forge.zeb.learner.run \
  --repo-id ${HF_WEIGHTS_REPO} \
  --examples-repo-id ${HF_EXAMPLES_REPO} \
  --weights-name ${EXP} \
  --checkpoint ${BOOTSTRAP} \
  --batch-size 64 \
  --replay-buffer-size 500000 \
  --eval-aux-replay-buffer-size 100000 \
  --training-steps-per-cycle 1000 \
  --keep-example-files 128 \
  --amp --amp-dtype fp16 \
  --local-replay-cache-enabled \
  --local-replay-cache-dir ~/.cache/forge/zeb/replay \
  --local-replay-cache-save-every 25 \
  --no-eval-aux-enabled \
  --eval-aux-batch-fraction 0.0 \
  --wandb --wandb-project ${WANDB_PROJECT} --run-name ${WANDB_RUN}
```

If your bootstrap file is `${EXP}-bootstrap.pt`, set:

```bash
export BOOTSTRAP=forge/zeb/checkpoints/${EXP}-bootstrap.pt
```

Launch self-play workers:

```bash
cd forge/zeb/vast
ZEB_WEIGHTS_NAME=${EXP} \
ZEB_FLEET=${FLEET_SELFPLAY} \
ZEB_WORKER_MODE=selfplay \
./go-belief.sh 2
```

### Learn Stage 1 (self-play + eval-aux)

Restart learner with eval-aux enabled:

```bash
python -u -m forge.zeb.learner.run \
  --repo-id ${HF_WEIGHTS_REPO} \
  --examples-repo-id ${HF_EXAMPLES_REPO} \
  --weights-name ${EXP} \
  --checkpoint ${BOOTSTRAP} \
  --batch-size 64 \
  --replay-buffer-size 500000 \
  --eval-aux-replay-buffer-size 100000 \
  --training-steps-per-cycle 1000 \
  --keep-example-files 128 \
  --amp --amp-dtype fp16 \
  --local-replay-cache-enabled \
  --local-replay-cache-dir ~/.cache/forge/zeb/replay \
  --local-replay-cache-save-every 25 \
  --eval-aux-enabled \
  --eval-aux-batch-fraction 0.05 \
  --eval-aux-policy-weight 0.0 \
  --eval-aux-value-weight 1.0 \
  --eval-aux-belief-weight 0.5 \
  --eval-aux-max-model-lag 400 \
  --eval-aux-lag-half-life 200 \
  --eval-aux-min-keep-weight 0.10 \
  --wandb --wandb-project ${WANDB_PROJECT} --run-name ${EXP}-stage1
```

Launch eval-aux worker fleet (separate producer stream):

```bash
cd forge/zeb/vast
ZEB_WEIGHTS_NAME=${EXP} \
ZEB_FLEET=${FLEET_EVALAUX} \
./go-experiment.sh 1
```

## Verify

```bash
cd forge/zeb/vast
./vast_status.sh ${FLEET_SELFPLAY}
./vast_status.sh ${FLEET_EVALAUX}
```

Expect:
- self-play files and eval-aux files under `${EXP}/` in examples repo
- learner logs show non-zero eval mix in Stage 1
- learner logs show `eval_aux_policy_w=0`

## Rollback

Disable eval-aux immediately by restarting learner with:

```bash
--no-eval-aux-enabled --eval-aux-batch-fraction 0.0
```

No fleet teardown required for rollback.
