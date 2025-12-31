# Crystal Forge - ML Pipeline Orientation

**New to the ML pipeline?** This document provides the mental models needed to navigate the forge architecture.

## Related Resources

- **Skills** (invoke these for implementation guidance):
  - `pytorch-lightning` - LightningModule patterns, Trainer configuration, callbacks, logging
  - `pytorch` - Core PyTorch patterns, distributed training, optimization
- **Game Architecture**: [docs/ORIENTATION.md](../docs/ORIENTATION.md) - TypeScript game engine
- **Game Rules**: [docs/rules.md](../docs/rules.md) - Official Texas 42 rules

---

## Core Architecture Pattern

The entire ML pipeline is built around this fundamental transformation:

```
ORACLE SHARDS → TOKENIZED DATA → TRAINED MODEL
```

Everything exists to:
1. **Generate** perfect play data (GPU solver computes optimal moves)
2. **Transform** game states into learnable representations (tokenization)
3. **Train** a neural network to imitate the oracle (Lightning training loop)

## Philosophy

1. **Separation of Concerns**: Oracle generates data, tokenizer transforms it, trainer learns from it
2. **Reproducibility**: Per-shard deterministic RNG, Lightning seed_everything, RNG state in checkpoints
3. **PyTorch Lightning First**: Structure prevents slop - LightningModule for model, DataModule for data
4. **Team 0 Perspective**: All V/Q values from Team 0's viewpoint (positive = Team 0 ahead)

---

## The Stack

```
┌─────────────────────────────────┐
│  forge/cli/                     │  User-facing commands
│  - train.py                     │  Train models with Wandb
│  - eval.py                      │  Evaluate checkpoints
│  - tokenize.py                  │  Preprocess shards
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  forge/ml/                      │  ★ LIGHTNING CORE ★
│  - module.py                    │  DominoLightningModule
│  - data.py                      │  DominoDataModule
│  - metrics.py                   │  Q-gap, accuracy, blunder rate
│  - tokenize.py                  │  Tokenization logic
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  forge/oracle/                  │  GPU tablebase solver
│  - solve.py                     │  Minimax with alpha-beta
│  - schema.py                    │  Parquet format, V/Q semantics
│  - generate.py                  │  CLI entry point
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Data Storage                   │  Versioned directories
│  - data/shards/                 │  Oracle parquet output
│  - data/tokenized/              │  Preprocessed numpy arrays
│  - runs/                        │  Training outputs
└─────────────────────────────────┘
```

---

## Data Flow

```
1. GENERATE (GPU solver)
   forge/oracle/generate.py --seeds 0-99 --out data/shards

   Output: data/shards/seed_XXXXXXXX_decl_Y.parquet
           └── Each file: ~50k game states with perfect V/Q values

2. TOKENIZE (CPU preprocessing)
   forge/cli/tokenize --input data/shards --output data/tokenized

   Output: data/tokenized/{train,val,test}/
           └── tokens.npy, masks.npy, targets.npy, legal.npy, qvals.npy, teams.npy

3. TRAIN (GPU learning)
   forge/cli/train --data data/tokenized --wandb

   Output: runs/domino/version_X/
           └── checkpoints/, hparams.yaml, metrics.csv
```

---

## Mental Models

### The Oracle as Ground Truth

The GPU solver computes **perfect play** via minimax search. For every game state, it knows:
- **V (value)**: Expected point differential if both teams play optimally
- **Q (action-values)**: Expected value after each legal move
- **Best move**: The action with highest Q (for Team 0) or lowest Q (for Team 1)

The neural network learns to approximate this oracle. Training loss measures how well predictions match oracle decisions.

### Team 0 Perspective (Critical!)

All values are from Team 0's viewpoint:
- **Positive V** = Team 0 is ahead
- **Negative V** = Team 1 is ahead
- **Team 0 maximizes Q**, Team 1 minimizes Q

When computing metrics, we apply a team sign:
```python
team_sign = 1 if team == 0 else -1
q_for_team = q_values * team_sign
```

### Q-Gap as the Key Metric

**Q-gap** measures oracle regret - how many points the model's choice loses compared to optimal:

```python
q_gap = oracle_best_q - oracle_q[model_prediction]
```

- **Q-gap = 0**: Model chose optimally
- **Q-gap = 5**: Model's choice costs 5 points on average
- **Blunder**: Q-gap > 10 (catastrophic mistake)

### Lightning as Guardrails

PyTorch Lightning enforces structure that prevents slop:

| Component | Responsibility | Lightning Provides |
|-----------|---------------|-------------------|
| **DominoTransformer** | Model architecture | Pure nn.Module |
| **DominoLightningModule** | Training logic | Logging, checkpoints, optimization |
| **DominoDataModule** | Data pipeline | prepare_data/setup separation, DataLoaders |
| **Trainer** | Training loop | Devices, precision, callbacks |

**Invoke the `pytorch-lightning` skill** for implementation patterns.

### Per-Shard Deterministic RNG

Sampling must be reproducible regardless of which shards are processed:

```python
# BAD: Global RNG - adding shards changes all samples
rng = np.random.default_rng(seed)
for shard in shards:
    samples = rng.choice(...)  # Depends on processing order!

# GOOD: Per-shard RNG - each shard is independent
for shard in shards:
    shard_rng = np.random.default_rng((global_seed, shard.seed, shard.decl_id))
    samples = shard_rng.choice(...)  # Same samples every time
```

---

## File Map

**Oracle** (GPU solver):
- `forge/oracle/solve.py` - Minimax search with alpha-beta pruning
- `forge/oracle/schema.py` - Parquet format, state packing, V/Q semantics
- `forge/oracle/state.py` - Game state representation (41 bits packed in int64)
- `forge/oracle/rng.py` - Deterministic deal generation from seed
- `forge/oracle/generate.py` - CLI: generate shards

**ML Core** (Lightning):
- `forge/ml/module.py` - DominoTransformer + DominoLightningModule
- `forge/ml/data.py` - DominoDataset + DominoDataModule
- `forge/ml/metrics.py` - Q-gap, accuracy, blunder rate computation
- `forge/ml/tokenize.py` - State → token conversion logic

**CLI** (user-facing):
- `forge/cli/train.py` - Training with Wandb, callbacks, checkpointing
- `forge/cli/eval.py` - Evaluate checkpoint on test set
- `forge/cli/tokenize.py` - Preprocess shards to numpy arrays

**Data** (versioned storage):
- `data/shards/` - Oracle parquet output
- `data/tokenized/{train,val,test}/` - Preprocessed numpy arrays
- `runs/domino/version_X/` - Training outputs (Lightning convention)

**Archive** (historical reference):
- `forge/archive/solver2/` - Original solver scripts (frozen, do not modify)

---

## Key Abstractions

### Oracle Shard (Parquet)

Each shard contains ~50k game states from one (seed, declaration) pair:

| Column | Type | Description |
|--------|------|-------------|
| `state` | int64 | Packed game state (41 bits) |
| `V` | int8 | Value: expected point differential (Team 0 perspective) |
| `Q0`-`Q6` | int8 | Q-values for each action (-128 = illegal) |

Metadata in parquet schema: `seed`, `decl_id`, `generator_version`

### Token Format

Neural network input is `int8[batch, 32, 12]`:
- **32 positions**: 7 hand + 21 trick history + 4 metadata
- **12 features per position**: pip values, trump rank, player info, etc.

### Split Assignment

Deterministic train/val/test split by seed:

```python
def get_split(seed: int) -> str:
    bucket = seed % 1000
    if bucket >= 950:
        return 'test'   # 5% - sacred, never touched
    elif bucket >= 900:
        return 'val'    # 5% - model selection
    else:
        return 'train'  # 90%
```

---

## Essential Commands

### Generate Oracle Data

```bash
# Generate shards for seeds 0-99, all 10 declarations
python -m forge.oracle.generate --seeds 0-99 --out data/shards
```

### Tokenize for Training

```bash
# Preprocess shards into train/val/test splits
python -m forge.cli.tokenize --input data/shards --output data/tokenized
```

### Train Model

```bash
# Quick sanity check (1 batch)
python -m forge.cli.train --fast-dev-run --no-wandb

# Full training with Wandb
python -m forge.cli.train --epochs 10 --wandb

# Resume from checkpoint
python -m forge.cli.train --resume runs/domino/version_0/checkpoints/last.ckpt
```

### Evaluate Model

```bash
# Test set evaluation
python -m forge.cli.eval --checkpoint runs/domino/version_0/checkpoints/best.ckpt

# Validation set
python -m forge.cli.eval --checkpoint runs/domino/version_0/checkpoints/best.ckpt --split val
```

---

## Common Patterns

### Adding a New Metric

1. Implement in `forge/ml/metrics.py`:
```python
def compute_my_metric(logits: Tensor, targets: Tensor, ...) -> Tensor:
    ...
```

2. Log in `DominoLightningModule.validation_step`:
```python
self.log('val/my_metric', value, sync_dist=True)
```

### Modifying the Model

1. Change architecture in `DominoTransformer` (pure nn.Module)
2. Training logic stays in `DominoLightningModule`
3. Use `self.save_hyperparameters()` to track changes

### Adding a Callback

```python
# In forge/cli/train.py
from lightning.pytorch.callbacks import MyCallback

callbacks = [
    ...,
    MyCallback(...),
]
```

**Invoke the `pytorch-lightning` skill** for callback patterns.

---

## Debugging Tips

| Issue | Where to Look |
|-------|---------------|
| Model not learning | Check Q-gap trend, verify data loading |
| OOM errors | Reduce batch size, enable gradient checkpointing |
| Slow training | Check num_workers, enable pin_memory |
| Metrics not logging | Verify `self.log()` calls, check sync_dist |
| Checkpoint won't load | Check hparams match, verify Lightning version |
| Inconsistent results | Check RNG seeds, verify deterministic mode |

### Quick Diagnostics

```bash
# Verify imports work
python -c "from forge.ml import module, data, metrics; print('OK')"

# Check data structure
python -c "
from pathlib import Path
import numpy as np
p = Path('data/tokenized/train')
print(f'tokens: {np.load(p/\"tokens.npy\", mmap_mode=\"r\").shape}')
"

# Fast training test
python -m forge.cli.train --fast-dev-run --no-wandb
```

---

## Architectural Invariants

1. **Team 0 Perspective**: All V/Q values from Team 0's viewpoint
2. **Deterministic Splits**: seed % 1000 determines train/val/test
3. **Per-Shard RNG**: Sampling independent of shard processing order
4. **Lightning Structure**: LightningModule for training, DataModule for data
5. **No Models in data/**: Checkpoints go to runs/, not data/
6. **Manifests Track Provenance**: Each stage has manifest.yaml

**Violation of any invariant = pipeline regression.**

---

## What's Not Here Yet

- **Hyperparameter tuning**: Optuna integration
- **Distributed training**: Multi-GPU with DDP/FSDP
- **Model serving**: Export to ONNX or TorchScript
- **Active learning**: Mine hard examples for retraining

---

## Quick Start

```bash
# 1. Verify environment
python -c "import torch; import lightning; print('PyTorch:', torch.__version__)"

# 2. Check existing data
ls data/tokenized/

# 3. Run fast training test
python -m forge.cli.train --data data/tokenized --fast-dev-run --no-wandb

# 4. Train for real
python -m forge.cli.train --data data/tokenized --epochs 10 --wandb
```

---

**Skill References**:
- For Lightning patterns → invoke `pytorch-lightning` skill
- For PyTorch fundamentals → invoke `pytorch` skill
