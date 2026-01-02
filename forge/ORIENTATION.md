# Crystal Forge - ML Pipeline Orientation

**New to the ML pipeline?** This document provides the mental models needed to navigate the forge architecture.

## Related Resources

**Skills** (invoke these for implementation guidance):
- `pytorch-lightning` - LightningModule patterns, Trainer configuration, callbacks, logging
- `pytorch` - Core PyTorch patterns, distributed training, optimization
- `wandb` - Experiment tracking, sweeps, artifact versioning, reports

**Documentation**:
- [forge/models/README.md](models/README.md) - Model catalog, loading, architecture details
- [forge/bidding/README.md](bidding/README.md) - Bidding evaluation (System 2)
- [forge/flywheel/RUNBOOK.md](flywheel/RUNBOOK.md) - Iterative training operations

**Game Context**:
- [docs/ORIENTATION.md](../docs/ORIENTATION.md) - TypeScript game engine architecture
- [docs/rules.md](../docs/rules.md) - Official Texas 42 rules

---

## Setup

### Dependencies

```bash
pip install -r forge/requirements.txt
```

Core requirements:
- `torch>=2.0` - PyTorch for neural networks
- `lightning>=2.0` - PyTorch Lightning for training structure
- `pyarrow>=14.0` - Parquet file I/O
- `wandb>=0.16` - Experiment tracking
- `numpy>=1.26,<2` - Pinned for torch compatibility

### Verification

```bash
# Verify imports work
python -c "from forge.ml import module, data, metrics; from forge.oracle import schema; print('OK')"

# Quick training sanity check
python -m forge.cli.train --fast-dev-run --no-wandb
```

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
2. **Reproducibility**: Per-shard deterministic RNG, Lightning seed_everything
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
│  - flywheel.py                  │  Iterative training
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
│  forge/bidding/                 │  Bid strength evaluation
│  - evaluate.py                  │  P(make) matrix
│  - simulator.py                 │  Monte Carlo simulation
│  - poster.py                    │  Visual PDF output
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  forge/scripts/                 │  Cloud training automation
│  - cloud-setup.sh               │  Environment setup on Lambda/etc
│  - cloud-run.sh                 │  End-to-end training pipeline
│  - cloud-train*.sh              │  Training job variants
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Data Storage                   │  Versioned directories
│  - data/shards/                 │  Oracle parquet output
│  - data/tokenized/              │  Preprocessed numpy arrays
│  - runs/                        │  Training outputs
│  - forge/models/                │  Pre-trained checkpoints
└─────────────────────────────────┘
```

---

## Data Flow

```
1. GENERATE (GPU solver)
   python -m forge.oracle.generate --seed-range 0:100 --out data/shards

   Output: data/shards/seed_XXXXXXXX_decl_Y.parquet
           └── Each file: ~50k game states with perfect V/Q values

2. TOKENIZE (CPU preprocessing)
   python -m forge.cli.tokenize --input data/shards --output data/tokenized

   Output: data/tokenized/{train,val,test}/
           └── tokens.npy, masks.npy, targets.npy, legal.npy, qvals.npy

3. TRAIN (GPU learning)
   python -m forge.cli.train --data data/tokenized --wandb

   Output: runs/domino/version_X/
           └── checkpoints/, hparams.yaml, metrics.csv
```

Use `--help` on any command for full options.

---

## Mental Models

### The Oracle as Ground Truth

The GPU solver computes **perfect play** via minimax search. For every game state, it knows:
- **V (value)**: Expected point differential if both teams play optimally
- **Q (action-values)**: Expected value after each legal move
- **Best move**: The action with highest Q (for Team 0) or lowest Q (for Team 1)

The neural network learns to approximate this oracle.

### Team 0 Perspective (Critical!)

All values are from Team 0's viewpoint:
- **Positive V** = Team 0 is ahead
- **Negative V** = Team 1 is ahead
- **Team 0 maximizes Q**, Team 1 minimizes Q

### Q-Gap as the Key Metric

**Q-gap** measures oracle regret - how many points the model's choice loses compared to optimal:

```python
q_gap = oracle_best_q - oracle_q[model_prediction]
```

- **Q-gap = 0**: Model chose optimally
- **Q-gap = 5**: Model's choice costs 5 points on average
- **Blunder**: Q-gap > 10 (catastrophic mistake)

### Lightning as Guardrails

| Component | Responsibility | Lightning Provides |
|-----------|---------------|-------------------|
| **DominoTransformer** | Model architecture | Pure nn.Module |
| **DominoLightningModule** | Training logic | Logging, checkpoints, optimization |
| **DominoDataModule** | Data pipeline | prepare_data/setup separation, DataLoaders |
| **Trainer** | Training loop | Devices, precision, callbacks |

---

## Key Abstractions

### Oracle Shard (Parquet)

Each shard contains ~50k game states from one (seed, declaration) pair:

| Column | Type | Description |
|--------|------|-------------|
| `state` | int64 | Packed game state (41 bits) |
| `V` | int8 | Value: expected point differential (Team 0 perspective) |
| `q0`-`q6` | int8 | Q-values for each action (-128 = illegal) |

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

```bash
# Generate oracle shards
python -m forge.oracle.generate --seed-range 0:100 --out data/shards

# Tokenize for training
python -m forge.cli.tokenize --input data/shards --output data/tokenized

# Quick sanity check
python -m forge.cli.train --fast-dev-run --no-wandb

# Full training with Wandb
python -m forge.cli.train --epochs 10 --wandb

# Evaluate on test set
python -m forge.cli.eval --checkpoint runs/domino/version_0/checkpoints/best.ckpt

# Bidding evaluation
python -m forge.bidding.evaluate --hand "6-4,5-5,4-2,3-1,2-0,1-1,0-0" --samples 100
```

Training auto-detects hardware. Use `--precision bf16-mixed` for A100/H100 servers.

---

## Bidding Evaluation (System 2)

Monte Carlo simulation evaluates bid strength by playing out complete games with the trained model.

**Quick start:**
```bash
python -m forge.bidding.evaluate --hand "6-6,6-5,6-4,6-3,6-2,6-1,6-0" --samples 100
```

**Output:** P(make) matrix showing probability of making each bid (30-42) for each trump choice.

See [bidding/README.md](bidding/README.md) for module overview, and [bidding/EXAMPLES.md](bidding/EXAMPLES.md) for worked examples with analysis.

---

## Flywheel: Iterative Fine-Tuning

Automates iterative model improvement: generate new seeds → train → evaluate → repeat.

```bash
python -m forge.cli.flywheel status   # Check current state
python -m forge.cli.flywheel          # Run continuously
python -m forge.cli.flywheel --once   # Run one iteration
```

See [flywheel/RUNBOOK.md](flywheel/RUNBOOK.md) for full operations guide.

---

## Debugging Tips

| Issue | Where to Look |
|-------|---------------|
| Model not learning | Check Q-gap trend, verify data loading |
| OOM errors | Reduce batch size, enable gradient checkpointing |
| Slow training | Check num_workers, enable pin_memory |
| Checkpoint won't load | Check hparams match, verify Lightning version |
| Inconsistent results | Check RNG seeds, verify deterministic mode |

### Quick Diagnostics

```bash
# Verify imports work
python -c "from forge.ml import module, data, metrics; print('OK')"

# Fast training test
python -m forge.cli.train --fast-dev-run --no-wandb
```

---

## Architectural Invariants

1. **Team 0 Perspective**: All V/Q values from Team 0's viewpoint
2. **Deterministic Splits**: seed % 1000 determines train/val/test
3. **Per-Shard RNG**: Sampling independent of shard processing order
4. **Lightning Structure**: LightningModule for training, DataModule for data
5. **No Models in data/**: Checkpoints go to runs/ or forge/models/

**Violation of any invariant = pipeline regression.**

---

## LLM Workflow

This section is for you, the LLM working on this codebase.

### State Diagnosis

Before making changes, understand the current state:

```bash
# What shards exist?
ls data/shards/*.parquet 2>/dev/null | wc -l

# What's tokenized?
cat data/tokenized/manifest.yaml 2>/dev/null || echo "No tokenized data"

# Any training runs?
ls runs/domino/ 2>/dev/null || echo "No runs yet"

# What's currently running?
pgrep -af "forge.oracle\|forge.cli" || echo "Nothing running"
```

### Change Impact

| If you modify... | Regenerate... | Invalidates... |
|------------------|---------------|----------------|
| `forge/oracle/*.py` | shards | data/shards/, data/tokenized/ |
| `forge/ml/tokenize.py` | tokenized | data/tokenized/ |
| `forge/ml/module.py` | train | runs/ (retrain needed) |
| `forge/ml/metrics.py` | nothing | (metrics-only change) |

### Verification Commands

```bash
# After any forge/ changes - verify imports
python -c "from forge.ml import module, data, metrics; from forge.oracle import schema; print('OK')"

# After model changes - quick training test
python -m forge.cli.train --fast-dev-run --no-wandb
```

---

## Glossary

| Term | Definition |
|------|------------|
| **q0-q6** | Q-values: optimal value of each action. Parquet columns, saved as qvals.npy |
| **V** | State value. V = max(q) for Team 0, min(q) for Team 1 |
| **seed** | RNG seed for deal generation. Determines train/val/test split via seed % 1000 |
| **shard** | One parquet file per (seed, decl) pair. Contains all reachable game states |
| **declaration (decl)** | Trump suit choice, 0-9. See `DECL_NAMES` in schema.py |
| **Team 0 perspective** | All values from Team 0's view. Positive = Team 0 winning |

