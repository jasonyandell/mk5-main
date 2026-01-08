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

### Relationship to Game Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                     DEVELOPMENT TIME                            │
├─────────────────────────────────────────────────────────────────┤
│  TypeScript Game Engine    Oracle (Python/PyTorch)              │
│  (src/core/)               (forge/oracle/)                      │
│       │                          │                              │
│       │ Same game rules          │ GPU-parallel                 │
│       │ (authoritative)          │ (reimplemented)              │
│       ▼                          ▼                              │
│  Browser gameplay          Perfect play data                    │
│                                  │                              │
│                                  ▼                              │
│                            forge/ml/ → Trained Model (.ckpt)    │
│                                  │                              │
│                                  ▼                              │
│                            ONNX export (.onnx)                  │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RUNTIME                                  │
├─────────────────────────────────────────────────────────────────┤
│  Browser loads ONNX model → AI opponent plays in real-time      │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: The oracle **reimplements** game rules in PyTorch for GPU solving. It must match the TypeScript engine exactly, or the trained model learns wrong behavior.

## ML Pipeline Principles

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
│  - module.py                    │  DominoLightningModule + DominoTransformer
│  - data.py                      │  DominoDataModule
│  - metrics.py                   │  Q-gap, accuracy, blunder rate
│  - tokenize.py                  │  State → token conversion, split logic
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  forge/oracle/                  │  GPU tablebase solver
│  - generate.py                  │  CLI: explicit decl selection
│  - campaign.py                  │  CLI: random decl sampling (diversity)
│  - solve.py                     │  Backward induction algorithm
│  - expand.py                    │  Child state expansion (vectorized)
│  - context.py                   │  SeedContext: per-deal lookup tables
│  - state.py                     │  State packing/unpacking (GPU tensors)
│  - tables.py                    │  Game rules: suits, tricks, points
│  - declarations.py              │  Trump types (0-9) and parsing
│  - schema.py                    │  Parquet format, V/Q semantics
│  - rng.py                       │  Deterministic deal generation
│  - output.py                    │  Parquet/PT file writing utilities
│  - timer.py                     │  Per-shard timing and metrics
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  forge/bidding/                 │  Bid strength evaluation
│  - evaluate.py                  │  CLI entry point
│  - simulator.py                 │  Batched game simulation
│  - inference.py                 │  PolicyModel wrapper for fast batching
│  - estimator.py                 │  P(make) calculation, Wilson CI
│  - poster.py                    │  Visual PDF output with heatmaps
│  - parallel.py                  │  Multi-GPU cluster simulation
│  - convergence.py               │  Sample size vs accuracy analysis
│  - stability_experiment.py      │  N=200 vs N=500 variance analysis
│  - investigate.py               │  Debug losing games trick-by-trick
│  - benchmark.py                 │  Performance profiling
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  forge/scripts/                 │  Cloud training automation
│  - cloud-setup.sh               │  Environment setup on Lambda/etc
│  - cloud-run.sh                 │  End-to-end training pipeline
│  - cloud-train*.sh              │  Training job variants
│  - campaign_marginalized.py     │  Marginalized Q-value generation
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Data Storage                   │  Versioned directories
│  - data/shards-standard/        │  Standard oracle output (1 decl/seed)
│  - data/shards-marginalized/    │  Marginalized oracle output (N opp/seed)
│  - data/tokenized/              │  Preprocessed numpy arrays
│  - data/flywheel-shards/        │  Flywheel oracle output
│  - data/flywheel-tokenized/     │  Flywheel preprocessed data
│  - data/bidding-results/        │  P(make) Monte Carlo evaluations
│  - runs/                        │  Training outputs
│  - forge/models/                │  Pre-trained checkpoints
└─────────────────────────────────┘
```

---

## Data Flow

```
1. GENERATE (GPU solver - continuous mode recommended)
   python -m forge.cli.generate_continuous              # Standard mode
   python -m forge.cli.generate_continuous --marginalized  # Marginalized mode

   Output: data/shards-{standard,marginalized}/{train,val,test}/
           └── Each file: ~50k-100k game states with perfect V/Q values

2. TOKENIZE (CPU preprocessing)
   python -m forge.cli.tokenize --input data/shards-standard --output data/tokenized

   Output: data/tokenized/{train,val,test}/
           └── tokens.npy, masks.npy, targets.npy, legal.npy, qvals.npy

3. TRAIN (GPU learning)
   python -m forge.cli.train --data data/tokenized --wandb

   Output: runs/domino/version_X/
           └── checkpoints/, hparams.yaml, metrics.csv
```

Use `--help` on any command for full options.

### Generation Strategies

**Continuous generation** (recommended - runs unattended, fills gaps):
```bash
# Standard mode: 1 decl per seed (decl = seed % 10)
python -m forge.cli.generate_continuous

# Marginalized mode: 3 opp seeds per P0 hand
python -m forge.cli.generate_continuous --marginalized

# Start at specific seed (no backfill before)
python -m forge.cli.generate_continuous --start-seed 1000

# Preview what would be generated
python -m forge.cli.generate_continuous --dry-run
```

**Standard vs Marginalized** (separate experiments):
| Mode | Output Directory | Strategy | Training Goal |
|------|-----------------|----------|---------------|
| Standard | `data/shards-standard/` | 1 decl per seed | Perfect play for specific deals |
| Marginalized | `data/shards-marginalized/` | N opp seeds per P0 hand | Robust/averaged play |

**Legacy tools** (still available for debugging/one-off generation):

| Tool | Command | Use Case |
|------|---------|----------|
| **generate.py** | `python -m forge.oracle.generate` | Debugging with `--show-qvals` |
| **campaign.py** | `python -m forge.oracle.campaign` | Batch generation with random decls |

**Key insight**: Declaration diversity (100 seeds × 1 decl each) outperforms sample volume (10 seeds × 10 decls). The continuous generator uses `decl = seed % 10` for optimal coverage.

---

## Mental Models

### The Oracle as Ground Truth

The GPU solver computes **perfect play** via minimax search. For every game state, it knows:
- **V (value)**: Expected point differential if both teams play optimally
- **Q (action-values)**: Expected value after each legal move
- **Best move**: The action with highest Q (for Team 0) or lowest Q (for Team 1)

The neural network learns to approximate this oracle.

### How the Oracle Works

The solver uses **backward induction** on a complete game tree:

1. **Enumerate** all reachable game states from the initial deal (~50k states per seed/decl)
2. **Build child index**: For each state, compute which states result from each legal move
3. **Solve backwards**: Starting from terminal states (all dominoes played), propagate V/Q values up:
   - Terminal: V = 0 (no remaining points to win)
   - Non-terminal: V = max(Q) for Team 0's turn, min(Q) for Team 1's turn
   - Q[action] = V of resulting child state

Key implementation details:
- **GPU-parallel**: All states processed in batches on GPU (`expand.py`, `solve.py`)
- **SeedContext**: Precomputed lookup tables for trick resolution (`context.py`)
- **41-bit state packing**: Compact representation for GPU efficiency (`state.py`)

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

Deterministic train/val/test split by seed (see `forge.ml.tokenize.get_split()`):

| Bucket (seed % 1000) | Split | Percentage |
|---------------------|-------|------------|
| 0-899 | train | 90% |
| 900-949 | val | 5% |
| 950-999 | test | 5% - **sacred, never touched during development** |

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

## Model Inference

See [models/README.md](models/README.md) for:
- Model catalog with architecture details
- Loading checkpoints for inference
- Training configs and provenance

---

## Bidding Evaluation (System 2)

Monte Carlo simulation for P(make) estimation. Uses trained policy model to simulate games.

**Continuous generation** (recommended - runs unattended, fills gaps):
```bash
python -m forge.cli.bidding_continuous              # Run forever, N=500 samples
python -m forge.cli.bidding_continuous --start-seed 1000  # Start at seed 1000
python -m forge.cli.bidding_continuous --limit 1    # Single seed test
python -m forge.cli.bidding_continuous --dry-run    # Preview gaps
```

Output: `data/bidding-results/{train,val,test}/seed_XXXXXXXX.parquet`
- Raw points: `points_{decl}` (int8 array, N samples per declaration)
- P(make): `pmake_{decl}_{bid}` for bids 30-42
- Wilson CI: `ci_low_{decl}_{bid}`, `ci_high_{decl}_{bid}`
- Metadata: seed, hand string, n_samples, model checkpoint, timestamp

See [bidding/README.md](bidding/README.md) for:
- Module structure and key classes
- Sample size guidelines
- Single-hand evaluation CLI

See [bidding/EXAMPLES.md](bidding/EXAMPLES.md) for worked examples with analysis.

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

### Debugging with Custom Hands

When investigating whether the oracle computes correct Q-values for a specific hand, use `--p0-hand` to fix P0's hand and `--show-qvals` to display the Q-values:

```bash
# Check oracle Q-values for P0 opening with a specific hand
python -m forge.oracle.generate \
    --seed 0 \
    --decl sixes \
    --p0-hand "6-6,6-5,6-4,6-2,6-1,6-0,2-2" \
    --show-qvals \
    --out /dev/null

# Compare across multiple opponent distributions
for seed in 0 1 2 3 4; do
    python -m forge.oracle.generate \
        --seed $seed \
        --decl sixes \
        --p0-hand "6-6,6-5,6-4,6-2,6-1,6-0,2-2" \
        --show-qvals \
        --out /dev/null
done
```

The output shows each legal move with its Q-value and gap from optimal:
- **Q-value**: Expected game value (Team 0 perspective) after playing that domino
- **Δbest**: Points lost compared to the optimal move (0 = best move)

Use this to verify the oracle agrees with expert intuition about correct play.

---

## Marginalized Q-Values (Imperfect Information Training)

### The Problem

The oracle computes perfect-play Q-values for **one specific deal**. But in real play, the model doesn't know opponents' hands. When two moves have equal Q-values in one deal (e.g., 6-6 and 2-2 both show Q=+42), the model can't distinguish which is **universally optimal** vs **situationally optimal**.

Example: With a trump-heavy hand (6-6, 6-5, 6-4, 6-2, 6-1, 6-0, 2-2) calling sixes:
- 6-6 (double-six) is **always** the best lead (unbeatable highest trump)
- 2-2 **might** work if opponents can't beat it, but often fails

Training on single-deal data causes the model to learn fragile strategies.

### The Solution: Sample Multiple Opponent Distributions

Instead of one shard per P0 hand, generate **N shards** with different opponent distributions:

```bash
# Continuous marginalized generation (recommended)
python -m forge.cli.generate_continuous --marginalized

# With custom opponent seeds per P0 hand (default: 3)
python -m forge.cli.generate_continuous --marginalized --n-opp-seeds 5
```

This creates files like:
- `seed_00000000_opp0_decl_0.parquet` - Base seed 0, opponent shuffle 0
- `seed_00000000_opp1_decl_0.parquet` - Base seed 0, opponent shuffle 1
- `seed_00000000_opp2_decl_0.parquet` - Base seed 0, opponent shuffle 2

The model sees the same P0 hand with different Q-values depending on opponent distribution. Through gradient descent, it implicitly learns to prefer **robust** moves (high Q across all samples) over **fragile** ones (high Q only sometimes).

### Data Organization

Marginalized data is stored in `data/shards-marginalized/{train,val,test}/`:
- Seeds routed by `seed % 1000`: 0-899 → train, 900-949 → val, 950-999 → test
- Gap-filling: CLI automatically detects and generates missing shards
- Changing `--n-opp-seeds` backfills new opp seeds for existing base seeds

### Shard Naming

| Pattern | Meaning |
|---------|---------|
| `seed_XXXXXXXX_decl_Y.parquet` | Standard shard (seed X, decl Y) |
| `seed_XXXXXXXX_oppZ_decl_Y.parquet` | Marginalized shard (base seed X, opp seed Z, decl Y) |

The **base_seed** in the filename determines train/val/test split (via `seed % 1000`).

### References

- Paper: ["Efficiently Training NNs for Imperfect Information Games"](https://arxiv.org/abs/2407.05876) - Key finding: ~3 samples per position is sufficient
- Bead t42-elle for implementation context

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

### Checking Seed Inventory

Shards may be stored locally or on external drives. Use these patterns to find them:

```bash
# Find all shard directories (standard and marginalized)
find /mnt -name "shards-*" -type d 2>/dev/null
find data -name "shards-*" -type d 2>/dev/null

# Count files per split (adjust path as needed)
SHARD_DIR="/mnt/d/shards-standard"  # or data/shards-standard
echo "train: $(ls $SHARD_DIR/train/*.parquet 2>/dev/null | wc -l)"
echo "val: $(ls $SHARD_DIR/val/*.parquet 2>/dev/null | wc -l)"
echo "test: $(ls $SHARD_DIR/test/*.parquet 2>/dev/null | wc -l)"

# Find seed range (highest/lowest seed numbers)
ls $SHARD_DIR/*/*.parquet | grep -o 'seed_[0-9]*' | sed 's/seed_//' | sort -n | head -1  # lowest
ls $SHARD_DIR/*/*.parquet | grep -o 'seed_[0-9]*' | sed 's/seed_//' | sort -n | tail -1  # highest

# Find gaps (missing seeds in a range)
for seed in $(seq 0 999); do
  f=$(printf "$SHARD_DIR/*/seed_%08d_*.parquet" $seed)
  ls $f >/dev/null 2>&1 || echo "missing: $seed"
done

# Total disk usage
du -sh $SHARD_DIR/
```

### State Diagnosis

Before making changes, understand the current state:

```bash
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
| **declaration (decl)** | Trump suit choice, 0-9. See `DECL_ID_TO_NAME` in declarations.py |
| **Team 0 perspective** | All values from Team 0's view. Positive = Team 0 winning |
