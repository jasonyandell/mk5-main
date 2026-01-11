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
│  - tokenize_data.py             │  Preprocess shards
│  - flywheel.py                  │  Iterative training
│  - generate_continuous.py       │  Continuous oracle generation
│  - bidding_continuous.py        │  Continuous P(make) evaluation
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
│  - investigate.py               │  Debug losing games trick-by-trick
│  - benchmark.py                 │  Performance profiling
│  - schema.py                    │  Parquet schema for bidding results
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  forge/flywheel/                │  Iterative fine-tuning pipeline
│  - state.yaml                   │  State machine configuration
│  - RUNBOOK.md                   │  Operational instructions
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
│  Data Storage                   │  Actual locations
│                                 │
│  External drive (/mnt/d/):      │
│  - shards-standard/             │  215GB, 1124 train parquet files
│  - shards-marginalized/         │  191GB, marginalized oracle output
│                                 │
│  Local (mk5-tailwind/data/):    │
│  - tokenized-full/              │  5GB, 11.2M samples (ready to use)
│  - bidding-results/             │  P(make) Monte Carlo evaluations
│                                 │
│  Forge-relative:                │
│  - runs/                        │  Training outputs
│  - forge/models/                │  Pre-trained checkpoints
│  - forge/data/tokenized/        │  Small test tokenized data
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
   python -m forge.cli.tokenize_data --input data/shards-standard --output data/tokenized

   Output: data/tokenized/{train,val,test}/
           └── tokens.npy, masks.npy, targets.npy, legal.npy, qvals.npy, etc.

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
   - Q[action] = V of resulting child state + reward for trick completion

Key implementation details:
- **GPU-parallel**: All states processed in batches on GPU (`expand.py`, `solve.py`)
- **SeedContext**: Precomputed lookup tables for trick resolution (`context.py`)
- **41-bit state packing**: Compact representation for GPU efficiency (`state.py`)

### Team 0 Perspective (Critical!)

All values are from Team 0's viewpoint:
- **Positive V** = Team 0 is ahead
- **Negative V** = Team 1 is ahead
- **Team 0 maximizes Q**, Team 1 minimizes Q

### V/Q Semantics (Critical!)

**V is value-to-go**, not cumulative score:
- V represents expected remaining (Team 0 − Team 1) point differential from this state to end
- Terminal states have V = 0 (no remaining points to win)
- At initial state, V equals the final hand point differential

**Q-values** follow the same semantics:
- Q[action] = expected value-to-go after playing that action
- All Q-values from Team 0's perspective regardless of whose turn
- -128 indicates illegal move

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
| `V` | int8 | Value-to-go: expected remaining point differential (Team 0 perspective) |
| `q0`-`q6` | int8 | Q-values for each action (-128 = illegal) |

### State Packing (41 bits)

```
Bits 0-27:   Remaining hands (4 × 7-bit masks)
   [0:7]    P0's local indices still in hand
   [7:14]   P1's local indices
   [14:21]  P2's local indices
   [21:28]  P3's local indices

Bits 28-29:  leader (2 bits) - current trick leader (0-3)
Bits 30-31:  trick_len (2 bits) - plays so far (0-3)

Bits 32-40:  Current trick plays (3 × 3 bits)
   [32:35]  p0 - leader's local index (0-6, or 7 if N/A)
   [35:38]  p1 - 2nd player's local index
   [38:41]  p2 - 3rd player's local index
```

**Why local indices?** State encoding is deal-independent; same bit layout for all seeds.

### Token Format

Neural network input is `int8[batch, 32, 12]`:
- **32 positions**: 7 hand + 21 (4×7 domino positions) + 4 (trick/metadata)
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

### Oracle Generation

```bash
# Continuous generation (recommended)
python -m forge.cli.generate_continuous           # Standard mode
python -m forge.cli.generate_continuous --marginalized  # Marginalized mode
python -m forge.cli.generate_continuous --dry-run      # Preview gaps

# Single-seed debugging with Q-value inspection
python -m forge.oracle.generate --seed 0 --decl sixes --show-qvals --out /dev/null
```

### Tokenization

```bash
python -m forge.cli.tokenize_data --input data/shards-standard --output data/tokenized
python -m forge.cli.tokenize_data --dry-run  # Preview file counts
```

### Training

```bash
# Quick sanity check
python -m forge.cli.train --fast-dev-run --no-wandb

# Full training with Wandb (use full tokenized dataset, BS=4096-8192 for RTX 3060)
python -m forge.cli.train --data ../data/tokenized-full --batch-size 4096 --epochs 20 --wandb

# Large model with A100 precision
python -m forge.cli.train --precision bf16-mixed --n-layers 4 --n-heads 8 --embed-dim 128
```

### Evaluation

```bash
# Evaluate checkpoint on test set
python -m forge.cli.eval --checkpoint runs/domino/version_0/checkpoints/best.ckpt
```

### Bidding Evaluation

```bash
# Single hand evaluation
python -m forge.bidding.evaluate --hand "6-4,5-5,4-2,3-1,2-0,1-1,0-0" --samples 100

# Continuous evaluation (fills gaps)
python -m forge.cli.bidding_continuous
python -m forge.cli.bidding_continuous --dry-run  # Preview gaps
```

### Flywheel (Iterative Training)

```bash
python -m forge.cli.flywheel status   # Check current state
python -m forge.cli.flywheel --once   # Run one iteration
python -m forge.cli.flywheel          # Run continuously
```

Training auto-detects hardware. Use `--precision bf16-mixed` for A100/H100 servers.

---

## Model Catalog

See [models/README.md](models/README.md) for complete details.

| Model | File | Params | Val Acc | Val Q-Gap |
|-------|------|--------|---------|-----------|
| Large v2 (value head) | `domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt` | 817K | 97.79% | 0.072 |
| Large v1 | `domino-large-817k-acc97.1-qgap0.11.ckpt` | 817K | 97.09% | 0.112 |

**Current best architecture**:
```python
DominoTransformer(
    embed_dim=128,
    n_heads=8,
    n_layers=4,
    ff_dim=512,
    dropout=0.1,
)
```

**Loading for inference**:
```python
from forge.ml.module import DominoLightningModule

model = DominoLightningModule.load_from_checkpoint(
    "forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt"
)
model.eval()

# Get predictions
logits, value = model(tokens, mask, current_player)
probs = torch.softmax(logits, dim=-1)
best_move = probs.argmax(dim=-1)
```

---

## Bidding Evaluation (System 2)

Monte Carlo simulation for P(make) estimation. Uses trained policy model to simulate games.

**Why not use the value head?** The value head predicts smooth game values, but bidding thresholds (30, 31, 32, 36, 42) create a "cliff" landscape that doesn't suit MSE regression. Simulation naturally handles this.

**Continuous generation** (recommended):
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

**Sample size guidelines**:

| Use Case | N | CI Width | Time/Trump |
|----------|---|----------|------------|
| Quick screening | 50 | ±0.15 | ~7s |
| Standard analysis | 100 | ±0.10 | ~14s |
| Publication quality | 200 | ±0.07 | ~28s |
| Ground truth | 500 | ±0.04 | ~70s |

See [bidding/README.md](bidding/README.md) for module structure and examples.

---

## Flywheel: Iterative Fine-Tuning

Automates iterative model improvement: generate new seeds → train → evaluate → repeat.

**State machine**:
```
ready ──run──> running ──success──> ready (next iteration)
                  │
                  └──failure──> failed
```

**Per-iteration workflow**:
1. Ensure golden val/test shards exist (seeds 900-902 for val, 950 for test)
2. Generate oracle shards for current seed range
3. Tokenize all data
4. Fine-tune from previous checkpoint (2 epochs, LR=3e-5)
5. Evaluate against golden benchmarks
6. Log metrics to W&B with lineage tracking

**Commands**:
```bash
python -m forge.cli.flywheel init --wandb-group my-experiment --start-seed 200
python -m forge.cli.flywheel status
python -m forge.cli.flywheel --once   # One iteration
python -m forge.cli.flywheel          # Run continuously
```

See [flywheel/RUNBOOK.md](flywheel/RUNBOOK.md) for full operations guide.

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

---

## E[Q] Training Pipeline (Stage 2)

### The Problem

Stage 1 (the oracle) sees all 4 hands - it "cheats." In real play, you only see your own hand and the play history. We need a model that makes decisions from **imperfect information** but plays as well as if it could see everything.

### The Solution: Two-Stage Distillation

```
Stage 1 Oracle (sees all hands, trained)
        ↓
   Sample 100 possible opponent hands
        ↓
   Query oracle for each → Average logits → E[Q]
        ↓
Stage 2 Model (sees transcript only, learns to predict E[Q])
        ↓
   At runtime: single forward pass, no sampling
```

Stage 2 **distills** the expensive sampling process into a fast neural net.

### Directory Structure

```
forge/eq/
├── voids.py               # Infer void suits from play history
├── sampling.py            # Backtracking sampler (guaranteed valid hands)
├── oracle.py              # Stage1Oracle wrapper for batch queries
├── game.py                # GameState for simulation
├── generate.py            # Generate single game with 28 decisions
├── generate_dataset.py    # Batch generation with train/val split
├── transcript_tokenize.py # Stage 2 tokenizer (public info only)
├── viewer.py              # Interactive debug viewer
├── stage2.py              # Stage 2 model (TODO)
└── train_stage2.py        # Training loop (TODO)
```

### Data Format

Each training example is one **decision point**:

| Field | Shape | Description |
|-------|-------|-------------|
| `transcript_tokens` | (36, 8) | Padded sequence: [decl, hand..., plays...] |
| `transcript_lengths` | scalar | Actual sequence length |
| `e_logits` | (7,) | Target: averaged Q-values across sampled worlds |
| `legal_mask` | (7,) | Which actions were legal |
| `action_taken` | scalar | Which slot the E[Q] policy chose |

### Key Commands

```bash
# Generate training data (1000 games, ~8 min on 3050 Ti)
PYTHONPATH=. python -m forge.eq.generate_dataset \
    --n-games 1000 \
    --output forge/data/eq_dataset.pt

# Interactive viewer for human evaluation
PYTHONPATH=. python -m forge.eq.viewer forge/data/eq_dataset.pt

# Jump to specific game
PYTHONPATH=. python -m forge.eq.viewer forge/data/eq_dataset.pt --game 42
```

### Viewer Controls

- `←` / `→` - Navigate decisions
- `j` - Jump to example number
- `q` - Quit

### Data Storage

- `forge/data/eq_dataset.pt` - Generated training data (gitignored, ~66 MB for 1000 games)
- `scratch/eq_test.pt` - Small test datasets

### The Backtracking Sampler

Unlike rejection sampling (which fails with tight void constraints), the backtracking sampler uses **MRV heuristic** (Minimum Remaining Values) to guarantee finding valid opponent hand distributions:

1. Build candidate sets per opponent (dominoes respecting void constraints)
2. Always assign to player with minimum slack first
3. Early pruning: if slack < 0, backtrack immediately
4. Guaranteed to find solution if one exists (the real game state is always valid)

Ported from `src/game/ai/hand-sampler.ts`.

---

## Debugging Tips

| Issue | Where to Look |
|-------|---------------|
| Model not learning | Check Q-gap trend, verify data loading |
| OOM errors | Reduce batch size, reduce chunk sizes in generate |
| Slow training | Check num_workers, enable pin_memory |
| Checkpoint won't load | Check hparams match, verify Lightning version |
| Inconsistent results | Check RNG seeds, verify deterministic mode |

### Quick Diagnostics

```bash
# Verify imports work
python -c "from forge.ml import module, data, metrics; print('OK')"

# Fast training test
python -m forge.cli.train --fast-dev-run --no-wandb

# Check GPU available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
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

### Debugging Bidding Simulation

When a near-certain hand loses unexpectedly:

```bash
# Find and replay losing games trick-by-trick
python -m forge.bidding.investigate \
    --hand "6-6,6-5,6-4,6-2,6-1,6-0,2-2" \
    --trump sixes \
    --below 42 \
    --samples 500
```

---

## Architectural Invariants

1. **Team 0 Perspective**: All V/Q values from Team 0's viewpoint
2. **V is Value-to-Go**: Not cumulative score; terminal states have V=0
3. **Deterministic Splits**: seed % 1000 determines train/val/test
4. **Per-Shard RNG**: Sampling keyed by (global_seed, base_seed, decl_id)
5. **Lightning Structure**: LightningModule for training, DataModule for data
6. **No Models in data/**: Checkpoints go to runs/ or forge/models/

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
SHARD_DIR="data/shards-standard"  # or /mnt/d/shards-standard
echo "train: $(ls $SHARD_DIR/train/*.parquet 2>/dev/null | wc -l)"
echo "val: $(ls $SHARD_DIR/val/*.parquet 2>/dev/null | wc -l)"
echo "test: $(ls $SHARD_DIR/test/*.parquet 2>/dev/null | wc -l)"

# Find seed range (highest/lowest seed numbers)
ls $SHARD_DIR/*/*.parquet | grep -o 'seed_[0-9]*' | sed 's/seed_//' | sort -n | head -1  # lowest
ls $SHARD_DIR/*/*.parquet | grep -o 'seed_[0-9]*' | sed 's/seed_//' | sort -n | tail -1  # highest

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

# Flywheel status?
python -m forge.cli.flywheel status 2>/dev/null || echo "Flywheel not initialized"

# What's currently running?
pgrep -af "forge.oracle\|forge.cli" || echo "Nothing running"
```

### Change Impact

| If you modify... | Regenerate... | Invalidates... |
|------------------|---------------|----------------|
| `forge/oracle/*.py` | shards | data/shards-*, data/tokenized/ |
| `forge/ml/tokenize.py` | tokenized | data/tokenized/ |
| `forge/ml/module.py` | train | runs/ (retrain needed) |
| `forge/ml/metrics.py` | nothing | (metrics-only change) |
| `forge/bidding/*.py` | bidding results | data/bidding-results/ |

### Verification Commands

```bash
# After any forge/ changes - verify imports
python -c "from forge.ml import module, data, metrics; from forge.oracle import schema; print('OK')"

# After model changes - quick training test
python -m forge.cli.train --fast-dev-run --no-wandb

# After bidding changes - quick simulation test
python -m forge.bidding.evaluate --hand "6-4,5-5,4-2,3-1,2-0,1-1,0-0" --samples 10
```

---

## Glossary

| Term | Definition |
|------|------------|
| **q0-q6** | Q-values: optimal value-to-go of each action. Parquet columns, saved as qvals.npy |
| **V** | State value-to-go. V = max(q) for Team 0, min(q) for Team 1. Terminal states = 0 |
| **seed** | RNG seed for deal generation. Determines train/val/test split via seed % 1000 |
| **shard** | One parquet file per (seed, decl) pair. Contains all reachable game states |
| **declaration (decl)** | Trump suit choice, 0-9. See `DECL_ID_TO_NAME` in declarations.py |
| **Team 0 perspective** | All values from Team 0's view. Positive = Team 0 winning |
| **Q-gap** | Oracle regret: optimal_q - predicted_q. Lower is better |
| **Blunder** | A move with Q-gap > 10 points. Catastrophic mistake |
| **P(make)** | Probability of making a bid, estimated via Monte Carlo simulation |
| **Wilson CI** | Confidence interval for P(make) using Wilson score method |
| **Marginalized** | Training with multiple opponent distributions per P0 hand |
| **Flywheel** | Iterative fine-tuning loop: generate → train → evaluate → repeat |
