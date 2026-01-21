---
name: texas-42-forge
description: Crystal Forge ML pipeline for Texas 42 oracle training. Use when generating oracle shards, tokenizing data, training transformer models, evaluating checkpoints, running Modal GPU jobs, promoting models to the catalog, or generating E[Q] Stage 2 training data. Covers the full pipeline from GPU solver to trained model.
---

# Crystal Forge ML Pipeline

**CRITICAL**: Invoke `pytorch-lightning` skill for training patterns, `pytorch` skill for model architecture, and `modal` skill for cloud GPU jobs. These provide essential guidance this skill builds upon.

## Quick Start

```bash
# Verify setup
python -c "from forge.ml import module, data, metrics; print('OK')"

# Quick training test
python -m forge.cli.train --fast-dev-run --no-wandb

# Full training with wandb
python -m forge.cli.train --data ../data/tokenized-full --batch-size 4096 --epochs 20 --wandb
```

## Core Architecture

```
ORACLE SHARDS → TOKENIZED DATA → TRAINED MODEL → ONNX EXPORT
```

The oracle (GPU minimax solver) generates perfect-play data. The model learns to imitate it.

**Team 0 Perspective**: All V/Q values from Team 0's viewpoint (positive = Team 0 ahead).

## Common Workflows

### Workflow 1: Train a Model

```
Task Progress:
- [ ] Step 1: Verify tokenized data exists
- [ ] Step 2: Run training with wandb
- [ ] Step 3: Monitor metrics (q_gap, accuracy, blunder_rate)
- [ ] Step 4: Evaluate on test set
```

**Step 1: Verify tokenized data**

```bash
# Check data exists
ls data/tokenized-full/{train,val,test}/tokens.npy
```

**Step 2: Run training**

```bash
python -m forge.cli.train \
    --data data/tokenized-full \
    --batch-size 4096 \
    --epochs 20 \
    --precision bf16-mixed \
    --wandb
```

**Step 3: Monitor metrics**

Target values during training:

| Metric | Target | Meaning |
|--------|--------|---------|
| `q_gap` | < 0.10 | Points lost vs oracle per move |
| `q_mae` | < 3.0 | Q-value prediction error (points) |
| `blunder_rate` | < 1% | Moves with q_gap > 10 |

**Step 4: Evaluate**

```bash
python -m forge.cli.eval --checkpoint runs/domino/version_X/checkpoints/best.ckpt
```

### Workflow 2: Generate Oracle Data

```
Task Progress:
- [ ] Step 1: Run continuous generation
- [ ] Step 2: Monitor progress
- [ ] Step 3: Tokenize shards
```

**Step 1: Generate**

```bash
# Standard mode (1 decl per seed)
python -m forge.cli.generate_continuous

# Marginalized mode (3 opp distributions per P0 hand)
python -m forge.cli.generate_continuous --marginalized

# Preview gaps
python -m forge.cli.generate_continuous --dry-run
```

**Step 2: Monitor**

```bash
# Count shards
ls data/shards-standard/train/*.parquet | wc -l
```

**Step 3: Tokenize**

```bash
python -m forge.cli.tokenize_data --input data/shards-standard --output data/tokenized
```

### Workflow 3: Modal GPU Jobs

```
Task Progress:
- [ ] Step 1: Authenticate with Modal
- [ ] Step 2: Run job
- [ ] Step 3: Monitor and cost-track
- [ ] Step 4: Download results
```

**Step 1: Auth**

```bash
modal token new  # Opens browser
```

**Step 2: Run**

```bash
modal run forge/modal_app.py::generate_valtest
```

**Step 3: Monitor**

```bash
# Check running jobs
modal app list

# Container count
modal container list --json | jq length

# CRITICAL: Ctrl+C does NOT stop Modal jobs!
# To stop: modal app stop ap-XXXXXXXXXXXX
```

**Step 4: Download**

```bash
modal run forge/modal_app.py::download --split val --output-dir data/shards-marginalized
```

### Workflow 4: Promote Model to Catalog

```
Task Progress:
- [ ] Step 1: Name using convention
- [ ] Step 2: Copy to forge/models/
- [ ] Step 3: Update forge/models/README.md
- [ ] Step 4: Track with beads
```

**Step 1: Naming convention**

```
domino-{type}-{size}-{params}-{key-metrics}.ckpt

# Examples:
domino-large-817k-acc97.8-qgap0.07.ckpt      # Policy model
domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt  # Q-value model
```

**Step 2: Copy**

```bash
cp runs/domino/version_XX/checkpoints/epoch=*.ckpt \
   forge/models/domino-{name}.ckpt
```

**Step 3: Document in README**

Add to table in `forge/models/README.md` with architecture, config, results, provenance (bead ID, wandb run).

**Step 4: Track**

```bash
bd create --title="Promote {model} to catalog" --type=task
bd close {id} --reason="Promoted with documentation"
```

### Workflow 5: Generate E[Q] Stage 2 Data

Stage 2 trains on imperfect information - it learns E[Q] by sampling possible opponent hands.

```
Task Progress:
- [ ] Step 1: Verify Stage 1 checkpoint exists
- [ ] Step 2: Run GPU generation (continuous)
- [ ] Step 3: Monitor output and gaps
- [ ] Step 4: (Optional) Enable posterior weighting
```

**Step 1: Verify checkpoint**

```bash
ls forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt
```

**Step 2: Generate**

```bash
# Preview what would be generated
python -m forge.cli.generate_eq_continuous \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
    --dry-run

# Generate training data (GPU, ~100x faster than CPU)
python -m forge.cli.generate_eq_continuous \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt

# Generate specific seed range
python -m forge.cli.generate_eq_continuous \
    --checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt \
    --start-seed 1000 --limit 500
```

**Step 3: Monitor**

```bash
# Count generated files per split (90/5/5 split, same as Stage 1)
ls data/eq-games/train/*.pt | wc -l
ls data/eq-games/val/*.pt | wc -l
ls data/eq-games/test/*.pt | wc -l
```

**Step 4: Posterior weighting (optional)**

Improves E[Q] estimates mid-game by reweighting worlds based on transcript likelihood:

```bash
python -m forge.cli.generate_eq_continuous \
    --checkpoint model.ckpt \
    --posterior --posterior-window 4
```

**Output format**: Per-seed `.pt` files with `e_q_mean`, `e_q_var`, `legal_mask`, `transcript_tokens`, etc.

**Performance**: RTX 3050 Ti, batch=32, samples=50 → ~300k examples/hour

See [docs/EQ_STAGE2_TRAINING.md](docs/EQ_STAGE2_TRAINING.md) for full spec.

## Key Metrics

| Metric | What it Measures | Good Value |
|--------|------------------|------------|
| **q_gap** | Points lost per move vs oracle | < 0.10 |
| **q_mae** | Q-value prediction error | < 3.0 |
| **value_mae** | Game value prediction error | < 3.0 |
| **accuracy** | Top-1 match with oracle | ~97% (policy), ~75% (Q-value) |
| **blunder_rate** | Moves with q_gap > 10 | < 1% |

## Model Types

| Type | Loss | Output | Use Case |
|------|------|--------|----------|
| **Policy (Soft CE)** | KL divergence | Action probabilities | Move selection via sampling |
| **Q-Value (MSE)** | Q-value MSE | Direct Q estimates | Value-based selection |

**Loading models**:

```python
from forge.ml.module import DominoLightningModule

model = DominoLightningModule.load_from_checkpoint("forge/models/xxx.ckpt")
model.eval()

# Policy model
logits, value = model(tokens, mask, current_player)
best_move = torch.softmax(logits, dim=-1).argmax(dim=-1)

# Q-value model
q_values, value = model(tokens, mask, current_player)
best_move = q_values.argmax(dim=-1)
```

## Common Issues

**Issue: Low GPU utilization during training**

Increase batch size. Use AMP precision:
```bash
--batch-size 4096 --precision bf16-mixed
```

**Issue: Model not learning (q_gap not decreasing)**

Check data loading, verify tokens.npy exists, check num_workers > 0.

**Issue: Modal jobs keep running after Ctrl+C**

```bash
modal app list  # Find app ID
modal app stop ap-XXXXXXXXXXXX
```

**Issue: OOM on large seeds**

Reduce batch size or use `--max-states 500000000` for oracle generation.

## GPU Recommendations

| Hardware | batch_size | precision | Notes |
|----------|------------|-----------|-------|
| RTX 3060 | 4096 | 16-mixed | Local dev |
| A10 | 4096-8192 | bf16-mixed | Modal standard |
| H100/H200 | 8192+ | bf16-mixed | Fast training |

## Advanced Topics

**Marginalized training**: See [forge/ORIENTATION.md](forge/ORIENTATION.md) § "Marginalized Q-Values"
**E[Q] pipeline (Stage 2)**: See [forge/ORIENTATION.md](forge/ORIENTATION.md) § "E[Q] Training Pipeline" and [docs/EQ_STAGE2_TRAINING.md](docs/EQ_STAGE2_TRAINING.md) for full research spec
**Bidding evaluation**: See [forge/bidding/README.md](forge/bidding/README.md)
**Modal optimization**: See [forge/MODAL_ORIENTATION.md](forge/MODAL_ORIENTATION.md)
**Modal monitoring**: See [forge/MODAL_MONITOR.md](forge/MODAL_MONITOR.md)

## File Map

```
forge/
├── cli/                         # User commands
│   ├── train.py                # Training entry point
│   ├── eval.py                 # Checkpoint evaluation
│   ├── tokenize_data.py        # Shard → tokenized
│   ├── generate_continuous.py  # Oracle generation (Stage 1)
│   └── generate_eq_continuous.py  # E[Q] generation (Stage 2)
├── ml/                          # Lightning core (Stage 1)
│   ├── module.py               # DominoLightningModule + DominoTransformer
│   ├── data.py                 # DominoDataModule
│   └── metrics.py              # q_gap, accuracy, blunder
├── eq/                          # E[Q] pipeline (Stage 2)
│   ├── generate_gpu.py         # GPU generation (~100x faster)
│   ├── generate_pipelined.py   # Async pipelined generation
│   ├── sampling_gpu.py         # GPU world sampling
│   ├── posterior_gpu.py        # GPU posterior weighting
│   ├── oracle.py               # Stage1Oracle wrapper
│   ├── transcript_tokenize.py  # Public-info tokenizer
│   └── viewer.py               # Debug viewer
├── oracle/                      # GPU solver
│   ├── solve.py                # Backward induction
│   └── schema.py               # Parquet format
├── models/                      # Promoted checkpoints
│   └── README.md               # Model catalog
└── modal_app.py                # Modal GPU jobs
```

## Resources

- Model catalog: [forge/models/README.md](forge/models/README.md)
- Full orientation: [forge/ORIENTATION.md](forge/ORIENTATION.md)
- Game rules: [docs/rules.md](docs/rules.md)
