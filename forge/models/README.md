# Crystal Forge Model Catalog

Pre-trained models for Texas 42 domino play prediction.

## Models

### Soft Cross-Entropy Models (Policy)

| Model | File | Params | Val Acc | Val Q-Gap | Date |
|-------|------|--------|---------|-----------|------|
| Large v2 (value head) | `domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt` | 817K | 97.79% | 0.072 | 2026-01-01 |
| Large v1 | `domino-large-817k-acc97.1-qgap0.11.ckpt` | 817K | 97.09% | 0.112 | 2025-12-31 |

### Q-Value Models (Value Estimation)

| Model | File | Params | Q-MAE | Val Q-Gap | Date |
|-------|------|--------|-------|-----------|------|
| Q-Val Large | `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt` | 3.3M | 0.94 | 0.071 | 2026-01-13 |
| Q-Val Small | `domino-qval-small-816k-qgap0.094-qmae1.49.ckpt` | 816K | 1.49 | 0.094 | 2026-01-13 |

## Current Best: Large v2 (value head)

**File**: `domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt`

### Architecture

Same as Large v1, with added value head:

```
DominoTransformer(
    embed_dim=128,
    n_heads=8,
    n_layers=4,
    ff_dim=512,
    dropout=0.1,
    value_head=Linear(128, 1)  # NEW: predicts game value
)
```

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 20 |
| Batch Size | 512 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 (norm) |
| Precision | bfloat16 mixed |
| Loss | Soft cross-entropy + 0.5 * value MSE |
| Temperature | 3.0 |
| Soft Weight | 0.7 |
| Value Weight | 0.5 |

### Training Data

| Split | Seeds | Declarations | Shards | Samples |
|-------|-------|--------------|--------|---------|
| Train | 0-199 | 1 per seed (seed % 10) | 200 | ~10M |
| Val | 900-904 | All 10 | 50 | ~2.5M |
| Test | 950-954 | All 10 | 50 | ~2.5M |

**Key change**: 2x more training seeds (200 vs 100). Declaration diversity strategy maintained.

### Results

| Metric | Value | Description |
|--------|-------|-------------|
| Val Accuracy | 97.79% | Model's top choice matches oracle's best move |
| Val Q-Gap | 0.072 | Mean regret in points (lower is better) |
| Val Value MAE | 7.4 pts | Mean absolute error on game value prediction |
| Val Blunder Rate | 0.15% | Moves with Q-gap > 10 points |

### Value Head Findings

The value head experiment revealed important insights:

| Finding | Evidence |
|---------|----------|
| Value head doesn't hurt policy | 97.8% acc (up from 97.1%) |
| More data helps | q_gap 0.072 vs 0.112 |
| Value prediction is hard | 7.4 MAE plateaued after epoch 9 |

**Conclusion**: Value head regression is the wrong approach for bidding. The "cliff" landscape of bid thresholds (30, 31, 32, 36, 42, 84) doesn't suit smooth MSE regression. Simulation-based bidding (System 2) is the path forward.

### Loading the Model

```python
from forge.ml.module import DominoLightningModule

# Load for inference
model = DominoLightningModule.load_from_checkpoint(
    "forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt"
)
model.eval()

# Get predictions (policy)
logits, value = model(tokens, mask, current_player)
probs = torch.softmax(logits, dim=-1)
best_move = probs.argmax(dim=-1)

# Value prediction (not recommended for bidding - use simulation instead)
predicted_value = value * 42  # denormalize to points
```

### Provenance

- **Bead**: t42-1e1y (value head experiment)
- **Wandb**: crystal-forge v2-valuehead run
- **Git commit**: (see git log)

---

## Large v1

**File**: `domino-large-817k-acc97.1-qgap0.11.ckpt`

### Architecture

```
DominoTransformer(
    embed_dim=128,
    n_heads=8,
    n_layers=4,
    ff_dim=512,
    dropout=0.1
)
```

| Component | Value |
|-----------|-------|
| Parameters | 816,775 |
| Layers | 4 |
| Attention Heads | 8 |
| Embedding Dim | 128 |
| Feed-Forward Dim | 512 |
| Dropout | 0.1 |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 20 |
| Batch Size | 512 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 (norm) |
| Precision | bfloat16 mixed |
| Loss | Soft cross-entropy |
| Temperature | 3.0 |
| Soft Weight | 0.7 |

### Training Data

| Split | Seeds | Declarations | Shards | Samples |
|-------|-------|--------------|--------|---------|
| Train | 0-99 | 1 per seed (seed % 10) | 100 | 5.0M |
| Val | 900-909 | All 10 | 100 | 5.0M |
| Test | 950-959 | All 10 | 100 | 5.0M |

**Key insight**: Declaration diversity (100 seeds × 1 decl) outperformed sample volume (10 seeds × 10 decls). Each declaration has 10 unique seeds for balanced coverage.

### Results

| Metric | Value | Description |
|--------|-------|-------------|
| Val Accuracy | 97.09% | Model's top choice matches oracle's best move |
| Val Q-Gap | 0.112 | Mean regret in points (lower is better) |
| Val Blunder Rate | 0.33% | Moves with Q-gap > 10 points |

### Hardware

- **GPU**: NVIDIA H100 80GB HBM3 (Lambda Labs)
- **Training Time**: ~60 minutes
- **torch.compile**: Disabled (system package conflicts)

### Comparison to Prior Work

| Model | Params | Val Acc | Val Q-Gap | Notes |
|-------|--------|---------|-----------|-------|
| Prior best (solver2) | 73K | 94.6% | — | Plateau, limited diversity |
| Baseline v1 | 73K | 93.6% | 0.367 | Same arch, more diversity |
| Medium v1 | 275K | 95.8% | 0.198 | |
| Large v1 | 817K | 97.1% | 0.112 | 100 seeds |
| **Large v2** | **817K** | **97.8%** | **0.072** | 200 seeds, value head |

### Reproducing This Model

```bash
# 1. Generate training data
python -m forge.oracle.generate --seed-range 0:100 --decl all --out data/shards  # train
python -m forge.oracle.generate --seed-range 900:910 --decl all --out data/shards  # val
python -m forge.oracle.generate --seed-range 950:960 --decl all --out data/shards  # test

# 2. Tokenize
python -m forge.cli.tokenize --input data/shards --output data/tokenized

# 3. Train
python -m forge.cli.train \
    --epochs 20 \
    --embed-dim 128 --n-heads 8 --n-layers 4 --ff-dim 512 \
    --precision bf16-mixed \
    --no-compile \
    --wandb
```

### Loading the Model

```python
from forge.ml.module import DominoLightningModule

# Load for inference
model = DominoLightningModule.load_from_checkpoint(
    "forge/models/domino-large-817k-acc97.1-qgap0.11.ckpt"
)
model.eval()

# Get predictions
logits = model(tokens, mask)  # tokens: [B, 32, 12], mask: [B, 7]
probs = torch.softmax(logits, dim=-1)
best_move = probs.argmax(dim=-1)
```

### File Sizes

| Format | Size | Use Case |
|--------|------|----------|
| Checkpoint (.ckpt) | 9.5 MB | Resume training, full reproducibility |
| Weights only (.pt) | 3.1 MB | PyTorch inference |
| ONNX (.onnx) | ~3 MB | Browser/mobile inference |
| ONNX quantized (int8) | ~800 KB | Mobile-optimized |

### Provenance

- **Bead**: t42-ep5j (closed)
- **Wandb**: https://wandb.ai/jasonyandell-forge42/crystal-forge
- **Git commit**: (see git log)

---

## Q-Value Models

A new family of models trained directly on Q-value regression instead of soft cross-entropy. These models predict Q-values for each action rather than matching oracle move distributions.

| Model | File | Params | q_gap | q_mae | value_mae | Accuracy | Date |
|-------|------|--------|-------|-------|-----------|----------|------|
| Q-Val Large | `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt` | 3.3M | 0.071 | 0.94 | 1.89 | 79.0% | 2026-01-13 |
| Q-Val Small | `domino-qval-small-816k-qgap0.094-qmae1.49.ckpt` | 816K | 0.094 | 1.49 | 2.39 | 74.7% | 2026-01-13 |

### Current Best: Q-Val Large

**File**: `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`

#### Architecture

```
DominoTransformer(
    embed_dim=256,
    n_heads=8,
    n_layers=6,
    ff_dim=512,
    dropout=0.1,
    value_head=Linear(256, 1)
)
```

| Component | Value |
|-----------|-------|
| Parameters | 3.3M |
| Layers | 6 |
| Attention Heads | 8 |
| Embedding Dim | 256 |
| Feed-Forward Dim | 512 |
| Dropout | 0.1 |

#### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 20 |
| Batch Size | 512 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 (norm) |
| Precision | 16-mixed AMP |
| **Loss** | **Q-value MSE** (not soft CE) |
| Q-value range | [-1, 1] (normalized from 0-42 pts) |

#### Training Data

Same as Large v2: 200 seeds × 1 declaration per seed, ~10M samples.

#### Results

| Metric | Value | Description |
|--------|-------|-------------|
| q_gap | 0.071 | Mean regret in points vs oracle |
| q_mae | 0.94 | Mean absolute error on Q-value prediction (points) |
| value_mae | 1.89 | Mean absolute error on game value prediction (points) |
| Accuracy | 79.0% | Model's argmax Q matches oracle's best move |
| Blunder Rate | 0.23% | Moves with Q-gap > 10 points |

#### Loading the Model

```python
from forge.ml.module import DominoLightningModule

# Load for inference
model = DominoLightningModule.load_from_checkpoint(
    "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
)
model.eval()

# Get Q-value predictions
q_values, value = model(tokens, mask, current_player)
# q_values: [B, num_actions] in [-1, 1], multiply by 42 for points
best_move = q_values.argmax(dim=-1)
```

#### Provenance

- **Bead**: t42-dove (Q-value training experiment), t42-mnaz (model promotion)
- **Wandb**: qval-202601120501-3.3M-6L-8H-d256
- **Run dir**: runs/domino/version_38

---

### Q-Val Small

**File**: `domino-qval-small-816k-qgap0.094-qmae1.49.ckpt`

#### Architecture

```
DominoTransformer(
    embed_dim=128,
    n_heads=8,
    n_layers=4,
    ff_dim=512,
    dropout=0.1,
    value_head=Linear(128, 1)
)
```

| Component | Value |
|-----------|-------|
| Parameters | 816K |
| Layers | 4 |
| Attention Heads | 8 |
| Embedding Dim | 128 |
| Feed-Forward Dim | 512 |

#### Results

| Metric | Value |
|--------|-------|
| q_gap | 0.094 |
| q_mae | 1.49 |
| value_mae | 2.39 |
| Accuracy | 74.7% |
| Blunder Rate | 0.28% |

#### Provenance

- **Bead**: t42-dove, t42-mnaz
- **Wandb**: qval-202601120105-816K-4L-8H-d128
- **Run dir**: runs/domino/version_37

---

### Q-Value vs Soft Cross-Entropy

| Aspect | Soft CE Models | Q-Value Models |
|--------|----------------|----------------|
| Loss | KL divergence from oracle distribution | MSE on Q-values |
| Output | Logits → softmax probabilities | Direct Q-value estimates |
| Accuracy metric | 97%+ (matches oracle's top choice) | 74-79% (argmax Q matches) |
| Use case | Move selection via sampling | Direct value estimation |
| q_gap | 0.07-0.11 | 0.07-0.09 |

The lower "accuracy" for Q-value models is expected—they're optimizing for accurate Q-values across all actions, not just getting the top action right.

### Best Model for E[Q] Marginalization

For the E[Q] pipeline (`forge/eq/`), **Q-value models are preferred** because they produce directly interpretable marginals:

```python
# Logit model: arbitrary scale, needs softmax interpretation
e_logits = [-0.72, -1.09, -1.87, ...]  # What does -0.72 mean?

# Q-value model: expected points, directly interpretable
e_q = [-17.89, -21.05, -21.45, ...]  # "-17.89 points expected"
```

When debugging E[Q] decisions, Q-value outputs make it easy to verify values are in valid game range (±42 points) and understand the strategic implications of each choice.

---

## Metrics Glossary

- **Accuracy**: Fraction of moves where model's argmax matches oracle's best move
- **Q-Gap**: Expected point loss per move vs oracle (oracle_best_q - oracle_q[model_choice])
- **Q-MAE**: Mean absolute error of predicted Q-values vs oracle Q-values (in points)
- **Value-MAE**: Mean absolute error of predicted game value vs oracle value (in points)
- **Blunder Rate**: Fraction of moves with Q-gap > 10 (catastrophic mistakes)
- **Oracle**: GPU minimax solver with perfect play (ground truth)
