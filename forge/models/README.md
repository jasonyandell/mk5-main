# Crystal Forge Model Catalog

Pre-trained models for Texas 42 domino play prediction.

## Models

| Model | File | Params | Val Acc | Val Q-Gap | Date |
|-------|------|--------|---------|-----------|------|
| Large v1 | `domino-large-817k-acc97.1-qgap0.11.ckpt` | 817K | 97.09% | 0.112 | 2025-12-31 |

## Current Best: Large v1

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
| **Large v1** | **817K** | **97.1%** | **0.112** | Current best |

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

## Metrics Glossary

- **Accuracy**: Fraction of moves where model's argmax matches oracle's best move
- **Q-Gap**: Expected point loss per move vs oracle (oracle_best_q - oracle_q[model_choice])
- **Blunder Rate**: Fraction of moves with Q-gap > 10 (catastrophic mistakes)
- **Oracle**: GPU minimax solver with perfect play (ground truth)
