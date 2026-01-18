# Crystal Forge Model Catalog

Pre-trained models for Texas 42 domino play prediction.

## Recommended Models

**For most use cases, use Q-value models.** They output expected points directly, making outputs interpretable and debugging straightforward.

| Model | File | Params | Q-MAE | Q-Gap | Use Case |
|-------|------|--------|-------|-------|----------|
| **Q-Val Large** | `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt` | 3.3M | 0.94 | 0.071 | Best accuracy |
| Q-Val Small | `domino-qval-small-816k-qgap0.094-qmae1.49.ckpt` | 816K | 1.49 | 0.094 | Faster inference |

For E[Q] marginalization (`forge/eq/`), Q-value models are strongly preferred—see [Why Q-Value Models?](#why-q-value-models) below.

---

## Q-Value Models

Models trained with `loss_mode='qvalue'` that directly predict expected points for each action.

### Training Data (Validated)

All Q-value models trained on `data/tokenized-full`:

| Split | Seed Range | Shards | Samples |
|-------|------------|--------|---------|
| Train | 0-1223 (excl. 900-999) | 1,124 | 11.24M |
| Val | 900-949 | 50 | 500K |
| Test | 950-999 | 50 | 500K |

**Source**: `/mnt/d/shards-standard/` → `data/tokenized-full/manifest.yaml`

Each shard uses 1 declaration per seed (`decl = seed % 10`) for balanced coverage across all 10 declarations.

### Q-Val Large (Recommended)

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

#### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 19 (best checkpoint) |
| Batch Size | 512 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Precision | 16-mixed AMP |
| **Loss** | Q-value MSE |
| Value Weight | 0.5 |

#### Results

| Metric | Value | Description |
|--------|-------|-------------|
| Q-Gap | 0.071 | Mean regret in points vs oracle |
| Q-MAE | 0.94 | Mean absolute error on Q-value prediction (points) |
| Value-MAE | 1.89 | Mean absolute error on game value prediction (points) |
| Accuracy | 79.0% | Model's argmax Q matches oracle's best move |
| Blunder Rate | 0.23% | Moves with Q-gap > 10 points |

#### Loading

```python
from forge.ml.module import DominoLightningModule

model = DominoLightningModule.load_from_checkpoint(
    "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
)
model.eval()

# Get Q-value predictions (in normalized [-1, 1] range)
q_values, value = model(tokens, mask, current_player)

# Convert to points: multiply by 42
q_points = q_values * 42  # Now in [-42, 42] range
best_move = q_points.argmax(dim=-1)
```

#### Provenance

- **Bead**: t42-dove (Q-value training), t42-mnaz (promotion)
- **Wandb**: qval-202601120501-3.3M-6L-8H-d256
- **Run dir**: `runs/domino/version_38`
- **Data**: `data/tokenized-full` (1124 seeds, 11.24M samples)

---

### Q-Val Small

**File**: `domino-qval-small-816k-qgap0.094-qmae1.49.ckpt`

Same architecture as policy models but trained with Q-value loss.

| Component | Value |
|-----------|-------|
| Parameters | 816K |
| Layers | 4 |
| Embed Dim | 128 |

| Metric | Value |
|--------|-------|
| Q-Gap | 0.094 |
| Q-MAE | 1.49 |
| Value-MAE | 2.39 |
| Accuracy | 74.7% |

**Provenance**: `runs/domino/version_37`, same training data as Q-Val Large.

---

## Why Q-Value Models?

### Interpretable Outputs

Q-value models output expected points directly:

```python
# Q-value model output (after * 42 denormalization)
e_q = [-17.89, -21.05, -21.45, ...]
# Interpretation: "Action 0 expects -17.89 points (17.89 behind)"
```

Policy models output logits that require softmax:

```python
# Policy model output
logits = [-0.72, -1.09, -1.87, ...]
# Interpretation: ??? (arbitrary scale)
```

### For E[Q] Marginalization

The E[Q] pipeline (`forge/eq/`) samples N consistent worlds and averages model outputs. With Q-value models:

- **E[Q]** is directly in points (range ±42)
- Values are **verifiable** against game outcome bounds
- **Debugging** is straightforward (e.g., "why does E[Q] = -30?")

With policy models, E[logits] has no clear interpretation—it's an average of arbitrary-scale values.

**Recommendation**: Always use Q-value models for E[Q] marginalization.

### Trade-offs

| Aspect | Q-Value Models | Policy Models |
|--------|----------------|---------------|
| Output | Expected points | Probability-like logits |
| Accuracy | 74-79% | 97%+ |
| Q-Gap | 0.07-0.09 | 0.07-0.11 |
| Interpretability | High | Low |
| Use case | Value estimation, E[Q] | Move sampling |

The "lower accuracy" for Q-value models is expected—they optimize for accurate Q-values across **all** actions, not just getting the top action right.

---

## Policy Models (Legacy)

These models were trained with soft cross-entropy loss and are kept for reference. For new work, prefer Q-value models.

### Large v2 (Value Head)

**File**: `domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt`

| Metric | Value |
|--------|-------|
| Val Accuracy | 97.79% |
| Val Q-Gap | 0.072 |
| Val Value MAE | 7.4 pts |

**Training Data** (validated from checkpoint):
- Dataset: `data/tokenized`
- Samples: ~500K (10 shards × 50K samples/shard)
- Note: Smaller dataset than Q-value models

**Architecture**: 817K params (embed=128, heads=8, layers=4, ff=512)

**Loss**: Soft cross-entropy + 0.5 × value MSE

### Large v1

**File**: `domino-large-817k-acc97.1-qgap0.11.ckpt`

Earlier version without value head. Same architecture as v2.

| Metric | Value |
|--------|-------|
| Val Accuracy | 97.09% |
| Val Q-Gap | 0.112 |

---

## Metrics Glossary

| Metric | Description |
|--------|-------------|
| **Q-Gap** | Expected point loss per move vs oracle: `oracle_best_q - oracle_q[model_choice]` |
| **Q-MAE** | Mean absolute error of predicted Q-values vs oracle Q-values (in points) |
| **Value-MAE** | Mean absolute error of predicted game value vs oracle value (in points) |
| **Accuracy** | Fraction of moves where model's argmax matches oracle's best move |
| **Blunder Rate** | Fraction of moves with Q-gap > 10 (catastrophic mistakes) |
| **Oracle** | GPU minimax solver with perfect play (ground truth) |

---

## File Sizes

| Model | Checkpoint | Weights Only |
|-------|------------|--------------|
| Q-Val Large (3.3M) | 39 MB | ~13 MB |
| Q-Val Small (816K) | 9.9 MB | ~3 MB |
| Policy Large (817K) | 9.5 MB | ~3 MB |

---

## Model Evolution

| Date | Model | Key Change | Q-Gap |
|------|-------|-----------|-------|
| 2025-12-31 | Large v1 | 100 seeds, soft CE | 0.112 |
| 2026-01-01 | Large v2 | Added value head | 0.072 |
| 2026-01-13 | Q-Val Small | Q-value loss | 0.094 |
| 2026-01-13 | **Q-Val Large** | 6L, 256 dim, 1124 seeds | **0.071** |

**Key finding**: Q-value loss with more data (1124 vs 10 seeds) and larger architecture achieves best Q-gap while providing interpretable outputs.
