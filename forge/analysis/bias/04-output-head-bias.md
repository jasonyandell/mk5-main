# Investigation 04: Output Head Bias Analysis

**Date**: 2026-01-25
**Checkpoint**: `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`
**Question**: Does the Q-value output head have learned bias against output position 0?

## Background

The model shows systematic degradation at slot 0:

| Metric             | Slot 0 | Slots 1-6    |
|--------------------|--------|--------------|
| r(model, oracle)   | 0.81   | 0.99+        |
| Variance explained | 65%    | 98.7-99.0%   |
| Mean bias          | -0.26  | -0.04 to +0.02 |

Slot 0 predictions are 19x worse than other slots. This investigation examines whether the output head weights/bias contain position-specific artifacts.

## Model Architecture

The DominoTransformer uses this architecture for Q-value prediction:

```python
# Output layer (shared across all 7 positions)
self.output_proj = nn.Linear(embed_dim, 1)

# Forward pass
hand_repr = torch.gather(x, dim=1, index=gather_indices)  # (batch, 7, embed_dim)
logits = self.output_proj(hand_repr).squeeze(-1)          # (batch, 7)
```

**Key Insight**: The same `output_proj` layer is applied identically to ALL 7 hand positions. The layer cannot distinguish position 0 from positions 1-6.

## Checkpoint Analysis

### Model Hyperparameters
- embed_dim: 256
- n_heads: 8
- n_layers: 6
- ff_dim: 512
- loss_mode: qvalue
- Training: 18 epochs, 417,107 global steps

### Output Projection Layer

```
Weight shape: (1, 256)
  Mean: 0.0196
  Std: 0.3201
  Min: -0.5636
  Max: 0.6631
  L2 norm: 5.121

Bias: -0.0450 (single scalar)
```

**Finding**: The output layer has a single bias term of -0.0450, applied uniformly to all positions. There is no position-specific output bias.

### Weight Analysis

The output projection is `Linear(256, 1)`:
- Input: 256-dimensional embedding from transformer
- Output: 1 Q-value per position

Since the same weights are applied to all positions, any slot 0 degradation must come from **upstream in the transformer**, not from the output head itself.

## Root Cause Analysis

The bias cannot be in the output head. Looking at the architecture and tokenization:

### Token Sequence Structure (tokenize.py)

```
Position 0:  Context token
Position 1-7:  Player 0's 7 dominoes (hand slots 0-6)
Position 8-14: Player 1's 7 dominoes
Position 15-21: Player 2's 7 dominoes
Position 22-28: Player 3's 7 dominoes
Position 29-31: Trick tokens (0-3 played cards)
```

For the **current player**, hand representations are gathered from:
- If current_player=0: positions 1-7 (slot 0 = position 1)
- If current_player=1: positions 8-14 (slot 0 = position 8)
- If current_player=2: positions 15-21 (slot 0 = position 15)
- If current_player=3: positions 22-28 (slot 0 = position 22)

**Slot 0 of the hand is always the FIRST position in that player's hand block.**

### Potential Upstream Causes

1. **Transformer Attention Patterns**
   - Self-attention may systematically underweight the first position in each hand block
   - "First token" effects are well-documented in NLP transformers

2. **Domino Ordering**
   - Hands are dealt in a fixed order from `deal_from_seed()`
   - If domino ordering correlates with strategic value, position 0 could have different statistical properties

3. **Token Type Embedding**
   - Each player's dominoes share the same `token_type` embedding (TOKEN_TYPE_PLAYER0 + p)
   - Position 0 in each block receives the same amount of context from other positions

4. **Gradient Flow**
   - During backpropagation, position 0 may receive different gradient magnitude
   - Possible "boundary effects" at the start of each hand block

## Embedding Layer Observations

Several embedding layers show differences at position 0:

| Embedding | Pos 0 L2 Norm | Other Avg L2 | Difference |
|-----------|---------------|--------------|------------|
| high_pip_embed | 3.67 | 2.84 | +0.84 |
| is_double_embed | 0.46 | 1.64 | -1.18 |
| is_current_embed | 0.77 | 1.72 | -0.95 |
| token_type_embed | 3.72 | 1.92 | +1.80 |

These differences could propagate through the transformer and affect slot 0 representations. However, these embeddings are indexed by **feature values** (pip counts, booleans), not by **positional indices** within the hand.

The token_type_embed shows notable difference at index 0 (TOKEN_TYPE_CONTEXT = 0), which is the context token position. This is expected - the context token plays a different role.

## Conclusion

**The output head does NOT have position-specific bias.**

The Q-value projection layer uses:
- A single (1, 256) weight matrix applied to all positions
- A single scalar bias (-0.045) applied uniformly

The slot 0 degradation must originate from:

1. **The transformer encoder** - Attention patterns or representations that systematically differ for the first position in each hand block

2. **The tokenization** - How dominoes are ordered within each hand, and whether position 0 tends to contain systematically different dominoes

3. **Implicit positional effects** - The transformer (without explicit positional encodings) may still learn implicit position-dependent patterns through the attention mechanism

**Recommended Next Investigation**: Analyze whether the domino at hand slot 0 has systematically different properties (e.g., strategic value, point count, trump status) than dominoes at other slots, since hand ordering comes from the deterministic `deal_from_seed()` function.
