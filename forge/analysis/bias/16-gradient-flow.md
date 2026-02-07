# 16: Gradient Flow Analysis - Value Head vs Q-Head

## Hypothesis

**Question**: Does gradient interference from the value head (using context token at position 0) explain the slot 0 Q-value degradation (r=0.81 vs r=0.99+)?

**Mechanism proposed**:
- Value head reads from position 0 (context token)
- Q-head reads from positions 1-7 (player's hand, where position 1 = slot 0)
- Position 1 is adjacent to position 0
- During backprop, gradients from the value head might "spill over" to position 1 through:
  1. Direct gradient flow via shared transformer layers
  2. Attention-mediated gradient propagation

## Architecture Review

```
Forward pass:
  Input tokens (batch, 32, 12)
      ↓
  Embedding layers (12 feature embeddings concatenated)
      ↓
  input_proj: Linear(concatenated_dim, 256)
      ↓
  transformer: 6-layer TransformerEncoder
      ↓ [output shape: (batch, 32, 256)]
      ├─→ x[:, 0, :] → value_head → scalar state value V
      └─→ gather(x, positions 1-7 for current player) → output_proj → 7 Q-values

Loss:
  total_loss = q_loss + 0.5 * value_loss
```

## Experimental Method

Ran gradient flow analysis on trained model (`domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`) with 256 validation samples:

1. Hook transformer output to capture gradients
2. Separately backprop: (a) total loss, (b) value loss only, (c) Q-loss only
3. Measure L2 gradient magnitude per position

## Results

### Transformer Output Gradients

| Loss Source | Position 0 | Position 1 | Mean(2-7) | Ratio P1/Mean(2-7) |
|-------------|------------|------------|-----------|-------------------|
| **Total**   | 2.46e-6    | 2.69e-7    | 2.37e-7   | **1.13x**         |
| **Value only** | 6.23e-6 | 0.00e+0    | 0.00e+0   | N/A               |
| **Q only**  | 0.00e+0    | 3.24e-7    | 2.50e-7   | **1.29x**         |

### Key Finding 1: Value Head Gradient is Isolated

**Value loss gradient is exactly zero at positions 1-7.**

The value head performs `value = value_head(x[:, 0, :])`, a simple linear projection. During backprop, `d_loss/d_x[:, 0, :]` receives gradient, but `d_loss/d_x[:, 1:, :]` = 0 because these positions are not used in the value computation.

This rules out direct gradient spillover at the transformer output level.

### Key Finding 2: Q-Loss Shows Mild Position 1 Elevation

Position 1 (slot 0) receives **1.29x** the gradient of positions 2-7 from Q-loss alone. This is a real but small asymmetry that exists independent of the value head.

Possible causes:
- **Domino sorting**: Slot 0 always contains the minimum-ID domino (0-0, 1-0, 1-1, etc.). These have narrower Q-value distributions (confirmed in investigation 12).
- **Output projection**: The output head applies `nn.Linear(256, 1)` identically to all positions, but the Q-values being predicted differ by slot.

### Key Finding 3: Input Embedding Gradients Show Inverse Pattern

| Position | Gradient Magnitude | vs Mean(2-7) |
|----------|-------------------|---------------|
| 0        | 1.28e-7           | 0.29x         |
| 1        | 1.85e-7           | 0.42x         |
| 2-7 mean | 4.37e-7           | 1.00x         |

At the input embedding level, position 1 has **lower** gradient than positions 2-7. This is counterintuitive but suggests:
- The transformer redistributes gradients during backprop
- Position 1's lower input gradient may indicate that the model has learned to rely less on slot 0 features

## Detailed Position-by-Position Gradient (Q-loss only)

```
Position  0: 0.00e+0  (context - no Q-loss gradient)
Position  1: 3.24e-7  (slot 0) ← 1.29x elevation
Position  2: 2.18e-7  (slot 1)
Position  3: 1.12e-7  (slot 2)
Position  4: 1.63e-7  (slot 3)
Position  5: 1.42e-7  (slot 4)
Position  6: 7.83e-7  (slot 5) ← highest
Position  7: 8.27e-8  (slot 6) ← lowest
```

Slots 5 and 6 show high variance in gradient magnitude. This aligns with investigation 12's finding that slot 6 contains high-value dominoes (6-5, 6-6) with more complex strategic decisions.

## Analysis: Why Value Head Doesn't Cause Position 1 Degradation

### Theoretical Analysis of Transformer Backprop

In a transformer encoder layer, the gradient flow during backprop is:

```
d_loss/d_x_in = W_o^T @ d_loss/d_attn_out
             + d_loss/d_ff_out @ d_ff/d_x
```

For position `i`, the attention contribution to gradient is:
```
d_loss/d_x_in[i] = sum_j(A[i,j] * W_v @ d_loss/d_x_out[j])
```

where `A[i,j]` is the attention from position `i` to position `j`.

**Key insight**: Even though position 0 receives large gradient from value loss, this doesn't directly affect position 1's gradient unless:
1. Position 1 attends heavily to position 0, AND
2. The gradient flows through that attention path

From investigation 10 (attention routing), we found that attention between position 0 and position 1 is NOT elevated compared to other position pairs.

### Why Gradient Interference is Unlikely

1. **Linear independence**: The output heads (value_head and output_proj) are separate linear layers. Their gradients don't interact until reaching the shared transformer.

2. **Attention acts as a router**: Attention weights determine gradient flow. If position 0 and position 1 don't have elevated mutual attention, gradients don't preferentially flow between them.

3. **Multi-head diversity**: With 8 attention heads per layer, any single attention pattern is diluted.

4. **6 layers of mixing**: After 6 transformer layers, any local gradient concentration is heavily dispersed.

## Conclusion

**Gradient interference from the value head does NOT explain the slot 0 bias.**

Evidence:
- Value loss gradient is exactly zero at positions 1-7 (no direct spillover)
- Position 1's elevated Q-loss gradient (1.29x) exists independently of the value head
- Input embedding gradients show position 1 is LOWER than positions 2-7

The slot 0 bias is more likely caused by:
1. **Training distribution bias** (investigation 12): Slot 0 always contains low-ID dominoes with narrower Q-value distributions
2. **Simpler prediction task**: Low-ID dominoes (0-0, 1-0, 1-1) may have more predictable values, leading to higher r but potentially worse generalization when rare high-value dominoes appear in slot 0

## Recommendations

1. **Not needed**: Architectural changes to isolate value/Q-head gradients
2. **Recommended**: Address the training distribution bias by:
   - Shuffling slot indices during training (data augmentation)
   - Using a permutation-invariant architecture (DeepSets, Set Transformer)

## Script Reference

Analysis script: `scratch/gradient_flow_analysis.py`

Model checkpoint: `forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`
