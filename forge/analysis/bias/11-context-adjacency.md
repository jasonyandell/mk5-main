# Investigation 11: Context Token Adjacency Effect

**Date**: 2026-01-25
**Checkpoint**: `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`
**Question**: Does position 1's proximity to the context token (position 0) cause it to behave differently?

## Summary

**Finding: PARTIAL EVIDENCE for context adjacency effect.** The context token at position 0 shows a **3.51x attention bias toward position 1 (slot 0)** in early layers, and position 1 shows **1.5x elevated attention toward context** in middle layers. However, this attention asymmetry likely explains only part of the slot 0 degradation. The value head's use of the context token for state value prediction creates a gradient competition scenario that may compound the effect.

## Confirmed Slot 0 Bias

Re-measured on 50,000 validation samples:

| Slot | Correlation | R^2    | MAE  | Bias |
|------|-------------|--------|------|------|
| 0    | **0.6212**  | 21.1%  | 6.05 | -0.43 |
| 1    | 0.9897      | 97.6%  | 1.03 | -0.03 |
| 2    | 0.9907      | 97.8%  | 1.02 | -0.02 |
| 3    | 0.9931      | 98.3%  | 1.00 | -0.12 |
| 4    | 0.9920      | 98.1%  | 1.02 | -0.04 |
| 5    | 0.9918      | 98.0%  | 1.08 | +0.01 |
| 6    | 0.9919      | 98.0%  | 1.05 | -0.10 |

**Slot 0 explains only 21% of oracle variance vs 98% for other slots - a 4.6x degradation.**

The bias is consistent across all current player positions:

| Current Player | Token Position for Slot 0 | Slot 0 r | Slots 1-6 mean r |
|----------------|---------------------------|----------|------------------|
| 0              | 1 (adjacent to context)   | 0.5558   | 0.9917           |
| 1              | 8                         | 0.6538   | 0.9914           |
| 2              | 15                        | 0.6099   | 0.9906           |
| 3              | 22                        | 0.7178   | 0.9917           |

**Key observation**: Player 0 (whose slot 0 is at position 1, directly adjacent to context) has the **worst** slot 0 correlation (r=0.5558). Player 3 (whose slot 0 is at position 22, farthest from context) has the **best** slot 0 correlation (r=0.7178).

## Attention Analysis

### Context Token Attending to Hand Positions

The context token (position 0) shows elevated attention to the first hand position in early layers:

| Layer | Context -> Slot 0 | Context -> Slots 1-6 (mean) | Ratio |
|-------|-------------------|----------------------------|-------|
| 0     | **0.0866**        | 0.0247                     | **3.51x** |
| 1     | 0.0266            | 0.0232                     | 1.15x |
| 2     | 0.0289            | 0.0285                     | 1.01x |
| 3     | 0.0314            | 0.0282                     | 1.11x |
| 4     | 0.0307            | 0.0321                     | 0.96x |
| 5     | 0.0330            | 0.0282                     | 1.17x |

**Layer 0 shows a striking 3.51x attention bias** from context toward slot 0. This could corrupt the slot 0 representation by mixing in context-specific information that the value head optimizes for state value prediction rather than Q-value prediction.

### Hand Positions Attending to Context

| Layer | Slot 0 -> Context | Slots 1-6 -> Context (mean) | Ratio |
|-------|-------------------|----------------------------|-------|
| 0     | 0.1058            | 0.0971                     | 1.09x |
| 1     | **0.0352**        | 0.0228                     | **1.55x** |
| 2     | **0.0305**        | 0.0202                     | **1.51x** |
| 3     | 0.0251            | 0.0241                     | 1.04x |
| 4     | 0.0320            | 0.0297                     | 1.08x |
| 5     | 0.0338            | 0.0341                     | 0.99x |

**Layers 1-2 show 1.5x elevated attention from slot 0 toward context.** This suggests slot 0 is "distracted" by the context token, potentially contaminating its representation with value-prediction information rather than action-specific Q-value information.

### Mean Across Layers

- Context -> Slot 0 / Others: **1.44x**
- Slot 0 -> Context / Others: **1.15x**

## Value Head Gradient Competition

The architecture uses position 0 (context token) for state value prediction:

```python
# From forge/ml/module.py line 122
value = self.value_head(x[:, 0, :]).squeeze(-1)
```

During training, the context token representation is optimized for **two competing objectives**:

1. **Value prediction**: The value head directly uses position 0 to predict game outcome
2. **Information provision**: Other tokens attend to position 0 to gather game context

### Gradient Analysis (100 samples)

| Position | Gradient from Q-loss | Gradient from V-loss |
|----------|---------------------|---------------------|
| 0 (context) | 0.0 (not used for Q) | **4.6e-5** (used for V) |
| 1 (slot 0)  | 1e-6 | 0.0 |
| 2-7 (slots 1-6) | 2e-6 | 0.0 |

The value head gradient is **100% concentrated at position 0**, while Q-value gradients flow to hand positions. This creates a "gradient isolation" scenario where position 0's representation is primarily shaped by value prediction, potentially degrading its usefulness for nearby positions that attend to it.

## Mechanism Hypothesis

The slot 0 degradation appears to result from a cascading effect:

1. **Attention Asymmetry (Layer 0)**: Context token attends 3.51x more to slot 0 than other slots, potentially because position 1 is the first "content" position after the context token.

2. **Reciprocal Attention (Layers 1-2)**: Slot 0 attends 1.5x more to context than other slots, creating a "closed loop" where slot 0 and context exchange more information than slot 0 exchanges with other hand positions.

3. **Value Head Contamination**: The context token is optimized for state value prediction. When slot 0 attends strongly to context, it absorbs information optimized for value prediction rather than Q-value prediction.

4. **Representation Corruption**: Slot 0's representation becomes "contaminated" with value-prediction-oriented features, degrading its ability to predict accurate Q-values.

## Position-Dependent Severity

The correlation between slot 0 position (distance from context) and accuracy:

| Distance from Context | Slot 0 Position | Slot 0 r |
|----------------------|-----------------|----------|
| 1 (adjacent)         | Position 1 (P0) | 0.5558   |
| 8                    | Position 8 (P1) | 0.6538   |
| 15                   | Position 15 (P2)| 0.6099   |
| 22                   | Position 22 (P3)| 0.7178   |

**Correlation between distance and accuracy: r = 0.77** (based on 4 points)

This supports the hypothesis that proximity to the context token causes the degradation.

## Representation Similarity Analysis

Cosine similarity between context token and hand positions (500 samples):

| Position | Similarity to Context |
|----------|----------------------|
| 1 (P0 slot 0) | 0.8181 |
| 2 (P0 slot 1) | 0.8449 |
| 3 (P0 slot 2) | 0.8328 |
| 4 (P0 slot 3) | 0.8575 |
| 5 (P0 slot 4) | 0.8579 |
| 6 (P0 slot 5) | 0.8187 |
| 7 (P0 slot 6) | 0.8540 |

**Position 1 (slot 0) has the lowest similarity to context (0.8181)**, which is unexpected if the attention hypothesis were simply about "copying" context information. This suggests the relationship is more nuanced - perhaps slot 0's representation is being pulled in a different direction by the attention interaction.

## Conclusion

**Context adjacency is a contributing factor to slot 0 degradation, but not the sole cause.**

Evidence supporting context adjacency effect:
- 3.51x elevated attention from context to slot 0 in layer 0
- 1.5x elevated attention from slot 0 to context in layers 1-2
- Player 0 (position 1, adjacent to context) has the worst slot 0 correlation
- Player 3 (position 22, farthest from context) has the best slot 0 correlation

Evidence that other factors also contribute:
- All players show slot 0 degradation, not just player 0
- The degradation persists even at position 22 (r=0.7178 vs 0.99+)
- Slot 0 has **lower** cosine similarity to context, not higher

## Recommendations

1. **Test attention masking**: Mask attention between context (pos 0) and first hand position (pos 1) during training to see if this reduces slot 0 bias.

2. **Use separate context tokens**: Instead of one context token at position 0, use two: one for value prediction (attended only by value head) and one for Q-value context (attended by hand positions).

3. **Investigate other factors**: The consistent slot 0 degradation across all players suggests additional causes beyond context adjacency:
   - Fixed output extraction order (slot 0 is always first-gathered)
   - Training data distribution (systematic differences in first-slot dominoes)
   - Implicit positional effects in attention computation

## References

- `forge/ml/module.py`: Model architecture showing value head using `x[:, 0, :]`
- `forge/ml/tokenize.py`: Token layout with context at position 0
- `forge/analysis/scripts/attention_analysis.py`: Attention extraction methodology
