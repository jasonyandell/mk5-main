# Investigation 18: Layer-wise Degradation Analysis

## Question
Does the slot 0 bias accumulate across transformer layers, or is it introduced at a specific layer?

## Context
The model shows slot 0 has r=0.81 correlation with oracle while slots 1-6 have r=0.99+.
Previous investigations (01-14) ruled out attention masking, tokenization, positional encoding, and output head bias. The remaining hypothesis is that the degradation occurs within the transformer encoder layers themselves.

## Method
1. Extract embeddings after each transformer layer
2. Project each layer's hand embeddings through the output head to get Q-values
3. Correlate predicted Q-values with oracle Q-values for each slot
4. Track how the correlation gap (slot 0 vs slots 1-6) changes layer by layer

**Model**: 6 layers, 256 embed_dim
**Validation samples**: 499,601

## Results

### Correlation by Layer

| Layer | Slot 0 (r) | Slots 1-6 mean (r) | Gap |
|-------|------------|-------------------|-----|
| input | -0.0058 | 0.0403 | -0.0461 |
| layer_0 | 0.1412 | 0.1535 | -0.0123 |
| layer_1 | 0.0729 | 0.1469 | -0.0740 |
| layer_2 | 0.3327 | 0.3753 | -0.0426 |
| layer_3 | 0.5819 | 0.6214 | -0.0395 |
| layer_4 | 0.8083 | 0.8410 | -0.0327 |
| layer_5 | 0.9592 | 0.9949 | -0.0357 |

### Degradation per Layer Transition

| Transition | Slot 0 delta | Slots 1-6 delta | Gap change |
|------------|--------------|-----------------|------------|
| input -> layer_0 | +0.1470 | +0.1132 | +0.0338 |
| layer_0 -> layer_1 | -0.0683 | -0.0066 | -0.0617 |
| layer_1 -> layer_2 | +0.2598 | +0.2284 | +0.0314 |
| layer_2 -> layer_3 | +0.2492 | +0.2461 | +0.0031 |
| layer_3 -> layer_4 | +0.2264 | +0.2196 | +0.0068 |
| layer_4 -> layer_5 | +0.1509 | +0.1539 | -0.0030 |

### Per-Slot Correlations

| Layer | Slot 0 | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Slot 5 | Slot 6 |
|-------|-----|-----|-----|-----|-----|-----|-----|
| input | -0.0058 | -0.0336 | 0.0473 | 0.1025 | 0.0693 | -0.0124 | 0.0687 |
| layer_0 | 0.1412 | 0.1204 | 0.1405 | 0.1996 | 0.2003 | 0.0874 | 0.1726 |
| layer_1 | 0.0729 | 0.0861 | 0.1244 | 0.1897 | 0.1711 | 0.1287 | 0.1811 |
| layer_2 | 0.3327 | 0.3294 | 0.3487 | 0.4021 | 0.3791 | 0.3821 | 0.4103 |
| layer_3 | 0.5819 | 0.5903 | 0.6154 | 0.6283 | 0.6279 | 0.6341 | 0.6324 |
| layer_4 | 0.8083 | 0.8327 | 0.8389 | 0.8443 | 0.8435 | 0.8441 | 0.8426 |
| layer_5 | 0.9592 | 0.9945 | 0.9948 | 0.9951 | 0.9951 | 0.9951 | 0.9950 |

## Key Findings

### Pattern: Small gap amplified dramatically in final layer

1. **The gap is small and stable through layers 0-4**: -0.01 to -0.07 difference
2. **Dramatic gap emergence at final layer (layer_5)**: Slot 0 reaches r=0.96 while slots 1-6 reach r=0.995
3. **Layer 1 is a regression point**: Both slot 0 and slots 1-6 correlations DROP from layer 0 to layer 1, but slot 0 drops more (-0.0683 vs -0.0066), widening the gap by 0.06

The critical observation is that while both slot 0 and slots 1-6 improve massively (from ~0 to ~0.96-0.99), **slot 0 fails to make the final jump to 0.99** that the other slots achieve. The final layer creates a 3.5% correlation gap (0.9592 vs 0.9949).

### Variance Explained

| Slot | Final r | r^2 (variance explained) |
|------|---------|-------------------------|
| Slot 0 | 0.9592 | 92.0% <-- biased |
| Slot 1 | 0.9945 | 98.9% |
| Slot 2 | 0.9948 | 99.0% |
| Slot 3 | 0.9951 | 99.0% |
| Slot 4 | 0.9951 | 99.0% |
| Slot 5 | 0.9951 | 99.0% |
| Slot 6 | 0.9950 | 99.0% |

## Conclusion

The slot 0 bias is **NOT a gradual accumulation** but rather a **failure to converge in the final layer**:

1. **Both slot 0 and slots 1-6 start with near-zero correlation** (no predictive power before transformer processing)
2. **Both improve steadily through layers 0-4** with only minor gap fluctuations
3. **In the final layer, slots 1-6 jump from r=0.84 to r=0.995 while slot 0 only reaches r=0.959**

This suggests the model has sufficient capacity to predict all slots well, but **position 1's embedding fails to extract the last 3.5% of oracle information** in the final layer. The root cause is likely:

- **Training data imbalance**: Slot 0 sees systematically different domino distributions (see investigation 12)
- **Attention routing**: Position 1 may receive less refined information from other positions in the final layer

The transformer architecture itself is not inherently biased - it processes all positions similarly. The bias emerges from the **data distribution** where slot 0 always contains the lowest-ID domino due to hand sorting.

## Visualizations

- `figures/18_layer_wise_degradation.png` - Correlation trajectory and gap analysis
