# 10: Deep Attention Routing Analysis

## Question
Is position 0 (or hand slot 0) systematically isolated in attention patterns?

## Method
- Loaded model: `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`
- Ran inference on 1000 validation samples
- Extracted attention weights from all transformer layers
- Analyzed attention received/given by each position

## Model Architecture
- layer_0: 8 heads, sequence length 32
- layer_1: 8 heads, sequence length 32
- layer_2: 8 heads, sequence length 32
- layer_3: 8 heads, sequence length 32
- layer_4: 8 heads, sequence length 32
- layer_5: 8 heads, sequence length 32

## Position Layout
```
Position 0:     Context token
Positions 1-7:  P0's hand (slot 0-6 for current player when P0)
Positions 8-14: P1's hand
Positions 15-21: P2's hand
Positions 22-28: P3's hand
Positions 29+:  Trick history
```

## Results

### Attention Flow Summary

| Layer | Pos 0 Recv | Pos 1-7 Recv | Ratio | Pos 0 Given | Pos 1-7 Given | Ratio |
|-------|------------|--------------|-------|-------------|---------------|-------|
| layer_0 | 3.193 | 0.885 | 3.61 | 1.000 | 1.000 | 1.00 |
| layer_1 | 0.922 | 0.951 | 0.97 | 1.000 | 1.000 | 1.00 |
| layer_2 | 0.625 | 0.843 | 0.74 | 1.000 | 1.000 | 1.00 |
| layer_3 | 0.765 | 0.926 | 0.83 | 1.000 | 1.000 | 1.00 |
| layer_4 | 0.628 | 0.954 | 0.66 | 1.000 | 1.000 | 1.00 |
| layer_5 | 1.043 | 0.992 | 1.05 | 1.000 | 1.000 | 1.00 |

### Isolation Metrics

| Layer | Attn 0->1-7 | Attn 1-7->0 | Internal 1-7 | Isolation Score |
|-------|-------------|-------------|--------------|----------------|
| layer_0 | 0.0333 | 0.0990 | 0.0342 | 1.03 |
| layer_1 | 0.0246 | 0.0322 | 0.0292 | 1.19 |
| layer_2 | 0.0251 | 0.0185 | 0.0236 | 0.94 |
| layer_3 | 0.0261 | 0.0222 | 0.0293 | 1.12 |
| layer_4 | 0.0309 | 0.0202 | 0.0301 | 0.97 |
| layer_5 | 0.0304 | 0.0330 | 0.0312 | 1.03 |

### Hand Slot Analysis (Key for Output Bias)

The bias manifests in **output slots** (the 7 Q-value predictions for a player's hand).
For P0, these correspond to sequence positions 1-7.

| Layer | Slot 0 Recv | Slots 1-6 Recv (mean) | Ratio |
|-------|-------------|----------------------|-------|
| layer_0 | 0.1972 | 0.2465 | 0.80 |
| layer_1 | 0.2266 | 0.2009 | 1.13 |
| layer_2 | 0.2389 | 0.1529 | 1.56 |
| layer_3 | 0.2299 | 0.2012 | 1.14 |
| layer_4 | 0.2283 | 0.2079 | 1.10 |
| layer_5 | 0.2136 | 0.2195 | 0.97 |

## Visualizations

- `figures/10_attention_heatmaps.png` - Mean attention matrices per layer
- `figures/10_attention_by_position.png` - Attention received/given by position
- `figures/10_per_head_attention_layer0.png` - Per-head patterns in first layer
- `figures/10_position_0_isolation.png` - Analysis of position 0 connectivity
- `figures/10_hand_slot_attention.png` - Attention among hand slots (1-7)
- `figures/10_slot0_vs_others_attention.png` - Slot 0 vs slots 1-6 comparison

## Interpretation

**Finding: Attention to slot 0 is similar to slots 1-6 (ratio = 0.97).**

Attention routing does NOT explain the slot 0 bias.

### Key Observations

1. **Position 0 (context token) is NOT isolated** - Layer 0 actually shows position 0 receiving 3.6x MORE attention than positions 1-7. This is expected since the context token contains global game state information (declaration, leader).

2. **Hand slot 0 (output position 0, sequence position 1) receives balanced attention** - In the final layer, the ratio of attention to slot 0 vs slots 1-6 is 0.97 (essentially equal). Some middle layers show slot 0 receiving MORE attention (ratio up to 1.56 in layer 2).

3. **Attention "given" is uniform** - All positions contribute equally to queries (ratio = 1.00 across all layers), as expected with softmax normalization over keys.

4. **Isolation scores near 1.0** - The internal attention among positions 1-7 is comparable to cross-attention between position 0 and positions 1-7, indicating no information barrier.

### Relationship to Domino Ordering Bias (Investigation 12)

Investigation 12 found that `deal_from_seed()` sorts hands, causing slot 0 to always contain low-ID dominoes (0-0, 1-0, 1-1, etc.) while high-value dominoes (6-4, 6-5, 6-6) never appear in slot 0.

The attention analysis rules out the hypothesis that the model learned to "ignore" slot 0. Instead:
- Slot 0 receives normal attention
- The degraded Q-value prediction (r=0.81 vs r=0.99+) likely stems from **training distribution asymmetry**
- Slot 0 sees a restricted subset of dominoes during training, making generalization harder

## Conclusion

**Attention routing does NOT cause the slot 0 bias.**

The bias likely originates from the training distribution asymmetry identified in investigation 12:
1. Slot 0 only sees low-ID dominoes during training (sorting artifact)
2. The model learns slot-specific patterns rather than generalizing across positions
3. At inference time, when the domino distribution differs from training, slot 0 predictions degrade

**Recommendation**: Train with shuffled hands (randomized domino ordering within each hand) to eliminate the positional correlation with domino ID.
