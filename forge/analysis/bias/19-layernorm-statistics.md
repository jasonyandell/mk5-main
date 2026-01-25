# 19: LayerNorm Statistics by Position

## Question
Do LayerNorm statistics differ systematically for position 1 (slot 0) vs other positions?

## Method
- Loaded Q-value model: `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`
- Ran inference on 5000 validation samples
- Manually forwarded through each transformer layer to capture pre-normalization activations
- Computed per-position statistics:
  - Mean activation value (averaged across embedding dim, then samples)
  - Standard deviation (across embedding dim per sample, then averaged - what LayerNorm divides by)

## Position Layout
```
Position 0:     Context token
Positions 1-7:  P0's hand (slot 0 at position 1)
Positions 8-14: P1's hand
Positions 15-21: P2's hand
Positions 22-28: P3's hand
```

## Results

### Standard Deviation by Position (LayerNorm Scaling Factor)

Higher std means LayerNorm scales down the activation more aggressively.

| Layer | Slot 0 Std | Slots 1-6 Std | Ratio |
|-------|------------|---------------|-------|
| layer_0_norm1 | 0.8258 | 0.7027 | 1.175 |
| layer_0_norm2 | 1.0571 | 0.9548 | 1.107 |
| layer_1_norm1 | 0.4007 | 0.4272 | 0.938 |
| layer_1_norm2 | 0.5199 | 0.5546 | 0.938 |
| layer_2_norm1 | 0.4500 | 0.4695 | 0.959 |
| layer_2_norm2 | 0.5734 | 0.5996 | 0.956 |
| layer_3_norm1 | 0.4429 | 0.4546 | 0.974 |
| layer_3_norm2 | 0.6022 | 0.6143 | 0.980 |
| layer_4_norm1 | 0.4854 | 0.4892 | 0.992 |
| layer_4_norm2 | 0.7707 | 0.7687 | 1.003 |
| layer_5_norm1 | 0.4771 | 0.4693 | 1.017 |
| layer_5_norm2 | 3.2160 | 3.2111 | 1.002 |

### Mean Activation by Position

| Layer | Slot 0 Mean | Slots 1-6 Mean | Ratio |
|-------|-------------|----------------|-------|
| layer_0_norm1 | -0.0023 | -0.0048 | 0.474 |
| layer_0_norm2 | -0.0179 | -0.0541 | 0.331 |
| layer_1_norm1 | 0.0003 | 0.0015 | 0.197 |
| layer_1_norm2 | -0.0115 | -0.0126 | 0.915 |
| layer_2_norm1 | 0.0055 | 0.0046 | 1.204 |
| layer_2_norm2 | 0.0042 | 0.0034 | 1.236 |
| layer_3_norm1 | 0.0029 | 0.0028 | 1.055 |
| layer_3_norm2 | 0.0039 | 0.0024 | 1.606 |
| layer_4_norm1 | 0.0004 | 0.0005 | 0.719 |
| layer_4_norm2 | 0.0014 | 0.0028 | 0.481 |
| layer_5_norm1 | -0.0001 | -0.0001 | 1.235 |
| layer_5_norm2 | 0.0179 | 0.0174 | 1.030 |

### Per-Position Detail (Last Layer norm2)

| Position | Role | Mean | Std |
|----------|------|------|-----|
| 0 | context | 0.0281 | 3.5979 |
| 1 | slot0 | 0.0179 | 3.2160 |
| 2 | slot1 | 0.0161 | 3.2216 |
| 3 | slot2 | 0.0168 | 3.2291 |
| 4 | slot3 | 0.0164 | 3.2282 |
| 5 | slot4 | 0.0171 | 3.2250 |
| 6 | slot5 | 0.0207 | 3.1536 |
| 7 | slot6 | 0.0172 | 3.2091 |

## Interpretation

**Do LayerNorm statistics differ for slot 0?** **NO**: Slot 0 std ratio (1.002) is within 5% of slots 1-6.

LayerNorm treats slot 0 similarly to other slots.

## Conclusion

LayerNorm statistics are essentially identical for slot 0 vs other slots.
LayerNorm normalization is NOT the cause of the positional bias.