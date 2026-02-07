# 17: Attention Head Specialization Analysis

## Question
Do specific attention heads consistently treat position 1 (hand slot 0) differently?

## Background
The model shows systematic bias against slot 0:
- Slot 0: r=0.81 correlation with oracle
- Slots 1-6: r=0.99+ correlation with oracle

This analysis examines whether individual attention heads specialize on
ignoring or suppressing information flow to/from position 1.

## Method
- Model: `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`
- Samples: 2000 validation examples
- For each (layer, head) pair:
  - Computed attention TO slot 0 from slots 1-6
  - Computed attention TO slots 1-6 (internal)
  - Computed attention FROM slot 0 to slots 1-6
  - Calculated ratio and statistical significance
- A head is classified as a 'suppressor' if:
  - TO ratio < 0.8 (slots 1-6 attend to slot 0 less than to each other)
  - p-value < 0.01 (statistically significant)

## Position Layout Reminder
```
For current player P:
  Position 1 + P*7 = Slot 0 (first domino in hand)
  Positions 2-7 + P*7 = Slots 1-6 (remaining dominoes)
```

## Results

### Per-Layer Summary

| Layer | Mean TO Ratio | Mean FROM Ratio | Suppressors |
|-------|---------------|-----------------|-------------|
| layer_0 | 0.819 | 1.298 | 6/8 |
| layer_1 | 0.969 | 1.055 | 1/8 |
| layer_2 | 1.076 | 0.985 | 0/8 |
| layer_3 | 1.163 | 0.908 | 0/8 |
| layer_4 | 1.100 | 1.010 | 1/8 |
| layer_5 | 0.886 | 1.079 | 3/8 |

### Per-Head TO Ratios

| Layer | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 | 
|-------|----|----|----|----|----|----|----|----|
| layer_0 |  **0.47** |  **0.72** |  **0.69** | 1.52 |  **0.78** | 1.45 |  **0.39** |  **0.54** | 
| layer_1 |  **0.51** | 0.93 | 0.91 | 0.83 | 1.06 | 1.09 | 1.46 | 0.96 | 
| layer_2 | 0.85 | 0.82 | 1.41 | 1.34 | 0.96 | 1.54 | 0.87 | 0.80 | 
| layer_3 | 1.11 | 1.39 | 1.38 | 1.08 | 1.32 | 0.99 | 1.09 | 0.95 | 
| layer_4 | 1.24 |  **0.70** | 1.19 | 1.12 | 1.09 | 1.26 | 1.23 | 0.98 | 
| layer_5 | 1.02 |  **0.66** | 0.89 | 1.10 |  **0.74** | 0.92 | 0.97 |  **0.78** | 

**Bold** indicates suppressor heads (ratio < 0.8, p < 0.01)

### Identified Position 1 Suppressors

Found **11 suppressor heads** that significantly reduce attention to slot 0:

| Layer | Head | TO Ratio | Effect Size | p-value |
|-------|------|----------|-------------|----------|
| layer_0 | 4 | 0.777 | 0.0328 | 2.43e-110 |
| layer_0 | 6 | 0.385 | 0.0214 | 0.00e+00 |
| layer_5 | 1 | 0.663 | 0.0116 | 6.47e-83 |
| layer_0 | 7 | 0.536 | 0.0116 | 1.46e-97 |
| layer_1 | 0 | 0.509 | 0.0110 | 4.58e-63 |
| layer_5 | 4 | 0.743 | 0.0095 | 1.56e-55 |
| layer_0 | 2 | 0.690 | 0.0083 | 2.27e-35 |
| layer_5 | 7 | 0.775 | 0.0083 | 2.57e-47 |
| layer_4 | 1 | 0.697 | 0.0080 | 8.75e-42 |
| layer_0 | 1 | 0.716 | 0.0045 | 3.25e-98 |

### Identified Position 1 Amplifiers

Found **17 amplifier heads** that increase attention to slot 0:

| Layer | Head | TO Ratio | Effect Size | p-value |
|-------|------|----------|-------------|----------|
| layer_2 | 2 | 1.410 | 0.0167 | 1.71e-41 |
| layer_3 | 2 | 1.382 | 0.0159 | 2.14e-17 |
| layer_4 | 5 | 1.262 | 0.0156 | 6.77e-07 |
| layer_3 | 1 | 1.393 | 0.0143 | 1.12e-09 |
| layer_2 | 5 | 1.542 | 0.0135 | 2.96e-16 |
| layer_0 | 5 | 1.454 | 0.0129 | 1.14e-13 |
| layer_1 | 6 | 1.457 | 0.0112 | 1.39e-07 |
| layer_4 | 6 | 1.234 | 0.0084 | 3.14e-08 |
| layer_3 | 4 | 1.316 | 0.0084 | 4.33e-11 |
| layer_4 | 0 | 1.241 | 0.0080 | 3.52e-05 |

## Interpretation

Out of 48 total attention heads:
- **11** are position 1 suppressors
- **17** are position 1 amplifiers
- **20** show no significant differential treatment

**Net effect: More heads amplify slot 0 than suppress it.**

Attention patterns alone do not explain the slot 0 bias.
The root cause must lie elsewhere (training distribution, output head, etc.).

### Layer-by-Layer Pattern

- **layer_0**: 6 suppressors, mean ratio = 0.819
- **layer_1**: 1 suppressors, mean ratio = 0.969
- layer_2: No suppressors, mean ratio = 1.076
- layer_3: No suppressors, mean ratio = 1.163
- **layer_4**: 1 suppressors, mean ratio = 1.100
- **layer_5**: 3 suppressors, mean ratio = 0.886

## Visualizations

- `figures/17_per_head_ratios.png` - TO/FROM ratios for each head
- `figures/17_suppressor_heatmap.png` - Layer x Head suppressor map
- `figures/17_extreme_heads.png` - Most extreme suppressors and amplifiers
- `figures/17_layer_aggregate.png` - Per-layer aggregate statistics

## Conclusion

**PARTIAL: Some attention heads suppress slot 0, but effect is limited.**

The attention patterns show some differential treatment, but
this alone may not fully explain the slot 0 bias.
Other factors (training distribution, domino ordering) likely contribute.
