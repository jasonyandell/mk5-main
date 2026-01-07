# 15: Core Visualizations

Publication-quality visualizations of key findings.

## 15a: Risk-Return Scatter Plot

### The Headline Finding

**Correlation**: r = -0.38 (95% CI: [-0.49, -0.26], p < 0.001)

Texas 42 exhibits an **inverse risk-return relationship**:
- Good hands (high E[V]) have LOWER variance (low σ(V))
- Bad hands have HIGHER variance
- This is the opposite of typical financial markets

### Visualizations Created

1. **15a_risk_return_scatter.png** (528 KB, 300 DPI)
   - Publication-quality scatter plot
   - Points colored by n_doubles
   - Regression line with correlation coefficient
   - Quadrant annotations
   - Statistics annotation box

2. **15a_risk_return_scatter.pdf** (28 KB, vector)
   - Vector format for publication
   - Scalable without loss

3. **15a_risk_return_clean.png** (278 KB, 300 DPI)
   - Simplified version
   - No quadrant annotations
   - Clean grid

4. **15a_risk_return_hexbin.png** (100 KB, 150 DPI)
   - Density visualization
   - Shows clustering patterns

### Interpretation

**Why inverse correlation?**

Good hands provide **control**:
- Many doubles → guaranteed trick winners
- Strong trumps → can trump opponents
- Less dependence on luck
- More predictable outcomes

Bad hands leave outcomes to **chance**:
- Dependent on unknown opponent holdings
- High variance in possible results
- Some opponent configurations are favorable, others disastrous

### Key Visual Elements

1. **Downward slope**: Clear negative trend line
2. **Color gradient**: Higher n_doubles → upper left (high E[V], low risk)
3. **Empty upper right**: No hands with both high E[V] AND high risk
4. **Dense lower right**: Many mediocre hands cluster here

---

## Remaining Tasks

- 15b: UMAP hand space visualization
- 15c: Pareto frontier
- 15d: Phase transition plots
