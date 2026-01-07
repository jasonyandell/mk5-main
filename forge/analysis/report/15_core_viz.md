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

## 15b: UMAP Hand Space

### Key Question
How do hands cluster in feature space? Are there distinct archetypes?

### Method
- UMAP (Uniform Manifold Approximation and Projection)
- 10 features → 2D embedding
- n_neighbors=15, min_dist=0.1, metric='euclidean'

### Key Findings

#### Hand Space is Continuous

UMAP reveals **no sharp clusters** of hand archetypes:
- Hands form a continuous manifold
- Gradual transitions between good and bad hands
- No distinct "hand types" - more like a spectrum

#### Feature Correlations with UMAP

| Feature | UMAP1 Corr | UMAP2 Corr |
|---------|------------|------------|
| has_trump_double | 0.57 | 0.67 |
| n_voids | 0.44 | -0.51 |
| trump_count | 0.33 | 0.42 |
| n_6_high | 0.23 | 0.10 |
| n_doubles | 0.17 | -0.01 |

**E[V] vs UMAP1**: r = 0.23 (modest gradient in embedding space)

#### Extreme Hands Location

- **Best hand (E[V]=42)**: Located in high-doubles region
- **Worst hand (E[V]=-29)**: Located in low-doubles region
- High/low risk hands also separate spatially

### Interpretation

**No natural hand archetypes** - the hand space is continuous:
1. You can't categorize hands into "types"
2. Feature importance is a gradient, not categories
3. UMAP confirms the linear relationships found in regression

### Files Generated

- `results/tables/15b_umap_coordinates.csv` - UMAP coordinates for all 200 hands
- `results/figures/15b_umap_hand_space.png` - Side-by-side E[V] and σ(V) coloring
- `results/figures/15b_umap_doubles.png` - Colored by n_doubles
- `results/figures/15b_umap_annotated.png` - With extreme hands labeled

---

## Remaining Tasks

- 15c: Pareto frontier
- 15d: Phase transition plots
