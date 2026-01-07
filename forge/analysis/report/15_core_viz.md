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

## 15c: Pareto Frontier

### Key Question
Which hands offer the best risk-return tradeoff?

### Method
- Pareto optimality: max E[V], min σ(V)
- A hand is Pareto-optimal if no other hand dominates it
- Dominated = another hand has higher E[V] AND lower σ(V)

### Key Findings

#### Extreme Dominance

**Only 3 hands (1.5%) are Pareto-optimal**:
- All have E[V] = 42 (maximum) and σ(V) = 0 (no risk)
- 197 hands (98.5%) are dominated

This is a consequence of the **inverse risk-return relationship**:
- High E[V] hands also have low σ(V)
- The best hands dominate almost everything else

#### Pareto Frontier Shape

Unlike typical portfolio theory (upward-sloping frontier):
- Texas 42 has a **degenerate** Pareto frontier
- It collapses to a few points at E[V]=42, σ(V)=0
- No meaningful risk-return tradeoff exists

#### Optimal Hand Characteristics

The 3 Pareto-optimal hands:
| Feature | Mean |
|---------|------|
| n_doubles | 2.33 |
| trump_count | 1.67 |

These are the "perfect" hands with deterministic outcomes.

### Implications for Bidding

1. **No risk-return tradeoff**: Just maximize E[V]
2. **Perfect hands exist**: Some hands guarantee +42
3. **Most hands are dominated**: Better alternatives exist in the sample

### Files Generated

- `results/tables/15c_pareto_frontier.csv` - All hands with Pareto classification
- `results/figures/15c_pareto_frontier.png` - Visualization with frontier

---

## 15d: Phase Transition

### Key Question
How does move consistency change as the game progresses?

### Method
- Data source: 11c stability analysis (best-move consistency by depth)
- Depth = dominoes remaining (28 = start, 0 = end)
- Consistency = percentage of states where best move is unique

### Key Findings

#### Three Phases of the Game

| Phase | Depth | Dominoes Played | Consistency | # States |
|-------|-------|-----------------|-------------|----------|
| **Early game** | 24-28 | 0-4 | 40% | 18 |
| **Mid-game** | 5-23 | 5-23 | 22% | 147,529 |
| **End-game** | 0-4 | 24-28 | **100%** | 19,472 |

#### Game Progression

1. **Opening (first 4 dominoes)**:
   - Few unique states exist (only 18)
   - Consistency around 40%
   - Declarer controls the game

2. **Mid-game (dominoes 5-23)**:
   - Maximum uncertainty phase
   - Consistency drops to 22% average
   - Minimum consistency 0% at depth 18
   - Multiple good strategies often exist
   - Game is in "chaotic" phase

3. **End-game (last 5 dominoes)**:
   - Consistency rises to **100%**
   - 19,472 unique states
   - Outcomes largely locked in
   - Mechanical, deterministic play

### Interpretation

The phase transition reflects **information revelation**:
- Early: Few cards played, but opener sets tempo
- Mid: Hands revealed, many strategic options
- Late: Most cards known, outcome determined

### Implications for Play

1. **Opening matters most**: Declarer's first few moves set the trajectory
2. **Mid-game is chaotic**: Multiple good strategies exist - don't overoptimize
3. **Endgame is mechanical**: Outcomes are largely fixed by this point

### Files Generated

- `results/figures/15d_phase_transition.png` - Progress-based view (dominoes played)
- `results/figures/15d_phase_by_depth.png` - Depth-based view with state counts
