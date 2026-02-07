# 15: Core Visualizations

Publication-quality visualizations of key findings.

> **Epistemic Status**: This report visualizes findings from oracle (perfect-information minimax) data. All correlations, clusters, and phase transitions describe oracle outcome distributions. The "inverse risk-return relationship" and "phase transition" are properties of the oracle game tree. Whether these patterns characterize human gameplay dynamics is untested.

## 15a: Risk-Return Scatter Plot

### The Headline Finding

**Correlation**: r = -0.38 (95% CI: [-0.49, -0.26], p < 0.001)

Oracle data exhibits an **inverse risk-return relationship**:
- Good hands (high oracle E[V]) have LOWER variance (low oracle σ(V))
- Bad hands have HIGHER oracle variance
- This is the opposite of typical financial markets

**Note**: This describes oracle (perfect-information) outcome variance, not necessarily human game outcomes.

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

### Interpretation (Oracle Dynamics)

**Why inverse correlation in oracle data?**

Good hands provide **control** in oracle simulations:
- Many doubles → guaranteed trick winners
- Strong trumps → can trump opponents
- Less dependence on opponent configuration
- More predictable oracle outcomes

Bad hands leave oracle outcomes to **chance**:
- Dependent on unknown opponent holdings in marginalized data
- High variance in oracle V across opponent configurations
- Some opponent configurations are favorable, others disastrous

**Note**: This interpretation describes oracle game tree dynamics. Whether human players experience similar "control vs chance" dynamics is untested.

### Key Visual Elements (Oracle Data)

1. **Downward slope**: Clear negative trend line in oracle data
2. **Color gradient**: Higher n_doubles → upper left (high oracle E[V], low oracle risk)
3. **Empty upper right**: No hands with both high oracle E[V] AND high oracle risk
4. **Dense lower right**: Many mediocre hands cluster here in oracle feature space

---

## 15b: UMAP Hand Space

### Key Question
How do hands cluster in oracle feature space? Are there distinct archetypes?

### Method
- UMAP (Uniform Manifold Approximation and Projection)
- 10 features → 2D embedding
- n_neighbors=15, min_dist=0.1, metric='euclidean'

### Key Findings (Oracle Feature Space)

#### Hand Space is Continuous in Oracle Data

UMAP reveals **no sharp clusters** of hand archetypes in oracle feature space:
- Hands form a continuous manifold
- Gradual transitions between high and low oracle E[V] hands
- No distinct "hand types" in oracle data - more like a spectrum

#### Feature Correlations with UMAP

| Feature | UMAP1 Corr | UMAP2 Corr |
|---------|------------|------------|
| has_trump_double | 0.57 | 0.67 |
| n_voids | 0.44 | -0.51 |
| trump_count | 0.33 | 0.42 |
| n_6_high | 0.23 | 0.10 |
| n_doubles | 0.17 | -0.01 |

**Oracle E[V] vs UMAP1**: r = 0.23 (modest gradient in embedding space)

#### Extreme Hands Location (Oracle Data)

- **Best hand (oracle E[V]=42)**: Located in high-doubles region
- **Worst hand (oracle E[V]=-29)**: Located in low-doubles region
- High/low oracle risk hands also separate spatially

### Interpretation (Oracle Feature Space)

**No natural hand archetypes in oracle data** - the oracle feature space is continuous:
1. Hands can't be categorized into distinct "types" based on oracle E[V]
2. Feature importance is a gradient in oracle space, not categories
3. UMAP confirms the linear relationships found in regression on oracle outcomes

**Note**: Whether human players perceive or should use hand "archetypes" is a separate question from this oracle feature space analysis.

### Files Generated

- `results/tables/15b_umap_coordinates.csv` - UMAP coordinates for all 200 hands
- `results/figures/15b_umap_hand_space.png` - Side-by-side E[V] and σ(V) coloring
- `results/figures/15b_umap_doubles.png` - Colored by n_doubles
- `results/figures/15b_umap_annotated.png` - With extreme hands labeled

---

## 15c: Pareto Frontier

### Key Question
Which hands offer the best oracle risk-return tradeoff?

### Method
- Pareto optimality: max oracle E[V], min oracle σ(V)
- A hand is Pareto-optimal if no other hand dominates it in oracle metrics
- Dominated = another hand has higher oracle E[V] AND lower oracle σ(V)

### Key Findings (Oracle Data)

#### Extreme Dominance in Oracle Metrics

**Only 3 hands (1.5%) are Pareto-optimal in oracle space**:
- All have oracle E[V] = 42 (maximum) and oracle σ(V) = 0 (no oracle risk)
- 197 hands (98.5%) are dominated in oracle metrics

This is a consequence of the **inverse oracle risk-return relationship**:
- High oracle E[V] hands also have low oracle σ(V)
- The best hands dominate almost everything else in oracle space

#### Pareto Frontier Shape (Oracle)

Unlike typical portfolio theory (upward-sloping frontier):
- Oracle data shows a **degenerate** Pareto frontier
- It collapses to a few points at oracle E[V]=42, σ(V)=0
- No meaningful risk-return tradeoff exists in oracle outcomes

#### Optimal Hand Characteristics (Oracle)

The 3 Pareto-optimal hands in oracle space:
| Feature | Mean |
|---------|------|
| n_doubles | 2.33 |
| trump_count | 1.67 |

These are the hands with deterministic oracle outcomes (σ(V)=0 across all opponent configurations).

### Hypothetical Implications for Bidding (Oracle-Derived)

The following are hypotheses extrapolated from oracle analysis. **None have been validated against human gameplay.**

1. **Hypothesis**: No oracle risk-return tradeoff means bidders might focus solely on maximizing E[V]
2. **Hypothesis**: "Perfect" hands (oracle E[V]=42, σ(V)=0) might guarantee wins in human play
3. **Hypothesis**: Most hands being dominated in oracle space might inform human hand evaluation

**Note**: Human bidding involves strategic considerations (signaling, deception, partner coordination) not captured by oracle Pareto analysis.

### Files Generated

- `results/tables/15c_pareto_frontier.csv` - All hands with Pareto classification
- `results/figures/15c_pareto_frontier.png` - Visualization with frontier

---

## 15d: Phase Transition

### Key Question
How does oracle best-move consistency change as the game progresses?

### Method
- Data source: 11c stability analysis (oracle best-move consistency by depth)
- Depth = dominoes remaining (28 = start, 0 = end)
- Consistency = percentage of oracle states where oracle best move is unique

### Key Findings (Oracle Game Tree)

#### Three Phases of the Oracle Game Tree

| Phase | Depth | Dominoes Played | Oracle Consistency | # States |
|-------|-------|-----------------|-------------------|----------|
| **Early game** | 24-28 | 0-4 | 40% | 18 |
| **Mid-game** | 5-23 | 5-23 | 22% | 147,529 |
| **End-game** | 0-4 | 24-28 | **100%** | 19,472 |

#### Oracle Game Progression

1. **Opening (first 4 dominoes)**:
   - Few unique oracle states exist (only 18)
   - Oracle consistency around 40%
   - Declarer controls the oracle game tree

2. **Mid-game (dominoes 5-23)**:
   - Maximum oracle uncertainty phase
   - Oracle consistency drops to 22% average
   - Minimum oracle consistency 0% at depth 18
   - Multiple oracle-optimal strategies often exist
   - Oracle game tree is in "chaotic" phase

3. **End-game (last 5 dominoes)**:
   - Oracle consistency rises to **100%**
   - 19,472 unique oracle states
   - Oracle outcomes largely locked in
   - Mechanical, deterministic oracle play

### Interpretation (Oracle Game Tree)

The phase transition in oracle data reflects **information revelation in perfect-information play**:
- Early: Few cards played in oracle tree, but opener sets tempo
- Mid: Full hands visible in oracle; many oracle-optimal strategies exist
- Late: Oracle outcomes become deterministic as the game tree narrows

**Note**: In human play with hidden information, the "information revelation" dynamic differs significantly. Human mid-game involves inference about hidden cards, not the oracle's perfect information.

### Hypothetical Implications for Play (Oracle-Derived)

The following are hypotheses extrapolated from oracle game tree analysis. **None have been validated against human gameplay.**

1. **Hypothesis**: Opening moves may matter most - declarer's first few moves might set the trajectory
2. **Hypothesis**: Mid-game might have multiple good strategies - over-optimization may be counterproductive
3. **Hypothesis**: Endgame outcomes may be largely fixed by mid-game choices

**Note**: Human gameplay involves hidden information, bluffing, and inference. Whether the oracle's "order → chaos → resolution" pattern applies to human games is untested.

### Files Generated

- `results/figures/15d_phase_transition.png` - Progress-based view (dominoes played)
- `results/figures/15d_phase_by_depth.png` - Depth-based view with state counts

---

## Further Investigation

### Validation Needed

1. **Human risk-return relationship**: Does the inverse E[V]-σ(V) correlation hold for human game outcomes? The oracle correlation may not transfer to games with hidden information and suboptimal play.

2. **Human hand clustering**: Do experienced human players perceive hand "archetypes" even though oracle feature space is continuous? Human heuristics may differ from oracle optimality.

3. **Human game phases**: Does the oracle's "order → chaos → resolution" pattern characterize human gameplay? Human mid-game may be more or less chaotic depending on skill level.

### Methodological Questions

1. **Pareto sample dependence**: The 3 Pareto-optimal hands come from 200 samples. Would a larger sample reveal more optimal hands or confirm the degenerate frontier?

2. **UMAP hyperparameters**: n_neighbors=15, min_dist=0.1 are choices. Would different parameters reveal structure not visible in current embedding?

3. **Phase boundary definitions**: The "early/mid/late" boundaries are somewhat arbitrary. Is there a principled way to define phase transitions in the oracle game tree?

### Open Questions

1. **Why degenerate Pareto frontier?**: The collapse of the frontier to 3 points is striking. What game mechanism causes high oracle E[V] to coincide with low oracle σ(V)?

2. **Transfer of phases to AI training**: If the oracle game tree has distinct phases, should AI training treat them differently? Does curriculum learning across phases help?

3. **Human expert intuitions**: Do experienced 42 players have intuitions about "game phases" that align or conflict with the oracle phase structure?
