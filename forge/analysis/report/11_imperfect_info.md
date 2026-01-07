# 11: Imperfect Information Analysis

Analysis of how hidden opponent hands affect game outcomes, using marginalized oracle data.

## Data Source

**Marginalized shards**: `/mnt/d/shards-marginalized/train/`
- 201 unique base_seeds × 3 opponent configurations = 603 shards
- P0's hand fixed per base_seed, opponent hands vary

## 11a: Count Lock Rate / V Distribution Analysis

### Key Findings

#### V Variance Across Opponent Configurations

| Metric | Value |
|--------|-------|
| Mean V spread | 34.8 points |
| Median V spread | 34.0 points |
| Max V spread | 82 points |
| Hands with spread > 40 | 76 (38%) |
| Hands with spread < 10 | 22 (11%) |

**Insight**: Opponent hands matter enormously. The same P0 hand can swing from -42 to +40 depending on who holds what. Only 11% of hands are "stable" (spread < 10).

#### Count Holdings vs Expected Value

| Count Points Held | Mean V | Std V | Mean Spread | n |
|-------------------|--------|-------|-------------|---|
| 0 | 9.5 | 16.2 | 37.9 | 37 |
| 5 | 9.3 | 16.3 | 37.4 | 54 |
| 10 | 18.6 | 13.7 | 31.5 | 47 |
| 15 | 15.9 | 14.3 | 33.4 | 37 |
| 20 | 19.3 | 13.8 | 31.9 | 19 |
| 25 | 15.7 | 15.4 | 35.7 | 7 |

**Insights**:
- Holding 10-20 count points correlates with best mean V (18-19 points)
- Holding extreme amounts (0 or 25) gives lower expected V
- Holding more counts slightly reduces variance (spread)

#### Correlations

| Variables | Correlation |
|-----------|-------------|
| n_counts_held vs V_mean | 0.148 |
| p0_count_points vs V_mean | 0.197 |
| n_counts_held vs V_std | -0.055 |

**Insight**: Holding count dominoes is weakly correlated with better outcomes. The weak correlation (0.15-0.20) suggests count holdings explain only ~4% of V variance. Most of the game is determined by other factors (trump length, domino ranks, opponent distribution).

### Implications for Bidding

1. **High variance is the norm**: Most hands have 30+ point swings across opponent configs
2. **Count holdings help but don't guarantee**: Holding counts improves expected V by ~10 points on average
3. **Middle count holdings are best**: Holding 10-20 points of counts is optimal; 0 or 25 both underperform
4. **Risk assessment**: Use V_std as a measure of hand volatility

### Files Generated

- `results/tables/11a_base_seed_analysis.csv` - Full analysis of 201 base seeds

### Methodology

For each of 201 base_seeds:
1. Load all 3 opponent configurations
2. Extract root state V value (depth=28)
3. Compute V_mean, V_std, V_spread across the 3 configs
4. Track which count dominoes P0 holds

This analysis uses only root V values, not individual count capture tracking (which would require PV tracing through multi-million state shards).

---

## 11c: Best Move Stability Analysis

### Key Question
Does the optimal move change with opponent hands?

### Method
For states that appear in all 3 opponent configurations (same P0 hand, different opponent deals), check if argmax(Q) is consistent.

### Key Findings

| Metric | Value |
|--------|-------|
| Overall consistency | **54.5%** |
| Common states analyzed | 167,019 |
| States with consistent best move | 91,094 |

**Insight**: About half of all game positions have a "dominant" best move that's optimal regardless of opponent hands. The other half are situation-dependent - hidden information matters!

#### Consistency by Game Phase

| Depth | States | Consistent | Rate | Interpretation |
|-------|--------|------------|------|----------------|
| 0-4 (endgame) | 19,472 | 19,472 | **100%** | Perfect information at end |
| 5-8 (late game) | 136,805 | 68,851 | **50%** | Moderate uncertainty |
| 9-16 (mid game) | 10,557 | 2,368 | **22%** | High uncertainty |
| 17+ (early game) | 115 | 12 | **10%** | Maximum uncertainty |

**Key Insights**:

1. **Endgame is deterministic**: With ≤4 dominoes left, the best move is always the same regardless of opponent hands
2. **Mid-game is most complex**: 9-16 dominoes remaining shows only 22% consistency - this is where "reading" opponents matters most
3. **Early game is chaos**: With 17+ dominoes, best moves are almost entirely opponent-dependent

### Implications for Strategy

1. **Play heuristics early, calculate late**: In early game, general principles matter more than exact calculation since you can't know opponent hands
2. **Focus calculation on endgame**: Perfect play becomes possible once you've seen most cards
3. **Mid-game adaptation**: This is where inference about opponent hands pays off most

### Files Generated

- `results/tables/11c_stability_summary.csv` - Overall metrics
- `results/tables/11c_stability_by_depth.csv` - Breakdown by depth
- `results/tables/11c_best_move_stability_by_seed.csv` - Per-seed analysis
- `results/figures/11c_best_move_stability.png` - Visualization

---

## 11d: Q-Value Variance Analysis

### Key Question
How much do Q-values vary per position across opponent configurations?

### Method
For common states across 3 opponent configs, compute σ(Q) for each legal action. This measures confidence in move evaluation under uncertainty.

### Key Findings

| Metric | Value |
|--------|-------|
| Mean σ(Q) | **6.43 points** |
| Actions with high variance (σ > 5) | **76.8%** |
| States analyzed | 1,189 |
| Legal actions analyzed | 1,588 |

**Insight**: Most action evaluations vary significantly (6+ points) across opponent configurations. This means move quality is genuinely uncertain - a move that's great against one opponent distribution may be mediocre against another.

#### Q-Variance by Action Slot

| Slot | Mean σ(Q) | % High Variance | n |
|------|-----------|-----------------|---|
| 0 | 5.77 | 65% | 261 |
| 1 | 6.13 | 89% | 280 |
| 2 | 6.08 | 70% | 251 |
| 3 | 6.37 | 70% | 248 |
| 4 | 7.03 | 80% | 217 |
| 5 | 6.98 | 75% | 213 |
| 6 | 7.44 | 89% | 118 |

**Insight**: Later action slots (5-6) show higher variance than earlier slots (0-2). This suggests that as players have fewer options, the remaining choices become more situational.

#### Q-Variance by Depth

| Depth Range | Mean σ(Q) | Interpretation |
|-------------|-----------|----------------|
| 1-4 (endgame) | 4.5-6.0 | Lower uncertainty |
| 5-8 (late game) | 6.1-6.6 | Moderate uncertainty |
| 9-12 (mid game) | 6.5-9.0 | High uncertainty |
| 15+ (early game) | 9.2-21.2 | Maximum uncertainty |

**Insight**: Consistent with 11c, Q-value uncertainty peaks in early game and decreases toward endgame. Early game moves have σ(Q) > 20 points - opponent hands completely change which move is best.

### Implications for Strategy

1. **Trust early-game Q-values less**: With σ(Q) > 10, the "best" move by expected value may actually be worse than alternatives in many opponent configurations
2. **Endgame Q-values are reliable**: σ(Q) < 5 means move evaluations are stable across opponent distributions
3. **Prefer robust moves over optimal moves**: A move with slightly lower mean Q but lower σ(Q) may be preferable (risk aversion)

### Files Generated

- `results/tables/11d_q_variance_summary.csv` - Overall metrics
- `results/tables/11d_q_variance_by_slot.csv` - By action slot
- `results/tables/11d_q_variance_by_depth.csv` - By depth
- `results/tables/11d_q_variance_by_seed.csv` - Per-seed analysis
- `results/figures/11d_q_value_variance.png` - Visualization

---

*Analysis date: 2026-01-06*
