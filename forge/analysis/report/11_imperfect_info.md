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

*Analysis date: 2026-01-06*
