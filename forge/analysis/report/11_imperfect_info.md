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

### Key Findings

| Metric | Value |
|--------|-------|
| Overall consistency | 52.5% |
| States analyzed | 181 |
| States with consistent best move | 95 |

**Insight**: About half of game positions have a "dominant" best move that's optimal regardless of opponent hands. The other half are situation-dependent.

#### Consistency by Depth

| Depth | States | Consistent | Rate |
|-------|--------|------------|------|
| 1-4 | 8 | 8 | 100% |
| 5 | 57 | 32 | 56% |
| 6 | 46 | 17 | 37% |
| 7 | 29 | 16 | 55% |
| 8 | 20 | 16 | 80% |
| 9-16 | 21 | 6 | 29% |

**Insights**:
- Early/late game states (low remaining dominoes) have more stable best moves
- Mid-game (5-7 dominoes remaining) has lowest consistency
- This matches intuition: early game has obvious plays, mid-game is most complex

### Files Generated

- `results/tables/11c_stability_summary.csv` - Overall metrics
- `results/tables/11c_stability_by_depth.csv` - Breakdown by depth
- `results/tables/11c_best_move_stability_by_seed.csv` - Per-seed analysis
- `results/figures/11c_best_move_stability.png` - Visualization

---

*Analysis date: 2026-01-06*
