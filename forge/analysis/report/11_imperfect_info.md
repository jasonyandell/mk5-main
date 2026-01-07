# 11: Imperfect Information Analysis

Analysis of hand strength, count control, and decision stability under imperfect information.

## 11a: Count Lock Rate Analysis

**Question**: Which counts does a hand control? P(I capture count_i) across opponent configs.

**Method**: Trace principal variation from depth-0 for 100 seeds, track which team captures each count domino.

### Key Findings

1. **Overall holder advantage: 60.6%**
   - Holding a count domino gives ~10.6% advantage over random (50%)
   - This is the "ownership premium"

2. **10-counts are much easier to lock than 5-counts**
   - 10-count average lock rate: **75.4%**
   - 5-count average lock rate: **50.8%**
   - 10-counts (5-5, 6-4) are nearly locked when held; 5-counts are contested

3. **Individual count domino lock rates**

   | Domino | Points | Lock Rate | Status |
   |--------|--------|-----------|--------|
   | 5-5    | 10     | 80.0%     | LOCKED |
   | 6-4    | 10     | 70.8%     | LOCKED |
   | 4-1    | 5      | 56.9%     | CONTESTED |
   | 5-0    | 5      | 52.3%     | VULNERABLE |
   | 3-2    | 5      | 43.1%     | VULNERABLE |

### Implications for Bidding

- **10-counts are reliable**: If you hold 5-5 or 6-4, count those points
- **5-counts are speculative**: 3-2 and 5-0 are essentially coin flips
- **The 4-1 is marginal**: Slightly better than random, but not reliable

### Heritage Insight

> "When bidding, count your 10-counts as solid and your 5-counts as half."

This analysis validates the folk wisdom: 10-counts held = 20 points expected, 5-counts held = 7.5 points expected.

### Outputs
- Figure: `results/figures/11a_count_lock_rates.png`
- Table: `results/tables/11a_count_lock_rates.csv`
- Table: `results/tables/11a_lock_by_declaration.csv`

## 11b: V Distribution Per Hand

**Question**: What's E[V] and σ(V) for each hand across opponent configurations?

**Method**: Analyze 200 shards (500 hand-V pairs), extract hand features, compute correlations with initial V.

### Key Findings

1. **Best predictor: Trump count (r = 0.261)**
   - Number of trump dominoes in hand is the strongest single predictor of V
   - More trump = better expected outcome

2. **Trump count effect is dramatic**

   | Trump Count | E[V] | σ(V) | n |
   |-------------|------|------|---|
   | 0           | -4.3 | 25.2 | 81 |
   | 1           | -7.4 | 28.9 | 174 |
   | 2           | +1.0 | 28.8 | 151 |
   | 3           | +14.2 | 25.1 | 82 |
   | 4           | +27.8 | 15.9 | 12 |

   - Going from 0-1 trump (E[V] = -6.4) to 4+ trump (E[V] = +27.8) = **34 point swing**
   - Note: σ(V) decreases with more trump - stronger hands are more predictable

3. **Count points matter but less than trump**

   | Count Points | E[V] | σ(V) | n |
   |--------------|------|------|---|
   | 0            | -5.1 | 28.2 | 95 |
   | 5            | -7.8 | 28.4 | 136 |
   | 10           | +2.5 | 28.1 | 133 |
   | 15           | +3.3 | 27.2 | 88 |
   | 20           | +23.1 | 22.0 | 32 |
   | 25           | +10.5 | 27.4 | 15 |

   - 20+ count points: E[V] = +19.2 vs 0 count: E[V] = -5.1 = **24 point swing**

4. **Feature correlations with V**

   | Feature | Correlation |
   |---------|-------------|
   | n_trump | +0.261 |
   | count_points | +0.234 |
   | n_doubles | +0.195 |
   | total_pips | +0.193 |
   | high_pips | +0.094 |
   | n_suits | -0.099 |
   | longest_suit | -0.102 |

   - Positive: more trump, more count points, more doubles = better
   - Negative: more suits spread (short in all) = worse

### Implications for Bidding

- **Trump length is king**: Even more predictive than count points
- **High variance hands require caution**: σ(V) ≈ 27-29 for most hands means outcomes vary by ±27 points across opponent configs
- **4+ trump is a safe bid**: Lower variance (σ = 15.9) makes these hands more predictable

### Heritage Insight

> "Count your trumps, then your points."

This analysis validates that trump count should be weighted heavily in bidding decisions - it's the best single predictor of hand value.

### Outputs
- Figure: `results/figures/11b_v_distribution_per_hand.png`
- Table: `results/tables/11b_v_feature_correlations.csv`
- Table: `results/tables/11b_v_by_count_points.csv`
- Table: `results/tables/11b_v_by_trump.csv`
