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

## 11e: Contest State Distribution

### Key Question
What's P(Team 0 captures) for each count domino?

### Method
For each base_seed with fixed P0 hand across 3 opponent configurations:
1. Track which counts are held by Team 0 (P0+P2) vs Team 1 (P1+P3)
2. Use V distribution as proxy for capture outcomes
3. Estimate 5-vector of capture probabilities

### Key Findings

| Count | Mean P(capture) | Std | V when Team 0 holds | V when Team 1 holds | V Diff |
|-------|-----------------|-----|---------------------|---------------------|--------|
| 3-2 | 0.43 | 0.21 | 12.3 | 16.6 | **-4.2** |
| 4-1 | 0.34 | 0.26 | 12.7 | 14.9 | **-2.2** |
| 5-0 | 0.28 | 0.27 | 15.3 | 13.1 | +2.2 |
| 5-5 | 0.44 | 0.23 | 20.1 | 5.4 | **+14.7** |
| 6-4 | 0.30 | 0.24 | 18.2 | 11.2 | **+7.0** |

**Key Insights**:

1. **5-5 (double-five) is the most valuable count**: +14.7 point advantage when Team 0 holds it. This 10-point domino is often a trump-stopper and hard to steal.

2. **6-4 is second most valuable**: +7.0 point advantage. Also 10 points and often protected by holding the 6-suit.

3. **5-0 provides modest advantage**: +2.2 points when held. Being a 5-point count, less impactful but controllable.

4. **3-2 and 4-1 show *negative* holding advantage**: Counterintuitively, Team 0 does *worse* when holding these. Possible explanations:
   - These low-value counts often appear in weak hands overall
   - Opponent hands with these counts may have compensating strengths
   - Small sample size effect

5. **All counts are contested**: Mean capture probabilities range 0.28-0.44, all far from deterministic. No count is a "lock" based on ownership alone.

### Capture Probability Correlations

| | 3-2 | 4-1 | 5-0 | 5-5 | 6-4 |
|---|-----|-----|-----|-----|-----|
| 3-2 | 1.00 | 0.08 | -0.02 | **0.24** | 0.04 |
| 4-1 | 0.08 | 1.00 | 0.00 | -0.02 | 0.08 |
| 5-0 | -0.02 | 0.00 | 1.00 | **0.16** | 0.00 |
| 5-5 | **0.24** | -0.02 | **0.16** | 1.00 | 0.01 |
| 6-4 | 0.04 | 0.08 | 0.00 | 0.01 | 1.00 |

**Insight**: Capture probabilities are largely independent across counts. The 5-5 shows weak positive correlation with 3-2 (0.24) and 5-0 (0.16), suggesting hands that capture the double-five also tend to capture other 5-suit counts.

### Implications for Bidding

1. **5-5 is king**: Holding the double-five provides the largest expected value swing (+14.7 points)
2. **10-point counts matter more**: 5-5 and 6-4 provide larger advantages than 5-point counts
3. **Don't overvalue low counts**: Holding 3-2 or 4-1 doesn't predict winning - other factors dominate
4. **Count control is contested**: Even when holding a count, capture is ~40-45% likely (not guaranteed)

### Files Generated

- `results/tables/11e_contest_state_by_seed.csv` - Per-seed capture probabilities
- `results/tables/11e_count_ownership.csv` - Ownership statistics
- `results/tables/11e_capture_probabilities.csv` - 5-vector statistics
- `results/tables/11e_capture_correlations.csv` - Correlation matrix
- `results/figures/11e_contest_state_distribution.png` - Visualization

---

## 11f: Hand Features → E[V] Regression

### Key Question
What hand features predict expected value (E[V])?

### Method
Linear regression with features extracted from P0's hand:
- Number of doubles
- Trump count (dominoes containing trump pip)
- High dominoes (6-high, 5-high, 4-high)
- Count points held
- Has trump double
- Max suit length
- Total pip count

### Key Findings (200 seeds)

#### Model Performance

| Metric | Value |
|--------|-------|
| R² | **0.247** |
| CV R² | 0.182 ± 0.08 |

**Insight**: Hand features explain only ~25% of E[V] variance. The remaining 75% comes from opponent hands (imperfect information). This quantifies the "luck factor" - even with a great hand, opponent distribution matters enormously.

#### Feature Correlations with E[V]

| Feature | Correlation |
|---------|-------------|
| n_doubles | **+0.40** |
| has_trump_double | **+0.24** |
| trump_count | **+0.23** |
| count_points | +0.20 |
| n_6_high | **-0.16** |
| max_suit_length | -0.08 |
| n_5_high | +0.08 |
| total_pips | +0.04 |

**Key Insights**:

1. **Doubles are king**: The strongest predictor of E[V] is number of doubles (+0.40). Each double adds ~6.4 points expected value.

2. **Trump suit matters**: Both trump_count (+0.23) and has_trump_double (+0.24) are strong predictors. Having the trump double alone adds ~2.2 points.

3. **6-high is a trap**: Counterintuitively, n_6_high has *negative* correlation (-0.16). Having 6-high dominoes without trump suit strength may make them vulnerable to capture.

4. **Count points modestly helpful**: +0.20 correlation - holding counts helps but isn't decisive.

5. **Total pips irrelevant**: Near-zero correlation (+0.04). Raw hand strength doesn't predict success.

### The Napkin Formula

```
E[V] ≈ -4.1 + 6.4×(doubles) + 3.2×(trump_count) + 2.2×(trump_double) - 1.2×(6_highs)
```

**Example applications**:
- 0 doubles, 2 trumps, no trump double, 1 six-high: E[V] ≈ -4.1 + 6.4 + 0 - 1.2 = +1.1
- 2 doubles, 3 trumps, trump double, 0 six-high: E[V] ≈ -4.1 + 12.8 + 9.6 + 2.2 = +20.5
- 3 doubles (one is trump): E[V] ≈ -4.1 + 19.2 + 3.2 + 2.2 = +20.5

### Implications for Bidding

1. **Count doubles first**: The most reliable bidding signal. Each double is worth ~6 points expected value.

2. **Trump length matters but isn't everything**: 3 trumps with no doubles may be worse than 2 doubles with 1 trump.

3. **Be wary of "strong" hands**: 6-high dominoes can be liabilities if you're not calling that suit.

4. **R² = 0.25 means uncertainty**: Even optimal bidding has 75% unexplained variance from opponent hands.

### Files Generated

- `results/tables/11f_hand_features_by_seed.csv` - Per-seed features and V
- `results/tables/11f_feature_correlations.csv` - Correlation analysis
- `results/tables/11f_regression_coefficients.csv` - Model coefficients
- `results/tables/11f_napkin_formula.csv` - Formula parameters
- `results/figures/11f_hand_features_to_ev.png` - Visualization

---

## 11g: Hand Features → Count Locks

### Key Question
What hand features predict count locks (consistently capturing a count across opponent configurations)?

### Method
For each hand, track whether Team 0 captures each count in all 3 opponent configurations. Regress lock rate against hand features.

### Key Findings (Preliminary - 10 seeds)

**Note**: Small sample size (n=10) causes overfitting. Full analysis needed.

#### Lock Rates by Count

| Count | Avg Lock Rate | % Fully Locked |
|-------|---------------|----------------|
| 3-2 | 0.37 | 0% |
| 4-1 | 0.27 | 0% |
| 5-0 | 0.10 | 0% |
| 5-5 | 0.37 | 0% |
| 6-4 | 0.20 | 0% |

**Insight**: No count was fully locked in this sample. Count control is contested, not deterministic.

#### Does Holding a Count Predict Locking It?

| Count | Holding→Lock Correlation |
|-------|-------------------------|
| 5-0 | **+0.89** (strong) |
| 6-4 | **+0.70** (strong) |
| 3-2 | **+0.67** (strong) |
| 4-1 | +0.47 (moderate) |
| 5-5 | +0.37 (weak) |

**Insight**: Holding a count strongly predicts locking it for 5-0, 6-4, and 3-2. The 5-5 is hardest to lock even when held.

#### Feature Correlations with Total Lock Rate

| Feature | Correlation |
|---------|-------------|
| total_pips | **-0.93** |
| trump_count | **+0.56** |
| n_6_high | -0.24 |

**Insights** (tentative):
1. **High total pips = fewer locks**: Counterintuitive but consistent with 11f. Strong hands by pip count may lack trump control.
2. **Trump count helps**: More trumps = more control = more locks.
3. **The 5-5 paradox**: Holding it barely predicts locking it (0.37), suggesting opponents can often steal it.

### Files Generated

- `results/tables/11g_count_locks_by_seed.csv` - Per-seed lock rates
- `results/tables/11g_lock_correlations.csv` - Feature correlations
- `results/tables/11g_per_count_predictors.csv` - Per-count analysis
- `results/tables/11g_regression_coefficients.csv` - Model coefficients
- `results/figures/11g_hand_features_to_locks.png` - Visualization

---

## 11h: Path Divergence Analysis

### Key Question
When do paths diverge across opponent configs?

### Finding
**This analysis is redundant with 11c** (Best Move Stability), which answered the same question more efficiently.

From 11c:

| Depth Range | Consistency | Interpretation |
|-------------|-------------|----------------|
| 0-4 (endgame) | **100%** | Paths never diverge |
| 5-8 (late) | **50%** | Moderate divergence |
| 9-16 (mid) | **22%** | High divergence |
| 17+ (early) | **10%** | Maximum divergence |

**Conclusion**: Paths diverge almost immediately (10% consistency at depth 17+). By endgame, paths are deterministic (100% consistency).

---

*Analysis date: 2026-01-07*
