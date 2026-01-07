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

## 11i: Basin Convergence Analysis

### Key Question
Do different opponent configurations reach the same outcome basin (V category)?

### Method
Divide V into 5 basins (outcome categories):
- Big Loss: V < -20
- Loss: -20 ≤ V < -5
- Draw: -5 ≤ V < 5
- Win: 5 ≤ V < 20
- Big Win: V ≥ 20

For each hand, check if all 3 opponent configs land in the same basin.

### Key Findings (Preliminary - 10 seeds)

**Note**: Preliminary analysis with n=10. Full analysis recommended.

#### Basin Convergence

| Metric | Value |
|--------|-------|
| Basin convergence rate | **10%** |
| Mean V spread | 44.8 points |
| Median V spread | 48.0 points |
| Hands crossing ≥3 basins | 80% |

**Insight**: Only 10% of hands reach the same outcome basin regardless of opponent hands. Most hands (80%) swing across 3+ basins depending on who holds what.

#### Hand Dominance Classification

| Classification | Criteria | % of Hands |
|----------------|----------|------------|
| Dominant | V spread < 15 | **10%** |
| Moderate | 15 ≤ spread ≤ 35 | **10%** |
| Luck-dependent | V spread > 35 | **80%** |

**Insight**: 80% of hands are "luck-dependent" - the outcome can swing from Big Loss to Big Win based on opponent card distribution. Only 10% of hands are "dominant" with predictable outcomes.

### Interpretation

This is the strongest evidence yet for the **high luck factor** in Texas 42:

1. **Outcomes are not predictable from own hand**: Even a "great" hand can lose big or win big depending on opponents
2. **Bidding is inherently risky**: 80% of hands have 35+ point outcome swings
3. **Risk assessment > point estimation**: Understanding variance matters as much as expected value
4. **Basin spread quantifies uncertainty**: Most hands cross 3-4 basins - practically anything can happen

### Relationship to Other Analyses

- **11a** found mean V spread of 34.8 points - basin analysis confirms this is the norm
- **11f** found R² = 0.25 for hand features - basin analysis shows why (75% unexplained = luck)
- **11c** found 10% early-game consistency - basin analysis confirms outcomes diverge wildly

### Files Generated

- `results/tables/11i_basin_convergence_by_seed.csv` - Per-seed basin analysis
- `results/tables/11i_basin_convergence_summary.csv` - Summary statistics
- `results/figures/11i_basin_convergence.png` - Visualization

---

## 11s: σ(V) vs Hand Features Regression

### Key Question
What hand features predict outcome variance (risk)?

### Method
Regression with features: trump count, high dominoes, doubles → σ(V) and V_spread

### Key Findings (Preliminary - 10 seeds)

**Note**: Preliminary analysis with n=10. CV shows severe overfitting - full analysis required.

#### Feature Correlations with σ(V)

| Feature | Correlation with σ(V) | Interpretation |
|---------|----------------------|----------------|
| total_pips | **+0.63** | High pip count = HIGH risk |
| n_6_high | **+0.53** | More 6-highs = HIGH risk |
| trump_count | **-0.40** | More trumps = LOWER risk |
| has_trump_double | **-0.32** | Trump double = LOWER risk |
| n_doubles | **-0.25** | More doubles = LOWER risk |

**Key Insight**: The features that predict high expected value (doubles, trumps) ALSO predict low variance. Strong hands are both better AND safer.

#### E[V] vs σ(V) Relationship

| Metric | Value |
|--------|-------|
| Correlation E[V] vs σ(V) | **-0.55** |
| Correlation E[V] vs V_spread | **-0.60** |

**Critical Finding**: E[V] and σ(V) are **negatively correlated**. Good hands are not just higher EV - they're also more consistent. This has major bidding implications.

#### Risk Profiles

| Classification | Criteria | % | Avg Doubles | Avg Trumps | Avg Pips |
|----------------|----------|---|-------------|------------|----------|
| Low risk | spread < 20 | 10% | 3.0 | 3.0 | 29 |
| Medium risk | 20-45 | 30% | - | - | - |
| High risk | spread > 45 | 60% | 1.7 | 1.5 | 42 |

**Insight**: Low-risk hands have MORE doubles, MORE trumps, and FEWER total pips. High-risk hands are "pip-heavy" with few trumps.

### Implications for Bidding

1. **Doubles reduce risk**: Each double not only adds EV but also reduces variance
2. **Trump length is protective**: Long trump suits create predictable outcomes
3. **Beware high-pip hands**: Hands with many high pips but few trumps are volatile
4. **No risk-return tradeoff**: Unlike financial markets, Texas 42 has NEGATIVE risk-return correlation. Aggressive bidding on strong hands is doubly justified.

### The Risk Formula (Tentative)

```
V_spread ≈ -124 + 80×(trump_double) + 72×(6_highs) + 37×(singletons) - 21×(trump_count)
```

(Requires full-sample validation - these coefficients are unstable)

### Files Generated

- `results/tables/11s_sigma_v_by_seed.csv` - Per-seed features and variance
- `results/tables/11s_sigma_correlations.csv` - Feature correlations
- `results/tables/11s_regression_coefficients.csv` - Model coefficients
- `results/figures/11s_sigma_v_regression.png` - Visualization

---

## 11t: Lock Count → Bid Level Correlation

### Key Question
Does number of locked counts predict optimal bid level?

### Method
Track count holdings per hand, correlate with E[V], translate to bid recommendations.

### Key Findings (Full 201 seeds)

#### Correlations

| Metric | Value |
|--------|-------|
| n_counts_held vs E[V] | **+0.305** |
| total_count_points vs E[V] | **+0.197** |
| likely_locks vs E[V] | **+0.607** |

**Key Finding**: Holding more counts does predict higher E[V], confirming traditional bidding wisdom.

#### Bidding Heuristics by Count Holdings

| Counts Held | E[V] | E[V] Range | Recommended Bid | n |
|-------------|------|------------|-----------------|---|
| 0 | +5 | [-26, +37] | Pass | 42 |
| 1 | +14 | [-29, +42] | Pass | 80 |
| 2 | +18 | [-29, +42] | Low bid (~30) | 60 |
| 3 | +23 | [-28, +39] | ~30-31 | 17 |
| 4 | +19 | [19, 19] | Pass | 1 |

**Key Insight**: Each additional count held adds approximately **6 expected points**.

#### E[V] by Count Points Held

| Points Held | E[V] | V Spread | n |
|-------------|------|----------|---|
| 0 | +9.5 | 38 | 37 |
| 5 | +9.3 | 37 | 54 |
| 10 | +18.7 | 32 | 46 |
| 15 | +15.9 | 33 | 37 |
| 20 | +19.3 | 32 | 19 |
| 25 | +15.7 | 36 | 7 |

**Insight**: The sweet spot is 10-20 count points held (E[V] ~18-19).

#### Bid Recommendations

| Recommendation | % of Hands |
|----------------|------------|
| Pass | 70% |
| 30 | 10% |
| 31-34 | 10% |
| 38-42 | 10% |

**Insight**: 70% of hands don't justify bidding even with E[V] data. Conservative bidding is appropriate.

### The Count Rule of Thumb

```
E[V] ≈ 5 + 6 × (counts held)
```

**Examples**:
- 0 counts: E[V] ≈ 5 (pass)
- 2 counts: E[V] ≈ 17 (marginal bid)
- 3 counts: E[V] ≈ 23 (low bid justified)

### Implications for Bidding

1. **Counts matter**: Each count adds ~6 expected points
2. **But variance is high**: Range spans 60+ points regardless of count holdings
3. **Likely locks strongly predictive**: +0.607 correlation suggests lock potential matters more than mere possession
4. **Conservative bidding wise**: Only 30% of hands justify any bid

### Files Generated

- `results/tables/11t_lock_count_by_seed.csv` - Per-seed data
- `results/tables/11t_bidding_heuristics.csv` - Bid level heuristics
- `results/tables/11t_correlations.csv` - Correlation summary
- `results/figures/11t_lock_count_bid_level.png` - Visualization

---

## 11l: Lock Rate by Count Value

### Key Question
Are 10-point counts easier to lock than 5-point counts?

### Method
Define "lock" = Team 0 owns count in all 3 opponent configurations.
Compare lock rates between 5-point counts (3-2, 4-1, 5-0) and 10-point counts (5-5, 6-4).

### Key Findings (Full 201 seeds)

#### Lock Rate Comparison

| Type | Counts | Lock Rate | Capture Rate |
|------|--------|-----------|--------------|
| 5-point | 3-2, 4-1, 5-0 | **26.8%** | 52.6% |
| 10-point | 5-5, 6-4 | **23.5%** | 51.5% |
| Difference | | **-3.3%** | -1.1% |

**Finding**: 5-point counts are slightly EASIER to lock than 10-point counts! No significant difference in capture rates.

#### Individual Count Rankings (by Lock Rate)

| Count | Points | Lock Rate |
|-------|--------|-----------|
| 5-0 | 5 | **32.5%** |
| 4-1 | 5 | **27.0%** |
| 5-5 | 10 | 24.0% |
| 6-4 | 10 | 23.0% |
| 3-2 | 5 | 21.0% |

**Insight**: The 5-0 is easiest to lock, the 3-2 hardest. The 10-point counts (5-5, 6-4) are in the middle.

#### Lock Rates vs E[V]

| Metric | Correlation |
|--------|-------------|
| total_locks vs E[V] | **+0.305** |
| five_pt_locks vs E[V] | **+0.344** |
| ten_pt_locks vs E[V] | **+0.034** |

**Critical Finding**: Locking 5-point counts correlates MORE strongly with E[V] (+0.344) than locking 10-point counts (+0.034). This suggests 5-point count control is more strategically valuable than raw point totals might suggest.

#### E[V] by Total Locks

| Locks | E[V] | V Spread | n |
|-------|------|----------|---|
| 0 | +5 | 36 | 42 |
| 1 | +14 | 37 | 80 |
| 2 | +18 | 35 | 60 |
| 3 | +23 | 25 | 17 |
| 4 | +19 | 2 | 1 |

**Insight**: Each additional lock adds ~6-8 expected points. Locking 3 counts reduces V spread significantly (25 vs 35-37).

### Implications for Bidding

1. **Don't overvalue 10-point counts**: They're no easier to lock and their locks correlate weakly with E[V]
2. **The 5-0 is king for locks**: 32.5% lock rate - if you hold 5-0, you'll often capture it across all opponent hands
3. **Beware the 3-2**: Lowest lock rate (21%) despite being a count
4. **Lock quantity matters**: Each lock adds ~6 E[V] regardless of point value

### Files Generated

- `results/tables/11l_lock_by_count_by_seed.csv` - Per-seed data
- `results/tables/11l_lock_rates_summary.csv` - Count summaries
- `results/tables/11l_five_vs_ten_summary.csv` - Comparison
- `results/figures/11l_lock_by_count_value.png` - Visualization

---

## 11m: Lock Rate by Trump Length

### Key Question
Does holding trump lock more counts?

### Method
Count trump dominoes in P0's hand (any domino with trump pip). Track whether having the trump double affects lock rates.

### Key Findings (Full 201 seeds)

#### Trump Length vs Lock Rate

| Metric | Correlation |
|--------|-------------|
| trump_count vs total_locks | **-0.051** |
| trump_count vs capture_rate | **-0.070** |
| trump_count vs E[V] | **+0.229** |
| trump_count vs V_spread | **-0.094** |

**Critical Finding**: Trump LENGTH does NOT predict count locks! The correlation is essentially zero (-0.05). Having more trumps doesn't help you capture counts.

#### Lock Rate by Trump Count

| Trumps | Avg Locks | Capture Rate | E[V] | V Spread | n |
|--------|-----------|--------------|------|----------|---|
| 0 | 1.28 | 52% | +15 | 33 | 72 |
| 1 | 1.42 | 55% | +4 | 43 | 43 |
| 2 | 1.22 | 52% | +13 | 38 | 45 |
| 3 | 1.17 | 49% | +21 | 30 | 30 |
| 4 | 1.22 | 53% | +33 | 22 | 9 |
| 5 | 1.00 | 47% | +42 | 0 | 1 |

**Insight**: Lock rate is flat (~1.2-1.4) across trump lengths. However, E[V] increases and V spread decreases with more trumps - trump length helps you WIN but not by locking counts.

#### Trump Double Effect

| Has Trump Double | n | Avg Locks | Capture Rate | E[V] |
|------------------|---|-----------|--------------|------|
| No | 166 | 1.20 | 51.4% | +12.1 |
| Yes | 34 | **1.62** | **55.9%** | **+22.7** |
| **Difference** | | **+0.41** | **+4.5%** | **+10.7** |

**Key Finding**: The **trump double** matters enormously for count control:
- +0.41 additional locks on average
- +4.5% higher capture rate
- +10.7 E[V] advantage

#### Per-Count Correlations with Trump Length

| Count | Correlation |
|-------|-------------|
| 3-2 | -0.063 |
| 4-1 | -0.023 |
| 5-0 | -0.029 |
| 5-5 | -0.038 |
| 6-4 | +0.012 |

**Insight**: All counts show near-zero correlation with trump length. Trump dominoes don't help lock any specific count.

### Interpretation

This finding is **counterintuitive**. Conventional wisdom says "long trump = control = locks." The data shows:

1. **Trump length helps E[V] but not locks**: More trumps improve expected value (+0.23 correlation) but not by capturing counts
2. **The trump DOUBLE is what matters**: Having the highest trump (e.g., 6-6 when 6 is trump) is the key to count control
3. **Trump control ≠ count control**: Winning tricks (high E[V]) and capturing counts are different skills

**Why?** Possible explanation: Trump length lets you win tricks, but counts are won by having the count domino + timing. Having the trump double protects your count plays.

### Implications for Bidding

1. **Don't bid on trump length alone**: 4 trumps without the trump double may lock fewer counts than 1 trump with the double
2. **The trump double is critical**: Worth ~11 E[V] points and +0.4 locks on average
3. **Separate decisions**: "Can I win tricks?" (trump length) vs "Can I lock counts?" (trump double + count holdings)

### Files Generated

- `results/tables/11m_lock_by_trump_by_seed.csv` - Per-seed data
- `results/tables/11m_lock_by_trump_summary.csv` - Summary by trump count
- `results/tables/11m_correlations.csv` - Correlation summary
- `results/figures/11m_lock_by_trump.png` - Visualization

---

## 11u: Hand Ranking by Risk-Adjusted Value

### Key Question
Which hands are objectively strongest considering both expected value and risk?

### Method
Rank hands by utility function: `U = E[V] - λ×σ(V)`
- λ = 0: Risk-neutral (rank by E[V] only)
- λ = 1: Standard risk penalty
- λ = 2: Highly risk-averse

### Key Findings (Full 201 seeds)

#### Top 10 Hands by Risk-Adjusted Utility (λ=1)

| Rank | E[V] | σ(V) | Utility | Hand |
|------|------|------|---------|------|
| 1 | +42.0 | 0.0 | +42.0 | 6-4 4-4 4-3 4-1 4-0 2-2 1-1 |
| 2 | +42.0 | 0.0 | +42.0 | 6-4 5-4 4-4 4-3 4-0 3-3 3-0 |
| 3 | +42.0 | 0.0 | +42.0 | 5-4 4-4 4-3 2-1 2-0 1-1 1-0 |
| 4 | +41.3 | 0.9 | +40.4 | 5-2 4-4 4-0 3-1 2-2 2-0 0-0 |
| 5 | +41.3 | 0.9 | +40.4 | 6-6 6-1 5-2 5-0 3-3 2-0 0-0 |
| 6 | +39.3 | 0.9 | +38.4 | 5-3 5-1 5-0 4-4 2-2 2-1 1-0 |
| 7 | +40.0 | 1.6 | +38.4 | 6-5 6-4 5-5 4-2 3-3 2-1 0-0 |
| 8 | +38.7 | 0.9 | +37.7 | 6-6 5-5 5-1 5-0 3-3 2-1 1-1 |
| 9 | +40.0 | 2.8 | +37.2 | 6-2 5-5 5-2 4-0 3-0 2-2 1-0 |
| 10 | +36.7 | 0.9 | +35.7 | 5-5 5-4 5-2 3-3 2-1 2-0 1-0 |

**Key Pattern**: The top hands all have E[V] > 36 AND σ(V) < 3. They combine high expected value with consistency.

#### Ranking Stability Across Risk Preferences

| Comparison | Spearman ρ |
|------------|------------|
| λ=0 vs λ=1 | **0.923** |
| λ=0 vs λ=2 | **0.822** |
| λ=1 vs λ=2 | **0.974** |

**Critical Finding**: Rankings are **VERY STABLE** across risk preferences. The best hands by E[V] are also the best hands when accounting for risk. This is because E[V] and σ(V) are negatively correlated.

#### Dominated Hands Analysis (Pareto Frontier)

| Metric | Value |
|--------|-------|
| Total dominated hands | **197 / 200 (98.5%)** |
| Pareto-optimal hands | **3** |

**Finding**: Only **3 hands** are Pareto-optimal (no other hand has higher E[V] with lower σ(V)). All three have:
- E[V] = +42 (maximum)
- σ(V) = 0 (no variance across opponent configs)
- Average 2.3 doubles

**Interpretation**: These are the only "unambiguously best" hands - all others could be improved in at least one dimension.

#### Bidding Thresholds by Risk Preference

| Risk Level (λ) | % Would Bid (U ≥ 25) | Avg E[V] | Avg σ(V) | Avg Doubles |
|----------------|----------------------|----------|----------|-------------|
| 0 (neutral) | **30%** | +33.0 | 9.1 | 2.2 |
| 1 (standard) | **14%** | +36.6 | 3.6 | 2.4 |
| 2 (risk-averse) | **7%** | +39.3 | 2.0 | 2.4 |

**Insight**: Risk aversion dramatically reduces the number of "biddable" hands:
- Risk-neutral: 30% would bid
- Standard risk penalty: Only 14%
- Highly risk-averse: Just 7%

This explains the wide range of bidding styles in practice - conservative players bid ~7% of hands, aggressive players ~30%.

#### Feature Correlations with Utility

| Feature | λ=0 (E[V] only) | λ=1 (risk-adjusted) |
|---------|-----------------|---------------------|
| n_doubles | **+0.395** | **+0.359** |
| trump_count | **+0.229** | **+0.212** |
| count_points | +0.197 | +0.187 |
| n_6_high | **-0.161** | **-0.202** |
| total_pips | +0.035 | **-0.035** |

**Insights**:
1. **Doubles remain the best predictor** regardless of risk preference
2. **6-high becomes MORE negative** with risk adjustment (-0.16 → -0.20)
3. **Total pips FLIPS** from slightly positive to slightly negative with risk adjustment

#### Risk-Return Relationship

| Metric | Value |
|--------|-------|
| E[V] vs σ(V) correlation | **-0.381** |

**Critical Finding**: This is the **reverse** of typical financial markets. In Texas 42:
- Higher expected value → LOWER risk
- Strong hands are both better AND safer
- No risk-return tradeoff to navigate

### Implications for Bidding

1. **Risk preference matters less than you'd think**: Rankings are 92% correlated across risk levels. If a hand is good, it's good.

2. **The Pareto-optimal hands are exceptional**: Only 3/200 hands are unambiguously best. Recognize these when you see them.

3. **Conservative bidding is reasonable**: With risk adjustment, only 14% of hands justify bidding. "When in doubt, pass" is mathematically sound.

4. **Doubles predict everything**: They correlate with high E[V], low σ(V), and high utility regardless of λ.

### Files Generated

- `results/tables/11u_hand_rankings.csv` - Full rankings
- `results/tables/11u_top_hands.csv` - Top 20 hands by each λ
- `results/tables/11u_ranking_summary.csv` - Summary statistics
- `results/figures/11u_hand_ranking.png` - Visualization

---

## 11o: Robust vs Fragile Moves

### Key Question
Which moves are "always good" vs "depends on opponent hands"?

### Method
For common states across 3 opponent configurations:
- **Robust**: Same best move in all 3 configs
- **Fragile**: Best move varies by opponent configuration

### Key Findings (Full 201 seeds, 283K common states)

#### Best Move Classification

| Classification | Count | Percentage |
|----------------|-------|------------|
| **Robust** (same best move) | 274,750 | **97.0%** |
| **Fragile** (varies) | 8,529 | **3.0%** |

**Critical Finding**: The vast majority of positions (97%) have a clear "best" move regardless of opponent hands. Only 3% of positions have situationally-dependent optimal play.

**Note**: This is much higher than 11c's 54.5% because we're analyzing only **common states** (states reachable in all 3 configs), which tend to be later in the game where paths have converged.

#### Q-Variance by Move Type

| Move Type | Mean σ(Q) | Std σ(Q) |
|-----------|-----------|----------|
| Robust | 9.68 | 22.29 |
| Fragile | **69.70** | 14.56 |

**Finding**: Fragile moves have **7.2x more Q-variance** than robust moves. When the best move varies by opponent configuration, the Q-values are highly unstable.

#### Robustness by Game Depth

| Depth Range | Robust | Fragile | Total | Robust % |
|-------------|--------|---------|-------|----------|
| Endgame (0-4) | 3,592 | 0 | 3,592 | **100%** |
| Late (5-8) | 195,310 | 2,613 | 197,923 | **98.7%** |
| Mid (9-16) | 75,846 | 5,915 | 81,761 | **92.8%** |
| Early (17+) | 2 | 1 | 3 | 66.7% |

**Key Insight**: Robustness increases dramatically as the game progresses:
- **Endgame is deterministic**: 100% of positions have a robust best move
- **Late game is nearly certain**: 98.7% robust
- **Mid game is mostly settled**: 92.8% robust

#### Robustness by Action Slot

| Slot | Robust | Fragile | Robust % | Mean σ(Q) |
|------|--------|---------|----------|-----------|
| 0 | 70,558 | 13,943 | **83.5%** | 6.4 |
| 1 | 58,295 | 11,970 | **83.0%** | 6.4 |
| 2 | 46,937 | 12,652 | **78.8%** | 6.3 |
| 3 | 47,881 | 12,229 | **79.7%** | 6.7 |
| 4 | 42,746 | 11,885 | **78.2%** | 6.9 |
| 5 | 33,822 | 12,582 | **72.9%** | 6.6 |
| 6 | 22,677 | 9,587 | **70.3%** | 6.4 |

**Insight**: Earlier action slots (0-1) are more robust than later slots (5-6). When you have more dominoes, your first choices are more clearly correct.

### Reconciling with 11c

11c found only 54.5% best move consistency, while 11o finds 97% robust moves. The difference:

- **11c** counted **all states** across configs (including divergent paths)
- **11o** counts only **common states** (reachable from all 3 configs)

This reveals that:
1. Early in the game, paths diverge significantly (11c: only 10% consistency at depth 17+)
2. Where paths converge (common states), the optimal move is almost always clear (11o: 97%)
3. The "fragile" positions are concentrated where paths haven't yet converged

### Implications for Strategy

1. **Trust convergent positions**: Once a game state is reached regardless of opponent hands, the best move is almost certainly robust (97%)

2. **Be cautious at divergence points**: The 3% of fragile positions are where opponent-reading skills matter most

3. **Later dominoes require more judgment**: Slots 5-6 (when you have few dominoes left to play) are 30% less robust than slots 0-1

4. **Endgame calculation is reliable**: If you can calculate to depth 0-4, your analysis is 100% reliable regardless of hidden information

### Files Generated

- `results/tables/11o_robust_fragile_by_seed.csv` - Per-seed data
- `results/tables/11o_robust_fragile_summary.csv` - Summary statistics
- `results/tables/11o_robust_by_depth.csv` - Depth analysis
- `results/tables/11o_robust_by_slot.csv` - Slot analysis
- `results/figures/11o_robust_vs_fragile.png` - Visualization

---

## 11j: Basin Variance Analysis

### Key Question
How many outcome basins are reachable from a hand?

### Method
Divide V into 5 basins (Big Loss, Loss, Draw, Win, Big Win). For each hand across 3 opponent configs, count unique basins reached. "Converged" = same basin in all 3 configs.

### Key Findings (Full 201 seeds, 200 hands analyzed)

#### Basin Convergence

| Metric | Value |
|--------|-------|
| Hands converging to same basin | **37 / 200 (18.5%)** |
| Mean basin spread | **1.89** |
| Median basin spread | **2.0** |

**Key Finding**: Only 18.5% of hands reach the same outcome category regardless of opponent distribution. Most hands (81.5%) cross multiple outcome categories.

#### Distribution of Unique Basins Reached

| Basins Reached | Count | Percentage |
|----------------|-------|------------|
| 1 (converged) | 37 | **18.5%** |
| 2 | 102 | **51.0%** |
| 3 | 61 | **30.5%** |

**Insight**: The majority (51%) of hands span exactly 2 basins across opponent configs. Nearly a third (30.5%) span all 3 sampled basins.

#### High Variance vs Low Variance Hands

| Metric | Low Variance (Converged) | High Variance (Diverged) |
|--------|--------------------------|--------------------------|
| Count | 37 | 163 |
| Mean E[V] | **+30.9** | **+10.0** |
| Mean doubles | 2.1 | 1.6 |
| Mean trump count | 1.6 | 1.3 |
| % with trump double | 22% | 16% |

**Critical Finding**: Converged (safe) hands have:
- **3x higher E[V]** (+30.9 vs +10.0)
- More doubles (2.1 vs 1.6)
- More likely to have trump double (22% vs 16%)

This confirms the negative risk-return relationship: strong hands are both higher EV AND more predictable.

#### Feature Correlations with Basin Spread

| Feature | Correlation |
|---------|-------------|
| V_std | **+0.899** |
| V_mean | **-0.529** |
| n_doubles | **-0.181** |
| trump_count | **-0.183** |
| n_6_high | **+0.177** |
| count_points | -0.073 |

**Insights**:
1. **V_std perfectly tracks basin spread** (0.899) - variance and basin crossing are essentially the same metric
2. **High E[V] = low basin spread** (-0.529) - the best hands don't swing across categories
3. **More doubles/trumps = less spread** - the same features that predict E[V] also predict convergence
4. **6-high is risky** (+0.177) - hands heavy in 6-suit dominoes tend to span more basins

#### Safest Hands (Lowest Basin Spread)

Top 3 hands with E[V] = +42 and Spread = 0 (always Big Win):
1. 6-4 4-4 4-3 4-1 4-0 2-2 1-1 (4s trump, 3 doubles)
2. 6-4 5-4 4-4 4-3 4-0 3-3 3-0 (4s trump, 2 doubles)
3. 5-4 4-4 4-3 2-1 2-0 1-1 1-0 (4s trump, 2 doubles)

**Pattern**: All three are heavy in the trump suit with multiple doubles.

#### Riskiest Hands (Highest Basin Spread)

Top hand: Spread = 82 points, basins = Big Win / Loss / Big Loss
- E[V] = -3.3
- Hand: 6-5 5-5 4-3 3-3 3-2 3-0 2-2

**Pattern**: High-risk hands often have:
- Multiple 5s or 6s but not as trump
- Moderate doubling (2 doubles) but wrong suits
- Basins that span the entire range (Big Win to Big Loss)

### Relationship to Other Analyses

- **11i** (preliminary 10 seeds) found 10% convergence - full analysis shows 18.5%
- **11u** (Pareto analysis) found only 3 hands with σ(V) = 0 - these are the only guaranteed "same basin" hands
- **11s** found E[V] vs σ(V) correlation of -0.55 - 11j confirms this pattern with basin categories

### Implications for Bidding

1. **Recognize convergent hands**: Hands with 2+ doubles in trump suit tend to stay in the same outcome category
2. **Beware high-variance hands**: 82% of hands cross multiple outcome basins - bidding confidently is risky
3. **The 30/50/20 rule**: ~20% converge, ~50% span 2 basins, ~30% span 3+ basins
4. **E[V] and stability go together**: The highest EV hands are also the most predictable

### Files Generated

- `results/tables/11j_basin_variance_by_seed.csv` - Per-seed data
- `results/tables/11j_basin_variance_summary.csv` - Summary statistics
- `results/figures/11j_basin_variance.png` - Visualization

---

## 11k: Hand Classification Clustering

### Key Question
Can we cluster hands by outcome profile into meaningful categories?

### Method
K-means clustering on standardized (E[V], σ(V), n_unique_basins) feature vectors. Optimal k found via silhouette score.

### Key Findings (200 hands from 11j data)

#### Optimal Cluster Count

| k | Silhouette Score |
|---|------------------|
| 2 | 0.399 |
| 3 | 0.375 |
| 4 | 0.414 |
| 5 | 0.433 |
| 6 | 0.451 |
| 7 | 0.474 |

Statistically optimal k=7, but k=3 provides interpretable hand types.

#### Three Natural Hand Types

| Type | Count | % | E[V] | σ(V) | Basins | Spread |
|------|-------|---|------|------|--------|--------|
| **STRONG** | 35 | 18% | +33.7 | 4.4 | 1.0 | 10 |
| **VOLATILE** | 81 | 40% | +16.9 | 11.9 | 2.0 | 27 |
| **WEAK** | 84 | 42% | +2.7 | 22.7 | 2.7 | 53 |

**Key Finding**: Hands naturally cluster into three interpretable categories:
- **STRONG** (18%): High E[V], low variance, single basin outcome
- **VOLATILE** (40%): Medium E[V], medium variance, outcome varies
- **WEAK** (42%): Low E[V], high variance, unpredictable

#### Hand Features by Cluster

| Feature | STRONG | VOLATILE | WEAK |
|---------|--------|----------|------|
| n_doubles | **2.14** | 1.83 | 1.46 |
| trump_count | **1.66** | 1.40 | 1.11 |
| has_trump_double | **23%** | 22% | 10% |
| count_points | **10.43** | 10.00 | 7.92 |
| n_6_high | 1.34 | 1.74 | **1.90** |

**Insights**:
1. **Doubles separate STRONG from WEAK**: 2.1 vs 1.5 average
2. **Trump count matters**: STRONG have 50% more trumps than WEAK
3. **Trump double is key**: 23% of STRONG vs 10% of WEAK have it
4. **6-high is a liability**: WEAK hands have most 6-highs (1.9 avg)

#### Sample Hands

**STRONG** (bid confidently):
- E[V]=+42.0, σ=0.0: 6-4 5-4 4-4 4-3 4-0 3-3 3-0
- E[V]=+38.7, σ=0.9: 6-6 5-5 5-1 5-0 3-3 2-1 1-1

**VOLATILE** (bid cautiously):
- E[V]=+32.7, σ=10.5: 5-5 5-4 5-2 4-0 3-1 2-0 1-0
- E[V]=+18.0, σ=17.0: 6-5 6-2 6-1 5-5 4-2 3-0 0-0

**WEAK** (pass):
- E[V]=-12.0, σ=21.2: 6-5 6-0 5-4 5-2 3-2 3-0 1-1
- E[V]=+2.7, σ=22.7: 6-5 6-4 6-1 4-4 4-3 3-1 2-2

### Bidding Recommendations

| Type | Recommendation | Expected Outcome |
|------|----------------|------------------|
| **STRONG** | Bid 30-42 confidently | +33.7 ± 4.4 |
| **VOLATILE** | Cautious bid or pass | +16.9 ± 11.9 |
| **WEAK** | Pass | +2.7 ± 22.7 |

### The 18/40/42 Rule

- **18%** of hands are STRONG - bid with confidence
- **40%** of hands are VOLATILE - judgment call
- **42%** of hands are WEAK - pass and wait

This explains why experienced players pass most hands - 82% of hands are either weak or volatile.

### Relationship to Other Analyses

- **11j** (basin variance) provides the clustering features
- **11f** (hand features → E[V]) explains why doubles/trumps predict cluster membership
- **11u** (risk-adjusted ranking) confirms STRONG hands are both high EV and low risk

### Files Generated

- `results/tables/11k_hand_classification.csv` - Per-hand clusters
- `results/tables/11k_cluster_summary.csv` - Cluster statistics
- `results/tables/11k_silhouette_scores.csv` - Cluster quality metrics
- `results/figures/11k_hand_classification.png` - Visualization

---

## 11n: Decision Point Consistency (Preliminary)

### Key Question
Are critical decisions (Q-gap > 5) the same across opponent configs?

### Method
Track positions where Q-gap > threshold in ALL 3 opponent configs, then check if best move is consistent.

### Key Findings (Preliminary - 50 seeds, 84K states)

#### Critical Decision Frequency

| Condition | Count | Percentage |
|-----------|-------|------------|
| Critical in ALL configs | 545 | **0.6%** |
| Critical in ANY config | 22,150 | **26.3%** |

**Key Finding**: Only 0.6% of positions have Q-gap > 5 in ALL opponent configs. Most high-stakes decisions are opponent-dependent.

#### Consistency of Critical Decisions

| Outcome | Count | Percentage |
|---------|-------|------------|
| Same best move | 347 | **63.7%** |
| Different best moves | 198 | **36.3%** |

**Finding**: When a decision is critical in all configs, there's only 63.7% chance the best move is the same. Over a third of critical decisions depend on opponent hands.

#### Q-Gap Analysis

| Metric | Value |
|--------|-------|
| Mean Q-gap | 1.86 |
| Median Q-gap | 0.00 |
| % with Q-gap > 5 | 13.2% |
| % with Q-gap > 10 | 4.4% |

**Insight**: Most positions (87%) have Q-gap < 5 - the moves are roughly equivalent. Only 4.4% have Q-gap > 10 (high-stakes).

#### Critical Decisions by Depth

| Depth | Count | Percentage |
|-------|-------|------------|
| 5 (late game) | 396 | **72.7%** |
| 9 (mid game) | 148 | **27.2%** |
| 13 (early) | 1 | 0.2% |

**Insight**: Critical decisions concentrate in late game (depth 5). Early decisions rarely have large Q-gaps because outcomes depend heavily on future play.

### Implications for Strategy

1. **Most decisions don't matter much**: 87% of positions have Q-gap < 5. Playing "reasonably" is usually good enough.

2. **Critical decisions ARE opponent-dependent**: 36.3% of high-stakes positions have different best moves depending on who holds what.

3. **Late game is where it counts**: 73% of critical decisions occur at depth 5. Focus your calculation here.

4. **Opponent inference helps at key moments**: When Q-gap is large AND opponents' hands affect the answer (36% of the time), reading opponents is valuable.

### Relationship to Other Analyses

- **11c** found 54.5% overall best-move consistency
- **11o** found 97% robustness on common states
- **11n** adds nuance: for CRITICAL decisions specifically, only 64% are consistent

### Files Generated

- `results/tables/11n_decision_consistency_by_seed.csv` - Per-seed data
- `results/tables/11n_decision_consistency_summary.csv` - Summary
- `results/figures/11n_decision_consistency.png` - Visualization

---

## 11v: Hand Similarity Clustering

### Key Question
Do structurally similar hands have similar outcomes?

### Method
Cluster hands by FEATURES (doubles, trump count, count points, etc.), then measure within-cluster E[V] variance.

### Key Findings (200 hands)

#### Feature-Based Clusters

| Cluster | n | % | E[V] | σ(E[V]) | Characteristics |
|---------|---|---|------|---------|-----------------|
| Multi-Double/Trump-Heavy | 34 | 17% | +22.7 | 13.9 | 2.2 doubles, 2.4 trumps |
| Multi-Double/Trump-Light | 33 | 16% | +20.1 | 17.7 | 2.9 doubles, 0.4 trumps |
| Count-Rich/Six-Heavy | 37 | 18% | +13.0 | 10.6 | 17 count points, 2.7 6-highs |
| Few-Double/Count-Poor | 55 | 28% | +5.8 | 18.2 | 1.0 doubles, 4 count points |
| Mixed | 41 | 20% | +13.2 | 15.0 | Average features |

**Observation**: Multi-double hands have highest E[V] (~+21) regardless of trump count.

#### Within-Cluster Variance Analysis

| Metric | Value |
|--------|-------|
| Overall E[V] std | 16.62 |
| Within-cluster E[V] std | 15.08 |
| Variance reduction | **9%** |
| Improvement over random | **9%** |

**Critical Finding**: Feature clustering only explains **9% of E[V] variance**. Structurally similar hands do NOT guarantee similar outcomes.

#### Best and Worst Clusters

| Cluster | Variance Reduction | Interpretation |
|---------|-------------------|----------------|
| Count-Rich/Six-Heavy | **36%** | Most predictable |
| Multi-Double/Trump-Heavy | 16% | Moderately predictable |
| Multi-Double/Trump-Light | **-7%** | More variance than average |
| Few-Double/Count-Poor | **-10%** | Highly unpredictable |

**Insight**: Holding many counts makes outcomes more predictable, while few doubles/counts leads to extreme variance.

### Interpretation

This confirms the "luck factor" finding from 11a and 11f:

1. **Hand features explain ~25% of E[V]** (from 11f regression)
2. **Feature clustering explains ~9% of E[V] variance** (from 11v)
3. **Opponent distribution dominates** - the remaining 75%+ comes from who holds what

### Why Feature Similarity Fails

Two hands with identical (doubles, trumps, counts) can have very different outcomes because:
- The specific dominoes matter (which suits, which ranks)
- Opponent hands can be favorable or unfavorable
- The interaction between your hand and opponents' is unpredictable

### Implications for Bidding

1. **Don't assume similar = equivalent**: Two hands with "3 doubles, 2 trumps" can have wildly different outcomes.

2. **Features are necessary but not sufficient**: Count doubles/trumps/counts for general guidance, but recognize high variance.

3. **The 28% problem**: "Few-Double/Count-Poor" hands (28% of all hands) are the most unpredictable. Pass these.

4. **Count-rich hands are safest**: Best variance reduction (36%) - holding counts makes you more predictable.

### Relationship to Other Analyses

- **11f** found R² = 0.25 for features → E[V]
- **11k** clustered by outcomes (STRONG/VOLATILE/WEAK)
- **11v** clusters by features - confirms features don't fully determine outcomes

### Files Generated

- `results/tables/11v_hand_similarity.csv` - Per-hand clusters
- `results/tables/11v_cluster_summary.csv` - Cluster statistics
- `results/tables/11v_variance_analysis.csv` - Variance analysis
- `results/figures/11v_hand_similarity.png` - Visualization

---

## 11p: Path Similarity Analysis (DTW) (Preliminary)

### Key Question
How similar are V-trajectories across opponent configs?

### Method
For each hand across 3 opponent configurations:
1. Sample V distributions at depth levels (28, 24, 20, 16, 12, 8, 4, 1)
2. Compute mean V at each depth level
3. Compare depth-V trajectories using DTW and Pearson correlation

### Key Findings (Preliminary - 50 seeds)

#### Path Stability Summary

| Metric | Value |
|--------|-------|
| Mean trajectory correlation | **0.169** |
| Median trajectory correlation | 0.125 |
| Min trajectory correlation | **-0.980** |
| Mean DTW distance | 5.79 |

**Critical Finding**: V-trajectories show very LOW correlation across opponent configurations. The mean correlation of 0.169 indicates trajectories are almost uncorrelated.

#### Stability Categories

| Category | Count | Percentage |
|----------|-------|------------|
| High stability (corr > 0.9) | 3 | **6.0%** |
| Medium stability (0.7-0.9) | 3 | **6.0%** |
| Low stability (corr ≤ 0.7) | 44 | **88.0%** |

**Finding**: 88% of hands have low stability - the V trajectory through the game varies significantly based on opponent hands.

#### Root vs Terminal V Spread

| Depth | Mean Spread | Max Spread |
|-------|-------------|------------|
| Root (depth 28) | **40.9 points** | 82 points |
| Terminal (depth 1) | **3.0 points** | 9 points |

**Key Insight**: Games START with high V divergence (41 points) but CONVERGE toward similar endpoints (3 points). The path to get there varies wildly.

#### Path-Value Correlations

| Correlation | Value |
|-------------|-------|
| DTW vs root V spread | **+0.843** |
| Corr vs root V spread | **-0.725** |
| DTW vs terminal V spread | **+0.731** |

**Critical Finding**: Strong correlation (+0.84) between initial V spread and path divergence. Hands that start with high uncertainty in outcomes also have the most divergent trajectories.

### Reconciling with 11x (Information Value)

This seems to contradict 11x which found 75% action agreement:

- **11x**: At any given position, the best *move* is usually the same regardless of opponents (75%)
- **11p**: The overall *V trajectory* through the game diverges significantly (88% low correlation)

**Resolution**: Individual moves are robust, but the cumulative effect of different opponent distributions leads to very different game progressions. You often make the same move, but the consequences (V values at each stage) differ dramatically.

### Implications for Strategy

1. **Games converge at the end**: Terminal V spread is only 3 points despite 41 points at start. Endgames are predictable.

2. **Early/mid game trajectories diverge**: The path from start to end varies based on opponents, even if individual moves are similar.

3. **High-variance hands have divergent paths**: The +0.84 correlation means that hands with uncertain V also have the most unpredictable game progressions.

4. **Focus on endgame**: Since games converge, endgame calculation becomes increasingly reliable.

### Files Generated

- `results/tables/11p_path_similarity_by_seed.csv` - Per-seed data
- `results/tables/11p_path_similarity_summary.csv` - Summary
- `results/figures/11p_path_similarity.png` - Visualization

---

## 11q: Per-Hand PCA Analysis (Preliminary)

### Key Question
Is the 5D structure (V at multiple depths) preserved within a fixed hand?

### Method
For each hand across 3 opponent configurations:
1. Extract V statistics at depth levels (28, 24, 20, 16, 12, 8, 4, 1)
2. Build feature matrix: mean V, std V, and spread (max-min) at each depth
3. PCA to find intrinsic dimensionality

### Key Findings (Preliminary - 50 seeds)

#### PCA Variance Explained

| Component | Variance | Cumulative |
|-----------|----------|------------|
| PC1 | **45.6%** | 45.6% |
| PC2 | 17.3% | 62.9% |
| PC3 | 15.5% | 78.4% |
| PC4 | 7.4% | 85.8% |
| PC5 | 4.5% | 90.3% |

**Key Finding**: 5 components explain 90% of variance from 24 original features.

#### Dimensionality Metrics

| Metric | Value |
|--------|-------|
| Original features | 24 |
| Components for 90% variance | **5** |
| Components for 95% variance | 7 |
| Components for 99% variance | 10 |
| Effective dimensionality | **4.9** |
| Dimensionality compression | **4.8x** |

**Critical Finding**: Fixing P0's hand constrains the outcome manifold from 24 dimensions to ~5. This is a significant compression.

#### PC1 Loadings (Top Features)

| Feature | Loading |
|---------|---------|
| v_spread_d8 | **+0.385** |
| v_spread_d12 | +0.377 |
| v_spread_d4 | +0.366 |
| v_spread_d16 | +0.355 |
| v_spread_d20 | +0.325 |

**Insight**: PC1 is dominated by V SPREAD at mid-game depths. This represents the "uncertainty dimension" - how much V varies across opponent configurations.

#### V Spread by Depth

| Depth | Mean V Spread |
|-------|---------------|
| 28 (start) | **40.6** |
| 24 | 18.3 |
| 20 | 11.5 |
| 16 | 8.6 |
| 12 | 6.3 |
| 8 | 5.0 |
| 4 | 4.0 |
| 1 (end) | **3.0** |

**Key Pattern**: V spread decreases monotonically from 41 points at game start to 3 points at end. Games converge as they progress.

### Interpretation

1. **Manifold structure exists**: The 4.8x compression shows that hand-conditioned outcomes live on a ~5D manifold, not the full 24D feature space.

2. **Uncertainty is the main axis**: PC1 (45.6%) captures V spread - the primary variation between hands is how much their outcomes depend on opponents.

3. **Convergence is universal**: The V spread → depth relationship is consistent across hands, suggesting a shared funnel structure.

4. **Hand constrains but doesn't determine**: 5 dimensions still allow significant outcome variation within a fixed hand.

### Relationship to Other Analyses

- **11p** found 88% low-correlation trajectories - 11q explains this via the high-spread PC1 loadings
- **11j** found 82% of hands cross multiple basins - aligns with the 5D manifold allowing diverse outcomes
- **11f** found R² = 0.25 - the remaining 75% variance maps to the 5 PCA dimensions

### Implications

1. **Bidding heuristics are 5D**: A good bidding formula needs ~5 independent factors

2. **Mid-game depth matters most**: PC1 loadings peak at depths 8-16, not early or late game

3. **Convergence is reliable**: The funnel structure (41→3 spread) means endgame analysis is stable

### Files Generated

- `results/tables/11q_per_hand_pca_features.csv` - Per-hand features
- `results/tables/11q_pca_variance.csv` - Variance explained
- `results/tables/11q_pca_loadings.csv` - Component loadings
- `results/tables/11q_pca_summary.csv` - Summary statistics
- `results/figures/11q_per_hand_pca.png` - Visualization

---

## 11x: Information Value (Perfect vs Imperfect) (Preliminary)

### Key Question
How much does knowing opponent hands help?

### Method
Compare Q-values with perfect info (knowing which opponent holds which cards) vs imperfect info (average Q across opponent configurations).

- **Perfect info**: Best move under each config separately, value = Q[best_action]
- **Imperfect info**: Best move using average Q across configs, value = mean(Q[avg_best_action])
- **Information gain**: Perfect value - Imperfect value

### Key Findings (Preliminary - 50 seeds, 84K states)

#### Information Value Summary

| Metric | Value |
|--------|-------|
| Mean information gain | **0.84 points** |
| Median information gain | **0.00 points** |
| Std information gain | 2.58 points |
| Max information gain | 31.33 points |

**Critical Finding**: Knowing opponent hands gains only **0.8 points on average**. The median is 0 - in most positions, perfect information provides no advantage.

#### How Often Does Perfect Info Help?

| Threshold | Percentage |
|-----------|------------|
| Any benefit (>0) | **26.7%** |
| Significant (>2 pts) | **16.6%** |
| Large (>5 pts) | **6.3%** |

**Insight**: Only 27% of positions benefit from perfect info. The vast majority (73%) have the same optimal move regardless of whether you know opponent hands.

#### Action Agreement Rate

| Metric | Value |
|--------|-------|
| Perfect/Imperfect agreement | **74.7%** |

**Key Finding**: 75% of the time, the best move under perfect information is the SAME as the best move under imperfect information. Opponent inference provides only marginal improvement.

#### Information Value by Game Phase

| Depth | Mean Info Gain | n |
|-------|----------------|---|
| 1 (near end) | +0.00 | 915 |
| 5 (late) | +0.55 | 56,567 |
| 9 (mid) | +1.43 | 25,969 |
| 13 (early) | +2.70 | 729 |
| 17 (very early) | +8.83 | 2 |

**Insight**: Information value increases with game depth. Early in the game, knowing opponent hands is worth ~3-9 points. By endgame, it's worth essentially nothing.

### Interpretation

This is perhaps the most surprising finding of the entire imperfect information analysis:

1. **Information has marginal value**: Contrary to intuition, knowing opponent hands typically doesn't change what you should do

2. **Most positions are "dominant"**: The best move is best regardless of opponent distribution

3. **Early game is where it matters**: Information value peaks at depth 17+ (very early game), where there's maximum uncertainty

4. **Endgame is deterministic**: At depth 1-5, perfect information adds virtually nothing

### Reconciling with Other Findings

- **11c** found 54.5% best move consistency → different from 75% agreement here
- **Explanation**: 11c counted divergent paths separately; 11x averages across opponent configs within the same position
- **11n** found 36% of critical decisions are opponent-dependent → aligns with 27% benefiting from perfect info
- **11o** found 97% of common states have robust best moves → strong alignment

### Implications for Strategy

1. **Don't overthink opponent hands**: 75% of the time, the right move is the right move regardless

2. **Focus on your own play**: With only 0.8 expected points from perfect info, execution matters more than reading

3. **Early game exceptions**: The 6% of positions with >5 point info value ARE worth careful analysis

4. **Simplify decision-making**: For most positions, use heuristics instead of modeling opponents

### The Practical Takeaway

> "Play the board, not the player."

Opponent inference adds <1 point of expected value on average. Unless you're in the rare (6%) high-value situations, focus on basic strategy.

### Files Generated

- `results/tables/11x_information_value_by_seed.csv` - Per-seed data
- `results/tables/11x_information_value_summary.csv` - Summary
- `results/figures/11x_information_value.png` - Visualization

---

*Analysis date: 2026-01-07*
