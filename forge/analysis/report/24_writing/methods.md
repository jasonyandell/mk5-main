# Methods

## 2.1 Game Description

Texas 42 is a four-player partnership domino game using the standard double-six set (28 tiles). Partners sit across from each other (Team 0: P0+P2; Team 1: P1+P3). Each round begins with a bidding phase where players compete for the right to declare trump (any suit 0-6 or "no trump"). The declarer's team must capture at least as many count points as bid, or the bid becomes a penalty.

The game uses seven tricks, each containing four dominoes. Five dominoes carry "count" points: 5-0 (5), 5-5 (10), 4-1 (5), 3-2 (5), 6-4 (10). Trick points (1 per trick) and count points sum to 42 per round. Victory is scored as: **V = Team0_points - Team1_points**, ranging from -42 to +42.

## 2.2 Oracle Construction

We implemented a minimax solver with alpha-beta pruning and perfect recall. The oracle computes optimal play under perfect information—each player can see all 28 dominoes. The solver is exhaustive: at depth 28 (game start), it evaluates all legal play sequences to determine V, the value of the root state.

**State representation**: Game states are packed into 64-bit integers encoding trump (3 bits), lead suit (3 bits), current trick (4 domino slots × 5 bits each), player hands (28 bits), and game phase. This enables efficient memoization via hash maps.

## 2.3 Marginalization Over Opponent Configurations

Since real players cannot see opponent hands, we marginalize over opponent configurations to compute expected outcomes. For each of 200 base seeds:

1. **Fixed P0 hand**: `hands = deal_from_seed(base_seed)`; P0's hand is fixed
2. **Opponent sampling**: The remaining 21 dominoes are shuffled among P1, P2, P3 using three different opponent seeds (opp0, opp1, opp2)
3. **Oracle evaluation**: V computed for each configuration under perfect play

This yields:
- **E[V]**: Expected value = mean(V) across opponent configurations
- **σ(V)**: Risk/variance = std(V) across configurations

## 2.4 Feature Extraction

We extracted 10 features from each hand:

| Feature | Description |
|---------|-------------|
| n_doubles | Number of doubles (0-7) |
| trump_count | Number of trump-suit dominoes |
| n_6_high | Dominoes where 6 is high pip |
| n_5_high | Dominoes where 5 is high pip |
| count_points | Sum of count points in hand (0-35) |
| total_pips | Sum of all pips |
| has_trump_double | 1 if hand contains trump double |
| max_suit_length | Longest suit |
| n_voids | Number of empty suits |
| n_singletons | Number of single-domino suits |

## 2.5 Statistical Analysis

### Primary Analysis: E[V] vs σ(V) Correlation

We computed Pearson correlation with 95% confidence intervals via Fisher z-transformation:

$$r = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sqrt{\sum(x - \bar{x})^2 \sum(y - \bar{y})^2}}$$

$$z = \frac{1}{2}\ln\left(\frac{1+r}{1-r}\right), \quad SE_z = \frac{1}{\sqrt{n-3}}$$

### Regression Analysis

Ordinary least squares regression with bootstrap confidence intervals (B=1000 iterations):

$$E[V] = \beta_0 + \beta_1 \cdot \text{n\_doubles} + \beta_2 \cdot \text{trump\_count} + \ldots + \epsilon$$

Cross-validation: 10-fold CV with 10 repeats to assess generalization.

### Multiple Comparison Correction

Benjamini-Hochberg FDR correction applied to all 16 correlation tests:

$$p_{adj,i} = \min\left(1, \frac{m \cdot p_{(i)}}{i}\right)$$

where m=16 tests, sorted by p-value.

### Effect Size Interpretation

- Correlation: |r| < 0.1 small, 0.1-0.3 medium, > 0.3 large (Cohen)
- Cohen's d: d < 0.2 small, 0.2-0.8 medium, > 0.8 large
- R²: < 0.02 small, 0.02-0.13 medium, > 0.13 large

### Power Analysis

Post-hoc power analysis using G*Power formulas:

$$\text{Power} = 1 - \Phi\left(z_{1-\alpha/2} - |r|\sqrt{n-3}\right)$$

### SHAP Analysis

We trained GradientBoostingRegressor (100 trees, max_depth=3) and computed SHAP values using TreeExplainer:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

### Enrichment Analysis

Fisher's exact test for 2×2 contingency tables (domino present/absent × winner/loser):

$$p = \frac{\binom{a+b}{a}\binom{c+d}{c}}{\binom{n}{a+c}}$$

with BH FDR correction for 28 dominoes.

## 2.6 Sample Size Justification

With n=200 hands:
- Power > 99% for |r| ≥ 0.38 (our primary finding)
- Power > 80% for |r| ≥ 0.20
- Minimum detectable effect at 80% power: |r| ≈ 0.14

## 2.7 Software and Reproducibility

- **Oracle**: Custom C++ with Python bindings
- **Analysis**: Python 3.10, scikit-learn 1.3, SHAP 0.42, statsmodels 0.14
- **Visualization**: matplotlib 3.7, seaborn 0.12
- **Code**: Available at [repository URL]

All random seeds are fixed for reproducibility. Data pipeline: base_seed → deal_from_seed → oracle → features → analysis.
