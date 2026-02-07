# 26: Austin 42 Verification

Validating Texas 42 folk wisdom analytically using oracle data.

## 26a: Threshold Cliffs

### Key Question
Do P(make bid) drops at 30→31 and 35→36 show significantly larger cliffs than other transitions?

### Folk Wisdom Claims
- 30→31 is "about losing one 10-count" (first point requiring count capture)
- 35→36 is "about losing one 5-count" (can only afford to lose one 5-count)

### Method
- Use unified features (V_mean, V_std) for 200 hands
- Compute P(make bid B) = 1 - Φ((B - V_mean) / V_std) using normal approximation
- Calculate Δ(P) = P(make n) - P(make n+1) for each transition
- Test if key transitions show larger drops than baseline

### Key Findings

#### Transition Drop Statistics

| Transition | Mean Δ (%) | Excess vs Baseline | Key? |
|------------|-----------|-------------------|------|
| 30→31 | 1.50% | -0.08% | Yes |
| 31→32 | 1.48% | -0.09% | No |
| 32→33 | 1.48% | -0.10% | No |
| 33→34 | 1.48% | -0.10% | No |
| 34→35 | 1.48% | -0.10% | No |
| 35→36 | 1.54% | -0.04% | Yes |
| 36→37 | 1.63% | +0.05% | No |
| 37→38 | 1.66% | +0.08% | No |
| 38→39 | **1.74%** | **+0.16%** | No |
| 39→40 | 1.72% | +0.14% | No |
| 40→41 | 1.63% | +0.05% | No |
| 41→42 | 1.48% | -0.09% | No |

**Baseline mean Δ**: 1.58%

#### Statistical Test Results

| Threshold | Mean Drop | Excess | p-value | Result |
|-----------|-----------|--------|---------|--------|
| 30→31 (10-count cliff) | 1.50% | -0.08% | >0.05 | **NOT CONFIRMED** |
| 35→36 (5-count cliff) | 1.54% | -0.04% | >0.05 | **NOT CONFIRMED** |

### Interpretation

**FOLK WISDOM NOT SUPPORTED BY DATA**

Using the normal approximation method, neither claimed threshold cliff shows a significantly larger drop than surrounding transitions:

1. **30→31 drop is average**: 1.50% drop is actually slightly below the baseline (1.58%)
2. **35→36 drop is average**: 1.54% drop is also near the baseline
3. **Largest drop at 38→39**: The biggest excess (+0.16%) occurs at 38→39, not at the claimed thresholds

### Caveats

1. **Normal approximation**: This analysis assumes V is normally distributed. The true distribution may have discrete jumps at count thresholds that the approximation misses.

2. **Limited sample**: 200 hands may not have sufficient power to detect threshold effects.

3. **Marginalized data needed**: A more rigorous test would use the full V distribution from marginalized data (3 opponent configurations per hand).

### Files Generated

- `results/tables/26a_threshold_cliffs.csv` - Transition statistics
- `results/figures/26a_threshold_cliffs.png` - 4-panel visualization

---

## 26f: Coverage vs Trump Count

### Key Question
Does "coverage" (ability to compete in multiple suits) matter as much as trump count?

### Folk Wisdom Claim
"2 trumps + perfect coverage beats 4 trumps + naked lows"

Translation: Off-suit quality matters as much as raw trump count.

### Method
- Define `coverage_score` = composite measure of off-suit quality:
  - +1 per domino beyond first in each off-suit (depth bonus)
  - +1 if max rank in suit ≥ 5 (high card bonus)
  - -2 for singleton with rank ≤ 2 (naked low penalty)
- Regress E[V] on trump_count + coverage_score
- Compare standardized coefficients (β) for fair comparison

### Key Findings

#### Bivariate Correlations with E[V]

| Feature | r | p-value | Interpretation |
|---------|---|---------|----------------|
| trump_count | **+0.229** | 0.0011 | More trumps → higher E[V] |
| coverage_score | **-0.333** | 1.4×10⁻⁶ | More coverage → LOWER E[V] |

#### Multivariate Regression

| Feature | Coef | 95% CI | p-value | β (standardized) |
|---------|------|--------|---------|------------------|
| trump_count | +1.55 | [-0.34, 3.44] | 0.107 | +0.117 |
| coverage_score | **-1.63** | [-2.44, -0.82] | **0.0001** | **-0.288** |

**R² = 0.123**

#### Effect Size Comparison

- |β(coverage)| / |β(trump)| = **2.45×**
- Coverage effect is 2.45× larger in magnitude than trump effect
- BUT coverage effect is **NEGATIVE** - higher coverage → worse outcomes

### Interpretation

**FOLK WISDOM REFUTED (INVERTED)**

The data shows the **opposite** of folk wisdom:

1. **Coverage hurts, not helps**: Higher coverage_score is associated with **lower** E[V]
2. **Trump count helps**: More trumps correlate with higher E[V] (though not significant after controlling for coverage)
3. **Voids are valuable**: The negative coverage effect suggests that voids (enabling trump plays) are more valuable than being "covered" in all suits

### Why Coverage Hurts

1. **Voids enable trumping**: When you have no cards in a suit, you can trump when opponents lead it
2. **Following suit is weak**: If you must follow with a low card, you lose the trick
3. **Trumping is powerful**: Cutting in with a trump often wins even against high leads
4. **Coverage = commitment**: Being spread across all suits means fewer trumps and fewer void-based ruff opportunities

### Revised Folk Wisdom

**Correct interpretation**: "4 trumps + voids beats 2 trumps + coverage"

Having voids (which the coverage metric penalizes as depth=0) is actually beneficial because it enables trump plays. The folk wisdom appears to have the relationship backwards.

### Files Generated

- `results/tables/26f_coverage_vs_trump.csv` - Summary statistics
- `results/figures/26f_coverage_vs_trump.png` - 4-panel visualization

---
