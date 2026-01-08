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
