# 19: Bayesian Modeling

Full Bayesian inference on E[V] regression with PyMC.

## 19a: PyMC Bayesian Regression for E[V]

### Key Question
What are the posterior distributions for regression coefficients? How do they compare to bootstrap CIs?

### Method
- Bayesian linear regression with weakly informative priors
- NUTS sampler: 4 chains, 2000 draws, 1000 tune
- Diagnostics: R-hat, ESS, divergences

### Model Specification

```
α ~ Normal(0, 5)
β ~ Normal(0, 5)  # per feature
σ ~ HalfNormal(10)
y ~ Normal(α + Xβ, σ)
```

### Diagnostics

| Metric | Value | Status |
|--------|-------|--------|
| R-hat | all < 1.01 | PASS |
| ESS bulk | min 4908 | PASS |
| ESS tail | min 4852 | PASS |
| Divergences | 0 | PASS |

### Posterior Coefficients (95% HDI)

| Feature | Mean | 95% HDI | Significant? |
|---------|------|---------|--------------|
| **n_doubles** | +5.38 | [+2.25, +8.45] | **Yes** |
| **trump_count** | +3.01 | [+1.26, +4.76] | **Yes** |
| has_trump_double | +3.09 | [-2.83, +9.02] | No |
| n_voids | +2.99 | [-2.75, +8.40] | No |
| count_points | +0.22 | [-0.09, +0.55] | No |
| total_pips | +0.10 | [-0.36, +0.53] | No |
| n_singletons | -0.15 | [-3.40, +3.25] | No |
| n_5_high | -0.47 | [-3.19, +2.14] | No |
| max_suit_length | -0.88 | [-5.19, +3.65] | No |
| n_6_high | -1.52 | [-4.64, +1.42] | No |

### Key Findings

1. **Only n_doubles and trump_count significant**: 95% HDIs exclude zero
2. **Bayesian confirms frequentist**: Similar to bootstrap CIs from 13a
3. **R² = 0.26**: Model explains ~26% of E[V] variance

### Advantages of Bayesian Approach

- Full posterior distributions (not just point estimates)
- Proper probabilistic interpretation of intervals
- Valid for small samples without asymptotic assumptions

### Files Generated

- `results/tables/19a_pymc_ev_posterior.csv` - Posterior summary
- `results/models/19a_pymc_ev_idata.nc` - Full inference data
- `results/figures/19a_forest_plot.png` - Coefficient forest plot
- `results/figures/19a_trace_*.png` - MCMC trace plots
- `results/figures/19a_ppc.png` - Posterior predictive check

---

## 19b: Heteroskedastic Bayesian Model

### Key Question
Can we jointly predict E[V] and σ(V) using a heteroskedastic model?

### Method
- Mean model: μ = α_μ + X @ β_μ
- Variance model: log(σ) = α_σ + X @ β_σ
- Joint NUTS inference

### Coefficient Estimates

**Mean Model (β_μ)** - predicting E[V]:

| Feature | Coefficient | 95% CI | Significant? |
|---------|-------------|--------|--------------|
| n_doubles | +5.76 | [+3.54, +7.99] | **Yes** |
| trump_count | +4.34 | [+2.49, +6.06] | **Yes** |
| n_voids | +3.47 | [+0.22, +6.61] | **Yes** |
| has_trump_double | +3.41 | [-1.58, +8.32] | No |
| total_pips | -0.14 | [-0.42, +0.16] | No |

**Variance Model (β_σ)** - predicting log(σ):

| Feature | Coefficient | 95% CI | Significant? |
|---------|-------------|--------|--------------|
| trump_count | -0.107 | [-0.215, +0.004] | Marginal |
| has_trump_double | -0.286 | [-0.588, +0.036] | No |
| n_voids | -0.152 | [-0.328, +0.031] | No |
| n_doubles | -0.002 | [-0.117, +0.116] | No |
| total_pips | +0.001 | [-0.017, +0.019] | No |

### Model Comparison (LOO-CV)

| Model | ELPD_LOO | Weight |
|-------|----------|--------|
| Heteroskedastic | -822.4 | 0.64 |
| Homoskedastic | -823.8 | 0.36 |

### Key Findings

1. **Mean prediction R² = 0.23**: Reasonable E[V] prediction
2. **Variance prediction r = 0.11**: Near-zero correlation
3. **No significant variance predictors**: All β_σ CIs include zero
4. **Model comparison**: Heteroskedastic slightly preferred but marginal difference

### Critical Insight

The heteroskedastic model confirms that **variance is fundamentally unpredictable** from hand features:
- CV r ≈ 0.1 for σ(V) prediction
- Outcome uncertainty comes from opponent hands, not your own

### Files Generated

- `results/tables/19b_heteroskedastic_coefs.csv` - Coefficient estimates
- `results/tables/19b_model_comparison.csv` - LOO-CV comparison
- `results/figures/19b_heteroskedastic_coefs.png` - Forest plots
- `results/figures/19b_heteroskedastic_ppc.png` - Posterior predictive

---

## 19c: Model Comparison (LOO-CV)

Leave-one-out cross-validation using Pareto-smoothed importance sampling (PSIS-LOO).

### Key Question
Which model complexity is optimal? Does adding features beyond the napkin formula help?

### Method
- 6 nested models compared via LOO-CV
- PSIS-LOO for efficient cross-validation
- Stacking weights for model averaging

### Model Ranking

| Model | Rank | ELPD_LOO | Weight | Description |
|-------|------|----------|--------|-------------|
| M2_Plus_Voids | 1 | -822.4 | 67.5% | Napkin + n_voids |
| M1_Napkin | 2 | -822.8 | 28.8% | n_doubles + trump_count only |
| M3_Plus_TrumpDouble | 3 | -822.9 | 0.0% | Napkin + has_trump_double |
| M4_Core5 | 4 | -823.9 | 0.0% | 5 features |
| M5_Full | 5 | -828.1 | 0.0% | All 10 features |
| M0_Intercept | 6 | -847.3 | 3.8% | Intercept only |

### Key Findings

1. **Napkin model (M1) is nearly optimal**: ΔELPD from best model is only 0.4
2. **Adding features hurts**: M5_Full has ELPD 5.7 worse than napkin
3. **Stacking weights favor simplicity**: 28.8% napkin, 67.5% napkin+voids, 0% for complex models
4. **No warnings**: All models passed PSIS diagnostics

### Critical Insight

The Bayesian model comparison confirms that **complexity beyond the napkin formula is penalized**. The full 10-feature model performs worse than the 2-feature napkin model in predictive accuracy.

### Files Generated

- `results/figures/19c_model_comparison.png` - ELPD comparison plot
- `results/tables/19c_loo_comparison.csv` - Full LOO-CV results
- `results/tables/19c_incremental_elpd.csv` - Incremental ELPD gains

---

## 19d: Hierarchical Archetype Model

Bayesian hierarchical model with archetype-specific regression coefficients.

### Key Question
Do the effects of doubles and trumps vary by hand archetype (control/balanced/volatile)?

### Method
- Hierarchical linear model with random slopes by archetype
- 3 archetypes from k-means clustering (18_clustering)
- Partial pooling toward global mean

### Archetype-Specific Effects

| Archetype | β_doubles Mean | β_doubles 95% CI | β_trumps Mean | β_trumps 95% CI |
|-----------|---------------|------------------|---------------|-----------------|
| control | **+8.21** | [+5.27, +11.20] | +3.00 | [+0.83, +4.96] |
| balanced | +6.26 | [+3.56, +8.91] | **+4.29** | [+2.39, +6.59] |
| volatile | +5.74 | [+2.60, +8.92] | +2.66 | [-0.24, +5.08] |

### Key Findings

1. **Control archetype**: Doubles have the strongest effect (+8.21 pts/double)
2. **Balanced archetype**: Both doubles (+6.26) and trumps (+4.29) contribute significantly
3. **Volatile archetype**: Effects are weaker and less certain; trumps CI includes zero
4. **Doubles matter more when outcomes are predictable**: Strong effect in control (+8.21) vs volatile (+5.74)

### Critical Insight

Hand archetype moderates the napkin formula:
- **Control hands** (low σ(V)): Focus on doubles - they're worth +8 pts each
- **Volatile hands** (high σ(V)): Effects are attenuated - less predictable outcomes

### Files Generated

- `results/figures/19d_hierarchical_archetype.png` - Archetype coefficient comparison
- `results/tables/19d_hierarchical_archetype.csv` - Posterior summary by archetype

---

## Summary

Bayesian analysis confirms and extends frequentist findings:

1. **Two significant predictors**: n_doubles (+5.4) and trump_count (+3.0)
2. **Risk unpredictable**: Heteroskedastic model shows σ(V) cannot be predicted from hand features
3. **Proper uncertainty quantification**: Full posteriors available for any inference
4. **Napkin formula validated**: Bayesian posterior means match frequentist estimates
