# 19: Bayesian Modeling

Full Bayesian inference on E[V] regression with PyMC.

> **Epistemic Status**: This report applies Bayesian inference to oracle (perfect-information minimax) E[V] and σ(V). All posterior distributions, model comparisons, and archetype-specific effects describe oracle outcomes. The "napkin formula" validation and hierarchical archetype findings characterize oracle predictions. Whether these Bayesian relationships hold for human gameplay outcomes is untested.

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

### Key Findings (Oracle Data)

1. **Only n_doubles and trump_count significant for oracle E[V]**: 95% HDIs exclude zero
2. **Bayesian confirms frequentist on oracle**: Similar to bootstrap CIs from 13a
3. **R² = 0.26**: Model explains ~26% of oracle E[V] variance

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

### Key Findings (Oracle Data)

1. **Mean prediction R² = 0.23**: Reasonable oracle E[V] prediction
2. **Variance prediction r = 0.11**: Near-zero correlation for oracle σ(V)
3. **No significant variance predictors for oracle**: All β_σ CIs include zero
4. **Model comparison**: Heteroskedastic slightly preferred but marginal difference for oracle

### Critical Insight (Oracle Variance)

The heteroskedastic model confirms that **oracle variance is fundamentally unpredictable** from hand features:
- CV r ≈ 0.1 for oracle σ(V) prediction
- Oracle outcome uncertainty comes from opponent hands in marginalized data, not your own hand

**Note**: In human play with inference and signaling, variance dynamics might differ from the oracle. This finding describes oracle game tree variance.

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

### Key Findings (Oracle Model Comparison)

1. **Napkin model (M1) is nearly optimal for oracle**: ΔELPD from best model is only 0.4
2. **Adding features hurts oracle prediction**: M5_Full has ELPD 5.7 worse than napkin
3. **Stacking weights favor simplicity for oracle**: 28.8% napkin, 67.5% napkin+voids, 0% for complex models
4. **No warnings**: All models passed PSIS diagnostics

### Critical Insight (Oracle Complexity)

The Bayesian model comparison confirms that **complexity beyond the napkin formula is penalized for oracle prediction**. The full 10-feature model performs worse than the 2-feature napkin model in oracle predictive accuracy.

**Note**: This model comparison uses oracle outcomes. The optimal model complexity for predicting human gameplay outcomes might differ.

### Files Generated

- `results/figures/19c_model_comparison.png` - ELPD comparison plot
- `results/tables/19c_loo_comparison.csv` - Full LOO-CV results
- `results/tables/19c_incremental_elpd.csv` - Incremental ELPD gains

---

## 19d: Hierarchical Archetype Model

Bayesian hierarchical model with archetype-specific regression coefficients.

### Key Question
Do the effects of doubles and trumps on oracle E[V] vary by oracle hand archetype (control/balanced/volatile)?

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

### Key Findings (Oracle Archetypes)

1. **Control archetype**: Doubles have the strongest oracle effect (+8.21 oracle pts/double)
2. **Balanced archetype**: Both doubles (+6.26) and trumps (+4.29) contribute significantly to oracle E[V]
3. **Volatile archetype**: Oracle effects are weaker and less certain; trumps CI includes zero
4. **Doubles matter more when oracle outcomes are predictable**: Strong effect in control (+8.21) vs volatile (+5.74)

### Critical Insight (Oracle Archetype Effects)

Hand archetype (defined by oracle σ(V) clustering) moderates the napkin formula:
- **Control hands** (low oracle σ(V)): Doubles worth +8 oracle pts each
- **Volatile hands** (high oracle σ(V)): Oracle effects are attenuated - less predictable oracle outcomes

**Note**: These archetypes come from k-means clustering on oracle features (18_clustering). Whether human players perceive or should use similar archetypes is untested. The "control" vs "volatile" distinction describes oracle outcome variance, not necessarily human-perceived risk.

### Files Generated

- `results/figures/19d_hierarchical_archetype.png` - Archetype coefficient comparison
- `results/tables/19d_hierarchical_archetype.csv` - Posterior summary by archetype

---

## Summary (Oracle Bayesian Analysis)

Bayesian analysis of oracle data confirms and extends frequentist findings:

1. **Two significant predictors for oracle E[V]**: n_doubles (+5.4) and trump_count (+3.0)
2. **Oracle risk unpredictable**: Heteroskedastic model shows oracle σ(V) cannot be predicted from hand features
3. **Proper uncertainty quantification for oracle**: Full posteriors available for any oracle inference
4. **Oracle napkin formula validated**: Bayesian posterior means match frequentist estimates on oracle data

**Scope limitation**: These Bayesian findings describe oracle (perfect-information) outcomes. Whether the same posterior distributions and model comparisons hold for human gameplay—with hidden information and strategic complexity—is untested.

---

## Further Investigation

### Validation Needed

1. **Bayesian regression on human data**: Would the posterior distributions for n_doubles and trump_count differ when predicting human game outcomes? Human gameplay might show different coefficient posteriors.

2. **Hierarchical model transfer**: Do the oracle archetypes (control/balanced/volatile) exist in human gameplay? Would a hierarchical model on human data show similar archetype-specific effects?

3. **Model comparison on human outcomes**: The LOO-CV favors the napkin formula for oracle prediction. Would human game outcomes favor a different model complexity?

### Methodological Questions

1. **Prior sensitivity**: The weakly informative priors (Normal(0, 5)) may influence posteriors with n=200. How sensitive are the posterior CIs to prior choices?

2. **Archetype definition**: The archetypes come from k-means on oracle features. Would different clustering algorithms or feature sets yield different archetype-specific effects?

3. **Heteroskedastic model assumptions**: The log-link for variance ensures positivity. Is this the appropriate functional form for oracle variance?

### Open Questions

1. **Why are archetype effects attenuated for volatile hands?**: Is this an inherent property of oracle outcomes, or could human players show different patterns?

2. **Bayesian vs frequentist agreement**: The Bayesian and bootstrap CIs largely agree. Is this expected for n=200, or does it suggest the priors have minimal influence?

3. **Practical use of posteriors**: Full posteriors enable probabilistic statements like P(β_doubles > 4). What practical questions about Texas 42 could benefit from such posterior inferences?
