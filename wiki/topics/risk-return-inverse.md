---
title: Risk-return inverse correlation (oracle Texas 42)
kind: topic
first_seen: 2026-01-06
last_updated: 2026-07-13
status: complete
---

## The headline finding

In the [[forge-analysis]] oracle data, **expected value and risk are negatively
correlated** at n=200 seeds:

| Metric | r | 95% CI | p |
|---|---|---|---|
| r(E[V], σ[V]) | **−0.381** | [−0.494, −0.256] | 2.6×10⁻⁸ |
| r(E[V], V_spread) | **−0.398** | [−0.509, −0.274] | 5.4×10⁻⁹ |

Effect size is medium by Cohen's conventions (\|r\| ≈ 0.38–0.40). Power ≈ 1.00 at
n=200; would only need n≈51 to detect at 80% power.

(`forge/analysis/report/13_statistical_rigor.md`; figure
`results/figures/15a_risk_return_scatter.png`/`.pdf` is the publication-quality
rendering.)

## Why it's striking

Inverted from typical financial markets: **good hands are also predictable hands**.
A hand with high E[V] (from doubles, trumps, count points) tends to have low σ(V)
across opponent realizations — the win is robust to where the rest of the dominoes
land.

## Why the inversion holds (mechanism)

The two top E[V] predictors — `n_doubles` and `trump_count` (the book's napkin formula; no page)
— are also negatively associated with σ(V). 5-5 in particular shows up enriched in
**both** high-E[V] AND low-σ(V) groups (report/17a, 17b). Doubles are trick winners
**and** dampeners — they reduce the degrees of freedom available to opponent hand
realizations to swing the outcome.

## Pareto consequence

Because the relationship is inverse, the Pareto frontier (max E[V], min σ(V)) is
**degenerate**: only ~1.5% of hands are non-dominated, and those are the few hands
that already lock in E[V]=42, σ=0. There is **no meaningful risk-return tradeoff**
in oracle Texas 42 (`report/15c`, `results/tables/15c_pareto_frontier.csv`).

## Caveat

This is an oracle finding. See [[oracle-vs-human-play]] — whether human Texas 42
shows the same inverse relationship is untested, and the mechanism above (doubles
dampen opponent variance under perfect play) doesn't directly transfer to imperfect
information.

## Cross-validation

Holds under multiple lenses:

- Bivariate (Pearson r) — `report/12_validate_scale.md`
- Multivariate regression — `report/13a` shows σ(V) prediction R² < 0.10, but the
  inverse correlation with E[V] survives.
- Effect-size summary — `report/13c`, medium effect size.
- Fisher z-transform CIs — `report/13d`, [−0.49, −0.26] at 95%.
- BH FDR correction — `report/13f`, survives across all 16 simultaneous tests.
- Cross-validation — `report/13g`, σ(V) model has negative CV R² (the *prediction*
  is unpredictable but the *correlation* itself is solid).
