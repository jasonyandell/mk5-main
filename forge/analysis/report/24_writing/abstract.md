# Abstract Draft

## Texas 42: Oracle-Based Analysis of Bidding Under Imperfect Information

**Background**: Texas 42 is a partnership domino game where players bid for the right to declare trump, then compete to win counting dominoes. Despite 150+ years of play, no rigorous analysis of optimal bidding strategy has been published.

**Methods**: We constructed an oracle (minimax solver) for Texas 42 and evaluated 200 unique hands across multiple opponent configurations (marginalization). For each hand, we computed expected value E[V] and risk σ(V) under perfect play. We extracted 10 hand features and applied linear regression, SHAP analysis, and enrichment testing to identify predictors of bidding success.

**Results**: We discovered an inverse risk-return relationship (r = -0.38, p < 10⁻⁷): strong hands have lower outcome variance, opposite to typical financial markets. Feature importance analysis identified two dominant predictors: number of doubles (r = +0.40) and trump count (r = +0.23). These explain ~26% of E[V] variance via a simple "napkin formula": **E[V] ≈ 14 + 6×(doubles) + 3×(trumps)**. Risk (σ[V]) proved nearly unpredictable from hand features (R² = 0.08), confirming that outcome uncertainty derives from unknown opponent holdings rather than one's own hand. Game phase analysis revealed order→chaos→resolution transitions: early game shows 40% best-move consistency, mid-game drops to 22%, then end-game reaches 100%.

**Conclusions**: In Texas 42, good hands are safer hands. Bidding decisions should weight doubles twice as heavily as trumps. The residual ~74% variance in E[V] and ~92% variance in σ(V) reflect irreducible uncertainty from imperfect information.

---

**Word count**: 253

**Keywords**: Texas 42, dominoes, game theory, minimax, imperfect information, SHAP, feature importance, risk-return
