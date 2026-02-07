# 21: Survival Analysis

Defining and analyzing "decision time" - when game outcomes become determined.

> **Epistemic Status**: This report analyzes when oracle (minimax) outcomes become "decided" based on σ(V) thresholds. All findings describe oracle game tree dynamics. The "decision time" and "survival curves" are properties of perfect-information optimal play. Gameplay advice (cognitive investment, bidding implications) is hypothetical extrapolation—none has been validated against human gameplay.

## 21a: Decision Time Definition

### Key Question
When does a game's outcome become "decided"?

### Method
- Define σ(V) thresholds for different confidence levels
- Find depth where each threshold is first crossed
- Map to trick numbers for practical interpretation

### Decision Thresholds

| Level | σ Threshold | Meaning |
|-------|-------------|---------|
| Very Uncertain | σ < 20 | Still highly volatile |
| Uncertain | σ < 15 | Moderately uncertain |
| Leaning | σ < 12 | One team likely ahead |
| Probable | σ < 10 | Outcome probable |
| Decided | σ < 8 | Outcome essentially decided |
| Locked | σ < 5 | No realistic comeback |

### Key Findings

#### Decision Points

Based on 20a trajectory data:

| Level | First Reached | Trick | Plays |
|-------|---------------|-------|-------|
| Probable (σ<10) | Depth ~8 | 5-6 | ~20 |
| Decided (σ<8) | Depth ~6 | 6 | ~22 |
| Locked (σ<5) | Depth ~4 | 7 | ~24 |

### Hypothetical Implications (Oracle-Derived)

The following extrapolations from oracle data have NOT been validated against human gameplay:

1. **First 3 tricks**: High oracle uncertainty; **hypothesis**: human decisions matter most here
2. **Tricks 4-5**: Oracle outcome becoming clear; **hypothesis**: still room for human impact
3. **Tricks 6-7**: Oracle outcome essentially decided; **hypothesis**: mechanical play suffices

### Cognitive Investment Strategy (Hypothesis)

Based on oracle σ(V) patterns. **Untested** whether human games follow these dynamics.

| Game Phase | Depth | Oracle σ(V) | Hypothetical Recommendation |
|------------|-------|-------------|----------------------------|
| Early | 20+ | >15 | Decisions may matter most |
| Mid | 8-20 | 8-15 | Strategic focus potentially valuable |
| Late | <8 | <8 | Outcomes may be largely fixed |

### Files Generated

- `results/tables/21a_decision_time.csv` - Decision time thresholds
- `results/figures/21a_decision_time.png` - Visualization

---

## 21b: Survival Archetype Analysis

### Key Question
Do different hand archetypes have different oracle "survival curves" (time to decision)?

### Method
- Group hands by k-means cluster from 18a
- Compare σ(V) trajectory by archetype
- Analyze time-to-decision by hand type

### Archetype Summary

| Archetype | E[V] Mean | E[V] Std | σ(V) Mean | σ(V) Std | n_doubles | trumps | n_hands |
|-----------|-----------|----------|-----------|----------|-----------|--------|---------|
| **balanced** | 15.0 | 15.4 | 14.7 | 2.9 | 1.67 | 1.35 | 75 |
| **control** | 19.6 | 19.4 | **5.0** | 3.0 | 1.89 | 1.34 | 64 |
| **volatile** | 6.5 | 11.7 | **26.1** | 4.7 | 1.64 | 1.26 | 61 |

### Key Findings (Oracle Data)

**Control archetype** (n=64):
- **Lowest oracle σ(V)** = 5.0 (oracle outcomes predictable)
- Highest n_doubles (1.89)
- Reach oracle "decided" threshold earliest
- Oracle outcomes locked by trick 4-5

**Volatile archetype** (n=61):
- **Highest oracle σ(V)** = 26.1 (oracle outcomes highly uncertain)
- Lowest n_doubles (1.64)
- Stay uncertain longest in oracle tree
- May not reach oracle "decided" until trick 6-7

**Balanced archetype** (n=75):
- Middle oracle σ(V) = 14.7
- Average hand composition
- Oracle decision time matches overall average

### Oracle Survival Curve Interpretation

| Archetype | Time to oracle σ<10 | Time to oracle σ<5 | Oracle Interpretation |
|-----------|---------------------|--------------------|-----------------------|
| Control | Trick 3-4 | Trick 5 | Fast oracle convergence |
| Balanced | Trick 4-5 | Trick 6 | Normal oracle progression |
| Volatile | Trick 5-6 | Trick 7+ | Late oracle convergence or never |

### Key Insights (Oracle)

1. **Doubles drive oracle predictability**: Control hands have more doubles → lower oracle σ(V) → faster oracle decision
2. **Volatility is sticky in oracle**: High-σ hands stay uncertain because opponent configurations dominate oracle outcomes
3. **Hypothesis for bidding**: Control hands may warrant confident bids; volatile hands may be risky. **Untested** in human play.

### Files Generated

- `results/tables/21b_archetype_summary.csv` - Archetype statistics
- `results/figures/21b_survival_archetype.png` - Kaplan-Meier style curves
- `results/figures/21b_archetype_scatter.png` - E[V] vs σ(V) by archetype

---

## Summary (Oracle Dynamics)

Survival analysis of oracle data reveals:

1. **Oracle decision time varies**: Oracle games reach "decided" status between tricks 5-7
2. **Archetype matters for oracle convergence**: Control hands converge faster than volatile hands in oracle
3. **Hypothesis for play**: Focus cognitive effort on tricks 1-4 where oracle outcomes are still malleable
4. **Hypothesis for late game**: After trick 5, oracle outcomes are largely fixed, suggesting mechanical play may suffice

**Scope limitation**: These patterns describe oracle (perfect-information) game dynamics. Whether human games with hidden information have similar decision timing is untested.

---

## Further Investigation

### Validation Needed

1. **Human decision timing**: Do human games show similar "decision points" where outcomes become determined? This requires human gameplay data with move-by-move analysis.

2. **Archetype validation**: Do human players with "control" hands (high doubles) experience faster convergence than those with "volatile" hands?

3. **Cognitive strategy testing**: Does focusing cognitive effort on tricks 1-4 actually improve human outcomes? A/B testing with human players could validate this hypothesis.

### Methodological Questions

1. **Threshold sensitivity**: The σ thresholds (σ<10, σ<8, σ<5) are arbitrary. Would different thresholds change the decision timing conclusions?

2. **Archetype stability**: The k=3 clustering (control/balanced/volatile) differs from 18a's k=2. Which clustering is more meaningful for survival analysis?

3. **Sample size**: With 200 hands split into 3 archetypes (61-75 per group), survival curves have limited precision. Larger samples could sharpen the archetype differences.

### Open Questions

1. **Hidden information effect**: Human players don't know opponent hands. Does hidden information delay perceived "decision time" or accelerate it (premature resignation)?

2. **Psychological decision time**: When do human players "feel" a game is decided? Does this correlate with oracle σ(V) thresholds?

3. **Strategy adaptation**: If a player knows their hand is "volatile", should they play more aggressively early? More conservatively? The oracle doesn't tell us optimal human strategy.
