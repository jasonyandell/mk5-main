# 21: Survival Analysis

Defining and analyzing "decision time" - when game outcomes become determined.

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

### Practical Implications

1. **First 3 tricks**: High uncertainty, decisions matter most
2. **Tricks 4-5**: Outcome becoming clear, still room for impact
3. **Tricks 6-7**: Game essentially decided, play mechanically

### Cognitive Investment Strategy

| Game Phase | Depth | σ(V) | Recommendation |
|------------|-------|------|----------------|
| Early | 20+ | >15 | Think carefully, decisions matter |
| Mid | 8-20 | 8-15 | Strategic focus, key decisions |
| Late | <8 | <8 | Execute optimally, outcomes fixed |

### Files Generated

- `results/tables/21a_decision_time.csv` - Decision time thresholds
- `results/figures/21a_decision_time.png` - Visualization

---

## 21b: Survival Archetype Analysis

### Key Question
Do different hand archetypes have different "survival curves" (time to decision)?

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

### Key Findings

**Control archetype** (n=64):
- **Lowest σ(V)** = 5.0 (outcomes predictable)
- Highest n_doubles (1.89)
- Reach "decided" threshold earliest
- Outcomes locked by trick 4-5

**Volatile archetype** (n=61):
- **Highest σ(V)** = 26.1 (highly uncertain)
- Lowest n_doubles (1.64)
- Stay uncertain longest
- May not reach "decided" until trick 6-7

**Balanced archetype** (n=75):
- Middle σ(V) = 14.7
- Average hand composition
- Decision time matches overall average

### Survival Curve Interpretation

| Archetype | Time to σ<10 | Time to σ<5 | Interpretation |
|-----------|--------------|-------------|----------------|
| Control | Trick 3-4 | Trick 5 | Fast convergence |
| Balanced | Trick 4-5 | Trick 6 | Normal progression |
| Volatile | Trick 5-6 | Trick 7+ | Late or never |

### Key Insights

1. **Doubles drive predictability**: Control hands have more doubles → lower σ(V) → faster decision
2. **Volatility is sticky**: High-σ hands stay uncertain because opponent configurations dominate
3. **Bidding implication**: Control hands bid confidently; volatile hands are risky bids

### Files Generated

- `results/tables/21b_archetype_summary.csv` - Archetype statistics
- `results/figures/21b_survival_archetype.png` - Kaplan-Meier style curves
- `results/figures/21b_archetype_scatter.png` - E[V] vs σ(V) by archetype

---

## Summary

Survival analysis reveals:

1. **Decision time varies**: Games reach "decided" status between tricks 5-7
2. **Archetype matters**: Strong hands converge faster than weak hands
3. **Practical guidance**: Focus cognitive effort on tricks 1-4 where outcomes are still malleable
4. **Late game is mechanical**: After trick 5, optimal play requires less strategic thinking
