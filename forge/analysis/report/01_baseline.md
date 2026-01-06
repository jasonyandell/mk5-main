# 01: Baseline Distributions

## Overview

Before structural analysis, we characterize the marginal distributions of our key variables: the minimax value V, action values Q, and state counts across game depth.

---

## 1.1 Data Generation Process

**State space definition**: A state encodes:
- Which dominoes remain in each player's hand (4 × 7-bit masks initially)
- Which dominoes have been played in the current trick (0-3 dominoes)
- Trick history (which team won each completed trick)
- Current player to act

**Minimax computation**: For each terminal state (all dominoes played), V equals the declaring team's score minus 21 (centering at zero). For non-terminal states:
- If declaring team to play: V = max over actions of successor V
- If defending team to play: V = min over actions of successor V

**Sampling**: We analyze 10 seed-declaration pairs in detail, representing ~300M total states.

---

## 1.2 V Distribution by Seed and Declaration

V distributions vary substantially across configurations:

| Seed | Decl | States | V̄ | σ(V) | V Range |
|------|------|--------|-----|------|---------|
| 0 | 0 | 7.6M | +9.5 | 11.4 | [-18, +42] |
| 1 | 1 | 5.2M | +3.6 | 12.3 | [-26, +38] |
| 2 | 2 | 51.1M | +0.2 | 13.1 | [-34, +42] |
| 3 | 3 | 75.4M | -2.5 | 14.4 | [-42, +36] |
| 5 | 5 | 30.2M | -11.0 | 10.0 | [-42, +4] |
| 8 | 8 | 24.7M | +12.2 | 15.3 | [-26, +40] |

**Observations**:
1. Mean V ranges from -11.0 to +12.2 — declaration quality varies dramatically
2. Standard deviation ranges from 10.0 to 15.3
3. State counts span 5.2M to 75.4M (14× variation)
4. Full theoretical range [-42, +42] is approached but not always achieved

**Statistical question**: What determines state count variation? Is it correlated with V distribution properties?

---

## 1.3 V Distribution by Depth

"Depth" = dominoes remaining across all hands (28 at start, 4 at final trick, 0 at terminal).

**Key structural feature**: Depths 1-4 are identical because the first trick hasn't resolved. V at these depths equals the expected V over all possible first tricks.

Sample from seed 0, declaration 0:

| Depth | n | V̄ | σ(V) | Unique V | Entropy(V) |
|-------|---|-----|------|----------|------------|
| 0 | 4 | 0.0 | 0.0 | 1 | 0.0 |
| 1-4 | 7,756 | +3.4 | 7.6 | 12 | 2.73 |
| 5 | 1.09M | +9.5 | 9.7 | 25 | 3.46 |
| 9 | 2.23M | +15.8 | 10.1 | 39 | 3.64 |
| 13 | 502K | +21.7 | 10.0 | 44 | 3.67 |
| 17 | 31K | +26.8 | 10.3 | 37 | 3.64 |
| 21 | 1,088 | +30.9 | 10.8 | 25 | 3.45 |
| 25 | 27 | +40.7 | 3.7 | 3 | 0.75 |
| 28 | 1 | +42.0 | 0.0 | 1 | 0.0 |

**Observations**:
1. V̄ increases monotonically with depth (declaring team's advantage clarifies)
2. σ(V) peaks mid-game (~10-11) and decreases at extremes
3. Entropy peaks around depth 9-13, reflecting maximum uncertainty
4. Late game (depth > 20) has very few unique V values

![V Distribution by Depth](../results/figures/01b_q_by_depth.png)

---

## 1.4 State Count Distribution

State counts follow a characteristic pattern tied to the trick structure:

| Depth | Mean States | Std Dev | Min | Max |
|-------|-------------|---------|-----|-----|
| 5 | 1.82M | 646K | 927K | 3.67M |
| 9 | 8.66M | 6.75M | 1.48M | 27.1M |
| 13 | 3.16M | 3.20M | 261K | 11.3M |
| 17 | 196K | 191K | 17.6K | 702K |
| 21 | 3,593 | 2,907 | 456 | 12,222 |
| 25 | 35 | 17 | 10 | 69 |

**Pattern**: Counts peak at depth 9 (second trick boundary), with high variance across seeds.

The coefficient of variation (σ/μ) ranges from 0.35 (depth 5) to 0.81 (depth 21), indicating substantial heterogeneity in game tree sizes.

---

## 1.5 Q-Value Structure

Q(s,a) = minimax value of taking action a in state s. We analyze:
- **Gap**: Q(s, best) - Q(s, chosen) for the second-best action
- **Spread**: max(Q) - min(Q) across available actions
- **Optimal count**: number of actions achieving max Q

| Depth | Gap (mean) | Gap (median) | Spread (mean) | Forced % |
|-------|------------|--------------|---------------|----------|
| 5 | 0.011 | 0.0 | 48.1 | 97.1% |
| 9 | 0.038 | 0.0 | 48.4 | 91.1% |
| 13 | 0.101 | 0.0 | 49.1 | 70.0% |
| 17 | 0.223 | 0.0 | 50.6 | 55.5% |
| 21 | 6.08 | 0.0 | 52.3 | 49.0% |

**Key finding**: Median gap is 0.0 at all depths, meaning the majority of positions have a unique optimal action. The mean gap increases with depth as more positions become "live" (non-forced).

![Q-Value Structure](../results/figures/01b_q_structure.png)

**Forced moves**: At depth 5, 97% of positions have only one reasonable move. This decreases to ~50% by depth 21 as the game tree collapses.

---

## 1.6 Summary Statistics

| Metric | Value |
|--------|-------|
| Total states analyzed | ~300M |
| Seeds | 20 |
| V range | [-42, +42] |
| Mean state count per (seed, decl) | 15.2M |
| Median Q-gap | 0.0 (all depths) |
| Forced move rate | 50-97% by depth |

---

## 1.7 Questions for Statistical Review

1. **Heterogeneity**: State counts vary 14× across configurations. Should we stratify analyses by seed, or is pooling appropriate?

2. **Distributional form**: V appears roughly Gaussian by inspection but hasn't been tested. Is normality expected given the game's structure?

3. **Entropy measure**: We compute H(V) by discretizing to integer values. Is this appropriate, or should we treat V as continuous?

4. **Depth correlation**: Depths 1-4 are perfectly correlated (identical V). Should these be collapsed for analysis?

---

*Next: [02 Information Theory](02_information.md)*
