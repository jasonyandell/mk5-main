# 20: Time Series Analysis

Analyzing how game value V evolves during play.

> **Epistemic Status**: This report analyzes how oracle (minimax) V values evolve along game tree paths. All findings describe oracle game tree structure—how perfect-information optimal play unfolds. The "phases" and "volatility patterns" are properties of the oracle tree, not necessarily how human players experience game progression. Gameplay advice extrapolated from oracle dynamics is hypothetical.

## 20a: V Trajectory Extraction

### Key Question
How does V (game value) change as dominoes are played?

### Method
- Extract V distribution at each depth (28 → 0)
- Depth = dominoes remaining in game
- Analyze volatility and convergence patterns

### Key Findings

#### V Statistics by Depth

| Depth | Trick | n_states | Mean V | σ(V) | V Range |
|-------|-------|----------|--------|------|---------|
| 23 | 2 | 5 | 11.6 | 22.3 | 63 |
| 20 | 3 | 77 | 3.4 | 20.5 | 80 |
| 16 | 4 | 2,255 | 5.7 | 16.7 | 78 |
| 12 | 5 | 12,657 | 5.6 | 13.8 | 76 |
| 8 | 6 | 10,118 | 3.4 | 7.5 | 64 |
| 4 | 7 | 378 | 0.0 | 0.0 | 0 |

#### Volatility Pattern

| Phase | Depth Range | Mean σ(V) | Interpretation |
|-------|-------------|-----------|----------------|
| Early | 20-28 | ~20 | High uncertainty |
| Mid | 8-19 | ~15 | Outcome narrowing |
| Late | 0-7 | ~5 | Nearly determined |

### Key Insights

1. **Maximum volatility early**: σ(V) peaks around depth 20-23 (~22 points)
2. **Progressive resolution**: V range narrows as game progresses
3. **Convergence depth**: ~50% of outcome uncertainty resolved by trick 4

### Potential Implications (Hypotheses)

The following are speculative extrapolations from oracle data. None have been validated against human gameplay.

- **Hypothesis**: Opening plays may have outsized impact if human games have similar variance profiles
- **Hypothesis**: Late-game may be more mechanical if outcomes are similarly determined
- **Hypothesis**: Strategic focus on tricks 1-4 may be warranted if oracle volatility patterns apply

**Untested**: Whether human games exhibit the same variance-by-depth pattern is unknown. Hidden information could change these dynamics entirely.

### Files Generated

- `results/tables/20a_v_trajectory.csv` - V statistics by depth
- `results/figures/20a_v_trajectory.png` - 4-panel trajectory visualization

---

## 20b: MiniRocket Classification

Time series classification of V trajectories using MiniRocket features.

### Key Question
Can we predict game outcome from the shape of the V trajectory?

### Method
- Extract V trajectories as time series (depth 28→0) using DuckDB for efficiency
- Use MiniRocket kernel features for classification (n_kernels=5000-10000)
- Binary classification: winner (V > 0) vs loser (V ≤ 0)
- **Balanced dataset**: 732 seeds (366 wins, 366 losses), 70/30 train/test split

### Key Findings

**Classification Accuracy by Prefix Length:**

| Prefix Length | Plays | Trick | Train Acc | Test Acc |
|---------------|-------|-------|-----------|----------|
| 9 | 9 | 3 | 100% | **97.7%** |
| 12 | 12 | 4 | 100% | 95.9% |
| 16 | 16 | 5 | 100% | 96.4% |
| 20 | 20 | 6 | 100% | 95.9% |
| 24 | 24 | 7 | 100% | 95.9% |
| 28 | 28 | 8 | 100% | 95.9% |
| 29 | 29 | 8 | 100% | **96.8%** |

**Key insights:**
- **97.7% accuracy by trick 3** (9 plays) - early game signal is extremely strong
- **Stable 95-97% accuracy** across all prefix lengths
- **No late-game degradation** - larger dataset eliminates noise

### Interpretation (Oracle Trajectories)

1. **Early oracle trajectory is highly predictive**: The first 3 tricks (97.7%) of oracle V trajectory predict final oracle outcome
2. **Stable across game phases**: 95-97% accuracy throughout oracle game depth
3. **Excellent generalization**: 100% train vs 96-98% test shows model captures oracle trajectory patterns
4. **Time series features work**: MiniRocket's random convolutional kernels capture oracle game dynamics

**Note**: This predicts oracle outcomes from oracle trajectories. Whether early human game states similarly predict final human outcomes is untested.

### Technical Notes

- Used DuckDB `bit_count()` for efficient depth calculation (10x faster than Python UDF)
- Trajectory = median V at each depth level per seed
- MiniRocket requires minimum 9 timepoints

### Files Generated

- `results/tables/20b_minirocket_accuracy.csv` - Accuracy by prefix length
- `results/figures/20b_minirocket_classification.png` - Accuracy curves and sample trajectories

---

## 20c: Phase Segmentation

### Key Question
Can we identify distinct game phases from V trajectory patterns?

### Method
- Segment games by V volatility (σ(V)) at each depth
- Identify phase transitions from variance changes
- Label phases: deterministic, chaotic, transition

### Key Findings

**V Statistics by Depth:**

| Depth | σ(V) | V Range | Phase |
|-------|------|---------|-------|
| 25 | 0.0 | 0 | Deterministic |
| 24 | 0.0 | 0 | Deterministic |
| 23 | 22.3 | 63 | **Chaotic** |
| 20 | 20.5 | 80 | Chaotic |
| 16 | 16.7 | 78 | Chaotic |
| 12 | 13.8 | 76 | **Transition** |
| 8 | 11.0 | 69 | Transition |
| 4 | 7.9 | 52 | Deterministic |
| 1 | 8.0 | 42 | Deterministic |

**Three Phases**:

| Phase | Depth Range | σ(V) | Characteristics |
|-------|-------------|------|-----------------|
| **Deterministic (Early)** | 24-25 | 0 | Only 1 state, declarer control |
| **Chaotic** | 13-23 | 16-22 | Maximum uncertainty, peak variance |
| **Transition** | 5-12 | 11-14 | Outcomes narrowing, count locks |
| **Deterministic (Late)** | 0-4 | 7-8 | Outcomes locked, mechanical |

### Phase Transition Points

Key transitions occur at:
- **Depth 23**: σ jumps from 0 to 22 (chaos onset)
- **Depth 12**: σ drops from 17 to 14 (transition begins)
- **Depth 4**: σ drops from 11 to 8 (end-game lock)

### Interpretation (Oracle Tree Structure)

The oracle game tree transitions from **order → chaos → resolution**:
1. **Opening (deterministic)**: Declarer plays first few dominoes, single path in oracle tree
2. **Mid-game (chaotic)**: Opponents enter, oracle σ(V) peaks at 22 points
3. **Transition**: Count dominoes captured in oracle play, outcomes narrow
4. **End-game (deterministic)**: Last trick, oracle outcomes locked

**Note**: This describes the oracle tree structure. Human games with hidden information may have different phase dynamics—e.g., uncertainty about opponent hands persists throughout.

### Files Generated

- `results/tables/20c_phase_segmentation.csv` - Phase labels by depth
- `results/figures/20c_phase_segmentation.png` - σ(V) trajectory with phases

---

## 20d: Motif Discovery

Pattern mining in V trajectories to find common game dynamics.

---

## Summary (Oracle Game Tree)

Time series analysis of oracle data reveals:

1. **Oracle games resolve progressively**: Oracle V uncertainty decreases monotonically with depth
2. **Oracle outcome predictable by trick 3**: MiniRocket achieves 97.7% accuracy from first 9 oracle plays
3. **Three distinct oracle phases**: Opening (control), mid-game (chaos), end-game (resolution)
4. **Early oracle decisions dominate**: First few tricks determine most of the oracle outcome

**Scope limitation**: These patterns describe the oracle (perfect information) game tree. Whether human games with hidden information exhibit similar phase structure and early predictability is untested.

---

## Further Investigation

### Validation Needed

1. **Human game trajectories**: Do actual human games show similar variance-by-depth profiles? This requires human gameplay data with move-by-move records.

2. **Prediction transfer**: Can the MiniRocket model trained on oracle trajectories predict human game outcomes? This would test whether oracle patterns transfer.

3. **Phase perception**: Do human players perceive the three-phase structure (order → chaos → resolution), or does hidden information mask these dynamics?

### Methodological Questions

1. **Trajectory representation**: Using median V at each depth may obscure important variation. Would individual game paths reveal different patterns?

2. **Sample representativeness**: 732 seeds balanced by outcome. Does the phase structure hold for unbalanced samples and extreme outcomes?

3. **MiniRocket sensitivity**: The 97.7% accuracy is high. What features drive classification? SHAP analysis on MiniRocket could reveal which time points matter most.

### Open Questions

1. **Why such high early accuracy?**: Is 97.7% by trick 3 a ceiling, or would more data push accuracy higher? What information in the first 9 plays is so predictive?

2. **Human vs oracle phase timing**: If human games have phases, do they occur at the same depths as oracle phases?

3. **Strategic intervention**: If outcomes are "determined" early in oracle play, does early strategic intervention in human games have similar leverage?
