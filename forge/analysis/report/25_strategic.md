# 25: Strategic Analysis

Actionable insights for optimal play.

## 25a: Mistake Cost by Phase

### Key Question
When do mistakes hurt most? Where should players focus their thinking?

### Method
- Compute Q_best - Q_second for every state (gap between best and second-best move)
- Aggregate by depth/trick
- Identify critical decision points

### Key Findings

#### Mistake Cost by Depth

| Depth | Trick | Mean Cost | % Forced | n_states |
|-------|-------|-----------|----------|----------|
| 20 | 3 | 4.2 pts | 58% | 77 |
| 17 | 3 | 3.6 pts | 76% | 2,944 |
| 16 | 4 | 4.0 pts | 64% | 2,255 |
| 12 | 5 | 3.5 pts | 66% | 12,657 |
| 8 | 6 | 3.4 pts | 70% | 10,118 |
| 4 | 7 | 0.0 pts | 100% | 378 |

#### Phase Comparison

| Phase | Depth Range | Mean Mistake Cost | Forced Plays |
|-------|-------------|-------------------|--------------|
| Early | 20-28 | 4.9 pts | 69% |
| Mid | 8-19 | 2.7 pts | 75% |
| Late | 0-7 | 1.0 pts | 92% |

### Key Insights

1. **Early/mid-game most costly**: Mistakes average 3-5 points in tricks 2-5
2. **End-game is forced**: 90%+ of late-game positions have only one legal move
3. **Peak mistake cost**: Depth 16-20 (tricks 3-4) has highest average cost

### When to Think Hard

| Trick | Recommendation |
|-------|----------------|
| 1-2 | Medium focus - setting tempo |
| 3-4 | **HIGH focus** - peak mistake cost |
| 5-6 | Medium focus - outcomes narrowing |
| 7 | Low focus - mostly forced plays |

### Practical Implications

1. **Concentrate on tricks 3-4**: This is where suboptimal play costs the most
2. **Don't overthink the end-game**: With 90%+ forced plays, there's little to decide
3. **Early mistakes are recoverable**: Wide game tree allows compensation
4. **Mid-game mistakes compound**: Narrowing tree means fewer recovery options

### Files Generated

- `results/tables/25a_mistake_cost_by_depth.csv` - Summary statistics
- `results/figures/25a_mistake_cost_by_phase.png` - 4-panel visualization
- `results/figures/25a_mistake_cost_main.png` - Publication figure
- `results/figures/25a_mistake_cost_main.pdf` - Vector format

---

## 25b: Trick Importance

### Key Question
Which tricks matter most for final outcome?

### Method
- Analyze correlation between trick-level decisions and final V
- Identify pivotal tricks

### Key Findings

**Trick importance (by outcome variance explained)**:
1. Trick 1-2: Moderate (declarer sets tempo)
2. **Tricks 3-4**: Highest (maximum branching)
3. Tricks 5-6: Moderate (narrowing outcomes)
4. Trick 7: Low (mostly determined)

---

## 25c: Bid Optimization

### Key Question
How should bidding strategy account for hand features?

### Method
- Map E[V] to bid thresholds
- Account for uncertainty (σ)

### Key Findings

**Recommended bid adjustments**:

| Feature Change | E[V] Impact | Bid Adjustment |
|----------------|-------------|----------------|
| +1 double | +5.7 pts | +1 mark |
| +1 trump | +3.2 pts | +0.5 mark |
| Trump double | +3 pts (uncertain) | Conservative |

**Napkin bidding formula**:
```
Bid estimate = 30 + 6×(doubles) + 3×(trumps)
```

Adjust down for high σ(V) (uncertain hands).

---

## 25d: Domino Timing

### Key Question
When should specific dominoes be played?

### Method
- Analyze Q-value patterns for domino plays by depth
- Identify optimal timing for doubles, trumps, count

### Key Findings

General patterns:
- **Lead doubles early**: Establish control
- **Save trumps**: Use to capture count or control late tricks
- **Count dominoes**: Value depends on trick context

---

## 25e: Lead Analysis

### Key Question
What makes a good opening lead?

### Method
- Analyze Q-values for depth=28 (opening lead)
- Identify optimal lead patterns

### Key Findings

**Opening lead priorities**:
1. High doubles (if available)
2. Trump suit (establish control)
3. Avoid leading low off-suit

---

## Summary

Strategic analysis provides actionable guidance:

1. **Focus on tricks 3-4**: Highest mistake cost, most strategic value
2. **Don't overthink endgame**: 90%+ forced plays
3. **Bid with napkin formula**: ~30 + 6×doubles + 3×trumps
4. **Lead doubles early**: Establish control immediately
5. **Mistakes average 2-5 points**: Meaningful but not catastrophic
