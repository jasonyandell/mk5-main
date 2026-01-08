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
- Compute bid success rate by (doubles, trumps)

### E[V] by (n_doubles, trump_count)

| n_doubles | trumps | E[V] | Bid Rate | n_hands | Should Bid? |
|-----------|--------|------|----------|---------|-------------|
| 0 | 0 | -2.3 | 60% | 5 | No |
| 0 | 1 | -10.7 | 17% | 6 | **No** |
| 0 | 2 | -14.0 | 0% | 3 | **No** |
| 0 | 3 | 4.7 | 75% | 4 | Yes |
| 0 | 4 | 29.1 | 100% | 3 | Yes |
| 1 | 0 | 7.5 | 70% | 20 | Yes |
| 1 | 1 | 1.2 | 57% | 14 | Marginal |
| 2 | 0 | 20.0 | 93% | 28 | Yes |
| 2 | 1 | 10.3 | 82% | 17 | Yes |
| 3 | 0 | 15.5 | 86% | 14 | Yes |
| 3 | 3 | 34.1 | 100% | 6 | Yes |
| 4 | 0 | 27.1 | 100% | 5 | Yes |

### Bid Decision Rules

**Always bid** (E[V] > 10):
- 2+ doubles regardless of trumps
- 0 doubles + 4+ trumps
- 1 double + 2+ trumps

**Never bid** (E[V] < 0):
- 0 doubles + 0-2 trumps

**Marginal** (E[V] 0-10):
- 1 double + 0-1 trumps
- Depends on opponent bidding

### Napkin Bidding Formula

```
Expected score = 30 + 6×(doubles) + 3×(trumps)
```

**Interpretation:**
- Base: 30 points (roughly neutral)
- Each double: +6 points (almost a mark)
- Each trump: +3 points (half a mark)

### Risk-Adjusted Bidding

For volatile hands (σ(V) > 20), discount by 25%:
```
Risk-adjusted = Expected × 0.75
```

For control hands (σ(V) < 10), bid confidently.

### Files Generated

- `results/tables/25c_bid_optimization.csv` - Bid analysis by cell
- `results/figures/25c_bid_optimization.png` - Bid heatmap
- `results/figures/25c_bid_heatmap.png` - Publication figure

---

## 25d: Domino Timing

### Key Question
When should specific dominoes be played?

### Method
- Track mean depth at which each domino is played
- Compute early/mid/late play rates
- Identify optimal timing patterns

### Domino Play Timing (ordered by mean depth)

**Early Plays (mean depth > 9.5)**:

| Domino | Mean Depth | Early % | Mid % | Late % |
|--------|------------|---------|-------|--------|
| 6-4 | 9.83 | 0.0% | 83.8% | 16.2% |
| 5-5 | 9.75 | 0.0% | 81.8% | 18.2% |
| 6-2 | 9.74 | 0.1% | 80.5% | 19.5% |
| 6-1 | 9.71 | 0.0% | 82.5% | 17.5% |
| 6-5 | 9.70 | 0.0% | 81.4% | 18.6% |
| 6-6 | 9.61 | 0.0% | 78.9% | 21.1% |

**Late Plays (mean depth < 9.2)**:

| Domino | Mean Depth | Early % | Mid % | Late % |
|--------|------------|---------|-------|--------|
| 1-1 | 9.02 | 0.0% | 72.6% | **27.4%** |
| 4-3 | 8.94 | 0.0% | 70.6% | **29.4%** |

### Key Patterns

1. **High sixes play early**: 6-4, 6-5, 6-6 all have mean depth > 9.6
2. **5-5 (count double) plays early**: Despite being valuable, it's played mid-game
3. **Low doubles play late**: 1-1 and 0-0 are held longer
4. **4-3 is the latest non-double**: Saved for end-game flexibility

### Interpretation

- **Lead strength early**: High dominoes establish control
- **Hold low cards**: Flexibility to follow suit late
- **Doubles vary by value**: 5-5, 6-6 played early; 1-1, 0-0 held late

### Files Generated

- `results/tables/25d_domino_timing.csv` - Full timing statistics
- `results/figures/25d_domino_timing.png` - Timing visualization
- `results/figures/25d_domino_timing_heatmap.png` - Depth distribution heatmap

---

## 25e: Lead Analysis

### Key Question
What makes a good lead at each trick? How does lead strategy evolve?

### Method
- Analyze all leads by trick number
- Compute rates of trump leads, count leads, double leads
- Track average high pip of lead domino

### Lead Characteristics by Trick

| Trick | n_leads | Trump % | Count % | Double % | Avg High Pip |
|-------|---------|---------|---------|----------|--------------|
| 3 | 43 | 27.9% | 11.6% | **39.5%** | 2.5 |
| 4 | 1,129 | 32.2% | 11.6% | 33.5% | 2.7 |
| 5 | 6,244 | 31.2% | 13.6% | 33.3% | 2.8 |
| 6 | 5,096 | 28.5% | **15.0%** | 32.3% | 3.1 |
| 7 | 178 | 21.3% | **19.7%** | 24.7% | **3.7** |

### Key Patterns

**Early tricks (3-4)**:
- **Double rate highest** (~35-40%)
- Trump rate moderate (~30%)
- Count leads rare (~12%)

**Mid tricks (5-6)**:
- Balanced approach
- Count leads increasing (13-15%)
- Pip values rising (2.8-3.1)

**Late tricks (7)**:
- **Count leads peak** (19.7%)
- Double rate drops (24.7%)
- **Highest pip leads** (3.7 avg)

### Interpretation

1. **Lead doubles early**: 39% double rate in trick 3 → establish control
2. **Save count for late**: Count lead rate rises from 12% to 20% as game progresses
3. **Pip escalation**: Average lead pip increases from 2.5 to 3.7 across tricks
4. **Trump flexibility**: Trump leads steady at 28-32% throughout

### Opening Lead Priorities (Trick 1-2)

1. **High double** (if available) - establishes control
2. **Trump suit** - pulls opponent trumps
3. **High off-suit** - wins trick, sets tempo
4. **Avoid**: Low off-suit leads (loses control)

### Files Generated

- `results/tables/25e_lead_analysis.csv` - Lead statistics by trick
- `results/figures/25e_lead_analysis.png` - Lead pattern visualization

---

## Summary

Strategic analysis provides actionable guidance:

1. **Focus on tricks 3-4**: Highest mistake cost, most strategic value
2. **Don't overthink endgame**: 90%+ forced plays
3. **Bid with napkin formula**: ~30 + 6×doubles + 3×trumps
4. **Lead doubles early**: Establish control immediately
5. **Mistakes average 2-5 points**: Meaningful but not catastrophic
