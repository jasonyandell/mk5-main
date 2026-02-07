# Action Frequency Distribution Analysis

**Investigation**: Does training data have uneven action frequency distribution across slots?

**Context**: The Q-value model shows slot 0 has r=0.81 correlation with oracle while slots 1-6 have r=0.99+. This investigation checks whether the oracle recommends slot 0 less often, potentially undertrained.

**Data**: `data/tokenized-full/train/` (11,240,000 samples)

---

## Key Finding: Slot 0 is NOT Undertrained

**The hypothesis is rejected.** Slot 0 is actually the **most frequently recommended** action by the oracle and receives comparable training signal to other slots.

| Metric | Slot 0 | Slots 1-6 Avg | Conclusion |
|--------|--------|---------------|------------|
| Oracle's best choice | 24.4% | 12.6% | **1.9x MORE often** |
| Legal action frequency | 30.5% | 30.4% | Equal |
| Training signal (Q-loss) | 3.43M | 3.42M avg | Equal |

---

## Detailed Analysis

### 1. Oracle's Best Action Distribution

When the oracle picks the best action, which slot does it recommend?

| Slot | N Best | % Best | Rank |
|------|--------|--------|------|
| 0 | 2,743,972 | 24.41% | 1 |
| 1 | 2,006,261 | 17.85% | 2 |
| 2 | 1,590,453 | 14.15% | 3 |
| 3 | 1,367,474 | 12.17% | 4 |
| 4 | 1,241,678 | 11.05% | 5 |
| 5 | 1,198,621 | 10.66% | 6 |
| 6 | 1,091,541 | 9.71% | 7 |

**Observation**: Strong monotonic decrease from slot 0 to slot 6. This is due to hand sorting by domino ID - lower IDs (slot 0) contain smaller dominoes which are played earlier in the game.

### 2. Legal Action Frequency

How often is each slot a legal action?

| Slot | N Legal | % of Samples | Relative Weight |
|------|---------|--------------|-----------------|
| 0 | 3,426,712 | 30.49% | 1.002x |
| 1 | 3,433,524 | 30.55% | 1.004x |
| 2 | 3,442,232 | 30.62% | 1.007x |
| 3 | 3,433,923 | 30.55% | 1.004x |
| 4 | 3,466,983 | 30.85% | 1.014x |
| 5 | 3,486,820 | 31.02% | 1.020x |
| 6 | 3,248,528 | 28.90% | 0.950x |

**Observation**: Legal action frequency is nearly uniform across slots (within 5%). Slot 0 is NOT legal less often.

### 3. Conditional Probability P(Best | Legal)

Given an action is legal, how often is it the oracle's best choice?

| Slot | P(Best\|Legal) |
|------|---------------|
| 0 | **80.08%** |
| 1 | 58.43% |
| 2 | 46.20% |
| 3 | 39.82% |
| 4 | 35.81% |
| 5 | 34.38% |
| 6 | 33.60% |

**Observation**: When slot 0 is legal, it's the best choice 80% of the time! This is the highest of all slots.

### 4. Tie Analysis

When the oracle recommends a slot, is it strictly best or tied with others?

| Slot | N Best | Strictly Best | Tied | Only Option |
|------|--------|---------------|------|-------------|
| 0 | 2.74M | 4.4% | 69.2% | 13.5% |
| 1 | 2.01M | 6.6% | 56.5% | 23.5% |
| 2 | 1.59M | 8.5% | 42.9% | 34.1% |
| 3 | 1.37M | 10.3% | 30.4% | 44.0% |
| 4 | 1.24M | 11.2% | 21.7% | 51.7% |
| 5 | 1.20M | 12.1% | 12.8% | 60.9% |
| 6 | 1.09M | 13.6% | 0.0% | 73.2% |

**Observation**: Slot 0 has the HIGHEST tie rate (69.2%). When it's recommended, 69% of the time there are other equally good options. This could make learning harder - the model sees many examples where slot 0 is "best" but other actions work equally well.

### 5. Q-Value Statistics by Slot

| Slot | Mean Q | Std Q | P25 | P50 | P75 |
|------|--------|-------|-----|-----|-----|
| 0 | 0.47 | 15.05 | -11 | 1 | 12 |
| 1 | 0.56 | 15.26 | -11 | 1 | 12 |
| 2 | 0.53 | 15.54 | -11 | 1 | 12 |
| 3 | 0.52 | 15.81 | -12 | 1 | 13 |
| 4 | 0.43 | 15.95 | -12 | 1 | 13 |
| 5 | 0.40 | 16.17 | -12 | 1 | 13 |
| 6 | 0.48 | 16.16 | -12 | 1 | 13 |

**Observation**: Q-value distributions are remarkably similar across all slots. No slot has systematically higher/lower Q-values.

### 6. Forced Moves (Single Legal Action)

When only one action is legal, which slot?

| Slot | N | % |
|------|---|---|
| 0 | 369,998 | 8.90% |
| 1 | 470,599 | 11.32% |
| 2 | 542,122 | 13.04% |
| 3 | 601,999 | 14.48% |
| 4 | 642,356 | 15.45% |
| 5 | 730,285 | 17.57% |
| 6 | 799,073 | 19.22% |

**Observation**: Higher slots are more likely to be the only legal option (end-game states where most dominoes are played).

---

## Root Cause Hypothesis Update

The data frequency hypothesis is **rejected**. Slot 0 gets:
- MORE training signal as the oracle's best choice (24% vs 13% avg)
- EQUAL training signal for Q-value prediction (~30% legal rate)

**New hypothesis**: The slot 0 degradation may be related to:
1. **High tie rate**: 69% of slot 0 "best" recommendations are ties with other slots
2. **Game mechanics**: Slot 0 contains lowest-ID dominoes, which have distinct strategic properties
3. **Attention patterns**: Something in the transformer attention mechanism treats position 1 (slot 0's token position) differently

---

## Technical Note: Hand Sorting

Hands are sorted by domino ID (`deal_from_seed` uses `sorted()`). Domino IDs are assigned by (high_pip, low_pip):
- ID 0 = 0-0 (blank double)
- ID 1 = 1-0
- ID 2 = 1-1 (ace double)
- ...
- ID 27 = 6-6 (double six)

So slot 0 always contains the lowest-ID domino in the player's hand, which tends to be a low-pip domino.
