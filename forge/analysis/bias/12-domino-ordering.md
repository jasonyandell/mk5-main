# 12: Domino Ordering in deal_from_seed()

## Summary

**CONFIRMED: Sorting creates systematic slot 0 bias.** The `deal_from_seed()` function sorts hands by domino ID, causing slot 0 to always contain the minimum-ID domino in each hand. This creates a 1.74-bit KL divergence from uniform for slot 0, with low-pip "blank" dominoes appearing 2.5x more often than expected.

## The Sorting Mechanism

From `forge/oracle/rng.py:13`:
```python
hands = [sorted(dominos[i * 7 : (i + 1) * 7]) for i in range(4)]
```

This sorts each hand by domino ID after the random deal. Since domino IDs are ordered by (high_pip, low_pip), this means:
- **Slot 0**: Always the minimum-ID domino (lowest pips)
- **Slot 6**: Always the maximum-ID domino (highest pips)

## Domino ID Ordering

From `forge/oracle/tables.py`, dominoes are ordered:

| ID Range | Dominoes | Characteristics |
|----------|----------|-----------------|
| 0-6 | 0-0, 1-0, 1-1, 2-0, 2-1, 2-2, 3-0 | Low pips, 3 doubles, many blanks |
| 21-27 | 6-0, 6-1, 6-2, 6-3, 6-4, 6-5, 6-6 | High pips, 1 double, 1 count (6-4) |

## Empirical Frequency Distribution (500K samples)

### Slot 0 Distribution
| Domino | Frequency | vs Uniform (3.57%) |
|--------|-----------|-------------------|
| 0-0 (ID 0) | 25.1% | 7.0x |
| 1-0 (ID 1) | 19.4% | 5.4x |
| 1-1 (ID 2) | 15.0% | 4.2x |
| 2-0 (ID 3) | 11.4% | 3.2x |
| 2-1 (ID 4) | 8.5% | 2.4x |
| 2-2 (ID 5) | 6.3% | 1.8x |
| 3-0 (ID 6) | 4.5% | 1.3x |

**Dominoes NEVER in slot 0**: IDs 21-27 (6-0 through 6-6)

### Slot 6 Distribution (mirror image)
| Domino | Frequency | vs Uniform |
|--------|-----------|-----------|
| 6-6 (ID 27) | 25.1% | 7.0x |
| 6-5 (ID 26) | 19.4% | 5.4x |
| 6-4 (ID 25) | 14.8% | 4.2x |
| 6-3 (ID 24) | 11.4% | 3.2x |

**Dominoes NEVER in slot 6**: IDs 0-6 (0-0 through 3-0)

## KL Divergence from Uniform

| Slot | KL Divergence (bits) | Interpretation |
|------|---------------------|----------------|
| 0 | **1.74** | Extreme bias |
| 1 | 1.06 | High bias |
| 2 | 0.81 | Moderate |
| 3 | 0.74 | Moderate (most uniform) |
| 4 | 0.81 | Moderate |
| 5 | 1.06 | High bias |
| 6 | **1.74** | Extreme bias |

## Strategic Property Differences

### Domino Type Frequencies in Slot 0

| Property | Slot 0 Freq | Uniform Expectation | Ratio |
|----------|-------------|---------------------|-------|
| Contains blank (0) | 61.4% | 25.0% | **2.46x** |
| Is a double | 48.0% | 25.0% | **1.92x** |
| Is a count domino | 5.7% | 25.0% | **0.23x** |

### Strategic Implications

Low-ID dominoes (dominating slot 0):
- **Low pip sums** (avg 3.7 vs 8.3 for high IDs)
- **More blanks** (weak in most trump declarations)
- **More doubles** (strategically important but context-dependent)
- **Fewer count points** (3-2, 4-1, 5-0, 5-5, 6-4 are all ID >= 8)

High-ID dominoes (dominating slot 6):
- **High pip sums** (stronger trick-winning potential)
- **Contains 6-4** (10-point count domino)
- **Contains 5-5** (ID 20, 10-point double - but CAN appear in slot 0 rarely)
- **More valuable in off-suit play**

## Impact on Model Training

### Distribution Shift Problem
The model learns Q-value estimation on a biased training distribution:
- Slot 0 examples are dominated by weak, low-value dominoes
- Slot 6 examples are dominated by strong, high-value dominoes
- Slots 0 and 6 never see 7 dominoes from the opposite end

### Potential Causes of r=0.81 vs r=0.99+

1. **Limited domino diversity**: Slot 0 sees only ~21 unique dominoes vs 28 possible
2. **Value asymmetry**: Low-ID dominoes have systematically lower Q-value variance
3. **Edge effects**: Minimum/maximum of a random sample has different statistics than middle values
4. **Generalization gap**: Model trained heavily on 0-0 in slot 0, rarely on strategically complex dominoes

### The "Easy Examples" Hypothesis
Slot 0 dominoes (0-0, 1-0, 1-1, etc.) may have more predictable Q-values because:
- Blanks are often "throw-away" dominoes
- Low doubles have limited trick-winning power
- Less interaction with count points and strategic plays

If slot 0 examples are "easier" on average, the model might:
- Overfit to simple patterns
- Fail to learn nuanced valuation
- Show lower correlation when evaluated on harder examples

## Recommendation

**Shuffle slot order during training** or **use canonical domino ordering** that doesn't correlate with strategic value. Options:

1. **Random permutation**: After sorting for deduplication, randomly permute slot indices
2. **Hash-based ordering**: Use a position-independent hash to order dominoes
3. **Set representation**: Use a permutation-invariant architecture (set transformer, DeepSets)
4. **Data augmentation**: Randomly permute slot indices during training

The current sorting is convenient for deduplication and state comparison, but it creates an unintended training bias that degrades slot 0 predictions.
