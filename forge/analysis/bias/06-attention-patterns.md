# 06 - Attention Pattern Analysis

## Executive Summary

**Attention sink is NOT the cause of slot 0 degradation.** The root cause is **data bias from hand sorting** - the oracle/tokenization pipeline sorts hands by domino ID, causing specific dominoes to always appear in slot 0.

## Investigation Methodology

1. Loaded the Q-value model (`domino-qval-large-3.3M`)
2. Extracted attention weights from all 6 transformer layers using custom forward pass
3. Analyzed attention patterns to/from position 0 and intra-hand attention
4. Computed Q-prediction correlations per slot and per declaration type
5. Examined tokenized data for domino distribution patterns

## Findings

### 1. Attention Patterns Show No Sink

Final layer attention ratios (slot 0 vs slots 1-6):

| Metric | Layer 0 | Layer 5 (Final) |
|--------|---------|-----------------|
| Attention to P0 slot 0 | 0.046 | 0.031 |
| Attention to P0 slots 1-6 | 0.025 | 0.031 |
| **Ratio** | **1.86x** | **0.98x** |

Early layers show mild positional bias (1.86x), but by the final layer, attention is perfectly balanced (0.98x ratio). This is **opposite** to attention sink behavior, where early positions would accumulate excess attention.

Intra-hand attention (within current player's 7 slots):
- Slot 0 receives: 0.037 (vs 0.036 mean for other slots)
- Slot 0 gives: 0.036 (vs 0.036 mean for other slots)
- **No asymmetry detected**

### 2. Representation Norms Are Equal

| Position | Mean Norm |
|----------|-----------|
| Context (pos 0) | 4.10 |
| Slot 0 (current player) | 4.08 |
| Slots 1-6 (mean) | 4.12 |

Ratio: **1.00x** - slot 0 representations have identical norm to other slots.

### 3. Correlation Degradation Is Declaration-Specific

Running correlation analysis on 100k validation samples:

| Declaration | r(model, oracle) for Slot 0 |
|-------------|----------------------------|
| blanks (0) | 0.991 |
| ones (1) | 0.995 |
| **twos (2)** | **0.159** |
| **threes (3)** | **0.737** |
| **fours (4)** | **0.491** |
| fives (5) | 0.996 |
| sixes (6) | 0.995 |
| doubles-trump (7) | 0.998 |
| doubles-suit (8) | 0.997 |
| notrump (9) | 0.992 |

**Three specific declarations (twos, threes, fours) account for ALL of the slot 0 degradation.**

### 4. Root Cause: Hand Sorting Creates Data Bias

Examining the tokenized data reveals the root cause:

| Declaration | Domino in Slot 0 | Frequency |
|-------------|-----------------|-----------|
| blanks | 0-0 | 96.2% |
| ones | 0-0 | 100% |
| twos | 0-0 | 100% |
| threes | 0-0 | 100% |
| fours | 2-0 | 100% |
| fives | 1-1 | 100% |
| sixes | 2-1 | 100% |

**Slot 0 is NOT randomly populated** - it contains nearly the same domino for all samples within each declaration type.

Source: `forge/oracle/rng.py`, line 13:
```python
hands = [sorted(dominos[i * 7 : (i + 1) * 7]) for i in range(4)]
```

Hands are sorted by domino ID, so the lowest-ID domino in each hand always occupies slot 0.

### 5. Why Specific Declarations Fail

For declarations where the sorted-first domino has varied strategic value:
- **twos (decl_id 2)**: 0-0 is the lowest trump but not always present; varied game contexts create Q-variance the model can't predict because position is confounded with identity
- **threes, fours**: Similar confounding issues

For declarations where it works:
- **blanks**: 0-0 IS the top trump, so its Q-value is predictable
- **fives, sixes, etc.**: Different dominoes in slot 0 with consistent strategic patterns

## Implications

### This Is NOT an Attention Problem

The attention mechanism is working correctly. Position 0 receives appropriate attention and has normal representation norms. The transformer architecture is not exhibiting attention sink behavior.

### This IS a Data/Tokenization Problem

The correlation failure is caused by:
1. **Deterministic slot assignment**: Sorting hands by domino ID
2. **Confounded representations**: Model can't distinguish "slot 0" from "this specific domino"
3. **Declaration-specific effects**: Some declaration/domino combinations have unpredictable Q-values given the confounding

### Recommended Fix

**Shuffle the domino order within each player's hand** during tokenization or training augmentation. This would:
- Break the spurious correlation between slot position and domino identity
- Allow the model to learn position-independent representations
- Likely improve slot 0 correlation from ~0.4-0.8 to ~0.99

## Technical Details

### Model Architecture
- Embed dim: 256
- Layers: 6
- Heads per layer: 8
- Total parameters: 3.3M

### Data Analyzed
- 500 samples for attention extraction (memory-intensive)
- 20,000 samples for Q-prediction analysis
- 100,000 samples for declaration-specific breakdown

### Files
- Analysis script: `forge/analysis/scripts/attention_analysis.py`
- Original correlation script: `forge/analysis/scripts/model_oracle_correlation.py`
- Hand sorting source: `forge/oracle/rng.py`
