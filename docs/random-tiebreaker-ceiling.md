# Random Tiebreaker Ceiling Analysis

A study of optimal action uniqueness in Texas 42 game states.

## The Ceiling Metric

When training a model to predict optimal actions, we define the **random tiebreaker ceiling**:

```python
ceiling = (1.0 / (q_values == q_values.max(dim=-1)).sum(dim=-1)).mean()
```

This measures: if a model always selects an action with maximum Q-value but breaks ties uniformly at random, what accuracy would it achieve?

- If there's 1 best action: contributes 1.0
- If there are k tied best actions: contributes 1/k
- Average across all states gives the ceiling

## Results (10M Samples)

| Metric | Value |
|--------|-------|
| **Ceiling** | **0.7396** (73.96%) |
| 95% CI | [0.7394, 0.7398] |
| Mean ties at max | 1.694 |

### Tie Distribution

| Ties at Max | Count | Percentage |
|-------------|-------|------------|
| 1 (unique best) | 5,569,215 | 55.69% |
| 2 | 2,272,924 | 22.73% |
| 3 | 1,820,778 | 18.21% |
| 4 | 324,096 | 3.24% |
| 5 | 12,837 | 0.13% |
| 6 | 149 | 0.001% |
| 7 | 1 | 0.00001% |

**Key insight**: Only 55.7% of states have a unique optimal action. The remaining 44.3% have ties, reducing the theoretical maximum accuracy achievable by any argmax-based policy.

## The Singular 7-Way Tie

Out of 10 million game states, exactly **one** has all 7 actions tied at the maximum Q-value. This represents a state where the player has **zero agency** - their choice literally does not affect the game outcome.

### Game State Reconstruction

**Sample index**: 7,851,428
**Trump declaration**: 6s
**Current player**: P2 (Team 0)
**Value**: V = -36 (Team 0 loses by 36 points)

#### Current Trick (First Trick of the Hand)

| Position | Player | Team | Domino | Trump Rank |
|----------|--------|------|--------|------------|
| Lead | P0 | 0 | 6-0 | 7th (lowest) |
| 2nd | P1 | 1 | 6-5 | 2nd (very high) |
| 3rd | P2 | 0 | ??? | - |
| 4th | P3 | 1 | (waiting) | - |

#### All Hands (Remaining Dominoes)

**P0 (Team 0, Partner of P2):**
```
0-0 (double)     - not trump
3-1              - not trump
4-2              - not trump
5-0 [5 pts]      - not trump
6-3              - TRUMP (rank 4)
6-6 (double)     - TRUMP (rank 1, HIGHEST!)
[played: 6-0]
```

**P1 (Team 1):**
```
2-1              - not trump
3-3 (double)     - not trump
4-3              - not trump
4-4 (double)     - not trump
5-4              - not trump
6-2              - TRUMP (rank 5)
[played: 6-5, rank 2]
```

**P2 (Team 0, Current Player):**
```
1-1 (double)     - not trump
2-0              - not trump
2-2 (double)     - not trump
4-0              - not trump
5-1              - not trump
5-2              - not trump
5-3              - not trump
*** NO TRUMPS ***
```

**P3 (Team 1):**
```
1-0              - not trump
3-0              - not trump
3-2 [5 pts]      - not trump
4-1 [5 pts]      - not trump
5-5 (double) [10 pts] - not trump
6-1              - TRUMP (rank 6)
6-4 [10 pts]     - TRUMP (rank 3)
```

### Why All 7 Actions Are Equivalent

1. **P0 led with 6-0** (the weakest trump)
2. **P1 responded with 6-5** (second-highest trump, beaten only by 6-6)
3. **P2 has zero trumps** in their hand
4. **P2 cannot win this trick** regardless of what they play
5. **P3 (Team 1)** has 6-1 and 6-4, but neither beats P1's 6-5
6. **P1 wins this trick** no matter what P2 or P3 play

The game tree from this point is **fully deterministic**. P2's choice affects nothing - not trick ownership, not count capture, not the final score. All paths lead to V = -36.

### The Situation

P0 still holds **6-6 (the highest trump)** but led with 6-0 (the lowest). This allowed P1 to safely play 6-5 knowing only 6-6 could beat it, and P0 already committed.

Whether this was a mistake requires examining P0's opening Q-values - perhaps all leads were equivalent, or perhaps leading 6-6 would have changed the outcome. What we know for certain is that by the time P2 acts, the game is already decided.

P2, dealt no trumps at all in a sixes game, is a spectator to their own defeat.

### Trump Hierarchy (6s Trump)

| Domino | Trick Rank | Notes |
|--------|------------|-------|
| 6-6 | 46 | Highest (P0 has it, didn't play it) |
| 6-5 | 43 | Currently winning trick |
| 6-4 | 42 | P3 has it |
| 6-3 | 41 | P0 has it |
| 6-2 | 40 | P1 has it |
| 6-1 | 39 | P3 has it |
| 6-0 | 38 | Lowest (P0 led with this) |

## Implications for Model Training

1. **Accuracy ceiling**: ~74% is the maximum achievable by argmax selection
2. **Tie-breaking matters**: 44% of states have ties; sophisticated tie-breaking could recover some accuracy
3. **Zero-agency states are rare**: Only 1 in 10M states has complete action equivalence
4. **Most ties are small**: 96% have ≤3 tied actions

## Data Source

- Dataset: `data/tokenized-full/train/`
- Samples analyzed: 10,000,000
- Source: Oracle-solved game trees from `/mnt/d/shards-standard/`
