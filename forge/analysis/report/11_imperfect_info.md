# 11: Imperfect Information Analysis

Analysis of hand strength, count control, and decision stability under imperfect information.

## 11a: Count Lock Rate Analysis

**Question**: Which counts does a hand control? P(I capture count_i) across opponent configs.

**Method**: Trace principal variation from depth-0 for 100 seeds, track which team captures each count domino.

### Key Findings

1. **Overall holder advantage: 60.6%**
   - Holding a count domino gives ~10.6% advantage over random (50%)
   - This is the "ownership premium"

2. **10-counts are much easier to lock than 5-counts**
   - 10-count average lock rate: **75.4%**
   - 5-count average lock rate: **50.8%**
   - 10-counts (5-5, 6-4) are nearly locked when held; 5-counts are contested

3. **Individual count domino lock rates**

   | Domino | Points | Lock Rate | Status |
   |--------|--------|-----------|--------|
   | 5-5    | 10     | 80.0%     | LOCKED |
   | 6-4    | 10     | 70.8%     | LOCKED |
   | 4-1    | 5      | 56.9%     | CONTESTED |
   | 5-0    | 5      | 52.3%     | VULNERABLE |
   | 3-2    | 5      | 43.1%     | VULNERABLE |

### Implications for Bidding

- **10-counts are reliable**: If you hold 5-5 or 6-4, count those points
- **5-counts are speculative**: 3-2 and 5-0 are essentially coin flips
- **The 4-1 is marginal**: Slightly better than random, but not reliable

### Heritage Insight

> "When bidding, count your 10-counts as solid and your 5-counts as half."

This analysis validates the folk wisdom: 10-counts held = 20 points expected, 5-counts held = 7.5 points expected.

### Outputs
- Figure: `results/figures/11a_count_lock_rates.png`
- Table: `results/tables/11a_count_lock_rates.csv`
- Table: `results/tables/11a_lock_by_declaration.csv`
