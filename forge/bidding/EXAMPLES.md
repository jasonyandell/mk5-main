# Bidding Evaluation Examples

This document shows worked examples of bidding evaluation with analysis, including a spot-check that validates the simulation is working correctly.

## How to Read the Matrix

```
Trump          30  31  32  33  34  35  36  37  38  39  40  41  42
-----------------------------------------------------------------
fives         100 100 100 100 100 100 100 100 100 100 100 100 100
```

- **Rows** = trump choice
- **Columns** = bid level (points needed to make)
- **Cells** = P(make) as percentage

**Decision rule:** If P(make) > 50%, the bid has positive expected value.

---

## Example 1: Monster Hand (All Sixes)

**Hand:** 6-6, 6-5, 6-4, 6-3, 6-2, 6-1, 6-0

```
Trump          30  31  32  33  34  35  36  37  38  39  40  41  42
-----------------------------------------------------------------
blanks         24  22  22  22  22  22  16  12  12  12  12  10  10
ones           18  14  12  12  12  12  10  10  10  10  10  10  10
twos           10  10   8   8   8   8   8   8   8   8   8   8   8
threes         12   8   8   8   6   6   6   6   6   6   6   6   6
fours          14  14  14  14  14  14  12  10  10  10  10  10   8
fives          14  12  12  12  10   8   8   6   6   6   6   6   6
sixes         100 100 100 100 100 100 100 100 100 100 100 100 100
doubles-trump  44  42  38  38  38  38  30  28  28  28  28  26  26
notrump       100 100 100 100 100 100 100 100 100 100 100 100 100
```

**Analysis:** With all 7 sixes, you control every trick regardless of what opponents have. Both sixes and notrump show 100% across the board. This is a "walk" - bid 42 confidently.

---

## Example 2: Near-Monster (6 Fives + Double-Six)

**Hand:** 5-5, 5-4, 5-3, 5-2, 5-1, 5-0, 6-6

```
Trump          30  31  32  33  34  35  36  37  38  39  40  41  42
-----------------------------------------------------------------
blanks         26  26  22  22  22  20  20  18  18  18  18  16  14
ones           32  30  28  28  26  24  20  18  18  18  16  16  14
twos           22  22  16  16  16  14  12  12  12  12  12  12   8
threes         16  14  12  12  12  10   8   8   8   8   8   8   6
fours          18  14  14  14  14  14  14   8   8   8   8   6   6
fives         100 100 100 100 100 100 100 100 100 100 100 100 100
sixes          28  28  18  18  18  16  14   6   6   6   4   4   4
doubles-trump  72  70  70  70  68  66  58  56  56  56  56  52  44
notrump        98  98  98  98  98  98  98  98  98  98  98  98  96
```

**Analysis:** With 6 of the 7 fives (missing only 5-6), fives is 100% - bid 42. Notrump drops to 96% because the opponent with 5-6 can occasionally steal a trick. The 6-6 helps in notrump since it's the highest in its suit.

---

## Example 3: Doubles Hand

**Hand:** 4-4, 3-3, 2-2, 1-1, 0-0, 6-5, 4-3

```
Trump          30  31  32  33  34  35  36  37  38  39  40  41  42
-----------------------------------------------------------------
blanks         56  48  42  42  42  40  34  28  28  28  28  28  16
ones           54  50  44  44  44  40  40  32  32  32  32  30  22
twos           58  52  46  46  46  46  38  30  30  30  30  30  22
threes         74  62  56  56  54  52  42  38  38  38  36  30  30
fours          66  58  46  46  46  46  46  42  42  42  42  36  28
fives          32  28  22  22  22  20  20  20  20  20  20  20  12
sixes          44  38  30  30  30  30  22  18  18  18  18  16  10
doubles-trump  54  54  40  40  40  40  40  28  28  28  28  28  20
notrump        80  74  58  58  58  58  58  54  54  54  54  54  38
```

**Analysis:** Counter-intuitively, **notrump beats doubles-trump** (80% vs 54%)! In notrump, each double is the highest card of its suit - you can't be trumped. With doubles-trump, opponents can trump your small doubles. The 6-5 count domino also helps in notrump.

**Verdict:** Bid 30 on notrump confidently.

---

## Example 4: Mixed Count Hand

**Hand:** 6-4, 5-5, 4-2, 3-1, 2-0, 1-1, 0-0

```
Trump          30  31  32  33  34  35  36  37  38  39  40  41  42
-----------------------------------------------------------------
blanks         54  46  46  46  46  44  42  24  24  24  24  22  14
ones           58  52  46  46  44  42  36  26  26  26  24  22  14
twos           54  52  42  42  42  36  32  24  24  24  22  20  10
threes         42  42  42  42  36  32  28  26  26  26  22  18  12
fours          56  56  56  56  54  48  36  32  32  32  30  22  16
fives          50  42  42  42  42  34  26  20  20  20  18  18  16
sixes          38  28  26  26  24  18  16  14  14  14  14  12   2
doubles-trump  64  62  62  62  62  50  42  32  32  32  26  24  16
notrump        66  56  56  56  56  50  40  34  34  34  34  30  20
```

**Analysis:** Notrump (66%) and doubles-trump (64%) are best for 30. This is a marginal hand - >50% means positive expected value, but not by much. Note the steep drop from 30→31; you should bid exactly 30, not higher.

**Verdict:** Marginal hand. Bid 30 if you must, but it's close to break-even.

---

## Example 5: Weak Hand

**Hand:** 6-1, 5-2, 4-3, 3-0, 2-1, 1-0, 0-0

```
Trump          30  31  32  33  34  35  36  37  38  39  40  41  42
-----------------------------------------------------------------
blanks         50  42  36  36  36  34  26  20  20  20  20  20  12
ones           46  40  38  38  38  36  30  28  28  28  26  26  18
twos           44  36  32  32  32  26  22  18  18  18  18  18  10
threes         36  28  22  22  22  16  14  14  14  14  14  12   4
fours          16  14  14  14  12  12   8   6   6   6   6   4   2
fives          12  10  10  10  10  10  10   4   4   4   2   2   2
sixes          12  12  12  12  10  10   8   4   4   4   4   4   4
doubles-trump   8   6   6   6   6   4   4   2   2   2   2   2   2
notrump        14  14   8   8   8   6   4   2   2   2   2   2   .
```

**Analysis:** The best option (blanks at 30) is exactly 50% - break-even. Everything else is negative EV.

**Verdict:** **Don't bid this hand.** Pass and let your partner or opponents take it.

---

## Spot Check: Validating the Simulation

A natural question: "How can a weak hand sometimes make 42 with an unlikely trump?"

Looking at Example 4 (mixed count hand), sixes shows 2% at 42. But the hand only has one six (6-4)! How is this possible?

**Investigation:** We found a specific case (seed=8) where this happens:

```
Player 0 (Bidder):  6-4                        (1 six)
Player 1 (Opponent): 6-0                       (1 six)
Player 2 (Partner):  6-6, 6-3, 6-2, 6-1        (4 sixes!)
Player 3 (Opponent): 6-5                       (1 six)

Team 0 (bidder + partner): 5 sixes including 6-6
Team 1 (opponents): 2 sixes
```

**The partner got incredibly lucky!** The random deal gave player 2 four sixes, including the 6-6 (highest trump). Combined with the bidder's 6-4, Team 0 has 5 of 7 sixes and controls the game.

**Key insight:** The 2% isn't noise - it's the actual probability that the random deal gives your partner a monster hand in your chosen trump suit. The simulation correctly models that **your bid strength = your hand + what your partner might have**.

This is why simulation beats heuristics. You can't easily compute "probability my partner has 4+ sixes" analytically, but the Monte Carlo simulation naturally captures it.

---

## Summary Table

| Hand | Best Trump | Bid 30 | Bid 42 | Verdict |
|------|------------|--------|--------|---------|
| All sixes | sixes | 100% | 100% | Walk - bid 42 |
| 6 fives + 6-6 | fives | 100% | 100% | Walk - bid 42 |
| 5 doubles | notrump | 80% | 38% | Strong 30 |
| Mixed count | notrump | 66% | 20% | Marginal 30 |
| Scattered | blanks | 50% | 12% | Pass! |

---

## Usage

```bash
# Default: greedy (optimal) play, matrix output
python -m forge.bidding.evaluate --hand "6-4,5-5,4-2,3-1,2-0,1-1,0-0" --samples 100

# More samples for tighter confidence (slower)
python -m forge.bidding.evaluate --hand "..." --samples 500

# Sorted list format (old style)
python -m forge.bidding.evaluate --hand "..." --samples 100 --list

# JSON output for programmatic use
python -m forge.bidding.evaluate --hand "..." --samples 100 --json
```

---

## Poster Generation

Generate visual PDF posters with domino tiles and P(make) heatmaps:

```bash
# Generate a poster for a specific hand
python -m forge.bidding.poster --hand "6-4,5-5,4-2,3-1,2-0,1-1,0-0" --output poster.pdf

# More samples for accuracy (default: 50)
python -m forge.bidding.poster --hand "..." --output poster.pdf --samples 100

# Reproducible output
python -m forge.bidding.poster --hand "..." --output poster.pdf --seed 42
```

### Poster Layout

- **Top**: Visual domino tiles with pips
- **Bottom**: P(make) heatmap (9 trumps × 13 bid levels)
- **Color scale**: Red (0%) → Yellow (50%) → Green (100%)

### Pre-generated Examples

See `scratch/posters/` for posters of all example hands:

| File | Hand |
|------|------|
| `01-monster-all-sixes.pdf` | All 7 sixes |
| `02-near-monster-fives.pdf` | 6 fives + 6-6 |
| `03-doubles-hand.pdf` | 5 doubles |
| `04-mixed-count.pdf` | Marginal hand |
| `05-weak-hand.pdf` | Weak/pass hand |
