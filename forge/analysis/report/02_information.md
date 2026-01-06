# 02: Information Theory Analysis

## Context

Information theory asks: how much structure exists in V? Random data doesn't compress; structured data does. This tells us whether a neural network *can* learn the patterns.

## Compression Results: Significant Structure Exists

We compressed serialized V values with LZMA:

![Compression Results](../results/figures/02b_compression_single.png)

**Results:**
- Compression ratio: ~0.3-0.5 (compresses to 30-50% of original)
- Random data would compress to ~100%

**Model relevance**: The 60-70% redundancy is what our model exploits. There's substantial learnable structure—which the 97.8% accuracy confirms.

![Multi-seed Compression](../results/figures/02b_compression_multi.png)

## Entropy by Depth

Entropy measures unpredictability. We found:
- **Early game**: High entropy (many possible outcomes)
- **Late game**: Low entropy (outcomes determined)

![Cumulative Information](../results/figures/02a_cumulative_info.png)

**Model relevance**: This suggests curriculum learning could help—train on late-game (easy, low entropy) first, then early-game (hard, high entropy). However, our current model achieves 97.8% without curriculum, so the benefit may be marginal.

## Information Gain Per Move

Each move reveals information and changes the game state. We measured cumulative information across game progression:

![Information Gain](../results/figures/02a_info_gain.png)

**Model relevance**: Information peaks in midgame, matching where our model's errors concentrate. The 2.2% error rate isn't uniform—it clusters in high-information positions.

## What This Means for the Model

| Finding | Implication |
|---------|-------------|
| 40% compression | Substantial learnable structure |
| Depth-correlated entropy | Game phase affects difficulty |
| Midgame information peak | Where model errors concentrate |

The information theory analysis confirms there's structure to learn—and our 97.8% model is successfully capturing most of it. The remaining ~2% likely represents genuinely ambiguous positions where even perfect information doesn't uniquely determine the optimal move.

---

*Next: [03 Count Domino Analysis](03_counts.md)*
