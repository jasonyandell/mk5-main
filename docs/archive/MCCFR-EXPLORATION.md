# MCCFR Exploration for Texas 42

This document archives the Monte Carlo Counterfactual Regret Minimization (MCCFR) exploration that was conducted for Texas 42 AI development. The approach was ultimately abandoned in favor of other methods.

## What is MCCFR?

**Counterfactual Regret Minimization (CFR)** is a family of algorithms that compute Nash equilibrium strategies for imperfect information games. MCCFR is the Monte Carlo variant that samples opponent actions rather than exhaustively enumerating them, making it tractable for larger games.

The core idea:
- At each decision point, compute the "regret" for not having played each action
- Accumulate regrets over many iterations
- Convert regrets to a strategy via regret matching (positive regrets → proportional probabilities)
- The average strategy over all iterations converges to a Nash equilibrium

## Why We Tried It

Texas 42 is an imperfect information game (hidden opponent hands), making game-theoretic approaches appealing:

1. **Nash equilibrium** - An unexploitable strategy that performs well against any opponent
2. **Offline training** - Train once, deploy a small strategy table for instant decisions
3. **Proven success** - CFR solved Heads-Up Limit Texas Hold'em and achieved superhuman performance in No-Limit

## The Implementation

### Core Components

```
src/game/ai/cfr/
├── types.ts              # InfoSetKey, ActionKey, CFRNode, MCCFRConfig
├── regret-table.ts       # Storage with getStrategy(), updateRegrets()
├── action-abstraction.ts # actionToKey(), sampleAction(), selectBestAction()
├── mccfr-trainer.ts      # External sampling MCCFR trainer
├── mccfr-strategy.ts     # AIStrategy using trained regrets
├── compact-format.ts     # CFD1 compression format
└── compact-format-v2.ts  # CFD2 ultra-compact format (1.3MB for 250k iterations)
```

### Information Set Abstraction

The key challenge in CFR is managing the number of unique **information sets** (game situations from a player's perspective). We explored three abstraction levels:

1. **Raw** - Full game history (explodes exponentially)
2. **Canonical** - Permute non-trump suits to canonical order (minor compression)
3. **Count-centric** - Abstract away non-strategic details (32.5x compression)

### The Count-Centric Abstraction

The insight: Texas 42 has only 5 "count" dominoes worth points (5-0, 5-5, 6-4, 3-2, 4-1 = 35 points total). Everything else exists to control when count can be safely played.

The abstraction captured:
- Which count dominoes are in my hand
- Points captured by each team so far
- Points at stake in current trick
- Trump control (am I leading? how many trump do I hold?)
- Game progress (trick number, position)

This collapsed ~millions of canonical info sets down to ~100k count-centric info sets.

## Why It Failed

### The Fundamental Problem: Suit Identity Matters

The count-centric abstraction threw away suit identity for non-count dominoes. This proved fatal.

**Example failure case**: With treys as trump, the strategy couldn't learn "don't lead 5-0 when opponent might have the 3-5 (trump)." The abstraction saw:
- "I have count domino 5-0"
- "Trump type: regular suit"

But it **didn't see**:
- "5-0 contains a trey pip, which is trump"
- "Leading 5-0 is vulnerable to 3-5 trump"

This led to systematically bad plays like leading count into obvious trumping situations.

### Training Metrics

From our metrics collection (`cfr-metrics.ts`):
- ~4.5 average branching factor
- ~100k unique count-centric info sets (at 1000 games)
- 32.5x compression ratio over canonical
- But high singleton rate (many info sets visited only once)

The singleton rate indicated the abstraction was still too fine-grained for the training budget, while simultaneously being too coarse to capture crucial suit relationships.

### Performance Reality

Training results at 100k iterations:
- ~250k unique info sets discovered
- Strategy files: 172MB raw JSON, ~1.3MB CFD2 compressed
- Play quality: Noticeably worse than simple heuristic rollouts

The trained strategy made moves that were "count-aware" but "trump-blind" - it would protect count in general, but fail to account for specific trump vulnerabilities.

## Key Learnings

### 1. Abstraction is the Hard Problem

CFR's convergence guarantees apply to the abstracted game, not the original. A lossy abstraction can converge to a "Nash equilibrium" that performs poorly in the real game.

### 2. Suit Identity is Strategic in Texas 42

Unlike poker (where suit is irrelevant for hand strength), Texas 42's trump mechanism makes suit identity strategically crucial. The 5-0 plays very differently when fives are trump vs. when zeros are trump.

### 3. Count-Centrism is Necessary but Insufficient

Points matter, but HOW you win points matters equally. A correct abstraction needs to capture:
- Which count I hold
- **Whether my count is vulnerable to trump**
- **Which suits I control for safe count plays**

### 4. CFR May Need Full State for Texas 42

The game tree might simply be too small to benefit from abstraction. With only 28 dominoes, 7 tricks, and ~4.5 branching factor, perhaps:
- Exhaustive CFR (no sampling) on full state, or
- Perfect Information Monte Carlo (PIMC) with minimax

...are more appropriate than abstracted MCCFR.

## Decision: Punt CFR

"Boring and competent" (Nash equilibrium play) isn't worth the squeeze when:
1. We could achieve similar quality with fixed PIMC + minimax
2. Neural networks offer more upside for developing interesting "style"
3. The abstraction problem is deep and the game is small enough for other approaches

## Reference Commits

The MCCFR implementation can be recovered from these commits:

- `665c749` - CFD2 ultra-compact format implementation
- `53d8a40` - CFD2 implementation complete
- `eec9ee6` - Training up to 100k iterations
- `l4t` issue closed at `dfa3ef2` - Last commit with integrated MCCFR

## Future Directions

If CFR is revisited, consider:

1. **Suit-aware abstraction** - Include trump suit and suit composition in info set
2. **Card isomorphism only** - Preserve relative suit ordering, just collapse equivalent positions
3. **Warm-start from heuristics** - Initialize regrets from hand-coded strategy
4. **Hybrid approach** - CFR for bidding, minimax for trick-play

## Files Removed

```
src/game/ai/cfr/           # Entire directory
src/game/ai/cfr-metrics.ts # Metrics and abstraction code
scripts/train-mccfr.ts     # Single-process trainer
scripts/train-mccfr-parallel.ts  # Parallel trainer
src/tests/ai/cfr/          # Test directory
public/trained-strategy.json     # 172MB trained model
trained-strategy-100k.json       # 100k iteration checkpoint
```

---

*Archived: December 2024*
