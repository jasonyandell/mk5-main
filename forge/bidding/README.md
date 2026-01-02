# Bidding Evaluation (System 2)

Monte Carlo simulation to evaluate bid strength by playing out complete games.

## How It Works

1. Fix the bidder's 7 dominoes (player 0)
2. Randomly deal remaining 21 dominoes to players 1, 2, 3
3. Simulate complete games using the policy model (greedy action selection)
4. Count how often team 0 scores >= threshold for each bid level

**Key insight**: The simulation naturally captures partner synergy. A 2% chance of making 42 with a weak trump suit means "occasionally your partner gets dealt a monster hand."

## Entry Points

| Script | Purpose | Example |
|--------|---------|---------|
| `evaluate.py` | P(make) matrix for a hand | `python -m forge.bidding.evaluate --hand "6-4,5-5,..." --samples 100` |
| `poster.py` | Visual PDF with heatmap | `python -m forge.bidding.poster --hand "..." --output poster.pdf` |
| `investigate.py` | Debug losing games | `python -m forge.bidding.investigate --hand "..." --trump sixes --below 42` |

Use `--help` on any script for full options.

## Documentation

- **[EXAMPLES.md](EXAMPLES.md)** - Worked examples with analysis (monster hands, marginal hands, when to pass)
- **[CONVERGENCE.md](CONVERGENCE.md)** - Sample size vs accuracy tradeoffs

## Module Structure

| File | Purpose |
|------|---------|
| `simulator.py` | Core game simulation engine |
| `inference.py` | Model loading and batched inference |
| `estimator.py` | P(make) estimation with confidence intervals |
| `parallel.py` | Multi-process simulation for speed |
| `benchmark.py` | Performance profiling |
| `convergence.py` | Sample size analysis |
