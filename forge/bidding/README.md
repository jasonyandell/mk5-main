# Bidding Evaluation Module

Monte Carlo simulation for evaluating bid strength in Texas 42.

## Overview

Given a hand of 7 dominoes, this module answers: **"What's P(make) for each trump/bid combination?"**

The approach:
1. Load a trained policy model (from `forge/models/`)
2. Simulate many games with random opponent deals
3. Count wins to estimate P(make) for each trump × bid combination

## Quick Start

```bash
# Evaluate a hand (default: 100 samples)
python -m forge.bidding.evaluate --hand "6-4,5-5,4-2,3-1,2-0,1-1,0-0" --samples 100

# Output as matrix (default) or list
python -m forge.bidding.evaluate --hand "..." --list

# JSON output for programmatic use
python -m forge.bidding.evaluate --hand "..." --json

# Generate visual PDF poster
python -m forge.bidding.poster --hand "..." --output poster.pdf
```

## Module Structure

| File | Purpose |
|------|---------|
| `evaluate.py` | CLI entry point, hand parsing |
| `simulator.py` | Batched game simulation (vectorized PyTorch) |
| `inference.py` | Model loading and batched forward passes |
| `estimator.py` | P(make) calculation, Wilson CI, output formatting |
| `poster.py` | PDF generation with domino tiles and heatmaps |
| `parallel.py` | Multi-process simulation for multi-GPU clusters |
| `benchmark.py` | Performance benchmarking and profiling |
| `convergence.py` | Sample size vs accuracy analysis |
| `stability_experiment.py` | Variance analysis across RNG seeds |
| `investigate.py` | Debugging tools for specific simulation cases |

## Key Classes

### `PolicyModel` (inference.py)

Wrapper around trained model for fast batched inference:

```python
from forge.bidding.inference import PolicyModel

# Load model (uses default checkpoint if none specified)
model = PolicyModel()

# Warmup for consistent batch size (avoids recompilation)
model.warmup(batch_size=500)

# Get greedy actions
actions = model.greedy_actions(tokens, mask, current_player, legal_mask)

# Or sample from policy distribution
actions = model.sample_actions(tokens, mask, current_player, legal_mask)
```

### `simulate_games()` (simulator.py)

Runs batched game simulations:

```python
from forge.bidding.simulator import simulate_games

# Simulate 500 games with sixes trump
points = simulate_games(
    model=model,
    bidder_hand=[0, 1, 2, 3, 4, 5, 6],  # domino IDs
    decl_id=6,  # sixes trump
    n_games=500,
    greedy=True,  # vs sampling
    seed=42,
)
# points: Tensor of shape (500,) with Team 0 points per game
```

### `evaluate_bids()` (estimator.py)

Computes P(make) for all bid thresholds:

```python
from forge.bidding.estimator import evaluate_bids

results: TrumpResult = evaluate_bids(points, decl_id=6)
# results.bid_results[i].p_make = P(make) for bid threshold i
# results.bid_results[i].ci_low/ci_high = Wilson confidence interval
```

## Sample Size Guidelines

| Use Case | N | CI Width | Time/Trump |
|----------|---|----------|------------|
| Quick screening | 50 | ±0.15 | ~7s |
| Standard analysis | 100 | ±0.10 | ~14s |
| Publication quality | 200 | ±0.07 | ~28s |
| Ground truth | 500 | ±0.04 | ~70s |

See [CONVERGENCE.md](CONVERGENCE.md) for detailed analysis.

## Related Documentation

- [EXAMPLES.md](EXAMPLES.md) - Worked examples with analysis
- [CONVERGENCE.md](CONVERGENCE.md) - Sample size vs accuracy tradeoffs
- [../models/README.md](../models/README.md) - Model architecture and training details
