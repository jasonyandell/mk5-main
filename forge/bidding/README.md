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

## Continuous Evaluation

For generating large-scale bidding evaluation datasets, use the continuous CLI:

```bash
# Run forever with N=500 samples per declaration
python -m forge.cli.bidding_continuous

# Single seed test
python -m forge.cli.bidding_continuous --limit 1

# Preview gaps without running
python -m forge.cli.bidding_continuous --dry-run

# Start at specific seed
python -m forge.cli.bidding_continuous --start-seed 1000
```

### Output Schema

Each seed produces a parquet file at `data/bidding-results/{train,val,test}/seed_XXXXXXXX.parquet`:

| Column Pattern | Type | Description |
|----------------|------|-------------|
| `seed` | int64 | RNG seed for hand generation |
| `hand` | string | Hand as "6-4,5-5,..." format |
| `n_samples` | int32 | Samples per declaration |
| `points_{decl}` | list[int8] | Raw simulation results (N values, 0-84) |
| `pmake_{decl}_{bid}` | float32 | P(make) for bid threshold |
| `ci_low_{decl}_{bid}` | float32 | Wilson CI lower bound |
| `ci_high_{decl}_{bid}` | float32 | Wilson CI upper bound |

- **Declarations**: 0-7 and 9 (skips 8=doubles-suit)
- **Bid thresholds**: 30-42 (13 values)
- **Total columns**: 365 (5 metadata + 9 points arrays + 351 stats)

### Performance

| N | CI Width | Time/seed | Seeds/day (1 GPU) |
|---|----------|-----------|-------------------|
| 500 | ±0.04 | ~10 min | ~137 |

## Debugging & Analysis Tools

### investigate.py - Debug Losing Games

When a near-certain hand loses unexpectedly, use `investigate.py` to replay games trick-by-trick:

```bash
# Find games where Team 0 scored below 42 with sixes trump
python -m forge.bidding.investigate \
    --hand "6-6,6-5,6-4,6-2,6-1,6-0,2-2" \
    --trump sixes \
    --below 42 \
    --samples 500

# Show more losing games (default: 5)
python -m forge.bidding.investigate --hand "..." --trump sixes --max-show 10

# Use sampling instead of greedy play
python -m forge.bidding.investigate --hand "..." --trump sixes --sample
```

Output includes:
- All 4 players' hands (showing how opponents were dealt)
- Trick-by-trick replay with winner and points
- Identification of where Team 0 lost tricks

### stability_experiment.py - Variance Analysis

Compare sample size stability across random hands:

```bash
# Run stability experiment on 100 random hands
python -m forge.bidding.stability_experiment --hands 100 --output results.csv
```

Compares N=200 vs N=500 vs N=1000 to determine if rankings are stable at lower sample counts.

## Related Documentation

- [EXAMPLES.md](EXAMPLES.md) - Worked examples with analysis
- [CONVERGENCE.md](CONVERGENCE.md) - Sample size vs accuracy tradeoffs
- [../models/README.md](../models/README.md) - Model architecture and training details
