# P(make) Convergence Analysis

How does P(make) estimate quality scale with sample size N?

## Key Questions

1. **Variance vs N**: How does CI width shrink as N increases?
2. **Stability**: At what N do rankings stabilize (best trump doesn't change)?
3. **Speed**: What's the time/accuracy tradeoff?

## Theoretical CI Widths (95% Wilson)

| N | p=0.1 | p=0.5 | p=0.9 |
|---|-------|-------|-------|
| 10 | 0.39 | 0.53 | 0.39 |
| 25 | 0.23 | 0.36 | 0.26 |
| 50 | 0.17 | 0.27 | 0.17 |
| 100 | 0.12 | 0.19 | 0.12 |
| 200 | 0.08 | 0.14 | 0.08 |
| 500 | 0.05 | 0.09 | 0.05 |

Note: CI is widest at p=0.5 (maximum uncertainty) and narrower at extremes.

## Empirical Results (seed=42)

Tested on 4 hands spanning the difficulty range:
- **monster**: 6-6, 6-5, 6-4, 6-3, 6-2, 6-1, 6-0 (guaranteed win)
- **strong**: 6-6, 6-5, 6-4, 5-5, 5-4, 3-3, 2-2 (near-lock)
- **marginal**: 6-4, 5-3, 4-2, 3-1, 2-0, 1-0, 0-0 (~50% make)
- **weak**: 5-0, 4-1, 3-2, 2-1, 1-0, 0-0, 4-0 (pass this)

### Summary by Sample Size

| N | Avg CI Width | Stability Rate | Avg Time/Trump |
|---|--------------|----------------|----------------|
| 10 | 0.37 | 75% | 1.7s |
| 25 | 0.20 | 50% | 3.7s |
| 50 | 0.15 | 100% | 7.1s |

### Raw Data

| Hand | N | Best Trump/Bid | P(make) | CI Width | Stable? | Time |
|------|---|----------------|---------|----------|---------|------|
| monster | 10 | sixes/30 | 100% | 0.28 | Yes | 14.4s |
| monster | 25 | sixes/30 | 100% | 0.13 | Yes | 29.4s |
| monster | 50 | sixes/30 | 100% | 0.07 | Yes | 51.5s |
| strong | 10 | fives/30 | 100% | 0.28 | No | 13.4s |
| strong | 25 | fives/30 | 100% | 0.13 | No | 29.2s |
| strong | 50 | sixes/30 | 100% | 0.07 | Yes | 56.4s |
| marginal | 10 | blanks/30 | 50% | 0.53 | Yes | 13.4s |
| marginal | 25 | fours/30 | 52% | 0.36 | No | 30.3s |
| marginal | 50 | blanks/30 | 42% | 0.26 | Yes | 57.5s |
| weak | 10 | blanks/30 | 90% | 0.39 | Yes | 14.0s |
| weak | 25 | blanks/30 | 96% | 0.19 | Yes | 30.4s |
| weak | 50 | blanks/30 | 84% | 0.20 | Yes | 61.6s |

## Observations

1. **Extreme probabilities converge fast**: The "monster" hand shows 100% at all sample sizes with tight CI.

2. **Marginal hands have high variance**: The 50% hands fluctuate the most between runs (CI width ~0.26-0.53).

3. **Rankings can flip at low N**: The "strong" hand picked fives at N=10,25 but sixes at N=50. Both are near 100%, so the "best" is essentially a tie.

4. **Time scales linearly**: ~1.7s/trump at N=10 → ~7s/trump at N=50.

## Recommendations

| Use Case | N | CI Width | Time | Notes |
|----------|---|----------|------|-------|
| Quick screening | 50 | ±0.15 | ~7s/trump | Good for "is this hand biddable?" |
| Standard analysis | 100 | ±0.10 | ~14s/trump | Default for most use cases |
| Publication quality | 200 | ±0.07 | ~28s/trump | When precision matters |
| Ground truth | 500 | ±0.04 | ~70s/trump | Benchmarking and validation |

**Default recommendation**: Use N=100 for routine evaluation. The CI width of ±0.10 is tight enough that rankings are stable, and 14s per trump (2 minutes total) is acceptable.

## Running the Analysis

```bash
# Quick analysis with default settings
python -m forge.bidding.convergence

# With custom seed
python -m forge.bidding.convergence --seed 123

# With specific checkpoint
python -m forge.bidding.convergence --checkpoint path/to/model.ckpt
```

To modify the sample sizes tested, edit `SAMPLE_SIZES` in `convergence.py`.
