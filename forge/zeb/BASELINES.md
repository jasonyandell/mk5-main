# Zeb Baseline Metrics

Initial evaluation of the Zeb self-play system with random and heuristic players.

## Summary

| Player Type | vs Random Win Rate | Avg Margin |
|------------|-------------------|------------|
| Random | ~50% | 0 |
| Heuristic | ~57% | +4.5 pts |
| Untrained Neural | ~50% | 0 |

## Methodology

- 200 games per matchup for statistical confidence
- Random assignment of bidder and declaration (skip_bidding=True)
- Team assignments: Player 0,2 (team 0) vs Player 1,3 (team 1)
- Seed-controlled for reproducibility (base_seed=42)

## Key Findings

### 1. Random vs Random (~50%)
- Validates game logic correctness
- Win rate 43-57% range expected due to variance
- ~4,800 games/sec on CPU

### 2. Heuristic Player (~57% vs Random)
Simple rules:
- Lead with highest trump, then highest non-trump
- Follow with lowest legal domino
- Minimize points given up when sloughing

This provides a reasonable target for trained models.

### 3. Trajectory Collection
- 28 steps per game (7 tricks × 4 players)
- ~12 games/sec with neural inference on GPU
- Tensor shapes validated for training

### 4. Model Configuration
- Small model: 75K params (64d, 2 heads, 2 layers)
- Medium model: ~300K params (128d, 4 heads, 4 layers)
- Inference: single forward pass per decision

## Training Targets

| Milestone | vs Random | vs Heuristic |
|-----------|-----------|--------------|
| Untrained | ~50% | ~43% |
| Epoch 5 | >52% | ~48% |
| Epoch 10 | >55% | >50% |
| Epoch 20 | >60% | >55% |

## Running Evaluation

```bash
python -m forge.zeb.scripts.baseline_eval
```

Generated: 2026-01-31
