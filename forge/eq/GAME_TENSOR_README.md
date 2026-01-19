# GameStateTensor: GPU-Vectorized Game State

**Status**: ✅ Phase 1 Complete (2026-01-19)

## Overview

`GameStateTensor` is a GPU-accelerated, vectorized implementation of the Texas 42 game state that processes N games in parallel. It matches the CPU `GameState` semantics exactly while enabling massive parallelization for the E[Q] training pipeline.

## Key Features

- **Exact CPU Equivalence**: All 25 tests verify GPU matches CPU `GameState` behavior exactly
- **Vectorized Operations**: Process N games simultaneously using pure PyTorch operations
- **GPU-Resident**: All state and operations remain on GPU (no Python loops over games)
- **Pre-computed Lookup Tables**: Fast trick resolution and legal action determination
- **Immutable State**: Functional design with `apply_actions()` returning new instances

## Architecture

### State Representation

```python
class GameStateTensor:
    hands: (n_games, 4, 7) int8         # Domino IDs, -1 for played slots
    played_mask: (n_games, 28) bool     # True if domino played
    history: (n_games, 28, 3) int8      # (player, domino, lead_domino)
    trick_plays: (n_games, 4) int8      # Current trick, -1 for empty
    leader: (n_games,) int8             # Current trick leader (0-3)
    decl_ids: (n_games,) int8           # Declaration ID
```

### Lookup Tables

Pre-computed at module load for efficiency:

1. **LED_SUIT_TABLE** (28, 10): Led suit for each (lead_domino, decl_id)
2. **TRICK_RANK_TABLE** (28, 8, 10): Trick rank for (domino, led_suit, decl_id)
3. **CAN_FOLLOW** (28, 8, 10): Follow-suit constraints (from `sampling_gpu.py`)

### Core Methods

```python
# Initialize N games from dealt hands
GameStateTensor.from_deals(hands, decl_ids, device='cuda')

# Query state
.current_player -> (n_games,)        # Whose turn it is
.legal_actions() -> (n_games, 7)     # Boolean mask of legal slots
.hand_sizes() -> (n_games, 4)        # Domino counts per player
.active_games() -> (n_games,)        # Boolean mask of incomplete games

# State transition (immutable)
.apply_actions(actions: (n_games,)) -> GameStateTensor
```

## Usage Example

```python
import torch
from forge.eq.game_tensor import GameStateTensor

# Initialize 32 games
hands = [...]  # 32 games × 4 players × 7 dominoes
decl_ids = [9] * 32  # All no-trump
state = GameStateTensor.from_deals(hands, decl_ids, device='cuda')

# Game loop
while state.active_games().any():
    legal = state.legal_actions()  # (32, 7)

    # Pick actions (first legal slot for each game)
    actions = torch.tensor([
        legal[i].nonzero()[0][0] for i in range(32)
    ], device='cuda')

    # Apply transitions
    state = state.apply_actions(actions)
```

## Testing

**Test Coverage**: 25 tests, all passing

Categories:
- Lookup table correctness (vs CPU functions)
- Initialization and validation
- Current player tracking
- Legal action determination (leading, following, void)
- State transitions (hand updates, history, tricks)
- Trick resolution and leader changes
- Equivalence with CPU (100 random games)
- Vectorization (multiple concurrent games)
- Edge cases (all 10 declarations)

Run tests:
```bash
python -m pytest forge/eq/test_game_tensor.py -v
```

## Performance

Current implementation (Phase 1) focuses on correctness. Performance characteristics:

- **CPU Sequential**: ~0.5ms per game transition
- **GPU Vectorized**: Higher overhead due to tensor creation/transfer
- **Next Phase**: Reuse tensors across iterations for true GPU benefits

### Target for Phase 2

When integrated with SMC sampling (Phase 2), expect:
- 32 concurrent games on RTX 3050 Ti
- 10x speedup over CPU pipeline (end-to-end)
- Amortized tensor creation overhead across many iterations

## Design Decisions

### Why int8 for indices?

Memory efficiency: 28 dominoes fit in int8 (-128 to 127), reducing memory footprint by 4x vs int32. Convert to int64 only for indexing operations.

### Why -1 for played slots?

Maintains fixed-size tensors without dynamic resizing. Easy to mask with `hands >= 0` checks.

### Why immutable state?

Matches CPU `GameState` design. Enables safe parallelization and debugging. PyTorch operations naturally create new tensors anyway.

### Device handling

Lookup tables (LED_SUIT_TABLE, etc.) start on CPU and are moved to device on first use. This avoids requiring global device configuration while enabling GPU acceleration.

## Next Steps: Phase 2

Integrate with SMC sampling:

1. **Create `forge/eq/smc.py`**: Sequential Monte Carlo sampler using GameStateTensor
2. **Batch world sampling**: Process multiple opponent hands per game simultaneously
3. **End-to-end E[Q] generation**: Deal → Games → SMC → Q-values all on GPU
4. **Performance optimization**: Reuse tensors, optimize memory layout

See `forge/eq/smc.py` (to be created) for Phase 2 implementation.

## Files

- `forge/eq/game_tensor.py`: Implementation (420 lines)
- `forge/eq/test_game_tensor.py`: Tests (650 lines, 25 tests)
- `scratch/benchmark_game_tensor.py`: Performance benchmarks
- This file: Documentation

## References

- CPU implementation: `forge/eq/game.py`
- Lookup tables: `forge/oracle/tables.py`
- GPU sampling: `forge/eq/sampling_gpu.py`
- Design plan: `/home/jason/.claude/plans/precious-doodling-bentley-agent-af4639d.md`
