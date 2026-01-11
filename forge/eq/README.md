# Stage 2: E[Q] Training (Transcript Tokenization)

This directory contains the tokenizer for **Stage 2** of the training pipeline.

## Key Difference from Stage 1

**Stage 1** (`forge/ml/tokenize.py`):
- Sees ALL hands (omniscient oracle perspective)
- Used to train policy network from perfect information
- Input: Full game state with all 28 dominoes positioned

**Stage 2** (`forge/eq/transcript_tokenize.py`):
- Sees ONLY PUBLIC information
- Used to predict E[Q] from observable game state
- Input: My hand + play transcript + declaration

## Why Public Information Only?

Stage 2 learns to estimate expected quality (E[Q]) from what's visible during actual play. This is critical for:

1. **Realistic inference**: Can't see opponent hands during real games
2. **Transfer learning**: Model learns patterns that generalize
3. **Marginalization**: Averages over possible opponent hands

## Token Format

```python
tokenize_transcript(
    my_hand=[0, 1, 2],           # Current player's remaining dominoes
    plays=[(0, 3), (1, 4)],      # (absolute_player, domino_id) pairs
    decl_id=5,                   # Trump declaration
    current_player=0             # Perspective for relative player IDs
)
```

Returns tensor of shape `(seq_len, 8)` where `seq_len = 1 + len(my_hand) + len(plays)`:

| Index | Token Type | Description |
|-------|------------|-------------|
| 0 | Declaration | Trump type (0-9) |
| 1..N | Hand | Current player's remaining dominoes |
| N+1.. | Plays | Chronological play transcript |

### Feature Vector (8 dimensions)

Each token has 8 features:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | high_pip | 0-6 | High pip value |
| 1 | low_pip | 0-6 | Low pip value |
| 2 | is_double | 0-1 | 1 if double, 0 otherwise |
| 3 | count_value | 0-2 | Points: 0→0pts, 1→5pts, 2→10pts |
| 4 | player | 0-3 | Relative player (0=me, 1=left, 2=partner, 3=right) |
| 5 | is_in_hand | 0-1 | 1 if in current hand, 0 if played |
| 6 | decl_id | 0-9 | Declaration ID |
| 7 | token_type | 0-2 | 0=decl, 1=hand, 2=play |

## Relative Player IDs

All player IDs are **relative to `current_player`**:

```python
relative_player = (absolute_player - current_player) % 4
```

From any player's perspective:
- `0` = me (current player)
- `1` = left opponent
- `2` = partner
- `3` = right opponent

This ensures position-invariant learning - the model learns "my partner played X" rather than "player 2 played X".

## Usage

```python
from forge.eq import tokenize_transcript

# Game in progress: player 1's turn after 2 plays
my_hand = [5, 7, 12]  # 3 dominoes remaining
plays = [
    (0, 20),  # Player 0 led with (5,5)
    (1, 15),  # Player 1 played (5,0)
]
decl_id = 5  # Trump is fives
current_player = 1

tokens = tokenize_transcript(my_hand, plays, decl_id, current_player)
# Shape: (6, 8) = 1 decl + 3 hand + 2 plays

# Feed to transformer for E[Q] prediction
logits = model(tokens)
```

## Testing

Tests are designed for **speed** (< 5 seconds total):

```bash
python -m pytest forge/eq/test_transcript_tokenize.py -v
```

All tests use minimal inputs (1-3 plays) to catch bugs quickly without wasting time on large inputs.

## Design Philosophy

1. **Simplicity**: Flat sequence of tokens, no fancy structure
2. **Position-invariant**: Relative player IDs ensure transferable learning
3. **Minimal features**: 8 features capture essential domino properties
4. **Fast tests**: Keep development velocity high

## Stage 1 Oracle for E[Q] Marginalization

The `Stage1Oracle` class wraps a trained Stage 1 model for efficient batch querying during Stage 2 training.

### Why We Need This

Stage 2 trains on **public information only**, but we need target Q-values. The oracle:

1. Takes N sampled world states (possible opponent hands)
2. Queries Stage 1 model for Q-values in each world
3. Returns logits for averaging/marginalization

### Usage Example

```python
from forge.eq import Stage1Oracle
from forge.oracle.rng import deal_from_seed
import numpy as np
import torch

# Load trained Stage 1 checkpoint
oracle = Stage1Oracle("checkpoints/stage1.ckpt", device="cuda")

# Sample 100 possible worlds
worlds = [deal_from_seed(i) for i in range(100)]

# Game state (same for all worlds)
game_state_info = {
    'decl_id': 3,  # threes trump
    'leader': 0,
    'trick_plays': [],  # Start of hand
    'remaining': np.ones((100, 4), dtype=np.int64) * 0x7F,
}

# Query all worlds in batch
logits = oracle.query_batch(
    worlds=worlds,
    game_state_info=game_state_info,
    current_player=0,
)  # Shape: (100, 7)

# Marginalize to get E[Q]
probs = torch.softmax(logits, dim=-1)
avg_probs = probs.mean(dim=0)  # (7,) - expected quality per action
```

### Key Features

- **Reuses Stage 1 tokenization**: No code duplication
- **Batch efficient**: Single forward pass for N worlds
- **Returns raw logits**: Caller handles softmax/averaging
- **Fast tests**: < 5s with mocks, no real checkpoint needed

### Testing

```bash
# Fast unit tests (mocked model)
python -m pytest forge/eq/test_oracle.py -v --timeout=30

# Skip slow integration tests
python -m pytest forge/eq/test_oracle.py -m "not slow"

# Performance test (100 worlds in < 2s)
python -m pytest forge/eq/test_oracle.py::test_batch_query_performance -v
```

## Future Extensions

Possible additions (not implemented yet):

- Trump rank precomputation (like Stage 1)
- Attention masks for trick boundaries
- Lead suit encoding
- Running score features

For now, we keep it simple and let the transformer learn these patterns.
