# Tokenization Layout Analysis

## Summary

**Finding**: There is NO off-by-one bug in the tokenization or Q-value indexing. Token position 0 contains a context token (not a hand token), and the model correctly extracts hand representations from positions 1-7 for the current player. The positional bias at slot 0 must have a different root cause.

## Token Layout (32 positions)

| Position | Type | Description |
|----------|------|-------------|
| **0** | Context | Global game context (decl_id, normalized_leader) |
| **1-7** | Player 0 hand | 7 dominoes for player 0 |
| **8-14** | Player 1 hand | 7 dominoes for player 1 |
| **15-21** | Player 2 hand | 7 dominoes for player 2 |
| **22-28** | Player 3 hand | 7 dominoes for player 3 |
| **29-31** | Trick plays | Up to 3 dominoes played in current trick |

### Token Type Values (feature index 9)

```python
TOKEN_TYPE_CONTEXT = 0   # Position 0
TOKEN_TYPE_PLAYER0 = 1   # Positions 1-7  (token_type = 1 + player_id)
TOKEN_TYPE_PLAYER1 = 2   # Positions 8-14
TOKEN_TYPE_PLAYER2 = 3   # Positions 15-21
TOKEN_TYPE_PLAYER3 = 4   # Positions 22-28
TOKEN_TYPE_TRICK_P0 = 5  # Position 29
TOKEN_TYPE_TRICK_P1 = 6  # Position 30
TOKEN_TYPE_TRICK_P2 = 7  # Position 31
```

## Feature Encoding (12 features per token)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | high_pip | 0-6 | Higher pip count on domino |
| 1 | low_pip | 0-6 | Lower pip count on domino |
| 2 | is_double | 0-1 | 1 if double (high == low) |
| 3 | count_value | 0-2 | 0=0pts, 1=5pts, 2=10pts |
| 4 | trump_rank | 0-7 | Rank in trump suit (0=best, 7=not trump) |
| 5 | normalized_player | 0-3 | Player relative to current (0=self) |
| 6 | is_current | 0-1 | 1 if this is current player's token |
| 7 | is_partner | 0-1 | 1 if this is partner's token |
| 8 | is_remaining | 0-1 | 1 if domino still in hand (not played) |
| 9 | token_type | 0-7 | See above |
| 10 | decl_id | 0-9 | Declaration (trump suit) |
| 11 | normalized_leader | 0-3 | Trick leader relative to current |

### Context Token (position 0)

The context token at position 0 has:
- Features 0-8: All zeros (no domino-specific info)
- Feature 9: TOKEN_TYPE_CONTEXT (0)
- Feature 10: decl_id
- Feature 11: normalized_leader

## Model Architecture: Hand Extraction

From `forge/ml/module.py` (lines 110-118):

```python
# Extract hand representations for current player
# Player's 7 dominoes start at index 1 + player_id * 7
start_indices = 1 + current_player * 7
offsets = torch.arange(7, device=device)
gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)

hand_repr = torch.gather(x, dim=1, index=gather_indices)
logits = self.output_proj(hand_repr).squeeze(-1)  # [batch, 7]
```

### Key Insight: No Off-By-One

The model computes `start_indices = 1 + current_player * 7`:
- If current_player = 0: extracts positions 1-7 (player 0's hand)
- If current_player = 1: extracts positions 8-14 (player 1's hand)
- etc.

This means:
- **Output slot 0** -> **token position 1** (first domino of current player)
- **Output slot 6** -> **token position 7** (seventh domino of current player)

The mapping is correct. Output slot 0 corresponds to the first domino in the current player's tokenized hand, which is at token position `1 + current_player * 7`.

## Q-Value Indexing in Oracle Data

From `forge/ml/tokenize.py`, the parquet files have columns `q0` through `q6`:
- `q[i]` = Q-value for playing the domino at **local slot i** (0-6) of the current player's hand
- -128 indicates illegal action

The tokenization places the same domino ordering:
- Domino at `hands[player][0]` -> token position `1 + player * 7`
- Domino at `hands[player][6]` -> token position `7 + player * 7`

## Verification: Tokenizer Input-Output Alignment

The `process_shard()` function in `forge/ml/tokenize.py`:

1. Loads parquet with Q-values `q0-q6` for local slots 0-6
2. Builds tokens where hand tokens are at positions 1-28
3. For current player P, their dominoes are at positions `1 + P*7` through `7 + P*7`
4. Targets (best action) are argmax of `q0-q6` after legal masking

The model's forward pass:
1. Takes token positions `1 + current_player * 7` through `7 + current_player * 7`
2. Projects each to a logit (Q-value prediction)
3. Returns `logits[batch, 0-6]` corresponding to the same slots

**Conclusion**: The tokenization and model indexing are aligned. Slot 0 in Q-values corresponds to slot 0 in model output.

## Root Cause Must Be Elsewhere

Since there's no tokenization offset bug, the positional bias (r=0.81 at slot 0 vs r=0.99+ elsewhere) must come from:

1. **Attention pattern artifacts**: Position 0 might have different attention behavior
2. **Embedding initialization**: First token position may have edge effects
3. **Training data distribution**: Slot 0 might have different action frequency
4. **Transformer positional encoding interaction**: The context token at position 0 might interfere with nearby hand tokens

The slot swap test (`forge/analysis/scripts/slot_swap_test.py`) found that swapping token features between slot 0 and another slot causes the Q-values to follow the features, not the position. This suggests the bias is **content-related** (learned feature associations) rather than a pure output position bias.

## References

- `forge/ml/tokenize.py`: Tokenization pipeline (lines 248-317)
- `forge/ml/module.py`: Model forward pass (lines 71-124)
- `forge/ml/data.py`: Data loading (line 53-78)
- `forge/analysis/scripts/slot_swap_test.py`: Slot swap experiment
