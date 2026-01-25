# 05: Padding Convention Investigation

**Question**: Is variable-length padding causing positional bias at slot 0 in the Q-value model?

**Summary**: NO - padding is not the cause. The tokenization uses **right-padding** and position 0 always contains meaningful content (context token for Stage 1, declaration token for Stage 2). PAD tokens never appear at position 0 during training.

---

## Analysis

### Stage 1 Tokenization (forge/ml/tokenize.py)

Stage 1 uses **fixed-length sequences** with the following structure:

```
Position 0:      Context token (always present)
Positions 1-28:  Hand tokens (4 players x 7 dominoes each)
Positions 29-31: Trick tokens (0-3 plays in current trick)
```

**Key observations**:

1. **No padding at position 0**: The context token is ALWAYS at position 0
   ```python
   # Line 255-258
   tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
   tokens[:, 0, 10] = decl_id
   tokens[:, 0, 11] = normalized_leader
   masks[:, 0] = 1
   ```

2. **Fixed structure for hand tokens**: Positions 1-28 always contain hand tokens for all 4 players. The model extracts the current player's hand using `current_player` to index into these positions.

3. **Variable-length trick section**: Only positions 29-31 are variable (0-3 trick plays). Trick tokens are right-padded (empty positions stay at 0).

4. **Mask usage**: The `masks` array marks valid tokens (1) vs padding (0). Padding only appears at positions 29-31 when tricks haven't been played.

### Stage 2 Tokenization (forge/eq/transcript_tokenize.py)

Stage 2 uses **variable-length sequences** for imperfect information:

```
Position 0:      Declaration token (always present)
Positions 1-N:   Current player's remaining hand (1-7 dominoes)
Positions N+1-M: Play history (0-28 plays)
```

**Key observations**:

1. **Declaration token always at position 0**:
   ```python
   # Line 103-106
   tokens[idx, FEAT_DECL] = decl_id
   tokens[idx, FEAT_TOKEN_TYPE] = TOKEN_TYPE_DECL
   idx += 1
   ```

2. **Right-padding in collate.py**: When batching, sequences are padded to MAX_TOKENS (36) by appending zeros:
   ```python
   # Lines 144-151
   if seq_len < MAX_TOKENS:
       padding = torch.zeros((MAX_TOKENS - seq_len, N_FEATURES), dtype=tokens.dtype)
       tokens = torch.cat([tokens, padding], dim=0)  # RIGHT-PADDING
   ```

3. **No left-padding**: There is no code that prepends padding to sequences. Position 0 always contains the declaration token.

### GPU Tokenization (forge/eq/tokenize_gpu.py)

The GPU tokenizer follows the same pattern as Stage 1:

```python
# Lines 253-256
tokens[:, 0, 9] = TOKEN_TYPE_CONTEXT
tokens[:, 0, 10] = decl_id
tokens[:, 0, 11] = normalized_leader
masks[:, 0] = 1
```

Position 0 always contains the context token.

---

## Why Position 0 is NOT Affected by Padding

1. **Context/Declaration token always at position 0**: Both Stage 1 and Stage 2 tokenization guarantee that position 0 has meaningful content.

2. **Right-padding only**: Padding zeros are appended to the END of sequences, not prepended.

3. **Mask handling in Transformer**: The model uses `src_key_padding_mask` (line 107-108 in module.py):
   ```python
   attn_mask = (mask == 0)  # True where padding
   x = self.transformer(x, src_key_padding_mask=attn_mask)
   ```
   This masks out padded positions in attention, but position 0 is never masked since `masks[:, 0] = 1`.

---

## Ruling Out Padding as the Cause

The positional bias at slot 0 cannot be caused by padding because:

1. **Slot 0 refers to output position, not input position**: The model outputs Q-values for 7 slots (0-6), corresponding to the current player's 7 hand positions. These are extracted from input positions 1-7 (for player 0), 8-14 (for player 1), etc.

2. **The extraction logic** (lines 111-117 in module.py):
   ```python
   start_indices = 1 + current_player * 7
   offsets = torch.arange(7, device=device)
   gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
   hand_repr = torch.gather(x, dim=1, index=gather_indices)
   logits = self.output_proj(hand_repr).squeeze(-1)
   ```

   This extracts 7 contiguous positions starting at `1 + current_player * 7`. For player 0, this is positions 1-7. Position 0 (context token) is not included.

3. **Value head uses position 0**: The value prediction specifically uses the context token:
   ```python
   value = self.value_head(x[:, 0, :]).squeeze(-1)
   ```
   If there were padding issues at position 0, this would be affected too.

---

## Conclusion

**Padding convention is NOT the cause of slot 0 positional bias.**

The model uses right-padding, and position 0 always contains the context/declaration token. The output slots 0-6 correspond to input positions that are never padded.

The cause of the slot 0 bias must be elsewhere - possibly:
- Positional encoding initialization or learned patterns
- Attention pattern asymmetries
- Training data distribution (if slot 0 actions are less common)
- The gather operation creating numerical instabilities at array boundaries

See other investigations in this directory for alternative hypotheses.
