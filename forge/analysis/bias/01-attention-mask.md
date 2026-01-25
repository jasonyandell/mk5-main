# Investigation 01: Attention Mask Analysis

## Question
Is causal masking causing the positional bias at slot 0?

## Finding: NO CAUSAL MASKING

The model uses **bidirectional (full) attention**, not causal masking. Position 0 can attend to all positions.

## Evidence

From `forge/ml/module.py`, lines 51-58 and 106-108:

```python
# Model construction (lines 51-58)
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=n_heads,
    dim_feedforward=ff_dim,
    dropout=dropout,
    batch_first=True,
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

# Forward pass (lines 106-108)
attn_mask = (mask == 0)  # Padding mask only
x = self.transformer(x, src_key_padding_mask=attn_mask)
```

### Key observations:

1. **No `is_causal=True` flag** - The `TransformerEncoderLayer` is called without the `is_causal` parameter (introduced in PyTorch 2.0)

2. **No explicit triangular mask** - There is no `nn.Transformer.generate_square_subsequent_mask()` or `torch.triu()` call anywhere in the codebase

3. **Only padding mask used** - The only mask applied is `src_key_padding_mask`, which masks padded positions (where `mask == 0`), NOT a causal/autoregressive mask

4. **Grep confirms** - Searched entire `forge/ml/` directory for `is_causal`, `causal`, `triangular`, `triu` - no matches

## PyTorch TransformerEncoder Default Behavior

Per PyTorch documentation:
- `src_mask` (square attention mask): NOT provided - defaults to None (full attention)
- `src_key_padding_mask` (key padding mask): Provided - masks padding tokens
- `is_causal` parameter: NOT provided - defaults to False (bidirectional)

When no `src_mask` is provided and `is_causal=False`, the transformer uses **full bidirectional attention** - every position can attend to every other non-padded position.

## Conclusion

**Causal masking is NOT the cause of slot 0 bias.**

Position 0 has full attention to all 7 hand positions (and all other tokens). The positional degradation must stem from another source:

- Positional encoding artifacts (learned vs sinusoidal)
- Token ordering in the sequence
- Value head using position 0 for state value (line 122: `value = self.value_head(x[:, 0, :])`)
- Training data distribution effects

## Next Investigation

Examine the tokenization order - are hand slots always in positions 1-7 relative to player, and is slot 0 systematically different in some way? Also investigate whether the value head's reliance on position 0 creates gradient competition.
