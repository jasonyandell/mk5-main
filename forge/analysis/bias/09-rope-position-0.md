# RoPE Position 0 Investigation

## Summary

**RoPE is NOT used.** The DominoTransformer uses no positional encoding of any kind, including Rotary Position Embedding (RoPE).

## Background

The hypothesis was that Rotary Position Embedding might cause degenerate behavior at position 0 because:
- RoPE applies rotation matrices based on position index
- At position 0, the rotation angle theta = 0
- cos(0) = 1, sin(0) = 0 creates an identity-like rotation that doesn't inject positional information
- This could cause the model to treat position 0 differently

## Investigation

### Architecture Review

The `DominoTransformer` in `forge/ml/module.py` uses:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=n_heads,
    dim_feedforward=ff_dim,
    dropout=dropout,
    batch_first=True,
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
```

PyTorch's `TransformerEncoderLayer` and `TransformerEncoder`:
- **Do NOT include any positional encoding by default**
- No sinusoidal PE is added
- No learned positional embeddings
- No RoPE or ALiBi

### Code Search Results

Searched `forge/ml/` for:
- `positional`, `RoPE`, `rotary`, `position_embed`: **No matches**
- `nn.Embedding.*position`, `pos_embed`, `pe_`: **No matches**

The only embeddings defined are semantic features:
- `high_pip_embed`, `low_pip_embed` (domino pips)
- `is_double_embed`, `count_value_embed` (domino properties)
- `trump_rank_embed` (game-specific ranking)
- `player_id_embed` (relative player position: 0=current, 2=partner, etc.)
- `token_type_embed` (context vs hand vs trick tokens)
- `decl_embed`, `leader_embed` (game state)

### Forward Pass Analysis

The forward pass (lines 71-124 in module.py) shows:
1. Embeddings are concatenated from semantic features only
2. Projected through `input_proj` linear layer
3. Passed to transformer encoder with only `src_key_padding_mask`
4. No positional encoding is added at any point

## Conclusion

**The RoPE hypothesis is ruled out.** The model uses no positional encoding whatsoever.

The slot 0 bias cannot be attributed to:
- RoPE degenerate rotation at position 0 (NOT using RoPE)
- Sinusoidal PE where sin(0) = 0 (NOT using sinusoidal PE)
- Learned positional embeddings (NOT using any PE)

The transformer is architecturally **position-agnostic** for input tokens. All tokens are treated as an unordered set (permutation equivariant), distinguished only by their semantic embeddings (player_id, token_type, etc.).

The slot 0 output bias must originate elsewhere - likely in the output extraction mechanism where `torch.gather` extracts representations in fixed deal order, mapping the first-dealt domino always to output slot 0.

## Reference

For a complete analysis of positional encoding absence, see: `03-positional-encoding.md`
