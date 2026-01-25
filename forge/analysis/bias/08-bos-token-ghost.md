# 08: BOS Token Ghost Investigation

**Question**: Is a "BOS token ghost" causing positional bias at slot 0 in the Q-value model?

**Summary**: NO - the BOS token ghost hypothesis is RULED OUT. The DominoTransformer is trained entirely from scratch with randomly initialized weights. There are no pretrained weights from HuggingFace or other sources, and PyTorch's `nn.TransformerEncoder` does not have any built-in position 0 special handling.

---

## Background

The "BOS token ghost" phenomenon occurs when:
1. A model uses pretrained weights (e.g., from BERT, GPT) that expect position 0 to contain a special token (BOS, CLS, etc.)
2. The model's positional encoding has special handling for position 0 (e.g., learned embeddings trained to represent "start of sequence")
3. The architecture has implicit assumptions that position 0 contains metadata rather than actionable content

This investigation checks whether any of these conditions apply to the DominoTransformer.

---

## Analysis

### 1. No Pretrained Weights

**Finding**: The model is initialized entirely from scratch.

From `forge/ml/module.py`, the `DominoTransformer` class creates all its own embeddings:

```python
class DominoTransformer(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4, n_layers=2, ...):
        super().__init__()

        # Feature embeddings - ALL created fresh
        self.high_pip_embed = nn.Embedding(7, embed_dim // 6)
        self.low_pip_embed = nn.Embedding(7, embed_dim // 6)
        # ... 10 more embeddings, all randomly initialized

        self.input_proj = nn.Linear(self._calc_input_dim(), embed_dim)

        # Transformer - PyTorch's nn.TransformerEncoder, fresh weights
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
```

There are **zero** calls to:
- `from_pretrained()`
- `AutoModel.from_pretrained()`
- `BertModel.from_pretrained()`
- Any HuggingFace model loading

The `--resume` flag in `forge/cli/train.py` loads weights from a previous DominoTransformer checkpoint, not from external pretrained models.

### 2. No HuggingFace Integration

**Finding**: The codebase does not use any HuggingFace transformers.

A comprehensive grep for `transformers\.`, `HuggingFace`, `AutoModel`, `BertModel`, `GPT` returned zero matches in the `forge/ml/` directory. The model uses only:
- `torch.nn.TransformerEncoder` (PyTorch native)
- `torch.nn.TransformerEncoderLayer` (PyTorch native)

### 3. No Positional Encoding

**Finding**: The model has NO positional encoding at all.

As documented in `03-positional-encoding.md`, the DominoTransformer deliberately omits positional encoding:

```python
# From forward():
x = torch.cat(embeds, dim=-1)
x = self.input_proj(x)
# NO positional encoding added here
x = self.transformer(x, src_key_padding_mask=attn_mask)
```

PyTorch's `nn.TransformerEncoder` does **not** include built-in positional encoding - the user must add it explicitly. This model chooses not to, meaning:
- No sinusoidal positional encoding (sin(0)=0 issue does not exist)
- No learned positional embeddings (no special learned "position 0" pattern)
- The transformer is permutation-equivariant over tokens of the same type

### 4. No CLS/BOS Token Convention

**Finding**: Position 0 contains a context token, but it is NOT used for action prediction.

The tokenization layout (from `02-tokenization-layout.md`):

```
Position 0:      Context token (global game info: decl_id, leader)
Positions 1-7:   Player 0's hand
Positions 8-14:  Player 1's hand
...
```

The model uses position 0 only for the **value head**:
```python
value = self.value_head(x[:, 0, :]).squeeze(-1)
```

For **action prediction** (Q-values), the model extracts from positions 1-7 (or 8-14, etc. depending on current player):
```python
start_indices = 1 + current_player * 7  # Position 1, 8, 15, or 22
hand_repr = torch.gather(x, dim=1, index=gather_indices)
logits = self.output_proj(hand_repr).squeeze(-1)  # Output slots 0-6
```

**Critical distinction**: The "slot 0" bias is in the **output** (action 0 of 7), not the **input** (position 0 of 32). Output slot 0 corresponds to input position 1 (for player 0), not input position 0.

### 5. PyTorch TransformerEncoder Internals

**Finding**: PyTorch's TransformerEncoder has no position-specific handling.

The `nn.TransformerEncoder` is a sequence of `nn.TransformerEncoderLayer` modules, each containing:
- Multi-head self-attention (no position bias without explicit PE)
- Feed-forward network (applied identically to all positions)
- LayerNorm (no position-specific parameters)

There is no built-in assumption that position 0 is special. The only way position 0 would be treated differently is if:
1. Explicit positional encoding was added (not done)
2. The attention mask treats position 0 specially (ruled out in `01-attention-mask.md`)
3. Padding appears at position 0 (ruled out in `05-padding-convention.md`)

---

## Conclusion

**The BOS token ghost hypothesis is RULED OUT.**

The DominoTransformer:
1. Uses no pretrained weights - all parameters are randomly initialized
2. Has no HuggingFace integration - uses only PyTorch native modules
3. Has no positional encoding - no learned or sinusoidal PE exists
4. Has no CLS/BOS token architecture - position 0 is a context token used only for value prediction, not Q-values
5. Uses PyTorch's vanilla TransformerEncoder which has no position-0 special handling

The slot 0 bias in Q-value prediction cannot be caused by BOS token ghost effects because:
- Output slot 0 maps to input position 1 (not 0)
- Input position 1 is a normal hand token with no special treatment
- The model has never seen pretrained weights that expect special tokens

The root cause must lie elsewhere. Based on previous investigations:
- Not attention masking (01)
- Not tokenization offset (02)
- Not positional encoding (03)
- Not padding (05)
- Not BOS token ghost (this investigation)

Remaining hypotheses to investigate:
- Training data statistics (is slot 0 action frequency different?)
- Gradient flow asymmetries during training
- Numerical precision at gather operation boundaries
- Implicit attention order effects from sequential computation
