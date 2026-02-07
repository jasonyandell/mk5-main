# 15: PyTorch TransformerEncoder Edge Effects Analysis

## Summary

**FINDING: PyTorch's nn.TransformerEncoder has no inherent edge effects that would explain slot 0 degradation in this model.** The attention sink phenomenon documented in recent research (2024-2025) occurs primarily due to softmax normalization pressure and positional encodings - neither of which applies to this architecture.

## Model Configuration Verified

From `forge/ml/module.py`:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=n_heads,
    dim_feedforward=ff_dim,
    dropout=dropout,
    batch_first=True,  # Explicit: (batch, seq, feature) ordering
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
```

Key settings (using PyTorch defaults):
- `batch_first=True` - Non-standard but correctly handled
- `norm_first=False` (default) - Post-LN architecture
- `layer_norm_eps=1e-5` (default) - Standard epsilon
- No positional encoding of any kind
- No causal masking (`is_causal=False` by default for encoder)

## Research on Transformer Position Bias

### Attention Sink Phenomenon (2024-2025 Research)

Recent papers have extensively studied the "attention sink" phenomenon where certain tokens (often position 0) receive disproportionate attention:

1. **"When Attention Sink Emerges in Language Models: An Empirical View"** (Gu et al., 2024/ICLR 2025)
   - >70% of attention heads show sink behavior toward first token
   - Emerges from softmax normalization requiring attention weights sum to 1
   - **Key finding**: Replacing softmax with sigmoid attention **eliminates** attention sinks
   - Sinks store "extra" attention when no strong semantic match exists

2. **"On the Emergence of Position Bias in Transformers"** (arxiv:2502.01951)
   - Causal masking inherently biases toward earlier positions
   - RoPE's distance decay compounds with causal masking
   - **Bidirectional encoders do NOT exhibit the same early-position bias**

3. **"What are you sinking? A geometric approach on attention sink"** (Ruscio et al., 2025)
   - Attention sinks serve as "geometric anchors" in representation space
   - Three reference frame types: centralized, distributed, bidirectional
   - In BERT-style encoders, [CLS] and [SEP] tokens establish dual anchors

4. **"On the Role of Attention Masks and LayerNorm in Transformers"** (Wu, 2024)
   - Pure self-attention suffers from rank collapse with depth
   - LayerNorm can actually **prevent** collapse to rank-1 subspace
   - No position-specific effects documented in LayerNorm

### Why Attention Sinks Don't Explain This Model's Slot 0 Bias

| Factor | LLM Attention Sinks | This Model |
|--------|---------------------|------------|
| Positional encoding | RoPE/sinusoidal | **None** |
| Attention type | Causal (triangular mask) | **Bidirectional** |
| Special tokens | [CLS], [BOS] at position 0 | Context token at position 0 |
| Output extraction | From same position as sink | From positions 1-7 (hand tokens) |
| Depth | 12-96 layers | **2 layers** |
| Training data scale | Billions of tokens | ~1M samples |

**Critical distinction**: The attention sink research focuses on:
1. **Causal masking** - which this model doesn't use
2. **Positional encodings** - which this model doesn't have
3. **Deep networks** - this model has only 2 layers

## PyTorch TransformerEncoder Specifics

### No Position-Specific Code Paths

Reviewed PyTorch source (`torch/nn/modules/transformer.py`):
- No special handling for first/last positions
- All positions processed uniformly through the same linear projections
- LayerNorm applied identically across all sequence positions

### Known PyTorch Issues

Searched GitHub issues for position-related bugs:

1. **Issue #97111**: `enable_nested_tensor=True` can truncate output when position 0 is masked
   - **NOT applicable**: We don't use nested tensors; position 0 (context) is never masked

2. **Issue #104595**: Future token leakage with causal mask
   - **NOT applicable**: We use bidirectional attention (no causal mask)

3. **Issue #97532**: Fast path mask dtype warning
   - Fixed in PyTorch 2.1.0
   - Cosmetic issue, doesn't affect computation

No issues found relating to position 0 degradation in normal encoder operation.

### LayerNorm Edge Effects

Research on LayerNorm behavior:
- **"LayerNorm Induces Recency Bias in Transformer Decoders"** (arxiv:2509.21042)
  - Shows LayerNorm + causal masking creates recency bias
  - **NOT applicable**: We use bidirectional attention

LayerNorm is applied per-token, computing mean/variance across the embedding dimension:
```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```
This is position-independent - no mathematical mechanism for edge effects.

## Permutation Equivariance Analysis

Without positional encoding, `nn.TransformerEncoder` is permutation-equivariant:

```
f(permute(X)) = permute(f(X))
```

**Implication**: If we permuted hand token positions, the Q-value outputs would permute identically. The slot 0 bias must arise from something that **breaks** this symmetry:

1. **Input tokenization order** - Dominoes are always fed in a fixed order (by domino ID)
2. **Fixed output extraction** - `torch.gather` extracts from positions 1-7 in order
3. **Training data distribution** - Slot 0 always contains the lowest-ID domino in hand

## Conclusion

**PyTorch's `nn.TransformerEncoder` has no inherent edge effects that could cause position 0 degradation.**

The attention sink literature is not applicable because:
1. This model has no positional encoding (permutation equivariant)
2. This model uses bidirectional attention (no causal masking)
3. The output is extracted from positions 1-7, not position 0

**The slot 0 bias must originate from training data distribution** - specifically, the fixed domino ordering where slot 0 always contains the player's lowest-ID domino (often 0-0, 0-1, etc.). This creates an implicit position signal through the domino ID embeddings that correlates with slot position.

## References

1. Gu et al. (2024). "When Attention Sink Emerges in Language Models: An Empirical View." ICLR 2025. https://arxiv.org/abs/2410.10781
2. "On the Emergence of Position Bias in Transformers." https://arxiv.org/abs/2502.01951
3. Wu (2024). "On the Role of Attention Masks and LayerNorm in Transformers." https://arxiv.org/abs/2405.18781
4. Ruscio et al. (2025). "What are you sinking? A geometric approach on attention sink." https://arxiv.org/abs/2508.02546
5. PyTorch TransformerEncoderLayer documentation. https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
6. Lee et al. (2019). "Set Transformer: A Framework for Attention-based Set-to-Set Functions." ICML 2019.
