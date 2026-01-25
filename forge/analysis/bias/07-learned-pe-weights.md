# Investigation 07: Learned Positional Embedding Weights

**Date**: 2026-01-25
**Checkpoint**: `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`
**Question**: Are learned positional embedding weights anomalous at position 0?

## Summary

**Finding: NO LEARNED POSITIONAL EMBEDDINGS EXIST.**

This model uses `nn.TransformerEncoder` without adding any positional encoding mechanism. There are no learned PE weights to analyze. The hypothesis that position 0 has anomalous positional embedding weights is definitively ruled out - there are no such weights.

## Evidence

### 1. Checkpoint Weight Analysis

Searched all state dict keys for positional embeddings:

```python
pe_keys = [k for k in state_dict.keys() if 'pos' in k.lower() or 'position' in k.lower()]
# Result: []  (empty list)
```

**No positional embedding weights exist in the checkpoint.**

### 2. Complete List of Model Weights

All weights in `domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`:

| Weight Category | Keys |
|-----------------|------|
| Feature Embeddings | `high_pip_embed`, `low_pip_embed`, `is_double_embed`, `count_value_embed`, `trump_rank_embed`, `player_id_embed`, `is_current_embed`, `is_partner_embed`, `is_remaining_embed`, `token_type_embed`, `decl_embed`, `leader_embed` |
| Projection | `input_proj`, `output_proj`, `value_head` |
| Transformer | `transformer.layers.{0-5}.{linear1,linear2,norm1,norm2,self_attn.*}` |

**No `pos_embed`, `positional_encoding`, `position_embed`, or similar keys exist.**

### 3. Architecture Confirmation

From `forge/ml/module.py` lines 51-58:

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

PyTorch's `nn.TransformerEncoder` does **not** include positional encoding by default. The user must explicitly add it, and this model does not.

## Position Information in the Model

While no explicit positional embeddings exist, the model has these sources of position information:

### Token Type Embedding (Partial Position Signal)

```
token_type_embed.weight: shape [8, 42]

Type 0 (CONTEXT):   L2=3.72, Var=0.329
Type 1 (PLAYER0):   L2=1.24, Var=0.037
Type 2 (PLAYER1):   L2=1.48, Var=0.053
Type 3 (PLAYER2):   L2=1.49, Var=0.052
Type 4 (PLAYER3):   L2=1.39, Var=0.041
Type 5 (TRICK_P0):  L2=2.78, Var=0.187
Type 6 (TRICK_P1):  L2=3.18, Var=0.245
Type 7 (TRICK_P2):  L2=3.71, Var=0.335
```

**Observations**:
- CONTEXT token (type 0) has the highest L2 norm (3.72)
- Player hand tokens (types 1-4) have similar low norms (1.24-1.49)
- Trick tokens (types 5-7) have progressively higher norms (2.78-3.71)

This embedding distinguishes token **categories**, not within-hand positions.

### Player ID Embedding

```
player_id_embed.weight: shape [4, 21]

Player 0 (CURRENT):  L2=1.79, Var=0.160
Player 1 (NEXT):     L2=2.03, Var=0.204
Player 2 (PARTNER):  L2=2.09, Var=0.202
Player 3 (PREV):     L2=2.41, Var=0.280
```

This embedding distinguishes player **ownership**, not domino positions within a hand.

### Within-Hand Position: COMPLETELY ABSENT

For the 7 dominoes within a player's hand, there is **no distinguishing signal**:
- All 7 hand tokens share the same `token_type` embedding (e.g., TOKEN_TYPE_PLAYER0)
- All 7 share the same `player_id` embedding (normalized to current player)
- The only differences are the domino-specific features (pips, trump rank, etc.)

**The model cannot distinguish slot 0 from slot 6 based on positional information alone.**

## Cosine Similarity Analysis

Token type embeddings have low inter-type similarity:

```
         T0    T1    T2    T3    T4    T5    T6    T7
T0:    1.00  0.06 -0.08  0.24 -0.01  0.15 -0.17  0.07
T1:    0.06  1.00  0.15  0.07  0.02 -0.14 -0.21 -0.00
T2:   -0.08  0.15  1.00  0.12  0.28 -0.09 -0.02 -0.23
T3:    0.24  0.07  0.12  1.00  0.20 -0.12 -0.10 -0.20
T4:   -0.01  0.02  0.28  0.20  1.00 -0.03 -0.23 -0.21
T5:    0.15 -0.14 -0.09 -0.12 -0.03  1.00  0.06  0.01
T6:   -0.17 -0.21 -0.02 -0.10 -0.23  0.06  1.00  0.40
T7:    0.07 -0.00 -0.23 -0.20 -0.21  0.01  0.40  1.00
```

Player types (T1-T4) are nearly orthogonal to each other and to trick types (T5-T7). The context token (T0) is most similar to PLAYER2 (partner, cos=0.24).

## Implications for Slot 0 Bias

Since there are no learned positional embeddings:

1. **PE degeneracy (sin(0)=0)**: NOT APPLICABLE - no sinusoidal PE
2. **Learned PE undertrained at position 0**: NOT APPLICABLE - no learned PE
3. **Positional bias in attention computation**: NOT APPLICABLE - no position info

The transformer is **permutation-equivariant** over tokens of the same type. The only reasons slot 0 could behave differently are:

### Potential Remaining Causes

1. **Output Extraction Order**: The model always extracts hand tokens in deal order via `torch.gather`. Output slot 0 is always the first-dealt domino.

2. **Implicit Attention Patterns**: Even without PE, attention may develop implicit ordering preferences through content patterns or learned query/key interactions.

3. **Gradient Flow Asymmetry**: During backpropagation, the first position in a sequence may receive different gradient magnitudes.

4. **Value Head Interference**: The value head uses the context token at position 0. During training, this may create gradient competition that affects nearby hand positions differently.

## Conclusion

**Learned positional embedding weights CANNOT be the cause of slot 0 bias because no such weights exist.**

The DominoTransformer architecture:
- Uses no explicit positional encoding (sinusoidal or learned)
- Has no RoPE or other relative position mechanism
- Relies entirely on feature embeddings (token type, player ID) for position-related information
- Cannot distinguish positions **within** a player's hand

The slot 0 degradation (r=0.81 vs r=0.99+) must originate from:
- The fixed output extraction mechanism (gather by position)
- Implicit attention ordering effects
- Gradient flow asymmetries during training
- Or data distribution effects specific to first-dealt dominoes

## Recommendation

Investigate:
1. Whether first-dealt dominoes have systematically different strategic properties
2. Attention pattern visualization to see if position 0 receives different attention
3. Gradient magnitude analysis during training for position 0 vs other positions
